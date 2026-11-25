"""Exact canonical-suit permutation data augmentation for BridgeEncoder tensors.

BridgeEncoder uses canonical suit ordering:
  - Suit contracts (state[:, 549] == 0): trump is suit 0; plain suits 1, 2, 3 are symmetric (3! = 6 perms).
  - NT contracts   (state[:, 549] == 1): all 4 suits 0, 1, 2, 3 are symmetric (4! = 24 perms).

Precomputed index tables let us permute an entire training minibatch in one vectorized gather on GPU.
"""

from __future__ import annotations

import itertools
import torch

# Exact feature offsets matching cpp/bridge/encoder.cpp (kBridgeStateDim=570, kBridgePlyDim=66, kBridgeActionDim=76)
STATE_DIM = 570
PLY_DIM = 66
ACTION_DIM = 76
IS_NT_IDX = 549

# 10 blocks of 52 cards (4 suits x 13 ranks) in state: [0..520)
_STATE_CARD_BLOCKS = tuple(range(0, 520, 52))
# 4-suit blocks in state: void_suits (3 x 4 at 520..532), led_suit (544..548), unseen_suit_cnt (566..570)
_STATE_SUIT_BLOCKS = (520, 524, 528, 544, 566)

_PERMS_SUIT = [tuple([0] + list(p)) for p in itertools.permutations((1, 2, 3))]  # 6 perms (fix trump 0)
_PERMS_NT = [tuple(p) for p in itertools.permutations((0, 1, 2, 3))]              # 24 perms (all 4 suits)


def _build_index_tables(perms: list[tuple[int, int, int, int]], state_dim: int = STATE_DIM) -> dict[str, torch.Tensor]:
    """Build gather indices so out[..., i] = in[..., idx[p, i]] (maps suit s -> perm[s])."""
    n = len(perms)
    state_idx = torch.arange(state_dim, dtype=torch.long).unsqueeze(0).repeat(n, 1)
    ply_idx = torch.arange(PLY_DIM, dtype=torch.long).unsqueeze(0).repeat(n, 1)
    act_idx = torch.arange(ACTION_DIM, dtype=torch.long).unsqueeze(0).repeat(n, 1)
    card_idx = torch.arange(52, dtype=torch.long).unsqueeze(0).repeat(n, 1)

    suit_blocks = _STATE_SUIT_BLOCKS + ((573, 577, 581) if state_dim >= 585 else ())
    for pi, perm in enumerate(perms):
        # inv[new_suit] = old_suit so gather puts old_suit at new_suit
        inv = [0] * 4
        for old_s, new_s in enumerate(perm):
            inv[new_s] = old_s
        for new_s in range(4):
            old_s = inv[new_s]
            for r in range(13):
                card_idx[pi, new_s * 13 + r] = old_s * 13 + r
                for base in _STATE_CARD_BLOCKS:
                    state_idx[pi, base + new_s * 13 + r] = base + old_s * 13 + r
                ply_idx[pi, new_s * 13 + r] = old_s * 13 + r
                act_idx[pi, new_s * 13 + r] = old_s * 13 + r
            for base in suit_blocks:
                state_idx[pi, base + new_s] = base + old_s
            ply_idx[pi, 60 + new_s] = 60 + old_s
            act_idx[pi, 52 + new_s] = 52 + old_s

    # Forward card map: fwd_card[pi, old_card] = new_card (for permuting card-slot indices like legal_slots)
    fwd_card = torch.zeros((n, 52), dtype=torch.long)
    for pi, perm in enumerate(perms):
        for old_s, new_s in enumerate(perm):
            for r in range(13):
                fwd_card[pi, old_s * 13 + r] = new_s * 13 + r

    return {"state": state_idx, "ply": ply_idx, "act": act_idx, "card": card_idx, "fwd_card": fwd_card}


class SuitPermAugmenter:
    """Applies random valid suit permutations (6x for Suit, 24x for NT) on GPU or CPU."""

    def __init__(self, device: torch.device | str, state_dim: int = STATE_DIM):
        dev = torch.device(device)
        # Put all 24 permutations in one table: first 6 are the suit-fixing ones!
        suit_set = set(_PERMS_SUIT)
        ordered = list(_PERMS_SUIT) + [p for p in _PERMS_NT if p not in suit_set]
        assert len(ordered) == 24
        tabs = _build_index_tables(ordered, state_dim=state_dim)
        self.state_idx = tabs["state"].to(dev)
        self.ply_idx = tabs["ply"].to(dev)
        self.act_idx = tabs["act"].to(dev)
        self.card_idx = tabs["card"].to(dev)
        self.fwd_card = tabs["fwd_card"].to(dev)
        self.device = dev

    def sample_perm_ids(self, state: torch.Tensor) -> torch.Tensor:
        """Per-row permutation id: 0..5 for suit contracts, 0..23 for NT contracts."""
        bsz = state.shape[0]
        is_nt = state[:, IS_NT_IDX] > 0.5
        p_suit = torch.randint(0, 6, (bsz,), device=self.device)
        p_nt = torch.randint(0, 24, (bsz,), device=self.device)
        return torch.where(is_nt, p_nt, p_suit)

    def permute_state(self, state: torch.Tensor, pid: torch.Tensor) -> torch.Tensor:
        return torch.gather(state, 1, self.state_idx[pid])

    def permute_dmc_batch(self, batch: dict) -> dict:
        """Permutes state, history, actions (and optional belief_targets/belief_marginals) consistently."""
        pid = self.sample_perm_ids(batch["state"])
        out = dict(batch)
        out["state"] = torch.gather(batch["state"], 1, self.state_idx[pid])

        hist = batch["history"]  # [B, T, PLY_DIM]
        h_idx = self.ply_idx[pid].unsqueeze(1).expand(-1, hist.shape[1], -1)
        out["history"] = torch.gather(hist, 2, h_idx)

        acts = batch["actions"]  # [B, K, ACTION_DIM]
        a_idx = self.act_idx[pid].unsqueeze(1).expand(-1, acts.shape[1], -1)
        out["actions"] = torch.gather(acts, 2, a_idx)

        if "belief_targets" in batch:
            out["belief_targets"] = torch.gather(batch["belief_targets"], 1, self.card_idx[pid])
        if "belief_marginals" in batch:
            bm = batch["belief_marginals"]  # [B, 52, 3]
            c_idx = self.card_idx[pid].unsqueeze(-1).expand(-1, -1, bm.shape[2])
            out["belief_marginals"] = torch.gather(bm, 1, c_idx)
        if "legal_slots" in batch:
            ls = batch["legal_slots"]  # [B, K], -1 for padding
            valid = ls >= 0
            mapped = torch.gather(self.fwd_card[pid], 1, ls.clamp(min=0))
            out["legal_slots"] = torch.where(valid, mapped, ls)
        return out

    def permute_distill_batch(self, state: torch.Tensor, mask: torch.Tensor, target: torch.Tensor):
        pid = self.sample_perm_ids(state)
        c_idx = self.card_idx[pid]
        return (
            torch.gather(state, 1, self.state_idx[pid]),
            torch.gather(mask, 1, c_idx),
            torch.gather(target, 1, c_idx),
        )
