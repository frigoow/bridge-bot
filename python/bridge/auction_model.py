"""PyTorch AuctionNet matching cpp/bridge/auction_model.h (BRA1 binary format) bit-for-bit.

Given an auction call sequence (step features [B, T+1, 45]) and the 4 players' initial 13-card
hands (hand features [B, 4, 62]):
  - Causal GRU produces per-step contexts ctx[:, 0..T-1, :] and final context h_final = ctx[:, T, :]
  - Head 1 (Bid likelihood): P_bid(call_t | hand_{seat(t)}, ctx[:, t, :]) over 38 calls (masked to legal calls)
  - Head 2 (Auction card prior): P_auc(card c belongs to seat rel_dealer | h_final) over [52, 4]
"""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

AUC_STEP_DIM = 45
AUC_HAND_DIM = 62
AUC_CALLS = 38
N_CARDS = 52


def _write_linear(f, layer: nn.Linear) -> None:
    f.write(layer.weight.detach().cpu().float().contiguous().numpy().tobytes())
    f.write(layer.bias.detach().cpu().float().contiguous().numpy().tobytes())


def _write_ln(f, ln: nn.LayerNorm) -> None:
    f.write(ln.weight.detach().cpu().float().contiguous().numpy().tobytes())
    f.write(ln.bias.detach().cpu().float().contiguous().numpy().tobytes())


def _write_gru(f, gru: nn.GRU, empty_emb: torch.Tensor) -> None:
    f.write(gru.weight_ih_l0.detach().cpu().float().contiguous().numpy().tobytes())
    f.write(gru.weight_hh_l0.detach().cpu().float().contiguous().numpy().tobytes())
    f.write(gru.bias_ih_l0.detach().cpu().float().contiguous().numpy().tobytes())
    f.write(gru.bias_hh_l0.detach().cpu().float().contiguous().numpy().tobytes())
    f.write(empty_emb.detach().cpu().float().contiguous().numpy().tobytes())


class AuctionNet(nn.Module):
    def __init__(self, gru_hidden: int = 192, hand_hidden: int = 128, head_hidden: int = 192):
        super().__init__()
        self.gru_hidden = gru_hidden
        self.hand_hidden = hand_hidden
        self.head_hidden = head_hidden

        self.auc_gru = nn.GRU(AUC_STEP_DIM, gru_hidden, batch_first=True)
        self.empty_emb = nn.Parameter(torch.zeros(gru_hidden))  # kept for cardbot::nn::readGru layout

        self.hand_fc1 = nn.Linear(AUC_HAND_DIM, hand_hidden)
        self.hand_ln1 = nn.LayerNorm(hand_hidden)
        self.hand_fc2 = nn.Linear(hand_hidden, hand_hidden)
        self.hand_ln2 = nn.LayerNorm(hand_hidden)

        self.bid_fc1 = nn.Linear(gru_hidden + hand_hidden, head_hidden)
        self.bid_ln1 = nn.LayerNorm(head_hidden)
        self.bid_fc2 = nn.Linear(head_hidden, AUC_CALLS)

        self.card_fc1 = nn.Linear(gru_hidden, head_hidden)
        self.card_ln1 = nn.LayerNorm(head_hidden)
        self.card_fc2 = nn.Linear(head_hidden, N_CARDS * 4)

    def encode_hands(self, hand_feats: torch.Tensor) -> torch.Tensor:
        """hand_feats: [..., 62] -> [..., hand_hidden]"""
        h = F.silu(self.hand_ln1(self.hand_fc1(hand_feats)))
        return F.silu(self.hand_ln2(self.hand_fc2(h)))

    def forward(
        self,
        steps: torch.Tensor,      # [B, T_max+1, 45]
        n_calls: torch.Tensor,    # [B] number of calls (valid bid steps 0..n_calls-1, final at index n_calls)
        hand_feats: torch.Tensor, # [B, 4, 62] indexed by seat relative to dealer (0..3)
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (bid_logits [B, T_max, 38], card_logits [B, 52, 4])."""
        bsz, seq_len, _ = steps.shape
        t_max = seq_len - 1
        ctx, _ = self.auc_gru(steps)  # [B, T_max+1, H]

        # Gather final context at index n_calls for each batch item
        idx_final = n_calls.clamp(min=0, max=t_max).long().view(bsz, 1, 1).expand(-1, 1, self.gru_hidden)
        h_final = ctx.gather(1, idx_final).squeeze(1)  # [B, H]
        ch = F.silu(self.card_ln1(self.card_fc1(h_final)))
        card_logits = self.card_fc2(ch).view(bsz, N_CARDS, 4)

        # Encode all 4 hands relative to dealer, then align each step t with seat (t % 4)
        h_emb = self.encode_hands(hand_feats)  # [B, 4, HH]
        step_rel = (torch.arange(t_max, device=steps.device) & 3).view(1, t_max, 1).expand(bsz, -1, self.hand_hidden)
        step_hand = h_emb.gather(1, step_rel)  # [B, T_max, HH]

        cat = torch.cat([ctx[:, :t_max, :], step_hand], dim=-1)  # [B, T_max, H + HH]
        bh = F.silu(self.bid_ln1(self.bid_fc1(cat)))
        bid_logits = self.bid_fc2(bh)  # [B, T_max, 38]
        return bid_logits, card_logits

    def export_bin(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            f.write(b"BRA1")
            f.write(
                struct.pack(
                    "<6i",
                    1,
                    AUC_STEP_DIM,
                    AUC_HAND_DIM,
                    self.gru_hidden,
                    self.hand_hidden,
                    self.head_hidden,
                )
            )
            _write_gru(f, self.auc_gru, self.empty_emb)
            _write_linear(f, self.hand_fc1)
            _write_ln(f, self.hand_ln1)
            _write_linear(f, self.hand_fc2)
            _write_ln(f, self.hand_ln2)
            _write_linear(f, self.bid_fc1)
            _write_ln(f, self.bid_ln1)
            _write_linear(f, self.bid_fc2)
            _write_linear(f, self.card_fc1)
            _write_ln(f, self.card_ln1)
            _write_linear(f, self.card_fc2)
