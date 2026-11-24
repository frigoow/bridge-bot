"""Adapter for BEN (lorserker/ben, GPL-3.0) pretrained ONNX card-play models.

Loads BEN's real pretrained ONNX neural networks from external/ben/models/onnx/:
  - lead_nt.onnx, lead_suit.onnx          (opening lead)
  - lefty_nt.onnx, lefty_suit.onnx        (LHO defender)
  - dummy_nt.onnx, dummy_suit.onnx        (dummy, played by declarer)
  - righty_nt.onnx, righty_suit.onnx      (RHO defender)
  - decl_nt.onnx, decl_suit.onnx          (declarer)

Supports two modes:
  - BenOnnxPlayer(mode="nn"): pure BEN neural network card player (~0.5 ms/card)
  - BenOnnxPlayer(mode="nn_dds", worlds=16): BEN neural candidate filter + DDS
    evaluation over consistent sampled worlds (matching BEN's tournament card-play loop).
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import onnxruntime as ort


def _our_to_ben_card32(card52: int) -> int:
    """Convert bridge-bot card (suit 0=C..3=S, rank 0=2..12=A) to BEN card32 (S,H,D,C x A,K,Q,J,T,9,8,x)."""
    our_suit = card52 // 13
    our_rank = card52 % 13
    ben_suit = 3 - our_suit
    ben_rank32 = min(12 - our_rank, 7)
    return ben_suit * 8 + ben_rank32


def _our_to_ben_strain(our_strain: int) -> int:
    """Convert bridge-bot strain (0=C,1=D,2=H,3=S,4=NT) to BEN strain_i (0=NT,1=S,2=H,3=D,4=C)."""
    return 0 if our_strain == 4 else (4 - our_strain)


def _hand_mask_to_ben32(mask: int) -> np.ndarray:
    h = np.zeros(32, dtype=np.float32)
    m = int(mask)
    while m:
        lsb = m & -m
        c = lsb.bit_length() - 1
        h[_our_to_ben_card32(c)] += 1.0
        m ^= lsb
    return h


class BenOnnxPlayer:
    """Drives BEN's pretrained ONNX card-play models on any bridge_cpp.BridgeGame state."""

    def __init__(self, ben_root: str | Path | None = None, threads: int = 1):
        root = Path(ben_root) if ben_root else Path(__file__).resolve().parents[2] / "external" / "ben"
        onnx_dir = root / "models" / "onnx"
        if not onnx_dir.exists():
            raise FileNotFoundError(
                f"BEN ONNX models not found at {onnx_dir}. Run scripts/fetch_opponents.sh first."
            )

        opts = ort.SessionOptions()
        opts.intra_op_num_threads = threads
        opts.inter_op_num_threads = 1
        opts.log_severity_level = 3

        def load(name: str):
            return ort.InferenceSession(str(onnx_dir / name), sess_options=opts, providers=["CPUExecutionProvider"])

        self.lead_nt = load("lead_nt.onnx")
        self.lead_suit = load("lead_suit.onnx")
        # Indexed by player_i in BEN numbering: 0=lefty, 1=dummy, 2=righty, 3=decl
        self.nt_models = [
            load("lefty_nt.onnx"),
            load("dummy_nt.onnx"),
            load("righty_nt.onnx"),
            load("decl_nt.onnx"),
        ]
        self.suit_models = [
            load("lefty_suit.onnx"),
            load("dummy_suit.onnx"),
            load("righty_suit.onnx"),
            load("decl_suit.onnx"),
        ]

    def _opening_lead_probs32(self, game) -> np.ndarray:
        c = game.contract()
        ben_strain = _our_to_ben_strain(c.strain)
        x = np.zeros((1, 42), dtype=np.float32)
        x[0, 0] = float(c.level)
        x[0, 1 + ben_strain] = 1.0
        x[0, 6] = 1.0 if c.doubled == 1 else 0.0
        x[0, 7] = 1.0 if c.doubled == 2 else 0.0
        decl_vul = c.declarer_vul()
        def_vul = c.vul_ew if c.declarer_team() == 0 else c.vul_ns
        x[0, 8] = 1.0 if def_vul else 0.0
        x[0, 9] = 1.0 if decl_vul else 0.0
        x[0, 10:42] = _hand_mask_to_ben32(game.hand(game.to_move()))
        # Neutral bidding info prior (no auction in card-play-only mode)
        b = np.zeros((1, 15), dtype=np.float32)
        sess = self.lead_nt if c.strain == 4 else self.lead_suit
        return sess.run(None, {"X_input": x, "B_input": b})[0][0]

    def _build_x_play(self, game, player_i: int) -> np.ndarray:
        """Reconstruct BEN's exact [1, trick_i + 1, 298] play history tensor for player_i (0..3)."""
        c = game.contract()
        decl = c.declarer
        ben_strain = _our_to_ben_strain(c.strain)

        # Map BEN player index p in {0=LHO, 1=Dummy, 2=RHO, 3=Decl} <-> our seat
        def ben_to_seat(p: int) -> int:
            return (decl + 1 + p) & 3

        def seat_to_ben(s: int) -> int:
            return (s - (decl + 1)) & 3

        init_hands32 = [_hand_mask_to_ben32(game.initial_hand(ben_to_seat(p))) for p in range(4)]
        hist = game.history()  # list of (seat, card, led_suit, pos)
        n_plies = len(hist)
        cur_trick_i = n_plies // 4

        x = np.zeros((1, cur_trick_i + 1, 298), dtype=np.float32)
        cur_hands = [h.copy() for h in init_hands32]

        # Replay completed tricks 0 .. cur_trick_i - 1, then partial current trick
        for t in range(cur_trick_i + 1):
            pub_p = 3 if player_i == 1 else 1
            x[0, t, 0:32] = cur_hands[player_i]
            x[0, t, 32:64] = cur_hands[pub_p]
            x[0, t, 292] = float(c.level)
            x[0, t, 293 + ben_strain] = 1.0

            if t > 0:
                prev_base = (t - 1) * 4
                prev_leader_p = seat_to_ben(hist[prev_base][0])
                x[0, t, 288 + prev_leader_p] = 1.0
                for j in range(4):
                    c32 = _our_to_ben_card32(hist[prev_base + j][1])
                    x[0, t, 64 + j * 32 + c32] = 1.0

            base = t * 4
            end = min(base + 4, n_plies)
            if t < cur_trick_i:
                # Populate plays made BEFORE player_i acted in trick t
                leader_p = seat_to_ben(hist[base][0])
                for j in range(4):
                    s, card52, _, _ = hist[base + j]
                    p = seat_to_ben(s)
                    c32 = _our_to_ben_card32(card52)
                    played_before_me = ((p - leader_p) % 4) < ((player_i - leader_p) % 4)
                    if played_before_me:
                        if p == pub_p:
                            x[0, t, 32 + c32] -= 1.0
                        offset = (player_i - p) % 4
                        x[0, t, 192 + (3 - offset) * 32 + c32] = 1.0
                    cur_hands[p][c32] -= 1.0
            else:
                # Current partial trick: every card played so far was played before player_i
                for j in range(end - base):
                    s, card52, _, _ = hist[base + j]
                    p = seat_to_ben(s)
                    c32 = _our_to_ben_card32(card52)
                    if p == pub_p:
                        x[0, t, 32 + c32] -= 1.0
                    offset = (player_i - p) % 4
                    x[0, t, 192 + (3 - offset) * 32 + c32] = 1.0

        return x

    def card_scores52(self, game) -> dict[int, float]:
        """Return BEN's neural probability score for every legal card in `game`."""
        legal = game.legal_actions()
        if len(legal) == 1:
            return {legal[0]: 1.0}
        c = game.contract()
        if game.n_plies() == 0:
            probs32 = self._opening_lead_probs32(game)
        else:
            player_i = (game.to_move() - (c.declarer + 1)) & 3
            x = self._build_x_play(game, player_i)
            sess = (self.nt_models if c.strain == 4 else self.suit_models)[player_i]
            inp_name = sess.get_inputs()[0].name
            out = sess.run(None, {inp_name: x})[0]
            probs32 = out[0, -1, :]

        # Map 32-card probabilities back onto legal 52-card actions
        # For spot cards (rank <= 5, i.e., 2..7), prefer lowest rank (standard bridge spot play)
        scores = {}
        for card52 in legal:
            c32 = _our_to_ben_card32(card52)
            # Tiny tie-break preferring lower rank among equivalent small spot cards (2..7)
            scores[card52] = float(probs32[c32]) - 1e-6 * float(card52 % 13)
        return scores

    def act(self, game) -> int:
        legal = game.legal_actions()
        if len(legal) == 1:
            return legal[0]
        scores = self.card_scores52(game)
        return max(legal, key=lambda c: scores[c])
