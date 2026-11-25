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
        self.bidder = load("bidder.onnx") if (onnx_dir / "bidder.onnx").exists() else None
        self.bidinfo = load("bidinfo.onnx") if (onnx_dir / "bidinfo.onnx").exists() else None
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

    @staticmethod
    def _mask_to_hand24(mask: int) -> np.ndarray:
        h = np.zeros(24, dtype=np.float32)
        m = int(mask)
        while m:
            lsb = m & -m
            c = lsb.bit_length() - 1
            m ^= lsb
            su = c // 13       # 0=C..3=S
            rk = c % 13        # 0=2..12=A
            ben_s = 3 - su     # 0=S, 1=H, 2=D, 3=C
            ben_r = (12 - rk) if rk >= 8 else 5  # A=0,K=1,Q=2,J=3,T=4,x=5
            h[ben_s * 6 + ben_r] += 1.0
        return h

    @staticmethod
    def _build_auction_x(auction_calls: list[int], dealer: int, seat_ix: int, hands24: np.ndarray,
                         vul_ns: bool, vul_ew: bool):
        """Exact BEN model_version=3 (NS=1, EW=1, 24-card) auction tensor [N, n_steps, 193]."""
        auc = [0] * dealer + [c + 2 for c in auction_calls]
        n_steps = 0
        bi = seat_ix
        while bi < len(auc):
            if auc[bi] != 0:
                n_steps += 1
            bi += 4
        if n_steps == 0:
            return None, []
        auc_pad = auc + [1] * (4 * n_steps + 4)
        n_samples = hands24.shape[0]
        X = np.zeros((n_samples, n_steps, 193), dtype=np.float32)
        X[:, :, 0] = 1.0  # NS=1
        X[:, :, 1] = 1.0  # EW=1
        X[:, :, 2] = float(vul_ns if (seat_ix % 2 == 0) else vul_ew)
        X[:, :, 3] = float(vul_ew if (seat_ix % 2 == 0) else vul_ns)
        h = hands24.reshape(n_samples, 4, 6)
        shp = (h.sum(axis=2) - 3.25) / 1.75
        hcp = ((h[:, :, 0] * 4 + h[:, :, 1] * 3 + h[:, :, 2] * 2 + h[:, :, 3] * 1).sum(axis=1, keepdims=True) - 10.0) / 4.0
        X[:, :, 4:5] = hcp[:, None, :]
        X[:, :, 5:9] = shp[:, None, :]
        X[:, :, 9:33] = hands24[:, None, :]

        bi = seat_ix
        while auc_pad[bi] == 0:
            bi += 4
        targets = []
        for st in range(n_steps):
            targets.append(auc_pad[bi])
            b_me = auc_pad[bi - 4] if bi - 4 >= 0 else 0
            b_lho = auc_pad[bi - 3] if bi - 3 >= 0 else 0
            b_par = auc_pad[bi - 2] if bi - 2 >= 0 else 0
            b_rho = auc_pad[bi - 1] if bi - 1 >= 0 else 0
            X[:, st, 33 + b_me] = 1.0
            X[:, st, 73 + b_lho] = 1.0
            X[:, st, 113 + b_par] = 1.0
            X[:, st, 153 + b_rho] = 1.0
            bi += 4
        return X, targets

    def _opening_lead_probs32(self, game, use_auction: bool = False) -> np.ndarray:
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
        b = np.zeros((1, 15), dtype=np.float32)
        if use_auction and self.bidinfo is not None and c.has_auction():
            h24 = self._mask_to_hand24(game.initial_hand(game.to_move()))[None, :]
            X, _ = self._build_auction_x(c.auction_calls, c.auction_dealer, game.to_move(), h24, c.vul_ns, c.vul_ew)
            if X is not None:
                inp_name = self.bidinfo.get_inputs()[0].name
                p_hcp, p_shp = self.bidinfo.run(None, {inp_name: X})
                b[0, :3] = p_hcp[0, -1, :]
                b[0, 3:] = p_shp[0, -1, :]
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

    def card_scores52(self, game, use_auction: bool = False) -> dict[int, float]:
        """Return BEN's neural probability score for every legal card in `game`."""
        legal = game.legal_actions()
        if len(legal) == 1:
            return {legal[0]: 1.0}
        c = game.contract()
        if game.n_plies() == 0:
            probs32 = self._opening_lead_probs32(game, use_auction=use_auction)
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

    def score_worlds_auction(self, game, worlds: list) -> np.ndarray:
        """Compute BEN bidder.onnx log-likelihood of hidden seats' bids for every sampled world."""
        n = len(worlds)
        ll = np.zeros(n, dtype=np.float64)
        c = game.contract()
        if self.bidder is None or not c.has_auction() or n == 0:
            return ll
        # Determine hidden seats for current observer
        seat = game.to_move()
        dummy = c.dummy()
        decl = c.declarer
        known = {seat}
        if game.dummy_visible():
            known.add(dummy)
        if seat == dummy or seat == decl:
            known.add(decl)
            known.add(dummy)
        hidden = [s for s in range(4) if s not in known]

        inp_name = self.bidder.get_inputs()[0].name
        calls = c.auction_calls
        dealer = c.auction_dealer
        for s in hidden:
            h24 = np.stack([self._mask_to_hand24(wg.initial_hand(s)) for wg in worlds], axis=0)
            X, targets = self._build_auction_x(calls, dealer, s, h24, c.vul_ns, c.vul_ew)
            if X is None:
                continue
            probs = self.bidder.run(None, {inp_name: X})[0]  # [N, n_steps, 40]
            for st, tgt in enumerate(targets):
                p = np.clip(probs[:, st, tgt].astype(np.float64), 1e-6, 1.0)
                ll += np.log(p)
        return ll

    def act(self, game) -> int:
        legal = game.legal_actions()
        if len(legal) == 1:
            return legal[0]
        scores = self.card_scores52(game)
        return max(legal, key=lambda c: scores[c])


class BenDdsPlayer:
    """BEN neural card-play policy + DDS evaluation over sampled worlds (BEN tournament style).

    Combines BEN's ONNX neural network preference with expected duplicate score from
    Bo Haglund's DDS across `worlds` consistent sampled worlds (matching BEN's
    `pick_card_after_dd_eval`: DDS expected score + small NN tiebreak/prior weight).
    """

    def __init__(self, bridge_cpp_mod, worlds: int = 16, nn_weight: float = 20.0,
                 seed: int = 1, ben_root: str | Path | None = None,
                 use_auction: bool = False, auc_pool: int = 6, auc_weight: float = 0.45):
        self.cpp = bridge_cpp_mod
        self.nn = BenOnnxPlayer(ben_root=ben_root)
        self.worlds = max(1, worlds)
        self.nn_weight = nn_weight
        self.seed = seed
        self.use_auction = use_auction
        self.auc_pool = max(1, auc_pool)
        self.auc_weight = auc_weight
        self._step = 0

    def act(self, game) -> int:
        legal = game.legal_actions()
        if len(legal) == 1:
            return legal[0]
        self._step += 1
        c = game.contract()
        do_auc = self.use_auction and c.has_auction() and (self.nn.bidder is not None)
        nn_scores = self.nn.card_scores52(game, use_auction=do_auc)
        n_draw = self.worlds * self.auc_pool if do_auc else self.worlds
        worlds = self.cpp.sample_worlds(game, n_draw, self.seed * 1000003 + self._step)
        if not worlds:
            return max(legal, key=lambda card: nn_scores[card])

        if do_auc and len(worlds) > self.worlds:
            ll = self.nn.score_worlds_auction(game, worlds)
            top_idx = np.argsort(ll)[-self.worlds:]
            worlds = [worlds[int(i)] for i in top_idx]
            ll_top = ll[top_idx]
            mx = float(ll_top.max())
            wts = np.exp(np.clip(self.auc_weight * (ll_top - mx), -40.0, 0.0))
            wts /= float(wts.sum())
        elif do_auc:
            ll = self.nn.score_worlds_auction(game, worlds)
            mx = float(ll.max())
            wts = np.exp(np.clip(self.auc_weight * (ll - mx), -40.0, 0.0))
            wts /= float(wts.sum())
        else:
            wts = np.full(len(worlds), 1.0 / len(worlds), dtype=np.float64)

        seat = game.to_move()
        seat_is_decl = (game.team(seat) == c.declarer_team())
        rem_total = 13 - (game.n_plies() // 4)
        decl_so_far = game.decl_tricks()
        decl_vul = c.declarer_vul()

        dd_sum = {card: 0.0 for card in legal}
        for wi, wg in enumerate(worlds):
            w = float(wts[wi])
            for card, rem_tr in self.cpp.dds_solve_moves(wg):
                if card not in dd_sum:
                    continue
                tot_decl = (decl_so_far + rem_tr) if seat_is_decl else (decl_so_far + (rem_total - rem_tr))
                ds = self.cpp.contract_score(c.level, c.strain, c.doubled, decl_vul, tot_decl)
                dd_sum[card] += w * float(ds if seat_is_decl else -ds)

        return max(legal, key=lambda card: dd_sum[card] + self.nn_weight * nn_scores[card])

