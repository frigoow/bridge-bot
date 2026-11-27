"""Full upstream BEN card-play driver adapter (external/ben/src, GPL-3.0).

Runs BEN's unmodified upstream card-play stack (`BotLead`, `Sample.init_rollout_states`,
`CardPlayer.play_card`, `carding.py`, `DDSolver`, and `config/onnx.conf`) directly
inside our mirrored duplicate arena (`tools/opponents/arena.py`).

Uses BEN's official ONNX config (`config/onnx.conf`) with a lightweight stub for the
unused top-level `import tensorflow` statements so no TensorFlow install is required.
"""

from __future__ import annotations

import os
import sys
import types
from pathlib import Path
from typing import Optional

import numpy as np


_BEN_BOOTSTRAPPED = False
_BEN_MODS: dict = {}


def _bootstrap_ben(ben_root: Path):
    """Add external/ben/src to sys.path and stub unused TF/colorama imports for ONNX mode."""
    global _BEN_BOOTSTRAPPED
    if _BEN_BOOTSTRAPPED:
        return
    src_dir = str(ben_root / "src")
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)

    if "tensorflow" not in sys.modules:
        sys.modules["tensorflow"] = types.ModuleType("tensorflow")
    if "colorama" not in sys.modules:
        col = types.ModuleType("colorama")

        class _Empty:
            def __getattr__(self, _):
                return ""

        col.Fore = _Empty()
        col.Back = _Empty()
        col.Style = _Empty()
        col.init = lambda *a, **k: None
        sys.modules["colorama"] = col

    for mod_name, attr in [
        ("nn.player_tf2", "BatchPlayer"),
        ("nn.bid_info_tf2", "BidInfo"),
        ("nn.leader_tf2", "Leader"),
        ("nn.lead_singledummy_tf2", "LeadSingleDummy"),
        ("nn.contract_tf2", "Contract"),
        ("nn.bidder_tf2", "Bidder"),
        ("nn.trick_tf2", "Trick"),
    ]:
        if mod_name not in sys.modules:
            m = types.ModuleType(mod_name)
            setattr(m, attr, None)
            sys.modules[mod_name] = m

    import conf
    import deck52
    from bidding import bidding
    from botcardplayer import CardPlayer
    from botopeninglead import BotLead
    from ddsolver.ddsolver import DDSolver
    from nn.models_tf2 import Models
    from sample import Sample
    from util import get_play_status, get_possible_cards, get_singleton

    _BEN_MODS.update(
        conf=conf,
        deck52=deck52,
        bidding=bidding,
        CardPlayer=CardPlayer,
        BotLead=BotLead,
        DDSolver=DDSolver,
        Models=Models,
        Sample=Sample,
        get_play_status=get_play_status,
        get_possible_cards=get_possible_cards,
        get_singleton=get_singleton,
    )
    _BEN_BOOTSTRAPPED = True


_RANKS_DESC = "AKQJT98765432"


def _mask_to_ben_hand_str(mask: int) -> str:
    """Convert bridge-bot 52-bit hand mask (0=C2..51=SA) to BEN 'S.H.D.C' PBN string."""
    parts = []
    for our_s in (3, 2, 1, 0):  # S, H, D, C
        s = []
        for our_r in range(12, -1, -1):  # A down to 2
            if mask & (1 << (our_s * 13 + our_r)):
                s.append(_RANKS_DESC[12 - our_r])
        parts.append("".join(s))
    return ".".join(parts)


def _contract_to_ben_str(c) -> str:
    """Convert bridge_cpp.Contract to BEN contract string, e.g. '3NE', '4SXN'."""
    strain_ch = ["C", "D", "H", "S", "N"][c.strain]
    dbl_str = ["", "X", "XX"][c.doubled]
    decl_ch = "NESW"[c.declarer]
    return f"{c.level}{strain_ch}{dbl_str}{decl_ch}"


def _our_to_ben_strain_i(our_strain: int) -> int:
    """bridge-bot strain (0=C,1=D,2=H,3=S,4=NT) -> BEN strain_i (0=NT,1=S,2=H,3=D,4=C)."""
    return 0 if our_strain == 4 else (4 - our_strain)


def _build_ben_auction(c) -> list[str]:
    bidding = _BEN_MODS["bidding"]
    if c.has_auction():
        return ["PAD_START"] * c.auction_dealer + [bidding.ID2BID[x + 2] for x in c.auction_calls]
    # Fallback synthetic minimal auction if deal lacks auction
    strain_ch = ["C", "D", "H", "S", "N"][c.strain]
    auc = ["PAD_START"] * c.declarer + [f"{c.level}{strain_ch}"]
    if c.doubled == 1:
        auc += ["X", "PASS", "PASS", "PASS"]
    elif c.doubled == 2:
        auc += ["X", "XX", "PASS", "PASS", "PASS"]
    else:
        auc += ["PASS", "PASS", "PASS"]
    return auc


class BenFullDriverPlayer:
    """Drives BEN's unmodified upstream BotLead + Sample + CardPlayer + DDSolver."""

    def __init__(
        self,
        ben_root: str | Path | None = None,
        sample_hands_play: Optional[int] = None,
        sample_boards_for_play: Optional[int] = None,
        use_pimc: bool = False,
        pimc_threads: int = 1,
        dds_threads: int = 1,
        seed: int = 1,
    ):
        self.root = Path(ben_root) if ben_root else Path(__file__).resolve().parents[2] / "external" / "ben"
        _bootstrap_ben(self.root)
        conf_mod = _BEN_MODS["conf"]
        c = conf_mod.load("config/onnx.conf")
        # Disable external Windows/BGA/ACE/BBA DLLs unless explicitly requested
        c.set("models", "use_bba", "False")
        c.set("models", "consult_bba", "False")
        c.set("models", "use_bba_to_count_aces", "False")
        c.set("models", "suppress_warnings", "True")
        c.set("bidding", "verbose", "False")
        c.set("pimc", "pimc_use_declaring", "True" if use_pimc else "False")
        c.set("pimc", "pimc_use_defending", "True" if use_pimc else "False")
        c.set("pimc", "pimc_max_threads", str(max(1, pimc_threads)))
        if c.has_section("ace"):
            c.set("ace", "ace_use_declaring", "False")
            c.set("ace", "ace_use_defending", "False")
        if sample_hands_play is not None:
            c.set("cardplay", "sample_hands_play", str(sample_hands_play))
            if sample_hands_play < 20:
                c.set("cardplay", "min_sample_hands_play", str(max(4, sample_hands_play // 2)))
                c.set("cardplay", "min_sample_hands_play_bad", str(max(4, sample_hands_play // 2)))
        if sample_boards_for_play is not None:
            c.set("cardplay", "sample_boards_for_play", str(sample_boards_for_play))

        self.models = _BEN_MODS["Models"].from_conf(c, str(self.root))
        self.sampler = _BEN_MODS["Sample"].from_conf(c, verbose=False)
        self.dds = _BEN_MODS["DDSolver"](dds_mode=1, max_threads=max(1, dds_threads))
        self.use_pimc = use_pimc
        self.seed = seed
        self._deal_key = None
        self._synced_plies = 0
        self._reset_state_vars()

    def _reset_state_vars(self):
        self._card_players = None
        self._auction = None
        self._hands_str = None
        self._contract_str = None
        self._strain_i = 0
        self._level = 1
        self._decl_i = 0
        self._vuln = [False, False]
        self._dealer_i = 0
        self._player_cards_played = [[] for _ in range(4)]
        self._player_cards_played52 = [[] for _ in range(4)]
        self._shown_out_suits = [set() for _ in range(4)]
        self._discards = [set() for _ in range(4)]
        self._tricks = []
        self._tricks52 = []
        self._current_trick = []
        self._current_trick52 = []
        self._leader_i = 0
        self._opening_lead52 = None

    def _init_deal(self, game):
        c = game.contract()
        ih = game.initial_hands()
        self._deal_key = (tuple(ih), c.level, c.strain, c.declarer, c.doubled, c.vul_ns, c.vul_ew)
        self._synced_plies = 0
        self._reset_state_vars()

        self._hands_str = [_mask_to_ben_hand_str(ih[s]) for s in range(4)]
        self._contract_str = _contract_to_ben_str(c)
        self._strain_i = _our_to_ben_strain_i(c.strain)
        self._level = c.level
        self._decl_i = c.declarer
        self._vuln = [bool(c.vul_ns), bool(c.vul_ew)]
        self._dealer_i = c.auction_dealer if c.has_auction() else c.declarer
        self._auction = _build_ben_auction(c)
        self.sampler.clear_sample_cache()

        is_decl_vuln = self._vuln[self._decl_i % 2]
        lefty_hand = self._hands_str[(self._decl_i + 1) % 4]
        dummy_hand = self._hands_str[(self._decl_i + 2) % 4]
        righty_hand = self._hands_str[(self._decl_i + 3) % 4]
        decl_hand = self._hands_str[self._decl_i]

        pimc = [None, None, None, None]
        if self.use_pimc:
            from pimc.PIMC import BGADLL
            from pimc.PIMCDef import BGADefDLL

            declarer_pimc = BGADLL(self.models, dummy_hand, decl_hand, self._contract_str, is_decl_vuln, self.sampler, False)
            pimc[1] = declarer_pimc
            pimc[3] = declarer_pimc
            pimc[0] = BGADefDLL(self.models, dummy_hand, lefty_hand, self._contract_str, is_decl_vuln, 0, self.sampler, False)
            pimc[2] = BGADefDLL(self.models, dummy_hand, righty_hand, self._contract_str, is_decl_vuln, 2, self.sampler, False)

        CardPlayer = _BEN_MODS["CardPlayer"]
        self._card_players = [
            CardPlayer(self.models, 0, lefty_hand, dummy_hand, self._contract_str, is_decl_vuln, self.sampler, pimc[0], self.dds, False),
            CardPlayer(self.models, 1, dummy_hand, decl_hand, self._contract_str, is_decl_vuln, self.sampler, pimc[1], self.dds, False),
            CardPlayer(self.models, 2, righty_hand, dummy_hand, self._contract_str, is_decl_vuln, self.sampler, pimc[2], self.dds, False),
            CardPlayer(self.models, 3, decl_hand, dummy_hand, self._contract_str, is_decl_vuln, self.sampler, pimc[3], self.dds, False),
        ]

    def _sync_history(self, game):
        deck52 = _BEN_MODS["deck52"]
        hist = game.history()
        n = len(hist)
        while self._synced_plies < n:
            p = self._synced_plies
            trick_i = p // 4
            seat, our_card52, _, _ = hist[p]
            player_i = (seat - (self._decl_i + 1)) & 3
            card52 = 51 - int(our_card52)
            card32 = deck52.card52to32(card52)

            if p == 0:
                self._opening_lead52 = card52
                self._current_trick = [card32]
                self._current_trick52 = [card52]
                self._card_players[0].hand52[card52] -= 1
                for cp in self._card_players:
                    cp.set_real_card_played(card52, 0)
                    cp.set_card_played(trick_i=0, leader_i=0, i=0, card=card32)
            else:
                if trick_i < 12:
                    for cp in self._card_players:
                        cp.set_real_card_played(card52, player_i)
                        cp.set_card_played(trick_i=trick_i, leader_i=self._leader_i, i=player_i, card=card32)
                    self._current_trick.append(card32)
                    self._current_trick52.append(card52)
                    self._card_players[player_i].set_own_card_played52(card52)
                    if player_i == 1:
                        for idx in (0, 2, 3):
                            self._card_players[idx].set_public_card_played52(card52)
                    elif player_i == 3:
                        self._card_players[1].set_public_card_played52(card52)
                    if card32 // 8 != self._current_trick[0] // 8:
                        self._shown_out_suits[player_i].add(self._current_trick[0] // 8)
                        self._discards[player_i].add((trick_i, card32))
                else:
                    self._current_trick.append(card32)
                    self._current_trick52.append(card52)
                    for cp in self._card_players:
                        cp.set_real_card_played(card52, player_i)

                if len(self._current_trick) == 4:
                    self._tricks.append(self._current_trick)
                    self._tricks52.append(self._current_trick52)
                    if self.use_pimc:
                        for cp in self._card_players:
                            if cp.pimc:
                                cp.pimc.reset_trick()
                    if trick_i < 12:
                        for i, c32 in enumerate(self._current_trick):
                            p_i = (self._leader_i + i) % 4
                            self._card_players[p_i].x_play[:, trick_i + 1, 0:32] = self._card_players[p_i].x_play[:, trick_i, 0:32]
                            self._card_players[p_i].x_play[:, trick_i + 1, 0 + c32] -= 1
                        for idx in (0, 2, 3):
                            self._card_players[idx].x_play[:, trick_i + 1, 32:64] = self._card_players[1].x_play[:, trick_i + 1, 0:32]
                        self._card_players[1].x_play[:, trick_i + 1, 32:64] = self._card_players[3].x_play[:, trick_i + 1, 0:32]
                        for cp in self._card_players:
                            for i, c32 in enumerate(self._current_trick):
                                cp.x_play[:, trick_i + 1, 64 + i * 32 + c32] = 1
                            cp.x_play[:, trick_i + 1, 288 + self._leader_i] = 1
                            cp.x_play[:, trick_i + 1, 292] = self._level
                            cp.x_play[:, trick_i + 1, 293 + self._strain_i] = 1

                    winner_rel = deck52.get_trick_winner_i(self._current_trick52, (self._strain_i - 1) % 5)
                    trick_winner = (self._leader_i + winner_rel) % 4
                    if trick_winner % 2 == 0:
                        self._card_players[0].n_tricks_taken += 1
                        self._card_players[2].n_tricks_taken += 1
                        if self.use_pimc:
                            if self._card_players[0].pimc:
                                self._card_players[0].pimc.update_trick_needed()
                            if self._card_players[2].pimc:
                                self._card_players[2].pimc.update_trick_needed()
                    else:
                        self._card_players[1].n_tricks_taken += 1
                        self._card_players[3].n_tricks_taken += 1
                        if self.use_pimc and self._card_players[3].pimc:
                            self._card_players[3].pimc.update_trick_needed()
                    for i, c32 in enumerate(self._current_trick):
                        self._player_cards_played[(self._leader_i + i) % 4].append(c32)
                    for i, c52 in enumerate(self._current_trick52):
                        self._player_cards_played52[(self._leader_i + i) % 4].append(c52)
                    self._leader_i = trick_winner
                    self._current_trick = []
                    self._current_trick52 = []

            self._synced_plies += 1

    def act(self, game) -> int:
        legal = game.legal_actions()
        if len(legal) == 1:
            return legal[0]

        c = game.contract()
        ih = game.initial_hands()
        key = (tuple(ih), c.level, c.strain, c.declarer, c.doubled, c.vul_ns, c.vul_ew)
        if key != self._deal_key or game.n_plies() < self._synced_plies:
            self._init_deal(game)
        self._sync_history(game)

        p = game.n_plies()
        if p == 0:
            BotLead = _BEN_MODS["BotLead"]
            lead_seat = (self._decl_i + 1) % 4
            bot_lead = BotLead(
                self._vuln,
                self._hands_str[lead_seat],
                self.models,
                self.sampler,
                lead_seat,
                self._dealer_i,
                self.dds,
                False,
            )
            card_resp = bot_lead.find_opening_lead(self._auction, {})
            ben_c52 = card_resp.card.code()
            our_c52 = 51 - int(ben_c52)
            assert our_c52 in legal, f"BEN lead {card_resp.card} ({our_c52}) not in legal {legal}"
            return our_c52

        trick_i = p // 4
        player_i = (game.to_move() - (self._decl_i + 1)) & 3
        cp = self._card_players[player_i]

        get_play_status = _BEN_MODS["get_play_status"]
        get_singleton = _BEN_MODS["get_singleton"]
        get_possible_cards = _BEN_MODS["get_possible_cards"]

        play_status = get_play_status(cp.hand52, self._current_trick52, self._strain_i)
        if play_status == "Forced" and self.models.autoplaysingleton:
            ben_c52 = get_singleton(cp.hand52, self._current_trick52)
            return 51 - int(ben_c52)
        if play_status == "Follow":
            result = get_possible_cards(cp.hand52, self._current_trick52)
            if result[0] != -1:
                ben_c52 = result[0] if player_i == 3 else result[1]
                return 51 - int(ben_c52)

        played_cards = [card for row in self._player_cards_played52 for card in row] + self._current_trick52
        features = {"aceking": {}}
        (
            rollout_states,
            bidding_scores,
            _c_hcp,
            _c_shp,
            quality,
            prob_occ,
            lead_scores,
            play_scores,
            logical_play_scores,
            discard_scores,
            worlds,
        ) = self.sampler.init_rollout_states(
            trick_i,
            player_i,
            self._card_players,
            played_cards,
            self._player_cards_played,
            self._shown_out_suits,
            self._discards,
            features["aceking"],
            self._current_trick,
            self._opening_lead52,
            self._auction,
            cp.hand_str,
            cp.public_hand_str,
            self._vuln,
            self.models,
            cp.get_random_generator(),
        )
        if self.use_pimc:
            cp.check_pimc_constraints(trick_i, rollout_states, quality)
        card_resp = cp.play_card(
            trick_i,
            self._leader_i,
            self._current_trick52,
            self._tricks52,
            rollout_states,
            worlds,
            bidding_scores,
            quality,
            prob_occ,
            self._shown_out_suits,
            play_status,
            lead_scores,
            play_scores,
            logical_play_scores,
            discard_scores,
            features,
        )
        ben_c52 = card_resp.card.code()
        our_c52 = 51 - int(ben_c52)
        assert our_c52 in legal, f"BEN card {card_resp.card} ({our_c52}) not in legal {legal}"
        return our_c52
