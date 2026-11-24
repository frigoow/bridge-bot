"""Python tests for bridge_cpp, QGRU/FastPolicy/BeliefNet C++ export bit-equivalence,
suit-permutation augmentation equivariance, and BEN ONNX opponent sanity."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "build"))
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "tools" / "opponents"))

import bridge_cpp
from cardbot.models import BeliefNet, FastPolicy, QGRU
from cardbot.train import dmc
from bridge.suit_perm import SuitPermAugmenter


def test_batch_env_and_belief_targets():
    env = bridge_cpp.BatchEnv(8, seed=1, threads=2)
    env.set_collect_belief(True)
    obs = env.observe()
    assert obs["state"].shape == (8, env.STATE_DIM)
    assert obs["history"].shape == (8, env.MAX_PLIES, env.PLY_DIM)
    assert obs["legal_actions"].shape == (8, env.MAX_LEGAL, env.ACTION_DIM)
    assert obs["belief_targets"].shape == (8, 52)
    assert obs["belief_marginals"].shape == (8, 52, 3)

    # At ply 0, 39 cards are hidden across 3 seats; every hidden card's marginal must be positive at its true seat!
    bt = obs["belief_targets"]
    bm = obs["belief_marginals"]
    for b in range(8):
        hidden = np.nonzero(bt[b] >= 0)[0]
        assert len(hidden) == 39
        for c in hidden:
            assert bm[b, c, bt[b, c]] > 0.0
            np.testing.assert_allclose(bm[b, c].sum(), 1.0, atol=1e-5)


def test_qgru_fast_belief_exports_match_cpp(tmp_path):
    torch.manual_seed(7)
    env = bridge_cpp.BatchEnv(4, seed=5, threads=1)
    env.set_collect_belief(True)
    for _ in range(6):
        env.step(np.zeros(4, dtype=np.int32), auto_reset=False)
    obs = env.observe()

    # 1. QGRU -> BRQ1 -> C++ QGruModel
    qnet = QGRU(env.STATE_DIM, env.PLY_DIM, env.ACTION_DIM, state_hidden=64, gru_hidden=32,
                act_hidden=32, adv_hidden=64).eval()
    qpath = tmp_path / "q.bin"
    qnet.export_bin(qpath, tag=b"BRQ1", version=1)
    qcpp = bridge_cpp.QGruModel()
    assert qcpp.load(str(qpath), "BRQ1")
    with torch.no_grad():
        valid = torch.arange(env.MAX_LEGAL).unsqueeze(0) < torch.as_tensor(obs["n_legal"]).unsqueeze(1)
        q_py = qnet(torch.as_tensor(obs["state"]), torch.as_tensor(obs["legal_actions"]),
                    torch.as_tensor(obs["history"]), torch.as_tensor(obs["n_plies"]), valid).numpy()
    for b in range(4):
        n = int(obs["n_legal"][b])
        q_c = qcpp.forward_q(obs["state"][b], obs["history"][b], int(obs["n_plies"][b]), obs["legal_actions"][b], n)
        np.testing.assert_allclose(q_c, q_py[b, :n], atol=1e-4)

    # 2. FastPolicy -> BRP1 -> C++ FastPolicyModel
    fnet = FastPolicy(env.STATE_DIM, env.POLICY_DIM, h1=64, h2=32).eval()
    fpath = tmp_path / "f.bin"
    fnet.export_bin(fpath, tag=b"BRP1", version=1)
    fcpp = bridge_cpp.FastPolicyModel()
    assert fcpp.load(str(fpath), "BRP1")
    with torch.no_grad():
        f_py = fnet(torch.as_tensor(obs["state"])).numpy()
    for b in range(4):
        np.testing.assert_allclose(fcpp.forward(obs["state"][b]), f_py[b], atol=1e-4)

    # 3. BeliefNet -> BRB1 -> C++ BeliefModel
    bnet = BeliefNet(env.STATE_DIM, env.PLY_DIM, n_cards=52, n_seats=3, state_hidden=48, gru_hidden=24).eval()
    torch.nn.init.normal_(bnet.delta_head.weight, std=0.1)
    bpath = tmp_path / "b.bin"
    bnet.export_bin(bpath, tag=b"BRB1", version=1)
    bcpp = bridge_cpp.BeliefModel()
    assert bcpp.load(str(bpath), 52, 3, "BRB1")
    with torch.no_grad():
        b_py = bnet(torch.as_tensor(obs["state"]), torch.as_tensor(obs["history"]),
                    torch.as_tensor(obs["n_plies"]), torch.as_tensor(obs["belief_marginals"])).numpy()
    for b in range(4):
        b_c = bcpp.forward_logits(obs["state"][b], obs["history"][b], int(obs["n_plies"][b]),
                                  obs["belief_marginals"][b])
        np.testing.assert_allclose(b_c, b_py[b], atol=1e-4)


def test_suit_perm_preserves_invariants():
    env = bridge_cpp.BatchEnv(16, seed=11, threads=2)
    env.set_collect_belief(True)
    for _ in range(5):
        env.step(np.zeros(16, dtype=np.int32), auto_reset=False)
    obs = env.observe()
    aug = SuitPermAugmenter("cpu")
    batch = {
        "state": torch.as_tensor(obs["state"]),
        "history": torch.as_tensor(obs["history"]),
        "actions": torch.as_tensor(obs["legal_actions"]),
        "belief_targets": torch.as_tensor(obs["belief_targets"]),
        "belief_marginals": torch.as_tensor(obs["belief_marginals"]),
    }
    perm = aug.permute_dmc_batch(batch)
    # Every 52-card block preserves total card count; trump suit (0) stays fixed for suit contracts
    for base in range(0, 520, 52):
        torch.testing.assert_close(perm["state"][:, base:base + 52].sum(dim=1),
                                   batch["state"][:, base:base + 52].sum(dim=1))
        is_suit = batch["state"][:, 549] < 0.5
        if is_suit.any():
            torch.testing.assert_close(perm["state"][is_suit, base:base + 13],
                                       batch["state"][is_suit, base:base + 13])
    # Belief marginals still sum to the same total per row
    torch.testing.assert_close(perm["belief_marginals"].sum(dim=(1, 2)),
                               batch["belief_marginals"].sum(dim=(1, 2)))


def test_mini_dmc_smoke(tmp_path):
    torch.manual_seed(3)
    env = bridge_cpp.BatchEnv(64, seed=2, threads=2)
    model = QGRU(env.STATE_DIM, env.PLY_DIM, env.ACTION_DIM, state_hidden=64, gru_hidden=32,
                 act_hidden=32, adv_hidden=64)
    cfg = dmc.DMCConfig(episodes=128, batch_size=256, mb_per_round=4, lr=1e-3,
                        ply_bucket=13, n_buckets=4, log=None)
    dmc.train(env, model, cfg)
    out = tmp_path / "mini_q.bin"
    model.export_bin(out, tag=b"BRQ1")
    # Ensure C++ searchAgent can load and play with exported QGRU!
    agent = bridge_cpp.Agent(f"search:4:dds4:rex4:q={out}:qb=0.2", 1)
    g = bridge_cpp.random_game(99, -1)
    c = agent.act(g)
    assert c in g.legal_actions()


def test_ben_onnx_opponent():
    ben_onnx = ROOT / "external" / "ben" / "models" / "onnx" / "decl_nt.onnx"
    if not ben_onnx.exists():
        pytest.skip("external/ben not fetched")
    from ben_player import BenOnnxPlayer

    ben = BenOnnxPlayer()
    g = bridge_cpp.random_game(42, -1)
    while not g.finished():
        c = ben.act(g)
        assert c in g.legal_actions()
        g.apply(c)
    assert g.tricks_ns() + g.tricks_ew() == 13


def test_collect_exit_dataset_and_train():
    from cardbot.train import exit as cb_exit

    raw = bridge_cpp.collect_exit_dataset(
        num_deals=4, worlds=4, endgame_tricks=4, root_exact_tricks=4, seed=7, threads=2
    )
    n = len(raw["n_legal"])
    assert n >= 40
    assert raw["state"].shape == (n, 570)
    assert raw["q_targets"].shape == (n, 13)
    assert raw["legal_slots"].shape == (n, 13)
    ds = cb_exit.concat_chunks([raw])
    q = QGRU(570, 66, 76, state_hidden=64, gru_hidden=32, act_hidden=32, adv_hidden=64)
    b = BeliefNet(570, 66, n_cards=52, n_seats=3, state_hidden=48, gru_hidden=24)
    f = FastPolicy(570, 52, h1=64, h2=32)
    aug = SuitPermAugmenter("cpu")
    cfg = cb_exit.ExitConfig(epochs=2, batch_size=32, log=None, device="cpu")
    m = cb_exit.train(ds, q, cfg, belief_model=b, fast_model=f, augment=aug.permute_dmc_batch)
    assert m["q_mse"] >= 0.0 and m["belief_ce"] > 0.0 and m["fast_ce"] > 0.0
