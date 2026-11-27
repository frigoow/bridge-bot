# bridge-bot

[![CI](https://github.com/frigoow/bridge-bot/actions/workflows/ci.yml/badge.svg)](https://github.com/frigoow/bridge-bot/actions/workflows/ci.yml)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)

A **Contract Bridge card-play bot** built on the same game-independent C++20 and PyTorch architecture ([cardbot-framework](https://github.com/frigoow/cardbot-framework)) that powers our [Manille](https://github.com/frigoow/manil-bot) and [Bulgarian Belot](https://github.com/frigoow/belote-bot) bots.

All card-play neural networks (`Q-GRU`, distilled `FastPolicy`, and play-history `BeliefNet`) are trained **from scratch via self-play on a single consumer GPU (NVIDIA RTX 3060)** — with **zero human or master card-play game logs**.

> **Scope:** Card play only. Every deal begins with all 52 cards dealt and a fixed contract (and optional bidding sequence) — either generated during self-play, produced by [BEN](https://github.com/lorserker/ben)'s bidder for fair card-play comparisons, or loaded from real tournament PBN files (including unseen human BBO Daylongs).

---

## How the Bot Works

Trick-taking card games share the same core structure: **imperfect information** (hidden opponent/partner hands), **card-counting constraints** (voids revealed by failing to follow suit), **suit-permutation symmetry**, and **exact solvability in the endgame**. Rather than hand-engineering Bridge card-play heuristics, `bridge-bot` instantiates the shared `cardbot-framework` pipeline with just ~1,060 lines of Bridge rules, scoring, and canonical state encoding:

### 1. Canonical Suit Representation & Exact Symmetry (`6×` Suit / `24×` NT)
- All card and suit features map the trump suit to canonical suit `0` and the three plain suits to `1..3` (in No-Trump, all four suits are symmetric).
- Unlike 32-card spot-card abstractions (which collapse `2..7` into a single bucket), the encoder represents all **52 distinct cards** natively (`570`-dim without auction; `585`-dim when augmented with 15 canonicalized HCP/shape summary floats from `bidinfo`).
- Every training batch is augmented with exact suit permutations (`3! = 6×` on Suit contracts, `4! = 24×` on No-Trump), multiplying effective data without introducing any bias.

### 2. Self-Play Value Learning (`Q-GRU`) → Distilled `FastPolicy`
- **Stage 1 — Deep Monte Carlo (`DMC`) + Expert Iteration (`ExIt`):** Vectorized C++20 environments (`BatchEnv<BridgeGame, Encoder>`) step 1,024 games in parallel while a recurrent **`Q-GRU`** learns action values from full deal payoffs (duplicate score) using the full play history.
- **Stage 2 — `FastPolicy` Distillation:** Because running a GRU inside every rollout step is too slow, we distill the learned policy into a lightweight stateless **`512 → 256` MLP (`FastPolicy`, 1.7 MB per strain)**. In C++, `cardbot::FastPolicyModel` stores all three weight matrices transposed (column-major) and runs **sparse-input + sparse-ReLU SAXPY**, evaluating a full legal-card distribution in microseconds (`~1.8–2.0 ms/choice` standalone, **2.3× faster** than BEN's ONNX card-play network).

### 3. Play-History Belief Inference (`BeliefNet` + `bdraw` Proposal Sampling)
Standard imperfect-information Bridge bots (including PIMC+DDS and BEN) filter worlds using the **auction** (`bidder.onnx`) and **hard voids**, but draw candidate hidden hands **uniformly** at random before scoring them with the bidder. Out of $10^{12}$ possible deals, drawing 96–192 uniform deals yields very few hands that simultaneously match the auction *and* several tricks of play.

We solve this in two steps:
1. **Dedicated GRU `BeliefNet` (`384 → 192`, 7.1 MB per strain):** Reads the full 52-card play history + current state + `bidinfo` in `< 0.5 ms` and outputs calibrated per-card posterior probabilities $P(\text{hidden seat} \mid \text{obs})$ over all remaining unseen cards (trained on 1–2M self-play deals). On unseen BEN-bid deals, `BeliefNet` lifts true-world top-16 inclusion over `bidder.onnx` alone by **`+9.0 pp` on Suit (`76.1% → 85.1%`)** and **`+33.9 pp` on No-Trump (`54.9% → 88.8%`)**.
2. **Belief-Guided Proposal Sampling (`bdraw`) + Multiplicity-Preserving Deduplication:** Instead of proposing `W × pool` uniform deals, `BridgeSampler::sampleWeighted` draws the `W × pool` candidate pool **directly from `BeliefNet`** (`sbt=2.0` Suit, `nbt=1.0` NT). Identical hands are merged with their exact empirical proposal multiplicity (`ll = 0.45 * bidder_ll + log(count)`), and the top `W` **distinct** worlds are passed to search (late-game exact enumerations score every consistent world directly with `0.45 * bidder_ll + 0.25 * belief_ll`).
   - Across 320 unseen states, this lifts top-32 hidden-card placement accuracy from **`57.1% → 60.8%` (Suit)** and **`56.1% → 60.2%` (NT)** while raising auction likelihood **`9×–20×`** — at **zero extra DDS or `bidder.onnx` calls** and **11–13% faster runtime**.

### 4. Unified Configurable Exact-Endgame Handoff (`K`) & Multi-Core Search
The shared search engine (`cardbot::raceMoves`) uses one unified parameter **`K`** controlling when neural playouts hand off to an exact solver:
- **Belot & Manille (8-trick games):** 4 neural rollout tricks + exact C++ alpha-beta endgame (`K = 3–4`).
- **Bridge — 4-Trick Neural Rollouts (`K = 9`, `dds9:rex9`, zero Trick 1–4 DDS):** `FastPolicy` plays Tricks 1–4 with touching-card equivalence collapsing (`collapse`), common-random-numbers candidate racing (`raceMoves`), and a tapered root `Q-GRU` blend (`qb=0.06`); [Bo Haglund's DDS](https://github.com/dds-bridge/dds) solves the last 9 tricks. Matches the exact 4-trick rollout horizon of Belot/Manille while cutting worst-case 16-thread tail latency **25×** compared to 13-trick DDS (`146 ms` vs `3,832 ms`).
- **Bridge — Exact-from-Trick-1 (`K = 13`, `dds13:rex13`):** Evaluates every world directly with Bo Haglund's DDS (matching BEN's evaluator for strict apples-to-apples comparisons).
- **Persistent Work-Stealing Multi-Core Pool (`t<N>`):** Parallelizes world evaluations within each move while keeping `thread_local` DDS solver contexts warm (`t1 == t16` bit-identical determinism verified).

---

## Evaluation Protocol & External Opponents

All head-to-head results use **mirrored duplicate deals**: Bot A plays North–South at Table 1 and East–West at Table 2 on the exact same 52 cards, contract, and auction. We report **WBF IMPs/board** and **Tricks/board** with paired **95% confidence intervals** (`± 1.96 × SE`), win/tie/loss %, and per-decision wall-clock latency (`ms/choice`, 1 core vs 1 core per deal unless explicitly testing multi-core `t<N>`).

### Attribution & Fair Comparison vs BEN
We evaluate against [**BEN**](https://github.com/lorserker/ben) (GPL-3.0, fetched externally into gitignored `external/ben/` via `./scripts/fetch_opponents.sh` — never bundled here), a widely used open-source neural + DDS Bridge engine:
- **`ben_nn_auc` / `ben_dds_auc` ([tools/opponents/ben_player.py](tools/opponents/ben_player.py)):** Runs BEN's unmodified pretrained ONNX networks (`bidder`, `bidinfo`, `lead_suit`, `lead_nt`, `suit[0..3]`, `nt[0..3]`) paired with Bo Haglund's DDS under matched `W × 6` candidate budgets (`nn_weight=20.0`, `auc_weight=0.45`).
- **`ben_full_pimc` ([tools/opponents/ben_full_driver.py](tools/opponents/ben_full_driver.py)):** Runs BEN's complete unmodified upstream `config/onnx.conf` production driver (`BotLead`, `Sample.init_rollout_states` drawing 5,000–30,000 boards → 200 samples, `CardPlayer`, `carding.py`, `libsuitc.so` suit combinations, `BGADLL.so` PIMC declaring/defending, and `dds3`, single-threaded per worker).
- **How Bidding Info Is Used:** In shared-auction comparisons, **both** bots share BEN's exact `bidder.onnx` (exact Bayesian auction log-likelihood ranking over candidate worlds) and `bidinfo.onnx` (15 canonicalized HCP/suit-length summary floats read by our `Q-GRU`, `FastPolicy`, and `BeliefNet`), isolating 100% pure card play. For zero-BEN standalone play, our own 1.2 MB C++ [AuctionModel](cpp/bridge/auction_model.h) (`checkpoints/auction_v1.bin`) replaces `bidder.onnx`.
- **Strictly Unseen Holdouts:** Every evaluation set is strictly disjoint from training — 1,000 unseen BEN-bid Suit deals (`seed=999999`), 1,000 unseen BEN-bid NT deals (`seed=999999`), 1,000 unseen natural All-Strain deals (`seed=888888`, 782 Suit + 218 NT), and **292 real human/GIB BBO Daylong boards** (`eval_bbo_unseen.pbn`, zero BBO boards in training).

---

## Results

Mirrored duplicate deals (WBF IMPs/board and tricks/board ± 95% CI, 1 core vs 1 core per deal) against [BEN](https://github.com/lorserker/ben)'s unmodified ONNX networks (`ben_nn_auc`) and ONNX+DDS engine (`ben_dds_auc`) with identical `bidder.onnx` and `W × 6` world budgets:

| Matchup (Strictly Unseen Deals) | Mode | Pairs | IMPs/board | Tricks/board | Speed (ours vs BEN) |
|---|---|---:|---:|---:|---:|
| **NT `Q-GRU`** vs `ben_nn_auc` | NN vs NN *(zero search)* | 1,000 | **+0.88 ± 0.34** (`p < 1e-6`) | **+0.088 ± 0.050** | 6.0 vs 5.7 ms |
| **NT `FastPolicy`** (`512→256`) vs `ben_nn_auc` | NN vs NN *(zero search)* | 1,000 | **+0.52 ± 0.33** (`p = 0.002`) | **+0.064 ± 0.049** | **2.0** vs 4.6 ms |
| **Suit `Q-GRU`** vs `ben_nn_auc` | NN vs NN *(zero search)* | 1,000 | +0.23 ± 0.27 | +0.017 ± 0.035 | 6.0 vs 5.6 ms |
| **Suit `FastPolicy`** (`512→256`) vs `ben_nn_auc` | NN vs NN *(zero search)* | 1,000 | +0.13 ± 0.27 | +0.017 ± 0.032 | **1.8** vs 4.4 ms |
| **Full Bot (`16w`)** vs `ben_dds_auc:16:6` | All unseen *(Suit+NT+All+BBO)* | 3,292 | **+0.32 ± 0.14** (`p = 5e-6`) | **+0.058 ± 0.019** | **101** vs 114 ms |
| **Full Bot (`32w`)** vs `ben_dds_auc:32:6` | All unseen *(Suit+NT+All+BBO)* | 3,292 | **+0.24 ± 0.13** (`p = 3e-4`) | **+0.054 ± 0.018** | **195** vs 225 ms |
| **Full Bot (`16w`)** vs `ben_dds_auc:16:6` | **Real Unseen BBO Daylongs** | 292 | **+0.56 ± 0.52** (`p = 0.034`) | **+0.074 ± 0.065** | **104** vs 116 ms |
| **Minimal Bot (`64w`)** vs **Full Upstream BEN (`ben_full_pimc`)** | **Real Unseen BBO Daylongs** | 292 | **+0.33 ± 0.48** | **+0.057 ± 0.054** (`p = 0.039`) | **224** vs 813 ms (**3.6× faster**) |

For full per-strain breakdowns, the 1,000-pair component ablation (`FastPolicy` + `BeliefNet` `bdraw` accounting for `+0.93 IMPs/bd` over bare PIMC+DDS), neural rollout (`dds9`) & endgame sweeps (`dds0 → dds13`), and multi-core latency scaling, see [docs/BENCHMARKS.md](docs/BENCHMARKS.md).

---

## Repository Layout

| Path | Description |
|---|---|
| `cpp/bridge/` | C++20 Bridge card-play engine (`card.h`, `scoring.h`, `deal.h`, `deal_gen.h`), exact hidden-hand sampler (`sampler.h` over `cardbot::HiddenDeal<13>`), alpha-beta endgame (`endgame.h`), Bo Haglund DDS wrapper (`dds_solver.h/.cpp`), canonical neural encoder (`encoder.h/.cpp`), standalone `AuctionModel` (`auction_model.h`), hybrid search (`search.h/.cpp` over `cardbot/search.h`), and agent factory (`agents.h/.cpp`) |
| `cpp/tools/duel.cpp` | `bridge_duel`: multi-threaded C++ duplicate-deal evaluator with WBF IMPs ± 95% CI, double-dummy par diagnostics (`--dd-bench`), and PBN tournament board support (`--pbn`) |
| `cpp/python/bridge_module.cpp` | `bridge_cpp` pybind11 bindings (`BatchEnv`, `SuitAucBatchEnv`, `NtAucBatchEnv`, `SearchOnWorlds`, `BridgeGame`, `Agent`, DDS helpers, PBN/BEN deal loaders) |
| `cpp/tests/` | C++ unit tests (`test_rules`, `test_sampler`, `test_dds` cross-checking 100% against Bo Haglund's DDS, `test_search`) |
| `python/bridge/` | Training scripts built on `cardbot`: `train_dmc.py`, `train_exit.py`, `train_distill.py`, `train_belief.py`, `train_lead.py`, `train_auction.py`, canonical `suit_perm.py`, and BEN deal generator `gen_ben_deals.py` |
| `tools/opponents/` | Cross-engine duplicate-deal arena (`arena.py`), BEN ONNX+DDS adapter (`ben_player.py`), and full upstream BEN driver adapter (`ben_full_driver.py`) |
| `scripts/` | Model downloader (`download_models.sh`), external opponent fetcher (`fetch_opponents.sh`), core training pipelines (`train_suit_auc.sh`, `train_nt_auc.sh`), and historical ablation scripts (`scripts/experiments/`) |
| `docs/BENCHMARKS.md` | Permanent, chronological log of every benchmark, sweep, and ablation |
| `cardbot-framework/` | Shared game-independent C++20 & PyTorch framework ([frigoow/cardbot-framework](https://github.com/frigoow/cardbot-framework), submodule, Apache-2.0) |
| `third_party/dds/` | Bo Haglund's Double Dummy Solver ([dds-bridge/dds](https://github.com/dds-bridge/dds), submodule, Apache-2.0) |
| `external/` | Gitignored external opponents fetched by `scripts/fetch_opponents.sh`: [BEN](https://github.com/lorserker/ben) (GPL-3.0) and [bridge_learning](https://github.com/hanyuu11/bridge_learning) |

---

## Build & Test

```bash
git clone --recursive https://github.com/frigoow/bridge-bot.git
cd bridge-bot

python3 -m venv .venv
.venv/bin/pip install -r python/requirements.txt
.venv/bin/pip install --no-build-isolation -e cardbot-framework/python

.venv/bin/cmake -B build -S cpp -DCMAKE_BUILD_TYPE=Release -DBRIDGE_PYTHON=ON \
    -DPython_EXECUTABLE=$PWD/.venv/bin/python \
    -Dpybind11_DIR=$(.venv/bin/python -c 'import pybind11; print(pybind11.get_cmake_dir())')
.venv/bin/cmake --build build -j

# Run C++ unit tests (4/4) and Python parity tests (7/7)
ctest --test-dir build --output-on-failure
.venv/bin/python -m pytest python/tests -v
```

### Download Pretrained Weights & Fetch External Opponents

```bash
# Download pretrained binary checkpoints (~30 MB, Apache-2.0) into checkpoints/
./scripts/download_models.sh

# Optionally fetch BEN (GPL-3.0) and bridge_learning into gitignored external/
./scripts/fetch_opponents.sh
```

---

## Running Duels, Arena & Training

```bash
# 1. Native C++ duplicate-deal duel (with double-dummy par diagnostics)
./build/bridge_duel --a pimc_dds:16 --b rule --deals 100 --threads 16 --dd-bench

# 2. Real BBO tournament PBN boards
./build/bridge_duel --a pimc_dds:16 --b rule \
    --pbn "external/ben/Boards/BBO/Ben & Friends Daily - 2024-07-11 Daylong.pbn" --threads 16

# 3. Cross-engine arena against BEN ONNX+DDS or full upstream BEN (minimal 2-component bot)
PYTHONPATH=build:python:tools/opponents .venv/bin/python tools/opponents/arena.py \
    --a "shared_auc+search:16:pool=6:dds13:rex13:sfast=checkpoints/suit_auc_exit2_fast.bin:nfast=checkpoints/nt_auc_dmc8m_fast.bin:sbelief=checkpoints/suit_auc_belief_v2_best.bin:nbelief=checkpoints/nt_auc_belief_v3_best.bin:bdraw:sbt=2.0:nbt=1.0:bw=0.25:pb=0.04:norace" \
    --b "ben_dds_auc:16:6" \
    --pbn "external/ben/Boards/BBO/Ben & Friends Daily - 2024-07-11 Daylong.pbn" --threads 8
# Use --b "ben_full_pimc" to pit against BEN's full unmodified upstream config/onnx.conf driver!

# 4. Train from scratch (DMC Q-GRU -> ExIt/Distill FastPolicy -> Dedicated GRU BeliefNet)
PYTHONPATH=build:python .venv/bin/python -m bridge.train_dmc \
    --mode nt_auc --ben-deals data/ben_nt_train_500k.bin --episodes 8000000 --envs 1024 \
    --train-belief --out checkpoints/nt_auc_dmc8m.bin --out-belief checkpoints/nt_auc_dmc8m_belief.bin

PYTHONPATH=build:python .venv/bin/python -m bridge.train_distill \
    --mode nt_auc --ben-deals data/ben_nt_train_500k.bin \
    --teacher checkpoints/nt_auc_dmc8m_best_ben.pt --out checkpoints/nt_auc_dmc8m_fast.bin

PYTHONPATH=build:python .venv/bin/python -m bridge.train_belief \
    --ben-deals data/ben_nt_train_500k.bin --eval-ben-deals data/ben_nt_eval_1000.bin \
    --actor-fast checkpoints/nt_auc_dmc8m_fast.pt --episodes 2000000 \
    --out checkpoints/nt_auc_belief_v3.bin
```

See [scripts/train_suit_auc.sh](scripts/train_suit_auc.sh), [scripts/train_nt_auc.sh](scripts/train_nt_auc.sh), and [scripts/experiments/](scripts/experiments/) for full reproducible pipelines.

---

## Credits & License

- **`bridge-bot`** and [**`cardbot-framework`**](https://github.com/frigoow/cardbot-framework) are licensed under the **Apache License 2.0** (see [LICENSE](LICENSE)).
- [**Bo Haglund's Double Dummy Solver (`dds`)**](https://github.com/dds-bridge/dds) is included as a git submodule under the **Apache License 2.0**.
- [**BEN (`lorserker/ben`)**](https://github.com/lorserker/ben) (GPL-3.0) and [**`bridge_learning`**](https://github.com/hanyuu11/bridge_learning) are **not bundled** in this repository; they are optional external benchmark opponents fetched locally into `external/` by `scripts/fetch_opponents.sh`.
- Sister projects built on the same `cardbot-framework`: [**manil-bot**](https://github.com/frigoow/manil-bot) (Manille) and [**belote-bot**](https://github.com/frigoow/belote-bot) (Bulgarian Belot).
