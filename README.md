# bridge-bot

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
- **`ben_nn_auc` / `ben_dds_auc` ([tools/opponents/ben_player.py](tools/opponents/ben_player.py)):** Runs BEN's official unmodified pretrained ONNX networks (`bidder`, `bidinfo`, `lead_suit`, `lead_nt`, `suit[0..3]`, `nt[0..3]`) paired with Bo Haglund's DDS under matched `W × 6` candidate budgets (`nn_weight=20.0`, `auc_weight=0.45`). *(Note: upstream BEN's full Python application also contains optional heuristic carding rules and configuration modes not modeled in the pure ONNX+DDS benchmark driver.)*
- **Shared-Auction Fairness Control:** In Sections 1–3 below, **both** bots share BEN's exact `bidder.onnx` + `bidinfo.onnx` and identical `W × 6` world-scoring budgets so every IMP measures **100% pure card play** with zero bidding mismatch.
- **Strictly Unseen Holdouts:** Every evaluation set is strictly disjoint from training — 1,000 unseen BEN-bid Suit deals (`seed=999999`), 1,000 unseen BEN-bid NT deals (`seed=999999`), 1,000 unseen natural All-Strain deals (`seed=888888`, 782 Suit + 218 NT), and **292 real human/GIB BBO Daylong boards** (`eval_bbo_unseen.pbn`, zero BBO boards in training).

---

## Headline Results

### 1. Fair Auction-Aware Card-Play Showdown vs BEN (`ben_nn_auc` & `ben_dds_auc`)

#### Standalone Neural Networks (Zero Search) vs `ben_nn_auc`
| Our Standalone Network (Zero Search) | Unseen Dataset | Pairs | IMPs/board | Tricks/board | Win / Tie / Loss | Speed (ours vs BEN) |
|---|---|---:|---:|---:|---:|---:|
| **`nt_auc_dmc8m` (`Q-GRU`)** | Unseen NT | 1,000 | **+0.88 ± 0.34** (`p < 1e-6`) | **+0.088 ± 0.050** | 39.0% / 28.5% / 32.5% | 6.0 vs 5.7 ms |
| **`nt_auc_dmc8m_fast` (`512→256` MLP)** | Unseen NT | 1,000 | **+0.52 ± 0.33** (`p = 0.002`) | **+0.064 ± 0.049** | 36.0% / 32.2% / 31.8% | **2.0** vs 4.6 ms |
| `suit_auc_exit2_q` (`Q-GRU`) | Unseen Suit | 1,000 | +0.23 ± 0.27 | +0.017 ± 0.035 | 28.9% / 43.1% / 28.0% | 6.0 vs 5.6 ms |
| `suit_auc_exit2_fast` (`512→256` MLP) | Unseen Suit | 1,000 | +0.13 ± 0.27 | +0.017 ± 0.032 | 28.4% / 44.5% / 27.1% | **1.8** vs 4.4 ms |

#### Full Search (`BeliefNet` + `bdraw` + `FastPolicy`) vs `ben_dds_auc` (Matched `W × 6` Pool, 1 Core vs 1 Core)
| Our Bot vs `ben_dds_auc` | Unseen Dataset | Pairs | IMPs/board | Tricks/board | Win / Tie / Loss | Speed (ours vs BEN) |
|---|---|---:|---:|---:|---:|---:|
| `16w:pool=6` (`dds13:rex13`) vs `ben_dds_auc:16:6` | Unseen Suit | 1,000 | **+0.28 ± 0.22** (`p = 0.013`) | **+0.055 ± 0.026** | 26.4% / 54.8% / 18.8% | **96.6** vs 109.6 ms |
| `16w:pool=6` (`dds13:rex13`) vs `ben_dds_auc:16:6` | Unseen NT | 1,000 | +0.17 ± 0.27 | **+0.056 ± 0.041** (`p = 0.007`) | 32.4% / 42.6% / 25.0% | **107.0** vs 119.2 ms |
| `16w:pool=6` (`dds13:rex13`) vs `ben_dds_auc:16:6` | Unseen All-Strain | 1,000 | **+0.42 ± 0.24** (`p = 5e-4`) | **+0.059 ± 0.032** | 29.3% / 50.5% / 20.2% | **98.4** vs 111.5 ms |
| `16w:pool=6` (`dds13:rex13`) vs `ben_dds_auc:16:6` | **Real Unseen BBO** | 292 | **+0.56 ± 0.52** (`p = 0.034`) | **+0.074 ± 0.065** | 29.5% / 49.7% / 20.9% | **103.8** vs 115.8 ms |
| **Pooled `16w:pool=6` (all unseen)** | **All Unseen** | **3,292** | **+0.32 ± 0.14** (`p = 4.9e-6`) | **+0.058 ± 0.019** (`p = 8.3e-10`) | **29.4% / 49.3% / 21.3%** | **101.0** vs 113.7 ms (**11% faster**) |
| `32w:pool=6` (`dds13:rex13`) vs `ben_dds_auc:32:6` | Unseen Suit | 1,000 | **+0.25 ± 0.21** (`p = 0.017`) | **+0.062 ± 0.025** | 25.6% / 57.3% / 17.1% | **189.1** vs 225.2 ms |
| `32w:pool=6` (`dds13:rex13`) vs `ben_dds_auc:32:6` | Unseen NT | 1,000 | +0.23 ± 0.26 | **+0.054 ± 0.039** (`p = 0.007`) | 31.9% / 44.6% / 23.5% | **206.8** vs 233.1 ms |
| `32w:pool=6` (`dds13:rex13`) vs `ben_dds_auc:32:6` | Unseen All-Strain | 1,000 | +0.20 ± 0.23 | **+0.040 ± 0.030** (`p = 0.009`) | 26.4% / 54.0% / 19.6% | **186.5** vs 215.5 ms |
| `32w:pool=6` (`dds13:rex13`) vs `ben_dds_auc:32:6` | **Real Unseen BBO** | 292 | +0.38 ± 0.52 | **+0.072 ± 0.059** (`p = 0.017`) | 28.4% / 51.0% / 20.5% | **199.4** vs 224.6 ms |
| **Pooled `32w:pool=6` (all unseen)** | **All Unseen** | **3,292** | **+0.24 ± 0.13** (`p = 3.3e-4`) | **+0.054 ± 0.018** (`p = 2.0e-9`) | **28.0% / 51.9% / 20.1%** | **194.6** vs 224.6 ms (**13% faster**) |
| **`64w:pool=6` (`dds13:rex13`)** vs `ben_dds_auc:32:6` | Unseen All-Strain | 300 | **+0.52 ± 0.41** (`p = 0.014`) | **+0.090 ± 0.061** (`p = 0.004`) | 28.3% / 53.7% / 18.0% | 355.6 *(107 ms @16t)* vs 218.2 ms |
| **4-Trick Neural Rollout `128w:pool=4` (`dds9:rex9`, NO Trick 1–4 DDS)** vs `ben_dds_auc:32:6` | Unseen All-Strain | 500 | **+0.27 ± 0.36** | +0.023 ± 0.047 | 26.6% / 51.2% / 22.2% | **241.4** *(51 ms @16t)* vs 256.3 ms |

---

### 2. 1,000-Pair Component Ablation (Isolating Every Component)

Controlled 1,000-mirrored-pair ablation on `data/ben_all_eval_1000.bin` (`16w:pool=6` vs `ben_dds_auc:16:6`, identical seed):

| # | Configuration | IMPs/board | Tricks/board | Win / Tie / Loss | Delta | Speed (ours vs BEN) |
|---|---|---:|---:|---:|---:|---:|
| **A1** | **Bare** (`bidder.onnx` + uniform + DDS, zero learned nets) | −0.518 ± 0.238 | −0.049 ± 0.032 | 21.2% / 51.4% / 27.4% | — | 101.5 vs 109.0 ms |
| **A2** | `+ Distilled FastPolicy` (`pb=0.04`) | −0.084 ± 0.233 | −0.004 ± 0.031 | 24.2% / 51.9% / 23.9% | **+0.434** | 101.2 vs 110.7 ms |
| **A3** | `+ Opening-Lead Specialists` (`slead + nlead`) | −0.139 ± 0.236 | −0.013 ± 0.031 | 22.8% / 53.9% / 23.3% | −0.055 *(redundant)* | 102.1 vs 110.7 ms |
| **A4** | `+ GRU BeliefNet Re-Scoring` (`UNIF + bw=0.25`) | +0.077 ± 0.237 | +0.028 ± 0.031 | 25.5% / 52.8% / 21.7% | **+0.216** | 105.3 vs 109.8 ms |
| **Full** | `+ BeliefNet Proposal Sampling + Dedup` (`bdraw`, with lead) | **+0.419 ± 0.235** | **+0.059 ± 0.032** | **29.3% / 50.5% / 20.2%** | **+0.342** | **98.4** vs 111.5 ms |
| **A5** | **Pure `bdraw + dedup` ONLY** *(no FastPolicy, no Lead)* | **+0.072 ± 0.256** | **+0.033 ± 0.032** | 27.7% / 49.3% / 23.0% | **+0.590** *(vs A1)* | **97.4** vs 109.0 ms |
| **A6** | **Minimal 2-Component** (`FastPolicy + BeliefNet bdraw+dedup`, **NO lead specialists**) | **+0.409 ± 0.236** (`p = 7e-4`) | **+0.054 ± 0.031** (`p = 6e-4`) | **28.1% / 52.1% / 19.8%** | **+0.927** *(vs A1)* | **97.4** vs 110.9 ms |

**Key insight:** Just **two generic `cardbot-framework` components** (`FastPolicy` + GRU `BeliefNet` with `bdraw + dedup`, `14.2 MB` total, zero Bridge-specific extra networks) deliver the entire **`+0.93 IMPs/bd`** gain over bare auction-filtered PIMC+DDS (`A6`: `+0.409 ± 0.236` vs `Full`: `+0.419 ± 0.235`).

---

### 3. Multi-Core Per-Move Latency Scaling (8-Core / 16-Thread CPU)

Measured across 120 unseen All-Strain decision states (plies 0–20) using `cardbot::detail::SearchPool` (`t1 == t16` bit-identical):

| Config | `T=1` Mean | `T=4` Mean | `T=8` Mean | `T=16` Mean | `T=16` p95 | `T=16` Max (tail) | Speedup |
|---|---:|---:|---:|---:|---:|---:|---:|
| `dds13:rex13 W=16` | 161.7 ms | 57.8 ms | 46.6 ms | **45.0 ms** | 187.0 ms | 626.4 ms | 3.60× |
| `dds13:rex13 W=32` | 423.5 ms | 149.6 ms | 119.3 ms | **116.8 ms** | 357.6 ms | **3,832.5 ms** | 3.62× |
| `dds13:rex13 W=64` | 673.0 ms | 185.2 ms | 119.6 ms | **107.0 ms** | 436.8 ms | 1,200.4 ms | 6.29× |
| `roll dds9 W=128` | 252.0 ms | 73.1 ms | 56.0 ms | **51.2 ms** | 124.2 ms | **146.2 ms** | 4.93× |
| `roll dds8 W=1,000` | 563.3 ms | 160.3 ms | 106.7 ms | **96.5 ms** | 274.2 ms | 402.5 ms | **5.84×** |

13-trick DDS suffers severe straggler stalls (one tangled board stalls `W=32` for `3.83s` even on 16 threads), whereas 4-trick neural rollouts (`roll dds9 W=128`) run in **`51.2 ms` mean / `146.2 ms` max** (**26× lower tail latency**).

---

### 4. No-Auction & Self-Contained `AuctionModel` Results (Zero BEN Bidder)

Without using BEN's `bidder.onnx` at all (1,000 mirrored pairs vs no-auction `ben_dds:16`):
- **Pure Neural Rollouts + Late Endgame (`96w:dds8:rex8`, NO Trick 1–5 DDS):** **`+0.33 ± 0.32 IMPs/bd`**, `+0.036 ± 0.037 trk`, **`82.4` vs `100.4 ms`** (18% faster).
- **Matched DDS (`20w:dds13:rex13`):** **`+0.45 ± 0.31 IMPs/bd`**, **`+0.070 ± 0.035 trk`** (`102.0` vs `89.9 ms`).
- **Our Own 1.2 MB `AuctionModel` (`auction_v1.bin`) + `96w:dds8:rex8` on 292 Real Unseen BBO Daylongs vs `ben_dds:16`:** **`+0.62 ± 0.55 IMPs/bd`**, **`91.5` vs `102.9 ms`**.

Full chronological benchmark logs, endgame-threshold sweeps (`dds0 → ab4 → dds6 → dds8 → dds13`), and baseline ladders (`random` → `rule` → `ismcts` → `ben_nn` → `pimc_dds` → `ben_dds` → `dd_oracle`) are documented in [docs/BENCHMARKS.md](docs/BENCHMARKS.md).

---

## Repository Layout

| Path | Description |
|---|---|
| `cpp/bridge/` | C++20 Bridge card-play engine (`card.h`, `scoring.h`, `deal.h`, `deal_gen.h`), exact hidden-hand sampler (`sampler.h` over `cardbot::HiddenDeal<13>`), alpha-beta endgame (`endgame.h`), Bo Haglund DDS wrapper (`dds_solver.h/.cpp`), canonical neural encoder (`encoder.h/.cpp`), standalone `AuctionModel` (`auction_model.h`), hybrid search (`search.h/.cpp` over `cardbot/search.h`), and agent factory (`agents.h/.cpp`) |
| `cpp/tools/duel.cpp` | `bridge_duel`: multi-threaded C++ duplicate-deal evaluator with WBF IMPs ± 95% CI, double-dummy par diagnostics (`--dd-bench`), and PBN tournament board support (`--pbn`) |
| `cpp/python/bridge_module.cpp` | `bridge_cpp` pybind11 bindings (`BatchEnv`, `SuitAucBatchEnv`, `NtAucBatchEnv`, `SearchOnWorlds`, `BridgeGame`, `Agent`, DDS helpers, PBN/BEN deal loaders) |
| `cpp/tests/` | C++ unit tests (`test_rules`, `test_sampler`, `test_dds` cross-checking 100% against Bo Haglund's DDS, `test_search`) |
| `python/bridge/` | Training scripts built on `cardbot`: `train_dmc.py`, `train_exit.py`, `train_distill.py`, `train_belief.py`, `train_lead.py`, `train_auction.py`, canonical `suit_perm.py`, and BEN deal generator `gen_ben_deals.py` |
| `tools/opponents/` | Cross-engine duplicate-deal arena (`arena.py`) and BEN ONNX+DDS opponent adapter (`ben_player.py`) |
| `scripts/` | Reproducible end-to-end training scripts (`train_suit_auc.sh`, `train_nt_auc.sh`, `overnight_*.sh`) and opponent fetcher (`fetch_opponents.sh`) |
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

### Fetch External Opponents (BEN & `bridge_learning`)

```bash
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

# 3. Cross-engine arena against BEN ONNX+DDS (minimal 2-component bot)
PYTHONPATH=build:python:tools/opponents .venv/bin/python tools/opponents/arena.py \
    --a "shared_auc+search:16:pool=6:dds13:rex13:sfast=checkpoints/suit_auc_exit2_fast.bin:nfast=checkpoints/nt_auc_dmc8m_fast.bin:sbelief=checkpoints/suit_auc_belief_v2_best.bin:nbelief=checkpoints/nt_auc_belief_v3_best.bin:bdraw:sbt=2.0:nbt=1.0:bw=0.25:pb=0.04:norace" \
    --b "ben_dds_auc:16:6" --pbn data/eval_bbo_unseen.pbn --threads 16

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

See [scripts/train_suit_auc.sh](scripts/train_suit_auc.sh), [scripts/overnight_suit_auc_v2.sh](scripts/overnight_suit_auc_v2.sh), and [scripts/train_nt_auc.sh](scripts/train_nt_auc.sh) for full reproducible pipelines.

---

## Credits & License

- **`bridge-bot`** and [**`cardbot-framework`**](https://github.com/frigoow/cardbot-framework) are licensed under the **Apache License 2.0** (see [LICENSE](LICENSE)).
- [**Bo Haglund's Double Dummy Solver (`dds`)**](https://github.com/dds-bridge/dds) is included as a git submodule under the **Apache License 2.0**.
- [**BEN (`lorserker/ben`)**](https://github.com/lorserker/ben) (GPL-3.0) and [**`bridge_learning`**](https://github.com/hanyuu11/bridge_learning) are **not bundled** in this repository; they are optional external benchmark opponents fetched locally into `external/` by `scripts/fetch_opponents.sh`.
- Sister projects built on the same `cardbot-framework`: [**manil-bot**](https://github.com/frigoow/manil-bot) (Manille) and [**belote-bot**](https://github.com/frigoow/belote-bot) (Bulgarian Belot).
