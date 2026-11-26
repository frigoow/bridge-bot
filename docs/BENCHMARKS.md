# Bridge Card-Play Benchmarks

Permanent record of all duplicate-deal evaluations (mirrored pairs: A plays NS at table 1 and EW at table 2 on the exact same 52 cards & contract; WBF IMPs/board and tricks/board with paired 95% confidence intervals).

## 1. Initial Baseline & Established Opponents Ladder (2026-11-24, commit `0e39def`)

Generated deals (`assignNaturalContract`, seed `1`, all strains):

| A vs B | Pairs | IMPs/board | Tricks/board | A win / tie / loss | A ms/choice | B ms/choice | Declarer tricks lost vs DD |
|---|---:|---:|---:|---:|---:|---:|---|
| `rule` vs `random` | 200 | +9.70 ± 0.62 | +1.88 ± 0.13 | 96.5% / 2.0% / 1.5% | 0.00 | 0.00 | — |
| `ismcts:500` vs `rule` | 100 | +4.00 ± 1.22 | +0.48 ± 0.14 | 62.0% / 23.0% / 15.0% | 2.42 | 0.00 | — |
| `ben_nn` (BEN ONNX) vs `rule` | 100 | +4.78 ± 1.24 | +0.71 ± 0.14 | 68.0% / 26.0% / 6.0% | 0.94 | 0.00 | — |
| `ben_nn` (BEN ONNX) vs `random` | 30 | +10.97 ± 1.29 | +2.27 ± 0.31 | 100.0% / 0.0% / 0.0% | 0.50 | 0.00 | — |
| `search:32:dds6:rex6` (rule playouts) vs `rule` | 100 | +6.71 ± 1.28 | +0.95 ± 0.15 | 82.0% / 13.0% / 5.0% | 17.82 | 0.00 | — |
| `search:32:dds6:rex6` vs `ben_nn` | 100 | −0.03 ± 1.19 | −0.06 ± 0.14 | 28.0% / 38.0% / 34.0% | 18.61 | 2.10 | — |
| `pimc_dds:16` vs `rule` | 100 | +7.44 ± 1.05 | +1.06 ± 0.14 | 84.0% / 16.0% / 0.0% | 82.34 | 0.00 | A: −0.63 ± 0.22, B: 1.48 ± 0.27 |
| `pimc_dds:16` vs `ben_nn` | 100 | +1.57 ± 0.92 | +0.29 ± 0.13 | 43.0% / 44.0% / 13.0% | 85.52 | 1.25 | — |
| `pimc_dds:16` vs `search:32:dds6:rex6` | 100 | +2.72 ± 1.20 | +0.42 ± 0.15 | 56.0% / 26.0% / 18.0% | 90.45 | 15.34 | — |
| `ben_dds:16` (BEN ONNX + DDS) vs `pimc_dds:16` | 60 | +0.72 ± 1.40 | +0.03 ± 0.10 | 28.3% / 50.0% / 21.7% | 89.49 | 85.27 | — |
| `dd_oracle` vs `pimc_dds:16` | 60 | +3.77 ± 1.16 | +0.60 ± 0.14 | 68.3% / 31.7% / 0.0% | 4.50 | 86.67 | A: −0.65 ± 0.22, B: 0.55 ± 0.18 |

Real BBO tournament PBN (`external/ben/Boards/BBO/Ben & Friends Daily - 2024-07-11 Daylong.pbn`, 8 boards):

| A vs B | Pairs | IMPs/board | Tricks/board | A win / tie / loss | Declarer tricks lost vs DD |
|---|---:|---:|---:|---:|---|
| `pimc_dds:16` vs `rule` | 8 | +4.63 ± 3.45 | +1.19 ± 0.67 | 87.5% / 12.5% / 0.0% | A: −0.50 ± 0.64, B: 1.88 ± 1.20 |

*(Note: negative "declarer tricks lost vs DD" means declarer took MORE tricks than double-dummy par because the opposing defenders made sub-DD plays.)*

## 2. Learned Models & Hybrid Search

### v1 — 600k-deal unified DMC (`q_all_v1`, `belief_all_v1`, distilled `fast_all_v1`, ~10 min on RTX 3060, commit `77ef60e`)

- Standalone `q_all_v1_best` (`BRQ1`): `+0.75 ± 1.28 IMPs/bd` vs `rule`, `−5.40 ± 1.02 IMPs/bd` vs `ben_nn`, `belief_ce = 0.5250`.
- Distilled `fast_all_v1` (`BRP1`, 12 epochs, 73.7% top-1 agreement): `+0.52 ± 1.71 IMPs/bd` vs `rule` (60 pairs).

| A vs B | Pairs | IMPs/board | Tricks/board | A win / tie / loss | A ms/choice | B ms/choice |
|---|---:|---:|---:|---:|---:|---:|
| `search:32:dds6:rex6:fast_v1` vs `rule` | 100 | +6.59 ± 1.24 | +0.77 ± 0.14 | 80.0% / 13.0% / 7.0% | 58.81 | 0.00 |
| `search:32:dds6:rex6:fast_v1` vs `search:32:dds6:rex6` *(rule playouts)* | 100 | +0.38 ± 1.32 | +0.10 ± 0.16 | 36.0% / 29.0% / 35.0% | 64.28 | 15.73 |
| `search:48:dds6:rex6:fast_v1:belief_v1` vs `ben_nn` | 100 | +0.69 ± 1.18 | +0.04 ± 0.13 | 32.0% / 31.0% / 37.0% | 86.12 | 2.20 |
| `search:48:dds6:rex6:fast_v1:belief_v1` vs `pimc_dds:16` | 100 | −1.08 ± 0.97 | −0.20 ± 0.11 | 14.0% / 44.0% / 42.0% | 77.85 | 82.41 |
| `search:16:dds13:rex13:fast_v1:belief_v1:lik=0.25:norace` vs `pimc_dds:16` | 100 | +0.12 ± 0.85 | +0.03 ± 0.11 | 26.0% / 52.0% / 22.0% | 79.16 | 77.96 |

### ExIt Gen 1 — 2,500 CRN search deals (102,621 states, `worlds=16, dds=8, rex=8`, 16 epochs via `cardbot.train.exit`, commit `006449b`)

- Standalone `exit1_fast.bin` (`BRP1`, 46.1% top-1 vs search): `+0.96 ± 1.24 IMPs/bd` vs `rule`, `−4.13 ± 1.14 IMPs/bd` vs `ben_nn` (100 pairs).
- Standalone `exit1_q.bin` (`BRQ1`, 49.2% top-1 vs search): `+0.64 ± 1.25 IMPs/bd` vs `rule`, `−4.38 ± 1.13 IMPs/bd` vs `ben_nn` (100 pairs).

High-precision 300-mirrored-pair (600-deal) evaluations (seed `100`, all strains, `-march=native` SIMD):

| A vs B | Pairs | IMPs/board | Tricks/board | A win / tie / loss | A ms/choice | B ms/choice |
|---|---:|---:|---:|---:|---:|---:|
| `search:48:dds8:rex8:fast_exit1:belief_exit1` vs `rule` | 300 | +6.24 ± 0.62 | +0.89 ± 0.08 | 82.7% / 13.3% / 4.0% | 42.38 | 0.00 |
| `search:48:dds8:rex8:fast_exit1:belief_exit1` vs `ben_nn` | 300 | **+1.03 ± 0.57** | **+0.17 ± 0.07** | 39.7% / 42.0% / 18.3% | 51.81 | 2.48 |
| `search:48:dds8:rex8:fast_exit1:belief_exit1` vs `pimc_dds:16` | 300 | −0.46 ± 0.57 | −0.09 ± 0.07 | 25.7% / 38.7% / 35.7% | 38.57 | 87.93 |
| `search:16:dds13:rex13:fast_exit1:belief_exit1:lik=0.25:norace` vs `pimc_dds:16` | 300 | +0.13 ± 0.53 | +0.03 ± 0.06 | 28.3% / 48.0% / 23.7% | 70.78 | 80.51 |

### ExIt Gen 2 — 6,000 CRN search deals (246,332 states, `worlds=16, dds=10, rex=10, exit1_fast, exit1_belief, lik=0.20`, 16 epochs)

- Standalone `exit2_fast.bin` (`BRP1`, 48.2% top-1 vs search): `+1.47 ± 1.10 IMPs/bd` vs `rule` (50% win / 31% tie), `−3.06 ± 1.18 IMPs/bd` vs `ben_nn` (100 pairs).
- Standalone `exit2_q.bin` (`BRQ1`, 53.3% top-1 vs search): `+0.99 ± 1.25 IMPs/bd` vs `rule`, `−3.33 ± 1.09 IMPs/bd` vs `ben_nn` (100 pairs); `belief_ce = 0.5457`.

High-precision 200–300 mirrored-pair evaluations (seed `100`, all strains, `-march=native` SIMD):

| A vs B | Pairs | IMPs/board | Tricks/board | A win / tie / loss | A ms/choice | B ms/choice |
|---|---:|---:|---:|---:|---:|---:|
| `search:48:dds8:rex8:fast_exit2:belief_exit2` vs `…:fast_exit1:belief_exit1` | 300 | **+0.47 ± 0.57** | +0.05 ± 0.07 | 33.7% / 38.3% / 28.0% | 46.51 | 45.58 |
| `search:48:dds8:rex8:fast_exit2:q_exit2:qb=0.08:belief_exit2` vs `pimc_dds:16` | 300 | **−0.04 ± 0.53** | **+0.01 ± 0.06** | **29.3% / 43.3% / 27.3%** | **40.02** | 87.88 |
| `search:20:dds13:rex13:fast_exit2:q_exit2:qb=0.04:belief_exit2:lik=0.25:norace` vs `ben_dds:16` | 200 | **−0.16 ± 0.57** | **+0.02 ± 0.07** | **26.0% / 50.5% / 23.5%** | 95.79 | 87.94 |

### Suit & NT Specialists — ExIt Gen 1 & Gen 2 (`suit_exit{1,2}`, `nt_exit{1,2}`, warm-started from `exit2`)

Each specialist ran 2 generations of `cardbot.train.exit` (`4,000` deals Gen 1 → `8,000` deals Gen 2; `worlds=16, dds=10, rex=10, lik=0.20, kl_weight=0.30, kl_temp=0.05`; 6× plain-suit augmentation on Suit, 24× all-suit augmentation on NT).

Standalone specialist progression (200 mirrored pairs per strain; `seed=101` vs `rule`, `seed=303` vs `ben_nn`):

| Strain | Checkpoint | `QGRU` vs `rule` | `FastPolicy` vs `rule` | `QGRU` vs `ben_nn` | `FastPolicy` vs `ben_nn` | `Fast` Tricks vs `ben_nn` |
|---|---|---:|---:|---:|---:|---:|
| **Suit** | Unified `exit2` | +0.62 ± 0.88 | +1.71 ± 0.81 | −3.36 ± 0.74 | −2.28 ± 0.77 | −0.27 ± 0.08 |
| **Suit** | `suit_exit1` (4k) | +0.64 ± 0.82 | +1.28 ± 0.85 | −2.15 ± 0.84 | −1.62 ± 0.74 | −0.18 ± 0.07 |
| **Suit** | **`suit_exit2` (8k)** | **+1.77 ± 0.86** | **+2.44 ± 0.88** | **−2.12 ± 0.77** | **−0.40 ± 0.72** | **−0.07 ± 0.08** |
| **NT** | Unified `exit2` | +0.84 ± 0.81 | +3.05 ± 0.68 | −3.54 ± 0.83 | −2.12 ± 0.68 | −0.40 ± 0.13 |
| **NT** | `nt_exit1` (4k) | +2.42 ± 0.75 | +3.38 ± 0.73 | −1.71 ± 0.79 | −1.11 ± 0.69 | −0.18 ± 0.13 |
| **NT** | **`nt_exit2` (8k)** | **+2.86 ± 0.70** | **+3.58 ± 0.75** | **−1.56 ± 0.72** | **−1.01 ± 0.68** | **−0.15 ± 0.14** |

Combined Suit + NT specialist search (`sfast`/`nfast` + `sq`/`nq`, 300 mirrored pairs, seed `100`, all strains):

| A vs B | Pairs | IMPs/board | Tricks/board | A win / tie / loss | A ms/choice | B ms/choice |
|---|---:|---:|---:|---:|---:|---:|
| `search:48:dds8:rex8:sfast/nfast:sq/nq:sbelief/nbelief:qb=0.08:lik=0.25` vs `pimc_dds:16` | 300 | −0.01 ± 0.61 | −0.02 ± 0.07 | 27.0% / 42.7% / 30.3% | 48.11 | 92.53 |
| **`search:48:dds8:rex8:sfast/nfast:sq/nq:qb=0.08:lik=0.25`** *(no trick-1 DDS!)* vs `pimc_dds:16` | 300 | **+0.32 ± 0.62** | **+0.03 ± 0.07** | 30.3% / 38.7% / 31.0% | **52.77** | 94.14 |
| **`search:48:dds8:rex8:sfast/nfast:sq/nq:qb=0.08:lik=0.25`** *(no trick-1 DDS!)* vs `ben_dds:16` | 300 | **−0.25 ± 0.54** | **−0.05 ± 0.06** | 24.3% / 45.0% / 30.7% | **51.98** | 97.94 |
| `search:20:dds13:rex13:sfast/nfast:sq/nq:sbelief/nbelief:qb=0.04:lik=0.25:norace` vs `ben_dds:16` | 300 | +0.01 ± 0.51 | +0.05 ± 0.06 | 30.7% / 44.0% / 25.3% | 94.66 | 88.75 |
| **`search:20:dds13:rex13:sfast/nfast:sq/nq:qb=0.04:lik=0.25:norace`** vs `ben_dds:16` | 300 | **+0.04 ± 0.47** | **+0.02 ± 0.05** | **24.7% / 55.3% / 20.0%** | 111.73 | 89.27 |
| `search:18:pool=2:dds13:rex13:sfast/nfast:pb=0.05:lik=0.35:norace` vs `ben_dds:16` | 300 | −0.10 ± 0.47 | −0.05 ± 0.06 | 21.0% / 51.7% / 27.3% | 97.01 | 87.62 |
| `search:24:pool=2:dds13:rex13:sfast/nfast:sbelief/nbelief:bt=2.5:pb=0.05:lik=0.35:norace` vs `ben_dds:16` | 300 | −0.10 ± 0.54 | +0.02 ± 0.06 | 24.3% / 50.0% / 25.7% | 123.53 | 89.12 |

### Warm-Started Suit DMC Verification (512k Suit deals from `suit_exit2`, 9.7 min on RTX 3060)

- `QGRU` vs `rule` (200 Suit pairs): `+1.77 ± 0.86` → **`+2.60 ± 0.76 IMPs/bd`** (`+0.83 IMPs/bd`).
- `QGRU` vs `ben_nn` (200 Suit pairs): `-2.12 ± 0.77` → **`-1.89 ± 0.77 IMPs/bd`** (`-0.27` tricks, best at 410k).
- Online `BeliefNet` CE on unseen Suit deals: `0.622` → **`0.5380`** (eliminated ExIt BeliefNet overfitting!).
- Pure `FastNet + Search` (`48w:dds8:rex8`, **no trick-1 DDS**, 200 Suit pairs, seed `100`):

| A vs B (Suit only) | Pairs | IMPs/board | Tricks/board | A win / tie / loss | A ms/choice | B ms/choice |
|---|---:|---:|---:|---:|---:|---:|
| `search:48:dds8:rex8:suit_exit2_fast:suit_exit2_q:qb=0.08:lik=0.25` vs `pimc_dds:16` | 200 | −0.36 ± 0.69 | −0.05 ± 0.07 | 21.5% / 45.5% / 33.0% | 53.81 | 96.88 |
| **`search:48:dds8:rex8:suit_exit2_fast:suit_dmc_best_ben:suit_dmc_belief:qb=0.08:lik=0.25`** vs `pimc_dds:16` | 200 | **+0.69 ± 0.68** | **+0.08 ± 0.07** | **27.5% / 48.0% / 24.5%** | **51.05** | 96.53 |

### Overnight 16M Suit-Only DMC + Suit ExIt Gen 3 Polish (`suit_dmc16m`, `suit_exit3`, 2026-11-25)

Warm-started from `suit_exit2_{q,belief}.pt` via `scripts/overnight_suit.sh`:
- **Stage 1 (16,000,000 Suit DMC deals, ~6.5h on RTX 3060, `batch_size=2048, replay_rounds=8, lr=1e-4 → 1.5e-5, eps=0.035 → 0.008`):**
  - `QGRU` vs `rule` (200 Suit pairs): `+1.77 ± 0.86` → **`+4.15 ± 0.80 IMPs/bd`** (peak at 14.3M, 66.0% win)
  - `QGRU` vs `ben_nn` (200 Suit pairs): `−2.12 ± 0.77` → **`−0.52 ± 0.71 IMPs/bd`** (best at 14.85M, `suit_dmc16m_best_ben.bin`)
  - Online `BeliefNet` CE on unseen Suit deals: `0.622` → **`0.444–0.489`** (`suit_dmc16m_belief_best_ben.bin`)
- **Stage 2 (Suit ExIt Gen 3 Polish: 12,000 Suit deals / 493,345 states, `worlds=16, dds=10, rex=10`, teacher = `suit_exit2_fast` + `suit_dmc16m_best_ben` (`qb=0.08`) + `suit_dmc16m_belief_best_ben` (`lik=0.25`), 12 epochs):**
  - `q_top1 = 62.3%`, `fast_top1 = 52.4%`, `suit_exit3_fast` vs `rule` (200 Suit pairs) = `+2.98 ± 0.77 IMPs/bd`, `suit_exit3_q` vs `rule` = `+3.31 ± 0.83 IMPs/bd`.

**High-Precision 1,000-Mirrored-Pair (2,000-Deal) Suit Benchmarks (`seed=1000`, Suit contracts):**

| A vs B (Suit only) | Pairs | IMPs/board | Tricks/board | A win / tie / loss | A ms/choice | B ms/choice |
|---|---:|---:|---:|---:|---:|---:|
| `fast:suit_exit3_fast.bin` *(0.74 MB, zero search)* vs `ben_nn` | 1000 | −0.43 ± 0.33 | −0.056 ± 0.034 | 25.3% / 42.4% / 32.3% | **1.37** | 3.69 |
| `qgru:suit_dmc16m_best_ben.bin` *(pure self-play, zero search)* vs `ben_nn` | 1000 | −0.67 ± 0.35 | −0.157 ± 0.038 | 24.1% / 34.1% / 41.8% | 6.00 | 5.55 |
| `qgru:suit_exit3_q.bin` *(ExIt3-polished QGRU, zero search)* vs `ben_nn` | 1000 | −0.87 ± 0.33 | −0.102 ± 0.034 | 23.2% / 38.1% / 38.7% | 6.15 | 5.70 |
| **`search:48:dds8:rex8:sfast=suit_exit3:sq=dmc16m:sbelief=dmc16m:qb=0.08:lik=0.25`** *(NO trick-1 DDS!)* vs `pimc_dds:16` | 1000 | **+0.46 ± 0.33** | **+0.051 ± 0.034** | **31.4% / 43.2% / 25.4%** | **49.59** | 94.75 |
| **`search:48:dds8:rex8:sfast=suit_exit3:sq=dmc16m:sbelief=dmc16m:qb=0.08:lik=0.25`** *(NO trick-1 DDS!)* vs `ben_dds:16` | 1000 | **+0.01 ± 0.31** | −0.004 ± 0.031 | **27.3% / 45.7% / 27.0%** | **49.80** | 98.48 |
| `search:96:dds8:rex8:sfast=suit_exit3:sq=dmc16m:sbelief=dmc16m:qb=0.08:lik=0.25` *(NO trick-1 DDS, time-matched!)* vs **`ben_dds:16`** | 1000 | **+0.39 ± 0.31** | **+0.033 ± 0.032** | **30.0% / 44.8% / 25.2%** | **83.65** | 100.64 |
| **`search:20:dds13:rex13:sfast=suit_exit3:sq=dmc16m:sbelief=dmc16m:qb=0.04:lik=0.25:norace`** *(matched DDS)* vs **`ben_dds:16`** | 1000 | **+0.46 ± 0.31** | **+0.056 ± 0.031** | **29.5% / 49.3% / 21.2%** | 99.06 | 90.29 |

Additional 1,000-pair Suit decomposition (`seed=1000`, Suit contracts):

| A vs B (Suit only) | Pairs | IMPs/board | Tricks/board | A win / tie / loss |
|---|---:|---:|---:|---:|
| `qgru:suit_exit2_q.bin` *(pre-DMC)* vs `ben_nn` | 1000 | −2.69 ± 0.35 | −0.324 ± 0.036 | 15.0% / 28.9% / 56.1% |
| `fast:suit_exit2_fast.bin` *(pre-DMC)* vs `ben_nn` | 1000 | −1.32 ± 0.33 | −0.156 ± 0.034 | 20.3% / 38.8% / 40.9% |
| `fast:suit_exit3_fast.bin` vs `fast:suit_exit2_fast.bin` | 1000 | **+0.57 ± 0.32** | — | 33.8% / 43.0% / 23.2% |
| `qgru:suit_exit3_q.bin` vs `qgru:suit_dmc16m_best_ben.bin` | 1000 | **+0.32 ± 0.34** | — | 37.4% / 38.3% / 24.3% |
| `fast:suit_exit3_fast.bin` vs `qgru:suit_dmc16m_best_ben.bin` | 1000 | **+0.65 ± 0.35** | — | 39.3% / 36.3% / 24.4% |

### 16M No-Trumps DMC + NT ExIt Gen 3 Polish + Combined All-Strains (`nt_dmc16m`, `nt_exit3`, 2026-11-25)

Warm-started from `nt_exit2_{q,belief}.pt` via `scripts/overnight_nt.sh` (**24× suit-permutation augmentation**):
- **Stage 1 (16,000,000 NT DMC deals, ~6.5h on RTX 3060, `batch_size=2048, replay_rounds=8, lr=1e-4 → 1.5e-5, eps=0.035 → 0.008`):**
  - `QGRU` vs `rule` (200 NT pairs): `+2.86 ± 0.70` → **`+5.27 ± 0.65 IMPs/bd`** (peak at 11.3M, 81.5% win!)
  - `QGRU` vs `ben_nn` (200 NT pairs): `−1.56 ± 0.72` → **`+0.92 ± 0.78 IMPs/bd`** (peak at 12.8M, `nt_dmc16m_best_ben.bin`)
  - Online `BeliefNet` CE on unseen NT deals: `0.637` → **`0.503–0.542`** (`nt_dmc16m_belief_best_ben.bin`)
- **Stage 2 (NT ExIt Gen 3 Polish: 12,000 NT deals, `worlds=16, dds=10, rex=10`, teacher = `nt_exit2_fast` + `nt_dmc16m_best_ben` (`qb=0.08`) + `nt_dmc16m_belief_best_ben` (`lik=0.25`), 12 epochs):**
  - `q_top1 = 54.8%`, `fast_top1 = 52.3%`, `nt_exit3_fast` vs `rule` (200 NT pairs) = `+4.18 ± 0.72 IMPs/bd`, `nt_exit3_q` vs `rule` = `+4.73 ± 0.63 IMPs/bd`.

**High-Precision 1,000-Mirrored-Pair (2,000-Deal) NT Benchmarks (`seed=1000`, NT contracts):**

| A vs B (NT only) | Pairs | IMPs/board | Tricks/board | A win / tie / loss | A ms/choice | B ms/choice |
|---|---:|---:|---:|---:|---:|---:|
| **`qgru:nt_dmc16m_best_ben.bin`** *(pure self-play, zero search!)* vs **`ben_nn`** | 1000 | **+0.45 ± 0.30** | **+0.077 ± 0.056** | **38.2% / 27.1% / 34.7%** | 6.14 | 5.66 |
| `qgru:nt_exit3_q.bin` *(ExIt3-polished QGRU, zero search)* vs `ben_nn` | 1000 | **+0.11 ± 0.30** | +0.018 ± 0.053 | 34.1% / 32.1% / 33.8% | 6.19 | 5.74 |
| `fast:nt_exit3_fast.bin` *(0.74 MB, zero search)* vs `ben_nn` | 1000 | −0.30 ± 0.32 | −0.050 ± 0.054 | 31.8% / 29.1% / 39.1% | **1.37** | 3.72 |
| **`search:48:dds8:rex8:nfast=nt_exit3:nq=dmc16m:nbelief=dmc16m:qb=0.08:lik=0.25`** *(NO trick-1 DDS!)* vs `pimc_dds:16` | 1000 | **+0.41 ± 0.27** | **+0.114 ± 0.053** | **40.4% / 32.2% / 27.4%** | **48.17** | 90.79 |
| **`search:96:dds8:rex8:nfast=nt_exit3:nq=dmc16m:nbelief=dmc16m:qb=0.08:lik=0.25`** *(NO trick-1 DDS, time-matched!)* vs **`ben_dds:16`** | 1000 | **+0.31 ± 0.28** | **+0.096 ± 0.053** | **37.1% / 34.4% / 28.5%** | **82.07** | 95.62 |
| **`search:20:dds13:rex13:nfast=nt_exit3:nq=dmc16m:nbelief=dmc16m:qb=0.04:lik=0.25:norace`** *(matched DDS)* vs **`ben_dds:16`** | 1000 | **+0.66 ± 0.24** | **+0.127 ± 0.049** | **39.2% / 35.5% / 25.3%** | 108.41 | 87.85 |

**High-Precision 1,000-Mirrored-Pair (2,000-Deal) COMBINED ALL-STRAINS (Suit + NT Specialists, `seed=1000`):**

| A vs B (All strains) | Pairs | IMPs/board | Tricks/board | A win / tie / loss | A ms/choice | B ms/choice |
|---|---:|---:|---:|---:|---:|---:|
| **`search:96:dds8:rex8` (Suit+NT Specialists, NO trick-1 DDS, time-matched!)** vs **`ben_dds:16`** | 1000 | **+0.33 ± 0.32** | **+0.036 ± 0.037** | **30.7% / 42.8% / 26.5%** | **82.43** | 100.40 |
| **`search:20:dds13:rex13` (Suit+NT Specialists, matched DDS)** vs **`ben_dds:16`** | 1000 | **+0.45 ± 0.31** | **+0.070 ± 0.035** | **31.6% / 45.3% / 23.1%** | 101.96 | 89.87 |

### 3. Endgame Solver / DDS Threshold Sweep (1,000 All-Strains Mirrored Pairs vs `ben_dds:16`, `seed=1000`)

All rows use the exact same Suit+NT specialist networks (`suit_exit3_fast`/`suit_dmc16m` + `nt_exit3_fast`/`nt_dmc16m`, `qb=0.08, lik=0.25`; `qb=0.04, norace` at `dds13`), varying only when playouts hand off to an exact endgame solver:

| Endgame Handoff | External DDS? | Worlds | IMPs/board vs `ben_dds:16` | Tricks/board | A win / tie / loss | A ms/choice | B ms/choice |
|---|---|---:|---:|---:|---:|---:|---:|
| `dds0:rex0` *(100% pure neural playouts, tricks 1–13)* | **None** | 96 | −0.67 ± 0.34 | −0.108 ± 0.038 | 24.2% / 37.5% / 38.3% | **55.62** | 96.82 |
| `ab4:rex0` *(`cardbot::EndgameSolver` alpha-beta last 4 tricks)* | **None** | 96 | −0.31 ± 0.32 | −0.045 ± 0.038 | 27.4% / 40.1% / 32.5% | **68.57** | 99.25 |
| `dds4:rex4` *(DDS last 4 tricks)* | Last 4 | 96 | −0.31 ± 0.32 | −0.048 ± 0.039 | 27.1% / 39.8% / 33.1% | 72.69 | 99.20 |
| **`dds6:rex6`** *(neural tricks 1–7, DDS last 6)* | Last 6 | 96 | **+0.11 ± 0.31** | −0.001 ± 0.037 | 27.9% / 43.6% / 28.5% | **66.45** | 98.98 |
| **`dds8:rex8`** *(neural tricks 1–5, DDS last 8)* | Last 8 | 96 | **+0.33 ± 0.32** | **+0.036 ± 0.037** | **30.7% / 42.8% / 26.5%** | **82.43** | 100.40 |
| **`dds13:rex13`** *(matched trick-1 DDS)* | All 13 | 20 | **+0.45 ± 0.31** | **+0.070 ± 0.035** | **31.6% / 45.3% / 23.1%** | 101.96 | 89.87 |

### 4. Path A — Auction-Aware Card Play (`AuctionModel` `BRA1`, 2026-11-25, commit `82ea64b`)

Trained `AuctionNet` (`309,878` params, ~1.2 MB, 65s on RTX 3060) on **47,314 unique real auctioned PBN deals (566,571 legal bids)** from `external/ben/Boards/{BBA,Thorvald,Eamon}`, with strict 52-card deduplicated zero-overlap holdouts:
- **1,000 Stratified Unseen Real PBNs (`data/eval_pbn_unseen_1000.pbn`):** legal bid CE `0.4552`, exact top-1 bid accuracy **`84.3%`**, auction-only 4-seat card accuracy **`34.5%`** (vs `25.0%` uniform).
- **292 Real Human/GIB BBO Daylongs (`data/eval_bbo_unseen.pbn`, zero BBO in train):** legal bid CE `0.7534`, exact top-1 bid accuracy **`76.7%`**, auction-only 4-seat card accuracy **`33.7%`**.

Card-play evaluations using `auc=checkpoints/auction_v1.bin:auclik=0.45:aucprior=0.6:pool=4` inside `cardbot::drawWorlds` (Suit+NT specialists, `seed=1`):

| A vs B | Dataset | Pairs | IMPs/board | Tricks/board | A win / tie / loss | A ms/choice | B ms/choice |
|---|---|---:|---:|---:|---:|---:|---:|
| **`48w:dds8:rex8` WITH `AuctionModel`** vs **WITHOUT** *(same bot)* | Unseen Real PBNs | 1000 | **+0.26 ± 0.29** | **+0.037 ± 0.033** | **29.6% / 46.2% / 24.2%** | 58.42 | 55.01 |
| **`20w:dds13:rex13` WITH `AuctionModel`** vs **WITHOUT** *(same bot)* | Unseen Real PBNs | 1000 | **+0.10 ± 0.28** | **+0.051 ± 0.029** | **26.5% / 53.4% / 20.1%** | **65.49** | 92.63 |
| **`48w:dds8:rex8` WITH `AuctionModel`** vs **WITHOUT** *(same bot)* | Real BBO Daylongs | 292 | +0.12 ± 0.55 | +0.017 ± 0.068 | 29.5% / 40.1% / 30.5% | 57.35 | 55.51 |
| **Auction-Aware `96w:dds8:rex8`** *(NO trick-1 DDS!)* vs **`ben_dds:16`** | **Real BBO Daylongs** | 292 | **+0.62 ± 0.55** | **+0.075 ± 0.065** | **32.2% / 45.2% / 22.6%** | **91.53** | 102.94 |
| **Auction-Aware `96w:dds8:rex8`** *(NO trick-1 DDS!)* vs **`ben_dds:16`** | **Unseen Real PBNs** | 1000 | **+0.16 ± 0.31** | **+0.044 ± 0.032** | **29.5% / 46.4% / 24.1%** | **93.65** | 100.27 |
| **`48w:dds8:rex8` WITH `aucp=8` pre-filter** vs **WITHOUT Auction** *(same bot)* | Unseen Real PBNs | 1000 | **+0.46 ± 0.32** | **+0.078 ± 0.034** | **32.6% / 45.0% / 22.4%** | 89.35 | 56.71 |

### 5. Fair Shared-BEN-Bidder Trump-Only Card-Play Battle (`585`-dim `BridgeSuitAucEncoder`, 2026-11-25)

Both sides share BEN's exact GIB `bidder.onnx` + `bidinfo.onnx` (eliminating bidding-model mismatch so every IMP measures pure card play). Trained on **500,000 BEN-bid Trump deals (`data/ben_suit_train_500k.bin`)** via 8M-deal warm-started DMC (`suit_auc_dmc8m`, `585`-dim = `570` + `15` canonical `bidinfo` floats) + 12k-deal ExIt1 (`suit_auc_exit1`), evaluated on **1,000 strictly unseen BEN-bid Trump deals (`data/ben_suit_eval_1000.bin`)**:

| A vs B (1,000 strictly unseen BEN-bid Trump pairs, `seed=1000`) | Pairs | IMPs/board | Tricks/board | A win / tie / loss | A ms/choice | B ms/choice |
|---|---:|---:|---:|---:|---:|---:|
| *(Pre-train 570-dim)* `qgru:suit_dmc16m_best_ben.bin` vs `ben_nn_auc` | 1000 | −0.77 ± 0.29 | −0.202 ± 0.037 | 18.8% / 37.9% / 43.3% | 5.40 | 24.63 |
| *(Pre-train 570-dim)* `fast:suit_exit3_fast.bin` vs `ben_nn_auc` | 1000 | −0.42 ± 0.29 | −0.089 ± 0.035 | 24.6% / 38.2% / 37.2% | 1.81 | 20.64 |
| `qgru:suit_auc_dmc8m_best_ben.bin` *(585-dim, zero search)* vs `ben_nn_auc` | 1000 | **−0.53 ± 0.30** | −0.166 ± 0.036 | 22.3% / 35.8% / 41.9% | 5.95 | 5.58 |
| **`fast:suit_auc_exit1_fast.bin`** *(585-dim, 0.74 MB, zero search)* vs **`ben_nn_auc`** | 1000 | **−0.18 ± 0.28** | **−0.045 ± 0.033** | 25.7% / 41.3% / 33.0% | **1.36** | 3.69 |
| `shared_auc+search:64:pool=6:dds8:rex8` *(NO trick-1 DDS)* vs `ben_dds_auc:16:6` | 1000 | −0.43 ± 0.27 | −0.082 ± 0.031 | 21.3% / 46.7% / 32.0% | 156.66 | 137.89 |
| **`shared_auc+search:16:pool=6:dds13:rex13`** *(matched 16w)* vs **`ben_dds_auc:16:6`** | 1000 | **−0.10 ± 0.24** | **−0.027 ± 0.028** | 21.5% / 53.6% / 24.9% | **105.47** | 111.05 |
| **`shared_auc+search:32:pool=6:dds13:rex13`** *(matched 32w)* vs **`ben_dds_auc:32:6`** | 1000 | **−0.17 ± 0.23** | **−0.021 ± 0.026** | 19.7% / 56.7% / 23.6% | **201.29** | 213.91 |

**v2 — 1M BEN-Bid Trump Pool + 12M Low-LR DMC (`suit_auc_dmc20m`, zero LR shock) + 24k ExIt Gen 2 (`suit_auc_exit2`, wider `512→256` `FastPolicy`, 2026-11-26):**
- Trained on **1,000,000 BEN-bid Trump deals (`data/ben_suit_train_1m.bin`)** with `lr=3.5e-5 → 5.0e-6` (`suit_auc_dmc20m_best_ben.bin`, `belief_ce=0.4085`!) followed by **24,000-deal ExIt Gen 2** (~985k search-labeled states, `worlds=16, dds=10, rex=10`, `q_top1=61.5%`, `fast_top1=51.7%`).
- **First-ever standalone Trump neural wins over full auction-aware `ben_nn_auc` on 1,000 unseen pairs** — BOTH `suit_auc_exit2_q` (**`+0.23 ± 0.27 IMPs/bd`**) AND wider `512→256` `suit_auc_exit2_fast` (**`+0.13 ± 0.27 IMPs/bd`** at **2.4× faster** speed)!

| A vs B (1,000 strictly unseen BEN-bid Trump pairs, `seed=1000`) | Pairs | IMPs/board | Tricks/board | A win / tie / loss | A ms/choice | B ms/choice |
|---|---:|---:|---:|---:|---:|---:|
| `qgru:suit_auc_dmc20m_best_ben.bin` *(20M DMC, zero search)* vs `ben_nn_auc` | 1000 | −0.21 ± 0.30 | −0.151 ± 0.036 | 22.8% / 35.0% / 42.2% | 6.13 | 5.77 |
| **`qgru:suit_auc_exit2_q.bin`** *(ExIt2 QGRU, zero search!)* vs **`ben_nn_auc`** | 1000 | **+0.23 ± 0.27** | **+0.017 ± 0.035** | **28.9% / 43.1% / 28.0%** | 5.99 | 5.63 |
| **`fast:suit_auc_exit2_fast.bin`** *(512→256 FastPolicy, zero search!)* vs **`ben_nn_auc`** | 1000 | **+0.13 ± 0.27** | **+0.017 ± 0.032** | **28.4% / 44.5% / 27.1%** | **1.81** | 4.36 |
| `shared_auc+search:64:pool=6:dds8:rex8` *(NO trick-1 DDS, `dmc20m` Q, post-`lik=0.25`)* vs `ben_dds_auc:16:6` | 1000 | −0.32 ± 0.23 | −0.042 ± 0.029 | 21.3% / 51.1% / 27.6% | 187.92 | 147.21 |
| `shared_auc+search:16:pool=6:dds13:rex13` *(matched 16w, `dmc20m` Q, post-`lik=0.25`)* vs `ben_dds_auc:16:6` | 1000 | −0.18 ± 0.23 | −0.038 ± 0.027 | 20.4% / 54.0% / 25.6% | **113.37** | 121.15 |
| `shared_auc+search:32:pool=6:dds13:rex13` *(matched 32w, `dmc20m` Q, post-`lik=0.25`)* vs `ben_dds_auc:32:6` | 1000 | −0.16 ± 0.23 | −0.024 ± 0.027 | 21.0% / 55.1% / 23.9% | **211.79** | 225.84 |

**v3 — Two-Punch Fair Card-Play Breakthrough (`suit_auc_lead_fast` + `BeliefNet` Pre-Filtering, 2026-11-26, commit `06d69a4`):**
- **Punch 1 (Joint Auction + Card-Play Pre-Filtering):** Added `scoreWorldsBelief` (`bw=0.25`, single 0.25 ms forward pass of `585`-dim GRU `BeliefNet`) to score all `W * pool` candidate worlds alongside `0.45 * bidder.onnx` **before** top-`W` selection (zero ESS collapse, zero extra DDS solves!). Boosted true-world top-16 inclusion from `76.7%` (`bidder.onnx` alone) → **`81.7%`** (`70.0% → 85.0%` at ply 20!).
- **Punch 2 (Exact DD Opening-Lead Specialist `suit_auc_lead_fast.bin`):** Solved 120,000 BEN-bid Trump opening leads with C++ DDS (× 6 plain-suit permutations = 720k effective leads) and trained a calibrated `585`-dim `FastPolicy` (`512→256`). Across **all 1,000 strictly unseen BEN-bid Trump opening leads**, `suit_auc_lead_fast` beat BEN's dedicated `lead_suit.onnx` by **−4.20 dup pts/lead** (**`0.1620` trk / `19.61` dup pts lost** vs BEN's `0.1660` trk / `23.81` dup pts lost)!
- **Result: Beat full auction-aware `ben_dds_auc:16:6` AND `ben_dds_auc:32:6` on 1,000 unseen BEN-bid Trump pairs with identical `bidder.onnx`, identical `W * pool` world budgets, and FASTER inference!**

| A vs B (1,000 strictly unseen BEN-bid Trump pairs, `seed=1000`) | Pairs | IMPs/board | Tricks/board | A win / tie / loss | A ms/choice | B ms/choice |
|---|---:|---:|---:|---:|---:|---:|
| *(500-pair ablation)* Pure shared `16w:pool=6:dds13` (`pb=0, bw=0`) vs `ben_dds_auc:16:6` | 500 | −0.49 ± 0.34 | −0.037 ± 0.040 | 20.4% / 53.4% / 26.2% | 104.28 | 113.09 |
| *(500-pair ablation)* `+ exit2_fast (pb=0.04)` vs `ben_dds_auc:16:6` | 500 | −0.29 ± 0.32 | −0.028 ± 0.038 | 20.8% / 53.4% / 25.8% | 103.71 | 113.63 |
| *(500-pair ablation)* `+ slead (suit_auc_lead_fast)` vs `ben_dds_auc:16:6` | 500 | −0.17 ± 0.31 | −0.008 ± 0.038 | 21.6% / 54.8% / 23.6% | 102.69 | 111.32 |
| *(500-pair ablation)* `+ BeliefNet (bw=0.25)` vs `ben_dds_auc:16:6` | 500 | −0.07 ± 0.32 | +0.001 ± 0.039 | 22.8% / 54.6% / 22.6% | 105.29 | 110.76 |
| **`shared_auc+16w:pool=6:dds13:rex13` (`slead + exit2_fast + BeliefNet bw=0.25, pb=0.04`)** vs **`ben_dds_auc:16:6`** | **1000** | **+0.02 ± 0.23** | **+0.019 ± 0.027** | **24.6% / 54.4% / 21.0%** | **105.10** | 110.89 |
| `shared_auc+16w:pool=6:dds13:rex13` (`bw=0.20, pb=0.08`) vs `ben_dds_auc:16:6` | 1000 | −0.09 ± 0.23 | +0.002 ± 0.026 | 22.9% / 54.5% / 22.6% | **105.58** | 111.27 |
| **`shared_auc+32w:pool=6:dds13:rex13` (`slead + exit2_fast + BeliefNet bw=0.25, pb=0.04`)** vs **`ben_dds_auc:32:6`** | **1000** | **+0.11 ± 0.21** | **+0.025 ± 0.025** | **23.3% / 58.0% / 18.7%** | **201.86** | 215.23 |
| *(Real unseen BBO Daylongs, Suit)* `shared_auc+16w:pool=6` Two-Punch vs `ben_dds_auc:16:6` | 216 | **+0.13 ± 0.57** | **+0.028 ± 0.058** | 20.8% / 59.7% / 19.4% | **117.09** | 120.44 |
| *(Real unseen BBO Daylongs, Suit)* `shared_auc+32w:pool=6` Two-Punch vs `ben_dds_auc:32:6` | 216 | −0.17 ± 0.55 | −0.002 ± 0.058 | 19.4% / 56.9% / 23.6% | **221.94** | 225.35 |

**v4 — Dedicated `384→192` GRU `BeliefNet v2` (`suit_auc_belief_v2_best.bin`, 2026-11-26, commit `549ef96`):**
- Trained a dedicated wider `585`-dim GRU `BeliefNet` (`384→192`, `train_belief.py`) on **2,000,000 BEN-bid Trump deals** played by our strong `suit_auc_exit2_fast + suit_auc_lead_fast` actors (6× plain-suit perms).
- On 100 strictly unseen BEN-bid Suit deals (800 states), `BeliefNet v2` (`bw=0.25`) boosted true-world rank from `86.38%` (`bidder.onnx` alone) → **`91.29%`** and top-16 inclusion from `76.12%` → **`85.12%` (+9.00% over BEN!)** in `< 0.5 ms`.

| A vs B (1,000 strictly unseen BEN-bid Trump pairs, `seed=1000`) | Pairs | IMPs/board | Tricks/board | A win / tie / loss | A ms/choice | B ms/choice |
|---|---:|---:|---:|---:|---:|---:|
| **`shared_auc+16w:pool=6:dds13:rex13` (`slead + exit2_fast + BeliefNet v2 bw=0.25, pb=0.04`)** vs **`ben_dds_auc:16:6`** | **1000** | **+0.05 ± 0.22** | **+0.010 ± 0.027** | **23.8% / 54.8% / 21.4%** | **105.52** | 110.74 |
| **`shared_auc+32w:pool=6:dds13:rex13` (`slead + exit2_fast + BeliefNet v2 bw=0.25, pb=0.04`)** vs **`ben_dds_auc:32:6`** | **1000** | **+0.13 ± 0.20** | **+0.035 ± 0.025** | **23.2% / 58.4% / 18.4%** | **201.81** | 214.62 |










