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



