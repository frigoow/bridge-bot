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

*(Updated as DMC Q-GRU, BeliefNet, FastPolicy, and specialist runs complete.)*
