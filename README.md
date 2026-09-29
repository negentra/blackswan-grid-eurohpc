# BLACKSWAN-GRID: EuroHPC compute-readiness fixture

BLACKSWAN-GRID develops methods for analysing operating states in distribution grids with large flexible loads, including 1–20 MW EV, truck and bus charging hubs with optional battery storage (BESS) and photovoltaics (PV). This repository is a **sanitized, reproducible compute fixture** for a connected-grid temporal graph workload. It is not the full BLACKSWAN-GRID research system.

**Limits:** This repository does not demonstrate operational grid safety, cross-topology scientific generalisation, a physical AC power-flow oracle, or superiority over conventional search methods. NVIDIA A100 performance has **not** been measured. The full planned 14-row connected-grid calibration matrix has **not** been executed. No Task A/B, held-out data, private method or scientific label is included.

## What is implemented and measured

The deterministic synthetic generator builds connected graphs with buses, line/transformer edges and EV/BESS/PV markers. `TemporalGraphKernel` has a graph message-passing path and recurrent temporal state. The fixture loss combines a synthetic target with an edge voltage-drop proxy; it is **not** an AC safety test. The development seed is `19092026`, unrelated to protected scientific streams.

| Scale | Nodes | Edges | Trainable parameters |
|---|---:|---:|---:|
| S1 | 96 | 110 | 67,649 |
| S2 | 384 | 440 | 332,161 |
| S3 | 1,024 | 1,200 | 1,582,337 |

Small FP32 forward/backward/inference CUDA smokes passed on a local NVIDIA GeForce RTX 3080 using PyTorch `2.8.0+cu128` and CUDA runtime `12.8`. The accepted V2 preflight ran 31/31 fixture tests, and two independent review-package builds had identical SHA-256. These are local technical checks, not target-system timing or scientific validation. This public candidate includes the exact raw [CUDA smoke JSON](results/local_connected_grid_cuda_smoke.json) and [CPU smoke JSONs](results/README.md). Physical solver calls and new scientific labels in that preflight were both zero.

## Reproduce the small fixture

Use Python 3.12 in a fresh virtual environment. Install a matching official PyTorch build first, then install this package without replacing it. For a CPU-only Linux environment:

```bash
python3.12 -m venv .venv
. .venv/bin/activate
python -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install numpy==2.4.6
python -m pip install --no-deps -e .
python -m unittest discover -s tests -v
python -m benchmarks.connected_grid_calibration --scale S1 --steps 2 --device cpu --output /tmp/s1.json
python tools/check_public.py verify
```

Repeat the smoke command with `S2` and `S3`. On a compatible NVIDIA machine, install an official CUDA PyTorch wheel in a **separate** environment and run `python -m benchmarks.local_connected_cuda_smoke`. That command deliberately requires an RTX 3080 and PyTorch `2.8.0+cu128` to reproduce the included local result. Other GPUs can run the generic small fixture with `--device cuda`, but their output is new hardware evidence and must not be presented as the included RTX 3080 measurement. No physical grid solver is called.

See [reproducibility and result provenance](docs/REPRODUCIBILITY.md) for exact commands, environment and output interpretation. `MANIFEST.sha256` covers release files; `python tools/check_public.py verify` checks every hash and the publication allowlist.

## EuroHPC calibration boundary

The proposed Leonardo Booster NVIDIA A100 Playground task is to **measure** step time, throughput and memory for the connected-grid fixture. [`results/playground_calibration_matrix_v2.csv`](results/playground_calibration_matrix_v2.csv) is a prospective plan, not measured A100 output. [`hpc/leonardo_calibration_v2.sbatch`](hpc/leonardo_calibration_v2.sbatch) is an unsubmitted example requiring an awarded account and host confirmation. Later access decisions depend on target-GPU measurements. See [scope](docs/EUROHPC_PLAYGROUND_SCOPE.md), [claims](docs/CLAIM_BOUNDARIES.md) and [security/IP boundary](docs/SECURITY_AND_IP_SCOPE.md).

Copyright © 2026 Negentra Yazılım ve Oyun Teknolojileri A.Ş. Use is governed by the [restricted research/evaluation LICENSE](LICENSE). Citation metadata is in `CITATION.cff`.
