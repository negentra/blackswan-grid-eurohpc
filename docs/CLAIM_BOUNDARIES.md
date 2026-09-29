# Claim boundaries

| Statement | Status | Public evidence |
|---|---|---|
| S1/S2/S3 connected synthetic graph sizes and model parameters | Implemented and testable | Kernel, config, CPU smoke JSONs, tests |
| Small FP32 CUDA passes on RTX 3080, PyTorch 2.8.0+cu128 | Locally measured technical smoke | Raw `results/local_connected_grid_cuda_smoke.json`, exact source and command |
| 31/31 accepted V2 fixture tests; deterministic V2 review ZIP rebuild | Historical accepted preflight fact | V2 package SHA in `results/README.md`; not a current CI count |
| A100 step time, throughput and memory | UNMEASURED | Proposed calibration matrix only |
| Operational safety, AC oracle, cross-topology generalisation, method superiority | NOT ESTABLISHED | Outside this repository |

This public fixture contains no scientific labels, physical solver outputs, Task A/B evidence or held-out data. Planning numbers are never relabelled as measurements.
