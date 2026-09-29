# Result provenance

`connected_grid_smoke_s1.json`, `s2.json`, and `s3.json` are raw small CPU fixture outputs from the reviewed kernel, with `--steps 2`, seed `19092026`, Python 3.12.10, PyTorch `2.8.0+cpu`. Exact commands from repository root:

```text
python -m benchmarks.connected_grid_calibration --scale S1 --steps 2 --device cpu --output results/connected_grid_smoke_s1.json
python -m benchmarks.connected_grid_calibration --scale S2 --steps 2 --device cpu --output results/connected_grid_smoke_s2.json
python -m benchmarks.connected_grid_calibration --scale S3 --steps 2 --device cpu --output results/connected_grid_smoke_s3.json
```

`local_connected_grid_cuda_smoke.json` is the raw V2 local RTX 3080 result. Source: `benchmarks/local_connected_cuda_smoke.py` and `benchmarks/connected_grid_calibration.py`; command: `python -m benchmarks.local_connected_cuda_smoke`; seed: `19092026`; environment: Windows CPython 3.12.10, driver 595.71, PyTorch `2.8.0+cu128`, CUDA runtime 12.8. It records small S1/S2/S3 FP32 passes and peak allocated/reserved VRAM. It is not a throughput benchmark. Its source kernel SHA-256 is recorded in the JSON. The included JSON is preserved byte-for-byte from the accepted V2 preflight package.

`playground_calibration_matrix_v2.csv` is a **prospective, unexecuted** 14-row A100 calibration plan. It contains configuration and planning values, not measured A100 results. The runner `benchmarks/run_connected_grid_matrix.py` records synchronized raw timing samples, summary statistics, memory and environment when an authorized target job eventually runs. Do not treat its empty future outputs as existing measurements.

The reviewed source ZIP is `BLACKSWAN_GRID_EUROHPC_PLAYGROUND_PREFLIGHT_V2_REVIEW.zip`, SHA-256 `ec0198f594330016432fa5e1af0b7772b6949dd4392f94ffbc4e4dedf195c7b5`. That private review ZIP is **not** included here because a release must use a strict public allowlist.
