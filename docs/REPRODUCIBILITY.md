# Reproducibility

This repository is self-contained for the small synthetic compute fixture. It does not require the Negentra research workspace, protected source rows or a power-flow solver.

Use Python 3.12. The CPU CI environment installs official CPU PyTorch 2.8.0 and NumPy 2.4.6, then `pip install --no-deps -e .`. On Linux:

```bash
python3.12 -m venv .venv
. .venv/bin/activate
python -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install numpy==2.4.6
python -m pip install --no-deps -e .
python -m unittest discover -s tests -v
python tools/check_public.py verify
python -m benchmarks.connected_grid_calibration --scale S1 --steps 2 --device cpu --output /tmp/connected-s1.json
python -m benchmarks.connected_grid_calibration --scale S2 --steps 2 --device cpu --output /tmp/connected-s2.json
python -m benchmarks.connected_grid_calibration --scale S3 --steps 2 --device cpu --output /tmp/connected-s3.json
```

The source implementation is `benchmarks/connected_grid_calibration.py`; it fixes graph construction, model, loss and seed. `configs/connected_grid_reference.json` independently records expected shapes and parameter counts; tests check agreement. Repeated outputs should match in a fixed environment. Floating-point bit patterns can differ between CPU/GPU, PyTorch builds and platforms; cross-platform validation checks graph/hash/shape, finite forward/loss/gradients and parameter counts rather than imposing an unjustified bitwise numerical equality.

For the **included RTX 3080** result, create a separate CUDA environment using an official PyTorch `2.8.0+cu128` wheel compatible with the installed driver, install NumPy 2.4.6 and this package with `--no-deps`, then run `python -m benchmarks.local_connected_cuda_smoke`. The script checks hardware/build identity, sets the deterministic cuBLAS workspace, executes two small FP32 repeats per scale and writes `results/local_connected_grid_cuda_smoke.json`. Save the included reference file elsewhere first if you want to preserve its bytes. Do not run the 14-row matrix to reproduce the preflight; it was never run in that evidence layer.

`python tools/check_public.py verify` validates `MANIFEST.sha256`, scans filenames and contents for prohibited material, and rejects unlisted publishable files. CI runs the same check and a synthetic CPU smoke. The manifest excludes itself to avoid a circular digest.
