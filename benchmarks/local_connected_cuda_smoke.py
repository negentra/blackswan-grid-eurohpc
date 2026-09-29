"""Small synthetic CUDA executability check; no oracle, training programme or labels."""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import torch

from benchmarks.connected_grid_calibration import (
    SEED, SCALES, TemporalGraphKernel, loss_value, tensors, topology,
)


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "results/local_connected_grid_cuda_smoke.json"


def file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def one_pass(scale: str) -> dict:
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.use_deterministic_algorithms(True)
    graph = topology(scale)
    batch = tensors(graph, 1, 2, "cuda")
    _, _, width, layers = SCALES[scale]
    model = TemporalGraphKernel(width, layers).cuda()
    model.train()
    torch.cuda.reset_peak_memory_stats()
    prediction = model(batch)
    loss = loss_value(batch, prediction)
    if not bool(torch.isfinite(prediction).all() and torch.isfinite(loss)):
        raise ArithmeticError(f"Nonfinite forward/loss at {scale}")
    loss.backward()
    gradients = [p.grad for p in model.parameters() if p.grad is not None]
    if not gradients or not all(bool(torch.isfinite(g).all()) for g in gradients):
        raise ArithmeticError(f"Nonfinite gradients at {scale}")
    model.eval()
    with torch.no_grad():
        inference = model(batch)
    if not bool(torch.isfinite(inference).all()):
        raise ArithmeticError(f"Nonfinite inference at {scale}")
    torch.cuda.synchronize()
    raw = bytes(inference.detach().cpu().contiguous().view(torch.uint8).flatten().tolist())
    return {
        "scale": scale, "nodes": graph["nodes"], "edges": len(graph["edges"]),
        "sequence_steps": 2, "batch_size": 1, "precision": "FP32",
        "model_parameters": sum(p.numel() for p in model.parameters()),
        "loss": float(loss.detach().cpu()),
        "output_sha256": hashlib.sha256(raw).hexdigest(),
        "forward_finite": True, "loss_finite": True,
        "gradients_finite": True, "inference_finite": True,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
        "topology_connected": graph["connected"],
    }


def main() -> None:
    if torch.__version__ != "2.8.0+cu128" or torch.version.cuda != "12.8":
        raise RuntimeError("Isolated PyTorch CUDA build does not match expected smoke build")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; no GPU result can be claimed")
    name = torch.cuda.get_device_name(0)
    if name != "NVIDIA GeForce RTX 3080":
        raise RuntimeError(f"Unexpected CUDA device: {name}")
    properties = torch.cuda.get_device_properties(0)
    s3_nodes, s3_edges, s3_width, s3_layers = SCALES["S3"]
    local_memory_guard = int(properties.total_memory * 0.65)
    s3_full_matrix_memory_preflight = [
        {
            "batch_size": batch_size,
            "sequence_steps": steps,
            "activation_bytes_proxy": 6 * 4 * batch_size * steps * s3_layers
                                      * (s3_nodes + s3_edges) * s3_width,
            "status": "SKIP_FULL_CONFIG_LOCAL_SAFE_MEMORY_GUARD",
        }
        for steps, batch_size in ((96, 1), (192, 1), (192, 2))
    ]
    if not all(row["activation_bytes_proxy"] > local_memory_guard
               for row in s3_full_matrix_memory_preflight):
        raise AssertionError("S3 local memory-preflight assumption changed")
    driver = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
        text=True, timeout=10).strip().splitlines()[0]
    rows = []
    for scale, expected in (("S1", 67649), ("S2", 332161), ("S3", 1582337)):
        # S3 uses the same deliberately small batch=1, T=2 check; this is
        # technical executability, not the 96/192-step calibration matrix.
        first = one_pass(scale)
        second = one_pass(scale)
        if first["model_parameters"] != expected:
            raise AssertionError(f"Actual parameter count drift at {scale}")
        if first["output_sha256"] != second["output_sha256"]:
            raise AssertionError(f"Non-deterministic synthetic inference at {scale}")
        first["deterministic_fixture_output_repeat_sha256"] = second["output_sha256"]
        first["deterministic_repeat_exact"] = True
        first["status"] = "PASS_SMALL_TECHNICAL_SMOKE"
        rows.append(first)
        torch.cuda.empty_cache()
    payload = {
        "status": "PASS_SMALL_CONNECTED_GRID_CUDA_SMOKE",
        "evidence_role": "SYNTHETIC_TECHNICAL_EXECUTABILITY_ONLY_NOT_SCIENTIFIC_EVALUATION",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "gpu_name": name, "gpu_memory_bytes": properties.total_memory,
        "driver_version": driver, "python_version": platform.python_version(),
        "torch_version": torch.__version__, "torch_cuda_runtime": torch.version.cuda,
        "installed_versions": {name: importlib.metadata.version(name)
                               for name in ("torch", "numpy", "filelock", "fsspec", "networkx", "sympy")},
        "isolated_virtual_environment": sys.prefix != sys.base_prefix,
        "cuda_available": True, "seed": SEED,
        "cublas_workspace_config": os.environ["CUBLAS_WORKSPACE_CONFIG"],
        "local_safe_memory_guard_bytes": local_memory_guard,
        "s3_full_matrix_memory_preflight": s3_full_matrix_memory_preflight,
        "s3_memory_proxy_note": "Activation proxy only; planned 96/192-step S3 rows were not allocated or executed locally",
        "kernel_sha256": file_sha(ROOT / "benchmarks/connected_grid_calibration.py"),
        "rows": rows,
        "full_calibration_matrix_executed": False,
        "convergence_training_steps": 0,
        "physical_solver_calls": 0, "scientific_labels": 0,
    }
    OUTPUT.write_bytes((json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8"))
    print(json.dumps({"status": payload["status"], "rows": [r["scale"] for r in rows]}, sort_keys=True))


if __name__ == "__main__":
    main()
