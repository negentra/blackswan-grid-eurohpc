"""Short connected-grid timing matrix for a *future* authorized Playground job.

This module never invokes a grid solver or accesses scientific source splits.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
from io import BytesIO
import json
import math
import platform
import statistics
import subprocess
import time
from pathlib import Path

import torch

from benchmarks.connected_grid_calibration import (
    SEED, SCALES, TemporalGraphKernel, loss_value, tensors, topology,
)


def synchronize(device: str) -> None:
    if device == "cuda":
        torch.cuda.synchronize()


def timed(call, device: str, count: int) -> list[float]:
    samples = []
    for _ in range(count):
        synchronize(device)
        start = time.perf_counter()
        call()
        synchronize(device)
        samples.append(time.perf_counter() - start)
    return samples


def stats(values: list[float]) -> dict:
    return {"raw_seconds": values, "mean": statistics.mean(values),
            "median": statistics.median(values),
            "sample_stdev": statistics.stdev(values) if len(values) > 1 else 0.0,
            "minimum": min(values), "maximum": max(values)}


def measure(row: dict, device: str, *, smoke: bool = False) -> dict:
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    torch.manual_seed(SEED)
    scale = row["scale"]
    graph = topology(scale)
    batch = tensors(graph, int(row["batch_size"]), int(row["time_steps"]), device)
    _, _, width, layers = SCALES[scale]
    model = TemporalGraphKernel(width, layers).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    precision = row["precision"]
    use_amp = precision == "BF16_CONDITIONAL"
    if use_amp and device != "cuda":
        return {"status": "SKIPPED_BF16_REQUIRES_CUDA", "row": row}
    if use_amp and not torch.cuda.is_bf16_supported():
        return {"status": "SKIPPED_BF16_UNSUPPORTED", "row": row}
    if device == "cuda":
        estimated = int(row["estimated_activation_bytes"])
        total = torch.cuda.get_device_properties(0).total_memory
        if estimated > .65 * total:
            return {"status": "SKIPPED_PREALLOC_MEMORY_GUARD", "row": row,
                    "estimated_activation_bytes": estimated, "gpu_total_bytes": total}
        torch.cuda.reset_peak_memory_stats()

    def forward():
        with torch.autocast("cuda", enabled=use_amp, dtype=torch.bfloat16):
            result = model(batch)
            loss = loss_value(batch, result)
        if not bool(torch.isfinite(loss)):
            raise ArithmeticError("Nonfinite synthetic loss")
        return loss

    def train():
        optimizer.zero_grad(set_to_none=True)
        loss = forward()
        loss.backward()
        if not all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in model.parameters()):
            raise ArithmeticError("Nonfinite synthetic gradient")
        optimizer.step()

    warmup = 1 if smoke else int(row["warmup_train_steps_per_repeat"])
    train_count = 2 if smoke else int(row["timed_train_steps_per_repeat"])
    forward_count = 2 if smoke else int(row["timed_forward_steps_per_repeat"])
    repeats = 1 if smoke else int(row["repeats"])
    all_train, all_forward = [], []
    try:
        for _ in range(repeats):
            for _ in range(warmup):
                train()
            all_train.extend(timed(train, device, train_count))
            with torch.no_grad():
                all_forward.extend(timed(forward, device, forward_count))
    except torch.cuda.OutOfMemoryError:
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()
        return {"status": "OOM_CAPACITY_FINDING", "row": row}
    checkpoint = BytesIO()
    checkpoint_start = time.perf_counter()
    torch.save(model.state_dict(), checkpoint)
    checkpoint_write_seconds = time.perf_counter() - checkpoint_start
    driver = None
    if device == "cuda":
        try:
            driver = subprocess.run(
                ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                capture_output=True, text=True, check=True, timeout=10,
            ).stdout.strip().splitlines()[0]
        except (OSError, subprocess.SubprocessError, IndexError):
            driver = "UNAVAILABLE"
    result = {
        "status": "PASS", "row": row, "seed": SEED,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "topology_connected": graph["connected"],
        "parameter_count_actual": sum(p.numel() for p in model.parameters()),
        "training_step_seconds": stats(all_train),
        "forward_seconds": stats(all_forward),
        "sequences_per_second_train": int(row["batch_size"]) / statistics.mean(all_train),
        "sequences_per_second_forward": int(row["batch_size"]) / statistics.mean(all_forward),
        "warmup_per_repeat": warmup, "measured_train_per_repeat": train_count,
        "measured_forward_per_repeat": forward_count, "repeats": repeats,
        "precision": precision, "device": device,
        "torch_version": torch.__version__, "torch_cuda_runtime": torch.version.cuda,
        "driver_version": driver,
        "gpu_count_visible": torch.cuda.device_count() if device == "cuda" else 0,
        "gpu_total_memory_bytes": torch.cuda.get_device_properties(0).total_memory if device == "cuda" else None,
        "host_architecture": platform.machine(),
        "gpu_name": torch.cuda.get_device_name(0) if device == "cuda" else None,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(0) if device == "cuda" else None,
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(0) if device == "cuda" else None,
        "checkpoint_bytes": checkpoint.tell(),
        "checkpoint_write_seconds": checkpoint_write_seconds,
        "data_loader_wait_seconds": 0.0,
        "data_loader_note": "Deterministic synthetic tensor already resident before timing",
        "measurement_note": "Synthetic compute calibration only; no AC safety labels",
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix", type=Path, default=Path("results/playground_calibration_matrix.csv"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--row-index", type=int)
    parser.add_argument("--allow-bf16", action="store_true")
    args = parser.parse_args()
    rows = list(csv.DictReader(args.matrix.open(newline="", encoding="utf-8")))
    if args.smoke:
        rows = [rows[0]]
    elif args.row_index is not None:
        if not 0 <= args.row_index < len(rows):
            raise ValueError("Invalid frozen matrix row index")
        rows = [rows[args.row_index]]
    if any(row["precision"] == "BF16_CONDITIONAL" for row in rows) and not args.allow_bf16:
        raise ValueError("BF16 rows require a separate FP32 equivalence review and --allow-bf16")
    results = [measure(row, args.device, smoke=args.smoke) for row in rows]
    payload = {"status": "SYNTHETIC_CALIBRATION_ONLY", "rows": results,
               "scientific_evaluation": False}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes((json.dumps(payload, indent=2, sort_keys=True) + "\n").encode())
    print(json.dumps({"rows": len(results), "statuses": [r["status"] for r in results]}))


if __name__ == "__main__":
    main()
