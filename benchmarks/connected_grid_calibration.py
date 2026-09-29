"""Synthetic connected-grid temporal *compute* calibration; never an AC oracle.

Inputs are generated from published shapes and arithmetic rules. No P0, CALIB,
EVAL, CINELDI, customer data, or reserved scientific RNG is imported.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import statistics
import time
from pathlib import Path

import torch
from torch import nn


SCALES = {"S1": (96, 110, 64, 4), "S2": (384, 440, 128, 6),
          "S3": (1024, 1200, 256, 8)}
SEED = 19092026  # Public compute-fixture seed, not a BLACKSWAN scientific stream.


def canonical_sha(value: object) -> str:
    return hashlib.sha256((json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()).hexdigest()


def topology(scale: str) -> dict:
    n, m, _, _ = SCALES[scale]
    # A spanning tree guarantees connectivity; deterministic chords create meshing.
    edges = [(i - 1, i) for i in range(1, n)]
    for j in range(m - n + 1):
        src = (j * 37) % (n - 3)
        edges.append((src, src + 2 + j % 2))
    assert len(edges) == m
    seen, stack = {0}, [0]
    adjacency = [[] for _ in range(n)]
    for src, dst in edges:
        adjacency[src].append(dst)
        adjacency[dst].append(src)
    while stack:
        for neighbour in adjacency[stack.pop()]:
            if neighbour not in seen:
                seen.add(neighbour)
                stack.append(neighbour)
    if len(seen) != n:
        raise ValueError("Graph is disconnected")
    kinds = ["transformer" if i == 0 or (i + 1) % 113 == 0 else "line"
             for i in range(m)]
    assets = {"EV": [i for i in range(7, n, 29)],
              "BESS": [i for i in range(13, n, 61)],
              "PV": [i for i in range(11, n, 37)]}
    return {"scale": scale, "nodes": n, "edges": edges, "edge_kinds": kinds,
            "assets": assets, "connected": True, "source_bus": 0,
            "provenance": "DETERMINISTIC_SYNTHETIC_CONNECTED_GRAPH_NOT_A_REAL_FEEDER"}


def tensors(graph: dict, batch: int, steps: int, device: str) -> dict:
    if batch < 1 or steps < 1:
        raise ValueError("batch and time must be positive")
    n, m = graph["nodes"], len(graph["edges"])
    edge_index = torch.tensor(graph["edges"], dtype=torch.long, device=device).T.contiguous()
    edge = torch.zeros(m, 8, device=device)
    for i, kind in enumerate(graph["edge_kinds"]):
        edge[i, 0] = float(kind == "line")
        edge[i, 1] = float(kind == "transformer")
        edge[i, 2] = .002 + (i % 7) * .0002  # synthetic r pu
        edge[i, 3] = .004 + (i % 11) * .0003  # synthetic x pu
        edge[i, 4] = 2.0 + (i % 5) * .2       # synthetic rating MVA
        edge[i, 5] = 1.0                     # tap
        edge[i, 6] = 0.0                     # phase shift
        edge[i, 7] = float(i % 2)            # corridor marker
    node = torch.zeros(batch, steps, n, 12, device=device)
    node[:, :, 0, 0] = 1.0  # source-bus marker
    node[:, :, :, 1] = 20.0 / 110.0  # nominal voltage scale
    node[:, :, 0, 1] = 1.0
    for i in range(n):
        node[:, :, i, 2] = .02 + .001 * (i % 13)
        node[:, :, i, 3] = .01 + .0005 * (i % 11)
        node[:, :, i, 7] = (i % 7) / 7.0
        node[:, :, i, 8] = (i % 11) / 11.0
        node[:, :, i, 9] = float(i % 3 == 0)
        node[:, :, i, 10] = (i % 17) / 17.0
        node[:, :, i, 11] = 1.0
    for kind, feature in (("EV", 4), ("BESS", 5), ("PV", 6)):
        for i in graph["assets"][kind]:
            node[:, :, i, feature] = .1
    for t in range(steps):
        temporal = .5 + .5 * math.sin(2 * math.pi * (t % 24) / 24)
        node[:, t, :, 2] *= .7 + .6 * temporal
        node[:, t, :, 4] *= .5 + temporal
        node[:, t, :, 6] *= max(0.0, math.sin(math.pi * (t % 24) / 24))
    target = 1.0 - .01 * node[..., 2] + .005 * node[..., 6]
    return {"node": node, "edge": edge, "edge_index": edge_index,
            "target": target, "graph": graph}


class TemporalGraphKernel(nn.Module):
    def __init__(self, width: int, layers: int):
        super().__init__()
        self.input = nn.Linear(12, width)
        self.edge = nn.Linear(2 * width + 8, width)
        self.node_layers = nn.ModuleList(nn.Linear(2 * width, width) for _ in range(layers))
        self.temporal = nn.GRUCell(width, width)
        self.head = nn.Linear(width, 1)

    def forward(self, batch: dict) -> torch.Tensor:
        x, edge = batch["node"], batch["edge"]
        src, dst = batch["edge_index"]
        b, t, n, _ = x.shape
        state = x.new_zeros((b, n, self.head.in_features))
        output = []
        for k in range(t):
            h = torch.tanh(self.input(x[:, k]))
            for layer in self.node_layers:
                message = torch.tanh(self.edge(torch.cat(
                    (h[:, src], h[:, dst], edge.expand(b, -1, -1)), dim=-1)))
                aggregate = torch.zeros_like(h).index_add_(1, dst, message)
                aggregate = aggregate.index_add_(1, src, message)
                h = torch.tanh(layer(torch.cat((h, aggregate), -1)))
            state = self.temporal(h.reshape(b * n, -1), state.reshape(b * n, -1)).reshape(b, n, -1)
            output.append(self.head(state).squeeze(-1))
        return torch.stack(output, dim=1)


def loss_value(batch: dict, predicted: torch.Tensor) -> torch.Tensor:
    src, dst = batch["edge_index"]
    voltage_drop = predicted[..., dst] - predicted[..., src]
    proxy = batch["edge"][:, 2].reshape(1, 1, -1) * batch["node"][..., dst, 2]
    return (predicted - batch["target"]).square().mean() + .1 * (voltage_drop + proxy).square().mean()


def smoke(scale: str = "S1", device: str = "cpu", steps: int = 2) -> dict:
    torch.manual_seed(SEED)
    torch.use_deterministic_algorithms(True)
    graph = topology(scale)
    batch = tensors(graph, 1, steps, device)
    if batch["node"].shape != (1, steps, graph["nodes"], 12):
        raise AssertionError("Node/sequence dimensions changed")
    if batch["edge"].shape != (len(graph["edges"]), 8):
        raise AssertionError("Edge feature dimensions changed")
    if not bool(torch.isfinite(batch["node"]).all() and torch.isfinite(batch["edge"]).all()):
        raise AssertionError("Nonfinite graph features")
    _, _, width, layers = SCALES[scale]
    model = TemporalGraphKernel(width, layers).to(device)
    prediction = model(batch)
    loss = loss_value(batch, prediction)
    if not bool(torch.isfinite(loss)):
        raise AssertionError("Nonfinite loss")
    loss.backward()
    if not all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in model.parameters()):
        raise AssertionError("Nonfinite gradient")
    with torch.no_grad():
        inference = model(batch)
    if not bool(torch.isfinite(inference).all()):
        raise AssertionError("Nonfinite inference")
    return {"status": "PASS", "scale": scale, "nodes": graph["nodes"],
            "edges": len(graph["edges"]), "connected": graph["connected"],
            "node_features": 12, "edge_features": 8, "sequence_steps": steps,
            "batch_size": 1, "transformers": graph["edge_kinds"].count("transformer"),
            "asset_counts": {k: len(v) for k, v in graph["assets"].items()},
            "model_parameters": sum(p.numel() for p in model.parameters()),
            "loss": float(loss.detach().cpu()), "inference_finite": True,
            "gradient_finite": True, "topology_sha256": canonical_sha(graph),
            "seed": SEED, "torch_version": torch.__version__, "device": device,
            "evidence_role": "SYNTHETIC_COMPUTE_SMOKE_NOT_SCIENTIFIC_EVALUATION"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scale", choices=SCALES, default="S1")
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = smoke(args.scale, args.device, args.steps)
    payload = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_bytes(payload.encode())
    print(payload, end="")


if __name__ == "__main__":
    main()
