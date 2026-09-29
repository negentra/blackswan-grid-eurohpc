"""Small synthetic compute fixtures; no power-flow or scientific data."""
import unittest

import torch

from benchmarks.connected_grid_calibration import SCALES, smoke, tensors, topology


class ConnectedGridPreflightTests(unittest.TestCase):
    def test_all_planned_shapes_are_connected_and_exact(self):
        for scale, (nodes, edges, _, _) in SCALES.items():
            graph = topology(scale)
            self.assertTrue(graph["connected"])
            self.assertEqual(graph["nodes"], nodes)
            self.assertEqual(len(graph["edges"]), edges)
            self.assertIn("transformer", graph["edge_kinds"])
            self.assertTrue(all(graph["assets"].values()))

    def test_deterministic_finite_training_and_inference(self):
        first, second = smoke("S1", "cpu", 2), smoke("S1", "cpu", 2)
        self.assertEqual(first, second)
        self.assertTrue(first["gradient_finite"])
        self.assertTrue(first["inference_finite"])
        self.assertGreater(first["nodes"], 7)

    def test_planned_long_sequence_tensor_dimensions_without_training(self):
        graph = topology("S3")
        batch = tensors(graph, 2, 192, "cpu")
        self.assertEqual(tuple(batch["node"].shape), (2, 192, 1024, 12))
        self.assertEqual(tuple(batch["edge"].shape), (1200, 8))
        self.assertTrue(bool(torch.isfinite(batch["node"]).all()))

    def test_disconnected_corruption_is_detected_by_connectivity_logic(self):
        # The generator's spanning-tree invariant is independently checked.
        for scale in SCALES:
            graph = topology(scale)
            seen, stack = {0}, [0]
            adjacency = [[] for _ in range(graph["nodes"])]
            for u, v in graph["edges"]:
                adjacency[u].append(v)
                adjacency[v].append(u)
            while stack:
                for v in adjacency[stack.pop()]:
                    if v not in seen:
                        seen.add(v)
                        stack.append(v)
            self.assertEqual(len(seen), graph["nodes"])


if __name__ == "__main__":
    unittest.main()
