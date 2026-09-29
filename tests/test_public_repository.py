"""Sanitized publication candidate checks; synthetic fixtures only."""
import json
import unittest

from benchmarks.connected_grid_calibration import SEED, SCALES, smoke, topology
from tools.check_public import ROOT, FILES, present_files, scan, verify


class PublicRepositoryTests(unittest.TestCase):
    def test_config_matches_implementation(self):
        config = json.loads((ROOT / "configs/connected_grid_reference.json").read_text(encoding="utf-8"))
        self.assertEqual(config["seed"], SEED)
        for scale, (nodes, edges, width, layers) in SCALES.items():
            expected = config["scales"][scale]
            self.assertEqual((nodes, edges, width, layers),
                             (expected["nodes"], expected["edges"], expected["width"], expected["layers"]))
            self.assertEqual(smoke(scale, "cpu", 2)["model_parameters"], expected["parameters"])
            self.assertTrue(topology(scale)["connected"])

    def test_raw_cuda_result_is_small_smoke_not_target_measurement(self):
        result = json.loads((ROOT / "results/local_connected_grid_cuda_smoke.json").read_text(encoding="utf-8"))
        self.assertEqual(result["gpu_name"], "NVIDIA GeForce RTX 3080")
        self.assertFalse(result["full_calibration_matrix_executed"])
        self.assertEqual(result["physical_solver_calls"], 0)
        self.assertEqual(result["scientific_labels"], 0)
        self.assertEqual({r["scale"] for r in result["rows"]}, set(SCALES))

    def test_allowlist_and_no_private_file(self):
        self.assertEqual(present_files() - {"MANIFEST.sha256"}, FILES)
        self.assertNotIn("data/sanitized/promotion_accounting.json", FILES)
        scan()

    def test_manifest(self):
        verify()


if __name__ == "__main__":
    unittest.main()
