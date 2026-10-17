"""Regression tests for the acceptance decision, without an accelerator."""
import sys
import types
import unittest
from unittest.mock import patch

# Linux production imports resource; Windows development only tests pure logic.
if sys.platform == "win32":
    sys.modules.setdefault("resource", types.SimpleNamespace())
from accept_m1 import stability_resource_evidence


class StabilityEvidenceTests(unittest.TestCase):
    def samples(self):
        return [{"elapsed_s": float(t), "device": "cambricon", "rss_bytes": 200 * 1024**2,
                 "aten_reserved_bytes": 100 * 1024**2, "managed_device_bytes": 300 * 1024**2,
                 "process_device_memory_bytes": 400 * 1024**2,
                 "infinitensor": {"activation_bytes": 1234, "heap_bytes": 0}}
                for t in range(0, 7201, 5)]

    def test_constant_memory_passes(self):
        evidence = stability_resource_evidence(self.samples(), 7200)
        self.assertEqual(evidence["rss"]["last_60min_slope_bytes_s"], 0)
        self.assertEqual(evidence["rss"]["trend_window_samples"], 721)

    def test_rising_memory_fails(self):
        samples = self.samples()
        for sample in samples:
            sample["rss_bytes"] += sample["elapsed_s"] * 2 * 1024**2 / 60
        with self.assertRaisesRegex(AssertionError, "RSS did not stabilize"):
            stability_resource_evidence(samples, 7200)

    def test_missing_resource_fails(self):
        samples = self.samples()
        for sample in samples[-200:]:
            del sample["aten_reserved_bytes"]
        with self.assertRaisesRegex(AssertionError, "evidence missing"):
            stability_resource_evidence(samples, 7200)

    def test_missing_time_coverage_fails(self):
        samples = [s for s in self.samples() if s["elapsed_s"] > 900]
        with self.assertRaisesRegex(AssertionError, "time coverage"):
            stability_resource_evidence(samples, 7200)

    def test_same_shape_allocation_change_fails(self):
        samples = self.samples()
        samples[-1]["infinitensor"]["activation_bytes"] += 1
        with self.assertRaisesRegex(AssertionError, "allocation changed"):
            stability_resource_evidence(samples, 7200)

    def test_device_process_resource_required_on_nvidia(self):
        samples = self.samples()
        for sample in samples:
            sample["device"] = "nvidia"
            del sample["process_device_memory_bytes"]
        with self.assertRaisesRegex(AssertionError, "process device memory evidence missing"):
            stability_resource_evidence(samples, 7200)


if __name__ == "__main__":
    unittest.main()
