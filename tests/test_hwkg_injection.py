import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from hwkg.injection import hardware_guidance


GRAPH = Path(__file__).resolve().parents[1] / "hwkg" / "graph.json"
SHA256 = "0d422cfe81e2531980204e71b49fa9e5ffed9c2606a9a5569e7a43ebfc0078b4"


class HardwareGuidanceTest(unittest.TestCase):
    def test_disabled_is_noop(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertIsNone(hardware_guidance("draft"))

    def test_a10_guidance_and_audit(self):
        with tempfile.TemporaryDirectory() as directory:
            env = {
                "MLEVOLVE_HWKG_GRAPH_PATH": str(GRAPH),
                "MLEVOLVE_HWKG_GRAPH_SHA256": SHA256,
                "MLEVOLVE_HWKG_HARDWARE_ID": "nvidia.ampere.a10.spec",
                "MLEVOLVE_HWKG_AUDIT_DIR": directory,
            }
            with patch.dict(os.environ, env, clear=True):
                guidance = hardware_guidance("draft")
            self.assertIn("NVIDIA A10", guidance)
            self.assertIn("feat:amp", guidance)
            self.assertNotIn("feat:muon_optimizer", guidance)
            self.assertLessEqual(len(guidance), 3200)
            audit = json.loads((Path(directory) / "prompt-audit.jsonl").read_text().strip())
            self.assertEqual(audit["graph_sha256"], SHA256)
            self.assertEqual(audit["hardware_node"], "hw:nvidia.ampere.a10.spec")
            self.assertEqual(audit["chars"], len(guidance))

    def test_wrong_graph_hash_fails_closed(self):
        env = {
            "MLEVOLVE_HWKG_GRAPH_PATH": str(GRAPH),
            "MLEVOLVE_HWKG_GRAPH_SHA256": "0" * 64,
            "MLEVOLVE_HWKG_HARDWARE_ID": "nvidia.ampere.a10.spec",
        }
        with patch.dict(os.environ, env, clear=True):
            with self.assertRaisesRegex(ValueError, "SHA-256"):
                hardware_guidance("draft")


if __name__ == "__main__":
    unittest.main()
