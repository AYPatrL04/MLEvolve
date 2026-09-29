import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from hwkg.injection import hardware_guidance


GRAPH = Path(__file__).resolve().parents[1] / "hwkg" / "graph.json"
SHA256 = "0d422cfe81e2531980204e71b49fa9e5ffed9c2606a9a5569e7a43ebfc0078b4"
A10 = SimpleNamespace(name="NVIDIA A10", major=8, minor=6, total_memory=23 * 1024**3)


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
            with patch.dict(os.environ, env, clear=True), patch(
                "hwkg.injection._visible_cuda_device", return_value=A10
            ):
                guidance = hardware_guidance(
                    "draft", task_description="Disaster Tweets text classification"
                )
            self.assertIn("NVIDIA A10", guidance)
            self.assertIn("Runtime-visible GPU memory: 23552 MiB total", guidance)
            self.assertIn("feat:amp", guidance)
            self.assertEqual(guidance.count("Selective FP16 AMP is an optional"), 1)
            self.assertNotIn("feat:muon_optimizer [unverified]", guidance)
            self.assertIn("Other unverified or experimental GPU-linked entries", guidance)
            self.assertNotIn("from utils.muon_optimizer import", guidance)
            self.assertLessEqual(len(guidance), 3200)
            audit = json.loads((Path(directory) / "prompt-audit.jsonl").read_text().strip())
            self.assertEqual(audit["graph_sha256"], SHA256)
            self.assertEqual(audit["hardware_node"], "hw:nvidia.ampere.a10.spec")
            self.assertEqual(audit["visible_gpu_name"], "NVIDIA A10")
            self.assertEqual(audit["visible_compute_capability"], "8.6")
            self.assertIn("feat:muon_optimizer", audit["background_unverified_feature_ids"])
            self.assertEqual(
                audit["background_reasons"]["feat:muon_optimizer"],
                "no explicit hidden-matrix model in candidate code",
            )
            self.assertEqual(
                audit["background_reasons"]["feat:nvimagecodec_gpu_decode"],
                "no image-task evidence",
            )
            self.assertIn("feat:soap_optimizer", audit["background_unverified_feature_ids"])
            self.assertEqual(audit["chars"], len(guidance))

    def test_image_decoder_requires_image_task_evidence(self):
        env = {
            "MLEVOLVE_HWKG_GRAPH_PATH": str(GRAPH),
            "MLEVOLVE_HWKG_GRAPH_SHA256": SHA256,
            "MLEVOLVE_HWKG_HARDWARE_ID": "nvidia.ampere.a10.spec",
        }
        with patch.dict(os.environ, env, clear=True), patch(
            "hwkg.injection._visible_cuda_device", return_value=A10
        ):
            image_guidance = hardware_guidance("draft", task_description="Classify JPEG photos")
            text_guidance = hardware_guidance("draft", task_description="Classify tweet text")
        self.assertIn("feat:nvimagecodec_gpu_decode [unverified]", image_guidance)
        self.assertNotIn("feat:nvimagecodec_gpu_decode [unverified]", text_guidance)
        self.assertIn("feat:nvimagecodec_gpu_decode", text_guidance)

    def test_muon_requires_hidden_matrices_and_implementation(self):
        env = {
            "MLEVOLVE_HWKG_GRAPH_PATH": str(GRAPH),
            "MLEVOLVE_HWKG_GRAPH_SHA256": SHA256,
            "MLEVOLVE_HWKG_HARDWARE_ID": "nvidia.ampere.a10.spec",
        }
        mlp = "class Model(nn.Module):\n    def __init__(self):\n        self.hidden = nn.Linear(8, 16)\n        self.head = nn.Linear(16, 2)\n"
        single_layer = "model = nn.Linear(8, 2)"
        unrelated_layers = "def make_layers():\n    return nn.Linear(8, 16), nn.Linear(16, 2)\n"
        with patch.dict(os.environ, env, clear=True), patch(
            "hwkg.injection._visible_cuda_device", return_value=A10
        ):
            with patch("hwkg.injection._muon_available", return_value=True):
                eligible = hardware_guidance("improve", candidate_code=mlp)
                no_hidden = hardware_guidance("improve", candidate_code=single_layer)
                unrelated = hardware_guidance("improve", candidate_code=unrelated_layers)
            with patch("hwkg.injection._muon_available", return_value=False):
                no_implementation = hardware_guidance("improve", candidate_code=mlp)
        self.assertIn("feat:muon_optimizer [unverified]", eligible)
        self.assertNotIn("feat:muon_optimizer [unverified]", no_hidden)
        self.assertNotIn("feat:muon_optimizer [unverified]", unrelated)
        self.assertNotIn("feat:muon_optimizer [unverified]", no_implementation)

    def test_language_model_candidates_need_task_evidence(self):
        env = {
            "MLEVOLVE_HWKG_GRAPH_PATH": str(GRAPH),
            "MLEVOLVE_HWKG_GRAPH_SHA256": SHA256,
            "MLEVOLVE_HWKG_HARDWARE_ID": "nvidia.ampere.a10.spec",
        }
        with patch.dict(os.environ, env, clear=True), patch(
            "hwkg.injection._visible_cuda_device", return_value=A10
        ):
            guidance = hardware_guidance(
                "draft", task_description="Autoregressive language modeling over long document sequences"
            )
        self.assertIn("feat:cut_cross_entropy [unverified]", guidance)
        self.assertIn("feat:dataset_decomposition [unverified]", guidance)

    def test_wrong_visible_gpu_fails_closed(self):
        env = {
            "MLEVOLVE_HWKG_GRAPH_PATH": str(GRAPH),
            "MLEVOLVE_HWKG_GRAPH_SHA256": SHA256,
            "MLEVOLVE_HWKG_HARDWARE_ID": "nvidia.ampere.a10.spec",
        }
        wrong_gpu = SimpleNamespace(
            name="NVIDIA A100-SXM4-80GB", major=8, minor=0, total_memory=80 * 1024**3
        )
        with patch.dict(os.environ, env, clear=True), patch(
            "hwkg.injection._visible_cuda_device", return_value=wrong_gpu
        ):
            with self.assertRaisesRegex(RuntimeError, "does not uniquely match pinned graph node"):
                hardware_guidance("draft")

    def test_other_exact_gpu_models(self):
        devices = (
            ("nvidia.ampere.a100_pcie_40gb.spec", "NVIDIA A100-PCIE-40GB", 8, 0, 40),
            ("nvidia.blackwell.geforce_rtx_5090.spec", "NVIDIA GeForce RTX 5090", 12, 0, 32),
        )
        for hardware_id, name, major, minor, gib in devices:
            env = {
                "MLEVOLVE_HWKG_GRAPH_PATH": str(GRAPH),
                "MLEVOLVE_HWKG_GRAPH_SHA256": SHA256,
                "MLEVOLVE_HWKG_HARDWARE_ID": hardware_id,
            }
            gpu = SimpleNamespace(name=name, major=major, minor=minor,
                                  total_memory=gib * 1024**3)
            with self.subTest(hardware_id=hardware_id), patch.dict(
                os.environ, env, clear=True
            ), patch("hwkg.injection._visible_cuda_device", return_value=gpu):
                guidance = hardware_guidance("draft")
                self.assertIn(f"hw:{hardware_id}", guidance)
                self.assertLessEqual(len(guidance), 3200)

    def test_gpu_memory_mismatch_fails_closed(self):
        env = {
            "MLEVOLVE_HWKG_GRAPH_PATH": str(GRAPH),
            "MLEVOLVE_HWKG_GRAPH_SHA256": SHA256,
            "MLEVOLVE_HWKG_HARDWARE_ID": "nvidia.ampere.a10.spec",
        }
        small_gpu = SimpleNamespace(name="NVIDIA A10", major=8, minor=6, total_memory=5 * 1024**3)
        with patch.dict(os.environ, env, clear=True), patch(
            "hwkg.injection._visible_cuda_device", return_value=small_gpu
        ):
            with self.assertRaisesRegex(RuntimeError, "does not uniquely match pinned graph node"):
                hardware_guidance("draft")

    def test_ambiguous_graph_variant_fails_closed(self):
        env = {
            "MLEVOLVE_HWKG_GRAPH_PATH": str(GRAPH),
            "MLEVOLVE_HWKG_GRAPH_SHA256": SHA256,
            "MLEVOLVE_HWKG_HARDWARE_ID": "nvidia.ada_lovelace.geforce_rtx_4070_gddr6.spec",
        }
        gpu = SimpleNamespace(name="NVIDIA GeForce RTX 4070", major=8, minor=9,
                              total_memory=12 * 1024**3)
        with patch.dict(os.environ, env, clear=True), patch(
            "hwkg.injection._visible_cuda_device", return_value=gpu
        ):
            with self.assertRaisesRegex(RuntimeError, "does not uniquely match pinned graph node"):
                hardware_guidance("draft")

    def test_no_or_multiple_visible_gpus_fail_closed(self):
        from hwkg.injection import _visible_cuda_device

        for available, count in ((False, 0), (True, 2)):
            cuda = SimpleNamespace(is_available=lambda: available, device_count=lambda: count)
            with self.subTest(available=available, count=count), patch.dict(
                "sys.modules", {"torch": SimpleNamespace(cuda=cuda)}
            ):
                with self.assertRaisesRegex(RuntimeError, "exactly one visible CUDA GPU"):
                    _visible_cuda_device()

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
