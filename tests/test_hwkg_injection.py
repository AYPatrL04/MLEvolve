import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from hwkg.injection import hardware_guidance


GRAPH = Path(__file__).resolve().parents[1] / "hwkg" / "graph.json"
SHA256 = "3842eff0a7742c42b3777a01a6f58cfbec42b37c442d9bb9f5b080321ff7ed62"
A10 = SimpleNamespace(name="NVIDIA A10", major=8, minor=6, total_memory=23 * 1024**3)


class HardwareGuidanceTest(unittest.TestCase):
    def evidence(self, code="", task="Disaster Tweets text classification", stage="draft"):
        env = {
            "MLEVOLVE_HWKG_GRAPH_PATH": str(GRAPH), "MLEVOLVE_HWKG_GRAPH_SHA256": SHA256,
            "MLEVOLVE_HWKG_HARDWARE_ID": "nvidia.ampere.a10.spec",
            "MLEVOLVE_HWKG_EVIDENCE_POLICY": "v2",
        }
        with tempfile.TemporaryDirectory() as directory:
            env["MLEVOLVE_HWKG_AUDIT_DIR"] = directory
            with patch.dict(os.environ, env, clear=True), patch(
                "hwkg.injection._visible_cuda_device", return_value=A10
            ), patch("hwkg.injection._missing_tokenizer_packages", return_value=["tiktoken", "sentencepiece"]):
                text = hardware_guidance(stage, task, code)
            audit = json.loads((Path(directory) / "prompt-audit.jsonl").read_text())
        self.assertLessEqual(len(text), 1600)
        self.assertLessEqual(len(audit["advice_ids"]), 3)
        self.assertEqual(text, audit["context_text"])
        return text, audit

    def test_evidence_draft_has_actionable_conditional_advice(self):
        text, audit = self.evidence()
        self.assertEqual(audit["advice_ids"], ["padding", "validation", "precision"])
        self.assertIn("never test/private labels", text)
        self.assertIn("is_bf16_supported", text)
        self.assertIn("not proof of generalization", text)
        self.assertIn("non-padding tokens/s", text)
        self.assertEqual(set(audit["evidence_sources"]), set(audit["advice_ids"]))

    def test_evidence_parent_targets_observed_bottlenecks(self):
        code = '''model = AutoModel.from_pretrained("roberta-base")
batch = tokenizer(text, padding="max_length")
loader = DataLoader(dataset, pin_memory=True)
best_thr = 0.5
'''
        _, audit = self.evidence(code, stage="improve")
        self.assertEqual(audit["advice_ids"], ["padding", "transfer", "precision"])

    def test_evidence_does_not_inject_gpu_changes_for_linear_classifier(self):
        text, audit = self.evidence("model = LogisticRegression()", stage="improve")
        self.assertEqual(audit["advice_ids"], ["validation"])
        self.assertNotIn("[precision]", text)

    def test_evidence_ignores_comments_and_irrelevant_tasks(self):
        _, audit = self.evidence("# AutoModel DataLoader padding='max_length'", task="tabular regression", stage="improve")
        self.assertEqual(audit["advice_ids"], [])

    def test_evidence_deberta_safety_survives_budget(self):
        text, audit = self.evidence('''model = AutoModel.from_pretrained("microsoft/deberta-v3-base")
with torch.autocast("cuda", dtype=torch.float16):
    loss = model(batch)
loader = DataLoader(dataset)
padding = "max_length"
''', stage="improve")
        self.assertTrue(audit["deberta_guard"])
        self.assertIn("finite loss/gradients", text)
        self.assertIn("Do not blanket-enable FP16", text)

    def test_evidence_does_not_repeat_existing_padding_transfer(self):
        _, audit = self.evidence('''model = AutoModel.from_pretrained("roberta-base")
collator = DataCollatorWithPadding(tokenizer)
loader = DataLoader(dataset)
batch = batch.to(device, non_blocking=True)
''', stage="improve")
        self.assertNotIn("padding", audit["advice_ids"])
        self.assertNotIn("transfer", audit["advice_ids"])

    def test_disabled_is_noop(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertIsNone(hardware_guidance("draft"))

    def test_a10_guidance_and_audit(self):
        with tempfile.TemporaryDirectory() as directory:
            env = {
                "MLEVOLVE_HWKG_EVIDENCE_POLICY": "legacy",
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
            self.assertNotIn("feat:amp", guidance)
            self.assertIn("Choose model and data strategy for the task first", guidance)
            self.assertNotIn("feat:muon_optimizer [unverified]", guidance)
            self.assertNotIn("Other unverified or experimental GPU-linked entries", guidance)
            self.assertNotIn("from utils.muon_optimizer import", guidance)
            self.assertLessEqual(len(guidance), 1600)
            audit = json.loads((Path(directory) / "prompt-audit.jsonl").read_text().strip())
            self.assertEqual(audit["graph_sha256"], SHA256)
            self.assertEqual(audit["hardware_node"], "hw:nvidia.ampere.a10.spec")
            self.assertEqual(audit["visible_gpu_name"], "NVIDIA A10")
            self.assertEqual(audit["visible_compute_capability"], "8.6")
            self.assertEqual(audit["selected_feature_ids"], [])
            self.assertEqual(audit["background_unverified_feature_ids"], [])
            self.assertFalse(audit["precision_guard"])
            self.assertEqual(audit["chars"], len(guidance))

    def test_precision_guard_is_candidate_specific(self):
        env = {
            "MLEVOLVE_HWKG_EVIDENCE_POLICY": "legacy",
            "MLEVOLVE_HWKG_GRAPH_PATH": str(GRAPH),
            "MLEVOLVE_HWKG_GRAPH_SHA256": SHA256,
            "MLEVOLVE_HWKG_HARDWARE_ID": "nvidia.ampere.a10.spec",
        }
        code = '''from transformers import AutoModel
model = AutoModel.from_pretrained("microsoft/deberta-v3-base")
with torch.autocast("cuda", dtype=torch.float16):
    output = model(batch)
'''
        with patch.dict(os.environ, env, clear=True), patch(
            "hwkg.injection._visible_cuda_device", return_value=A10
        ), patch("hwkg.injection._missing_tokenizer_packages", return_value=["tiktoken"]):
            guidance = hardware_guidance("improve", candidate_code=code)
        self.assertIn("Missing optional tokenizer packages in this runtime: tiktoken", guidance)
        self.assertIn("feat:amp", guidance)
        self.assertIn("DeBERTa attention has previously overflowed", guidance)
        self.assertIn("validate BF16 on one batch or use FP32", guidance)
        self.assertLessEqual(len(guidance), 1600)

    def test_no_precision_recommendation_without_candidate_precision(self):
        env = {
            "MLEVOLVE_HWKG_EVIDENCE_POLICY": "legacy",
            "MLEVOLVE_HWKG_GRAPH_PATH": str(GRAPH),
            "MLEVOLVE_HWKG_GRAPH_SHA256": SHA256,
            "MLEVOLVE_HWKG_HARDWARE_ID": "nvidia.ampere.a10.spec",
        }
        with patch.dict(os.environ, env, clear=True), patch(
            "hwkg.injection._visible_cuda_device", return_value=A10
        ):
            guidance = hardware_guidance("improve", candidate_code="model = nn.Linear(8, 2)")
        self.assertNotIn("feat:amp", guidance)

    def test_image_decoder_requires_image_task_evidence(self):
        env = {
            "MLEVOLVE_HWKG_EVIDENCE_POLICY": "legacy",
            "MLEVOLVE_HWKG_GRAPH_PATH": str(GRAPH),
            "MLEVOLVE_HWKG_GRAPH_SHA256": SHA256,
            "MLEVOLVE_HWKG_HARDWARE_ID": "nvidia.ampere.a10.spec",
            "MLEVOLVE_HWKG_EXPLORATORY": "1",
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
            "MLEVOLVE_HWKG_EVIDENCE_POLICY": "legacy",
            "MLEVOLVE_HWKG_GRAPH_PATH": str(GRAPH),
            "MLEVOLVE_HWKG_GRAPH_SHA256": SHA256,
            "MLEVOLVE_HWKG_HARDWARE_ID": "nvidia.ampere.a10.spec",
            "MLEVOLVE_HWKG_EXPLORATORY": "1",
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
            "MLEVOLVE_HWKG_EVIDENCE_POLICY": "legacy",
            "MLEVOLVE_HWKG_GRAPH_PATH": str(GRAPH),
            "MLEVOLVE_HWKG_GRAPH_SHA256": SHA256,
            "MLEVOLVE_HWKG_HARDWARE_ID": "nvidia.ampere.a10.spec",
            "MLEVOLVE_HWKG_EXPLORATORY": "1",
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
            "MLEVOLVE_HWKG_EVIDENCE_POLICY": "legacy",
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
                "MLEVOLVE_HWKG_EVIDENCE_POLICY": "legacy",
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
                self.assertLessEqual(len(guidance), 1600)

    def test_gpu_memory_mismatch_fails_closed(self):
        env = {
            "MLEVOLVE_HWKG_EVIDENCE_POLICY": "legacy",
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
            "MLEVOLVE_HWKG_EVIDENCE_POLICY": "legacy",
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
            "MLEVOLVE_HWKG_EVIDENCE_POLICY": "legacy",
            "MLEVOLVE_HWKG_GRAPH_PATH": str(GRAPH),
            "MLEVOLVE_HWKG_GRAPH_SHA256": "0" * 64,
            "MLEVOLVE_HWKG_HARDWARE_ID": "nvidia.ampere.a10.spec",
        }
        with patch.dict(os.environ, env, clear=True):
            with self.assertRaisesRegex(ValueError, "SHA-256"):
                hardware_guidance("draft")


if __name__ == "__main__":
    unittest.main()
