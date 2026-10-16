"""Offline contract checks for the single clean-CFAR B32 joint pilot."""
import ast
from collections import Counter
import hashlib
import json
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[2]
RUNNER = Path(__file__).with_name("run.py")
PREPARE = Path(__file__).with_name("prepare.py")
SOURCE = RUNNER.read_text()


def _literal(path, name):
    tree = ast.parse(path.read_text())
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == name for target in node.targets
        ):
            return ast.literal_eval(node.value)
    raise AssertionError(f"missing literal assignment {name}")


def _assignment_expression(path, name):
    tree = ast.parse(path.read_text())
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == name for target in node.targets
        ):
            return ast.unparse(node.value)
    raise AssertionError(f"missing assignment {name}")


def _loss_flag_calls():
    calls = []
    for node in ast.walk(ast.parse(SOURCE)):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if node.func.attr in ("pretrained", "restore_checkpoint"):
                calls.append((node.func.attr, node))
    return calls


def _exposure_functions():
    tree = ast.parse(SOURCE)
    wanted = {"sha", "update_examples", "exposure_receipt"}
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in wanted]

    def tasks(row, split, direction, variant):
        return tuple({"task": task} for task in ("G", "Q1", "E", "Q2"))

    namespace = {
        "Counter": Counter,
        "hashlib": hashlib,
        "json": json,
        "UPDATES": 512,
        "READER_SCHEDULES": ("alternating_q1_q2",),
        "DEFAULT_READER_SCHEDULE": "alternating_q1_q2",
        "tasks": tasks,
    }
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(RUNNER), "exec"), namespace)
    return namespace


def _read_data(data_dir):
    tree = ast.parse(SOURCE)
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "read_data")

    def load_records(path):
        return [json.loads(line) for line in Path(path).read_text().splitlines()]

    namespace = {
        "PROJECT": ROOT,
        "K": 2560,
        "MAXLEN": 8192,
        "STEPS": 32,
        "json": json,
        "load_records": load_records,
    }
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(RUNNER), "exec"), namespace)
    return namespace["read_data"](data_dir)


def _generation_gate():
    tree = ast.parse(SOURCE)
    function = next(
        (node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "seen8_generation_gate_passed"),
        None,
    )
    if function is None:
        raise AssertionError("missing seen8_generation_gate_passed")
    namespace = {}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(RUNNER), "exec"), namespace)
    return namespace["seen8_generation_gate_passed"]


def _task_length():
    tree = ast.parse(SOURCE)
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "task_length")
    namespace = {"ARMS": ("joint_ar_diffusion_b32",), "K": 2560, "MAXLEN": 8192}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(RUNNER), "exec"), namespace)
    return namespace["task_length"]


class JointRunnerContractTest(unittest.TestCase):
    def test_single_arm_uses_requested_model_and_clean_cfar_weight(self):
        self.assertEqual(_literal(RUNNER, "ARMS"), ("joint_ar_diffusion_b32",))
        self.assertEqual(_literal(RUNNER, "CORE_MODES"), {"joint_ar_diffusion_b32": "hybrid"})
        self.assertEqual(_literal(RUNNER, "BLOCK_SIZES"), {"joint_ar_diffusion_b32": 32})
        self.assertEqual(_literal(RUNNER, "MODE_OFFSETS"), {"joint_ar_diffusion_b32": 0})
        self.assertEqual(_literal(RUNNER, "CLEAN_CRYSTAL_AR_WEIGHT"), 1.0)
        self.assertEqual(_literal(RUNNER, "EXPECTED_INITIAL_FINGERPRINT"), "e97aed646b7289c06f10be376b37d67cd7aa6a42ad6214b61da008cd85b8404b")
        self.assertEqual(_literal(PREPARE, "MODEL"), "Qwen/Qwen2.5-1.5B-Instruct")
        self.assertEqual(_literal(PREPARE, "REVISION"), "989aa7980e4cf806f80c7fef2b1adb7bc71aa306")

        calls = _loss_flag_calls()
        self.assertEqual(len([name for name, _ in calls if name == "pretrained"]), 2)
        self.assertEqual(len([name for name, _ in calls if name == "restore_checkpoint"]), 1)
        for name, call in calls:
            weight_name = "clean_crystal_ar_weight" if name == "pretrained" else "expected_clean_crystal_ar_weight"
            weight = next((item.value for item in call.keywords if item.arg == weight_name), None)
            self.assertIsNotNone(weight, f"{name} must pin the clean-crystal AR weight")
            if isinstance(weight, ast.Name):
                self.assertEqual(weight.id, "CLEAN_CRYSTAL_AR_WEIGHT")
                self.assertEqual(_literal(RUNNER, weight.id), 1.0)
            else:
                self.assertEqual(ast.literal_eval(weight), 1.0)
        self.assertIn('stats.get("crystal_noise_steps")', SOURCE)
        self.assertTrue("actual_noise_step_exposure" in SOURCE)
        self.assertTrue('stats.get("crystal_ar")' in SOURCE)
        self.assertTrue('"clean_crystal_ar_loss"' in SOURCE)

    def test_frozen_data_remains_8_seen_16_heldout_and_0_5b_provenance(self):
        data = ROOT / "experiments/interleaved_relational_20261004/data_pilot_001"
        manifest, rows, train, heldout = _read_data(data)
        self.assertEqual(len(rows), 24)
        self.assertEqual((len(train), len(heldout)), (8, 16))
        self.assertIn("experiments/interleaved_relational_20261004/data_pilot_001", _assignment_expression(PREPARE, "DATA_DIR"))
        self.assertEqual(manifest["provenance"]["model"], "Qwen/Qwen2.5-0.5B-Instruct")
        self.assertEqual(manifest["provenance"]["revision"], "7ae557604adf67be50417f59c2c2f167def9a775")
        self.assertEqual(_literal(RUNNER, "SEED"), 17)
        self.assertEqual(_literal(RUNNER, "UPDATES"), 512)

    def test_exposure_receipt_counts_and_event_hash_are_stable(self):
        helpers = _exposure_functions()
        rows = [{"id": f"case-{index}"} for index in range(8)]
        receipt = helpers["exposure_receipt"](rows)
        self.assertEqual(receipt["total"], {"E": 512, "G": 1024, "Q1": 256, "Q2": 256})
        for material in receipt["per_material"].values():
            self.assertEqual(material["tasks"], {"E": 64, "G": 128, "Q1": 32, "Q2": 32})
            self.assertEqual(material["E_factors"], {"minus": 32, "plus": 32})
            self.assertEqual(material["Q2_factors"], {"minus": 16, "plus": 16})
        self.assertEqual(receipt["events_sha256"], "33b4ad8a7e4cf29c4cda59dbfb26c2ee81d909b69d8ce0132ffdc2c80d7283a2")

    def test_heldout_gate_opens_only_after_a_valid_seen_generation(self):
        gate = _generation_gate()
        self.assertFalse(gate({"counts": {"G": 0}}))
        self.assertTrue(gate({"counts": {"G": 1}}))

    def test_joint_pack_counts_two_full_k_crystal_streams(self):
        class Tokenizer:
            @staticmethod
            def encode(text, add_special_tokens=False):
                return list(text)

        body, packed = _task_length()(
            {"prefix": "abc", "lead": "xy", "crystal": "data"},
            Tokenizer(), "joint_ar_diffusion_b32",
        )
        self.assertEqual(body, 5)
        self.assertEqual(packed, 3 + 2 + 2 * 2560)


if __name__ == "__main__":
    unittest.main()
