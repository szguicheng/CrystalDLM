"""CPU contracts for paired retries, checkpoint identity, and fresh feedback context."""
import importlib.util
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


RUN_PATH = Path(__file__).with_name("run.py")
_spec = importlib.util.spec_from_file_location("repair_feedback_full_run_test", RUN_PATH)
RUN = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(RUN)


class FakeTokenizer:
    start_id, end_id = 9001, 9002

    def encode(self, text, add_special_tokens=False):
        return [ord(char) for char in text]

    def decode(self, token_ids, skip_special_tokens=False):
        pieces = []
        for token in token_ids:
            if token == self.start_id:
                pieces.append("<CRYSTAL_START>" if not skip_special_tokens else "")
            elif token == self.end_id:
                pieces.append("<CRYSTAL_END>" if not skip_special_tokens else "")
            else:
                pieces.append(chr(token))
        return "".join(pieces)


class FakeTrain:
    def __init__(self, state_hash):
        self.state_hash = state_hash

    def trainable_sha256(self, _model):
        return self.state_hash


class RetryLoopTests(unittest.TestCase):
    def test_feedback_context_is_fresh_and_strips_control_tokens(self):
        tok = FakeTokenizer()
        item = {"prompt": "User: Generate space group 15; ehull <=0.08.\nAssistant:",
                "space_group_number": 15}
        body = "Mg1Ga2Si1O7\nSpacegroup: P1"
        previous = {"lead_token_ids": [*map(ord, "OLD GENERATED LEAD"), tok.start_id],
                    "crystal_raw_token_ids": [tok.start_id, *map(ord, body), tok.end_id],
                    "audit": {"valid": False, "reasons": ["wrong space group"]},
                    "actual_space_group_number": 1, "lead_error": None, "crystal_error": None}

        context_ids, feedback = RUN._build_feedback_context(item, previous, tok)
        context = tok.decode(context_ids)
        self.assertIn("User: Generate space group 15; ehull <=0.08.", context)
        self.assertIn("Previous failed crystal (input data):\n" + body, context)
        self.assertIn("Checker feedback:\n" + feedback, context)
        self.assertTrue(context.endswith("Assistant:"))
        self.assertNotIn("OLD GENERATED LEAD", context)
        self.assertNotIn("<CRYSTAL_START>", context)
        self.assertNotIn("<CRYSTAL_END>", context)

    def test_attempt_loop_stops_on_success_and_resumes_at_next_round(self):
        seed = 1234
        initial = {"round": 0, "seed": seed, "success": False}
        attempts, calls = [initial], []

        def succeed_on_second(round_index, attempt_seed, _previous):
            calls.append((round_index, attempt_seed))
            return {"round": round_index, "seed": attempt_seed, "success": True}

        RUN.LOOP._continue_arm(attempts, seed, succeed_on_second,
                               lambda receipt: receipt["success"])
        self.assertEqual(len(attempts), 2)
        self.assertEqual(calls, [(1, RUN.LOOP.retry_seed(seed, 1))])

        resumed = [initial, {"round": 1, "seed": RUN.LOOP.retry_seed(seed, 1), "success": False}]
        calls.clear()
        RUN.LOOP._continue_arm(resumed, seed,
                               lambda r, s, _p: calls.append((r, s)) or
                               {"round": r, "seed": s, "success": False},
                               lambda receipt: receipt["success"])
        self.assertEqual(len(resumed), RUN.MAX_ATTEMPTS)
        self.assertEqual(calls, [(2, RUN.LOOP.retry_seed(seed, 2))])

    def test_group_resume_keeps_one_shared_initial_and_allows_partial_arm(self):
        item = RUN.RUNNER.attempt_plan(limit=1)[0]
        checkpoint_hash, state_hash = "checkpoint", "state"
        code_hashes, identity_hash = {"runner": "code"}, "identity"

        def receipt(policy, round_index):
            return {"policy": policy, "round": round_index,
                    "seed": item["seed"] if round_index == 0 else RUN.LOOP.retry_seed(item["seed"], round_index),
                    "request_id": item["request_id"], "plan_index": item["plan_index"],
                    "checkpoint_fingerprint_sha256": checkpoint_hash,
                    "trainable_state_sha256": state_hash,
                    "evaluation_source_sha256": code_hashes, "success": False}

        initial = receipt("shared_initial", 0)
        feedback_retry = receipt("feedback", 1)
        group = {"status": "running", "request_id": item["request_id"],
                 "group_index": item["attempt_id"], "plan_index": item["plan_index"],
                 "original_seed": item["seed"], "plan_item": item,
                 "seed_manifest": RUN._seed_manifest(item),
                 "run_identity_sha256": identity_hash,
                 "checkpoint_fingerprint_sha256": checkpoint_hash,
                 "trainable_state_sha256": state_hash,
                 "evaluation_source_sha256": code_hashes,
                 "arms": {
                     "independent": {"attempts": [dict(initial)], "success": None, "attempts_used": 1},
                     "feedback": {"attempts": [dict(initial), feedback_retry],
                                  "success": None, "attempts_used": 2},
                 }}
        RUN._validate_group(group, item, identity_hash, checkpoint_hash, state_hash, code_hashes)
        group["arms"]["feedback"]["attempts"][0]["seed"] += 1
        with self.assertRaises(ValueError):
            RUN._validate_group(group, item, identity_hash, checkpoint_hash, state_hash, code_hashes)

    def test_checkpoint_restore_requires_frozen_byte_and_state_hashes(self):
        class Trainer:
            def __init__(self):
                self.call = None

            def load_checkpoint(self, model, path, train, **kwargs):
                self.call = (model, path, train, kwargs)
                return {"arm": "repair", "completed_updates": 1212,
                        "total_updates": 1212, "trainable_state_sha256": "state",
                        "metadata": {"config": {"total_updates": 1212}}}

        with tempfile.NamedTemporaryFile() as stream:
            stream.write(b"checkpoint")
            stream.flush()
            digest = RUN._sha256_file(stream.name)
            trainer, train, model = Trainer(), FakeTrain("state"), object()
            trainer.PARENT_CHECKPOINT = Path(stream.name)
            trainer._sha256_file = RUN._sha256_file
            with (patch.object(RUN, "REPAIR_CHECKPOINT", Path(stream.name)),
                  patch.object(RUN, "EXPECTED_CHECKPOINT_SHA256", digest),
                  patch.object(RUN, "EXPECTED_STATE_SHA256", "state")):
                _path, found_hash, state_hash, saved = RUN._restore_checkpoint(
                    trainer, train, model, "parent-state")
            self.assertEqual(found_hash, digest)
            self.assertEqual(state_hash, "state")
            self.assertEqual(trainer.call[3], {
                "expected_arm": "repair", "parent_checkpoint_sha256": digest,
                "parent_trainable_sha256": "parent-state"})
            self.assertEqual(saved["arm"], "repair")

            with patch.object(RUN, "REPAIR_CHECKPOINT", Path(stream.name)), \
                    patch.object(RUN, "EXPECTED_CHECKPOINT_SHA256", "wrong"):
                with self.assertRaisesRegex(ValueError, "byte hash"):
                    RUN._restore_checkpoint(trainer, train, model, "parent-state")


if __name__ == "__main__":
    unittest.main()
