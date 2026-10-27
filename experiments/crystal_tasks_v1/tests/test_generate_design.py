import json
import tempfile
import unittest
from types import SimpleNamespace

from experiments.crystal_tasks_v1.evaluation.design import evaluate_design
from experiments.crystal_tasks_v1.generate_design import (
    decode_generated_tokens,
    design_instruction,
    derive_attempt_seed,
    encode_prompt,
    format_user_prompt,
    grade_overfit,
    persist_attempt_or_fail,
    shard_rows,
)


CRYSTAL = """Na
Spacegroup: Pm-3m
abc: 4 4 4
angles: 90 90 90
Sites (1)
Na 0 0 0 1a"""


def overfit_case():
    gold = f"A cubic sodium crystal.\n<CRYSTAL_START>{CRYSTAL}<CRYSTAL_END>"
    predictions = []
    attempts = []
    gold_attempts = []
    for index in range(32):
        material_id = f"mp-test-{index:02d}"
        request = {"space_group": 221, "ehull_upper": 0.08 + index / 1000}
        predictions.append({
            "material_id": material_id,
            "response": f" \n{gold}\n ",
            "reference_output": gold,
            "request": request,
        })
        attempts.append({
            "attempt_id": material_id, "seed": 42, "request": request,
            "output": predictions[-1]["response"],
        })
        gold_attempts.append({
            "attempt_id": material_id, "seed": 42, "request": request,
            "output": gold,
        })
    return predictions, attempts, gold_attempts


class GenerateDesignTests(unittest.TestCase):
    def test_attempt_seed_is_stable_and_attempt_specific(self):
        seed = derive_attempt_seed(42, "c003", 7)
        self.assertEqual(seed, derive_attempt_seed(42, "c003", 7))
        self.assertNotEqual(seed, derive_attempt_seed(42, "c003", 8))
        self.assertGreaterEqual(seed, 0)
        self.assertLess(seed, 2**64)

    def test_shards_cover_each_plan_id_once_in_input_order(self):
        rows = [{"attempt_id": f"a{index}"} for index in range(11)]
        shards = [shard_rows(rows, rank, 3) for rank in range(3)]
        combined = [row["attempt_id"] for shard in shards for row in shard]
        self.assertEqual(set(combined), {f"a{index}" for index in range(11)})
        self.assertEqual(len(combined), len(set(combined)))
        self.assertEqual([row["attempt_id"] for row in shards[0]], ["a0", "a3", "a6", "a9"])

    def test_design_prompt_matches_the_frozen_request_sentence(self):
        instruction = design_instruction({"space_group": 221, "ehull_upper": 0.08})
        self.assertEqual(
            instruction,
            "Generate a crystal with space group number 221 and energy above hull no greater than 0.08 eV/atom.",
        )
        self.assertEqual(format_user_prompt(instruction, ""), instruction)
        self.assertEqual(format_user_prompt(instruction, "extra facts"), f"{instruction}\nextra facts")

    def test_prompt_encodes_only_the_lf_user_source(self):
        class Template:
            def encode_oneturn(self, tokenizer, messages):
                self.messages = messages
                return [101, 102], [999]

        template = Template()
        processor = SimpleNamespace(
            template=template, tokenizer=object(),
            data_args=SimpleNamespace(cutoff_len=16),
        )
        prompt = encode_prompt("request", "input facts", processor)
        self.assertEqual(prompt, [101, 102])
        self.assertEqual(template.messages, [
            {"role": "user", "content": "request\ninput facts"},
            {"role": "assistant", "content": ""},
        ])

    def test_decode_keeps_crystal_tokens_and_trims_only_chat_eos_and_pad_ids(self):
        class Tokenizer:
            pad_token_id = 0

            def decode(self, token_ids, *, skip_special_tokens, clean_up_tokenization_spaces):
                self.skip_special_tokens = skip_special_tokens
                self.clean_up_tokenization_spaces = clean_up_tokenization_spaces
                return "".join({101: "x", 151669: "<CRYSTAL_START>", 102: "y", 151670: "<CRYSTAL_END>"}[i]
                               for i in token_ids)

        tokenizer = Tokenizer()
        text, stop = decode_generated_tokens(
            tokenizer, [101, 151669, 102, 151670, 151645, 0, 0]
        )
        self.assertEqual(text, "x<CRYSTAL_START>y<CRYSTAL_END>")
        self.assertFalse(tokenizer.skip_special_tokens)
        self.assertFalse(tokenizer.clean_up_tokenization_spaces)
        self.assertEqual(stop["terminal_chat_eos"], True)
        self.assertEqual(stop["trailing_chat_padding_tokens"], 2)

    def test_generation_error_is_fsynced_then_raises_worker_failure(self):
        attempt = {
            "attempt_id": "a0", "output": "", "generation_error": "RuntimeError: device failure",
        }
        attempts = []
        with tempfile.TemporaryFile(mode="w+", encoding="utf-8") as stream:
            with self.assertRaisesRegex(RuntimeError, "a0"):
                persist_attempt_or_fail(stream, attempts, attempt)
            stream.seek(0)
            self.assertEqual(json.loads(stream.readline()), attempt)
        self.assertEqual(attempts, [attempt])

    def test_overfit_grade_uses_full_output_validity_requested_group_and_nll(self):
        predictions, attempts, gold_attempts = overfit_case()
        generated = evaluate_design(attempts, training_references=[], test_references=[])["receipts"]
        gold = evaluate_design(gold_attempts, training_references=[], test_references=[])["receipts"]

        result = grade_overfit(predictions, generated, gold, {"nll_mean": 0.05})
        self.assertEqual(result["exact_full_output"], {"k": 32, "n": 32})
        self.assertEqual(result["exact_crystal_body"], {"k": 32, "n": 32})
        self.assertEqual(result["strict_valid"], {"k": 32, "n": 32})
        self.assertEqual(result["strict_valid_requested_sg"], {"k": 32, "n": 32})
        self.assertTrue(result["passed"])

        predictions[0]["response"] = f"Different description.\n<CRYSTAL_START>{CRYSTAL}<CRYSTAL_END>"
        result = grade_overfit(predictions, generated, gold, {"nll_mean": 0.05})
        self.assertEqual(result["exact_full_output"], {"k": 31, "n": 32})
        self.assertEqual(result["exact_crystal_body"], {"k": 32, "n": 32})
        self.assertFalse(result["passed"])

    def test_overfit_grade_reports_invalid_exact_gold(self):
        predictions, attempts, gold_attempts = overfit_case()
        bad_gold = "<CRYSTAL_START>not a PLaID structure<CRYSTAL_END>"
        predictions[0]["reference_output"] = bad_gold
        gold_attempts[0]["output"] = bad_gold
        generated = evaluate_design(attempts, training_references=[], test_references=[])["receipts"]
        gold = evaluate_design(gold_attempts, training_references=[], test_references=[])["receipts"]

        result = grade_overfit(predictions, generated, gold, {"nll_mean": 0.01})
        self.assertEqual(result["gold_strict_valid"], {"k": 31, "n": 32})
        self.assertEqual(result["gold_invalid_material_ids"], ["mp-test-00"])
        self.assertFalse(result["passed"])


if __name__ == "__main__":
    unittest.main()
