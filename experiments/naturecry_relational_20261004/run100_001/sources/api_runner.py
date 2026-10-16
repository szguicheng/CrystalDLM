"""Explicit-opt-in, first-pass-only SciencePal runner (never called by preflight)."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
from urllib.error import HTTPError, URLError
from urllib.request import HTTPRedirectHandler, Request, build_opener

from pymatgen.io.cif import CifWriter

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
sys.path.insert(0, str(PROJECT / "experiments/naturecry_llm_100_20261002"))
sys.path.insert(0, str(HERE))

import cohort
import codec
import geometry
import prepare

ENDPOINT = "https://llm.sciencepal.ai/v1/chat/completions"
MODEL = "tokenhub/kimi-k3"
MAX_REQUESTS = 200
RUNTIME_HASH_KEYS = ("geometry_py", "codec_py", "cohort_py", "public_wyckoff_module", "prepare_py",
                     "preflight_py", "api_runner_py", "inventory_schema_json", "decoder_schema_json",
                     "rules_json", "protocol_md", "public_wyckoff_fixture")


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, request, fp, code, message, headers, new_url):
        return None


class RequestFailure(RuntimeError):
    def __init__(self, message, stage, failure_kind):
        super().__init__(message)
        self.stage = stage
        self.failure_kind = failure_kind


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def redact(value, key):
    if not key:
        return value
    if isinstance(value, str):
        return value.replace(key, "[REDACTED]")
    if isinstance(value, list):
        return [redact(item, key) for item in value]
    if isinstance(value, dict):
        return {redact(name, key): redact(item, key) for name, item in value.items()}
    return value


def write_json(path, value, key=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(redact(value, key), indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def encoder_messages(facts, inventory):
    # Material-specific facts are sent only to the encoder.
    user_data = {"trusted_crystal_facts": facts,
                 "typed_inventory": [{"name": name, **spec} for name, spec in inventory.items()]}
    return [{"role": "system", "content": codec.ENCODER_PROMPT},
            {"role": "user", "content": json.dumps(user_data, ensure_ascii=False, indent=2, allow_nan=False)}]


def decoder_messages(inline_document):
    # Deliberately accepts only the final inline text: no facts, mappings, or CIF path.
    return [{"role": "system", "content": codec.DECODER_PROMPT},
            {"role": "user", "content": inline_document}]


class FirstPassRunner:
    def __init__(self, key, output, max_requests, opener=None):
        self.key = key
        self.output = output
        self.max_requests = min(max_requests, MAX_REQUESTS)
        self.request_count = 0
        self.usage = []
        self.opener = opener or build_opener(_NoRedirect)

    def request(self, messages, temperature, case_id, stage):
        if self.request_count >= self.max_requests or self.request_count >= MAX_REQUESTS:
            raise RequestFailure("First-pass request budget exhausted", "budget_exhausted", "budget_exhausted")
        payload = {"model": MODEL, "messages": messages, "temperature": temperature, "max_tokens": 8192}
        request_path = self.output / "requests" / f"{case_id}.{stage}.request.json"
        write_json(request_path, payload, self.key)
        request = Request(ENDPOINT, data=json.dumps(payload, ensure_ascii=False).encode(),
                          headers={"Authorization": f"Bearer {self.key}", "Content-Type": "application/json"},
                          method="POST")
        self.request_count += 1
        try:
            with self.opener.open(request, timeout=180) as response:
                raw = response.read()
        except HTTPError as error:
            write_json(self.output / "requests" / f"{case_id}.{stage}.http_error.json",
                       {"status": error.code})
            error.close()
            raise RequestFailure(f"HTTP {error.code}; no retry", f"{stage}_transport", "http_error") from None
        except (URLError, TimeoutError, OSError) as error:
            raise RequestFailure(f"Transport {type(error).__name__}; no retry", f"{stage}_transport", "transport_error") from None
        try:
            result = json.loads(raw)
            choice = result["choices"][0]
            content = choice["message"]["content"]
            finish_reason = choice.get("finish_reason")
        except (json.JSONDecodeError, UnicodeDecodeError, KeyError, IndexError, TypeError):
            response_path = self.output / "requests" / f"{case_id}.{stage}.response.raw"
            response_path.write_text(raw.decode("utf-8", "replace").replace(self.key, "[REDACTED]"))
            raise RequestFailure("Malformed Chat Completions response; no retry", f"{stage}_format", "response_format") from None
        usage = result.get("usage")
        if isinstance(usage, dict):
            self.usage.append(usage)
        write_json(self.output / "requests" / f"{case_id}.{stage}.response.json", result, self.key)
        if not isinstance(content, str) or finish_reason != "stop":
            raise RequestFailure("Model response is incomplete or not plain text; no retry", f"{stage}_incomplete", "incomplete_response")
        if self.key and self.key in content:
            raise RequestFailure("Model response contains a credential; redacted", f"{stage}_secret_redacted", "credential_redacted")
        return content, usage, finish_reason


def verify_runtime(prepared):
    hashes = json.loads((prepared / "source_hashes.json").read_text())
    paths = {"geometry_py": HERE / "geometry.py", "codec_py": HERE / "codec.py",
             "cohort_py": PROJECT / "experiments/naturecry_llm_100_20261002/cohort.py",
             "public_wyckoff_module": PROJECT / "experiments/explicit_full_mp20_20260923/codec/wyckoff_baseline.py",
             "prepare_py": HERE / "prepare.py", "preflight_py": HERE / "preflight.py",
             "api_runner_py": HERE / "api_runner.py", "inventory_schema_json": HERE / "inventory_schema.json",
             "decoder_schema_json": HERE / "decoder_schema.json", "rules_json": HERE / "rules.json",
             "protocol_md": HERE / "PROTOCOL.md"}
    for name, path in paths.items():
        if sha256(path.read_bytes()) != hashes[name]:
            raise RuntimeError(f"Prepared {name} hash does not match the current runtime")
    fixture = PROJECT / "experiments/explicit_full_mp20_20260923/codec/fixtures/wyckoff_list.csv"
    if sha256(fixture.read_bytes()) != hashes["public_wyckoff_fixture"]:
        raise RuntimeError("Frozen public table fixture hash mismatch")
    if sha256(codec.ENCODER_PROMPT.encode()) != hashes["encoder_prompt"] or sha256(codec.DECODER_PROMPT.encode()) != hashes["decoder_prompt"]:
        raise RuntimeError("Prepared prompt hashes do not match the current runtime")
    schema_hash = sha256(json.dumps(codec.DECODER_SCHEMA, sort_keys=True, separators=(",", ":")).encode())
    if schema_hash != hashes["decoder_schema"]:
        raise RuntimeError("Prepared decoder schema hash does not match the current runtime")
    if json.loads((HERE / "decoder_schema.json").read_text()) != codec.DECODER_SCHEMA:
        raise RuntimeError("Saved decoder schema content does not match the codec")
    if json.loads((HERE / "inventory_schema.json").read_text()) != prepare.INVENTORY_SCHEMA:
        raise RuntimeError("Saved inventory schema content does not match preparation code")
    if json.loads((HERE / "rules.json").read_text()) != prepare.GEOMETRY_RULES:
        raise RuntimeError("Saved public execution rules do not match preparation code")
    if sha256((prepared / "prompts.json").read_bytes()) != hashes["prompts_json"]:
        raise RuntimeError("Prepared prompt file hash mismatch")
    prompts = json.loads((prepared / "prompts.json").read_text())
    if prompts != {"encoder": codec.ENCODER_PROMPT, "decoder": codec.DECODER_PROMPT}:
        raise RuntimeError("Prepared prompt content does not match the current runtime")
    if hashes["public_wyckoff_fixture"] != geometry.FROZEN_TABLE_SHA256:
        raise RuntimeError("Frozen public table hash mismatch")
    return hashes


def verify_prepared(prepared, include_preflight=False):
    hashes = verify_runtime(prepared)
    manifest = json.loads((prepared / "manifest.json").read_text())
    selection_bytes = (prepared / "selection.json").read_bytes()
    selection = json.loads(selection_bytes)
    ids = manifest.get("case_ids_in_original_order", [])
    if manifest.get("selected") != 100 or selection.get("selected") != 100 or len(ids) != 100:
        raise RuntimeError("Prepared cohort must preserve exactly 100 selected cases")
    if len(set(ids)) != 100 or ids != [item["id"] for item in selection["cases"]]:
        raise RuntimeError("Prepared case order differs from the frozen selection")
    if sha256(selection_bytes) != manifest["selection_sha256"] or sha256(selection_bytes) != hashes["selection_json"]:
        raise RuntimeError("Frozen selection hash mismatch")
    if sha256(prepare.SELECTION.read_bytes()) != hashes["selection_json"]:
        raise RuntimeError("Original frozen selection file hash mismatch")
    if selection["dataset_sha256"] != hashes["selected_dataset"]:
        raise RuntimeError("Frozen selected-dataset hash reference mismatch")
    item_by_id = {item["id"]: item for item in selection["cases"]}
    record_by_id = {record["id"]: record for record in manifest["cases"]}
    if set(item_by_id) != set(ids) or set(record_by_id) != set(ids) or set(hashes["input_cifs"]) != set(ids):
        raise RuntimeError("Prepared manifest, selection, and source-hash inventories differ")
    input_count = fact_count = inventory_count = oracle_count = residual_map_count = 0
    prepared_count = sum(record["status"] == "prepared" for record in record_by_id.values())
    for case_id in ids:
        item, record = item_by_id[case_id], record_by_id[case_id]
        source = prepare.FROZEN / "inputs" / f"{case_id}.cif"
        source_hash = sha256(source.read_bytes())
        if source_hash != item["input_cif_sha256"] or source_hash != record["input_sha256_expected"] or source_hash != record["input_sha256_actual"] or source_hash != hashes["input_cifs"][case_id]:
            raise RuntimeError(f"Frozen input hash mismatch for {case_id}")
        input_count += 1
        if record["status"] != "prepared":
            continue
        case_dir = prepared / "cases" / case_id
        facts_path, inventory_path = case_dir / "encoder_facts.json", case_dir / "inventory.json"
        if sha256(facts_path.read_bytes()) != record["facts_sha256"]:
            raise RuntimeError(f"Prepared facts hash mismatch for {case_id}")
        facts = json.loads(facts_path.read_text())
        fact_count += 1
        stored = json.loads(inventory_path.read_text())
        expected_fields = [{"name": name, **spec} for name, spec in codec.make_inventory(facts).items()]
        if stored.get("codec_version") != codec.CODEC_VERSION or stored.get("fields") != expected_fields:
            raise RuntimeError(f"Prepared inventory content mismatch for {case_id}")
        if sha256(inventory_path.read_bytes()) != record["inventory_sha256"]:
            raise RuntimeError(f"Prepared inventory hash mismatch for {case_id}")
        inventory_count += 1
        if include_preflight:
            oracle_path = prepared / "preflight_only/oracle_plans" / f"{case_id}.json"
            map_path = prepared / "preflight_only/legacy_residual_maps" / f"{case_id}.json"
            if sha256(oracle_path.read_bytes()) != record["oracle_plan_sha256"]:
                raise RuntimeError(f"Preflight oracle-plan hash mismatch for {case_id}")
            if sha256(map_path.read_bytes()) != record["legacy_residual_map_sha256"]:
                raise RuntimeError(f"Preflight residual-map hash mismatch for {case_id}")
            oracle_count += 1
            residual_map_count += 1
    checks = {
        "selection_json": {"expected": 1, "verified": 1},
        "original_selection_json": {"expected": 1, "verified": 1},
        "selected_dataset_hash_reference": {"expected": 1, "verified": 1},
        "input_cifs": {"expected": 100, "verified": input_count},
        "source_facts": {"expected": prepared_count, "verified": fact_count},
        "inventories": {"expected": prepared_count, "verified": inventory_count},
        "runtime_code_protocol_and_rules": {"expected": len(RUNTIME_HASH_KEYS), "verified": len(RUNTIME_HASH_KEYS)},
        "prompt_and_decoder_schema_constants": {"expected": 3, "verified": 3},
        "prepared_prompt_file": {"expected": 1, "verified": 1},
    }
    if include_preflight:
        checks["oracle_plans"] = {"expected": prepared_count, "verified": oracle_count}
        checks["legacy_residual_maps"] = {"expected": prepared_count, "verified": residual_map_count}
    return hashes, manifest, selection, checks


def execute(prepared, output, key, max_requests, opener=None):
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite API run output: {output}")
    hashes, manifest, selection, hash_verification = verify_prepared(prepared)
    prepared_records = {record["id"]: record for record in manifest["cases"]}
    output.mkdir(parents=True)
    runner = FirstPassRunner(key, output, max_requests, opener)
    table = geometry.verify_public_table()
    input_root = prepare.FROZEN / "inputs"
    records = []
    for case_id in manifest["case_ids_in_original_order"]:
        record = {"case": case_id, "status": "fail", "stage": "encoder_transport",
                  "encoder_requests": 0, "decoder_requests": 0}
        if prepared_records[case_id]["status"] != "prepared":
            record.update(stage="source_preparation", failure_kind="source_preparation",
                          error="Source preparation failed; case remains in N=100 denominator")
            records.append(record)
            write_json(output / "records.json", records, key)
            continue
        if runner.request_count >= runner.max_requests:
            record.update(stage="budget_exhausted", failure_kind="budget_exhausted",
                          error="No request issued; retained in N=100 denominator")
            records.append(record)
            continue
        case_dir = prepared / "cases" / case_id
        try:
            facts = json.loads((case_dir / "encoder_facts.json").read_text())
            inventory_file = json.loads((case_dir / "inventory.json").read_text())
            inventory = {field["name"]: {key: field[key] for key in ("type", "unit", "frame", "value")}
                         for field in inventory_file["fields"]}
            record.update(input_sha256=prepared_records[case_id]["input_sha256_actual"],
                          facts_sha256=prepared_records[case_id]["facts_sha256"],
                          encoder_prompt_sha256=hashes["encoder_prompt"],
                          decoder_prompt_sha256=hashes["decoder_prompt"],
                          decoder_schema_sha256=hashes["decoder_schema"])
            before = runner.request_count
            try:
                encoder_text, encoder_usage, encoder_finish = runner.request(encoder_messages(facts, inventory), 0.4, case_id, "encoder")
            finally:
                record["encoder_requests"] = runner.request_count - before
            record["encoder_usage"] = encoder_usage
            record["encoder_finish_reason"] = encoder_finish
            record["stage"] = "encoder_format"
            inline = codec.bind_inline(encoder_text, inventory)
            codec.validate_encoder_document(inline, inventory)
            record["encoder_gate_passed"] = True
            record["prose_semantic_audit"] = "not_executed; independent review required"
            (output / "inline").mkdir(exist_ok=True)
            (output / "inline" / f"{case_id}.txt").write_text(inline + "\n")

            record["stage"] = "decoder_transport"
            before = runner.request_count
            try:
                decoder_text, decoder_usage, decoder_finish = runner.request(decoder_messages(inline), 0.0, case_id, "decoder")
            finally:
                record["decoder_requests"] = runner.request_count - before
            record["decoder_usage"] = decoder_usage
            record["decoder_finish_reason"] = decoder_finish
            record["inline_sha256"] = sha256(inline.encode())
            record["stage"] = "decoder_format_and_schema"
            plan = codec.decode_plan_response(decoder_text, inline, table)
            record["decoder_gate_passed"] = True
            write_json(output / "decoded_plans" / f"{case_id}.json", plan, key)
            record["stage"] = "geometry_execution"
            rebuilt = geometry.execute_geometry_plan(plan, table)
            record["geometry_completed"] = True
            decoded_cif = output / "decoded_cifs" / f"{case_id}.cif"
            decoded_cif.parent.mkdir(parents=True, exist_ok=True)
            CifWriter(rebuilt, symprec=None, significant_figures=14).write_file(decoded_cif)
            record["stage"] = "original_structure_match"
            source = input_root / f"{case_id}.cif"
            expected_hash = next(item["input_cif_sha256"] for item in selection["cases"] if item["id"] == case_id)
            if sha256(source.read_bytes()) != expected_hash:
                raise RuntimeError("Frozen input changed before scoring")
            reparsed = cohort.parse(decoded_cif)
            coord_error, lattice_error = geometry.compare_structures(cohort.parse(source), reparsed)
            record.update(status="pass", stage="complete", coordinate_error_A=coord_error,
                          lattice_error_A=lattice_error, atom_count=len(reparsed),
                          decoded_cif_sha256=sha256(decoded_cif.read_bytes()))
        except Exception as error:
            if isinstance(error, RequestFailure):
                record["stage"] = error.stage
                failure_kind = error.failure_kind
            else:
                failure_kind = getattr(error, "stage", "runtime_error")
            record.update(failure_kind=failure_kind, error=redact(f"{type(error).__name__}: {error}", key))
        records.append(record)
        write_json(output / "records.json", records, key)

    attempts = sum(record["encoder_requests"] + record["decoder_requests"] for record in records)
    if attempts != runner.request_count or attempts > MAX_REQUESTS or attempts > max_requests:
        raise RuntimeError("Request ledger does not reconcile with per-case records")
    passed = [record for record in records if record["status"] == "pass"]
    encoder_gate_count = sum(record.get("encoder_gate_passed", False) for record in records)
    decoder_gate_count = sum(record.get("decoder_gate_passed", False) for record in records)
    summary = {
        "provider": "SciencePal",
        "endpoint": ENDPOINT,
        "model_alias": MODEL,
        "selected_denominator": 100,
        "completed": len(records),
        "strict_firstpass_matches": len(passed),
        "firstpass_match_rate": len(passed) / 100,
        "failed": 100 - len(passed),
        "failures_by_stage": dict(Counter(record["stage"] for record in records if record["status"] != "pass")),
        "failures_by_kind": dict(Counter(record.get("failure_kind", "unspecified") for record in records if record["status"] != "pass")),
        "gate_conditioned_matches": {
            "encoder_gate_passed_cases": encoder_gate_count,
            "strict_matches_after_encoder_gate": len(passed),
            "strict_match_rate_after_encoder_gate": len(passed) / encoder_gate_count if encoder_gate_count else None,
            "decoder_gate_passed_cases": decoder_gate_count,
            "strict_matches_after_decoder_gate": len(passed),
            "strict_match_rate_after_decoder_gate": len(passed) / decoder_gate_count if decoder_gate_count else None,
        },
        "encoder_requests": sum(record["encoder_requests"] for record in records),
        "decoder_requests": sum(record["decoder_requests"] for record in records),
        "total_requests": attempts,
        "stage_counts": {
            "encoder_generation_attempts": sum(record["encoder_requests"] for record in records),
            "decoder_generation_attempts": sum(record["decoder_requests"] for record in records),
            "encoder_gate_passed_cases": encoder_gate_count,
            "decoder_gate_passed_cases": decoder_gate_count,
            "geometry_executed_cases": sum(record.get("geometry_completed", False) for record in records),
        },
        "request_limit": max_requests,
        "requests_reconciled_to_case_records": attempts == runner.request_count,
        "automatic_retries": 0,
        "repairs": 0,
        "semantic_prose_audit": "not_executed; first-pass prose needs independent review",
        "reported_token_usage": runner.usage,
        "token_total": sum(item.get("total_tokens", 0) for item in runner.usage if isinstance(item.get("total_tokens"), int)),
        "cost": None,
        "hash_verification": hash_verification,
        "runtime_hashes": hashes,
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    write_json(output / "records.json", records, key)
    write_json(output / "summary.json", summary, key)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path, default=HERE / "prepared_final")
    parser.add_argument("--out", type=Path, default=HERE / "api_runs/firstpass_001")
    parser.add_argument("--max-requests", type=int, default=MAX_REQUESTS)
    parser.add_argument("--execute-api", action="store_true", help="Explicitly enable up to 200 first-pass generation requests")
    args = parser.parse_args()
    if not args.execute_api:
        parser.error("network disabled; pass --execute-api after protocol review")
    if not 1 <= args.max_requests <= MAX_REQUESTS:
        parser.error("--max-requests must be between 1 and 200")
    key = os.environ.get("SCIENCEPAL_API_KEY")
    if not key:
        parser.error("SCIENCEPAL_API_KEY is not set; no credential prompt or fallback is used")
    summary = execute(args.prepared.resolve(), args.out.resolve(), key, args.max_requests)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
