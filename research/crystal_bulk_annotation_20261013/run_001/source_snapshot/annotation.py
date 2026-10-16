"""Run the approved English annotation prompt on a small, frozen fact cohort."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import time
import urllib.error
import urllib.request


def sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def load_key(path):
    for line in Path(path).read_text().splitlines():
        if line.startswith("SCIENCEPAL_API_KEY="):
            return line.split("=", 1)[1].strip().strip("\"'")
    return os.environ["SCIENCEPAL_API_KEY"]


def description_facts(row):
    local = row["local_structure_facts"]
    groups = {}
    for env in local["coordination_environments"]:
        key = (env["center_element"], env["coordination_number"],
               json.dumps(env["neighbor_element_counts"], sort_keys=True), env["recognized_motif"])
        if key not in groups:
            groups[key] = {"center_element": env["center_element"],
                "geometric_neighbors_per_center_atom": env["coordination_number"],
                "neighbor_elements_per_center_atom": env["neighbor_element_counts"],
                "recognized_geometric_motif": {"cp": "close-packed-like local arrangement",
                    "bcc": "body-centered-cubic-like local arrangement"}.get(env["recognized_motif"], env["recognized_motif"]),
                "number_of_independent_orbits_with_this_environment": 0,
                "number_of_unit_cell_atoms_with_this_environment": 0,
                "neighbor_distance_range_A": None}
        group = groups[key]
        group["number_of_independent_orbits_with_this_environment"] += 1
        group["number_of_unit_cell_atoms_with_this_environment"] += env["orbit_multiplicity"]
        interval = env["neighbor_distance_range_A"]
        if interval:
            previous = group["neighbor_distance_range_A"] or interval
            group["neighbor_distance_range_A"] = [min(previous[0], interval[0]), max(previous[1], interval[1])]
    sharing = local["polyhedra_sharing"]
    if sharing.get("status") == "computed":
        relations = []
        for pair in sharing["orbit_pair_relations"]:
            centers = [{"center_element": c["center_element"], "coordination_number": c["coordination_number"],
                        "polyhedron_type": c["polyhedron_label"] or f'{c["coordination_number"]}-vertex coordination polyhedron'}
                       for c in pair["centers"]]
            types = [kind for kind, count in pair["sharing_counts"].items() if count > 0]
            if types:
                named = sum(c["polyhedron_label"] is not None for c in pair["centers"])
                relations.append((named, pair["relation_count_per_cell"], {"centers": centers, "observed_sharing_types": types}))
        relations.sort(key=lambda r: (r[0], r[1]), reverse=True)
        sharing = {"status": "computed", "basis": "validated periodic convex coordination-polyhedron contacts",
                   "selected_observed_relations": [r[2] for r in relations[:4]],
                   "selection_scope": "selected observed patterns; does not assert an exclusive type for all polyhedra"}
    else:
        sharing = {"status": sharing.get("status", "not_computed")}
    return {"analysis": "CrystalNN geometry-only; neighbor graph edges are not chemical bonds",
            "coordination_environment_groups": list(groups.values()),
            "geometric_neighbor_graph_connectivity": local["connectivity"],
            "polyhedra_sharing": sharing}


def payload_for(row, config, template):
    assert row["status"] == "ok"
    core = {key: row["core_facts"][key] for key in ["cell_formula", "reduced_formula", "elements",
        "element_count", "cell_atom_count", "space_group_number", "space_group_symbol",
        "crystal_system", "independent_orbit_count"]}
    text = template.format(
        structure_version=f'PLaID-decoded structure, material {row["material_id"]}',
        core_facts=json.dumps(core, ensure_ascii=False),
        local_structure_facts=json.dumps(description_facts(row), ensure_ascii=False),
        property_labels="Not supplied to this language-rewriting step. Do not discuss energy above hull; its exact source label is rendered separately.",
    )
    return {"model": config["model"], "messages": [{"role": "user", "content": text}],
            "thinking": config["thinking"], "reasoning_effort": config["reasoning_effort"],
            "max_tokens": config["max_tokens"], "stream": True,
            "stream_options": {"include_usage": True}}


def render_description(structure_description, row):
    label = row["property_labels"]
    text = structure_description.replace("PLaID-decoded ", "")
    text = text.replace(" of material " + row["material_id"], "")
    text = re.sub(r"\s*\(" + re.escape(row["material_id"]) + r"\)", "", text)
    text = re.sub(r"\b(?:Theoretical density|Polyhedra-sharing information) was not (?:supplied|computed)\.\s*", "", text)
    text = re.sub(r"; polyhedra-sharing analysis was not computed, and no property labels are supplied here\.", ".", text)
    text = re.sub(r"\bNo (?:polyhedra-sharing analysis or )?property labels were supplied for this structure\.\s*", "", text)
    text = re.sub(r"; (?:polyhedra-sharing analysis was not computed|no polyhedral-sharing analysis was computed)\.", ".", text)
    sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z])", text.rstrip())
    if len(sentences) == 5:
        second = sentences[1]
        for word in ("Its", "The", "This", "It", "Geometry"):
            if second.startswith(word + " ") or second.startswith(word + "-"):
                second = second[0].lower() + second[1:]
                break
        text = sentences[0].rstrip(".") + "; " + second + " " + " ".join(sentences[2:])
    return (text.rstrip() + " The original archived MP structure has a supplied energy above hull label of "
            + str(label["energy_above_hull"]) + " " + label["unit"] + ".")


def read_stream(response, started_at=None):
    started_at = time.monotonic() if started_at is None else started_at
    parts, reasoning, usage, model, finish = [], [], None, None, None
    first_content, last_content, content_chunks = None, None, 0
    for raw in response:
        line = raw.decode("utf-8").strip()
        if not line.startswith("data:"):
            continue
        data = line[5:].strip()
        if data == "[DONE]":
            break
        event = json.loads(data)
        if event.get("error"):
            raise RuntimeError(json.dumps(event["error"]))
        model = event.get("model") or model
        usage = event.get("usage") or usage
        for choice in event.get("choices", []):
            delta = choice.get("delta", {})
            if delta.get("content"):
                now = time.monotonic()
                first_content = now if first_content is None else first_content
                last_content = now
                content_chunks += 1
            parts.append(delta.get("content") or "")
            reasoning.append(delta.get("reasoning_content") or "")
            finish = choice.get("finish_reason") or finish
    text = "".join(parts).strip()
    return {"description": text, "returned_model": model, "finish_reason": finish,
            "usage": usage, "reasoning_content_characters": len("".join(reasoning)),
            "time_to_first_content_seconds": None if first_content is None else round(first_content - started_at, 4),
            "content_stream_seconds": None if first_content is None else round(last_content - first_content, 4),
            "content_chunks": content_chunks}


def save_receipts(path, receipts):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in receipts))
    temporary.replace(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--config", default="research/wayfinder_crystal_20261011/annotation-model-config.json")
    ap.add_argument("--prompt", default="research/wayfinder_crystal_20261011/description-template-draft.md")
    ap.add_argument("--env", default=".env")
    ap.add_argument("--limit", type=int, default=30)
    args = ap.parse_args()
    assert 1 <= args.limit <= 30, "This client is limited to the authorized 30-material pilot"
    assert Path(args.input).resolve() != Path(args.output).resolve(), "Output must differ from input facts"
    config = json.loads(Path(args.config).read_text())
    assert config["thinking"] == {"type": "enabled"} and config["reasoning_effort"] == "max"
    template = Path(args.prompt).read_text().split("```text\n", 1)[1].split("```", 1)[0]
    rows = [json.loads(line) for line in Path(args.input).read_text().splitlines()]
    assert len(rows) <= args.limit and all(r["status"] == "ok" for r in rows)
    assert len({r["material_id"] for r in rows}) == len(rows)
    endpoint = config["base_url"].rstrip("/") + "/chat/completions"
    key = load_key(args.env)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = {}
    receipts = []
    if out.exists():
        for line in out.read_text().splitlines():
            item = json.loads(line)
            receipts.append(item)
            done[item["material_id"]] = item
    for row in rows:
        payload = payload_for(row, config, template)
        code_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        fingerprint = sha({"facts": row, "endpoint": endpoint, "payload": payload, "annotation_code_sha256": code_hash})
        previous = done.get(row["material_id"])
        if previous:
            assert previous["request_sha256"] == fingerprint, "Changed facts or configuration on resume"
            if previous["status"] == "ok":
                continue
        receipt = {"material_id": row["material_id"], "split": row["split"],
                   "request_sha256": fingerprint, "facts_sha256": sha(row),
                   "endpoint": endpoint, "payload": payload, "property_labels": row["property_labels"],
                   "annotation_code_sha256": code_hash}
        started = time.monotonic()
        request = urllib.request.Request(endpoint, json.dumps(payload).encode(),
            {"Authorization": "Bearer " + key, "Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                receipt.update(read_stream(response, started))
                receipt["http_status"] = response.status
            assert receipt["returned_model"] == config["model"], "Returned model mismatch"
            assert receipt["description"] and receipt["finish_reason"] == "stop", "Incomplete response"
            receipt["structure_description"] = receipt["description"]
            receipt["description"] = render_description(receipt["structure_description"], row)
            receipt["status"] = "ok"
        except Exception as exc:
            receipt.update(status="error", error_type=type(exc).__name__, error=str(exc).replace(key, "[redacted]"))
            if isinstance(exc, urllib.error.HTTPError):
                receipt["http_status"] = exc.code
            receipt["elapsed_seconds"] = round(time.monotonic() - started, 3)
            receipts.append(receipt)
            save_receipts(out, receipts)
            raise RuntimeError(f'Annotation failed for {row["material_id"]}: {type(exc).__name__}') from None
        receipt["elapsed_seconds"] = round(time.monotonic() - started, 3)
        receipts.append(receipt)
        save_receipts(out, receipts)
        print(json.dumps({"material_id": row["material_id"], "status": receipt["status"],
                          "seconds": receipt["elapsed_seconds"], "usage": receipt["usage"]}), flush=True)
        time.sleep(1)


if __name__ == "__main__":
    main()
