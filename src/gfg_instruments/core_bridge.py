"""Native Core v3 export, without modifying or impersonating the Core validator."""
from __future__ import annotations

from dataclasses import asdict
import hashlib
import importlib
import json
from pathlib import Path
import platform
import sys
from importlib.metadata import version

from .errors import require
from .evidence import validate_journal
from .identity import canonical, digest

VENDOR = Path(__file__).parent / "_vendor"
CORE = VENDOR / "core"


def verify_vendored_sources():
    document = json.loads((VENDOR / "IDENTITY.json").read_text("utf-8"))
    hashes = {}
    for row in document["files"]:
        path = (VENDOR / row["path"]).resolve()
        require(path.is_relative_to(VENDOR.resolve()), "VENDOR_PATH_ESCAPE")
        raw = path.read_bytes()
        require(len(raw) == row["size"] and hashlib.sha256(raw).hexdigest() == row["sha256"], "VENDORED_SOURCE_HASH_MISMATCH:" + row["path"])
        if row["path"].startswith("core/"):
            hashes[row["path"].removeprefix("core/")] = row["sha256"]
    return hashes


def load_core():
    hashes = verify_vendored_sources()
    if "generation_relation_core" in sys.modules:
        existing = Path(sys.modules["generation_relation_core"].__file__).resolve()
        require(existing.is_relative_to(CORE.resolve()), "OTHER_CORE_ALREADY_IMPORTED")
    for path in (CORE, CORE / "src"):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    core = importlib.import_module("generation_relation_core")
    identity = importlib.import_module("generation_relation_core.implementation_identity")
    expected = {name: hashes[name] for name, _ in identity.IMPLEMENTATION_FILES}
    return core, expected


def record_membership(support, query, predicate):
    return predicate == "membership" and support["record_id"] == query["record_id"]


def _registry(domain):
    from generation_relation_core import entities as e
    from generation_relation_core.predicate_registry import PredicateRegistry, implementation_sha256
    text = {"type": "string", "minLength": 1}
    schema = {"type": "object", "additionalProperties": False,
              "properties": {"record_id": text, "artifact_sha256": text},
              "required": ["record_id", "artifact_sha256"]}
    query = {"type": "object", "additionalProperties": False, "properties": {"record_id": text}, "required": ["record_id"]}
    space = e.support_space(domain_scope_id=domain, support_space_name="instrument_record_identity",
                            support_payload_schema=schema, query_payload_schema=query,
                            normalization_rule="exact registered record identity; no value-based coalescence")
    profile = e.predicate_profile(domain_scope_id=domain, support_space_id=space["support_space_id"],
                                 predicate_kind="exact_record_membership", supported_predicates=["membership"],
                                 predicate_authority="gfg-neural-instruments", authorized=True,
                                 implementation_module=__name__, implementation_symbol="record_membership",
                                 predicate_implementation_sha256=implementation_sha256(record_membership),
                                 normalization_rule=space["normalization_rule"], result_ordering_rule="content-addressed support_id")
    registry = PredicateRegistry([space], [profile], {profile["predicate_profile_id"]: record_membership})
    return space, profile, registry


def instrument_code_hashes():
    root = Path(__file__).parent
    return {p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(root.rglob("*.py")) if "_vendor" not in p.parts}


def export_core(root):
    document = validate_journal(root)
    core, expected = load_core()
    from generation_relation_core import entities as e
    from generation_relation_core.canonical import canonical_bytes, payload_sha256
    from compat.v2.projections import derive_legacy_projections
    domain = "neural-instruments:" + document["run_id"]
    space, profile, registry = _registry(domain)
    tables = core.CoreV3Tables(support_space_records=[space], predicate_profiles=[profile])
    code = instrument_code_hashes()
    code_hash = digest(code)
    environment = e.environment_record(runtime_name="CPython", runtime_version=platform.python_version(),
                                       operating_system=platform.platform(), dependency_hashes={"instrument_source": code_hash,
                                       "core_source": digest(expected),
                                       **{name + "_version": digest(version(name)) for name in ("torch", "numpy", "jsonschema")}})
    tables.environment_records.append(environment)
    manifest = e.generator_manifest(generator_name="gfg-neural-instruments", generator_version="0.1.0",
                                     generator_code_hash=code_hash, supported_support_space_ids=[space["support_space_id"]],
                                     supported_predicate_profile_ids=[profile["predicate_profile_id"]],
                                     supported_operations=sorted({r["operation"] for r in document["events"]}),
                                     authorized_evidence_authorities=["actual-local-instrument-journal"], dependency_hashes=[digest(expected)])
    tables.generator_manifests.append(manifest)
    origin_refs, outcomes, occurrences = {}, {}, {}
    for row in document["sources"]:
        source = e.source_information(domain_scope_id=domain, source_identity=row["id"], source_parent_id=None,
                                      source_granularity="registered_native_object",
                                      source_payload={"identity": row["identity"], "artifact": row["artifact"]})
        tables.source_information_records.append(source)
        origin_refs[row["id"]] = {"kind": "registered_source", "source_information_id": source["source_information_id"]}
    relation_materials = []
    for row in document["events"]:
        occurrence = e.generation_occurrence(domain_scope_id=domain, generator_manifest_id=manifest["generator_manifest_id"],
                                             occurrence_stage=row["operation"], occurrence_type="actual_instrument_execution",
                                             stable_instance_key=row["id"], occurrence_index=row["index"],
                                             transform_reference={"operation": row["operation"], "source_code_hash": code_hash},
                                             occurrence_payload={"receipt_sha256": row["receipt_sha256"], "artifact": row["artifact"]})
        tables.generation_occurrences.append(occurrence)
        occurrences[row["id"]] = occurrence["generation_occurrence_id"]
        if row["disposition"]:
            outcome = e.explicit_disposition(domain_scope_id=domain, core_disposition_category="unverifiable",
                                              domain_reason_code=row["disposition"], disposition_payload={"record_id": row["id"], "artifact": row["artifact"]})
            tables.explicit_dispositions.append(outcome)
            outcome_ref = {"kind": "disposition", "disposition_id": outcome["disposition_id"]}
        else:
            outcome = e.perceptual_support(domain_scope_id=domain, support_space_id=space["support_space_id"],
                                           predicate_profile_id=profile["predicate_profile_id"],
                                           support_payload={"record_id": row["id"], "artifact_sha256": row["artifact"]["sha256"]})
            tables.perceptual_support_records.append(outcome)
            outcome_ref = {"kind": "support", "support_id": outcome["support_id"]}
        outcomes[row["id"]] = outcome_ref
        produced, evidence_ids = [], []
        for parent in row["parents"]:
            if parent not in origin_refs:
                require(parent in outcomes and parent in occurrences, "GENERATED_ORIGIN_WITHOUT_EXECUTED_PRODUCER")
                origin = e.generated_origin(domain_scope_id=domain, generator_manifest_id=manifest["generator_manifest_id"],
                                             origin_type="instrument_result_from_actual_occurrence",
                                             origin_payload={"producer_record": parent, "producer_occurrence": occurrences[parent],
                                                             "producer_outcome": outcomes[parent]})
                tables.generated_origins.append(origin)
                origin_refs[parent] = {"kind": "generated_origin", "generated_origin_id": origin["generated_origin_id"]}
            origin_ref = origin_refs[parent]
            material = e.relation_material(domain_scope_id=domain, origin_reference=origin_ref,
                                            generation_occurrence_id=occurrence["generation_occurrence_id"],
                                            outcome_reference=outcome_ref, relation_role="explicit_input_dependency")
            related = [next(v for k, v in origin_ref.items() if k != "kind"), occurrence["generation_occurrence_id"],
                       next(v for k, v in outcome_ref.items() if k != "kind")]
            evidence = e.relation_evidence_for_material(material,
                        artifact_locator=f"candidate://relation_materials.jsonl#sha256={payload_sha256(material)}",
                        evidence_authority="actual-local-instrument-journal", extraction_method="exact_declared_execution_inputs",
                        extraction_code_hash=code_hash, environment_hash=environment["environment_payload_sha256"], related_record_ids=related)
            tables.evidence_records.append(evidence)
            binding = e.generation_binding(
                domain_scope_id=domain, origin_reference=origin_ref, generation_occurrence_id=occurrence["generation_occurrence_id"],
                outcome_reference=outcome_ref, relation_role="explicit_input_dependency", evidence_ids=[evidence["evidence_id"]])
            tables.generation_bindings.append(binding)
            tables.evidence_links.append(e.evidence_link(evidence_id=evidence["evidence_id"], subject_type="generation_binding",
                                                        subject_id=binding["generation_binding_id"], evidence_role="primary_generation_relation"))
            produced.append(binding["generation_binding_id"])
            evidence_ids.append(evidence["evidence_id"])
            relation_materials.append(canonical_bytes(material))
        tables.generator_operation_results.append(e.generator_operation_result(
            generator_manifest_id=manifest["generator_manifest_id"], operation_name=row["operation"],
            produced_entity_ids=produced, evidence_ids=evidence_ids))
    tables.legacy_source_binding_projections, tables.legacy_occurrence_binding_projections = derive_legacy_projections(
        tables.source_information_records, tables.generation_occurrences, tables.generation_bindings)
    snapshot = core.build_snapshot(tables, registry, expected_implementation_hashes=expected)
    validation = core.validate_snapshot(snapshot, registry, expected_implementation_hashes=expected)
    package = {"record": snapshot.record, "tables": asdict(tables), "instrument_source_hashes": code,
               "journal_sha256": document["journal_sha256"]}
    root = Path(root)
    require(not (root / "core_snapshot.json").exists(), "CORE_SNAPSHOT_ALREADY_EXISTS")
    (root / "core_snapshot.json").write_bytes(canonical(package))
    (root / "relation_materials.jsonl").write_bytes(b"\n".join(relation_materials) + b"\n")
    return dict(status="PASS", validator="unmodified Core v3", snapshot_id=validation.snapshot_id,
                relations=len(validation.relation_evidence), occurrences=len(tables.generation_occurrences),
                generated_origins=len(tables.generated_origins), dispositions=len(tables.explicit_dispositions))


def validate_core_export(root):
    document = validate_journal(root)
    core, expected = load_core()
    package = json.loads((Path(root) / "core_snapshot.json").read_text("utf-8"))
    require(package["journal_sha256"] == document["journal_sha256"], "CORE_JOURNAL_BINDING_MISMATCH")
    tables = core.CoreV3Tables(**package["tables"])
    _, _, registry = _registry("neural-instruments:" + document["run_id"])
    snapshot = core.ValidatedSnapshot(package["record"], tables)
    result = core.validate_snapshot(snapshot, registry, expected_implementation_hashes=expected)
    # Core validates material hashes; also verify that the promised artifact exists.
    raw = (Path(root) / "relation_materials.jsonl").read_bytes().splitlines()
    material_hashes = {hashlib.sha256(line).hexdigest() for line in raw}
    require(material_hashes == {b["relation_material_sha256"] for b in tables.generation_bindings}, "RELATION_MATERIAL_FILE_MISMATCH")
    by_event = {r["id"]: r for r in document["events"]}
    by_source = {r["id"]: r for r in document["sources"]}
    require(len(by_source) == len(tables.source_information_records), "CORE_SOURCE_COVERAGE_MISMATCH")
    origin_to_record = {}
    for source in tables.source_information_records:
        record = by_source.get(source["source_identity"])
        require(record is not None and source["source_payload"] == {"identity": record["identity"], "artifact": record["artifact"]}, "CORE_SOURCE_BINDING_MISMATCH")
        origin_to_record[source["source_information_id"]] = record["id"]
    occurrence_by_record = {row["stable_instance_key"]: row["generation_occurrence_id"] for row in tables.generation_occurrences}
    outcomes = {}
    for support in tables.perceptual_support_records:
        record = by_event[support["support_payload"]["record_id"]]
        require(support["support_payload"]["artifact_sha256"] == record["artifact"]["sha256"] and not record["disposition"], "CORE_OUTCOME_BINDING_MISMATCH")
        outcomes[record["id"]] = {"kind": "support", "support_id": support["support_id"]}
    for disposition in tables.explicit_dispositions:
        record = by_event[disposition["disposition_payload"]["record_id"]]
        require(disposition["domain_reason_code"] == record["disposition"] and disposition["disposition_payload"]["artifact"] == record["artifact"], "CORE_DISPOSITION_BINDING_MISMATCH")
        outcomes[record["id"]] = {"kind": "disposition", "disposition_id": disposition["disposition_id"]}
    require(set(outcomes) == set(by_event), "CORE_OUTCOME_COVERAGE_MISMATCH")
    for origin in tables.generated_origins:
        value = origin["origin_payload"]
        record_id = value["producer_record"]
        require(value["producer_occurrence"] == occurrence_by_record[record_id]
                and value["producer_outcome"] == outcomes[record_id], "CORE_GENERATED_PRODUCER_BINDING_MISMATCH")
        origin_to_record[origin["generated_origin_id"]] = record_id
    require(len(by_event) == len(tables.generation_occurrences), "CORE_OCCURRENCE_COVERAGE_MISMATCH")
    for occurrence in tables.generation_occurrences:
        event = by_event.get(occurrence["stable_instance_key"])
        require(event is not None and event["receipt_sha256"] == occurrence["occurrence_payload"]["receipt_sha256"]
                and event["artifact"] == occurrence["occurrence_payload"]["artifact"], "CORE_EXECUTION_BINDING_MISMATCH")
        bindings = [r for r in tables.generation_bindings if r["generation_occurrence_id"] == occurrence["generation_occurrence_id"]]
        parents = []
        for binding in bindings:
            origin = next(v for k, v in binding["origin_reference"].items() if k != "kind")
            parents.append(origin_to_record[origin])
            require(binding["outcome_reference"] == outcomes[event["id"]], "CORE_BINDING_OUTCOME_MISMATCH")
        require(sorted(parents) == sorted(event["parents"]), "CORE_EXACT_INPUT_BINDING_MISMATCH")
    return dict(status="PASS", snapshot_id=result.snapshot_id, relations=len(result.relation_evidence))
