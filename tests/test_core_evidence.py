from copy import deepcopy
import json

import pytest

from gfg_instruments.core_bridge import export_core, validate_core_export, verify_vendored_sources, load_core, _registry
from gfg_instruments.evidence import Journal, validate_journal
from gfg_instruments.errors import MeasurementError


def two_events(journal):
    a = journal.source("source-a", {"number": 1})
    b = journal.source("source-b", {"number": 2})
    first = journal.record("actual-add", [a, b], {"number": 3})
    second = journal.record("actual-identity", [first], {"number": 3})
    journal.record("unsupported-readout", [second], {"reason": "no-target"}, disposition="TARGET_UNAVAILABLE")
    journal.save()


def test_unmodified_core_full_validation_and_generated_origins(journal):
    two_events(journal)
    result = export_core(journal.root)
    assert result["status"] == "PASS" and result["relations"] == 4
    assert result["occurrences"] == 3 and result["generated_origins"] == 2
    assert result["dispositions"] == 1
    assert validate_core_export(journal.root)["status"] == "PASS"


def test_no_cartesian_product_unknown_origin_and_equal_results_distinct(journal):
    with pytest.raises(MeasurementError, match="UNKNOWN_INPUT"):
        journal.record("bad", ["unknown"], {})
    a = journal.source("one", {"v": 1})
    one = journal.record("first", [a], {"v": 2})
    two = journal.record("second", [a], {"v": 2})
    assert one != two and journal.events[0]["artifact"] == journal.events[1]["artifact"]
    journal.save()
    result = export_core(journal.root)
    assert result["occurrences"] == result["relations"] == 2


def test_core_negative_missing_primary_evidence(journal):
    two_events(journal)
    export_core(journal.root)
    core, expected = load_core()
    package = json.loads((journal.root / "core_snapshot.json").read_text("utf-8"))
    tables = core.CoreV3Tables(**package["tables"])
    tables.evidence_links.clear()
    _, _, registry = _registry("neural-instruments:" + journal.run_id)
    from generation_relation_core.errors import CoreV3Error
    with pytest.raises(CoreV3Error, match="BINDING_HAS_NO_PRIMARY_EVIDENCE"):
        core.build_snapshot(tables, registry, expected_implementation_hashes=expected)


def test_core_negative_source_coverage_and_bad_hash(journal):
    two_events(journal)
    export_core(journal.root)
    core, expected = load_core()
    package = json.loads((journal.root / "core_snapshot.json").read_text("utf-8"))
    tables = core.CoreV3Tables(**package["tables"])
    from generation_relation_core.errors import CoreV3Error
    from generation_relation_core import entities as e
    _, _, registry = _registry("neural-instruments:" + journal.run_id)
    tables.source_information_records.append(e.source_information(
        domain_scope_id="neural-instruments:" + journal.run_id, source_identity="unused", source_parent_id=None,
        source_granularity="test", source_payload={}))
    with pytest.raises(CoreV3Error, match="SOURCE_COVERAGE_FAILED"):
        core.build_snapshot(tables, registry, expected_implementation_hashes=expected)
    tables.source_information_records[0]["source_payload"] = {"tampered": True}
    with pytest.raises(CoreV3Error, match="HASH_OR_ID_MISMATCH"):
        core.build_snapshot(tables, registry, expected_implementation_hashes=expected)


def test_vendored_core_hashes():
    assert "src/generation_relation_core/snapshots.py" in verify_vendored_sources()
