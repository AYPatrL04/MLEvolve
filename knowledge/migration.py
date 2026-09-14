"""Resumable, evidence-preserving backfill for design knowledge v2.

The default CLI action is an inventory only. --apply writes a local migration
ledger and a validated export; --publish additionally updates derived domain
storage using the existing domain settings. Scheduler storage is never opened.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sqlite3
from typing import Any, Iterable

import yaml

from knowledge.lessons import publication_records
from knowledge.records import SCHEMA_VERSION, digest, from_source, source_identity, validate_record


def load_sources(path: Path, domain: str) -> list[dict[str, Any]]:
    if path.suffix in {".sqlite", ".sqlite3", ".db"}:
        if domain != "lesson":
            raise ValueError("Only the lesson-profile SQLite database is a migration source")
        with sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True) as connection:
            tables = {row[0] for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")}
            if not {"profiles", "profile_revisions", "qdrant_outbox", "lessons"}.issubset(tables):
                raise ValueError("Source is not a lesson-profile database")
            rows = connection.execute("SELECT payload_json FROM qdrant_outbox ORDER BY profile_key, revision_number").fetchall()
        return [{"record_id": f"{p['profile_key']}:{p['revision_number']}", "publication": p} for p in (json.loads(row[0]) for row in rows)]
    payload = yaml.safe_load(path.read_text())
    if isinstance(payload, list):
        return payload
    if "nodes" in payload:
        result = []
        for node in payload["nodes"]:
            props = dict(node.get("properties") or {})
            if node.get("label") == "Hardware":
                # Hardware identity is a fact, not an optimization recipe. Keep
                # per-stage conditions as separate complete source records.
                fields = ("name", "gpu_name", "architecture", "compute_capability", "vram_MB")
                summary = "; ".join(f"{key}={props[key]}" for key in fields if props.get(key) is not None)
                props["design_summary"] = summary or str(node["id"])
                props["hardware_id"] = props.get("hardware_id") or str(node["id"])
                props["topics"] = ["runtime"]
            result.append({**props, "record_id": str(node["id"]), "evidence_refs": [f"{path.name}#{node['id']}"]})
        by_id = {node["id"]: node for node in payload["nodes"]}
        for index, edge in enumerate(payload.get("edges") or []):
            if edge.get("type") != "HAS_FEATURE":
                continue
            if edge.get("from") not in by_id or edge.get("to") not in by_id:
                result.append({"record_id": f"edge:{index}", "migration_error": "Relationship references a missing graph node", "source_evidence": edge})
                continue
            hardware = by_id[edge["from"]]["properties"]
            feature = by_id[edge["to"]]["properties"]
            relationship = edge.get("properties") or {}
            result.append({**feature, **relationship,
                           "record_id": f"edge:{index}", "hardware_id": hardware.get("hardware_id") or edge["from"],
                           "avoid_patterns": list(feature.get("avoid_patterns") or []),
                           "evidence_refs": [f"{path.name}#edge:{index}"],
                           "source_evidence": {"hardware": hardware, "feature": feature, "relationship": relationship}})
        return result
    if isinstance(payload, dict):
        return list(payload.get("records") or [payload])
    raise ValueError("Expected knowledge records, a graph bundle, or a lesson database")


def migrate(sources: Iterable[dict[str, Any]], *, domain: str, output: Path, apply: bool = False,
            max_sources: int | None = None) -> dict[str, Any]:
    if domain not in {"hardware", "lesson"}:
        raise ValueError("Knowledge domain must be hardware or lesson")
    sources = list(sources)
    identities = [source_identity(source) for source in sources]
    if len(set(identities)) != len(identities):
        raise ValueError("Duplicate source identities; disambiguate before migration")
    ledger_path = output / "migration.sqlite3"
    previous = {}
    if ledger_path.exists():
        with sqlite3.connect(ledger_path.resolve().as_uri() + "?mode=ro", uri=True) as connection:
            previous = {row[0]: (row[1], row[2], json.loads(row[3]), row[4]) for row in connection.execute(
                "SELECT source_id, source_hash, revision, records_json, status FROM sources WHERE domain=? ORDER BY revision", (domain,)
            )}
    connection = None
    if apply:
        output.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(ledger_path)
        connection.execute("CREATE TABLE IF NOT EXISTS sources (domain TEXT, source_id TEXT, source_hash TEXT, revision INTEGER, source_json TEXT, records_json TEXT, status TEXT, PRIMARY KEY(domain, source_id, source_hash))")
    mapping = []
    all_records = {}
    processed = 0
    try:
        for identity, source in zip(identities, sources):
            source_hash = digest(source)
            prior = previous.get(identity)
            resumed = bool(prior and prior[0] == source_hash)
            if not resumed and max_sources is not None and processed >= max_sources:
                break
            revision = prior[1] if resumed else prior[1] + 1 if prior else 1
            reason = source.get("migration_error")
            try:
                records = [] if reason else prior[2] if resumed else publication_records(source["publication"]) if "publication" in source else from_source(source, domain=domain, source_id=identity, revision=revision)
                for record in records:
                    validate_record(record)
            except (TypeError, ValueError, KeyError) as exc:
                records = []
                reason = f"Invalid concise source: {type(exc).__name__}"
            status = "converted" if records else "awaiting_correction"
            duplicates = []
            for record in records:
                validate_record(record)
                key = digest({k: record[k] for k in ("summary", "applicability", "applies_when", "restrictions", "fallbacks", "verification_status", "strength")})
                if key in all_records:
                    duplicates.append(all_records[key]["record_id"])
                    all_records[key]["evidence_refs"] = sorted(set(all_records[key]["evidence_refs"] + record["evidence_refs"]))
                else:
                    all_records[key] = json.loads(json.dumps(record))
            if records and len(duplicates) == len(records):
                status = "deduplicated"
            if connection is not None and not resumed:
                with connection:
                    connection.execute("INSERT INTO sources VALUES (?, ?, ?, ?, ?, ?, ?)", (
                        domain, identity, source_hash, revision, json.dumps(source, sort_keys=True), json.dumps(records, sort_keys=True), status,
                    ))
            if not resumed:
                processed += 1
            mapping.append({"source_id": identity, "source_hash": source_hash, "revision": revision, "status": status,
                            "record_ids": [record["record_id"] for record in records], "duplicates": duplicates, "resumed": resumed,
                            "reason": (reason or "Source requires a complete concise summary") if status == "awaiting_correction" else None})
    finally:
        if connection is not None:
            connection.close()
    records = sorted(all_records.values(), key=lambda record: record["record_id"])
    report = {"schema_version": SCHEMA_VERSION, "domain": domain, "dry_run": not apply,
              "complete": len(mapping) == len(sources), "source_count": len(sources), "processed_sources": len(mapping),
              "record_count": len(records), "records_hash": digest(records), "mapping": mapping,
              "awaiting_correction": [row["source_id"] for row in mapping if row["status"] == "awaiting_correction"]}
    if apply and report["complete"]:
        export = output / f"{domain}-v2.json"
        temporary = export.with_suffix(".tmp")
        temporary.write_text(json.dumps({"manifest": report, "records": records}, indent=2) + "\n")
        temporary.replace(export)
    return report


def publish_lessons(source: Path, *, vector_store=None) -> dict[str, Any]:
    """Backfill only the derived lesson table/index, preserving frozen revisions."""
    from lesson_profile_database.config import LessonProfileSettings
    from lesson_profile_database.registry import LessonProfileRegistry

    sources = load_sources(source, "lesson")
    registry = LessonProfileRegistry(LessonProfileSettings(sqlite_path=str(source)))
    registry.initialize()
    count = 0
    for item in sources:
        payload = item["publication"]
        records = publication_records(payload)
        with registry.transaction() as connection:
            for record in records:
                connection.execute("INSERT OR IGNORE INTO design_knowledge_v2 VALUES (?, ?, ?, ?)",
                                   (payload["profile_key"], payload["revision_number"], record["record_id"], json.dumps(record, sort_keys=True)))
                count += 1
        if vector_store is not None:
            vector_store.upsert_design_publication({**payload, "design_records_v2": records})
    return {"ok": True, "record_count": count, "source_count": len(sources)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--domain", choices=["hardware", "lesson"], required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--publish", action="store_true")
    parser.add_argument("--settings", type=Path, help="Domain settings YAML for publishing graph/vector indexes")
    parser.add_argument("--max-sources", type=int)
    args = parser.parse_args()
    if args.publish and not args.apply:
        parser.error("--publish requires --apply")
    sources = load_sources(args.source, args.domain)
    report = migrate(sources, domain=args.domain, output=args.output, apply=args.apply, max_sources=args.max_sources)
    if args.publish:
        if not report["complete"]:
            parser.error("Complete the resumable backfill before publishing")
        settings = yaml.safe_load(args.settings.read_text()) if args.settings else {}
        if args.domain == "lesson":
            if args.source.suffix not in {".sqlite", ".sqlite3", ".db"}:
                parser.error("Lesson publication requires the authoritative lesson SQLite database")
            from lesson_profile_database.config import LessonProfileSettings
            from lesson_profile_database.vector_store import LessonVectorStore

            vector = LessonVectorStore(LessonProfileSettings.from_mapping(settings)) if args.settings else None
            report["publication"] = publish_lessons(args.source, vector_store=vector)
        else:
            if args.settings is None:
                parser.error("Hardware publication requires --settings")
            from hardware_knowledge_graph.config import HardwareKnowledgeSettings
            from localml_scheduler.hardware_knowledge.store import HardwareKnowledgeGraphStore
            from localml_scheduler.code_knowledge.store import CodeKnowledgeStore

            config = HardwareKnowledgeSettings.from_dict(settings)
            if "nodes" in (yaml.safe_load(args.source.read_text()) or {}):
                report["publication"] = HardwareKnowledgeGraphStore(config).ingest_schema_root(args.source.parent)
            else:
                report["publication"] = CodeKnowledgeStore(config).ingest_source(args.source)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
