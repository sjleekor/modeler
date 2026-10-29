"""Read saved daily briefing evidence without scoring, prices, or remote access."""

from __future__ import annotations

import math
import re
import subprocess
from datetime import date, datetime
from pathlib import Path

from .core import (
    SOURCE_IDENTITY_VERSION,
    Batch,
    canonical_bytes,
    identity,
    read_json,
    require_timestamp,
    sha256_bytes,
)

ADAPTER_VERSION = "briefing.v1"
CHECKER_VERSION = "briefing-quality.v1"
_HASH = re.compile(r"[0-9a-f]{64}")
_COMMIT = re.compile(r"[0-9a-f]{40}")
_UNIT_FILES = {"README.md", "market-sector.md", "kr-stocks.md", "us-stocks.md", "data-status.md"}
_SIDECARS = (
    "run-state.json",
    "coordinator-inference.json",
    "coordinator-render.json",
    "reports-publisher-config.json",
    "coordinator-publication.json",
    "markdown/manifest.json",
    "remote-verification.json",
    "monitor-0.json",
    "monitor-1.json",
    "monitor-2.json",
    "monitor-3.json",
    "coordinator-explain.json",
)


def _error(batch: Batch, source: str, reason: str) -> None:
    batch.errors.append({"source": source, "reason": reason})


def _hash(value: object) -> bool:
    return isinstance(value, str) and _HASH.fullmatch(value) is not None


def _day(value: object) -> bool:
    try:
        return isinstance(value, str) and date.fromisoformat(value).isoformat() == value
    except ValueError:
        return False


def _decision(value: object, day: str) -> bool:
    try:
        require_timestamp(value)
        return datetime.fromisoformat(value.replace("Z", "+00:00")) == datetime.fromisoformat(
            day + "T10:00:00+09:00"
        )
    except (ValueError, TypeError):
        return False


def _read(path: Path, root: Path, batch: Batch) -> dict:
    try:
        return read_json(path, root=root, batch=batch)
    except (OSError, ValueError) as exc:
        _error(batch, path.relative_to(root).as_posix(), str(exc))
        return {}


def _sidecars(day_dir: Path, batch: Batch) -> dict[str, dict]:
    result = {}
    for name in _SIDECARS:
        path = day_dir / name
        if path.exists() or path.is_symlink():
            result[name] = _read(path, day_dir, batch)
    return result


def _invocation(
    report: dict, sides: dict[str, dict], report_hash: str, day: str, batch: Batch
) -> str | None:
    values = set()
    if report.get("invocation_id") is not None:
        values.add(report["invocation_id"] if isinstance(report["invocation_id"], str) else "")
    bound = False
    bad = False
    for name, item in sides.items():
        if name.startswith("monitor-") or name in {
            "remote-verification.json",
            "coordinator-explain.json",
            "publication-receipt.json",
        }:
            continue
        if item.get("report_date", day) != day:
            _error(batch, name, "report_date_mismatch")
            bad = True
        if "report_sha256" in item and item["report_sha256"] != report_hash:
            _error(batch, name, "report_sha256_mismatch")
            bad = True
        value = item.get("invocation_id")
        if value is not None:
            values.add(value if isinstance(value, str) else "")
            if item.get("report_sha256") == report_hash and name == "coordinator-inference.json":
                bound = True
    if len(values) != 1 or not next(iter(values), ""):
        _error(batch, "report-" + day + ".json", "invocation_missing_or_mismatch")
        return None
    if bad or not bound:
        _error(batch, "coordinator-inference.json", "daily_producer_unverified")
        return None
    return next(iter(values))


def _explain(
    day_dir: Path,
    day: str,
    sides: dict[str, dict],
    report_hash: str,
    invocation: str | None,
    batch: Batch,
) -> tuple[dict, str | None]:
    path = day_dir / f"explain-{day}.json"
    if not path.exists() and not path.is_symlink():
        return {}, None
    item = _read(path, day_dir, batch)
    matches = [source["sha256"] for source in batch.sources if source["path"] == path.name]
    if not item or not matches:
        return {}, None
    digest = matches[-1]
    expected = sides.get("coordinator-explain.json", {}).get("document_sha256")
    manifest_hash = sides.get("markdown/manifest.json", {}).get("explain_sha256")
    errors = []
    if item.get("report_sha256") != report_hash:
        errors.append("explain_report_mismatch")
    if item.get("invocation_id") != invocation or invocation is None:
        errors.append("explain_invocation_mismatch")
    if expected is None or expected != digest:
        errors.append("explain_hash_mismatch")
    if manifest_hash is not None and manifest_hash != digest:
        errors.append("explain_manifest_hash_mismatch")
    if item.get("schema") != "briefing-explain.v1" or not isinstance(item.get("models"), dict):
        errors.append("explain_schema_invalid")
    if item.get("unit") != day:
        errors.append("explain_unit_mismatch")
    releases = {
        side["release"]
        for name, side in sides.items()
        if name in {"reports-publisher-config.json", "markdown/manifest.json"}
        and isinstance(side.get("release"), str)
        and side["release"]
    }
    if (
        len(releases) != 1
        or not isinstance(item.get("release"), str)
        or item.get("release") not in releases
    ):
        errors.append("explain_release_mismatch")
    for reason in errors:
        _error(batch, path.name, reason)
    return ({}, None) if errors else (item, digest)


def _git_unit(
    checkout: Path,
    commit: str,
    day: str,
    report_hash: str,
    release: str,
    expected_hash: str,
    batch: Batch,
) -> tuple[int, str, dict[str, bytes]] | None:
    """Only immutable local git objects are consulted; no working-tree fallback."""
    from modeler.reporting.markdown import parse_front_matter

    unit = f"reports/daily-briefing/{day[:4]}/{day[5:7]}/{day}"
    try:
        tree = subprocess.run(
            ["git", "-C", str(checkout), "ls-tree", "-rlz", commit, "--", unit + "/"],
            check=True,
            capture_output=True,
            timeout=10,
        ).stdout
        entries = []
        for entry in tree.split(b"\0"):
            if not entry:
                continue
            metadata, path = entry.split(b"\t", 1)
            mode, kind, _oid, size = metadata.decode("ascii").split()
            path = path.decode("utf-8")
            if mode != "100644" or kind != "blob" or not path.startswith(unit + "/"):
                raise ValueError("git_unit_nonregular")
            if int(size) > 32 * 1024 * 1024:
                raise ValueError("git_unit_oversize")
            entries.append((path[len(unit) + 1 :], path))
        if {name for name, _ in entries} != _UNIT_FILES:
            raise ValueError("git_unit_files_mismatch")
        files = {}
        revisions = set()
        for name, path in entries:
            raw = subprocess.run(
                ["git", "-C", str(checkout), "show", f"{commit}:{path}"],
                check=True,
                capture_output=True,
                timeout=10,
            ).stdout
            if len(raw) > 32 * 1024 * 1024:
                raise ValueError("git_unit_oversize")
            digest = sha256_bytes(raw)
            batch.artifacts[digest] = raw
            batch.sources.append(
                {
                    "path": f"git:{commit}:{path}",
                    "sha256": digest,
                    "hash_kind": "file_bytes_sha256",
                    "producer": "stock-reports.v1",
                }
            )
            meta, _ = parse_front_matter(raw.decode("utf-8"))
            if (
                not meta
                or not isinstance(meta.get("source"), dict)
                or meta.get("unit") != day
                or meta.get("schema") != "stock-reports.v1"
                or meta.get("source", {}).get("report_sha256") != report_hash
                or meta.get("source", {}).get("release") != release
            ):
                raise ValueError("git_unit_source_mismatch")
            revision = meta.get("revision")
            if isinstance(revision, bool) or not isinstance(revision, int) or revision < 1:
                raise ValueError("git_unit_revision_invalid")
            revisions.add(revision)
            files[name] = raw
        digest = sha256_bytes(
            b"".join(f"{name}\0{sha256_bytes(files[name])}\n".encode() for name in sorted(files))
        )
        if digest != expected_hash:
            raise ValueError("git_unit_hash_mismatch")
        if len(revisions) != 1:
            raise ValueError("git_unit_revision_mismatch")
        return next(iter(revisions)), digest, files
    except (OSError, ValueError, UnicodeError, subprocess.SubprocessError) as exc:
        reason = str(exc) if isinstance(exc, ValueError) else "git_unit_unavailable"
        _error(batch, "markdown/manifest.json", reason)
        return None


def _exposed_predictions(files: dict[str, bytes], batch: Batch) -> set[str]:
    """Bind actual ranking table rows, never infer visibility from top_n or report status."""
    from modeler.reporting.markdown import c, parse_front_matter

    exposed = set()
    for name, market in (("kr-stocks.md", "KR"), ("us-stocks.md", "US")):
        meta, body = parse_front_matter(files[name].decode("utf-8"))
        declared = meta.get("models") if meta else None
        if (
            not isinstance(declared, list)
            or not all(isinstance(model, str) for model in declared)
            or meta.get("section") != name.removesuffix(".md")
        ):
            _error(batch, name, "publication_models_unverified")
            continue
        active_model = None
        table = False
        header_pending = False
        for line in body.splitlines():
            stripped = line.strip()
            if market == "US" and stripped.startswith("## "):
                active_model = None
            if stripped.startswith("- 모델:"):
                links = re.findall(
                    r"reference/models/([A-Za-z0-9._-]+)\.md\)\s+\(`([^`]+)`\)", stripped
                )
                active_model = (
                    links[0][0]
                    if len(links) == 1 and links[0][0] == links[0][1] and links[0][0] in declared
                    else None
                )
                table = False
                header_pending = False
                continue
            if not stripped.startswith("|") or not stripped.endswith("|"):
                table = False
                header_pending = False
                continue
            cells = [cell.strip() for cell in re.split(r"(?<!\\)\|", stripped[1:-1])]
            if cells[:2] == ["순위", "코드"]:
                header_pending = True
                table = False
                continue
            if header_pending:
                header_pending = False
                table = len(cells) >= 2 and all(re.fullmatch(r":?-{3,}:?", cell) for cell in cells)
                continue
            if not table or len(cells) < 2:
                continue
            if active_model is None:
                _error(batch, name, "publication_ranking_model_unverified")
                continue
            if not cells[0].isdigit() or int(cells[0]) < 1:
                _error(batch, name, "publication_ranking_invalid")
                continue
            candidates = [
                row
                for row in batch.predictions
                if row["market"] == market
                and row["model_id"] == active_model
                and row["rank"] == int(cells[0])
                and c(row["symbol"]) == cells[1]
            ]
            if len(candidates) != 1:
                _error(batch, name, "publication_ranking_unmatched_or_ambiguous")
                continue
            exposed.add(candidates[0]["prediction_ref"])
    return exposed


def _publication(
    day_dir: Path,
    day: str,
    report_hash: str,
    invocation: str | None,
    sides: dict[str, dict],
    checkout: Path | None,
    batch: Batch,
    observed_at: str,
) -> None:
    manifest = sides.get("markdown/manifest.json", {})
    published = sides.get("coordinator-publication.json", {})
    config = sides.get("reports-publisher-config.json", {})
    commit = published.get("reports_commit")
    unit_hash = manifest.get("unit_sha256")
    binding = (
        invocation is not None
        and manifest.get("unit") == day
        and manifest.get("report_sha256") == report_hash
        and manifest.get("invocation_id") == invocation
        and published.get("report_sha256") == report_hash
        and published.get("status") == "published"
        and isinstance(manifest.get("release"), str)
        and bool(manifest["release"])
        and ("release" not in config or config["release"] == manifest["release"])
        and isinstance(commit, str)
        and _COMMIT.fullmatch(commit) is not None
        and _hash(unit_hash)
    )
    local = None
    if binding and checkout is not None:
        local = _git_unit(checkout, commit, day, report_hash, manifest["release"], unit_hash, batch)
    if manifest and not binding:
        _error(batch, "markdown/manifest.json", "publication_binding_invalid")
    remote = sides.get("remote-verification.json", {})
    historical = bool(
        local and remote.get("reports_commit") == commit and remote.get("status") == "contained"
    )
    monitors = [item for name, item in sides.items() if name.startswith("monitor-")]
    if historical and monitors:
        historical = any(
            item.get("status") == "verified"
            and item.get("report_date") == day
            and item.get("report_sha256", report_hash) == report_hash
            and isinstance(item.get("axes"), dict)
            and item["axes"].get("remote") == "contained"
            for item in monitors
        )
    exposed = _exposed_predictions(local[2], batch) if local else set()
    record = {
        "prediction_refs": sorted(exposed),
        "report_prediction_refs": [row["prediction_ref"] for row in batch.predictions],
        "status": "published_evidence" if historical else "publication_unverified",
        "report_sha256": report_hash,
        "invocation_id": invocation,
        "unit": day,
        "report_date": day,
        "release": manifest.get("release"),
        "unit_path": f"reports/daily-briefing/{day[:4]}/{day[5:7]}/{day}",
        "revision": local[0] if local else None,
        "unit_sha256": unit_hash if _hash(unit_hash) else None,
        "reports_commit": commit if isinstance(commit, str) and _COMMIT.fullmatch(commit) else None,
        "verified_by": None,
        "verified_at": None,
        "observed_at": observed_at,
        "historical_publication_evidence": historical,
        "unit_binding_verified": local is not None,
        "adapter_version": ADAPTER_VERSION,
        "source_identity_version": SOURCE_IDENTITY_VERSION,
        "evidence_sha256": sorted(batch.artifacts),
    }
    record["publication_ref"] = identity(
        "publication",
        {
            "binding": {
                key: record[key]
                for key in (
                    "report_sha256",
                    "invocation_id",
                    "unit",
                    "revision",
                    "unit_sha256",
                    "reports_commit",
                )
            }
        },
    )
    batch.publications.append(record)
    for prediction in batch.predictions:
        prediction["publication_status"] = (
            record["status"]
            if prediction["prediction_ref"] in exposed
            else "not_published" if local else "generated"
        )


def read_briefing(
    day_dir: Path, *, observed_at: str, historical_link: bool = True, checkout: Path | None = None
) -> Batch:
    """Adapt exactly one explicitly selected daily directory into metadata references."""
    observed_at = require_timestamp(observed_at)
    day_dir = Path(day_dir)
    batch = Batch()
    day = day_dir.name
    if not _day(day):
        _error(batch, ".", "day_directory_invalid")
        return batch
    report_name = f"report-{day}.json"
    report = _read(day_dir / report_name, day_dir, batch)
    if not report:
        return batch
    # A standalone report uses a different bundle hash contract and is never upgraded.
    if report.get("schema_version") != "1.0" or not isinstance(report.get("markets"), list):
        _error(batch, report_name, "daily_envelope_required")
        return batch
    report_hash = next(
        source["sha256"] for source in batch.sources if source["path"] == report_name
    )
    if report.get("report_date") != day or not _decision(report.get("decision_at"), day):
        _error(batch, report_name, "report_date_mismatch")
        return batch
    sides = _sidecars(day_dir, batch)
    invocation = _invocation(report, sides, report_hash, day, batch)
    explain, explain_hash = _explain(day_dir, day, sides, report_hash, invocation, batch)
    seen = set()
    for model_index, model in enumerate(report["markets"]):
        source = f"{report_name}#markets/{model_index}"
        if not isinstance(model, dict):
            _error(batch, source, "model_invalid")
            continue
        rows = model.get("rankings")
        if not isinstance(rows, list):
            _error(batch, source, "rankings_invalid")
            continue
        provenance = model.get("provenance", {})
        market = model.get("market")
        model_ok = (
            invocation is not None
            and isinstance(market, str)
            and market in {"KR", "US"}
            and model.get("report_date") == day
            and _day(model.get("feature_asof_date"))
            and model["feature_asof_date"] < day
            and _decision(model.get("decision_at"), day)
            and all(
                isinstance(model.get(key), str) and model[key]
                for key in ("model_id", "model_version")
            )
            and isinstance(provenance, dict)
            and all(
                _hash(provenance.get(key))
                for key in (
                    "bundle_manifest_sha256",
                    "input_sha256",
                    "code_inventory_sha256",
                    "native_prepare_manifest_sha256",
                )
            )
        )
        if not model_ok:
            _error(batch, source, "source_incomplete")
            for row_index in range(len(rows)):
                _error(batch, f"{source}/rankings/{row_index}", "source_incomplete")
            continue
        explanation_model = explain.get("models", {}).get(model["model_id"], {}) if model_ok else {}
        explanation_rows = (
            explanation_model.get("rows", []) if isinstance(explanation_model, dict) else []
        )
        if explain_hash and explanation_model and model_ok:
            inputs = explanation_model.get("inputs")
            if (
                not isinstance(inputs, dict)
                or inputs.get("prepared_sha256") != provenance.get("input_sha256")
                or inputs.get("bundle_sha256") != provenance.get("bundle_manifest_sha256")
            ):
                _error(batch, source, "explain_model_inputs_mismatch")
                explanation_rows = []
        for row_index, row in enumerate(rows):
            row_source = f"{source}/rankings/{row_index}"
            if not model_ok or not isinstance(row, dict):
                _error(batch, row_source, "source_incomplete")
                continue
            security_market = row.get("market") if market == "KR" else row.get("market", "US")
            symbol, rank, score = row.get("symbol"), row.get("rank"), row.get("score")
            if (
                not isinstance(symbol, str)
                or not symbol
                or not isinstance(security_market, str)
                or not security_market
                or (market == "KR" and security_market not in {"KOSPI", "KOSDAQ"})
                or isinstance(rank, bool)
                or not isinstance(rank, int)
                or rank < 1
                or isinstance(score, bool)
                or not isinstance(score, (int, float))
                or not math.isfinite(score)
            ):
                _error(batch, row_source, "ranking_invalid")
                continue
            fields = {
                "market": market,
                "security_market": security_market,
                "model_id": model["model_id"],
                "bundle_hash": provenance["bundle_manifest_sha256"],
                "report_sha256": report_hash,
                "invocation_id": invocation,
                "symbol": symbol,
            }
            ref = identity("prediction", fields)
            if ref in seen:
                _error(batch, row_source, "duplicate_prediction")
                continue
            seen.add(ref)
            reasons = row.get("quality_reasons", []) if rank <= 100 else []
            quality_valid = True
            if not isinstance(reasons, list) or not all(
                isinstance(reason, str) and reason for reason in reasons
            ):
                _error(batch, row_source, "quality_reasons_invalid")
                reasons = []
                quality_valid = False
            try:
                evidence_raw = canonical_bytes(
                    {
                        "report_sha256": report_hash,
                        "model_index": model_index,
                        "row_index": row_index,
                        "row": row,
                    }
                )
            except (ValueError, TypeError):
                _error(batch, row_source, "ranking_evidence_invalid")
                continue
            evidence = sha256_bytes(evidence_raw)
            batch.artifacts[evidence] = evidence_raw
            prediction = {
                **fields,
                "prediction_ref": ref,
                "kind": "briefing",
                "model_version": model["model_version"],
                "rank": rank,
                "score": score,
                "report_date": day,
                "feature_asof_date": model["feature_asof_date"],
                "decision_at": model.get("decision_at"),
                "observed_at": observed_at,
                "historical_link": historical_link,
                "quality_reasons": reasons,
                "quality_review": row.get("quality_review") if rank <= 100 else "not_checked",
                "quality_validation": (
                    "not_checked" if rank > 100 else "valid" if quality_valid else "source_invalid"
                ),
                "checker_version": CHECKER_VERSION,
                "source_evidence_sha256": evidence,
                "evaluation_status": (
                    "sealed"
                    if market == "US"
                    else "pending" if model["model_id"] == "kr_daily_h20_v1" else "not_defined"
                ),
                "source_identity_version": SOURCE_IDENTITY_VERSION,
                "adapter_version": ADAPTER_VERSION,
                "provenance": provenance,
                "bundle_hash_kind": "file_bytes_sha256",
                "explain_sha256": None,
            }
            if explain_hash and isinstance(explanation_rows, list):
                candidates = [
                    item
                    for item in explanation_rows
                    if isinstance(item, dict) and item.get("symbol") == symbol
                ]
                symbol_unique = (
                    sum(isinstance(item, dict) and item.get("symbol") == symbol for item in rows)
                    == 1
                )
                if candidates and (not symbol_unique or len(candidates) != 1):
                    _error(batch, row_source, "explain_symbol_ambiguous")
                elif candidates:
                    candidate = candidates[0]
                    if candidate.get("rank") != rank or candidate.get("score") != score:
                        _error(batch, row_source, "explain_row_mismatch")
                    else:
                        prediction["explain_sha256"] = explain_hash
            batch.predictions.append(prediction)
            for reason in reasons:
                batch.findings.append(
                    {
                        "prediction_ref": ref,
                        "reason_code": reason,
                        "checker_version": CHECKER_VERSION,
                        "source_evidence_sha256": evidence,
                        "observed_at": observed_at,
                    }
                )
    _publication(day_dir, day, report_hash, invocation, sides, checkout, batch, observed_at)
    return batch
