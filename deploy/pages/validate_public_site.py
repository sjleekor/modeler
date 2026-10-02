#!/usr/bin/env python3
"""Validate a SiteBuilder public projection before Pages delivery."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from datetime import date
from pathlib import Path, PurePosixPath
from typing import Any
from html.parser import HTMLParser

DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
REVISION = re.compile(r"^[0-9a-f]{16}$")
MODEL_ID = re.compile(r"^[A-Za-z0-9._-]+$")
MAX_FILE_BYTES = 5 * 1024 * 1024
MAX_FILES = 20_000


def _fail(message: str) -> None:
    raise ValueError(message)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"),
                           parse_constant=lambda token: _fail(f"non-finite JSON number: {token}"))
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        _fail(f"invalid JSON in {path.name}: {exc}")
    if not isinstance(value, dict):
        _fail(f"JSON root must be an object: {path.name}")
    return value


def _date(value: Any, field: str) -> str:
    if not isinstance(value, str) or not DATE.fullmatch(value):
        _fail(f"{field} must use YYYY-MM-DD")
    try:
        if date.fromisoformat(value).isoformat() != value:
            _fail(f"{field} is not a canonical date")
    except ValueError:
        _fail(f"{field} is not a calendar date")
    return value


def _safe_tree(root: Path) -> dict[str, Path]:
    if root.is_symlink() or not root.is_dir():
        _fail("site root must be a real directory")
    files: dict[str, Path] = {}
    for current, dirs, names in os.walk(root, followlinks=False):
        current_path = Path(current)
        for name in dirs:
            if (current_path / name).is_symlink():
                _fail("symlink directory is forbidden")
        for name in names:
            path = current_path / name
            if path.is_symlink() or not path.is_file():
                _fail("symlink or non-regular file is forbidden")
            rel = path.relative_to(root).as_posix()
            pure = PurePosixPath(rel)
            if pure.is_absolute() or any(part in {"", ".", ".."} for part in pure.parts):
                _fail("unsafe site path")
            if path.stat().st_size > MAX_FILE_BYTES:
                _fail(f"site file exceeds {MAX_FILE_BYTES} bytes: {rel}")
            files[rel] = path
            if len(files) > MAX_FILES:
                _fail("site contains too many files")
    return files


def _allowed_path(path: str) -> bool:
    if path in {"index.html", "archive/index.html", "models/index.html", "site-manifest.json"}:
        return True
    if re.fullmatch(r"archive/\d{4}-\d{2}/index\.html", path):
        return True
    if re.fullmatch(r"reports/\d{4}-\d{2}-\d{2}/(?:index\.html|report\.json)", path):
        return True
    if re.fullmatch(r"reports/\d{4}-\d{2}-\d{2}/revisions/[0-9a-f]{16}/(?:index\.html|report\.json)", path):
        return True
    if re.fullmatch(r"models/[A-Za-z0-9._-]+/[0-9a-f]{16}/(?:index\.html|model-card\.json)", path):
        return True
    if re.fullmatch(r"assets/site-[0-9a-f]{16}\.css", path):
        return True
    return False


def _exact_keys(value: dict[str, Any], keys: set[str], label: str) -> None:
    if set(value) != keys:
        _fail(f"{label} fields differ from the public SiteBuilder schema")


def _validate_market(row: Any) -> None:
    if not isinstance(row, dict):
        _fail("market projection must be an object")
    keys = {
        "schema_version", "market", "report_date", "decision_at", "feature_asof_date",
        "status", "model_id", "model_version", "inference_started_at", "publication",
        "rankings", "quality", "provenance",
    }
    _exact_keys(row, keys, "market projection")
    if row["market"] not in {"KR", "US"}:
        _fail("invalid public market")
    if row["schema_version"] != "1.0":
        _fail("invalid market schema version")
    if row["status"] not in {"ok", "partial", "stale", "withheld", "failed", "unavailable"}:
        _fail("invalid public market status")
    if row["model_id"] is not None and (not isinstance(row["model_id"], str) or not MODEL_ID.fullmatch(row["model_id"])):
        _fail("invalid public model id")
    if row["model_version"] is not None and not isinstance(row["model_version"], str):
        _fail("invalid public model version")
    if row["decision_at"] is not None and not isinstance(row["decision_at"], str):
        _fail("invalid decision_at")
    if row["inference_started_at"] is not None and not isinstance(row["inference_started_at"], str):
        _fail("invalid inference_started_at")
    _date(row["report_date"], "market.report_date")
    if row["feature_asof_date"] is not None:
        _date(row["feature_asof_date"], "market.feature_asof_date")
    publication = row["publication"]
    if not isinstance(publication, dict):
        _fail("market publication must be an object")
    _exact_keys(publication, {"status", "evidence"}, "market publication")
    if publication["status"] not in {"allowed", "withheld", "unresolved"}:
        _fail("invalid publication state")
    if not isinstance(publication["evidence"], list):
        _fail("public evidence must be a list")
    for item in publication["evidence"]:
        if not isinstance(item, dict) or set(item) != {"id"} or not isinstance(item["id"], str):
            _fail("public evidence can contain only safe evidence ids")
    rankings = row["rankings"]
    if not isinstance(rankings, list):
        _fail("rankings must be a list")
    if len(rankings) > 100:
        _fail("public projection may expose at most 100 rankings")
    for item in rankings:
        if not isinstance(item, dict):
            _fail("public ranking must be an object")
        _exact_keys(item, {"rank", "symbol", "name"}, "public ranking")
        if isinstance(item["rank"], bool) or not isinstance(item["rank"], int) or item["rank"] <= 0:
            _fail("public ranking has an invalid rank")
        if not all(isinstance(item[key], str) for key in ("symbol", "name")):
            _fail("public ranking symbol/name must be strings")
    if [item["rank"] for item in rankings] != list(range(1, len(rankings) + 1)):
        _fail("public ranking ranks must be contiguous and ordered")
    if not isinstance(row["quality"], dict) or not isinstance(row["provenance"], dict):
        _fail("public quality/provenance must be objects")
    if row["quality"] not in ({}, {"status": row["status"]}):
        _fail("quality contains fields outside public projection")
    if row["provenance"] not in ({}, {"model_id": row["model_id"], "model_version": row["model_version"]}):
        _fail("provenance contains fields outside public projection")
    if publication["status"] != "allowed" and rankings:
        _fail("rankings are present without an allowed publication decision")


def _validate_report_payload(payload: dict[str, Any], expected_date: str, allow_synthetic: bool) -> None:
    keys = {
        "schema_version", "report_date", "decision_at", "status", "markets", "opening",
        "model_cards", "synthetic_fixture", "historical_replay", "inference_started_at",
    }
    _exact_keys(payload, keys, "report payload")
    if payload["schema_version"] != "1.0" or payload["report_date"] != expected_date:
        _fail("report payload schema/date mismatch")
    if not isinstance(payload["synthetic_fixture"], bool) or not isinstance(payload["historical_replay"], bool):
        _fail("report fixture/replay markers must be booleans")
    if payload["synthetic_fixture"] and not allow_synthetic:
        _fail("synthetic fixture validation requires explicit opt-in")
    if not isinstance(payload["markets"], list):
        _fail("report markets must be a list")
    for market in payload["markets"]:
        _validate_market(market)
        if market["report_date"] != expected_date:
            _fail("market report date differs from containing report")
    opening = payload["opening"]
    if not isinstance(opening, dict) or not {"status", "publication"} <= set(opening) <= {
        "status", "publication", "observed_at", "session_state", "indices", "industries"
    }:
        _fail("opening contains fields outside public projection")
    if opening["status"] not in {"ok", "partial", "stale", "failed", "unavailable"}:
        _fail("invalid opening status")
    if opening["publication"] not in {"allowed", "withheld", "unresolved"}:
        _fail("invalid opening publication state")
    if opening["publication"] != "allowed" and any(key in opening for key in ("observed_at", "session_state", "indices", "industries")):
        _fail("opening observations are present without an allowed publication decision")
    if "observed_at" in opening:
        if not isinstance(opening["observed_at"], str):
            _fail("invalid opening observed_at")
    if opening.get("session_state") not in {None, "regular", "delayed_open", "closed", "halted", "pending_open", "unavailable"}:
        _fail("invalid opening session_state")
    for field in ("indices", "industries"):
        rows = opening.get(field, [])
        if not isinstance(rows, list):
            _fail(f"opening.{field} must be a list")
        for item in rows:
            allowed = {"code", "name", "observed_at", "open", "previous_close", "last", "change_pct"}
            if field == "industries":
                allowed.add("rank")
            if not isinstance(item, dict) or not set(item) <= allowed:
                _fail(f"opening.{field} contains fields outside public projection")
            if any(key in item and not isinstance(item[key], str) for key in ("code", "name", "observed_at")):
                _fail(f"opening.{field} text fields must be strings")
            for key in {"open", "previous_close", "last", "change_pct"} & set(item):
                if isinstance(item[key], bool) or not isinstance(item[key], (int, float)):
                    _fail(f"opening.{field} numeric fields must be numbers")
                if not math.isfinite(item[key]):
                    _fail(f"opening.{field} numeric fields must be finite")
            if "rank" in item and (isinstance(item["rank"], bool) or not isinstance(item["rank"], int) or item["rank"] <= 0):
                _fail("opening industry rank must be positive")
    if not isinstance(payload["model_cards"], dict):
        _fail("model_cards must be an object")
    for model_id, revision in payload["model_cards"].items():
        if not MODEL_ID.fullmatch(model_id) or not isinstance(revision, str) or not REVISION.fullmatch(revision):
            _fail("invalid model card reference")


def _validate_model_card(card: dict[str, Any]) -> None:
    allowed = {"model_id", "publication", "title", "summary", "target", "scope", "limitations"}
    if not set(card) <= allowed or not {"model_id", "publication"} <= set(card):
        _fail("model card contains fields outside public projection")
    if not isinstance(card["model_id"], str) or not MODEL_ID.fullmatch(card["model_id"]):
        _fail("invalid public model card id")
    publication = card["publication"]
    if not isinstance(publication, dict) or set(publication) != {"status", "evidence"}:
        _fail("invalid public model card publication")
    if publication["status"] not in {"allowed", "withheld", "unresolved"} or publication["evidence"] != []:
        _fail("invalid public model card gate")
    if publication["status"] != "allowed" and set(card) != {"model_id", "publication"}:
        _fail("unapproved model card text is present")
    if any(not isinstance(card[key], str) for key in set(card) - {"model_id", "publication"}):
        _fail("model card text must be strings")


def _validate_card_references(payload: dict[str, Any], files: dict[str, Path]) -> None:
    for model_id, revision in payload["model_cards"].items():
        prefix = f"models/{model_id}/{revision}"
        if f"{prefix}/index.html" not in files or f"{prefix}/model-card.json" not in files:
            _fail("report references a missing public model card")


class _Links(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.values: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self.values.extend(value for key, value in attrs if key in {"href", "src"} and value is not None)


def validate_site(root: Path, compare_to: Path | None = None, *,
                  base_path: str = "/market-briefing/", allow_synthetic: bool = False) -> dict[str, Any]:
    if (not isinstance(base_path, str) or not re.fullmatch(r"/[A-Za-z0-9._/-]+/", base_path)
            or any(part in {".", ".."} for part in PurePosixPath(base_path).parts)):
        _fail("base_path must be an explicit absolute Pages project path")
    files = _safe_tree(root)
    if not files or any(not _allowed_path(name) for name in files):
        unexpected = sorted(name for name in files if not _allowed_path(name))
        _fail(f"site contains files outside the public allowlist: {unexpected[:5]}")
    required = {"index.html", "archive/index.html", "models/index.html", "site-manifest.json"}
    if not required <= set(files):
        _fail(f"site is missing required files: {sorted(required - set(files))}")

    manifest = _read_json(files["site-manifest.json"])
    _exact_keys(manifest, {"schema_version", "latest_report_date", "reports", "synthetic_fixture"}, "site manifest")
    if manifest["schema_version"] != "1.0" or not isinstance(manifest["synthetic_fixture"], bool):
        _fail("invalid schema or synthetic site marker")
    if manifest["synthetic_fixture"] and not allow_synthetic:
        _fail("synthetic fixture validation requires explicit opt-in")
    rows = manifest["reports"]
    if not isinstance(rows, list) or not rows:
        _fail("site manifest must contain at least one report date")
    dates = []
    for row in rows:
        if not isinstance(row, dict):
            _fail("site report record must be an object")
        _exact_keys(row, {"report_date", "status", "markets", "revision", "url", "synthetic_fixture", "historical_replay"}, "site report record")
        day = _date(row["report_date"], "site manifest report_date")
        dates.append(day)
        if not isinstance(row["synthetic_fixture"], bool) or not isinstance(row["historical_replay"], bool) or not REVISION.fullmatch(str(row["revision"])):
            _fail("invalid report revision or synthetic marker")
        if row["synthetic_fixture"] and not allow_synthetic:
            _fail("synthetic fixture report requires explicit opt-in")
        if row["status"] not in {"ok", "partial", "stale", "withheld", "failed", "unavailable"}:
            _fail("invalid site report status")
        if not isinstance(row["markets"], list):
            _fail("site report markets must be a list")
        for market in row["markets"]:
            if not isinstance(market, dict):
                _fail("site report market row must be an object")
            _exact_keys(market, {"market", "model_id", "status", "publication"}, "site report market row")
            if market["market"] not in {"KR", "US"} or market["status"] not in {"ok", "partial", "stale", "withheld", "failed", "unavailable"}:
                _fail("invalid site report market record")
            if market["publication"] not in {"allowed", "withheld", "unresolved"}:
                _fail("invalid site report publication state")
        expected_url = f"{base_path}reports/{day}/revisions/{row['revision']}/"
        if row["url"] != expected_url:
            _fail("site report URL differs from the configured project Pages base path")
        expected_json = f"reports/{day}/report.json"
        immutable_json = f"reports/{day}/revisions/{row['revision']}/report.json"
        if expected_json not in files or immutable_json not in files:
            _fail(f"site manifest references a missing report for {day}")
        _validate_report_payload(_read_json(files[expected_json]), day, allow_synthetic)
        _validate_report_payload(_read_json(files[immutable_json]), day, allow_synthetic)
        if _read_json(files[expected_json])["synthetic_fixture"] != row["synthetic_fixture"]:
            _fail("manifest synthetic marker differs from current report payload")
        if f"reports/{day}/index.html" not in files or f"reports/{day}/revisions/{row['revision']}/index.html" not in files:
            _fail(f"site manifest references a missing report page for {day}")
        if files[expected_json].read_bytes() != files[immutable_json].read_bytes():
            _fail("mutable report JSON must match the manifest's current revision")
        current_html = files[f"reports/{day}/index.html"]
        immutable_html = files[f"reports/{day}/revisions/{row['revision']}/index.html"]
        if current_html.read_bytes() != immutable_html.read_bytes():
            _fail("mutable report page must match the manifest's current revision")
        _validate_card_references(_read_json(files[expected_json]), files)

    # Every date must have a manifest row. Older immutable revisions may remain
    # alongside the current revision so publishing never rewrites report history.
    manifest_dates = set(dates)
    for row in rows:
        day, revision = row["report_date"], row["revision"]
        referenced = {f"reports/{day}/index.html", f"reports/{day}/report.json",
                      f"reports/{day}/revisions/{revision}/index.html",
                      f"reports/{day}/revisions/{revision}/report.json"}
        if not referenced <= set(files):
            _fail(f"site is missing the current report files for {day}")
    for name in files:
        if name.startswith("reports/"):
            parts = PurePosixPath(name).parts
            if len(parts) < 3 or parts[1] not in manifest_dates:
                _fail("site contains report files without a manifest date")
            if len(parts) == 5 and parts[2] == "revisions":
                if not REVISION.fullmatch(parts[3]):
                    _fail("invalid historical revision path")
                if parts[4] == "report.json":
                    revision_payload = _read_json(files[name])
                    _validate_report_payload(revision_payload, parts[1], allow_synthetic)
                    _validate_card_references(revision_payload, files)
    if dates != sorted(set(dates)):
        _fail("site report dates must be unique and sorted")
    if manifest["latest_report_date"] != dates[-1]:
        _fail("latest_report_date must be the newest included date")

    for name, path in files.items():
        if name.endswith(".json") and name not in {"site-manifest.json"} and "/report.json" not in name:
            if "/model-card.json" in name:
                _validate_model_card(_read_json(path))
        if name.endswith((".html", ".css")):
            body = path.read_text(encoding="utf-8")
            is_fixture_page = manifest["synthetic_fixture"]
            match = re.match(r"reports/(\d{4}-\d{2}-\d{2})/", name)
            if match:
                current_record = next((record for record in rows if record["report_date"] == match.group(1)), None)
                is_fixture_page = bool(current_record and current_record["synthetic_fixture"])
                revision_match = re.match(r"reports/\d{4}-\d{2}-\d{2}/revisions/([0-9a-f]{16})/", name)
                if revision_match:
                    revision_json = files.get(name.rsplit("/", 1)[0] + "/report.json")
                    if revision_json:
                        is_fixture_page = _read_json(revision_json).get("synthetic_fixture") is True
            if not is_fixture_page and ("synthetic fixture" in body.lower() or "합성 fixture" in body):
                _fail("synthetic fixture text cannot be published")
            if name.endswith(".html"):
                links = _Links()
                links.feed(body)
                for value in links.values:
                    if value.startswith("#"):
                        continue
                    if not value.startswith(base_path):
                        _fail(f"HTML contains an external or out-of-project link: {name}")
                    target = value[len(base_path):].split("#", 1)[0].split("?", 1)[0]
                    if not target:
                        target = "index.html"
                    elif target.endswith("/"):
                        target += "index.html"
                    if target not in files:
                        _fail(f"HTML link points to a missing site file: {name}")

    if compare_to is not None and compare_to.exists():
        old_files = _safe_tree(compare_to)
        for name, old_path in old_files.items():
            if (re.fullmatch(r"reports/\d{4}-\d{2}-\d{2}/revisions/[0-9a-f]{16}/(?:index\.html|report\.json)", name)
                    or re.fullmatch(r"assets/site-[0-9a-f]{16}\.css", name)
                    or re.fullmatch(r"models/[A-Za-z0-9._-]+/[0-9a-f]{16}/(?:index\.html|model-card\.json)", name)):
                new_path = files.get(name)
                if new_path is None or old_path.read_bytes() != new_path.read_bytes():
                    _fail(f"immutable revision changed or disappeared: {name}")
        old_manifest_path = old_files.get("site-manifest.json")
        if old_manifest_path:
            old_manifest = _read_json(old_manifest_path)
            old_dates = {_date(row.get("report_date"), "prior report_date") for row in old_manifest.get("reports", [])}
            if not old_dates <= set(dates):
                _fail("new site manifest removes previously published report dates")
    return {"files": len(files), "latest_report_date": dates[-1],
            "synthetic_fixture": bool(manifest["synthetic_fixture"])}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--compare-to", type=Path)
    parser.add_argument("--base-path", required=True)
    parser.add_argument("--allow-synthetic", action="store_true")
    args = parser.parse_args()
    try:
        result = validate_site(args.root, args.compare_to, base_path=args.base_path,
                               allow_synthetic=args.allow_synthetic)
    except (ValueError, OSError) as exc:
        print(f"public site rejected: {exc}", file=sys.stderr)
        return 1
    print(f"public site accepted: files={result['files']} latest={result['latest_report_date']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
