from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from modeler.serving import release_build as rb
from modeler.serving.daily_inputs import _release_jobs

CSV = rb.DATA_FILES[0]


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _write(path: Path, data: bytes) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return _sha(data)


def _bundle(root: Path, kind: str, version: str) -> Path:
    directory = root / kind
    if kind == "kr":
        files = {"golden.json": _write(directory / "golden.json", b"{}"),
                 "model.joblib": _write(directory / "model.joblib", b"kr-weights")}
        manifest = {"market": "KR", "model_version": version, "files": files}
    else:
        manifest = {"variant": kind, "model_version": version,
                    "model_sha256": _write(directory / "model.joblib", kind.encode() + b"-w"),
                    "golden_features_sha256": _write(directory / "golden_features.parquet", b"gf"),
                    "golden_predictions_sha256": _write(directory / "golden_predictions.parquet",
                                                        kind.encode() + b"-gp")}
    _write(directory / "manifest.json", json.dumps(manifest).encode())
    return directory


@pytest.fixture()
def env(tmp_path: Path) -> dict:
    modeler, collector = tmp_path / "modeler", tmp_path / "collector"
    listing = {}

    def add(repo: str, root: Path, rel: str, data: bytes) -> None:
        listing[f"{repo}/{rel}"] = _write(root / rel, data)

    add("modeler", modeler, "src/modeler/__init__.py", b"")
    add("modeler", modeler, "src/modeler/serving/__init__.py", b"")
    add("modeler", modeler, "src/modeler/serving/adapters.py", b"# adapters\n")
    add("collector", collector, "src/collector/__init__.py", b"")
    add("collector", collector, CSV, b"date\n2026-01-01\n")
    listing["modeler/tests/test_x.py"] = _sha(b"ignored")  # listed, never copied
    manifest = tmp_path / "SHA256SUMS"

    def save() -> None:
        manifest.write_text("".join(f"{d}  {n}\n" for n, d in sorted(listing.items())))

    add("modeler", modeler, "deploy/prod/model-cards.json", b"[]")
    add("modeler", modeler, "deploy/prod/runtime.json", b"{}")
    add("modeler", modeler, "uv.lock", b"lock")
    save()
    return {"tmp": tmp_path, "modeler": modeler, "collector": collector, "listing": listing,
            "save": save, "manifest": manifest,
            "kwargs": dict(
                modeler_root=modeler, collector_root=collector,
                kr_bundle=_bundle(tmp_path / "b", "kr", "1.0.0"),
                us_lightgbm_bundle=_bundle(tmp_path / "b", "lightgbm", "1"),
                us_ridge_bundle=_bundle(tmp_path / "b", "ridge", "1"),
                model_cards=modeler / "deploy/prod/model-cards.json",
                runtime_manifest=modeler / "deploy/prod/runtime.json",
                uv_lock=modeler / "uv.lock", source_manifest=manifest,
                output=tmp_path / "out" / "release")}


def _build(env: dict, **override):
    (env["tmp"] / "out").mkdir(exist_ok=True)
    return rb.build_release(**{**env["kwargs"], **override})


def test_build_passes_release_jobs_and_records_data_files(env):
    summary = _build(env)
    release_path = env["tmp"] / "out" / "release" / "release.json"
    jobs, fixture = _release_jobs(release_path)
    assert len(jobs) == 3 and fixture is False
    release = json.loads(release_path.read_text())
    assert release["frozen"] is True and release["synthetic_fixture"] is False
    assert release["release_root"] == str(release_path.parent.resolve())
    assert release["source_manifest_sha256"] == _sha(env["manifest"].read_bytes())
    assert release["data_files"] == [{"path": CSV, "sha256": _sha(b"date\n2026-01-01\n")}]
    assert (release_path.parent / CSV).read_bytes() == b"date\n2026-01-01\n"
    assert summary["python_file_count"] == 4
    assert not (release_path.parent / "src/modeler/tests").exists()
    assert (release_path.parent / "model-cards.json").read_bytes() == b"[]"
    assert not [p for p in (env["tmp"] / "out").iterdir() if p.name.startswith(".")]


def test_synthetic_flag(env):
    _build(env, synthetic_fixture=True)
    assert _release_jobs(env["tmp"] / "out" / "release" / "release.json")[1] is True


def test_file_not_in_manifest_fails(env):
    _write(env["modeler"] / "src/modeler/extra.py", b"x = 1\n")
    with pytest.raises(rb.BuildError, match="not in the reviewed"):
        _build(env)
    assert not (env["tmp"] / "out" / "release").exists()
    assert list((env["tmp"] / "out").iterdir()) == []


def test_hash_mismatch_fails(env):
    (env["modeler"] / "src/modeler/serving/adapters.py").write_bytes(b"# tampered\n")
    with pytest.raises(rb.BuildError, match="differs from the reviewed"):
        _build(env)


def test_data_file_hash_mismatch_fails(env):
    (env["collector"] / CSV).write_bytes(b"other")
    with pytest.raises(rb.BuildError, match="differs from the reviewed"):
        _build(env)


def test_apple_double_and_pycache_ignored(env):
    _write(env["modeler"] / "src/modeler/serving/._adapters.py", b"\x00\x05junk")
    _write(env["modeler"] / "src/modeler/__pycache__/x.py", b"junk")
    _build(env)
    root = env["tmp"] / "out" / "release"
    assert not list(root.rglob("._*")) and not list(root.rglob("__pycache__"))
    _release_jobs(root / "release.json")


def test_symlink_fails(env):
    link = env["modeler"] / "src/modeler/serving/link.py"
    link.symlink_to(env["modeler"] / "src/modeler/serving/adapters.py")
    with pytest.raises(rb.BuildError, match="symlink"):
        _build(env)


def test_symlink_directory_fails(env):
    (env["modeler"] / "src/modeler/sub").symlink_to(env["modeler"] / "src/modeler/serving")
    with pytest.raises(rb.BuildError, match="symlink"):
        _build(env)


def test_existing_output_fails(env):
    (env["tmp"] / "out" / "release").mkdir(parents=True)
    with pytest.raises(rb.BuildError, match="already exists"):
        _build(env)


def test_bundle_hash_mismatch_fails(env):
    (env["kwargs"]["kr_bundle"] / "model.joblib").write_bytes(b"corrupt")
    with pytest.raises(rb.BuildError, match="differs before copy"):
        _build(env)
    assert not (env["tmp"] / "out" / "release").exists()


def test_us_golden_hash_mismatch_fails(env):
    (env["kwargs"]["us_ridge_bundle"] / "golden_predictions.parquet").write_bytes(b"corrupt")
    with pytest.raises(rb.BuildError, match="differs before copy"):
        _build(env)


def test_model_version_mismatch_fails(env):
    directory = _bundle(env["tmp"] / "other", "lightgbm", "2")
    with pytest.raises(rb.BuildError, match="model_version"):
        _build(env, us_lightgbm_bundle=directory)


def test_post_copy_tamper_is_caught_by_release_jobs(env):
    _build(env)
    root = env["tmp"] / "out" / "release"
    (root / "src/modeler/serving/adapters.py").write_bytes(b"# changed\n")
    with pytest.raises(ValueError):
        _release_jobs(root / "release.json")


def test_cli_main_reports_failure(env, capsys):
    argv = ["--modeler-root", str(env["modeler"]), "--collector-root", str(env["collector"]),
            "--kr-bundle", str(env["kwargs"]["kr_bundle"]),
            "--us-lightgbm-bundle", str(env["kwargs"]["us_lightgbm_bundle"]),
            "--us-ridge-bundle", str(env["kwargs"]["us_ridge_bundle"]),
            "--model-cards", str(env["modeler"] / "deploy/prod/model-cards.json"),
            "--runtime-manifest", str(env["modeler"] / "deploy/prod/runtime.json"),
            "--uv-lock", str(env["modeler"] / "uv.lock"),
            "--source-manifest", str(env["manifest"]),
            "--output", str(env["tmp"] / "out" / "release")]
    (env["tmp"] / "out").mkdir()
    assert rb.main(argv) == 0
    assert json.loads(capsys.readouterr().out)["code_sha256"]
    assert rb.main(argv) == 1


@pytest.mark.parametrize("field,rel", [("model_cards", "deploy/prod/model-cards.json"),
                                       ("runtime_manifest", "deploy/prod/runtime.json"),
                                       ("uv_lock", "uv.lock")])
def test_side_files_must_match_manifest(env, field, rel):
    (env["modeler"] / rel).write_bytes(b"changed")
    with pytest.raises(rb.BuildError, match="differs from the reviewed"):
        _build(env)
    env["listing"].pop(f"modeler/{rel}")
    env["save"]()
    (env["modeler"] / rel).write_bytes(b"changed")
    with pytest.raises(rb.BuildError, match="not in the reviewed"):
        _build(env)


def test_side_file_outside_modeler_root_fails(env):
    outside = _write(env["tmp"] / "elsewhere.json", b"[]")
    with pytest.raises(rb.BuildError, match="not under modeler_root"):
        _build(env, model_cards=env["tmp"] / "elsewhere.json")
    assert outside


def _mutate_release(env, mutate):
    _build(env)
    path = env["tmp"] / "out" / "release" / "release.json"
    release = json.loads(path.read_text())
    mutate(release, path.parent)
    path.write_text(json.dumps(release))
    return path


def test_data_files_tamper_fails(env):
    _build(env)
    root = env["tmp"] / "out" / "release"
    (root / CSV).write_bytes(b"tampered")
    with pytest.raises(ValueError, match="SHA-256"):
        _release_jobs(root / "release.json")


@pytest.mark.parametrize("bad", ["../escape.csv", "/abs.csv", "missing.csv"])
def test_data_files_bad_path_fails(env, bad):
    def mutate(release, root):
        release["data_files"][0]["path"] = bad
    with pytest.raises(ValueError):
        _release_jobs(_mutate_release(env, mutate))


def test_data_files_duplicate_and_symlink_fail(env):
    def dup(release, root):
        release["data_files"].append(dict(release["data_files"][0]))
    with pytest.raises(ValueError, match="duplicated"):
        _release_jobs(_mutate_release(env, dup))


def test_data_files_symlink_fails(env):
    _build(env)
    root = env["tmp"] / "out" / "release"
    real = root / CSV
    data = real.read_bytes()
    real.unlink()
    (root / "elsewhere.csv").write_bytes(data)
    real.symlink_to(root / "elsewhere.csv")
    with pytest.raises(ValueError, match="symlink"):
        _release_jobs(root / "release.json")


def test_release_without_data_files_still_passes(env):
    def mutate(release, root):
        release.pop("data_files")
    assert _release_jobs(_mutate_release(env, mutate))[1] is False
