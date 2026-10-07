"""T1-5 DSR·PBO 계측용 데이터셋 6종 빌드 — ``run_matrix.ensure_dataset`` 만 부른다.

만드는 것(준비 문서 ``my/milestones/kr/modeling/assessment/20261005_dsr_pbo.md`` §1.2):
FS0 rank · FS0 tree · FS1 · FS2 · FS1h native_t · FS1h+FS3(``_e5`` 루트).

* ``force=True`` 는 주지 않는다. 같은 폴더가 이미 full 이면 ``ensure_dataset`` 이 재사용한다.
* 기존 ``FS0_h20_lag1_tree`` 가 smoke(manifest ``period.end`` 2019-12-31 · ``n_folds`` 2)면
  빌드 전에 ``_SMOKE__FS0_h20_lag1_tree`` 로 이름을 바꾼다. 지우지 않는다.
* 동결본 ``FS1h_h20_lag1_rank`` 는 빌드 대상이 아니다. manifest 해시를 전후로 대조해 달라지면 실패한다.
* 빌드 뒤 보존 manifest(``discarded_datasets``)가 있는 4종은 행 수·design 컬럼·기간·null 비율을 대조한다.
* ``--dry-run`` 은 계획·디스크 여유만 출력하고 아무것도 바꾸지 않는다.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import time
from dataclasses import dataclass, replace
from pathlib import Path

from modeler.etl.config import REPO_ROOT
from modeler.models._02_updown_prob import build_dataset as bd
from modeler.models._02_updown_prob.experiments import run_matrix
from modeler.models._02_updown_prob.spec import PERIOD_END, PERIOD_START, ModelSpec, lake_config

HORIZON = 20
FROZEN_KEY = "FS1h_h20_lag1_rank"
SMOKE_PREFIX = "_SMOKE__"
SMOKE_PERIOD_END = run_matrix.SMOKE_PERIOD_END
SMOKE_FOLDS = run_matrix.SMOKE_FOLDS
PRESERVED_ROOT = (
    REPO_ROOT.parent
    / "my/milestones/kr/refactoring/20260912_project_split/02_plan_revised/discarded_datasets"
    / "02_updown_prob/snapshot_date=2026-08-23/source=sj2_remote"
)
# 디스크 추정(GB). rank 2.2~2.6, tree 8~10(smoke 2 fold 1.3GB 에서 키운 추정, 준비 문서 §1.2).
EST_GB = {"rank": 3.0, "tree": 10.0}


@dataclass(frozen=True)
class Target:
    name: str
    feature_set: str
    flow_variant: str
    profile: str
    e5: bool = False
    preserved: bool = False  # 보존 manifest 가 있는가

    def spec(self) -> ModelSpec:
        return ModelSpec(
            feature_set=self.feature_set,
            flow_variant=self.flow_variant,
            preprocess_profile=self.profile,
            seed=0,
        )

    @property
    def key(self) -> str:
        return bd.dataset_key(self.spec(), HORIZON)


TARGETS: tuple[Target, ...] = (
    Target("fs0_rank", "FS0", "lag1", "rank"),
    Target("fs0_tree", "FS0", "lag1", "tree"),
    Target("fs1", "FS1", "lag1", "rank", preserved=True),
    Target("fs2", "FS2", "lag1", "rank", preserved=True),
    Target("native_t", "FS1h", "native_t", "rank", preserved=True),
    Target("fs1h_fs3", "FS1h_FS3", "lag1", "rank", e5=True, preserved=True),
)
TARGET_NAMES: tuple[str, ...] = tuple(t.name for t in TARGETS)


def e5_lake():
    from modeler.models._02_updown_prob.experiments import isolated_lake

    return replace(lake_config(), root=isolated_lake._isolated_root())


def lake_of(target: Target, shared=None, e5=None):
    if target.e5:
        return e5 if e5 is not None else e5_lake()
    return shared if shared is not None else lake_config()


def target_dir(target: Target, shared=None, e5=None) -> Path:
    lake = lake_of(target, shared, e5)
    return lake.dataset_dir(target.spec().model_id) / target.key


# ------------------------------------------------------------------ 판정

def manifest_hash(dataset_dir: Path) -> str | None:
    path = Path(dataset_dir) / "dataset_manifest.json"
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None


def read_manifest(dataset_dir: Path) -> dict | None:
    path = Path(dataset_dir) / "dataset_manifest.json"
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def is_smoke(manifest: dict) -> bool:
    period_end = (manifest.get("period") or {}).get("end")
    n_folds = (manifest.get("extra") or {}).get("n_folds")
    return period_end == SMOKE_PERIOD_END or n_folds == SMOKE_FOLDS


def is_full(manifest: dict, target: Target) -> bool:
    """``ensure_dataset`` 이 재사용할 manifest 인지(같은 feature_set·flow·profile·horizon·기간)."""
    extra = manifest.get("extra") or {}
    return (
        extra.get("feature_set") == target.feature_set
        and extra.get("flow_variant") == target.flow_variant
        and extra.get("preprocess_profile") == target.profile
        and extra.get("horizon") == HORIZON
        and manifest.get("period") == {"start": PERIOD_START, "end": PERIOD_END}
        and extra.get("n_folds") == 5
        and extra.get("std_layout") is not None
    )


@dataclass
class Plan:
    target: Target
    path: Path
    action: str  # build | reuse | rename_then_build | abort
    reason: str = ""
    rename_to: Path | None = None


def plan_target(target: Target, shared=None, e5=None) -> Plan:
    path = target_dir(target, shared, e5)
    if not path.exists():
        return Plan(target, path, "build", "폴더 없음")
    manifest = read_manifest(path)
    if manifest is None:
        return Plan(target, path, "abort", "폴더가 있는데 dataset_manifest.json 이 없다 - 덮어쓸 위험")
    if is_smoke(manifest):
        new = path.with_name(SMOKE_PREFIX + path.name)
        if new.exists():
            return Plan(target, path, "abort", f"smoke 인데 바꿀 이름 {new.name} 이 이미 있다")
        return Plan(target, path, "rename_then_build", "smoke(period.end 2019-12-31 또는 n_folds 2)", new)
    if is_full(manifest, target):
        return Plan(target, path, "reuse", "이미 full - ensure_dataset 이 재사용")
    return Plan(target, path, "abort", "smoke 도 full 도 아닌 manifest - 덮어쓸 위험, 사람이 봐야 한다")


def apply_rename(plan: Plan) -> None:
    """smoke 폴더를 지우지 않고 이름만 바꾼다."""
    assert plan.action == "rename_then_build" and plan.rename_to is not None
    plan.path.rename(plan.rename_to)


class FrozenDatasetChanged(RuntimeError):
    pass


def assert_frozen_unchanged(before: str | None, after: str | None, key: str = FROZEN_KEY) -> None:
    if before != after:
        raise FrozenDatasetChanged(
            f"!!! 동결본 {key} 의 dataset_manifest.json 해시가 바뀌었다: {before} -> {after}"
        )


# ------------------------------------------------------------------ 대조

def preserved_manifest(target: Target, root: Path = PRESERVED_ROOT) -> dict | None:
    if not target.preserved:
        return None
    return read_manifest(root / target.key)


def compare_manifests(preserved: dict, built: dict) -> list[tuple[str, bool, str]]:
    """행 수 · design 컬럼 · 기간 · (있으면) null 비율. ``(항목, 일치, 설명)`` 목록."""
    out: list[tuple[str, bool, str]] = []
    out.append(("row_count", preserved.get("row_count") == built.get("row_count"),
                f"{preserved.get('row_count')} vs {built.get('row_count')}"))
    pe, be = preserved.get("extra") or {}, built.get("extra") or {}
    pd_, bd_ = pe.get("design_columns"), be.get("design_columns")
    same = pd_ == bd_
    detail = f"{len(pd_ or [])} vs {len(bd_ or [])}"
    if not same and pd_ is not None and bd_ is not None:
        detail += f" · 보존에만 {sorted(set(pd_) - set(bd_))} · 새로만 {sorted(set(bd_) - set(pd_))}"
        if set(pd_) == set(bd_):
            detail += " (집합은 같고 순서만 다름)"
    out.append(("design_columns", same, detail))
    out.append(("period", preserved.get("period") == built.get("period"),
                f"{preserved.get('period')} vs {built.get('period')}"))
    pn, bn = pe.get("null_ratios"), be.get("null_ratios")
    if pn is not None:
        if bn is None:
            out.append(("null_ratios", False, "새 manifest 에 없다"))
        else:
            diffs = {c: (pn[c], bn.get(c)) for c in pn if bn.get(c) is None or abs(pn[c] - bn[c]) > 1e-9}
            only_new = sorted(set(bn) - set(pn))
            out.append(("null_ratios", not diffs and not only_new,
                        f"어긋남 {len(diffs)}개 {dict(list(diffs.items())[:5])} · 새 열 {only_new}"))
    return out


# ------------------------------------------------------------------ 실행

def free_gb(path: Path) -> float:
    probe = Path(path)
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    return shutil.disk_usage(probe).free / 1e9


def print_plans(plans: list[Plan], frozen_path: Path) -> None:
    need = 0.0
    print(f"{'이름':10s} {'동작':18s} 경로 / 이유")
    for p in plans:
        print(f"{p.target.name:10s} {p.action:18s} {p.path}")
        print(f"{'':10s} {'':18s} - {p.reason}" + (f" -> 이름 바꾸기 {p.rename_to.name}" if p.rename_to else ""))
        if p.action in {"build", "rename_then_build"}:
            need += EST_GB[p.target.profile]
    print(f"디스크: 필요 추정 약 {need:.0f}GB (rank 약 3GB · tree 약 10GB 추정), 여유 {free_gb(frozen_path):.0f}GB")
    print(f"동결본 {FROZEN_KEY} manifest sha256 {manifest_hash(frozen_path)}")


def run(only: str | None, dry_run: bool, *, shared=None, e5=None, log=print) -> int:
    targets = [t for t in TARGETS if only in (None, t.name)]
    shared_lake = shared if shared is not None else lake_config()
    frozen_path = shared_lake.dataset_dir("02_updown_prob") / FROZEN_KEY
    plans = [plan_target(t, shared_lake, e5) for t in targets]
    print_plans(plans, frozen_path)
    aborts = [p for p in plans if p.action == "abort"]
    if aborts:
        log(f"중단: {[p.target.name for p in aborts]} 은 사람이 봐야 한다")
        return 2
    if dry_run:
        log("--dry-run: 아무것도 바꾸지 않았다")
        return 0
    frozen_before = manifest_hash(frozen_path)
    if frozen_before is None:
        log(f"중단: 동결본 {frozen_path} 의 manifest 가 없다")
        return 2
    failed = 0
    for p in plans:
        t0 = time.time()
        if p.action == "rename_then_build":
            apply_rename(p)
            log(f"[{p.target.name}] smoke 폴더 이름 바꿈 -> {p.rename_to}")
        if p.action == "reuse":
            log(f"[{p.target.name}] 이미 full - 재사용")
        lake = lake_of(p.target, shared_lake, e5)
        dataset_dir, manifest = run_matrix.ensure_dataset(p.target.spec(), lake, HORIZON)  # force 없음
        log(f"[{p.target.name}] {dataset_dir} · {manifest.get('row_count'):,}행 · {time.time() - t0:.1f}s")
        preserved = preserved_manifest(p.target)
        if preserved is None:
            log(f"[{p.target.name}] 보존 manifest 없음 - 대조 못 함")
            continue
        for item, ok, detail in compare_manifests(preserved, manifest):
            log(f"[{p.target.name}]   {item:15s} {'일치' if ok else '어긋남'}  {detail}")
            failed += 0 if ok else 1
    assert_frozen_unchanged(frozen_before, manifest_hash(frozen_path))
    log(f"동결본 {FROZEN_KEY} manifest 해시 그대로")
    if failed:
        log(f"보존 manifest 와 어긋난 항목 {failed}개 - 위 로그를 본다")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dry-run", action="store_true", help="계획·디스크 여유만 출력. 아무것도 바꾸지 않는다")
    parser.add_argument("--only", choices=TARGET_NAMES, help="하나만 만든다")
    args = parser.parse_args(argv)
    return run(args.only, args.dry_run, log=lambda m: print(m, flush=True))


if __name__ == "__main__":
    sys.exit(main())
