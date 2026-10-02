"""Label-free daily US scoring using the frozen feature definitions."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from dataclasses import dataclass
from datetime import date, datetime, time
from pathlib import Path
from typing import Any, Callable
from zoneinfo import ZoneInfo

import joblib
import numpy as np
import polars as pl

from modeler.etl.config import DataRoot
from modeler.serving.calendars import SessionCalendar
from modeler.serving.freshness import assess_freshness
from modeler.serving.orchestration import (
    US_FIN_SERVING_RULE,
    US_PARITY_CRITERIA,
    US_PARITY_EVIDENCE_SCHEMA,
    US_PARITY_MODELS,
    US_PARITY_SCORE_EQUIVALENT,
    us_native_block_reason,
    us_parity_failures,
)
from modeler.serving.schema import report_template, validate_report
from modeler.us.features import FAMILY_ORDER as BASE_FAMILIES
from modeler.us.features.fundamentals_ttm import check_rule
from modeler.us.features.ftd import add_ftd
from modeler.us.features.institutional import add_institutional
from modeler.us.features.investment import add_investment
from modeler.us.features.payout import add_payout
from modeler.us.features.profitability import add_profitability
from modeler.us.features.valuation import add_valuation
from modeler.us.features.order_flow import add_order_flow
from modeler.us.lake import UsLake
from modeler.us.m4_transform import rank_transform, to_design_arrays
from modeler.us.panel import month_first_trading_days
from modeler.us.prices import adjusted_daily

MODEL_ID = "us_exploratory_20260929_r1"
SCORING_VERSION = "us_scoring_daily_v1"
BASE_START = date(2018, 9, 7)
_EASTERN_TZ = ZoneInfo("America/New_York")
_SEOUL_TZ = ZoneInfo("Asia/Seoul")
GOLDEN_DATES = tuple(
    date.fromisoformat(day)
    for day in ("2019-01-02", "2020-04-01", "2021-07-01", "2022-10-03", "2024-01-02", "2025-04-01")
)
FEATURES = (
    "mom_12_1",
    "mom_6_1",
    "mom_1m",
    "rev_1w",
    "max_ret_1m",
    "rv_20",
    "rv_60",
    "idio_vol_60",
    "beta_252",
    "log_dvol_20",
    "amihud_20",
    "mcap_rank",
    "bm",
    "ep_ttm",
    "cfp_ttm",
    "sp_ttm",
    "roa_ttm",
    "roe_ttm",
    "gpa",
    "opm_ttm",
    "asset_growth",
    "accruals",
    "net_issuance",
    "div_yield",
    "buyback_yield",
    "sue_last",
    "days_since_earn",
    "days_to_earn",
    "n_estimates",
    "ins_netbuy_90",
    "ins_cluster_90",
    "ins_officer_buy_90",
    "si_ratio",
    "dtc",
    "si_chg",
    "sv_share_20",
    "n_8k_90",
    "filing_lag",
    "iv_rank",
    "iv_hv_spread",
    "iv_isna",
    "sp500_member",
    "sp500_days_since_add",
    "ftd_share_20",
    "ftd_days_20",
    "ftd_chg",
    "cancel_ratio_20",
    "hidden_share_20",
    "oddlot_share_20",
    "fill_ratio_20",
    "inst_n_log",
    "inst_breadth_chg",
    "inst_shares_chg",
)
if len(FEATURES) != 53 or len(set(FEATURES)) != 53:
    raise RuntimeError("US serving feature contract must contain 53 unique features")
DESIGN_COLUMNS = tuple(f"{name}_rank" for name in FEATURES) + tuple(
    f"{name}_isna" for name in FEATURES
)
#: 서빙이 쓰는 재무 선택 규칙. 연구 경로 기본값(``legacy``)과 달리 서빙은 규칙 A로 고정한다.
SERVING_FINANCIAL_RULE = US_FIN_SERVING_RULE
FINANCIAL_FEATURES = (
    "bm",
    "ep_ttm",
    "cfp_ttm",
    "sp_ttm",
    "roa_ttm",
    "roe_ttm",
    "gpa",
    "opm_ttm",
    "asset_growth",
    "accruals",
    "net_issuance",
)
#: 재무 11개 말고도 ``buyback_yield``는 ``market_cap``·``flow_ttm``을 써서 같은 fact 선택
#: 규칙을 탄다. 기존 규칙에서는 입력 순서에 따라 값이 달라졌으므로(2026-09-30 sj2 실측, 섞은 순서에서
#: 2셀) 서빙은 payout(F8)에도 규칙 A를 쓴다. gate는 이 12개의 결정성을 본다.
RULE_GOVERNED_FEATURES = (*FINANCIAL_FEATURES, "buyback_yield")
#: 규칙 A가 선택 규칙을 받는 family.
FINANCIAL_FAMILIES = frozenset({"F5_valuation", "F6_profitability", "F7_investment", "F8_payout"})
#: 재무 11개가 읽는 XBRL 태그(gate가 fact를 메모리에 올릴 때 거른다).
FINANCIAL_SOURCE_TAGS = (
    "Assets",
    "StockholdersEquity",
    "EntityCommonStockSharesOutstanding",
    "NetIncomeLoss",
    "NetCashProvidedByUsedInOperatingActivities",
    "Revenues",
    "RevenueFromContractWithCustomerExcludingAssessedTax",
    "GrossProfit",
    "CostOfRevenue",
    "OperatingIncomeLoss",
    "PaymentsForRepurchaseOfCommonStock",
)
FLOW_FAMILIES = (
    ("F17_ftd", add_ftd),
    ("F18_order_flow", add_order_flow),
    ("F19_institutional", add_institutional),
)
MANDATORY_TABLES = (
    "company_meta",
    "corp_actions",
    "cusip_symbol_pit",
    "earnings_calendar",
    "filings_index",
    "filings_sub",
    "ftd_fails",
    "fundamentals",
    "index_constituents",
    "inst_holdings_q",
    "insider_owners",
    "insider_trans",
    "listing_snapshots",
    "macro_series",
    "midas_security_daily",
    "prices_daily",
    "short_interest",
    "short_volume",
    "thirteenf_submissions",
    "trading_calendar",
    "universe_daily",
    "volatility_daily",
)


@dataclass(frozen=True)
class PinnedUsLake(UsLake):
    snapshots: dict[str, str]

    def latest_snapshot(self, table: str) -> date:
        return date.fromisoformat(self.snapshots[table])


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_hash(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode()
    return hashlib.sha256(encoded).hexdigest()


def _estimator_variant(estimator: Any) -> str:
    """Identify only the two reviewed estimator families from the loaded object."""
    estimator_type = type(estimator)
    module = estimator_type.__module__
    name = estimator_type.__name__
    if module.startswith("lightgbm.") and name == "LGBMRegressor":
        return "lightgbm"
    if module.startswith("sklearn.linear_model") and name == "Ridge":
        return "ridge"
    raise ValueError(
        f"지원하지 않는 US estimator 형식입니다: {module}.{name}"
    )


def _read_native_completion(
    native_manifest_path: Path,
    *,
    expected_marker_sha256: str | None = None,
    input_cutoff: datetime | None = None,
) -> dict[str, Any]:
    """Read and hash-check the post-publish marker for one immutable native artifact."""
    marker_path = native_manifest_path.parent / "completion.json"
    feature_path = native_manifest_path.parent / "features.parquet"
    if any(path.is_symlink() for path in (native_manifest_path, feature_path, marker_path)):
        raise ValueError("native prepared 파일은 symlink일 수 없습니다")
    if not native_manifest_path.is_file() or not feature_path.is_file() or not marker_path.is_file():
        raise ValueError("native prepared feature, manifest, completion marker가 모두 필요합니다")
    native = json.loads(native_manifest_path.read_text())
    marker_sha = sha256_file(marker_path)
    if expected_marker_sha256 is not None and marker_sha != expected_marker_sha256:
        raise ValueError("completion marker hash가 selection과 다릅니다")
    feature_sha = sha256_file(feature_path)
    native_sha = sha256_file(native_manifest_path)
    marker = json.loads(marker_path.read_text())
    if (
        marker.get("schema_version") != "prepared-features-completion.v1"
        or marker.get("availability_evidence_type") != "prepared_features_completion"
        or marker.get("features_sha256") != feature_sha
        or marker.get("features_sha256") != native.get("features_sha256")
        or marker.get("features_sha256") != native.get("input_sha256")
        or marker.get("native_prepare_manifest_sha256") != native_sha
        or native.get("availability_evidence_type") != "prepared_features_completion"
    ):
        raise ValueError("completion marker가 native manifest 또는 feature hash와 다릅니다")
    verified_at = datetime.fromisoformat(marker.get("verified_available_by", ""))
    if verified_at.tzinfo is None or verified_at.utcoffset() is None:
        raise ValueError("completion marker verified_available_by에는 시간대가 필요합니다")
    if input_cutoff is not None:
        if input_cutoff.tzinfo is None or input_cutoff.utcoffset() is None:
            raise ValueError("input_cutoff에는 시간대가 필요합니다")
        if verified_at > input_cutoff:
            raise ValueError("native prepared artifact가 D 09:30 입력 마감 뒤에 완료됐습니다")
    return {
        "marker": marker,
        "marker_path": marker_path,
        "marker_sha256": marker_sha,
        "features_sha256": feature_sha,
        "native_prepare_manifest_sha256": native_sha,
        "verified_available_by": verified_at,
    }


def _publish_native_directory(
    temporary_dir: Path,
    prep_dir: Path,
    *,
    now_fn: Callable[[], datetime] | None = None,
) -> dict[str, Any]:
    """Publish the immutable native directory, then publish its completion marker."""
    if prep_dir.exists():
        raise FileExistsError(f"같은 prep 경로가 이미 있습니다: {prep_dir}")
    os.rename(temporary_dir, prep_dir)
    verified_at = (now_fn or (lambda: datetime.now().astimezone()))()
    if verified_at.tzinfo is None or verified_at.utcoffset() is None:
        raise ValueError("completion marker 시각에는 시간대가 필요합니다")
    manifest_path = prep_dir / "manifest.json"
    feature_path = prep_dir / "features.parquet"
    marker_path = prep_dir / "completion.json"
    if marker_path.exists():
        raise FileExistsError(f"completion marker가 이미 있습니다: {marker_path}")
    marker = {
        "schema_version": "prepared-features-completion.v1",
        "verified_available_by": verified_at.isoformat(),
        "availability_evidence_type": "prepared_features_completion",
        "features_sha256": sha256_file(feature_path),
        "native_prepare_manifest_sha256": sha256_file(manifest_path),
    }
    marker_tmp = prep_dir / ".completion.json.tmp"
    marker_tmp.write_text(json.dumps(marker, sort_keys=True, indent=2) + "\n")
    os.rename(marker_tmp, marker_path)
    return marker


def _rank_golden(model: Any, frame: pl.DataFrame) -> pl.DataFrame:
    parts = [rank_daily(model, frame.filter(pl.col("date") == day)) for day in sorted(frame["date"].unique())]
    return pl.concat(parts).sort(["date", "rank"])


def export_model_bundle(
    *,
    source_model: Path,
    bundle_dir: Path,
    model_version: str,
    golden_features: pl.DataFrame,
    golden_source_sha256: dict[str, str],
) -> dict[str, Any]:
    """Copy an unchanged exploratory model and pin six feature-only golden sessions."""
    if model_version != "1":
        raise ValueError("US serving bundle model_version은 1이어야 합니다")
    model = joblib.load(source_model)
    if model.get("kind") != "exploratory" or tuple(model.get("features", ())) != FEATURES:
        raise ValueError("원본 모델의 kind 또는 53개 피쳐 순서가 다릅니다")
    variant = _estimator_variant(model.get("model"))
    golden_dates = tuple(sorted(golden_features["date"].unique().to_list()))
    if golden_dates != GOLDEN_DATES:
        raise ValueError(f"golden 날짜가 사전 고정 목록과 다릅니다: {golden_dates}")
    if any(d > date(2025, 6, 30) for d in golden_features["date"].unique().to_list()):
        raise ValueError("golden 입력 날짜가 학습 경계를 넘습니다")
    expected_features = {"date", "symbol", "price_ge_5", *FEATURES}
    missing = sorted(expected_features - set(golden_features.columns))
    if missing:
        raise ValueError(f"golden 입력에 피쳐가 빠졌습니다: {missing}")
    bundle_dir.mkdir(parents=True, exist_ok=False)
    copied_model = bundle_dir / "model.joblib"
    shutil.copy2(source_model, copied_model)
    golden_path = bundle_dir / "golden_features.parquet"
    prediction_path = bundle_dir / "golden_predictions.parquet"
    golden_features.sort(["date", "symbol"]).write_parquet(golden_path)
    predictions = _rank_golden(model["model"], golden_features)
    predictions.write_parquet(prediction_path)
    manifest = {
        "schema_version": 1,
        "model_id": MODEL_ID,
        "model_version": model_version,
        "variant": variant,
        "model_kind": model["kind"],
        "model_sha256": sha256_file(copied_model),
        "features": list(FEATURES),
        "feature_count": len(FEATURES),
        "design_columns": list(DESIGN_COLUMNS),
        "design_column_count": len(DESIGN_COLUMNS),
        "transform": {
            "eligible_filter": "price_ge_5 == true",
            "null_values": "nonfinite_to_null",
            "missing_flags": "feature_isna_is_null_before_fill",
            "cross_sectional_rank": "(rank(method=min)-1)/(n_nonnull-1)",
            "missing_rank_fill": 0.5,
            "feature_order": list(FEATURES),
            "design_order": ["all feature_rank columns", "all feature_isna columns"],
        },
        "train_end": str(model["train_end"]),
        "max_training_terminal": str(model["max_training_terminal"]),
        "golden_dates": [d.isoformat() for d in sorted(golden_features["date"].unique())],
        "golden_feature_rows": golden_features.height,
        "golden_source_sha256": dict(sorted(golden_source_sha256.items())),
        "golden_features_sha256": sha256_file(golden_path),
        "golden_predictions_sha256": sha256_file(prediction_path),
        "prediction_tolerance": {"rtol": 0, "atol": 1e-12},
    }
    (bundle_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True, default=str) + "\n"
    )
    return manifest


def export_frozen_model_bundles(
    *,
    frozen_root: Path,
    source_models: dict[str, Path],
    bundle_root: Path,
    model_version: str,
) -> dict[str, dict[str, Any]]:
    """Export both unchanged models with six pinned frozen feature-only sessions."""
    if set(source_models) != {"lightgbm", "ridge"}:
        raise ValueError("lightgbm·ridge 원본 model.joblib 둘 다 필요합니다")
    golden, source_hashes = read_frozen_golden_features(frozen_root)
    bundle_root.mkdir(parents=True, exist_ok=True)
    exported = {
        name: export_model_bundle(
            source_model=source_models[name],
            bundle_dir=bundle_root / name,
            model_version=model_version,
            golden_features=golden,
            golden_source_sha256=source_hashes,
        )
        for name in ("lightgbm", "ridge")
    }
    for name, manifest in exported.items():
        if manifest.get("variant") != name:
            raise ValueError(f"{name} bundle 경로에 다른 estimator variant가 지정됐습니다")
    return exported


def _monthly_membership(lake: UsLake, scoring_date: date) -> pl.DataFrame:
    """Use the first-session member list for that month, without rejudging it daily."""
    month_start = scoring_date.replace(day=1)
    first_sessions = month_first_trading_days(lake, month_start, scoring_date)
    if not first_sessions:
        raise ValueError(f"거래 캘린더에 {scoring_date:%Y-%m} 세션이 없습니다")
    membership_date = first_sessions[-1]
    members = (
        lake.scan("universe_daily")
        .filter((pl.col("date") == membership_date) & pl.col("in_universe"))
        .select("symbol", "cik", "sic", "mcap_rank", "adv_20d", "exchange")
        .with_columns(
            pl.when(pl.col("sic").is_not_null())
            .then(pl.col("sic").str.slice(0, 2))
            .otherwise(None)
            .alias("sic2")
        )
        .collect()
    )
    if members.is_empty():
        raise ValueError(f"월간 유니버스가 비었습니다: {membership_date}")
    if members["symbol"].n_unique() != members.height:
        raise ValueError(f"월간 유니버스에 중복 심볼이 있습니다: {membership_date}")
    return members.with_columns(pl.lit(scoring_date).cast(pl.Date).alias("date"))


def build_daily_panel(
    lake: UsLake, scoring_date: date, *, decision_at: datetime | None = None
) -> pl.DataFrame:
    """Build the requested completed-session panel from monthly membership and PIT prices."""
    calendar_row = (
        lake.scan("trading_calendar")
        .filter((pl.col("exchange") == "XNYS") & (pl.col("date") == scoring_date))
        .select("date", "close_local")
        .collect()
    )
    if calendar_row.height != 1:
        raise ValueError(f"scoring_date가 XNYS 완료 세션이 아닙니다: {scoring_date}")
    if decision_at is not None:
        if decision_at.tzinfo is None or decision_at.utcoffset() is None:
            raise ValueError("decision_at에는 시간대가 필요합니다")
        close_time = calendar_row["close_local"].item()
        session_close = datetime.combine(scoring_date, close_time, tzinfo=_EASTERN_TZ)
        if session_close >= decision_at.astimezone(_EASTERN_TZ):
            raise ValueError(f"decision_at 전에 장이 끝나지 않은 세션입니다: {scoring_date}")

    members = _monthly_membership(lake, scoring_date)
    raw = (
        lake.scan("prices_daily")
        .filter(pl.col("date") == scoring_date)
        .select("date", "symbol", pl.col("close").cast(pl.Float64).alias("close"))
    )
    adjusted = adjusted_daily(lake, base_date=scoring_date).filter(
        pl.col("date") == scoring_date
    )
    panel = (
        members.lazy()
        .join(raw, on=["date", "symbol"], how="left", validate="1:1")
        .join(
            adjusted.select("date", "symbol", "adj_close", "adj_volume"),
            on=["date", "symbol"],
            how="left",
            validate="1:1",
        )
        .with_columns((pl.col("close") >= 5).alias("price_ge_5"))
        .select(
            "date",
            "symbol",
            "cik",
            "sic",
            "sic2",
            "mcap_rank",
            "adv_20d",
            "exchange",
            "close",
            "adj_close",
            "adj_volume",
            "price_ge_5",
        )
        .sort("symbol")
        .collect()
    )
    if panel.is_empty() or panel["close"].null_count() == panel.height:
        raise ValueError(f"{scoring_date}에 월간 유니버스의 가격이 없습니다")
    return panel


def build_daily_features(
    lake: UsLake,
    scoring_date: date,
    *,
    decision_at: datetime | None = None,
    financial_selection_rule: str = SERVING_FINANCIAL_RULE,
) -> pl.DataFrame:
    """Build the frozen 53 features for one completed US session; labels are never opened."""
    return build_daily_features_for_dates(
        lake,
        [scoring_date],
        decision_at_by_date={scoring_date: decision_at},
        financial_selection_rule=financial_selection_rule,
    )


def _add_base_families(
    features: pl.DataFrame, lake: UsLake, financial_selection_rule: str
) -> pl.DataFrame:
    check_rule(financial_selection_rule)
    for family_name, add_family in BASE_FAMILIES:
        if family_name in FINANCIAL_FAMILIES:
            features = add_family(features, lake, selection_rule=financial_selection_rule)
        else:
            features = add_family(features, lake)
    return features


def build_daily_features_for_dates(
    lake: UsLake,
    scoring_dates: list[date],
    *,
    decision_at_by_date: dict[date, datetime | None] | None = None,
    financial_selection_rule: str = SERVING_FINANCIAL_RULE,
) -> pl.DataFrame:
    """Build multiple feature-only sessions together so each source scan is shared."""
    if not scoring_dates or len(set(scoring_dates)) != len(scoring_dates):
        raise ValueError("scoring_dates는 비어 있지 않고 중복이 없어야 합니다")
    decision_at_by_date = decision_at_by_date or {}
    panels = [
        build_daily_panel(lake, day, decision_at=decision_at_by_date.get(day))
        for day in sorted(scoring_dates)
    ]
    features = pl.concat(panels).sort(["date", "symbol"])
    features = _add_base_families(features, lake, financial_selection_rule)
    for _family_name, add_family in FLOW_FAMILIES:
        features = add_family(features, lake)

    missing = [name for name in FEATURES if name not in features.columns]
    if missing:
        raise ValueError(f"필수 US 피쳐가 없습니다: {missing}")
    if set(features["date"].unique().to_list()) != set(scoring_dates):
        raise ValueError("피쳐 출력 날짜가 요청 목록과 다릅니다")
    return features.sort("symbol")


def read_frozen_golden_features(frozen_root: Path) -> tuple[pl.DataFrame, dict[str, str]]:
    """Read only the six preselected feature sessions; never open label datasets."""
    base_dir = frozen_root / "us_features_v2"
    flow_dir = frozen_root / "us_features_flow_v1"
    base_path, flow_path = base_dir / "part.parquet", flow_dir / "part.parquet"
    if not base_path.is_file() or not flow_path.is_file():
        raise FileNotFoundError("label-free frozen v2/flow feature files가 없습니다")
    base_names = set(pl.scan_parquet(base_path).collect_schema().names())
    flow_names = set(pl.scan_parquet(flow_path).collect_schema().names())
    feature_flags = [f"{name}_isna" for name in FEATURES]
    base_cols = [name for name in FEATURES if name in base_names]
    base_flag_cols = [name for name in feature_flags if name in base_names]
    flow_cols = [name for name in FEATURES if name not in base_names and name in flow_names]
    flow_flag_cols = [name for name in feature_flags if name in flow_names]
    absent = sorted(set(FEATURES) - set(base_cols) - set(flow_cols))
    if absent:
        raise ValueError(f"frozen feature datasets에 컬럼이 없습니다: {absent}")
    date_filter = pl.col("date").is_in(GOLDEN_DATES)
    base = (
        pl.scan_parquet(base_path)
        .filter(date_filter)
        .select("date", "symbol", "price_ge_5", "close", "adv_20d", *base_cols, *base_flag_cols)
    )
    flow = (
        pl.scan_parquet(flow_path)
        .filter(date_filter)
        .select("date", "symbol", *flow_cols, *flow_flag_cols)
    )
    frozen = base.join(flow, on=["date", "symbol"], how="left", validate="1:1").collect()
    if frozen.height == 0 or set(frozen["date"].unique().to_list()) != set(GOLDEN_DATES):
        raise ValueError("frozen v2/flow에 사전 고정한 6개 날짜가 모두 없습니다")
    if frozen.select(pl.struct("date", "symbol").n_unique()).item() != frozen.height:
        raise ValueError("frozen v2 golden 입력에 중복 키가 있습니다")
    source_hashes = {
        "us_features_v2": sha256_file(base_path),
        "us_features_flow_v1": sha256_file(flow_path),
    }
    return frozen.sort(["date", "symbol"]), source_hashes


def compare_frozen_features(actual: pl.DataFrame, frozen: pl.DataFrame) -> pl.DataFrame:
    """Compare daily-built raw features to frozen v2 without reading any label table."""
    feature_flags = {f"{name}_isna" for name in FEATURES}
    required_values = {"date", "symbol", "price_ge_5", *FEATURES}
    comparable = []
    for label, frame in (("actual", actual), ("frozen", frozen)):
        missing_values = sorted(required_values - set(frame.columns))
        if missing_values:
            raise ValueError(f"{label} 피쳐 입력에 컬럼이 없습니다: {missing_values}")
        missing_flags = feature_flags - set(frame.columns)
        # F13 emits iv_isna as a non-null indicator, not its own missingness flag.
        # The original exploratory_run.prepare_features recomputes every input
        # flag from the cleaned feature values, so regenerate this one flag only.
        if missing_flags - {"iv_isna_isna"}:
            raise ValueError(f"{label} 피쳐 입력에 missing flag가 없습니다: {sorted(missing_flags)}")
        if "iv_isna_isna" in missing_flags:
            frame = frame.with_columns(pl.col("iv_isna").is_null().alias("iv_isna_isna"))
        comparable.append(frame)
    actual, frozen = comparable
    dates = tuple(sorted(actual["date"].unique().to_list()))
    if dates != GOLDEN_DATES or dates != tuple(sorted(frozen["date"].unique().to_list())):
        raise ValueError("대조 입력에는 같은 6개 사전 고정 날짜가 있어야 합니다")
    required = required_values | feature_flags
    actual = actual.select(sorted(required)).sort(["date", "symbol"])
    frozen = frozen.select(sorted(required)).sort(["date", "symbol"])
    if actual.select("date", "symbol").rows() != frozen.select("date", "symbol").rows():
        raise ValueError("일일 경로와 frozen v2의 키 또는 유니버스가 다릅니다")
    if actual["price_ge_5"].to_list() != frozen["price_ge_5"].to_list():
        raise ValueError("price_ge_5 유니버스가 frozen v2와 다릅니다")
    diagnostics: list[dict[str, Any]] = []
    for name in FEATURES:
        left, right = actual[name], frozen[name]
        if left.is_null().to_list() != right.is_null().to_list():
            raise ValueError(f"{name} null mask가 frozen v2와 다릅니다")
        expected_isna = left.is_null().to_list()
        if actual[f"{name}_isna"].to_list() != expected_isna:
            raise ValueError(f"일일 경로 {name}_isna 플래그가 값의 null mask와 다릅니다")
        if frozen[f"{name}_isna"].to_list() != expected_isna:
            raise ValueError(f"frozen v2 {name}_isna 플래그가 값의 null mask와 다릅니다")
        left_values = left.cast(pl.Float64).to_numpy()
        right_values = right.cast(pl.Float64).to_numpy()
        present = ~np.isnan(left_values)
        max_abs_delta = (
            float(np.max(np.abs(left_values[present] - right_values[present])))
            if present.any()
            else 0.0
        )
        if not np.allclose(left_values, right_values, rtol=1e-9, atol=1e-12, equal_nan=True):
            raise ValueError(f"{name} 값이 frozen v2와 다릅니다")
        diagnostics.append(
            {
                "feature": name,
                "rows": int(present.sum()),
                "missing_rows": int((~present).sum()),
                "max_abs_delta": max_abs_delta,
            }
        )
    return pl.DataFrame(diagnostics)


def transform_daily_features(frame: pl.DataFrame) -> tuple[pl.DataFrame, np.ndarray]:
    """Apply the original filter, null mask, cross-sectional ranks, and 106-column order."""
    missing = [name for name in FEATURES if name not in frame.columns]
    if missing:
        raise ValueError(f"모델 입력 피쳐가 없습니다: {missing}")
    selected = frame.filter(pl.col("price_ge_5")).sort("date", "symbol")
    selected = selected.with_columns(
        [
            pl.when(pl.col(name).cast(pl.Float64).is_finite())
            .then(pl.col(name).cast(pl.Float64))
            .otherwise(None)
            .alias(name)
            for name in FEATURES
        ]
    )
    selected = selected.with_columns(
        [pl.col(name).is_null().alias(f"{name}_isna") for name in FEATURES]
    )
    transformed = rank_transform(selected, FEATURES)
    matrix, columns = to_design_arrays(transformed, FEATURES)
    if tuple(columns) != DESIGN_COLUMNS:
        raise ValueError("모델 입력 컬럼 순서가 bundle 계약과 다릅니다")
    if matrix.shape != (transformed.height, 106) or not np.isfinite(matrix).all():
        raise ValueError("모델 입력 행렬의 크기 또는 값이 올바르지 않습니다")
    return transformed, matrix


def load_model(model_path: Path, manifest_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("model_id") != MODEL_ID or manifest.get("model_version") != "1":
        raise ValueError("US bundle model_id 또는 model_version이 지원되지 않습니다")
    if manifest.get("variant") not in {"lightgbm", "ridge"}:
        raise ValueError("US bundle에 명시된 model variant가 없습니다")
    actual_hash = sha256_file(model_path)
    if manifest.get("model_sha256") != actual_hash:
        raise ValueError(f"model.joblib hash가 manifest와 다릅니다: {model_path}")
    if tuple(manifest.get("features", ())) != FEATURES:
        raise ValueError("serving bundle feature order가 53개 계약과 다릅니다")
    if tuple(manifest.get("design_columns", ())) != DESIGN_COLUMNS:
        raise ValueError("serving bundle design matrix가 106열 계약과 다릅니다")
    bundle = joblib.load(model_path)
    if bundle.get("kind") != "exploratory" or tuple(bundle.get("features", ())) != FEATURES:
        raise ValueError("원본 탐색 모델 bundle 형식이 예상과 다릅니다")
    if _estimator_variant(bundle.get("model")) != manifest["variant"]:
        raise ValueError("manifest variant가 실제 estimator 형식과 다릅니다")
    for field in ("train_end", "max_training_terminal"):
        if date.fromisoformat(str(bundle[field])[:10]) > date(2025, 6, 30):
            raise ValueError(f"학습 경계가 2025-06-30을 넘습니다: {field}")
    golden_features_path = manifest_path.parent / "golden_features.parquet"
    golden_predictions_path = manifest_path.parent / "golden_predictions.parquet"
    if sha256_file(golden_features_path) != manifest.get("golden_features_sha256"):
        raise ValueError("golden 입력 hash가 manifest와 다릅니다")
    if sha256_file(golden_predictions_path) != manifest.get("golden_predictions_sha256"):
        raise ValueError("golden 예측 hash가 manifest와 다릅니다")
    golden_features = pl.read_parquet(golden_features_path)
    actual = _rank_golden(bundle["model"], golden_features)
    expected = pl.read_parquet(golden_predictions_path)
    if actual.select("date", "symbol", "rank").rows() != expected.select(
        "date", "symbol", "rank"
    ).rows():
        raise ValueError("golden 재로딩 순위가 일치하지 않습니다")
    np.testing.assert_allclose(
        actual["score"].to_numpy(),
        expected["score"].to_numpy(),
        rtol=manifest["prediction_tolerance"]["rtol"],
        atol=manifest["prediction_tolerance"]["atol"],
    )
    return bundle, manifest


def rank_daily(model: Any, frame: pl.DataFrame) -> pl.DataFrame:
    transformed, matrix = transform_daily_features(frame)
    scores = model.predict(matrix)
    if len(scores) != transformed.height or not np.isfinite(scores).all():
        raise ValueError("모델 점수가 비었거나 유한하지 않습니다")
    return (
        transformed.select("date", "symbol", "close", "adv_20d")
        .with_columns(pl.Series("score", scores))
        .sort(["score", "symbol"], descending=[True, False])
        .with_row_index("rank", offset=1)
        .with_columns(pl.col("rank").cast(pl.UInt32))
    )


def _source_revision(lake: UsLake) -> dict[str, str]:
    revisions = lake.snapshot_manifest()
    missing = sorted(set(MANDATORY_TABLES) - set(revisions))
    if missing:
        raise ValueError(f"US source snapshot이 없습니다: {missing}")
    return revisions


def input_cutoff_for_report_date(report_date: date) -> datetime:
    """Feature inputs must be available by 09:30 KST; 10:00 is inference start."""
    return datetime.combine(report_date, time(9, 30), tzinfo=_SEOUL_TZ)


def validate_prepared_cutoff(
    *, available_at: datetime, input_cutoff: datetime, report_date: date
) -> None:
    """Check an aware prepare completion against the fixed 09:30 KST cutoff."""
    if available_at.tzinfo is None or available_at.utcoffset() is None:
        raise ValueError("prepared available_at에는 시간대가 필요합니다")
    if input_cutoff.tzinfo is None or input_cutoff.utcoffset() is None:
        raise ValueError("input_cutoff에는 시간대가 필요합니다")
    expected = input_cutoff_for_report_date(report_date)
    if input_cutoff.astimezone(_SEOUL_TZ) != expected:
        raise ValueError("input_cutoff은 report_date 당일 09:30 Asia/Seoul이어야 합니다")
    if available_at > input_cutoff:
        raise ValueError("prepared features가 D 09:30 입력 마감 뒤에 준비됐습니다")


def validate_selection_completion(*, completed_at: datetime, decision_at: datetime) -> None:
    """Selection may finish after 09:30, but must be ready by inference time."""
    if completed_at.tzinfo is None or completed_at.utcoffset() is None:
        raise ValueError("selection completed_at에는 시간대가 필요합니다")
    if decision_at.tzinfo is None or decision_at.utcoffset() is None:
        raise ValueError("decision_at에는 시간대가 필요합니다")
    if completed_at > decision_at:
        raise ValueError("D별 selection이 10:00 추론 시작 뒤에 완성됐습니다")


def validate_inference_time(*, decision_at: datetime, now: datetime | None = None) -> None:
    """Allow production inference only at or after the exact scheduled instant."""
    if decision_at.tzinfo is None or decision_at.utcoffset() is None:
        raise ValueError("decision_at에는 시간대가 필요합니다")
    decision_kst = decision_at.astimezone(_SEOUL_TZ)
    if (
        decision_kst.hour != 10
        or decision_kst.minute != 0
        or decision_kst.second != 0
        or decision_kst.microsecond != 0
    ):
        raise ValueError("decision_at은 정확히 10:00:00 Asia/Seoul이어야 합니다")
    current = now if now is not None else datetime.now(decision_at.tzinfo)
    if current.tzinfo is None or current.utcoffset() is None:
        raise ValueError("현재 시각에는 시간대가 필요합니다")
    if current < decision_at:
        raise ValueError("실제 추론 시각이 decision_at보다 이릅니다")


def write_daily_run(
    *,
    root: DataRoot,
    scoring_date: date,
    features: pl.DataFrame,
    model_paths: dict[str, Path],
    model_manifests: dict[str, Path],
    source_revision: dict[str, str],
    code_hash: str,
    report_date: date,
    decision_at: datetime,
    prepared_available_at: str,
    input_cutoff: str,
    freshness_status: str,
    selection_manifest_sha256: str,
    freshness_metadata: dict[str, Any] | None = None,
) -> dict[str, Path]:
    """Write both immutable model outputs; same hash reuses, conflicts never overwrite."""
    if set(model_paths) != {"lightgbm", "ridge"} or set(model_manifests) != set(model_paths):
        raise ValueError("lightgbm·ridge 모델과 각 manifest가 모두 필요합니다")
    if freshness_status not in {"ok", "stale"}:
        raise ValueError(f"freshness가 {freshness_status}라 inference를 중단합니다")
    if decision_at.tzinfo is None or decision_at.utcoffset() is None:
        raise ValueError("decision_at에는 시간대가 필요합니다")
    prepared_at = datetime.fromisoformat(prepared_available_at)
    cutoff_at = datetime.fromisoformat(input_cutoff)
    validate_prepared_cutoff(
        available_at=prepared_at, input_cutoff=cutoff_at, report_date=report_date
    )
    decision_kst = decision_at.astimezone(_SEOUL_TZ)
    if decision_kst.date() != report_date:
        raise ValueError("decision_at은 report_date 당일 10:00이어야 합니다")
    validate_inference_time(decision_at=decision_at)
    model_inputs = {
        name: sha256_file(model_path) for name, model_path in sorted(model_paths.items())
    }
    freshness_metadata = freshness_metadata or {}
    input_digest = _canonical_hash(
        {
            "scoring_date": scoring_date,
            "features": features.hash_rows(seed=0).to_list(),
            "source_revision": source_revision,
        }
    )
    paths: dict[str, Path] = {}
    for name in ("lightgbm", "ridge"):
        model_manifest = json.loads(model_manifests[name].read_text())
        if (
            model_manifest.get("model_id") != MODEL_ID
            or model_manifest.get("model_version") != "1"
            or model_manifest.get("variant") != name
        ):
            raise ValueError(f"{name} 슬롯의 US bundle identity가 다릅니다")
        run_payload = {
            "scoring_version": SCORING_VERSION,
            "scoring_date": scoring_date.isoformat(),
            "report_date": report_date.isoformat(),
            "decision_at": decision_at.isoformat(),
            "prepared_available_at": prepared_available_at,
            "input_cutoff": input_cutoff,
            "freshness_status": freshness_status,
            "selection_manifest_sha256": selection_manifest_sha256,
            "source_revision": source_revision,
            "feature_hash": input_digest,
            "model_hash": model_inputs[name],
            "model_manifest_sha256": sha256_file(model_manifests[name]),
            "model_version": model_manifest.get("model_version"),
            "model_variant": model_manifest.get("variant"),
            "code_hash": code_hash,
        }
        run_hash = _canonical_hash(run_payload)
        run_id = run_hash[:16]
        output_dir = root.output / SCORING_VERSION / f"score_date={scoring_date}" / name / f"run_id={run_id}"
        manifest_path = output_dir / "manifest.json"
        rankings_path = output_dir / "rankings.parquet"
        report_path = output_dir / "report.json"
        if output_dir.exists():
            if not manifest_path.is_file() or not rankings_path.is_file() or not report_path.is_file():
                raise FileExistsError(f"완료되지 않은 출력이 있어 덮지 않습니다: {output_dir}")
            previous = json.loads(manifest_path.read_text())
            if (
                previous.get("run_hash") != run_hash
                or previous.get("model_sha256") != model_inputs[name]
                or previous.get("input_features_sha256")
                != sha256_file(output_dir / "input_features.parquet")
                or previous.get("design_matrix_sha256")
                != sha256_file(output_dir / "design_matrix.npy")
                or previous.get("rankings_sha256") != sha256_file(rankings_path)
                or previous.get("report_sha256") != sha256_file(report_path)
            ):
                raise FileExistsError(f"같은 run_id의 출력 내용이 달라 덮지 않습니다: {output_dir}")
            paths[name] = output_dir
            continue

        bundle, bundle_manifest = load_model(model_paths[name], model_manifests[name])
        if bundle_manifest.get("variant") != name:
            raise ValueError(f"{name} 슬롯에 다른 US 모델 variant가 지정됐습니다")
        transformed, matrix = transform_daily_features(features)
        ranked = rank_daily(bundle["model"], features)
        model_id = f"{MODEL_ID}_{name}"
        report = report_template(
            market="US",
            report_date=report_date.isoformat(),
            decision_at=decision_at,
            feature_asof_date=scoring_date.isoformat(),
            model_id=model_id,
            model_version=str(bundle_manifest["model_version"]),
        )
        report.update(
            status=freshness_status,
            rankings=[
                {
                    "rank": int(row["rank"]),
                    "symbol": row["symbol"],
                    "name": row["symbol"],
                    "score": float(row["score"]),
                }
                for row in ranked.iter_rows(named=True)
            ],
            quality={
                "freshness_status": freshness_status,
                "feature_count": 53,
                "design_column_count": 106,
                "eligible_rows": ranked.height,
                "monthly_membership_month": scoring_date.strftime("%Y-%m"),
                "latest_us_session": freshness_metadata.get("latest_us_session"),
                "expected_us_session": freshness_metadata.get("expected_us_session"),
                "actual_us_session": freshness_metadata.get("actual_us_session"),
                "delivery_lag_sessions": freshness_metadata.get("delivery_lag_sessions"),
                "market_lag_sessions": freshness_metadata.get("market_lag_sessions"),
                "market_lag_limit_sessions": freshness_metadata.get("market_lag_limit_sessions"),
            },
            provenance={
                "scoring_version": SCORING_VERSION,
                "run_id": run_id,
                "feature_hash": input_digest,
                "model_sha256": model_inputs[name],
                "source_revision": source_revision,
                "prepared_available_at": prepared_available_at,
                "input_cutoff": input_cutoff,
                "freshness_status": freshness_status,
                "source_first_available_at": freshness_metadata.get("source_first_available_at"),
                "verified_available_by": freshness_metadata.get("verified_available_by"),
                "availability_evidence_type": freshness_metadata.get("availability_evidence_type"),
                "score_semantics": "raw model ranking score; not a probability or confidence",
                "display_name_source": "symbol fallback; company-name mapping is not part of scoring",
                "labels_read": False,
            },
        )
        validate_report(report)
        output_dir.parent.mkdir(parents=True, exist_ok=True)
        temporary_dir = Path(tempfile.mkdtemp(prefix=f".{run_id}.", dir=output_dir.parent))
        temp_rankings = temporary_dir / "rankings.parquet"
        transformed.write_parquet(temporary_dir / "input_features.parquet")
        np.save(temporary_dir / "design_matrix.npy", matrix, allow_pickle=False)
        ranked.write_parquet(temp_rankings)
        manifest = {
            **run_payload,
            "schema_version": 1,
            "market": "US",
            "model_id": model_id,
            "model_version": bundle_manifest["model_version"],
            "model_variant": bundle_manifest["variant"],
            "run_id": run_id,
            "run_hash": run_hash,
            "feature_count": 53,
            "design_column_count": 106,
            "universe": "monthly_membership_extended_to_scoring_session_and_price_ge_5",
            "row_count": ranked.height,
            "model_sha256": model_inputs[name],
            "input_feature_rows": transformed.height,
            "input_features_sha256": sha256_file(temporary_dir / "input_features.parquet"),
            "design_matrix_sha256": sha256_file(temporary_dir / "design_matrix.npy"),
            "rankings_sha256": sha256_file(temp_rankings),
            "report_sha256": "",
            "freshness_status": freshness_status,
            "labels_read": False,
        }
        report_path_tmp = temporary_dir / "report.json"
        report_path_tmp.write_text(json.dumps(report, sort_keys=True, indent=2) + "\n")
        manifest["report_sha256"] = sha256_file(report_path_tmp)
        (temporary_dir / "manifest.json").write_text(
            json.dumps(manifest, sort_keys=True, indent=2) + "\n"
        )
        if output_dir.exists():
            raise FileExistsError(f"출력 경로가 생성되는 동안 이미 생겼습니다: {output_dir}")
        os.rename(temporary_dir, output_dir)
        paths[name] = output_dir
    return paths


def collect_source_revision(lake: UsLake) -> dict[str, str]:
    """Public wrapper for the exact table snapshot set recorded by a daily run."""
    return _source_revision(lake)


def fingerprint_source_snapshots(root: DataRoot, revisions: dict[str, str]) -> dict[str, list[dict[str, Any]]]:
    """Identify every parquet file read through the pinned 22-table lake."""
    result: dict[str, list[dict[str, Any]]] = {}
    for table, revision in sorted(revisions.items()):
        directory = root.derived / "snapshots" / table / f"snapshot_date={revision}"
        paths = sorted(directory.glob("*.parquet"))
        if not paths:
            raise FileNotFoundError(f"US source snapshot parquet is missing: {directory}")
        result[table] = [
            {
                "path": str(path.relative_to(root.base)),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in paths
        ]
    return result


def verify_universe_completion(root: DataRoot, revision: str,
                               source_revisions: dict[str, str]) -> str:
    """Require the monthly seed/stitch manifest for a live US preparation."""
    part = root.derived / "snapshots" / "universe_daily" / f"snapshot_date={revision}" / "part.parquet"
    marker = part.parent / "completion.json"
    if not part.is_file() or not marker.is_file() or part.is_symlink() or marker.is_symlink():
        raise ValueError("US universe_daily requires a completed incremental membership snapshot")
    record = json.loads(marker.read_text())
    if (record.get("schema_version") != 1 or record.get("table") != "universe_daily"
            or record.get("snapshot_date") != revision
            or record.get("snapshot_sha256") != sha256_file(part)
            or record.get("unjudged_months") != []
            or not isinstance(record.get("input_snapshot_sha256"), dict)
            or not isinstance(record.get("input_snapshots"), dict)
            or not isinstance(record.get("ticker_source_sha256"), dict)
            or not record.get("ticker_source_sha256")
            or not record.get("previous_month_seed_count")):
        raise ValueError("US universe_daily completion or monthly seed evidence is invalid")
    required_inputs = {"prices_daily", "listing_snapshots", "filings_sub", "midas_security_daily"}
    if (set(record["input_snapshot_sha256"]) != required_inputs or
            set(record["input_snapshots"]) != required_inputs):
        raise ValueError("US universe completion does not identify every required input")
    for table, expected_hash in record["input_snapshot_sha256"].items():
        if (not isinstance(table, str) or not isinstance(expected_hash, str)
                or source_revisions.get(table) != record["input_snapshots"].get(table)):
            raise ValueError("US universe input revision differs from prepared source revision")
        source = (root.derived / "snapshots" / table /
                  f"snapshot_date={source_revisions[table]}" / "part.parquet")
        if not source.is_file() or sha256_file(source) != expected_hash:
            raise ValueError(f"US universe source snapshot changed: {table}")
    for filename, expected_hash in record["ticker_source_sha256"].items():
        if Path(filename).is_absolute() or ".." in Path(filename).parts:
            raise ValueError("US universe ticker source path must stay within the lake root")
        path = root.base / filename
        if not path.is_file() or sha256_file(path) != expected_hash:
            raise ValueError("US universe ticker source changed")
    return sha256_file(marker)


def model_paths(root: DataRoot) -> tuple[dict[str, Path], dict[str, Path]]:
    bundle_root = Path(
        os.environ.get(
            "US_SERVING_BUNDLE_ROOT",
            str(root.output / "serving_bundles" / MODEL_ID),
        )
    )
    models = {name: bundle_root / name / "model.joblib" for name in ("lightgbm", "ridge")}
    manifests = {name: bundle_root / name / "manifest.json" for name in models}
    return models, manifests


def code_tree_hash() -> str:
    """Hash serving, panel, pricing, registry, and all feature-family source code."""
    us_root = Path(__file__).parents[1] / "us"
    module_paths = [
        Path(__file__),
        Path(__file__).with_name("schema.py"),
        us_root / "m4_transform.py",
        us_root / "panel.py",
        us_root / "prices.py",
        us_root / "lake.py",
        us_root / "features" / "__init__.py",
        us_root / "features" / "ftd.py",
        us_root / "features" / "order_flow.py",
        us_root / "features" / "institutional.py",
    ]
    module_paths.extend((us_root / "features").glob("*.py"))
    digest = hashlib.sha256()
    for path in sorted(set(module_paths)):
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def financial_code_hash() -> str:
    """Hash the code that decides the 11 financial features (selection rule and its callers)."""
    features_dir = Path(__file__).parents[1] / "us" / "features"
    digest = hashlib.sha256()
    for name in ("fundamentals_ttm.py", "valuation.py", "profitability.py", "investment.py", "payout.py"):
        digest.update(name.encode())
        digest.update((features_dir / name).read_bytes())
    return digest.hexdigest()


def validate_parity_evidence(evidence_path: Path) -> dict[str, Any]:
    """Load a parity evidence JSON and require a passing, current-code, rule-A result."""
    evidence = json.loads(evidence_path.read_text())
    if evidence.get("status") != US_PARITY_SCORE_EQUIVALENT:
        raise ValueError("parity evidence status가 score_equivalent가 아닙니다")
    if evidence.get("rule") != SERVING_FINANCIAL_RULE:
        raise ValueError("parity evidence의 재무 선택 규칙이 서빙 규칙과 다릅니다")
    if evidence.get("financial_code_hash") != financial_code_hash():
        raise ValueError("parity evidence가 현재 재무 코드와 다른 코드로 만들어졌습니다")
    failures = us_parity_failures(evidence)
    if failures:
        raise ValueError(f"parity evidence가 gate 기준을 넘지 못했습니다: {failures}")
    return evidence


def prepare_daily_features(scoring_date: date, *, diagnostic_only: bool = False,
                           raw_feature_parity_status: str = "unverified",
                           raw_feature_parity_evidence: str | None = None) -> Path:
    """Build an immutable native artifact keyed only by A and its input/code identity."""
    if raw_feature_parity_status not in {"passed", US_PARITY_SCORE_EQUIVALENT, "failed", "unverified"}:
        raise ValueError("invalid raw feature parity status")
    if raw_feature_parity_status == US_PARITY_SCORE_EQUIVALENT:
        if diagnostic_only:
            raise ValueError("score_equivalent parity cannot be combined with diagnostic_only")
        if not raw_feature_parity_evidence:
            raise ValueError("score_equivalent parity requires evidence")
        validate_parity_evidence(Path(raw_feature_parity_evidence))
    if raw_feature_parity_status == "failed" and not diagnostic_only:
        raise ValueError("failed raw feature parity requires diagnostic_only")
    if raw_feature_parity_status == "failed" and not raw_feature_parity_evidence:
        raise ValueError("failed raw feature parity requires evidence")
    parity_evidence_sha256 = (
        sha256_file(Path(raw_feature_parity_evidence)) if raw_feature_parity_evidence else None
    )
    serving_eligible = not diagnostic_only and raw_feature_parity_status in {
        "passed", US_PARITY_SCORE_EQUIVALENT
    }
    financial_rule = SERVING_FINANCIAL_RULE
    fin_code_hash = financial_code_hash()
    prepare_started_at = datetime.now().astimezone()
    root = DataRoot.resolve(market="us")
    live_lake = UsLake(root)
    revisions = collect_source_revision(live_lake)
    source_snapshot_files = fingerprint_source_snapshots(root, revisions)
    universe_completion_sha256 = verify_universe_completion(
        root, revisions["universe_daily"], revisions)
    lake = PinnedUsLake(root, revisions)
    features = build_daily_features(lake, scoring_date, financial_selection_rule=financial_rule)
    if fingerprint_source_snapshots(root, revisions) != source_snapshot_files:
        raise ValueError("US source snapshot changed during daily feature preparation")
    if raw_feature_parity_evidence and sha256_file(Path(raw_feature_parity_evidence)) != parity_evidence_sha256:
        raise ValueError("US raw feature parity evidence changed during preparation")
    feature_hash = _canonical_hash(features.hash_rows(seed=0).to_list())
    prep_hash = _canonical_hash(
        {
            "scoring_version": SCORING_VERSION,
            "scoring_date": scoring_date,
            "source_revision": revisions,
            "source_snapshot_files": source_snapshot_files,
            "universe_completion_sha256": universe_completion_sha256,
            "diagnostic_only": diagnostic_only,
            "raw_feature_parity_status": raw_feature_parity_status,
            "raw_feature_parity_evidence": raw_feature_parity_evidence,
            "raw_feature_parity_evidence_sha256": parity_evidence_sha256,
            "serving_eligible": serving_eligible,
            "financial_selection_rule": financial_rule,
            "financial_code_hash": fin_code_hash,
            "feature_hash": feature_hash,
            "code_hash": code_tree_hash(),
        }
    )
    prep_dir = root.output / SCORING_VERSION / "prepared" / f"score_date={scoring_date}" / f"prep_id={prep_hash[:16]}"
    manifest_path = prep_dir / "manifest.json"
    feature_path = prep_dir / "features.parquet"
    if prep_dir.exists():
        if not manifest_path.is_file() or not feature_path.is_file():
            raise FileExistsError(f"같은 prep_id의 결과가 달라 덮지 않습니다: {prep_dir}")
        native = json.loads(manifest_path.read_text())
        if (
            native.get("prep_hash") != prep_hash
            or native.get("features_sha256") != sha256_file(feature_path)
        ):
            raise FileExistsError(f"같은 prep_id의 결과가 달라 덮지 않습니다: {prep_dir}")
        _read_native_completion(manifest_path)
        return manifest_path
    prep_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary_dir = Path(tempfile.mkdtemp(prefix=f".{prep_hash[:16]}.", dir=prep_dir.parent))
    features.write_parquet(temporary_dir / "features.parquet")
    features_sha256 = sha256_file(temporary_dir / "features.parquet")
    manifest = {
        "schema_version": 1,
        "market": "US",
        "scoring_version": SCORING_VERSION,
        "scoring_date": scoring_date.isoformat(),
        "feature_asof_date": scoring_date.isoformat(),
        "prepare_started_at": prepare_started_at.isoformat(),
        "feature_hash": feature_hash,
        "input_sha256": features_sha256,
        "source_revision": revisions,
        "source_snapshot_files": source_snapshot_files,
        "universe_completion_sha256": universe_completion_sha256,
        "diagnostic_only": diagnostic_only,
        "raw_feature_parity_status": raw_feature_parity_status,
        "raw_feature_parity_evidence": raw_feature_parity_evidence,
        "raw_feature_parity_evidence_sha256": parity_evidence_sha256,
        "serving_eligible": serving_eligible,
        "financial_selection_rule": financial_rule,
        "financial_code_hash": fin_code_hash,
        "code_hash": code_tree_hash(),
        "prep_hash": prep_hash,
        "prep_id": prep_hash[:16],
        "row_count": features.height,
        "features_sha256": features_sha256,
        "availability_evidence_type": "prepared_features_completion",
        "freshness_status": "freshness_unverified",
        "labels_read": False,
    }
    (temporary_dir / "manifest.json").write_text(
        json.dumps(manifest, sort_keys=True, indent=2) + "\n"
    )
    _publish_native_directory(temporary_dir, prep_dir)
    return manifest_path


def create_daily_selection_manifest(
    *,
    native_manifest_path: Path,
    report_date: date,
    calendar_manifest: dict[str, Any],
    latest_us_session: date | None,
    expected_us_session: date | None,
    max_us_market_lag: int | None,
    source_first_available_at: datetime | None = None,
) -> Path:
    """Pin a D-specific 09:30 selection to a reusable A-specific native artifact."""
    native = json.loads(native_manifest_path.read_text())
    block = us_native_block_reason(native)
    if block is not None:
        raise ValueError(f"diagnostic US native features cannot be selected for serving: {block}")
    if native.get("market") != "US" or native.get("scoring_version") != SCORING_VERSION:
        raise ValueError("native prepared manifest의 market 또는 scoring version이 다릅니다")
    if (
        native.get("feature_asof_date") != native.get("scoring_date")
        or native.get("availability_evidence_type") != "prepared_features_completion"
    ):
        raise ValueError("native prepared manifest의 feature 날짜 또는 가용 증거가 다릅니다")
    if native.get("code_hash") != code_tree_hash():
        raise ValueError("native prepared artifact의 코드 hash가 현재 코드와 다릅니다")
    input_cutoff = input_cutoff_for_report_date(report_date)
    decision_at = datetime.combine(report_date, time(10), _SEOUL_TZ)
    completion = _read_native_completion(native_manifest_path, input_cutoff=input_cutoff)
    native_features_sha256 = completion["features_sha256"]
    native_manifest_sha256 = completion["native_prepare_manifest_sha256"]
    verified_available_by = completion["verified_available_by"]
    completion_marker_path = completion["marker_path"]
    completion_marker_sha256 = completion["marker_sha256"]
    calendar = SessionCalendar.from_manifest(calendar_manifest)
    actual_us_session = date.fromisoformat(native["scoring_date"])
    freshness = assess_freshness(
        report_date=report_date,
        market="US",
        feature_asof_date=actual_us_session,
        decision_at=decision_at,
        input_cutoff=input_cutoff,
        calendar=calendar,
        latest_us_session=latest_us_session,
        expected_us_session=expected_us_session,
        actual_us_session=actual_us_session,
        source_first_available_at=source_first_available_at,
        verified_available_by=verified_available_by,
        availability_evidence_type="prepared_features_completion",
        availability_evidence={
            "features_sha256": native_features_sha256,
            "native_prepare_manifest_sha256": native_manifest_sha256,
            "completion_marker_sha256": completion_marker_sha256,
        },
        max_us_market_lag=max_us_market_lag,
    )
    completed_at = datetime.now(_SEOUL_TZ)
    validate_selection_completion(completed_at=completed_at, decision_at=decision_at)
    root = DataRoot.resolve(market="us")
    selection_dir = root.output / SCORING_VERSION / "selections" / f"report_date={report_date}"
    selection_dir.mkdir(parents=True, exist_ok=True)
    selection_path = selection_dir / "selection.json"
    payload = {
        "schema_version": 1,
        "market": "US",
        "report_date": report_date.isoformat(),
        "decision_at": decision_at.isoformat(),
        "input_cutoff": input_cutoff.isoformat(),
        "calendar": calendar_manifest,
        "native_prepare_manifest_path": str(native_manifest_path.resolve()),
        "native_prepare_manifest_sha256": native_manifest_sha256,
        "completion_marker_path": completion_marker_path.name,
        "completion_marker_sha256": completion_marker_sha256,
        "native_features_sha256": native_features_sha256,
        "input_sha256": native_features_sha256,
        "feature_asof_date": actual_us_session.isoformat(),
        "latest_us_session": latest_us_session.isoformat() if latest_us_session else None,
        "expected_us_session": expected_us_session.isoformat() if expected_us_session else None,
        "actual_us_session": actual_us_session.isoformat(),
        "source_first_available_at": source_first_available_at.isoformat()
        if source_first_available_at else None,
        "verified_available_by": verified_available_by.isoformat(),
        "availability_evidence_type": "prepared_features_completion",
        "availability_evidence": {
            "features_sha256": native_features_sha256,
            "native_prepare_manifest_sha256": native_manifest_sha256,
            "completion_marker_sha256": completion_marker_sha256,
        },
        "completed_at": completed_at.isoformat(),
        "freshness_status": freshness.status,
        "freshness": freshness.as_dict(),
        "delivery_lag_sessions": freshness.delivery_lag,
        "market_lag_sessions": freshness.market_lag,
        "market_lag_limit_sessions": max_us_market_lag,
    }
    identity_payload = {
        key: value for key, value in payload.items()
        if key not in {"completed_at", "verified_available_by"}
    }
    canonical = json.dumps(identity_payload, sort_keys=True, separators=(",", ":"))
    selection_hash = hashlib.sha256(canonical.encode()).hexdigest()
    payload["selection_hash"] = selection_hash
    if selection_path.exists():
        old = json.loads(selection_path.read_text())
        if old.get("selection_hash") != selection_hash:
            raise FileExistsError(f"같은 report_date selection이 달라 덮지 않습니다: {selection_path}")
        return selection_path
    tmp = selection_path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n")
    os.rename(tmp, selection_path)
    return selection_path


def infer_prepared_features(selection_path: Path) -> dict[str, Path]:
    """Infer from a D selection that references one immutable native A artifact."""
    selection = json.loads(selection_path.read_text())
    if selection.get("market") != "US":
        raise ValueError("selection manifest의 market이 US가 아닙니다")
    identity_payload = {
        key: value for key, value in selection.items()
        if key not in {"selection_hash", "completed_at", "verified_available_by"}
    }
    expected_selection_hash = _canonical_hash(identity_payload)
    if selection.get("selection_hash") != expected_selection_hash:
        raise ValueError("selection manifest identity hash가 다릅니다")
    native_path = Path(selection["native_prepare_manifest_path"])
    if sha256_file(native_path) != selection.get("native_prepare_manifest_sha256"):
        raise ValueError("native prepared manifest hash가 selection과 다릅니다")
    native = json.loads(native_path.read_text())
    block = us_native_block_reason(native)
    if block is not None:
        raise ValueError(f"diagnostic US native features cannot be inferred for serving: {block}")
    if (
        native.get("market") != "US"
        or native.get("scoring_version") != SCORING_VERSION
        or native.get("code_hash") != code_tree_hash()
    ):
        raise ValueError("native prepared manifest의 version 또는 코드 hash가 다릅니다")
    if (
        selection.get("actual_us_session") != native.get("scoring_date")
        or selection.get("actual_us_session") != native.get("feature_asof_date")
        or selection.get("feature_asof_date") != native.get("feature_asof_date")
        or native.get("availability_evidence_type")
        != selection.get("availability_evidence_type")
    ):
        raise ValueError("selection의 A 날짜 또는 가용 증거가 native manifest와 다릅니다")
    native_features_path = native_path.parent / "features.parquet"
    verified_by = selection.get("verified_available_by")
    evidence = selection.get("availability_evidence")
    native_sha = selection.get("native_prepare_manifest_sha256")
    input_sha = selection.get("input_sha256")
    marker_rel = selection.get("completion_marker_path")
    if marker_rel != "completion.json":
        raise ValueError("selection completion marker 경로가 native artifact 경계를 벗어났습니다")
    input_cutoff = datetime.fromisoformat(selection["input_cutoff"])
    completion = _read_native_completion(
        native_path,
        expected_marker_sha256=selection.get("completion_marker_sha256"),
        input_cutoff=input_cutoff,
    )
    if (
        not isinstance(evidence, dict)
        or evidence.get("features_sha256") != input_sha
        or evidence.get("native_prepare_manifest_sha256") != native_sha
        or evidence.get("completion_marker_sha256") != selection.get("completion_marker_sha256")
        or input_sha != native.get("features_sha256")
        or input_sha != native.get("input_sha256")
        or sha256_file(native_features_path) != input_sha
        or verified_by != completion["verified_available_by"].isoformat()
    ):
        raise ValueError("selection의 prepared-feature 가용 증거가 native marker와 다릅니다")
    report_date = date.fromisoformat(selection["report_date"])
    decision_at = datetime.fromisoformat(selection["decision_at"])
    input_cutoff = datetime.fromisoformat(selection["input_cutoff"])
    completed_at = datetime.fromisoformat(selection["completed_at"])
    validate_selection_completion(completed_at=completed_at, decision_at=decision_at)
    actual = date.fromisoformat(selection["actual_us_session"])
    calendar = SessionCalendar.from_manifest(selection["calendar"])
    freshness = assess_freshness(
        report_date=report_date,
        market="US",
        feature_asof_date=date.fromisoformat(selection["feature_asof_date"]),
        decision_at=decision_at,
        input_cutoff=input_cutoff,
        calendar=calendar,
        latest_us_session=(date.fromisoformat(selection["latest_us_session"])
                           if selection.get("latest_us_session") else None),
        expected_us_session=(date.fromisoformat(selection["expected_us_session"])
                             if selection.get("expected_us_session") else None),
        actual_us_session=actual,
        source_first_available_at=(
            datetime.fromisoformat(selection["source_first_available_at"])
            if selection.get("source_first_available_at") else None
        ),
        verified_available_by=datetime.fromisoformat(selection["verified_available_by"]),
        availability_evidence_type=selection["availability_evidence_type"],
        availability_evidence=selection["availability_evidence"],
        max_us_market_lag=selection.get("market_lag_limit_sessions"),
    )
    if freshness.status != selection.get("freshness_status"):
        raise ValueError("selection freshness status가 재계산 결과와 다릅니다")
    if freshness.status not in {"ok", "stale"}:
        raise ValueError(f"freshness가 {freshness.status}라 inference를 중단합니다")
    feature_path = native_features_path
    if sha256_file(feature_path) != selection.get("native_features_sha256"):
        raise ValueError("native prepared features hash가 selection과 다릅니다")
    features = pl.read_parquet(feature_path)
    scoring_date = date.fromisoformat(native["scoring_date"])
    root = DataRoot.resolve(market="us")
    models, manifests = model_paths(root)
    return write_daily_run(
        root=root,
        scoring_date=scoring_date,
        features=features,
        model_paths=models,
        model_manifests=manifests,
        source_revision=native["source_revision"],
        code_hash=native["code_hash"],
        report_date=report_date,
        decision_at=decision_at,
        prepared_available_at=selection["verified_available_by"],
        input_cutoff=selection["input_cutoff"],
        freshness_status=freshness.status,
        selection_manifest_sha256=sha256_file(selection_path),
        freshness_metadata=selection,
    )


# --- 재무 feature 점수 수준 동등성 gate --------------------------------------------------
PARITY_SHUFFLE_SEEDS = (1, 2)
_FROZEN_CLOSE_TOLERANCE = {"rtol": 1e-9, "atol": 1e-12}


class _FactsLake:
    """``fundamentals`` held in memory so the gate controls fact row order.

    Other tables (``corp_actions`` for payout) come from ``fallback``."""

    def __init__(self, facts: pl.DataFrame, fallback: UsLake):
        self.facts = facts
        self.fallback = fallback

    def scan(self, table: str) -> pl.LazyFrame:
        if table == "fundamentals":
            return self.facts.lazy()
        return self.fallback.scan(table)


def _financial_frame(panel: pl.DataFrame, lake: Any, rule: str) -> pl.DataFrame:
    """The rule-governed features (financial 11 + buyback_yield), keyed by (date, symbol)."""
    frame: pl.DataFrame | None = None
    for add_family in (add_valuation, add_profitability, add_investment, add_payout):
        part = add_family(panel, lake, selection_rule=rule)
        part = part.select("date", "symbol", *[c for c in RULE_GOVERNED_FEATURES if c in part.columns])
        if frame is None:
            frame = part
        else:
            extra = [c for c in part.columns if c not in frame.columns]
            frame = frame.join(part.select("date", "symbol", *extra), on=["date", "symbol"], how="left")
    assert frame is not None
    return frame.sort("date", "symbol")


def _count_cell_diffs(
    actual: pl.DataFrame, expected: pl.DataFrame, columns: tuple[str, ...], *, exact: bool
) -> tuple[bool, dict[str, int]]:
    """Cell differences over ``columns`` for two frames with the same (date, symbol) keys."""
    a = actual.sort("date", "symbol")
    e = expected.sort("date", "symbol")
    keys_equal = a.select("date", "symbol").rows() == e.select("date", "symbol").rows()
    if not keys_equal:
        return False, {}
    diffs: dict[str, int] = {}
    for name in columns:
        x, y = a[name].cast(pl.Float64), e[name].cast(pl.Float64)
        null_mismatch = x.is_null().to_numpy() != y.is_null().to_numpy()
        xv, yv = x.to_numpy(), y.to_numpy()
        if exact:
            same = (xv == yv) | (np.isnan(xv) & np.isnan(yv))
        else:
            same = np.isclose(xv, yv, equal_nan=True, **_FROZEN_CLOSE_TOLERANCE)
        diffs[name] = int((~same | null_mismatch).sum())
    return True, diffs


def _rank_shift_metrics(frozen_ranks: pl.DataFrame, new_ranks: pl.DataFrame) -> dict[str, Any]:
    joined = frozen_ranks.select("date", "symbol", pl.col("rank").alias("r0")).join(
        new_ranks.select("date", "symbol", pl.col("rank").alias("r1")),
        on=["date", "symbol"], how="inner",
    )
    keys_equal = joined.height == frozen_ranks.height == new_ranks.height
    per_date: list[dict[str, Any]] = []
    shifts: list[np.ndarray] = []
    frozen_top100_max = 0
    for day in sorted(joined["date"].unique().to_list()):
        part = joined.filter(pl.col("date") == day)
        r0 = part["r0"].cast(pl.Int64).to_numpy()
        r1 = part["r1"].cast(pl.Int64).to_numpy()
        symbols = part["symbol"].to_numpy()
        spearman = float(np.corrcoef(r0, r1)[0, 1])
        overlap = {
            k: len(set(symbols[r0 <= k]) & set(symbols[r1 <= k])) / k for k in (50, 100)
        }
        shift = np.abs(r1 - r0)
        shifts.append(shift)
        top100_max = int(shift[r0 <= 100].max()) if (r0 <= 100).any() else 0
        frozen_top100_max = max(frozen_top100_max, top100_max)
        per_date.append({
            "date": day.isoformat(), "rows": int(len(r0)), "spearman": spearman,
            "top50_overlap": overlap[50], "top100_overlap": overlap[100],
            "changed_rows": int((shift > 0).sum()), "max_abs_rank_shift": int(shift.max()),
            "frozen_top100_max_abs_rank_shift": top100_max,
        })
    pooled = np.concatenate(shifts) if shifts else np.array([0])
    return {
        "keys_equal": bool(keys_equal),
        "rows": int(joined.height),
        "changed_rows": int((pooled > 0).sum()),
        "max_abs_rank_shift": int(pooled.max()),
        "p99_abs_rank_shift": int(np.percentile(pooled, 99, method="higher")),
        "min_daily_spearman": min(d["spearman"] for d in per_date) if per_date else float("nan"),
        "min_top50_overlap": min(d["top50_overlap"] for d in per_date) if per_date else 0.0,
        "min_top100_overlap": min(d["top100_overlap"] for d in per_date) if per_date else 0.0,
        "frozen_top100_max_abs_rank_shift": frozen_top100_max,
        "per_date": per_date,
    }


def _sha_or_none(path: Path) -> str | None:
    return sha256_file(path) if path.is_file() else None


def run_parity_check(
    *, lake_root: Path, frozen_root: Path, bundle_root: Path, rule: str, output: Path
) -> dict[str, Any]:
    """Score-level equivalence gate for the 11 financial features (label-free, read-only)."""
    check_rule(rule)
    if output.exists():
        raise FileExistsError(f"parity evidence 경로가 이미 있어 덮지 않습니다: {output}")
    base_manifest_path = frozen_root / "us_features_v2" / "manifest.json"
    flow_manifest_path = frozen_root / "us_features_flow_v1" / "manifest.json"
    base_map = dict(json.loads(base_manifest_path.read_text())["input_table_snapshots"])
    flow_map = dict(json.loads(flow_manifest_path.read_text())["input_table_snapshots"])
    # 모델 bundle은 오래 걸리는 재생성 전에 먼저 검증한다(golden 재로딩 포함).
    loaded = {
        name: load_model(bundle_root / name / "model.joblib", bundle_root / name / "manifest.json")
        for name in US_PARITY_MODELS
    }
    data_root = DataRoot(lake_root)
    base_lake = PinnedUsLake(data_root, base_map)
    flow_lake = PinnedUsLake(data_root, flow_map)
    flow_only_map = {t: d for t, d in flow_map.items() if base_map.get(t) != d}
    source_files = {
        "base": fingerprint_source_snapshots(data_root, base_map),
        "flow_only": fingerprint_source_snapshots(data_root, flow_only_map),
    }

    # 1) six golden sessions rebuilt from the frozen source map with the requested rule.
    panels = [build_daily_panel(base_lake, day) for day in sorted(GOLDEN_DATES)]
    features = pl.concat(panels).sort(["date", "symbol"])
    features = _add_base_families(features, base_lake, rule)
    for _name, add_family in FLOW_FAMILIES:
        features = add_family(features, flow_lake)
    features = features.sort(["date", "symbol"])
    frozen, frozen_hashes = read_frozen_golden_features(frozen_root)

    non_financial = tuple(name for name in FEATURES if name not in FINANCIAL_FEATURES)
    keys_ok, nf_diffs = _count_cell_diffs(features, frozen, non_financial, exact=False)
    eligible_equal = keys_ok and (
        features.sort("date", "symbol")["price_ge_5"].to_list()
        == frozen.sort("date", "symbol")["price_ge_5"].to_list()
    )
    _fin_ok, fin_vs_frozen = _count_cell_diffs(features, frozen, FINANCIAL_FEATURES, exact=False)
    fin_ref = features.select("date", "symbol", *RULE_GOVERNED_FEATURES).sort("date", "symbol")
    golden_keys = fin_ref.select("date", "symbol")

    # 2) determinism: shuffled fact order x two panel ranges, rule-governed 12 features.
    facts = (
        base_lake.scan("fundamentals")
        .filter(pl.col("tag").is_in(list(FINANCIAL_SOURCE_TAGS)))
        .select("cik", "tag", "fp", "start", "end", "val", "filed", "accn", "form", "unit")
        .collect()
    )
    frozen_panel = (
        pl.scan_parquet(frozen_root / "us_features_v2" / "part.parquet")
        .select("date", "symbol", "cik", "close")
        .collect()
    )
    frozen_dates = sorted(frozen_panel["date"].unique().to_list())
    wide_dates = set(GOLDEN_DATES)
    for day in GOLDEN_DATES:
        later = [d for d in frozen_dates if d > day]
        if later:
            wide_dates.add(later[0])
    panel_small = features.select("date", "symbol", "cik", "close")
    panel_wide = frozen_panel.filter(pl.col("date").is_in(sorted(wide_dates))).with_columns(
        pl.col("cik").cast(pl.Int64), pl.col("close").cast(pl.Float64)
    )
    runs: list[dict[str, Any]] = []
    for label, panel in (("panel6", panel_small), (f"panel{len(wide_dates)}", panel_wide)):
        for seed in PARITY_SHUFFLE_SEEDS:
            shuffled = facts.sample(fraction=1.0, shuffle=True, seed=seed)
            got = _financial_frame(panel, _FactsLake(shuffled, base_lake), rule)
            got = got.join(golden_keys, on=["date", "symbol"], how="inner")
            same_keys, diffs = _count_cell_diffs(got, fin_ref, RULE_GOVERNED_FEATURES, exact=True)
            runs.append({
                "name": f"{label}_shuffle{seed}", "panel_dates": panel["date"].n_unique(),
                "panel_rows": panel.height, "shuffle_seed": seed, "keys_equal": same_keys,
                "diff_cells": sum(diffs.values()), "diff_by_feature": diffs,
            })
            del shuffled, got
    del facts

    # 3) score equivalence on both unchanged models.
    models: dict[str, Any] = {}
    bundle_info: dict[str, Any] = {}
    bundle_matches_frozen = True
    for name in US_PARITY_MODELS:
        bundle, manifest = loaded[name]
        golden = pl.read_parquet(bundle_root / name / "golden_features.parquet")
        frozen_ranks = _rank_golden(bundle["model"], golden)
        new_ranks = _rank_golden(bundle["model"], features)
        models[name] = _rank_shift_metrics(frozen_ranks, new_ranks)
        bundle_info[name] = {
            key: manifest.get(key)
            for key in ("model_id", "model_version", "variant", "model_sha256",
                        "golden_features_sha256", "golden_predictions_sha256")
        }
        bundle_matches_frozen = bundle_matches_frozen and (
            dict(manifest.get("golden_source_sha256", {})) == frozen_hashes
        )

    evidence: dict[str, Any] = {
        "schema_version": US_PARITY_EVIDENCE_SCHEMA,
        "rule": rule,
        "financial_features": list(FINANCIAL_FEATURES),
        "rule_governed_features": list(RULE_GOVERNED_FEATURES),
        "criteria": dict(US_PARITY_CRITERIA),
        "created_at": datetime.now().astimezone().isoformat(),
        "labels_read": False,
        "code_hash": code_tree_hash(),
        "financial_code_hash": financial_code_hash(),
        "inputs": {
            "base_source_map": base_map,
            "flow_source_map": flow_map,
            "source_files": source_files,
            "frozen_manifest_sha256": {
                "us_features_v2": sha256_file(base_manifest_path),
                "us_features_flow_v1": sha256_file(flow_manifest_path),
            },
            "frozen_feature_sha256": frozen_hashes,
            "bundles": bundle_info,
            "bundle_matches_frozen": bundle_matches_frozen,
            "golden_dates": [d.isoformat() for d in GOLDEN_DATES],
            "shuffle_seeds": list(PARITY_SHUFFLE_SEEDS),
        },
        "determinism": {"runs": runs},
        "non_financial": {
            "keys_equal": keys_ok, "eligible_equal": eligible_equal,
            "features": len(non_financial), "diff_cells": sum(nf_diffs.values()),
            "diff_by_feature": nf_diffs,
        },
        "financial_vs_frozen_diff_cells": fin_vs_frozen,
        "models": models,
    }
    failures = us_parity_failures(evidence)
    if not bundle_matches_frozen:
        failures.append("bundle golden source differs from frozen inputs")
    evidence["failures"] = failures
    evidence["status"] = "failed" if failures else US_PARITY_SCORE_EQUIVALENT
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.with_suffix(output.suffix + ".tmp")
    tmp.write_text(json.dumps(evidence, indent=2, sort_keys=True, default=str) + "\n")
    os.rename(tmp, output)
    return evidence


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser("prepare", help="일일 derived 이후 daily feature를 미리 저장")
    prepare.add_argument("--as-of", required=True, help="완료된 XNYS 세션 (YYYY-MM-DD)")
    prepare.add_argument("--diagnostic-only", action="store_true", help="운영 선택을 금지한 진단 준비")
    prepare.add_argument("--raw-feature-parity-evidence", help="failed 또는 score_equivalent parity 근거 JSON 경로")
    prepare.add_argument(
        "--raw-feature-parity-status", choices=[US_PARITY_SCORE_EQUIVALENT],
        help="gate를 통과한 evidence가 있을 때만 score_equivalent로 서빙 가능한 준비를 만듭니다",
    )
    select = subparsers.add_parser("select", help="D 09:30에 native US A를 선택해 freshness 고정")
    select.add_argument("--prepared-manifest", required=True)
    select.add_argument("--report-date", required=True)
    select.add_argument("--calendar-json", required=True)
    select.add_argument("--latest-us-session")
    select.add_argument("--expected-us-session")
    select.add_argument("--max-us-market-lag", type=int)
    select.add_argument("--source-first-available-at")
    infer = subparsers.add_parser("infer", help="D별 선택 manifest로 두 모델을 순차 추론")
    infer.add_argument("--selection-manifest", required=True)
    parity = subparsers.add_parser("parity-check", help="재무 11개 점수 수준 동등성 gate (label 없음)")
    parity.add_argument("--rule", required=True, choices=["legacy", SERVING_FINANCIAL_RULE])
    parity.add_argument("--output", required=True)
    parity.add_argument("--lake-root", required=True, help="US lake 루트 (derived/snapshots 상위)")
    parity.add_argument("--frozen-root", required=True)
    parity.add_argument("--bundle-root", required=True, help="lightgbm/ridge bundle 상위 경로")
    bundle = subparsers.add_parser("bundle", help="원본 모델과 frozen feature golden을 감싼 bundle 생성")
    bundle.add_argument("--frozen-root", required=True)
    bundle.add_argument("--source-model-root", required=True)
    bundle.add_argument("--bundle-root", required=True)
    bundle.add_argument("--model-version", required=True)
    args = parser.parse_args(argv)
    if args.command == "prepare":
        score_equivalent = args.raw_feature_parity_status == US_PARITY_SCORE_EQUIVALENT
        if score_equivalent and (args.diagnostic_only or not args.raw_feature_parity_evidence):
            parser.error("score_equivalent requires evidence and forbids --diagnostic-only")
        if args.raw_feature_parity_evidence and not args.diagnostic_only and not score_equivalent:
            parser.error("--raw-feature-parity-evidence requires --diagnostic-only")
        if score_equivalent:
            status = US_PARITY_SCORE_EQUIVALENT
        else:
            status = "failed" if args.raw_feature_parity_evidence else "unverified"
        manifest = prepare_daily_features(
            date.fromisoformat(args.as_of),
            diagnostic_only=args.diagnostic_only,
            raw_feature_parity_status=status,
            raw_feature_parity_evidence=args.raw_feature_parity_evidence,
        )
        print(f"prepared: {manifest}")
        return 0
    if args.command == "select":
        path = create_daily_selection_manifest(
            native_manifest_path=Path(args.prepared_manifest),
            report_date=date.fromisoformat(args.report_date),
            calendar_manifest=json.loads(Path(args.calendar_json).read_text()),
            latest_us_session=date.fromisoformat(args.latest_us_session)
            if args.latest_us_session else None,
            expected_us_session=date.fromisoformat(args.expected_us_session)
            if args.expected_us_session else None,
            max_us_market_lag=args.max_us_market_lag,
            source_first_available_at=datetime.fromisoformat(args.source_first_available_at)
            if args.source_first_available_at else None,
        )
        print(f"selection: {path}")
        return 0
    if args.command == "parity-check":
        evidence = run_parity_check(
            lake_root=Path(args.lake_root), frozen_root=Path(args.frozen_root),
            bundle_root=Path(args.bundle_root), rule=args.rule, output=Path(args.output),
        )
        print(f"parity-check: status={evidence['status']} rule={evidence['rule']} output={args.output}")
        if evidence["failures"]:
            print("failures: " + ", ".join(evidence["failures"]))
        return 0 if evidence["status"] == US_PARITY_SCORE_EQUIVALENT else 1
    if args.command == "bundle":
        source_root = Path(args.source_model_root)
        manifests = export_frozen_model_bundles(
            frozen_root=Path(args.frozen_root),
            source_models={
                name: source_root / name / "model.joblib" for name in ("lightgbm", "ridge")
            },
            bundle_root=Path(args.bundle_root),
            model_version=args.model_version,
        )
        for model_name, manifest in sorted(manifests.items()):
            print(f"{model_name}: model_sha256={manifest['model_sha256']}")
        return 0
    outputs = infer_prepared_features(Path(args.selection_manifest))
    for model_name, output in sorted(outputs.items()):
        print(f"{model_name}: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
