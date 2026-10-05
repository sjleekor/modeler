"""시장 리포트 markdown 시험용 합성 fixture를 만듭니다.

운영 코드의 envelope 구조(orchestration.combine_reports·adapters·kr_serving)를 따라갑니다.
`modeler.serving.schema`의 report_template과 validate_report를 그대로 씁니다.
종목 코드와 이름, 점수는 합성 값이라 실제 시장과 무관합니다.
"""

from __future__ import annotations

import hashlib
import json
import random
from datetime import datetime, timedelta, timezone
from pathlib import Path

from modeler.serving import schema as OPS

KST = timezone(timedelta(hours=9))

KR_ID, KR_VER = "kr_daily_h20_v1", "1"
LGB_ID = "us_exploratory_20260929_r1_lightgbm"
RDG_ID = "us_exploratory_20260929_r1_ridge"


def sha(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


def decision_at(day: str) -> datetime:
    y, m, d = (int(x) for x in day.split("-"))
    return datetime(y, m, d, 10, 0, tzinfo=KST)


def template(market: str, day: str, asof, model_id: str, version: str) -> dict:
    return OPS.report_template(
        market=market,
        report_date=day,
        decision_at=decision_at(day),
        feature_asof_date=asof,
        model_id=model_id,
        model_version=version,
    )


def freshness(
    day: str,
    market: str,
    asof,
    status: str,
    *,
    delivery=0,
    market_lag=0,
    u=None,
    e=None,
    a=None,
    reason=None,
    verified="05:02:11",
) -> dict:
    return {
        "status": status,
        "report_date": day,
        "feature_asof_date": asof,
        "latest_us_session": u,
        "expected_us_session": e,
        "actual_us_session": a,
        "input_cutoff": day + "T09:30:00+09:00",
        "verified_available_by": day + "T" + verified + "+09:00",
        "delivery_lag": delivery,
        "market_lag": market_lag,
        "reason": reason,
        "availability_evidence_type": "prepared_features_completion",
        "source_first_available_at": None,
        "availability_evidence": {
            "features_sha256": sha("f" + day + market),
            "native_prepare_manifest_sha256": sha("n" + day + market),
            "completion_marker_sha256": sha("c" + day + market),
        },
    }


def kr_report(
    day: str,
    asof: str,
    *,
    rows: int = 120,
    lag: int = 0,
    status: str = "partial",
    fresh_status: str = "ok",
) -> dict:
    report = template("KR", day, asof, KR_ID, KR_VER)
    rng = random.Random(day + "kr")
    scores = sorted((rng.uniform(0.2, 0.8) for _ in range(rows)), reverse=True)
    rankings = []
    reasons_at = {
        3: ["K_halt_or_unknown"],
        17: ["K_price_jump_or_unknown", "K_share_change_or_unknown"],
        42: ["K_price_quality_unavailable"],
    }
    for i, score in enumerate(scores, start=1):
        name = f"샘플기업{i:03d}"
        if i == 5:
            name = "샘플|기업\n005"  # 표 셀 이스케이프 시험: 파이프와 줄바꿈
        if i == 6:
            name = "샘플[링크](x)*기업"  # 링크·강조 문자 이스케이프 시험
        row = {
            "rank": i,
            "symbol": f"{100000 + i:06d}",
            "market": "KOSPI" if i % 2 else "KOSDAQ",
            "name": name,
            "score": round(score, 6),
        }
        if i <= 100:
            reasons = reasons_at.get(i, [])
            row["quality_review"] = bool(reasons)
            row["quality_reasons"] = reasons
            if reasons:
                row["K_simple_return"] = -0.0123
        rankings.append(row)
    review = sum(1 for r in rankings[:100] if r.get("quality_review"))
    report.update(
        status=status,
        rankings=rankings,
        quality={
            "status": "review_required",
            "management_filter_available": False,
            "management_state": "unverified",
            "halted_rows_at_K": 2,
            "price_jump_review": "K_quality_mart_checked",
            "top100_quality_review_rows": review,
            "D_management_and_halt_state": "unverified",
            "eligible_rows_scored": rows,
            "excluded_balance_features": ["bs_a", "bs_b"],
            "freshness_status": fresh_status,
        },
        provenance={
            "bundle_manifest_sha256": sha("kr-bundle"),
            "prepared_manifest_sha256": sha("kr-prep" + day),
            "feature_snapshot_date": day,
            "source_model_run_id": "kr_run_20260818",
            "target": "y_up_20d; p_raw ranking score, uncalibrated",
            "label_used_for_inference": False,
            "bundle_manifest_content_sha256": sha("kr-bundle-content"),
            "native_prepare_manifest_content_sha256": sha("kr-native-content" + day),
            "input_sha256": sha("kr-input" + day),
            "native_prepare_manifest_sha256": sha("kr-native" + day),
            "code_inventory_sha256": sha("kr-code"),
            "freshness": freshness(day, "KR", asof, fresh_status, delivery=lag, market_lag=lag),
        },
        inference_started_at=day + "T01:00:09+00:00",
        synthetic_fixture=False,
    )
    return report


def us_report(
    day: str,
    model_id: str,
    a: str,
    u: str,
    e: str,
    *,
    rows: int = 150,
    lag: int = 0,
    status: str = "ok",
    overlap: int = 70,
) -> dict:
    report = template("US", day, a, model_id, "1")
    rng = random.Random(day + model_id)
    base = [f"TS{i:03d}" for i in range(1, rows + 1)]
    if model_id == RDG_ID:  # LightGBM 상위 100과 overlap개만 겹치게 만듭니다
        top = base[:overlap] + [f"TX{i:03d}" for i in range(1, 101 - overlap)]
        symbols = top + [s for s in base[overlap:] if s not in top]
        symbols = symbols[:rows]
    else:
        symbols = base
    scores = sorted((rng.uniform(-1.5, 2.5) for _ in range(len(symbols))), reverse=True)
    rankings = [
        {"rank": i, "symbol": s, "name": s, "score": round(v, 6)}
        for i, (s, v) in enumerate(zip(symbols, scores), start=1)
    ]
    fresh = freshness(
        day, "US", a, status, delivery=lag, market_lag=lag, u=u, e=e, a=a, verified="03:30:00"
    )
    report.update(
        status=status,
        rankings=rankings,
        quality={
            "freshness_status": status,
            "feature_count": 53,
            "design_column_count": 106,
            "eligible_rows": len(rankings),
            "monthly_membership_month": a[:7],
            "latest_us_session": u,
            "expected_us_session": e,
            "actual_us_session": a,
            "delivery_lag_sessions": lag,
            "market_lag_sessions": lag,
            "market_lag_limit_sessions": 2,
        },
        provenance={
            "scoring_version": "us-daily-1",
            "input_sha256": sha("us-input" + day),
            "bundle_manifest_sha256": sha("us-bundle" + model_id),
            "native_prepare_manifest_sha256": sha("us-native" + day),
            "code_inventory_sha256": sha("us-code"),
            "freshness": fresh,
            "score_semantics": "raw model ranking score; not a probability or confidence",
            "display_name_source": "symbol fallback; company-name mapping is not part of scoring",
            "labels_read": False,
        },
        inference_started_at=day + "T01:00:20+00:00",
        synthetic_fixture=False,
    )
    return report


def envelope(
    day: str, reports: list, failures: list, *, replay: bool = False, extra: dict | None = None
) -> dict:
    ok = any(r["status"] == "ok" for r in reports)
    if not reports and not failures:
        status = "failed"
    elif failures or any(
        r["status"] in {"failed", "stale", "withheld", "unavailable"} for r in reports
    ):
        status = "partial" if ok else "failed"
    elif any(r["status"] == "partial" for r in reports):
        status = "partial"
    else:
        status = "partial"  # opening 없음 -> partial (combine_reports와 같음)
    env = {
        "schema_version": "1.0",
        "report_date": day,
        "decision_at": decision_at(day).isoformat(),
        "status": status,
        "markets": sorted(reports, key=lambda r: (r["market"], r["model_id"])),
        "failures": failures,
        "opening": {
            "status": "unavailable",
            "publication": {"status": "unresolved", "evidence": []},
        },
        "synthetic_fixture": False,
        "historical_replay": replay,
    }
    if extra:
        env.update(extra)
    for r in reports:
        OPS.validate_report(r)
    return env


def ms_input(
    day: str,
    *,
    replay: bool = False,
    kr_asof: str | None = None,
    us_asof: str | None = None,
    macro_asof: str | None = None,
) -> dict:
    us_asof = us_asof or day
    kr_asof = kr_asof or day
    rng = random.Random("ms" + day)
    us = [
        ("us_spx", "SPY (S&P 500)", "market"),
        ("us_ndx", "QQQ (Nasdaq 100)", "market"),
        ("us_xlf", "XLF (금융)", "sector"),
        ("us_xlv", "XLV (헬스케어)", "sector"),
        ("us_xli", "XLI (산업재)", "sector"),
        ("us_xle", "XLE (에너지)", "sector"),
        ("us_xlk", "XLK (기술)", "sector"),
    ]
    kr = [
        ("kr_kospi", "코스피", "market"),
        ("kr_kosdaq", "코스닥", "market"),
        ("kr_bank", "KRX 은행", "sector"),
        ("kr_health", "KRX 헬스케어", "sector"),
        ("kr_mach", "KRX 기계장비", "sector"),
        ("kr_energy", "KRX 에너지화학", "sector"),
        ("kr_semi", "KRX 반도체", "sector"),
    ]
    assets = []
    for market, rows, basis, cash, asof in (
        ("US", us, "total_return", 0.042, us_asof),
        ("KR", kr, "price_only", 0.0345, kr_asof),
    ):
        for asset_id, name, group in rows:
            assets.append(
                {
                    "asset_id": asset_id,
                    "name": name,
                    "market": market,
                    "group": group,
                    "return_basis": basis,
                    "asof_date": asof,
                    "ret_1": round(rng.uniform(-0.02, 0.02), 4),
                    "ret_5": round(rng.uniform(-0.04, 0.04), 4),
                    "ret_20": round(rng.uniform(-0.08, 0.08), 4),
                    "ret_60": round(rng.uniform(-0.15, 0.15), 4),
                    "rvol_20": round(rng.uniform(0.08, 0.30), 4),
                    "dd_252": round(-rng.uniform(0.0, 0.2), 4),
                    "cash_rate": cash,
                    "b_opp_mean_pct": round(rng.uniform(-1.0, 2.0), 2),
                    "b_stab_prob_pct": round(rng.uniform(10, 30), 1),
                    "opportunity_score": round(rng.uniform(0, 100), 1),
                    "stability_score": round(rng.uniform(0, 100), 1),
                }
            )
    assets[0]["close"] = 5432.1  # 지수 종가 수준: 출력에 나오면 안 됩니다
    assets[7]["close"] = 2711.9
    return {
        "report_date": day,
        "historical_replay": replay,
        "asof": {"KR": kr_asof, "US": us_asof, "US_macro": macro_asof or "2026-09-27"},
        "assets": assets,
        "verdicts": {
            "US": {"opportunity": "실패", "sector_relative": "보류", "stability": "실패"},
            "KR": {"opportunity": "실패", "sector_relative": "실패", "stability": "실패"},
        },
        "provenance": {
            "ms_runs": ["ms_us_202609300940", "ms_kr_202609300941"],
            "config_hash": "b95fd065…",
            "calendar_basis": {"KR": "exchange_calendars==4.13.2"},
        },
        "notes": ["KR 지수는 K보다 1세션 늦음(T+1 공표)"],
    }


# --- 시나리오 4개 -----------------------------------------------------------------------------
def scenario_normal() -> tuple[dict, dict]:
    day = "2026-10-07"
    reports = [
        kr_report(day, "2026-10-06"),
        us_report(day, LGB_ID, "2026-10-06", "2026-10-06", "2026-10-06"),
        us_report(day, RDG_ID, "2026-10-06", "2026-10-06", "2026-10-06"),
    ]
    return envelope(day, reports, []), ms_input(
        day, kr_asof="2026-10-06", us_asof="2026-10-06", macro_asof="2026-10-04"
    )


def scenario_kr_unavailable_us_stale() -> tuple[dict, dict]:
    day = "2026-10-08"
    reports = [
        us_report(day, LGB_ID, "2026-10-02", "2026-10-07", "2026-10-07", lag=3, status="stale"),
        us_report(day, RDG_ID, "2026-10-02", "2026-10-07", "2026-10-07", lag=3, status="stale"),
    ]
    failures = [{"market": "KR", "model_id": KR_ID, "error_class": "ValueError"}]
    return envelope(day, reports, failures), ms_input(
        day, kr_asof="2026-10-06", us_asof="2026-10-07", macro_asof="2026-10-04"
    )


def scenario_replay() -> tuple[dict, dict]:
    day = "2026-09-30"
    reports = [
        kr_report(day, "2026-09-29"),
        us_report(day, LGB_ID, "2026-09-29", "2026-09-29", "2026-09-29"),
        us_report(day, RDG_ID, "2026-09-29", "2026-09-29", "2026-09-29"),
    ]
    replay = {
        "K": "2026-09-29",
        "A": "2026-09-29",
        "snapshot": {"kr_raw": "2026-10-06", "us_lake": "2026-10-06"},
        "executed_at": "2026-10-06T14:20:11+09:00",
        "skipped_time_checks": [
            "입력 완료 시각 <= D 09:30",
            "select 허용 창 D 09:30~10:00",
            "selection 불변 확인(D 이전 생성)",
        ],
    }
    env = envelope(day, reports, [], replay=True, extra={"replay": replay})
    return env, ms_input(
        day, replay=True, kr_asof="2026-09-29", us_asof="2026-09-29", macro_asof="2026-09-27"
    )


def scenario_all_failed() -> tuple[dict, None]:
    day = "2026-10-12"
    failures = [
        {"market": "KR", "model_id": KR_ID, "error_class": "MissingInference"},
        {"market": "US", "model_id": LGB_ID, "error_class": "TimeoutError"},
        {"market": "US", "model_id": RDG_ID, "error_class": "TimeoutError"},
    ]
    return envelope(day, [], failures), None


def scenario_lag_over_limit() -> tuple[dict, dict]:
    """KR·US 모두 5세션을 넘게 늦은 날. 순위 표 대신 사유와 마지막 정상 단위 링크가 나옵니다."""
    day = "2026-10-14"
    kr = kr_report(day, "2026-10-02", lag=6, status="stale", fresh_status="stale")
    kr["provenance"]["freshness"]["reason"] = "KR feature session does not match K"
    reports = [
        kr,
        us_report(day, LGB_ID, "2026-10-05", "2026-10-13", "2026-10-13", lag=6, status="stale"),
        us_report(day, RDG_ID, "2026-10-05", "2026-10-13", "2026-10-13", lag=6, status="stale"),
    ]
    return envelope(day, reports, []), ms_input(
        day, kr_asof="2026-10-12", us_asof="2026-10-13", macro_asof="2026-10-11"
    )


SCENARIOS = {
    "f1_normal": scenario_normal,
    "f2_kr_unavailable_us_stale": scenario_kr_unavailable_us_stale,
    "f3_replay": scenario_replay,
    "f4_all_failed": scenario_all_failed,
    "f5_lag_over_limit": scenario_lag_over_limit,
}


def write_all(out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    for name, builder in SCENARIOS.items():
        env, ms = builder()
        env_path = out_dir / (name + "_envelope.json")
        env_path.write_text(
            json.dumps(env, ensure_ascii=False, indent=1, sort_keys=True) + "\n", encoding="utf-8"
        )
        ms_path = None
        if ms is not None:
            ms_path = out_dir / (name + "_ms.json")
            ms_path.write_text(
                json.dumps(ms, ensure_ascii=False, indent=1, sort_keys=True) + "\n",
                encoding="utf-8",
            )
        paths[name] = (env_path, ms_path)
    return paths
