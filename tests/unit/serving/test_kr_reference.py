"""KR reference session: complete_K, fallback_K_prime or none, from the exported snapshot."""

from __future__ import annotations

import json
from datetime import date, datetime, timedelta
from pathlib import Path

import polars as pl

from modeler.serving import kr_reference as ref
from modeler.serving.kr_reference import (
    FLOW_GROUPS,
    SnapshotFacts,
    assess_session,
    main,
    read_snapshot_facts,
    select_reference,
)

K = date(2026, 10, 2)
# Newest first, the order ``_sessions_before`` hands over (10-01, 09-30, 09-29, 09-28, ...).
BEFORE = [
    date(2026, 10, 1),
    date(2026, 9, 30),
    date(2026, 9, 29),
    date(2026, 9, 28),
    date(2026, 9, 25),
    date(2026, 9, 24),
    date(2026, 9, 23),
]
ALL_DAYS = [K, *BEFORE]
FULL = 2760


def _facts(
    counts: dict[date, int],
    *,
    price_max: date | None = None,
    flows: dict[str, date | None] | None = None,
) -> SnapshotFacts:
    latest = price_max if price_max is not None else max(counts)
    groups = {name: K for name in FLOW_GROUPS}
    groups.update(flows or {})
    return SnapshotFacts(price_max=latest, price_counts=counts, flow_group_latest=groups)


def _full_counts(days: list[date] = ALL_DAYS, **overrides: int) -> dict[date, int]:
    counts = {day: FULL for day in days}
    for key, value in overrides.items():
        counts[date.fromisoformat(key.replace("_", "-"))] = value
    return counts


def test_complete_k_is_chosen_without_a_cut() -> None:
    result = select_reference(_facts(_full_counts()), k=K, sessions_before_k=BEFORE)
    assert result["verdict"] == "complete_K" and result["reference_date"] == "2026-10-02"
    assert result["lag_sessions"] == 0 and result["cut_required"] is False
    (only,) = result["candidates"]
    assert only["complete"] and only["price"]["ticker_ratio"] == 1.0
    assert {name: item["reaches_session"] for name, item in only["flows"].items()} == {
        name: True for name in FLOW_GROUPS
    }


def test_k_with_a_partial_price_session_falls_back_to_the_previous_complete_session() -> None:
    """The evening chain stopped half way (K has 60% of the tickers); K-prime is in the export."""
    counts = _full_counts(**{"2026_10_02": int(FULL * 0.6)})
    result = select_reference(_facts(counts), k=K, sessions_before_k=BEFORE)
    assert result["verdict"] == "fallback_K_prime" and result["reference_date"] == "2026-10-01"
    assert result["lag_sessions"] == 1
    assert result["cut_required"] is True  # K's partial rows lie after K' and must be hidden
    bad, good = result["candidates"]
    assert (
        not bad["complete"]
        and "ticker count 1656 is 0.6000 of the previous session" in bad["reasons"][0]
    )
    assert good["complete"]


def test_ratio_boundary_is_97_percent_inclusive() -> None:
    previous = 1000
    counts = {date(2026, 10, 1): previous, K: 970}
    assert assess_session(_facts(counts), K)["complete"]
    counts[K] = 969
    assessed = assess_session(_facts(counts), K)
    assert not assessed["complete"] and assessed["price"]["ticker_ratio"] == 0.969


def test_flow_group_that_stops_short_of_k_makes_k_incomplete() -> None:
    facts = _facts(_full_counts(), flows={"investor": date(2026, 10, 1)})
    result = select_reference(facts, k=K, sessions_before_k=BEFORE)
    assert result["verdict"] == "fallback_K_prime" and result["reference_date"] == "2026-10-01"
    assert any("flow group investor" in reason for reason in result["candidates"][0]["reasons"])


def test_a_flow_group_without_rows_is_never_judged_complete() -> None:
    facts = _facts(_full_counts(), flows={"shorting": None})
    assert select_reference(facts, k=K, sessions_before_k=BEFORE)["verdict"] == "none"


def _shrinking(complete_from: date | None) -> dict[date, int]:
    """Every session has half the tickers of the one before it (ratio 0.5): incomplete.

    ``complete_from`` and the session before it get the same count, so that session passes."""
    ordered = sorted(ALL_DAYS)  # oldest first: the oldest session has the most tickers
    counts = {day: max(1, FULL >> index) for index, day in enumerate(ordered)}
    if complete_from is not None:
        previous = max(day for day in counts if day < complete_from)
        counts[complete_from] = counts[previous]
    return counts


def test_the_lookback_reaches_five_sessions_below_k() -> None:
    result = select_reference(_facts(_shrinking(date(2026, 9, 25))), k=K, sessions_before_k=BEFORE)
    # K, 10-01, 09-30, 09-29, 09-28 are incomplete; 09-25 (the fifth below K) passes.
    assert result["verdict"] == "fallback_K_prime" and result["reference_date"] == "2026-09-25"
    assert result["lag_sessions"] == 5 and len(result["candidates"]) == 6


def test_none_when_nothing_within_five_sessions_is_complete() -> None:
    far = select_reference(_facts(_shrinking(date(2026, 9, 24))), k=K, sessions_before_k=BEFORE)
    assert (
        far["verdict"] == "none" and far["reference_date"] is None and far["lag_sessions"] is None
    )
    assert far["cut_required"] is False
    assert (
        len(far["candidates"]) == 1 + ref.MAX_LOOKBACK_SESSIONS
    )  # K and five below; the sixth is out of reach


def test_a_session_without_a_previous_session_cannot_be_judged() -> None:
    assessed = assess_session(_facts({K: FULL}), K)
    assert not assessed["complete"]
    assert "ticker ratio cannot be judged" in assessed["reasons"][0]


def test_k_is_complete_even_when_the_export_holds_later_rows_which_are_then_cut() -> None:
    counts = _full_counts()
    counts[date(2026, 10, 5)] = 12  # D's own first rows
    result = select_reference(
        _facts(counts, price_max=date(2026, 10, 5)), k=K, sessions_before_k=BEFORE
    )
    assert result["verdict"] == "complete_K" and result["cut_required"] is True
    assert result["candidates"][0]["price"]["rows_after_session"] is True


def test_prices_that_stop_before_the_session_are_incomplete() -> None:
    counts = _full_counts(ALL_DAYS[1:])  # nothing for K
    assessed = assess_session(_facts(counts, price_max=date(2026, 10, 1)), K)
    assert not assessed["complete"]
    assert any("older than 2026-10-02" in reason for reason in assessed["reasons"])
    assert any("no daily_ohlcv rows" in reason for reason in assessed["reasons"])


# ---- parquet snapshot + CLI ----


def _write_snapshot(root: Path, snapshot: str, *, k_count: int, flow_latest: date) -> Path:
    raw = root / "kr" / "raw" / "raw_postgres" / f"snapshot_date={snapshot}" / "source=sj2_remote"
    prices = []
    for day in ALL_DAYS:
        count = k_count if day == K else FULL
        prices += [(day, f"{index:06d}", "KOSPI") for index in range(count)]
    (raw / "daily_ohlcv").mkdir(parents=True)
    pl.DataFrame(prices, schema=["trade_date", "ticker", "market"], orient="row").write_parquet(
        raw / "daily_ohlcv" / "part-0.parquet"
    )
    flows = []
    for group, metrics in FLOW_GROUPS.items():
        for metric in metrics:
            latest = flow_latest if group == "investor" else K
            flows.append((latest, "000001", metric, "KRX"))
            flows.append(
                (K + timedelta(days=30), "000001", metric, "PYKRX")
            )  # other source: ignored
    (raw / "krx_security_flow_raw").mkdir(parents=True)
    pl.DataFrame(
        flows, schema=["trade_date", "ticker", "metric_code", "source"], orient="row"
    ).write_parquet(raw / "krx_security_flow_raw" / "part-0.parquet")
    return raw


def _calendar(path: Path) -> Path:
    sessions = sorted(day.isoformat() for day in ALL_DAYS + [date(2026, 10, 5)])
    path.write_text(
        json.dumps(
            {
                "market": "KR",
                "timezone": "Asia/Seoul",
                "coverage_start": "2026-09-21",
                "coverage_end": "2026-10-05",
                "sessions": sessions,
                "default_open_at": "09:00",
                "default_close_at": "15:30",
                "overrides": [],
                "unconfirmed_dates": [],
            }
        )
    )
    return path


def test_read_snapshot_facts_counts_prices_and_uses_only_krx_and_kis_flows(tmp_path: Path) -> None:
    raw = _write_snapshot(tmp_path, "2026-10-05", k_count=100, flow_latest=date(2026, 10, 1))
    facts = read_snapshot_facts(raw_root=raw, newest=K)
    assert (
        facts.price_max == K
        and facts.price_counts[K] == 100
        and facts.price_counts[date(2026, 10, 1)] == FULL
    )
    assert facts.flow_group_latest == {
        "foreign_holding": K,
        "investor": date(2026, 10, 1),
        "shorting": K,
    }


def test_cli_writes_the_evidence_and_prints_the_reference_session(tmp_path: Path, capsys) -> None:
    _write_snapshot(tmp_path, "2026-10-05", k_count=100, flow_latest=K)
    gate = tmp_path / "gate.json"
    gate.write_text(
        json.dumps(
            {
                "reasons": ["xbrl_parse: no ingestion_runs record"],
                "dart_chain_ended_at": None,
                "checks": [
                    {"check": "daily_ohlcv", "status": "ok"},
                    {
                        "check": "ingestion_run",
                        "run_type": "xbrl_parse",
                        "status": "not_yet",
                        "run_status": None,
                        "ended_at": None,
                    },
                ],
            }
        )
    )
    out = tmp_path / "evidence" / "reference-selection.json"
    code = main(
        [
            "--snapshot-date",
            "2026-10-05",
            "--report-date",
            "2026-10-05",
            "--k",
            "2026-10-02",
            "--calendar",
            str(_calendar(tmp_path / "cal.json")),
            "--stock-data-root",
            str(tmp_path),
            "--output",
            str(out),
            "--gate-exit",
            "75",
            "--gate-evidence",
            str(gate),
        ]
    )
    assert code == 0
    assert capsys.readouterr().out.splitlines() == ["2026-10-01", "cut"]
    evidence = json.loads(out.read_text())
    assert (
        evidence["schema"] == "kr-reference-selection.v1"
        and evidence["verdict"] == "fallback_K_prime"
    )
    assert evidence["k"] == "2026-10-02" and evidence["reference_date"] == "2026-10-01"
    assert evidence["lag_sessions"] == 1 and evidence["cut_required"] is True
    # The gate verdict is recorded, advisory, and never part of the choice.
    assert evidence["gate"]["verdict"] == "not_yet" and evidence["gate"]["advisory"] is True
    assert evidence["gate"]["run_checks"] == [
        {"run_type": "xbrl_parse", "status": "not_yet", "run_status": None, "ended_at": None}
    ]
    datetime.fromisoformat(evidence["checked_at"])


def test_cli_exit_32_when_nothing_is_complete_and_2_on_bad_input(tmp_path: Path, capsys) -> None:
    _write_snapshot(
        tmp_path, "2026-10-05", k_count=FULL, flow_latest=date(2026, 9, 1)
    )  # investor never reaches
    out = tmp_path / "e.json"
    args = [
        "--snapshot-date",
        "2026-10-05",
        "--report-date",
        "2026-10-05",
        "--k",
        "2026-10-02",
        "--calendar",
        str(_calendar(tmp_path / "cal.json")),
        "--stock-data-root",
        str(tmp_path),
        "--output",
        str(out),
    ]
    assert main(args) == ref.EXIT_NONE
    assert capsys.readouterr().out == ""
    assert json.loads(out.read_text())["verdict"] == "none"

    def with_value(flag: str, value: str) -> list[str]:
        changed = list(args)
        changed[changed.index(flag) + 1] = value
        return changed

    assert main(with_value("--k", "2026-10-05")) == 2  # K must precede D
    assert main(with_value("--snapshot-date", "2099-01-01")) == 2  # the snapshot does not exist


def test_cli_k_must_be_a_session_of_the_calendar(tmp_path: Path) -> None:
    _write_snapshot(tmp_path, "2026-10-05", k_count=FULL, flow_latest=K)
    calendar = _calendar(tmp_path / "cal.json")
    body = json.loads(calendar.read_text())
    body["sessions"].remove("2026-10-02")
    calendar.write_text(json.dumps(body))
    assert (
        main(
            [
                "--snapshot-date",
                "2026-10-05",
                "--report-date",
                "2026-10-05",
                "--k",
                "2026-10-02",
                "--calendar",
                str(calendar),
                "--stock-data-root",
                str(tmp_path),
                "--output",
                str(tmp_path / "e.json"),
            ]
        )
        == 2
    )


def test_the_selection_never_reads_dates_the_calendar_does_not_cover(tmp_path: Path) -> None:
    from modeler.serving.kr_reference import _sessions_before

    calendar = _calendar(tmp_path / "cal.json")
    body = json.loads(calendar.read_text())
    body["coverage_start"] = "2026-09-28"
    body["sessions"] = [s for s in body["sessions"] if s >= "2026-09-28"]
    calendar.write_text(json.dumps(body))
    # Coverage starts at 09-28: older sessions are unknown and are not guessed.
    assert _sessions_before(calendar, K) == [
        date(2026, 10, 1),
        date(2026, 9, 30),
        date(2026, 9, 29),
        date(2026, 9, 28),
    ]
