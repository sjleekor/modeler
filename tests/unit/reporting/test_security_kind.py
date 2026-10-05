"""순위 표 `종류` 열(06 문서 R0 발견 F1·F2, 1안): 이름으로 판정하고 표시만 합니다.

분류 사례는 2026-10-02 재현 유니버스(US 3,389·KR 2,008)의 실제 이름입니다. 규칙을 만들며
걸렸던 오탐(BNS, REIT 신탁, 메리츠, 성우, 심볼 재사용)을 그대로 시험으로 남깁니다.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import md_fixtures as mf
import pytest

from modeler.reporting import markdown as md

RELEASE = "r20261005"
STAMP = "2026-10-07T10:03:12+09:00"

US_CASES = [
    # 채권형
    ("F$D", "Ford Motor Company 6.500% Notes due August 15, 2062", "채권형"),
    (
        "NEE$U",
        "NextEra Energy, Inc. Series U Junior Subordinated Debentures due June 1, 2085",
        "채권형",
    ),
    (
        "MER$K",
        "Bank of America Corporation Income Capital Obligation Notes initially due "
        "December 15, 2066",
        "채권형",
    ),
    ("DUKB", "Duke Energy Corporation 5.625% Junior Subordinated Debentures due 2078", "채권형"),
    ("VXZ", "iPath Series B S&P 500 VIX Mid-Term Futures ETN", "채권형"),
    # 우선주
    ("GS$D", "Goldman Sachs Group, Inc. (The) Dep Shs repstg 1/1000 Pfd Ser D Fltg", "우선주"),
    ("MS$F", "Morgan Stanley Dep Shs Rpstg 1/1000th Int Prd Ser F Fxd to Flag", "우선주"),
    ("NLY$F", "Annaly Capital Management Inc 6.95% Series F", "우선주"),
    (
        "PSA$T",
        "Public Storage Depository Shares Representing 1/1000 Pfd Shares Beneficial "
        "Interest Series T",
        "우선주",
    ),
    ("SCE$M", "SCE Trust VII 7.50% Trust Preference Securities", "우선주"),
    (
        "LILAP",
        "Liberty Latin America Ltd. - 9.0% Fixed Rate Cumulative Perpetual Redeemable "
        "Series A Preference Shares",
        "우선주",
    ),
    ("BNS", "Bank Nova Scotia Halifax Pfd 3 Ordinary Shares", "보통주"),
    # SPAC
    ("BID", "Tribeca Strategic Acquisition Corp. - Class A Ordinary Shares", "SPAC"),
    ("NCO", "Southern Cross Acquisition I Corp. - Ordinary Shares", "SPAC"),
    ("CCXI", "Churchill Capital Corp XI - Class A Ordinary Shares", "SPAC"),
    ("CEPO", "Cantor Equity Partners I, Inc. - Class A Ordinary Shares", "SPAC"),
    ("NU", "Nu Holdings Ltd. Class A Ordinary Shares", "보통주"),
    ("ACN", "Accenture plc Class A Ordinary Shares (Ireland)", "보통주"),
    # 펀드
    (
        "BOE",
        "Blackrock Enhanced Global Dividend Trust Common Shares of Beneficial Interest",
        "펀드",
    ),
    ("BDJ", "Blackrock Enhanced Equity Dividend Trust", "펀드"),
    ("MUC", "Blackrock MuniHoldings California Quality Fund, Inc.  Common Stock", "펀드"),
    ("KTF", "DWS Municipal Income Trust", "펀드"),
    ("LEO", "BNY Mellon Strategic Municipals, Inc. Common Stock", "펀드"),
    ("NXP", "Nuveen Select Tax Free Income Portfolio Common Stock", "펀드"),
    ("ETO", "Eaton Vance Tax-Advantage Global Dividend Opp Common Stock", "펀드"),
    ("EIC", "Eagle Point Income Company Inc. Common Stock", "펀드"),
    ("RMT", "Royce Micro-Cap Trust, Inc. Common Stock", "펀드"),
    ("HQH", "abrdn Healthcare Investors Shares of Beneficial Interest", "펀드"),
    ("RFI", "Cohen & Steers Total Return Realty Fund, Inc. Common Stock", "펀드"),
    ("ARCC", "Ares Capital Corporation - Closed End Fund", "펀드"),
    ("GSBD", "Goldman Sachs BDC, Inc. Common Stock", "펀드"),
    # REIT·은행·로열티 신탁·운용사 자신은 보통주
    ("FCPT", "Four Corners Property Trust, Inc. Common Stock", "보통주"),
    ("PINE", "Alpine Income Property Trust, Inc. Common Stock", "보통주"),
    ("OPI", "Office Properties Income Trust - Common shares of beneficial interest", "보통주"),
    ("UHT", "Universal Health Realty Income Trust Common Stock", "보통주"),
    ("AMH", "American Homes 4 Rent Common Shares of Beneficial Interest", "보통주"),
    ("AAT", "American Assets Trust, Inc. Common Stock", "보통주"),
    ("TRTX", "TPG RE Finance Trust, Inc. Common Stock", "보통주"),
    ("NTRS", "Northern Trust Corporation - Common Stock", "보통주"),
    ("CTBI", "Community Trust Bancorp, Inc. - Common Stock", "보통주"),
    ("PBT", "Permian Basin Royalty Trust Common Stock", "보통주"),
    ("MSB", "Mesabi Trust Common Stock", "보통주"),
    ("CODI", "D/B/A Compass Diversified Holdings Shares of Beneficial Interest", "보통주"),
    ("BLK", "BlackRock, Inc. Common Stock", "보통주"),
    ("IVZ", "Invesco Ltd Common Stock", "보통주"),
    ("GOOG", "Alphabet Inc. - Class C Capital Stock", "보통주"),
    # 이름이 없으면 확인 안 됨, 다만 `$` 심볼은 우선주
    ("DGAC", None, "확인 안 됨"),
    ("XYZ$A", None, "우선주"),
]

KR_CASES = [
    ("473050", "유안타제15호스팩", "SPAC"),
    ("0165X0", "메리츠제2호스팩", "SPAC"),
    ("138040", "메리츠금융지주", "보통주"),
    ("088260", "이리츠코크렙", "리츠"),
    ("357120", "코람코라이프인프라리츠", "리츠"),
    ("395400", "SK리츠", "리츠"),
    ("088980", "맥쿼리인프라", "펀드"),
    ("415640", "KB발해인프라", "펀드"),
    ("199730", "바이오인프라", "보통주"),
    ("005935", "삼성전자우", "우선주"),
    ("005387", "현대차2우B", "우선주"),
    ("00104K", "CJ4우(전환)", "우선주"),
    ("00279K", "아모레퍼시픽홀딩스3우C", "우선주"),
    ("097955", "CJ제일제당 우", "우선주"),
    ("458650", "성우", "보통주"),
    ("005930", "삼성전자", "보통주"),
    ("005930", "005930", "확인 안 됨"),
    ("005930", None, "확인 안 됨"),
]


@pytest.mark.parametrize(("symbol", "name", "kind"), US_CASES)
def test_classify_us(symbol: str, name: str | None, kind: str) -> None:
    assert md.classify_us(symbol, name) == kind


@pytest.mark.parametrize(("symbol", "name", "kind"), KR_CASES)
def test_classify_kr(symbol: str, name: str | None, kind: str) -> None:
    assert md.classify_kr(symbol, name) == kind


def _no_log(_message: str) -> None:
    return None


def _inputs(tmp_path: Path) -> tuple[dict, bytes]:
    paths = mf.write_all(tmp_path / "fx")
    env_path, ms_path = paths["f1_normal"]
    return json.loads(env_path.read_text(encoding="utf-8")), ms_path.read_bytes()


def _names(us: dict | None = None, kr: dict | None = None) -> bytes:
    markets = {}
    if us is not None:
        markets["US"] = {"source": "US 시험 목록", "basis": "시험 기준", "symbols": us}
    if kr is not None:
        markets["KR"] = {"source": "KR 시험 목록", "basis": "-", "symbols": kr}
    return json.dumps({"schema": md.SECURITY_NAMES_SCHEMA, "markets": markets}).encode()


def _render(env: dict, ms: bytes, names: bytes | None) -> tuple[dict, dict]:
    ctx = md.build_context(
        Path("no-repo"),
        json.dumps(env).encode(),
        ms,
        RELEASE,
        STAMP,
        100,
        _no_log,
        security_names=names,
    )
    return ctx, md.render_unit(ctx)


def _table_rows(text: str, header_start: str) -> list[list[str]]:
    """header_start로 시작하는 첫 표의 본문 행을 셀 목록으로 돌려줍니다."""
    lines = text.split("\n")
    start = next(i for i, line in enumerate(lines) if line.startswith(header_start))
    rows = []
    for line in lines[start + 2 :]:
        if not line.startswith("|"):
            break
        rows.append([cell.strip() for cell in line.strip("|").split(" | ")])
    return rows


def _us_names_for(env: dict) -> dict:
    lgb = next(m for m in env["markets"] if m["model_id"] == mf.LGB_ID)
    symbols = [r["symbol"] for r in lgb["rankings"]]
    us = {s: {"name": f"Sample {s} Inc. Common Stock", "current": True} for s in symbols}
    us[symbols[0]] = {"name": "Ford Motor Company 6.500% Notes due 2062", "current": True}
    us[symbols[1]] = {"name": "Blackrock Enhanced Equity Dividend Trust", "current": True}
    us[symbols[2]] = {"name": "Old ETN Name", "current": False}  # 낡은 이름 → 확인 안 됨
    us[symbols[3]] = {"name": "Irenic Acquisition Corp. - Class A Ordinary Shares"}
    return us


def test_kind_column_and_count_line(tmp_path: Path) -> None:
    env, ms = _inputs(tmp_path)
    kr = next(m for m in env["markets"] if m["market"] == "KR")
    kr["rankings"][0].update(symbol="473050", name="유안타제15호스팩")
    kr["rankings"][1].update(symbol="005935", name="삼성전자우")
    ctx, files = _render(env, ms, _names(us=_us_names_for(env)))

    us_text = files["us-stocks.md"]
    rows = _table_rows(us_text, "| 순위 | 코드 | 종류 |")
    assert [r[2] for r in rows[:5]] == ["채권형", "펀드", "확인 안 됨", "SPAC", "보통주"]
    assert (
        "보통주가 아닌 종목 3개 포함 (채권형 1 · SPAC 1 · 펀드 1). 종류를 확인하지 못한 종목 1개."
        in us_text
    )

    kr_text = files["kr-stocks.md"]
    rows = _table_rows(kr_text, "| 순위 | 코드 | 이름 | 종류 |")
    assert [r[3] for r in rows[:3]] == ["SPAC", "우선주", "보통주"]
    assert "보통주가 아닌 종목 2개 포함 (우선주 1 · SPAC 1)." in kr_text
    assert "(data-status.md#증권-종류)" in kr_text

    summary = files["README.md"]
    assert "| 순위 | 코드 | 이름 | 종류 |" in summary  # KR 상위 10
    assert "| 순위 | 코드 | 종류 |" in summary  # US 상위 10
    status = files["data-status.md"]
    assert "## 증권 종류" in status and "US 시험 목록" in status
    assert "증권 이름 입력 sha256" in status
    assert md.validate_unit_files(ctx["unit"], files) == []


def test_kind_does_not_change_rank_or_score(tmp_path: Path) -> None:
    env, ms = _inputs(tmp_path)
    _, plain = _render(env, ms, None)
    _, kinded = _render(env, ms, _names(us=_us_names_for(env)))
    for name, header, kind_col in (
        ("us-stocks.md", "| 순위 | 코드 |", 2),
        ("kr-stocks.md", "| 순위 | 코드 | 이름 |", 3),
    ):
        before = _table_rows(plain[name], header)
        after = _table_rows(kinded[name], header)
        assert len(before) == len(after) == 100
        if name == "kr-stocks.md":  # KR은 행 이름으로 두 판 모두 종류 열이 있습니다
            assert before == after
        else:
            assert [r[:kind_col] + r[kind_col + 1 :] for r in after] == before


def test_kind_column_omitted_without_names(tmp_path: Path) -> None:
    env, ms = _inputs(tmp_path)
    for m in env["markets"]:
        if m["market"] == "KR":
            for r in m["rankings"]:
                r["name"] = r["symbol"]  # 서빙 KR 채점은 이름을 넣지 않습니다(F3)
    ctx, files = _render(env, ms, None)
    for name in ("us-stocks.md", "kr-stocks.md"):
        assert md.KIND_OMITTED in files[name]
        assert "| 종류 |" not in files[name]
        assert "보통주가 아닌 종목" not in files[name]
    assert "이름 원천이 없어 `종류` 열을 생략했습니다" in files["data-status.md"]
    assert md.validate_unit_files(ctx["unit"], files) == []


def test_kr_names_from_input_when_rows_are_unnamed(tmp_path: Path) -> None:
    env, ms = _inputs(tmp_path)
    kr = next(m for m in env["markets"] if m["market"] == "KR")
    for r in kr["rankings"]:
        r["name"] = r["symbol"]
    kr["rankings"][0].update(symbol="088980", name="088980")
    names = {r["symbol"]: {"name": "샘플기업"} for r in kr["rankings"]}
    names["088980"] = {"name": "맥쿼리인프라"}
    _, files = _render(env, ms, _names(kr=names))
    rows = _table_rows(files["kr-stocks.md"], "| 순위 | 코드 | 이름 | 종류 |")
    assert rows[0][2:4] == ["088980", "펀드"]  # 이름 열은 행 값(코드) 그대로입니다
    assert "envelope 순위 행의 이름, 없으면 KR 시험 목록" in files["data-status.md"]


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        ({"schema": "x", "markets": {}}, "schema"),
        ({"schema": md.SECURITY_NAMES_SCHEMA}, "markets"),
        ({"schema": md.SECURITY_NAMES_SCHEMA, "markets": {"US": {"symbols": []}}}, "US.symbols"),
    ],
)
def test_bad_names_input_is_an_input_error(payload: dict, message: str) -> None:
    with pytest.raises(md.InputError, match=re.escape(message)):
        md.load_security_names(json.dumps(payload).encode())


def test_kind_count_line() -> None:
    assert md.kind_count_line(["보통주"] * 3) == "보통주가 아닌 종목 0개 포함."
    line = md.kind_count_line(["펀드", "채권형", "우선주", "펀드", "확인 안 됨", "보통주"])
    assert line == (
        "보통주가 아닌 종목 4개 포함 (채권형 1 · 우선주 1 · 펀드 2). 종류를 확인하지 못한 종목 1개."
    )
