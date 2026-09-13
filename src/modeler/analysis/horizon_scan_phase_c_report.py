"""Phase C result report — ``03c_conditional_ic_results.md`` (Stage 1b §7.1).

Written from the rows every stage already produced, so the prose can never
disagree with the parquet beside it. The interpretation rules in §8 were fixed
*before* any result existed and are reproduced here verbatim, which is the
point: how to read the outcome was decided when it was still unknown.
"""

from __future__ import annotations

from typing import Any

# §8, written before the first delta was computed.
_READING_RULES = [
    "`screen_pass` 쌍이 하나 이상이면 `12`의 Phase C 개방 조건"
    '("경제적으로 설명할 수 있는 조건부 패턴")이 처음으로 충족된다. '
    "다음 단계(모델 입력 후보 승격, 국면 변수 확장)는 이 문서에서 정하지 않는다.",
    "P1·P2가 통과하고 P4·P5가 실패하면 "
    '"변동성 축은 VIX 국면 의존, 모멘텀 축은 국면 무관"으로 읽는다. 반대면 반대다. '
    '둘을 합쳐 "국면 조건화가 통했다/안 통했다"로 뭉뚱그리지 않는다.',
    "쌍들은 강하게 종속적이다. cell을 공유하는 쌍(P4·P5, P7·P11·P12)과 "
    "국면을 공유하는 쌍(`market_up` 5쌍, `liq_high` 6쌍)이 있다. "
    "같은 국면의 쌍이 여럿 통과했다고 그 수만큼 독립된 발견으로 세지 않는다.",
    "X1·X2에서 `|t_nw| > 2`가 나오면 규모 축에 국면 의존이 있다는 뜻이고 "
    "다음 사전등록의 후보가 된다. 이번 판정은 아니다.",
    "모든 쌍이 실패해도 결론은 "
    '"이 7개 국면 정의, 이 15개 쌍에서는 조건부 패턴이 없다"이지 '
    '"매크로는 쓸모없다"가 아니다.',
]


def _fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "–"
    if isinstance(value, bool):
        return "✅" if value else "✗"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def build_phase_c_report_context(
    *,
    run_spec: dict[str, Any],
    rows: list[dict[str, Any]],
    regime_summary: list[dict[str, Any]],
    q_threshold: float,
) -> dict[str, Any]:
    primary = [r for r in rows if r["regime_role"] == "primary"]
    return {
        "run_spec": run_spec,
        "rows": rows,
        "regime_summary": regime_summary,
        "q_threshold": q_threshold,
        "n_primary": len(primary),
        "n_valid": sum(1 for r in primary if r["status"] == "valid"),
        "n_discovery": sum(1 for r in primary if r.get("discovery")),
        "n_screen_pass": sum(1 for r in primary if r.get("screen_pass")),
        "grades": {
            grade: sum(1 for r in rows if r.get("evidence_grade") == grade)
            for grade in ("A", "B", "C", "D", "R")
        },
    }


def render_phase_c_report(context: dict[str, Any]) -> str:
    spec = context["run_spec"]
    lines = [
        "# Phase C 조건부 IC 결과",
        "",
        f"- run_id: `{spec.get('run_id')}`",
        f"- config_hash: `{spec.get('config_hash')}`",
        f"- contract: `{spec.get('contract')}`",
        f"- 입력 Phase A run: `{spec.get('phase_a_run_id')}`",
        f"- 입력 Phase B run: `{spec.get('phase_b_run_id')}`",
        f"- snapshot: `{spec.get('snapshot_date')}` / `{spec.get('source')}`",
        f"- 세션 격자: {spec.get('n_sessions')}개, "
        f"sha `{str(spec.get('session_grid_sha256'))[:12]}…`",
        "",
        "## 요약",
        "",
        f"- primary 쌍 {context['n_primary']}개 중 유효 {context['n_valid']}, "
        f"discovery {context['n_discovery']}, `screen_pass` {context['n_screen_pass']}",
        f"- BH q 임계 {context['q_threshold']}",
        "- 등급: " + ", ".join(f"{g} {n}" for g, n in context["grades"].items() if n),
        "",
        "## 쌍별 결과",
        "",
        "| id | family | cell | 국면 | 방향 | δ̂ | t_nw | p | q | G2 | G3 | G4 | screen | 등급 |",
        "|---|---|---|---|:-:|---:|---:|---:|---:|:-:|:-:|:-:|:-:|:-:|",
    ]
    for row in context["rows"]:
        cell = f"{row['scan_type']} {row['h_start']}→{row['h_end']}"
        lines.append(
            f"| {row['pair_id']} | `{row['family']}` | {cell} | `{row['regime_id']}` | "
            f"{row['direction_preregistered'] or '양방향'} | {_fmt(row.get('delta'))} | "
            f"{_fmt(row.get('t_nw'), 2)} | {_fmt(row.get('p_nw'))} | "
            f"{_fmt(row.get('q_fdr_phase_c'))} | {_fmt(row.get('period_sign_pass'))} | "
            f"{_fmt(row.get('tradable_pass'))} | {_fmt(row.get('placebo_pass'))} | "
            f"{_fmt(row.get('screen_pass'))} | {row.get('evidence_grade', '–')} |"
        )

    lines += [
        "",
        "## 표본과 국면 지속",
        "",
        "| id | 세션 | s=1 | s=0 | 전환 | 평균 지속 s=1/s=0 | 상태 |",
        "|---|---:|---:|---:|---:|---|---|",
    ]
    for row in context["rows"]:
        runs = (
            f"{_fmt(row.get('mean_run_length_s1'), 1)} / {_fmt(row.get('mean_run_length_s0'), 1)}"
        )
        reason = row.get("status_reason") or ""
        lines.append(
            f"| {row['pair_id']} | {row['n_dates']} | {row['n_dates_s1']} | {row['n_dates_s0']} | "
            f"{_fmt(row.get('n_regime_transitions'))} | {runs} | "
            f"{row['status']}{(' — ' + reason) if reason else ''} |"
        )

    if context["regime_summary"]:
        lines += [
            "",
            "## 국면 점유율 (G1)",
            "",
            "| regime | 세션 | s=1 | s=0 | 점유율 | G1 |",
            "|---|---:|---:|---:|---:|:-:|",
        ]
        for row in context["regime_summary"]:
            lines.append(
                f"| `{row['regime_id']}` | {row['n_dates']} | {row['n_dates_s1']} | "
                f"{row['n_dates_s0']} | {_fmt(row['share_s1'], 3)} | {_fmt(row['g1_pass'])} |"
            )

    lines += ["", "## 읽는 규칙 (사전등록, §8)", ""]
    lines += [f"{i}. {rule}" for i, rule in enumerate(_READING_RULES, start=1)]
    lines += [
        "",
        "> G4(국면 circular-shift placebo)는 모든 쌍에 필수다. 국면과 IC가 둘 다 지속 계열이라 "
        "관계가 없어도 δ̂가 크게 나올 수 있고, HAC는 국면 길이가 lag를 넘으면 잡지 못한다.",
        "",
    ]
    return "\n".join(lines)


def write_phase_c_report(path, context: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_phase_c_report(context), encoding="utf-8")
