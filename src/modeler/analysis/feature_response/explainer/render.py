"""Data + config + ``template.html`` -> one self-contained HTML page."""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from .paths import Roots
from .texts import code, kr_defaults, us_defaults

TEMPLATE = Path(__file__).with_name("template.html")
PACKAGE = "src/modeler/analysis/feature_response/explainer/"
KR_DEFAULT_LABELS = [
    "px_idio_vol_60d",
    "px_turnover_ma20",
    "px_dist_52w_high",
    "px_high_low_range_20d",
    "px_vol_60d",
]


def js(obj: Any) -> str:
    """JSON for an inline ``<script type="application/json">`` block."""
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")


def footer_common(cfg, name: str, doc_md: str, res_md: str, generated: dt.date) -> str:
    return (
        f"<li><b>생성일</b> {generated.isoformat()}. 생성기는 modeler "
        f"{code('feature-response-l3')} "
        f"브랜치의 {code(PACKAGE)}에 있습니다(config {code(name)}). 이 파일은 "
        f"{code('python -m modeler.analysis.feature_response.explainer build ' + name)}으로 "
        "다시 만들고, 중간 숫자(json)는 저장소가 아니라 캐시 폴더에 남습니다.</li>"
        f'<li><b>정본</b>: <a href="{doc_md}">{code(doc_md)}</a>(정의·직관·기대 부호) · '
        f'<a href="{res_md}">{code(res_md)}</a>(층 3 결과) · '
        f'<a href="README.md">{code("README.md")}</a>(코드로 확인한 함정 피쳐). '
        "이 페이지의 모양·안정·기대 부호 대조는 결과 문서의 값을 그대로 옮겼고, 보조 판정은 같은 "
        "규칙으로 곡선에서 다시 계산해 결과 문서 표와 일치함을 확인했습니다.</li>"
        "<li>곡선은 21개 격자점을 직선으로 이은 것입니다. 개별 종목 선(ICE)은 같은 표본에서 뽑은 "
        f"1,000개 중 {cfg.N_ICE}개이며, '수준 맞춤'은 각 선의 평균을 PDP 평균에 맞춰 종목 간 수준 "
        "차이를 뺀 것입니다.</li>"
    )


def render_html(
    cfg: SimpleNamespace,
    data: dict[str, Any],
    name: str,
    roots: Roots,
    generated: dt.date | None = None,
) -> str:
    generated = generated or dt.date.today()
    market = getattr(cfg, "MARKET", "kr")
    feats = [f for _, _, fs in cfg.THEMES for f in fs]
    doc_md = getattr(cfg, "DOC_MD", Path(cfg.MD_CARDS).name)
    res_md = getattr(cfg, "RESULT_MD", Path(cfg.MD_RESULT).name)
    make = kr_defaults if market == "kr" else us_defaults
    parts = make(cfg, data, feats, doc_md, res_md, roots.rel)
    override = getattr(cfg, "HTML_OVERRIDE", {})
    # FOOT override replaces only the paragraphs above the input/run_id lines; the common
    # footer (inputs, run_id, commit, generator location) is always appended.
    foot_top = override.get("FOOT", "")
    parts["FOOT"] = foot_top + parts["FOOT"] + footer_common(cfg, name, doc_md, res_md, generated)
    ui = parts.pop("UI")
    ui.update(getattr(cfg, "UI", {}))
    for key in list(parts):
        if key != "FOOT" and key in override:
            parts[key] = override[key]
    content = {
        "themes": cfg.THEMES,
        "cards": cfg.CARDS,
        "fmt": cfg.FMT,
        "basis": cfg.BASIS,
        "ui": ui,
        "group_names": getattr(cfg, "GROUP_NAMES", {}),
        "ex_notes": getattr(cfg, "EX_NOTES", {}),
        "scatter_labels": getattr(
            cfg, "SCATTER_LABELS", KR_DEFAULT_LABELS if market == "kr" else []
        ),
    }
    toc = "".join(f'<a href="#t{i}">{t[0]}</a>' for i, t in enumerate(cfg.THEMES))
    out = TEMPLATE.read_text(encoding="utf-8")
    out = out.replace("__TITLE__", cfg.PAGE_TITLE).replace("__TOC__", toc)
    for key, value in parts.items():
        out = out.replace(f"__{key}__", value)
    out = out.replace("__DATA__", js(data)).replace("__CONTENT__", js(content))
    if "__KICKER__" in out or "__FOOT__" in out:
        raise RuntimeError("template placeholder left unfilled")
    return out
