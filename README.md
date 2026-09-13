# modeler

피쳐 생성, 학습, 검증, 백테스트. 원천에서 데이터를 받아오는 일은 `../collector`다.

역할 경계와 지켜야 할 원칙은 [`CLAUDE.md`](CLAUDE.md)와 [`../CLAUDE.md`](../CLAUDE.md)에 있다.

## 개발

```bash
uv sync --extra dev                       # 기본
uv sync --extra dev --extra features      # 피쳐 생성까지
uv sync --extra dev --extra eval          # 검증·평가까지
uv run pytest
uv run ruff check src/ tests/
uv run black src/ tests/
```

## 전제

**미국 시장 검정 구간은 2018-09-07 이후 8.0년이다.** 가격 원천의 그 이전 구간에 상폐 종목이 없다.
근거는 [`08_universe_build.md`](../my/milestones/us/research/data/web_scraping/08_universe_build.md)에 있다.
