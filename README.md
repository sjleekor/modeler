# modeler

피쳐 생성, 학습, 검증, 백테스트. 원천에서 데이터를 받아오는 일은 `../collector`다.

역할 경계와 지켜야 할 원칙은 [`CLAUDE.md`](CLAUDE.md)와 [`../CLAUDE.md`](../CLAUDE.md)에 있다.
설계·계획 문서 대부분은 [`my/milestones/kr/modeling/`](../my/milestones/kr/modeling/)에 있다
(비공개 저장소라 외부에서는 링크가 열리지 않는다). 다만 코드가 실행 시점에 읽고
쓰는 baseline·실험 산출물(`docs/target/`, `docs/dev/20260907_model_experiment/`,
`docs/dev/20260907_additional_feature/`)은 golden 테스트 픽스처와 같은 이유로
이 저장소 안에 그대로 둔다.

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
