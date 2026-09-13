# CLAUDE.md — modeler

**모델링 코드 프로젝트다.** 피쳐 생성, 학습, 검증, 백테스트가 여기 있다.
원천에서 데이터를 받아오는 일은 `collector/`, **데이터 파일은 `stock_data/`다.**

배치 원칙은 [`../CLAUDE.md`](../CLAUDE.md)에 있다.

---

## 현재 상태

**뼈대만 있다 (2026-09-12).** Python 3.12 + `uv` 환경이 서 있고 테스트가 통과한다.
모델링 로직은 아직 없다.

```
src/modeler/
├── features/   피쳐 생성 — 미래를 보지 않는 변환만 넣는다
└── models/     학습·검증
```

```bash
uv sync --extra dev                    # 기본 (pandas·numpy·pyarrow·duckdb·polars·sklearn)
uv sync --extra dev --extra features   # + TA-Lib · feature-engine · tsfresh
uv sync --extra dev --extra eval       # + skfolio · alphalens-reloaded · lightgbm
uv run pytest
uv run ruff check src/ tests/
uv run black src/ tests/
```

**무거운 것은 extra 로 분리했다.** 가격 입력이 붙기 전에는 기본만으로 충분하다.
선택 근거는 [`90_stack_recommendation`](../my/milestones/us/research/libraries/90_stack_recommendation.md)에 있다.

설정은 [`../CLAUDE.md`](../CLAUDE.md)의 공통 툴체인을 따른다.

한국 시장 모델링 코드는 `stock_data_collector/research/`에 있고 이쪽으로 옮길 예정이다 —
[분리 계획](../my/milestones/kr/refactoring/20260912_project_split/00_candidate_plan/README.md).

---

## 모델링 코드가 지켜야 할 것

| 원칙 | 왜 |
|---|---|
| **미래를 보지 않는다** | 롤링 통계·정규화·결측 보간 전부. `feature-engine`처럼 sklearn 인터페이스를 쓰면 구조적으로 막힌다 |
| **유니버스는 그 시점 것을 쓴다** | 현재 종목 목록으로 과거를 돌리면 생존편향이 들어간다 |
| **검정 구간을 넘겨 학습하지 않는다** | purged CV를 쓴다 (`skfolio`의 `CombinatorialPurgedCV`) |
| **라이선스를 본다** | `vectorbt`(판매 금지), `backtesting.py`·`OpenBB`(AGPL), `lib-pybroker`(비상업) |

---

## 데이터를 어디서 읽고 어디에 쓰나

**둘 다 `stock_data/`다. 이 저장소 안에 데이터를 쓰지 않는다.**

**최상위가 시장이다** — `../stock_data/kr/`, `../stock_data/us/`.

| 항목 | 값 |
|---|---|
| 입력 | `<시장>/raw/` — `collector/`가 적재한 것. **읽기만 한다** |
| 출력 | `<시장>/derived/`(파생·피쳐) · `<시장>/datasets/`(모델 입력) · `<시장>/output/`(예측·리포트) |
| 경로 지정 | 환경변수. 기본값만 `../stock_data`를 가리킨다 |
| 재현 | 실행마다 snapshot 날짜·피쳐 집합·라벨 정의·코드 버전을 manifest에 남긴다 |

**`raw/`에 쓰지 않는다.** 거기는 재수집으로만 복원되는 유일한 계층이고 `collector/`의 것이다.

**`collector/`를 import하지 않는다.** 둘 다 필요한 정의(거래일 달력, 지표 매핑 규칙 등)가
생기면 공유 패키지로 뺀다. 한쪽이 다른 쪽을 import하면 배포가 같이 묶인다.

---

## 미국 시장 전제 (2026-09-12)

**검정 구간은 2018-09-07 이후 8.0년이다.**

가격 원천(DoltHub)의 **2018-09 이전 구간에 상폐 종목이 없다.** 21,592종목 중
2018-09-01 이전에 거래가 끝난 것이 1개뿐이라, 그 이전을 쓰면 생존편향이 그대로 들어간다.
근거는 [`08_universe_build.md`](../my/milestones/us/research/data/web_scraping/08_universe_build.md)에 있다.

**이 전제가 바뀌려면 2011~2018 상폐 종목의 가격 원천을 찾아야 한다.**
