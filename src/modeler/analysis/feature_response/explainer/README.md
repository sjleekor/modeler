# 서빙 피쳐 해설 그림판 생성기 (`explainer`)

카드 문서(`my/milestones/common/20261012_feature_explainer/0N_*.md`)와 층 3 결과(`09_kr_result.md`·`10_us_result.md`)를 읽어 문서마다 **단일 HTML 한 장**을 만든다. 숫자는 층 3 산출물(`feature_response_l3`)에서, 원값 눈금은 라벨 없는 동결 피쳐 패널에서, 예시 차트는 기준일 이하 일봉에서 온다. **라벨 파일은 열지 않는다**(`paths.guard_not_label`이 경로에 `label`이 든 것을 막는다).

이 코드는 `modeler` 브랜치 `feature-response-l3`에 있고 10-20 release 뒤 main에 합친다. 새 파일이라 release 검토 목록에 넣어야 빌드가 선다.

## 1. 실행

```bash
cd /Users/whishaw/wss_p/modeler      # 기본 경로가 ../stock_data, ../my 라서 modeler 디렉터리에서 돌린다
M=modeler.analysis.feature_response.explainer
uv run --offline --no-sync python -m $M list
uv run --offline --no-sync python -m $M build kr_price
uv run --offline --no-sync python -m $M build kr_flow --out /tmp/x/02.html --cache-dir /tmp/x/cache
uv run --offline --no-sync python -m $M build-all --out-dir /tmp/x --cache-dir /tmp/x/cache
uv run --offline --no-sync python -m $M build /path/to/my_config.py
```

worktree의 코드를 돌릴 때는 `PYTHONPATH=<worktree>/src`를 앞에 붙인다. `uv sync`·`uv add`는 쓰지 않는다.

| 인자·환경변수 | 뜻 |
|---|---|
| `STOCK_DATA_ROOT` / `--stock-data-root` | 데이터 루트. 기본 `../stock_data` |
| `MY_ROOT` / `--my-root` | 카드 md와 HTML이 있는 `my/` 루트. 기본 `../my` |
| `--out PATH` | (`build`) HTML 위치. 기본 `<MY_ROOT>/<DOC_DIR>/<OUT_NAME>` |
| `--out-dir DIR` | (`build-all`) 모든 HTML을 이 폴더에 쓴다 |
| `--cache-dir DIR` | 중간 숫자 `data_<이름>.json`과 `build_log_<이름>.txt`. 기본 `<STOCK_DATA_ROOT>/<kr 또는 us>/output/feature_response_l3/explainer_cache/` |

- 코드는 자기 디렉터리에 아무것도 쓰지 않는다. 쓰는 곳은 HTML 출력 위치와 캐시 폴더뿐이다.
- 종료 코드 1 = 로그에 `MISMATCH`·`no card line`·`no md row`·`no supp row`가 있다(md와 산출물 숫자가 어긋남). 파일은 그래도 쓴다. 0건이 될 때까지 원인을 찾는다.
- 번들 config 이름: `kr_price`(01) · `kr_flow`(02) · `kr_fin_event`(03) · `us_price_liquidity`(04) · `us_fundamental_event`(05) · `us_flow_holders`(06). 새 문서는 `configs/`에 모듈을 추가하고 `config.CONFIG_NAMES`에 이름을 넣는다.
- 경로는 코드에 박지 않는다. config는 `DOC_DIR`(`MY_ROOT` 아래), `RUN_ID`, `PANEL_REL`·`RAW_REL`·`PANEL_V2_REL` 등(`STOCK_DATA_ROOT` 아래 상대경로)만 적고, 절대경로는 `config.resolve_config`가 만든다.

## 2. 파일

| 파일 | 역할 |
|---|---|
| `cli.py`·`__main__.py` | 명령(build·build-all·list) |
| `paths.py` | 루트 해석(`STOCK_DATA_ROOT`·`MY_ROOT`), 라벨 가드, 최신 스냅샷 고르기 |
| `config.py` | config 로드·검증(`validate_config`·`validate_principle`)·경로 풀기 |
| `data_kr.py`·`data_us.py` | 숫자 만들기(KR: 섞기·SHAP·PDP, US: LightGBM·Ridge). md 표를 파싱해 교차 검증 |
| `common.py` | 순수 함수(md 정리, 보조 판정, 예시 종목 고르기, NaN 정리) |
| `render.py`·`texts.py`·`template.html` | HTML 조립. 기본 문구는 `texts.py`, 그림은 `template.html`의 JS |
| `configs/*.py` | 문서별 config 6개 |

시험은 `tests/unit/analysis/test_feature_response_explainer.py`(합성 자료, 실제 데이터 없이 돈다). 스크린샷 스크립트 같은 개발 도구는 코드에 넣지 않았다. 헤드리스 Chrome 사용법은 6절에 있다.

> 아래 3~7절은 config를 쓰는 사람을 위한 안내다.

## 3. config 필드

### 3.1 필수 필드

경로는 **상대 이름만** 적는다. 절대경로(`/Users/...`)를 config에 쓰지 않는다.

| 필드 | 뜻 | 예 |
|---|---|---|
| `MARKET` | `"kr"` 또는 `"us"` | `"us"` |
| `DOC_DIR` | 카드 md·HTML 폴더. `MY_ROOT` 아래 상대경로 | `"milestones/common/20261012_feature_explainer"` |
| `OUT_NAME` | 출력 HTML 파일명 (`DOC_DIR` 아래) | `"02_kr_flow.html"` |
| `DOC_MD`·`RESULT_MD` | 카드 md·결과 md 파일명 (`DOC_DIR` 아래, 읽기만) | `"02_kr_flow.md"`, `"09_kr_result.md"` / `"10_us_result.md"` |
| `RUN_ID` | 층 3 run id. 폴더는 `<STOCK_DATA_ROOT>/<market>/output/feature_response_l3/<RUN_ID>` | `"kr_l3_20261012T113305Z"` / `"us_l3_20261012T112931Z"` |
| `REF_DATE` | 원값 눈금·예시 기준일 `"YYYY-MM-DD"` | KR `2026-08-28`, US `2026-06-01` |
| `PAGE_TITLE` | 페이지 제목 (2~4단어 이름) | `"KR 수급 피쳐 그림 해설"` |
| `N_ICE` | 곡선에 그리는 ICE 선 수 | `40` |
| `EX_DAYS` | 예시 차트의 거래일 수(창이 252행이면 260 이상) | `260` |
| `THEMES` | `[(소주제 제목, 한 줄 설명, [피쳐 id, ...]), ...]` — 카드 순서가 이 순서 | |
| `CARDS` | 피쳐 id → 카드 dict (3.3) | |
| `FMT` | 피쳐 id → 원값 표시 형식 (3.4) | |
| `BASIS` | 계산 기준 코드 → 설명 문자열 (카드의 `basis` 값으로 씀) | |

KR 전용(필수): `PREFIX`(지도에서 색칠할 피쳐 접두사. **문자열 하나 또는 튜플** `("fin_", "ev_", "own_")`), `EX_MARKET`(`"KOSPI"`), `MODELER_COMMIT`(푸터 커밋 `"24eb8e0"`), `PANEL_REL`(`feat_panel.parquet`)·`RAW_REL`(raw_postgres 스냅샷 폴더) — 둘 다 `STOCK_DATA_ROOT` 아래 상대경로.

US 전용(필수): `PANEL_V2_REL`·`PANEL_FLOW_REL`(원값 분위용 동결 피쳐 패널, 상대경로), `PRICES_GLOB`·`COMPANY_META_GLOB`(`STOCK_DATA_ROOT` 아래 glob. **가장 최근 snapshot_date가 자동으로 뽑힌다**). 커밋은 manifest에서 자동이다.

`validate_config`가 빠진 필드와 카드 필드(`title`·`basis`·`intuition`·`win`·`what`·`model`·`caution`), 잘못된 `basis` 키, `principle` 인자 오류를 알려 준다. 시험하는 법: `python -m modeler.analysis.feature_response.explainer build <config>`가 맨 먼저 이 검증을 한다.

### 3.2 선택 필드

| 필드 | 뜻 |
|---|---|
| `GROUP_NAMES` | `{묶음 id: 이름}`. KR은 `c01`.., US는 `g1`.. (groups.json 순서, 이 문서 피쳐가 든 묶음만 그려짐) |
| `SCATTER_LABELS` | 한 장 지도에서 직접 라벨을 붙일 피쳐 id 목록(5개 안팎). KR 기본값은 px 피쳐 5개라 **다른 문서에서는 반드시 지정**. US 기본은 없음 |
| `UI` | JS가 쓰는 문구 덮어쓰기: `ylabel`, `total_label`, `half1`, `half2`, `ex_scope`, `ex_pool`, `ex_note`, `singles_prefix`, `disclaimer` 등. 대부분 MARKET 기본값으로 충분 |
| `HTML_OVERRIDE` | 머리말 조각 교체: 키 `LEDE`(머리 문단 HTML), `GUIDE`(`<li>…</li>` 연속), `MAPINTRO`, `MAPCAP`, `MAPLEGEND`, `MAPTITLE`, `GROUPSTITLE`, `GROUPSINTRO`, `FOOT`, `KICKER`. **`FOOT`는 입력·run_id 줄 앞에 들어가는 문단만 바꾼다. 입력 파일·run_id·커밋·생성기 위치·정본 링크 푸터는 항상 붙으니 `FOOT`에 다시 적지 않는다.** **KR px 문서가 아니면 `LEDE`·`MAPINTRO`·`MAPCAP`·`GROUPSINTRO`를 반드시 자기 문서에 맞게 덮어쓴다** (KR 기본 문구는 px 18개 기준이고 숫자가 박혀 있다) |
| `EX_NOTES` | 예시 종목 아래 설명 문장. `{피쳐 id: "문장"}` 또는 `{피쳐 id: {"p95": "...", "p05": "...", "default": "..."}}`. `{value}`·`{pct}`·`{name}` 치환 |
| `EXAMPLES`·`EX_WINDOWS` | 예시 종목 지정 (3.6). **`EX_WINDOWS`는 생략할 수 있고**, 피쳐 키가 없으면 창 없는 예시(음영·범례 없음)가 된다 |

> KR px 01번의 `MAPCAP`은 px 숫자(0.0547, 7개 등)를 직접 쓴다. 다른 KR 문서에서 같은 문구가 나가면 틀린 문서가 된다. 반드시 `HTML_OVERRIDE`로 교체하고, 숫자는 결과 md에서 옮긴다.

### 3.3 CARDS[id] 필드

```python
"px_ret_1d": dict(
  title="하루 수익률",                 # 제목 (한글 이름)
  basis="valid",                       # BASIS의 키
  intuition="한 줄 직관 (md '한 줄 직관'을 줄여서, 충실하게)",
  analogy="쉬운 비유 한 줄 (선택)",     # 4.4 참고
  win=dict(...),                       # ① 계산 창 (3.5)
  principle=dict(...),                 # ⓟ 원리 그림 (선택, 4절)
  what="무엇을 재나",                  # 3줄 요약 1
  model="모델이 어떻게 쓰나",          # 3줄 요약 2 — 불안정이면 '해석하지 않는다'
  caution="주의",                      # 3줄 요약 3
)
```
카드 순서(화면): 제목·한 줄 직관 → 쉬운 비유 → ① 계산 창 → ⓟ 원리 그림 → ③ 반응 곡선(+숫자) → 3줄 요약 → ② 예시. `analogy`·`principle`·예시는 **없으면 그 칸이 통째로 빠진다**(01번이 이렇게 나간다).
**숫자(방향 ρ, SHAP 순위, 섞기, 안정, 모양, 기대 대조)는 카드에 쓰지 않는다.** 산출물 parquet과 결과 md 표에서 빌더가 채운다. `what/model/caution`의 문장에 숫자를 쓸 때는 결과 md 값을 그대로 옮긴다.

### 3.4 FMT 형식

`pct`(×100 + %) · `mult`(0.00배) · `eok`(원 → 억/조) · `sci`(1.2e−9) · `ratio`(0.00) · `bin`(0/1) · `usd`($) · `big`(K/M/B) · `num`(유효숫자 3자리) · `int`(정수, 쉼표). 가격·금액 같은 원값이면 `usd`/`eok`/`big`, 비율·수익률이면 `pct`, 0~1 비율이면 `ratio`.

### 3.5 `win` — ① 계산 창

```python
def seg(a, b, kind="use", text=""): return {"a": a, "b": b, "kind": kind, "text": text}
win=dict(
  axis=70,                              # 보여 줄 거래일 수(오른쪽 끝 = 기준일 t). 8/30/70/270 중에서 고른다
  formula="ln( close[t] / close[t−20행] )",   # 아래 코드 줄
  rows=[dict(label="한 줄 설명", segs=[seg(-20, 0, "use", "20행")]), ...],  # 막대 줄들
  marks=[(-20, "t−20"), (0, "t")],      # 축 눈금 (오프셋, 글자)
  note="행 기준 주의·코드 한계 (없으면 빈 문자열)")
```
- `a`,`b`는 기준일 기준 오프셋(−60 = 60행 전, 0 = 오늘). `kind`: `use`(쓰는 창, 파랑) · `minus`(빼는 창, 주황) · `skip`(안 씀, 회색 빗금) · `fit`(적합 구간, 청록) · `need`(필요 조건, 점선 윤곽).
- 차분은 줄을 나눠(`use` 줄 + `minus` 줄 + 결과 줄) 보여 준다(`px_mom_20_60` 참고). 창이 좁아 글자가 안 들어가면 글자는 막대 옆에 자동 배치된다.
- 날짜 기준(`basis`)이 둘 이상이면(예: KR legacy vs 유효 세션) `BASIS`에 키를 만들어 카드마다 지정한다. md에 근거가 있는 것만 쓴다.

### 3.6 예시 종목

```python
EXAMPLES = {"px_mom_20_60": [dict(kind="p95"), dict(kind="p05")],
            "px_high_low_range_20d": [dict(kind="p95"), dict(kind="halt", ticker="010780"), dict(kind="p05")]}
EX_WINDOWS = {"px_mom_20_60": [dict(a=-60, b=-20, kind="use", text="쓰는 구간"), dict(a=-20, b=0, kind="skip", text="안 씀 (최근 20행)")]}
```
- `kind`: `p95`/`p05`는 기준일 해당 시장(KR=`EX_MARKET`, US=가격 5달러 이상 전체)의 순위 백분위가 목표에 가장 가까운(±0.5%p, 없으면 ±1%p) 종목 중 거래대금이 큰 종목을 자동으로 고른다. `ticker="..."`를 주면 그 종목을 직접 쓴다(`kind` 이름은 라벨용, `halt` 등).
- 예시는 **피쳐 수가 많은 문서에서는 대표 3~5개 피쳐에만** 건다(가격 차트가 의미 있는 피쳐: 창이 가격 위에 칠해지는 것). 수급·재무처럼 가격 창이 없는 피쳐는 예시 대신 `principle`을 쓴다.
- `EX_WINDOWS`의 오프셋은 **행 단위, 창 길이 N이면 a = −(N−1)**(오늘 포함 N행: `a=-19,b=0`). 창이 가격이 아니라 다른 계열(수급·공시)이면 `EX_WINDOWS`를 비워 두거나 예시를 걸지 않는다.
- 가격은 **기준일 이하만** 쓴다(빌더가 assert). 예시 종목 가격 차트는 종가·정지일 표시만 지원한다.

## 4. 원리 그림 `principle` (선택)

`type` 세 가지. 공통 선택 키: `title`(소제목), `caption`(그림 아래 한 줄). 그림은 SVG로 폭에 맞춰 줄바꿈된다. 한국어 글자 수를 짧게 쓴다(상자 하나에 글자 40자 이내).

### 4.1 `flow` — 단계 흐름

```python
principle=dict(type="flow", title="공매도가 값이 되기까지",
  steps=[dict(title="주식 빌림", text="증권사에서 주식을 빌립니다."),
         dict(title="시장에 팜", text="빌린 주식을 오늘 가격에 팝니다."),
         dict(title="나중에 사서 갚음", text="값이 내리면 싸게 사서 돌려줍니다.", hl=True, tag="여기서 재는 값")],
  measure="잔고가 많을수록 '하락에 거는 돈'이 많다는 뜻입니다.")
```
- `steps`는 2~5개. `hl=True`인 단계가 "여기서 재는 값"(`tag`로 문구 변경) 강조 상자. `measure`는 그림 맨 아래 한 줄.

### 4.2 `timeline` — 사건과 공개 지연

```python
principle=dict(type="timeline", title="13F가 우리에게 오기까지",
  events=[dict(at=0, label="분기 말", sub="보유 기준일"),
          dict(at=45, label="13F 공시", sub="법정 기한 45일"),
          dict(at=50, label="우리가 쓰는 날", sub="모델 입력", kind="today")],
  spans=[dict(a=0, b=45, label="지연 45일")],
  known={"from": 45, "to": 60, "label": "모델이 아는 시점"},
  caption="분기 말 보유는 약 45일 뒤에야 알 수 있습니다.")
```
- `at`은 임의 단위의 숫자(일 수 등, 같은 단위로). `kind`: `today`/`known`은 파랑 채움, 그 외는 빈 원. `spans`는 아래 주황 괄호 + 라벨("지연 N일"). `known`은 파랑 음영 영역(키 `from`은 파이썬 예약어가 아니므로 딕셔너리 리터럴 `{"from":..}`로 쓴다). `to`를 생략하면 오른쪽 끝까지.
- 이벤트는 위·아래로 번갈아 글자가 놓인다. 3~5개가 적당하다. **지연 일수는 md의 PIT·공개 지연 근거(공시 규칙·코드의 lag)에서만 옮긴다.**

### 4.3 `ratio` — 분자/분모 막대

```python
principle=dict(type="ratio", title="ROE: 분모가 음수면", unit="",
  rows=[dict(label="정상", numLabel="순이익", num=30, denLabel="자본", den=100, result="ROE = 30 ÷ 100 = +0.3"),
        dict(label="자본잠식", numLabel="순이익", num=-30, denLabel="자본", den=-10, result="ROE = −30 ÷ −10 = +3.0 → 적자인데 양수가 됩니다", note="부호가 같으면 나눈 값이 양수입니다.")])
```
- `num`·`den`은 숫자(음수 가능, 0이면 안 됨). 막대 길이는 모든 행의 최대 절댓값 기준. 양수 파랑·음수 주황. `result`는 행 아래 굵은 글씨, `note`는 그림 밑에 따로 나온다. `unit`은 값 뒤에 붙일 단위. 행 1~3개.

### 4.4 `analogy` — 쉬운 비유

카드당 한 문장(~80자). "~와 같습니다" 식으로 일상 장면에 비유한다. 정의를 바꾸는 비유는 쓰지 않는다(오해를 만든다). 모르면 생략한다.

**"더 쉽게 이해"가 목적이므로, 카드마다 analogy와 원리 그림(flow·timeline·ratio 중 맞는 것)을 가능하면 건다.** 함정 피쳐(README "코드로 확인한 것")는 `ratio`/`flow`로 함정을 그린다.

## 5. 모드별 화면과 숫자 (config가 건드리지 않는 것)

- **KR**: 지도 = x 평균|SHAP| × y 단독 섞기 ΔrankIC(0선), 곡선 = 확률 PDP(`subset=full/first/second`) + ICE. 카드의 숫자는 `shap_summary`·`importance_summary`·`pdp`에서 오고, md의 카드 줄과 09 표와 교차 검증한다.
- **US**: 지도 = x LightGBM 평균|SHAP| × y Ridge 평균|기여| + 합계 비율 기준선, 점 모양(둘 다 안정 ●/ 하나 ◆/ 없음 ○). 곡선 = LightGBM PDP(전체·앞·뒤)+ICE, 아래에 Ridge 직선 패널(기울기 = rank 계수, 순위 50%에서 표본 평균 예측을 지나도록 맞춤). **permutation 없음**, 라벨 안 읽음. 문구는 "모델 점수의 반응이지 수익 기여가 아님". 보조 판정(10 §1.1)은 곡선에서 다시 계산해 md와 맞춘다.
- 묶음 그림은 groups.json에서 이 문서의 피쳐가 든 묶음만 그린다(US id `g1`.. = groups.json 순서, KR `c01`..). 이 문서 밖 피쳐는 회색 원.

## 6. 검증 체크리스트 (끝내기 전에 모두)

1. `build_log_<이름>.txt`에 MISMATCH·no card line·no md row·no supp row **0건**. (KR/US 모두)
2. 스크린샷으로 본다. 헤드리스 Chrome:
   ```
   "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome" --headless=new --disable-gpu --enable-logging=stderr --v=0 --hide-scrollbars --window-size=1280,2300 --virtual-time-budget=5000 --screenshot=<out.png> file://<html>
   ```
   - **`| grep CONSOLE`로 콘솔 에러가 비어 있는지 확인.** (JS 문법 오류는 `Uncaught`로 나온다)
   - 창 높이 = 보고 싶은 만큼(최대 16000)이고 **위쪽부터 찍힌다**(URL `#앵커`는 스크롤되지 않는다). 아래쪽은 높이를 키워 찍고 Read로 본다(Read가 축소해 보여 줌). `sips -c`의 cropOffset은 중앙 기준이라 믿지 말고, 창 높이만 바꿔서 확인한다.
   - 폭 1280(다크 시스템 기본)과 **500**(헤드리스 최소 폭)을 본다. 라이트는 HTML 복사본에서 `<html lang="ko">`를 `<html lang="ko" data-theme="light">`로 바꿔 한 번 본다(저장소가 아닌 scratchpad에 둔다).
   - 볼 것: 글자 겹침(특히 원값 눈금·직접 라벨·묶음 그림 엣지 라벨), 상자 밖으로 나간 글자, 가로 넘침, 원리 그림 줄바꿈.
3. **숫자 대조**: 카드의 방향 ρ·순위·안정·섞기·모양은 빌더가 채운 것이라 로그 0건이면 일치. `what/model/caution`에 손으로 쓴 숫자는 결과 md 값과 직접 맞춘다.
4. **기준일 이후 가격 없음**: 예시 차트 마지막 날짜 = `REF_DATE`(빌더 assert + 로그 `last=` 확인). 정의에 쓰는 일 수가 가격 창에 들어가는지(예: 252행이면 `EX_DAYS`≥260).
5. **물결표**: 한국어 문서 규칙. 범위·근사 `~`는 md에서 `\~`로 쓰는 규칙이지만, **HTML에서는 렌더 문제가 없으므로 그냥 `~`를 써도 된다.** 단 **md 파일을 새로 쓰지는 않는다.** HTML 안 글자에 `\~`가 남아 있지 않은지 `grep -n '\\~' <html>`로 확인(md에서 옮겨 온 문장에서 `\~`가 섞이는 사고 방지).
6. **korean.md**: 합니다체, 결론 먼저, 짧게, 한자어 학술용어 풀어쓰기, `~에 대해서`·`~가능하다` 같은 번역투 금지, "쉽게 말하면"·"사실은" 같은 말버릇 금지, 이모지 금지. `analogy`는 일상어로.
7. 사실 검증: 정의·코드 인용·함정은 **md·README에 있는 것만** 쓴다. 지어내지 않는다. 확인 못 한 것은 "미확인"/"기록 없음". 지연 일수·공시 규칙은 md의 근거에서만.
8. 결과 문구: KR은 "진단이지 판정이 아님", US는 "모델 점수의 반응이지 수익 기여가 아님 · 라벨을 읽지 않음"이 머리말·카드에 들어 있는지.
9. 파일 크기 ~1.5MB 이내(HTML). 저장소에는 HTML 하나만. 생성기·data json·스크린샷은 scratchpad에.

## 7. 자주 나는 문제

- `no card line {id}` → md 카드의 "모델 반응 (층 3)" 행 형식이 달라 파싱 실패. 형식은 KR `섞기 ΔrankIC … · SHAP N위, 방향 ρ …, 모양 …(보조 …), 안정 · …`, US `LightGBM SHAP N위, ρ …, 모양 …(보조 …), 안정 … · Ridge N위, ρ …, 안정 …`. 카드 md를 고치지 말고 메인에게 알린다.
- `Uncaught SyntaxError … not valid JSON` → data json에 NaN. 빌더가 처리하므로 config에 NaN을 넣지 않았는지 확인.
- 한 장 지도에 점이 안 보임 → `PREFIX`(KR)가 피쳐 접두사와 맞는지, US는 `THEMES`의 피쳐 id가 `shap_summary`의 `feature`와 같은지(접두사 없는 원래 이름).
- 예시 종목이 비어 `row(0)` 오류 → 목표 백분위 ±1%p 안에 조건(261행, 하루 변동 31%/40% 미만, US는 기준일 거래)을 만족하는 종목이 없다. `kind` 대신 `ticker`를 직접 준다.
