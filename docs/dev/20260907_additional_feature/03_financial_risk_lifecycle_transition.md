# 03. 재무위험·생애주기·상태 전이 — F-4 (canonical만) · F-5 (`metric_rules` 확장)

- 작성일: 2026-09-07
- 근거: `11_feature_taxonomy.md` §5(전이 축 T3 = 0), §9.1(레버리지 0, 원천은 있음), §9.2(Dickinson 생애주기), §9.3(무형·R&D). 한국 복제율(Han-Lee-Kang 2020): 투자 24%, 수익성 5% — **기대값을 낮게 잡는다.**
- 원천: `stock_metric_vintage_fact`(strict PIT vintage, `fin_scan.py`가 쓰는 것). canonical metric 29종에 `total_liabilities`·`total_equity`·`total_assets`·`cash_and_cash_equivalents`·`operating_cash_flow`·`investing_cash_flow`·`financing_cash_flow`·`interest_paid`·`operating_income`·`net_income`·`revenue`·`sga`가 있다(`definitions/metric_rules.py`). **F-4는 새 수집이 없다.**

---

## 1. F-4 — family 정의 (9)

마트 `feat_fin_risk`(`research/etl/features/fin_risk.py`). `fin_scan.py`의 vintage 선택·`base_ok`·`available_from` 규칙을 함수로 공유해 **같은 PIT 규칙**을 쓴다. `FORMULA_VERSION = "fin_risk_v1"`.

### 1.1 레버리지·재무위험 (4)

| family | primary | 산식 (TTM은 최근 4분기 합, BS는 최근 분기) | 기대 부호 | horizon |
|---|---|---|---|---|
| `fin_debt_to_assets` | `fin_debt_to_assets` | `total_liabilities / total_assets` | **양방향**(문헌 부실위험은 `−`, 위험 보상 논리는 `+`. 한국 근거 없음 — `11` §11 규칙 3) | [60, 120] |
| `fin_net_debt_to_mcap` | `fin_net_debt_to_mcap` | `(total_liabilities − cash_and_cash_equivalents) / market_cap_pit` | 양방향 | [60, 120] |
| `fin_interest_coverage` | `fin_interest_coverage` | `operating_income_ttm / interest_paid_ttm`, 분모 ≤ 0이면 NULL, 상한 100으로 클립 | `+`(이자보상 높을수록 건전) | [60, 120] |
| `fin_ext_finance` | `fin_ext_finance_to_assets` | `financing_cash_flow_ttm / avg_total_assets`(외부 자금 의존. 차입·발행 세부가 없어 재무활동 현금흐름 합계로 근사) | `−`(외부 조달 많을수록 이후 수익 낮음 — 발행 이례현상 계열) | [60, 120] |

- `total_liabilities`는 이자부부채가 아니라 총부채다(N2 V6과 같은 한계). F-5에서 `borrowings`가 열리면 `fin_net_debt_to_mcap`의 분자를 차입금으로 바꾼 **새 family**를 만든다(정의 변경 아님).
- 금융·지주는 부채 개념이 다르다. 업종 코드가 소급값이라 제외 정책은 **진단으로만**(KSIC 64~66 제외 표본에서 부호 재확인) 낸다.

### 1.2 생애주기 (2)

Dickinson(2011) 현금흐름 부호 패턴. 8조합 → 5단계 매핑은 원 논문대로 **고정**한다.

| CFO | CFI | CFF | 단계 |
|:--:|:--:|:--:|---|
| − | − | + | Introduction (1) |
| + | − | + | Growth (2) |
| + | − | − | Mature (3) |
| − | − | − | Shake-out (4) |
| + | + | + | Shake-out (4) |
| + | + | − | Shake-out (4) |
| − | + | + | Decline (5) |
| − | + | − | Decline (5) |

| family | primary | 산식 | 기대 부호 | horizon |
|---|---|---|---|---|
| `fin_lifecycle_stage` | `fin_lifecycle_stage` | TTM 3대 현금흐름 부호로 정한 단계(1~5). 순서형으로 rank | **양방향**(원 논문: Mature 저평가 → 3단계가 높음. 단조가 아니라 rank IC로는 방향을 못 박는다) | [60, 120] |
| `fin_lifecycle_transition` | `fin_lifecycle_transition` | 직전 vintage 대비 단계가 바뀌었으면 1, 아니면 0 | 양방향(전이 자체가 정보. 방향은 exploratory `_up`/`_down` 분해로 진단) | [20, 60] |

- TTM 부호로 잰다(분기 부호는 계절성으로 튄다). 이 선택은 고정.
- **Decline 단계는 상장폐지 직전 기업이 많다.** S-1 잔여·S-2(상폐 법인 재무·가격 백필, `06` §3)가 끝나기 전에는 Decline 관련 결론을 내지 않는다(`11` §11 규칙 6). 마트는 만들되 카드에 "생존편향 미해결"을 적는다.

### 1.3 상태 전이 (3)

| family | primary | 산식 | 기대 부호 | horizon |
|---|---|---|---|---|
| `fin_profit_turn` | `fin_profit_turn` | `net_income_ttm` 부호가 직전 vintage 대비 −→+면 +1, +→−면 −1, 아니면 0 | **`+`**(적자→흑자 전환은 지연 반영된다는 가설) | [20, 60, 120] |
| `fin_dividend_initiation` | `fin_dividend_initiation` | 최근 3개 사업연도 `dps`(`dart_shareholder_return_raw`) = 0이었다가 이번에 > 0이면 1. 노출은 배당 공시 접수 다음 세션 | `+` | [60, 120] |
| `fin_negative_equity_exit` | `fin_negative_equity_exit` | `total_equity`가 직전 vintage ≤ 0 → 이번 > 0이면 1 | `+` | [60, 120] |

- 이벤트 성격(0이 대부분, 1이 드물다). Phase B의 event 스캔 경로(`fin_sue`처럼 event cohort)가 맞는지, continuous 스캔이 맞는지는 **event 빈도로 정한다**: 연 이벤트 수가 종목의 5% 미만이면 event cohort 스캔(`event_buckets`), 이상이면 continuous. 첫 마트 빌드에서 빈도를 재고 등록 전에 고정한다.
- 비율형 YoY로는 표현할 수 없는 부호 전환(`11` §5.1)을 정면으로 다룬다.

### 1.4 Δ 피쳐 — 이번에는 넣지 않는다

`fin_gross_profitability_chg_yoy` 같은 Δ는 원 수준 family가 B/C이고 표본이 0.58이라 정보량 대비 가설 수만 늘린다. `fin_profit_turn`이 가장 중요한 Δ(부호 전환)를 대신한다. 필요하면 다음 라운드.

---

## 2. PIT와 표본

| 항목 | 규칙 |
|---|---|
| vintage | `fin_scan.py`와 같은 strict PIT(`DEFAULT_VINTAGE_POLICY`, 접수 다음 세션, 같은 날 후보는 최신 회계기간) |
| TTM | 최근 4개 분기 vintage 합. 4분기 미만이면 NULL |
| 직전 vintage | 같은 `(ticker, metric)`의 바로 전 `available_from` 행 |
| `dps` | `dart_shareholder_return_raw`, 사업연도 기준. 접수 `rcept_no` 날짜 다음 세션 노출(`ev_payout_yield`와 같은 규칙) |
| 유효 시작 | TTM·직전 vintage 요구로 2016~2017. `fin_asset_growth_yoy`(2016-06-27)와 비슷할 것 |
| 상폐 법인 | S-1 잔여 백필 전에는 재무가 약 1,300 법인 빠져 있다. **Decline·부실 축 판정 보류** |

---

## 3. F-5 — `metric_rules` 확장

canonical에 없는 계정 넷을 매핑한다. **수집이 아니라 매핑**이다(`dart_financial_statement_raw`·`dart_xbrl_fact_raw`에 계정이 있을 가능성이 높다 — 인벤토리 §2.4).

| metric_code | 계정 | XBRL 태그 후보(PoC에서 확정) | 여는 것 |
|---|---|---|---|
| `current_assets` | 유동자산 | `ifrs-full_CurrentAssets` | 유동비율, Altman Z 운전자본항, Piotroski |
| `current_liabilities` | 유동부채 | `ifrs-full_CurrentLiabilities` | 같음 |
| `borrowings` | 차입금(단기+장기) | `ifrs-full_Borrowings`, `ifrs-full_ShorttermBorrowings`, `ifrs-full_LongtermBorrowings` — 합산 규칙 필요 | 이자부부채 기준 레버리지 |
| `rnd_expense` | 경상연구개발비 | `dart_ResearchAndDevelopmentExpense` 계열(dart 확장 태그, 표준 재무제표 주석에 있을 수 있음) | 무형자산 조정 B/M(Peters-Taylor), R&D/매출 |

### 3.1 절차

1. **PoC — 커버리지 측정**: `dart_xbrl_fact_raw`에서 후보 태그별 `(corp, period)` 커버리지를 잰다(2015~2025). 50% 미만이면 fallback 규칙(`_xbrl_fallback_rule`)과 `dart_financial_statement_raw` 계정명 매칭(`_financial_rule`)을 추가한다. `fin_gross_profitability`의 역산값 94.4% 사례처럼 **역산 비율을 카드에 적는다.**
2. `definitions/metric_rules.py`에 `MetricCatalogEntry` + 매핑 규칙 추가. **기존 29 metric의 규칙은 건드리지 않는다**(golden parity).
3. `bin/parquet-compute-all.sh`로 `stock_metric_fact`·vintage 재빌드. 기존 metric 값 불변을 golden으로 확인.
4. `feat_fin_risk`에 파생 family 추가(새 `FORMULA_VERSION`): `fin_current_ratio`, `fin_borrowings_to_mcap`, `fin_altman_z`(축소판이 아닌 완전판이 되면), `fin_rnd_to_sales`, `fin_bm_intangible_adj`.

F-5의 파생 family는 **F-4와 같은 F-HS config에 넣지 않는다.** PoC 결과에 따라 정의가 바뀔 수 있어 다음 config다.

---

## 4. 사전등록 초안 (F-HS)

| family | fdr_family | 부호 | horizon | 스캔 경로 | 비고 |
|---|---|---|---|---|---|
| `fin_debt_to_assets` | `financial_risk` | 양방향 | [60, 120] | continuous | |
| `fin_net_debt_to_mcap` | `financial_risk` | 양방향 | [60, 120] | continuous | `fin_log_mcap`과 분모 공유 — 상관 진단 |
| `fin_interest_coverage` | `financial_risk` | `+` | [60, 120] | continuous | 분모 ≤ 0 NULL |
| `fin_ext_finance` | `financial_risk` | `−` | [60, 120] | continuous | `ev_net_share_issuance_yoy`와 개념 겹침 — 상관 진단 |
| `fin_lifecycle_stage` | `lifecycle` | 양방향 | [60, 120] | continuous(순서형 rank) | Decline 보류 |
| `fin_lifecycle_transition` | `lifecycle` | 양방향 | [20, 60] | 빈도로 결정 | |
| `fin_profit_turn` | `transition` | `+` | [20, 60, 120] | 빈도로 결정 | |
| `fin_dividend_initiation` | `transition` | `+` | [60, 120] | event cohort | 연 1회 이벤트 |
| `fin_negative_equity_exit` | `transition` | `+` | [60, 120] | event cohort | 드물다. `min_events` 미달이면 `insufficient` |

Phase C 후보: `fin_debt_to_assets × credit_wide`(F-8 신용 국면, 레버리지가 신용 경색에서 벌 받는가), `fin_lifecycle_stage × market_up`.

---

## 5. 검증할 것 (마트 단위)

- vintage 규칙이 `fin_scan.py`와 동일(테스트: 같은 종목·날짜에서 `total_assets` 값 일치).
- TTM 합에 분기 누락이 있으면 NULL(4개 미만).
- 생애주기 매핑표 8조합 전부 테스트.
- 전이 플래그의 이벤트 빈도 리포트(연도별 1의 비율) → 스캔 경로 결정 근거로 기록.
- `fin_net_debt_to_mcap`·`fin_ext_finance`와 기존 규모·발행 family의 일별 순위상관.
- 상폐 법인 커버리지(S-1 잔여 전/후) 차이.

---

## 6. 하지 않는 것

- Piotroski F-score 9항목·Altman Z 완전판: F-5의 유동자산·유동부채가 열려야 한다.
- 무형자산 조정 밸류: F-5 R&D가 열려야 한다.
- 정의 여러 개(부채 범위, TTM vs 분기, 8패턴 vs 5단계) 만들어 고르기.
- 금융·지주 제외를 판정에 반영(업종이 소급값이라 진단만).
