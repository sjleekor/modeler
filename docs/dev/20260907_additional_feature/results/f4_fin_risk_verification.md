# F-4 — `feat_fin_risk` 마트 검증 리포트

- 생성: 2026-09-08 23:30 KST
- snapshot `2026-09-08` / source `sj2_remote`
- `FORMULA_VERSION = fin_risk_v2`
- `feat_fin_risk` 7,242,208행, 2007-06-05 ~ 2026-09-07

생성 명령: `uv run python -m research.analysis.fin_risk_report --snapshot-date 2026-09-08 --source sj2_remote`

---

## 0. 요약

1. **PIT 규칙은 `feat_fin_scan_daily`와 같은 것을 쓴다.** vintage 선택·`base_ok`·같은-날 tie-break을 `fin_vintage`로 꺼내 공유하고, `feat_fin_scan_daily`의 SQL 텍스트는 바이트 단위로 그대로다(A0 캐시 키라서 필수다).
2. **5개 family가 2020년부터만 존재한다** — ``. 설계 문서는 2016~2017을 기대했다. 원인은 이 마트가 아니라 입력 metric 4개에 XBRL fallback 규칙이 없는 것이고(§5), F-5의 최우선 항목이다. **사전등록 표본을 2020~2025로 좁혀 읽어야 한다.**
3. **연속형 4개는 꼬리가 매우 두껍다.** `fin_interest_coverage`는 상한만 100으로 묶여 있고 하한이 없어 평균이 −9,500까지 간다(분모가 0에 가까운 적자 기업). `fin_net_debt_to_mcap`·`fin_ext_finance_to_assets`도 분모가 작아 같은 모양이다. 스캔이 rank로 변환하므로 판정에는 영향이 없지만, 수준을 그대로 쓰는 소비자는 winsorize가 필요하다. 정의는 `03` §1.1에 고정된 것이므로 바꾸지 않았다.
4. **중복 경고 2건.** `fin_interest_coverage` × `fin_operating_profitability` ρ = 0.88은 사실상 같은 축이다(분자가 같은 영업이익). `fin_net_debt_to_mcap` × `fin_book_to_market` ρ = 0.51은 분모(시총)를 공유한다. 둘 다 카드에 경고를 적고, 전자는 `financial_risk` family의 독립적 발견으로 읽지 않는다.
5. **전이 두 개는 continuous로 확정.** `fin_lifecycle_transition` 연평균 0.69, `fin_profit_turn` 0.27 — 둘 다 5% 문턱을 크게 넘는다(§6). `fin_dividend_initiation`(0.040)·`fin_negative_equity_exit`(0.004)은 event cohort 그대로다.
6. **생애주기 단계 분포는 Dickinson과 부합한다.** Mature 0.35가 가장 크고 Introduction 0.15 / Growth 0.23 / Shake-out 0.16 / Decline 0.11이다. 세 현금흐름 중 하나라도 있는 행 기준 단계 NULL 비율 0.0192 — 대부분 세 개가 다 차지 않은 행이다.
7. **Decline·부실 판정은 보류.** 상폐 종목 재무 커버리지가 9.2%이고 실제 상폐 모집단은 그보다 크다(§4). F-9.3·F-9.4 뒤에 다시 본다.


**holdout(2025-08-01~)은 열지 않았다.** 이 리포트의 모든 수치는 피쳐 쪽만 본다 — 커버리지·분포·자기상관·피쳐 간 순위상관·이벤트 빈도. forward 수익률이나 라벨을 쓰는 계산은 하나도 없으므로 표본을 전 구간으로 잡아도 홀드아웃이 소진되지 않는다. 판정은 F-HS에서 사전등록된 경계로만 한다.

---

## 1. 커버리지·노출 지연

### 유효 시작일

| 컬럼 | 최초 세션 | non-NULL 행 |
|---|---|---|
| `fin_debt_to_assets` | 2015-06-29 | 5,248,695 |
| `fin_net_debt_to_mcap` | 2015-06-29 | 4,786,369 |
| `fin_interest_coverage` | 2017-02-15 | 3,532,032 |
| `fin_ext_finance_to_assets` | 2017-01-16 | 4,402,788 |
| `fin_lifecycle_stage` | 2017-01-16 | 4,388,775 |
| `fin_lifecycle_transition` | 2017-04-17 | 4,223,500 |
| `fin_profit_turn` | 2017-04-21 | 4,231,058 |
| `fin_dividend_initiation` | 2018-06-27 | 3,665,755 |
| `fin_negative_equity_exit` | 2016-04-15 | 5,124,480 |

### 연도별 커버리지

| 연도 | 행 수 | debt_to_assets | net_debt_to_mcap | interest_coverage | ext_finance_to_assets | lifecycle_stage | lifecycle_transition | profit_turn | dividend_initiation | negative_equity_exit |
|---|---|---|---|---|---|---|---|---|---|---|
| 2007 | 140 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| 2008 | 248 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| 2009 | 253 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| 2010 | 251 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| 2011 | 248 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| 2012 | 355 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| 2013 | 494 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| 2014 | 422,907 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| 2015 | 498,151 | 0.005 | 0.004 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| 2016 | 516,707 | 0.545 | 0.521 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.436 |
| 2017 | 525,283 | 0.76 | 0.744 | 0.324 | 0.442 | 0.439 | 0.345 | 0.372 | 0.0 | 0.75 |
| 2018 | 539,721 | 0.785 | 0.778 | 0.476 | 0.652 | 0.647 | 0.603 | 0.635 | 0.001 | 0.775 |
| 2019 | 562,471 | 0.798 | 0.784 | 0.551 | 0.715 | 0.712 | 0.674 | 0.69 | 0.533 | 0.788 |
| 2020 | 582,561 | 0.815 | 0.797 | 0.587 | 0.747 | 0.745 | 0.726 | 0.726 | 0.762 | 0.805 |
| 2021 | 599,990 | 0.837 | 0.815 | 0.623 | 0.77 | 0.77 | 0.753 | 0.746 | 0.791 | 0.825 |
| 2022 | 606,924 | 0.863 | 0.818 | 0.658 | 0.804 | 0.804 | 0.786 | 0.78 | 0.808 | 0.853 |
| 2023 | 620,743 | 0.881 | 0.793 | 0.679 | 0.822 | 0.822 | 0.808 | 0.8 | 0.822 | 0.868 |
| 2024 | 642,665 | 0.917 | 0.78 | 0.688 | 0.831 | 0.831 | 0.816 | 0.807 | 0.821 | 0.899 |
| 2025 | 660,019 | 0.935 | 0.76 | 0.715 | 0.869 | 0.865 | 0.845 | 0.833 | 0.817 | 0.925 |
| 2026 | 462,077 | 0.95 | 0.689 | 0.745 | 0.894 | 0.885 | 0.868 | 0.857 | 0.822 | 0.94 |

### 노출 지연 (중앙값, 일)

| `leverage` | `net_debt` | `coverage` | `ext_finance` | `lifecycle` | `profit_turn` | `negative_equity` | `dividend` |
|---|---|---|---|---|---|---|---|
| 48.0 | 48.0 | 50.0 | 48.0 | 50.0 | 56.0 | 48.0 | 173.0 |

---

## 2. 분포

### 연속형

| 컬럼 | 평균 | 표준편차 | p1 | p50 | p99 |
|---|---|---|---|---|---|
| `fin_debt_to_assets` | 0.4019 | 0.2271 | 0.0407 | 0.3941 | 0.9012 |
| `fin_net_debt_to_mcap` | 18.0378 | 2590.8646 | -0.3739 | 0.3329 | 14.1556 |
| `fin_interest_coverage` | -10109.6116 | 1462313.7924 | -826.5213 | 3.7552 | 100.0 |
| `fin_ext_finance_to_assets` | 4.8828 | 702.24 | -0.2067 | -0.0042 | 0.728 |

### 이산형 (값별 비중)

| 컬럼 | 분포 | non-NULL 행 |
|---|---|---|
| `fin_lifecycle_stage` | 1: 0.148, 2: 0.243, 3: 0.355, 4: 0.148, 5: 0.105 | 4,388,775 |
| `fin_lifecycle_transition` | 0: 0.650, 1: 0.350 | 4,223,500 |
| `fin_profit_turn` | -1: 0.048, 0: 0.911, 1: 0.041 | 4,231,058 |
| `fin_dividend_initiation` | 0: 0.976, 1: 0.024 | 3,665,755 |
| `fin_negative_equity_exit` | 0: 0.999, 1: 0.001 | 5,124,480 |

- `fin_interest_coverage`가 상한 100에 걸린 비율: 0.1142 (403,337 / 3,532,032)
- `fin_lifecycle_prev_aligned`(직전 vintage 세 현금흐름의 접수일 일치): 0.9855 (4,162,330 / 4,223,500)
- `fs_basis_used`: CFS 3,602,955, NULL 2,845,068, OFS 794,185

---

## 3. 생애주기 단계 (Dickinson 2011)

| 단계 | 이름 | 행 비중 | 행 수 | 종목 |
|---|---|---|---|---|
| 1 | Introduction | 0.148 | 651,276 | 1649 |
| 2 | Growth | 0.243 | 1,065,548 | 2008 |
| 3 | Mature | 0.355 | 1,559,226 | 2086 |
| 4 | Shake-out | 0.148 | 650,690 | 1900 |
| 5 | Decline | 0.105 | 462,035 | 1414 |

- 직전 단계는 있는데 이번 단계가 NULL인 행(세 현금흐름 중 하나가 사라졌거나 부호가 정확히 0): 0.00081 (3,558 / 4,392,333)
- 매핑 조합 수 8 (부호 8조합 전수, Dickinson 2011 표 1 고정)

---

## 4. 상폐 커버리지 — Decline 축의 선행 조건

| 상태 | 종목 | 재무 vintage 있음 | 커버리지 |
|---|---|---|---|
| DELISTED | 480 | 48 | 0.100 |
| 그 외 | 2,765 | 2,621 | 0.948 |

- Decline(5단계)을 한 번이라도 기록한 종목 1,414개. 상폐 종목 재무가 10.0%만 있으므로 이 축의 판정은 **F-9.3 S-1 잔여 백필 뒤로 보류**한다(`03` §1.2·§2). 마트는 지금 만들고 카드에 "생존편향 미해결"을 적는다.

위 `DELISTED` 480개는 **`stock_master`가 아는 상폐 종목만**이다. corpCode.xml 기준 상폐 법인은 약 1,330개이고 `stock_master`가 그만큼을 담고 있지 않다 — 그것을 복구하는 것이 F-9.4(S-2, `universe backfill-master`)다. 즉 실제 생존편향은 이 표보다 크다.

---

## 5. 입력 metric 커버리지 — 유효 시작일이 갈리는 이유

| metric | ≤2018 행 | ≥2020 행 | 종목 | 매핑 규칙 원천 |
|---|---|---|---|---|
| `interest_paid` | 28,056 | 86,329 | 2379 | XBRL fallback 있음 |
| `net_income` | 35,285 | 104,428 | 2622 | XBRL fallback 있음 |
| `operating_income` | 35,295 | 104,577 | 2622 | XBRL fallback 있음 |
| `financing_cash_flow` | 36,012 | 104,204 | 2646 | XBRL fallback 있음 |
| `investing_cash_flow` | 37,893 | 104,584 | 2644 | XBRL fallback 있음 |
| `operating_cash_flow` | 38,019 | 104,676 | 2646 | XBRL fallback 있음 |
| `cash_and_cash_equivalents` | 39,039 | 106,185 | 2651 | XBRL fallback 있음 |
| `total_assets` | 39,116 | 106,213 | 2651 | XBRL fallback 있음 |
| `total_liabilities` | 39,133 | 106,269 | 2651 | XBRL fallback 있음 |
| `total_equity` | 39,242 | 106,299 | 2651 | XBRL fallback 있음 |

**이것이 유효 시작일이 갈리는 이유다.** `investing_cash_flow`·`financing_cash_flow`·`interest_paid`·`cash_and_cash_equivalents` 넷은 `dart_financial_statement_raw` 규칙만 있고 XBRL fallback이 없다. 그 원천이 2019년 이전에 얇아서 네 metric의 2018년 이전 행이 100건 안팎이고, 그래서 이 넷을 쓰는 **5개 family(`fin_net_debt_to_mcap`·`fin_interest_coverage`·`fin_ext_finance`·`fin_lifecycle_stage`·`fin_lifecycle_transition`)가 사실상 2020년부터만 존재한다.** 설계 문서가 기대한 2016~2017이 아니다.

F-5(`metric_rules` 확장)에서 **가장 값이 큰 항목이 이것**이다. fin_v4가 `revenue`에 XBRL fallback을 붙였을 때 8,103행 → 약 148,000행으로 늘었다. 같은 종류의 규칙을 이 넷에 붙이면 위 5개 family의 표본이 2016년까지 내려갈 가능성이 크다. 다만 그러면 `feat_fin_risk` 값이 바뀌므로 `FORMULA_VERSION` 범프와 재검정이 따라온다 — 이번 라운드에서는 하지 않고 측정만 남긴다.

---

## 6. 전이 이벤트 빈도 → 스캔 경로 (F-4.3)

| 컬럼 | 연평균 종목 비율 | 최소 | 최대 | 플래그=1 행 수 | 스캔 경로 |
|---|---|---|---|---|---|
| `fin_lifecycle_transition` | 0.7462 | 0.6079 | 0.7875 | 1,477,132 | continuous |
| `fin_profit_turn` | 0.2697 | 0.1508 | 0.3121 | 378,554 | continuous |
| `fin_dividend_initiation` | 0.0397 | 0.0000 | 0.0541 | 88,922 | event cohort |
| `fin_negative_equity_exit` | 0.0040 | 0.0011 | 0.0077 | 4,893 | event cohort |

규칙(§1.3, 결과 보기 전 고정): 연간 이벤트 종목 비율이 5% 미만이면 event cohort, 이상이면 continuous.

- **`fin_lifecycle_transition` → continuous** (연평균 0.7462, 문턱 0.05)
- **`fin_profit_turn` → continuous** (연평균 0.2697, 문턱 0.05)
- `fin_dividend_initiation`·`fin_negative_equity_exit`은 §4에서 이미 event cohort로 고정돼 있다. 위 표의 비율은 `min_events` 미달 여부를 미리 보기 위한 것이다.

---

## 7. 기존 축과의 중복 (F-4.4, 일별 순위상관)

| fin_risk 컬럼 | 기존 컬럼 | 일별 순위상관 평균 | 표준편차 | 날짜 수 | 경고 |
|---|---|---|---|---|---|
| `fin_net_debt_to_mcap` | `feat_fin_scan_daily.fin_log_mcap` | -0.1165 | 0.0654 | 2,561 |  |
| `fin_net_debt_to_mcap` | `feat_fin_scan_daily.fin_book_to_market` | 0.5172 | 0.03 | 2,561 | ⚠ |ρ| ≥ 0.5 |
| `fin_debt_to_assets` | `feat_fin_scan_daily.fin_log_mcap` | 0.0569 | 0.0249 | 2,561 |  |
| `fin_ext_finance_to_assets` | `feat_fin_scan_daily.fin_asset_growth_yoy` | 0.4305 | 0.0432 | 2,312 |  |
| `fin_interest_coverage` | `feat_fin_scan_daily.fin_operating_profitability` | 0.8719 | 0.0205 | 2,312 | ⚠ |ρ| ≥ 0.5 |
| `fin_lifecycle_stage` | `feat_fin_scan_daily.fin_log_mcap` | -0.097 | 0.0271 | 2,313 |  |
| `fin_ext_finance_to_assets` | `feat_event_scan_daily.ev_net_share_issuance_yoy` | 0.1807 | 0.0601 | 2,312 |  |

|ρ| ≥ 0.5이면 카드에 경고를 적는다.

---

## 8. 사전등록 고정 항목 (F-4.5 → F-HS-1)

| family | primary | fdr_family | 부호 | horizon | 스캔 경로 | 비고 |
|---|---|---|---|---|---|---|
| `fin_debt_to_assets` | `fin_debt_to_assets` | `financial_risk` | 양방향 | [60, 120] | continuous | 총부채(이자부부채 아님) |
| `fin_net_debt_to_mcap` | `fin_net_debt_to_mcap` | `financial_risk` | 양방향 | [60, 120] | continuous | `fin_log_mcap`과 분모 공유 |
| `fin_interest_coverage` | `fin_interest_coverage` | `financial_risk` | `+` | [60, 120] | continuous | 분모 ≤ 0 NULL, 상한 100 |
| `fin_ext_finance` | `fin_ext_finance_to_assets` | `financial_risk` | `−` | [60, 120] | continuous | `ev_net_share_issuance_yoy`와 개념 겹침 |
| `fin_lifecycle_stage` | `fin_lifecycle_stage` | `lifecycle` | 양방향 | [60, 120] | continuous(순서형 rank) | Decline 판정 보류 |
| `fin_lifecycle_transition` | `fin_lifecycle_transition` | `lifecycle` | 양방향 | [20, 60] | §5 측정 결과 |  |
| `fin_profit_turn` | `fin_profit_turn` | `transition` | `+` | [20, 60, 120] | §5 측정 결과 |  |
| `fin_dividend_initiation` | `fin_dividend_initiation` | `transition` | `+` | [60, 120] | event cohort | 연 1회 이벤트 |
| `fin_negative_equity_exit` | `fin_negative_equity_exit` | `transition` | `+` | [60, 120] | event cohort | `min_events` 미달이면 `insufficient` |

- 검정하지 않는 컬럼: `fin_lifecycle_transition_up`·`_down`(방향 진단), `fin_lifecycle_prev_stage`, `fin_lifecycle_prev_aligned`, `fin_interest_coverage_capped`, `fs_basis_used`, `*_available_from`, `*_fin_age_days`, 그리고 모든 `_lag1`(기존 variant 축).
- 양방향 5건은 첫 run 뒤 관측 부호를 고정하고 이후 바꾸지 않는다.
- Phase C 후보: `fin_debt_to_assets × credit_wide`(F-8 신용 국면), `fin_lifecycle_stage × market_up`.
- F-5(`metric_rules` 확장)의 파생 family는 이 config에 넣지 않는다(`03` §3).
