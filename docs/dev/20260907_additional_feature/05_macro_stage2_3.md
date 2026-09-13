# 05. 매크로 2·3단계 — F-8

- 작성일: 2026-09-07
- 근거: `20260829_macro_features/00_survey/02_candidate_indicators_and_sources.md` §8 단계표(2단계: ECOS·FRED 시리즈 정의만, 3단계: 관세청 수출·SOX 어댑터), Phase C 결과(유동성·VIX 수준·시장 상태 국면 통과, 변동성·모멘텀 축 실패).
- 정본 코드: `definitions/common_features.py`(`CommonFeatureSeries` + `_feature`), `common seed`/`common sync`, `marts/common_build.py`, `horizon_scan_phase_c_regimes.py`.
- 목표: **국면 변수를 신용·기대인플레·정책금리 축으로 넓히고**, F-4 레버리지 family와 짝을 만든다. 매크로 level을 횡단면 피쳐로 넣는 것은 여전히 하지 않는다(`06_macro_regime.md` §1).

---

## 1. 2단계 — 시리즈 정의만 (새 어댑터 없음)

| series_id | 원천 | stat/series 코드 | 빈도 | availability | feature_code(파생) | 용도 |
|---|---|---|---|---|---|---|
| `rate_kr_corp_aa3y` | ECOS `817Y002` | 회사채(무보증 3년) AA- (item_code 확인) | D | `next_krx_session` | `rate_kr_corp_aa3y_level`, **`credit_spread_aa_gov3y`** = AA- − 국고3년 | 신용 국면 |
| `rate_kr_corp_bbb3y` | ECOS `817Y002` | 회사채 BBB- (item_code 확인) | D | `next_krx_session` | `rate_kr_corp_bbb3y_level`, **`credit_spread_bbb_aa`** | 저신용 스프레드 |
| `rate_kr_cd91` | ECOS `817Y002` | CD 91일 | D | `next_krx_session` | `rate_kr_cd91_level`, `rate_kr_cd_gov3y_spread` | 단기 자금 |
| `rate_kr_base` | ECOS `722Y001` | 한국은행 기준금리 | 변경 시(일 격자로 forward-fill) | `next_krx_session` | `rate_kr_base_level`, `rate_kr_base_chg_63d`(분기 변화) | 정책 국면 |
| `macro_esi` | ECOS `513Y001` | 경제심리지수 | M | `manual_lag_days` | `macro_esi_level`, `macro_esi_chg_3m` | 심리 |
| `rate_us_baa10y` | FRED `BAA10Y` | Moody's Baa − 10년 국채 | D | `same_krx_session_morning` | `credit_spread_us_baa` | 글로벌 신용 |
| `rate_us_hy_oas` | FRED `BAMLH0A0HYM2` | ICE BofA HY OAS | D | `same_krx_session_morning` | `credit_spread_us_hy` | 글로벌 위험선호 |
| `fx_usd_broad` | FRED `DTWEXBGS` | 광의 달러 지수 | D | `same_krx_session_morning` | `fx_usd_broad_level`, `_ret_20d` | 달러 국면 |
| `rate_us_breakeven10y` | FRED `T10YIE` | 10년 기대인플레 | D | `same_krx_session_morning` | `rate_us_breakeven10y_level` | 인플레 기대 |
| `macro_us_epu` | FRED `USEPUINDXD` | 경제정책 불확실성(일별) | D | `same_krx_session_morning` | `macro_us_epu_level`, `_ma20` | 불확실성 |

- ECOS `item_code`는 `common sync --sources ecos` 전에 **실호출로 확정**한다(N8-2에서 `item_code2` 미지정 시 원계열·계절조정이 섞여 온 사례). PoC 산출물 `poc/ecos_stage2_items.md`.
- 시리즈 추가는 `default_common_feature_series()`에 `CommonFeatureSeries(...)` 한 항목씩, 파생은 `default_common_feature_catalog()`에 `_feature(...)`/`_multi_input_feature(...)`. `commodity_wti_spot_level` 추가 때 확인한 대로 **행 추가만이면 golden이 바뀌지 않는다**(`02_stage1a` §2.5). 스프레드는 `rate_kr_term_spread_10y_3y`와 같은 `_multi_input_feature` 패턴.
- 백필: `common sync --sources ecos,fred --from 2014-01-01`. `history_start_date`를 시리즈마다 적는다. readiness 창 정렬 문제(N8-2: 창 시작을 2014-06-02로 두면 늦게 시작한 계열이 미달로 잡힘)는 시리즈별 `history_start_date`로 이미 처리된다 — 새 시리즈도 같은 필드를 채운다.
- 운영: 20:30 ECOS/FRED/FDR common 이벤트가 그대로 받는다. freshness 예산은 시리즈별 `max_stale_business_days`.
- **inactive 시리즈 정리**(N8 §B3)도 이때 같이 한다: `market_kospi`·`market_kosdaq`·`market_kospi200`(pykrx), `industry_*` 4종(데이터 0), `fx_usdkrw`(FDR), `macro_employed_persons`, `commodity_wti_fred_ret_20d`. 지우지 않고 `active=False` 유지 + 카탈로그 주석. 지우면 과거 fact 재현이 깨진다.

---

## 2. 국면 변수 후보 (Phase C 2라운드 사전등록 초안)

정의는 1라운드 규칙(과거 창만, 전체 표본 중앙값 금지, KRX 세션 격자, 이진 `1[z > 0]`)을 그대로 따른다.

| id | 이름 | `z_t` | `s_t=1` 뜻 | 원천 | 역할 |
|---|---|---|---|---|---|
| R7 | `credit_wide` | `spread_aa_gov3y_t − median_252` | 신용 스프레드가 1년 중앙값보다 넓음 | `credit_spread_aa_gov3y` | primary |
| R8 | `credit_widening` | `spread_aa_gov3y_t − spread_{t−20}` | 20세션 확대 | 같음 | primary |
| R9 | `hy_stress` | `credit_spread_us_hy_t − median_252` | 글로벌 HY 스트레스 | `credit_spread_us_hy` | exploratory |
| R10 | `policy_tightening` | `rate_kr_base_t − rate_kr_base_{t−126}` | 반년 안 기준금리 인상 | `rate_kr_base_level` | exploratory |
| R11 | `usd_strong_20d` | `ln(usd_broad_t / usd_broad_{t−20})` | 달러 20일 강세 | `fx_usd_broad_level` | exploratory(`krw_weak_20d`와 겹침 확인) |

- 점유율(G1 0.40~0.60)·지속·기간별 유효(G2 양쪽 ≥ 40일)를 **hash 전에** 계산해 기록한다(1라운드 `05_preregistration_record.md` 절차).
- 국면 사이 상관(`credit_wide` vs `vix_high` 등)을 진단으로 낸다. |ρ| ≥ 0.7이면 같은 축이라 쌍 수를 줄인다.

---

## 3. 검정 쌍 초안 (2라운드)

| 쌍 | family × 국면 | 방향 | 근거 |
|---|---|---|---|
| Q1 | `fin_debt_to_assets`(F-4) × `credit_wide` | `−`(신용 경색에서 고레버리지 종목이 더 나쁨 → δ 음수) | 부실위험 문헌 + 신용 사이클 |
| Q2 | `fin_interest_coverage`(F-4) × `credit_wide` | `+` | 같은 축의 반대편 |
| Q3 | `fin_log_mcap` × `credit_wide` | 양방향 | 소형주가 신용 경색에 취약한가. 1라운드 X1·X2(시장 방향·KOSDAQ 상대)에서는 국면 무관 |
| Q4 | `px_idio_vol_60d` × `credit_wide` | 양방향 | 1라운드 `vix_up` 실패의 대안 축 |
| Q5 | `flow_foreign_netbuy_to_volume` × `usd_strong_20d` | 양방향 | 외국인 수급과 달러(1라운드 `krw_weak_20d`는 exploratory였다) |
| Q6 | `ev_payout_yield` × `policy_tightening` | 양방향 | 금리 인상기 배당주 |
| X3 (reference) | `px_reversal_5d` × `credit_wide` | — | 1라운드 통과 쌍의 다른 국면 — 대조군 |

- 쌍 수는 F-4 완료 뒤 확정한다. F-4 없이 2라운드를 열면 Q1·Q2가 빠져 신용 국면의 핵심 가설이 없다. **2라운드는 F-4·F-8 둘 다 끝난 뒤 한 번.**
- 1라운드 통과 4쌍은 다시 검정하지 않는다(결과를 보고 재검정하지 않는다).

---

## 4. 3단계 — 새 어댑터 (D-F5 확정 2026-09-07: 2단계 먼저, 3단계는 2단계 국면이 Phase C에서 통하면)

| 원천 | 내용 | 어댑터 | PIT | 비고 |
|---|---|---|---|---|
| 관세청 수출입 잠정치(공공데이터포털 `15157908`) | 10일 단위 수출액 | `common_features_datago`(신규). `DATAGO_KEY`는 로컬 `.env`에만 있고 prod 미반영, `Settings` 필드는 있다 | 발표일(익일 오전) + 1세션 | 수출 국면 → `macro_beta_export`(exposure) 후보 |
| SOX(필라델피아 반도체) | 지수 레벨·수익률 | FDR 익명 경로(KRX 아님, 허용) — `common_features_fdr`에 시리즈 추가로 될 수 있어 어댑터가 필요 없을 수도 | `same_krx_session_morning` | 반도체 비중이 큰 시장의 spillover exposure `macro_beta_sox` |

- 3단계 산출물은 국면보다 **exposure 베타 family**(`macro_beta_export`, `macro_beta_sox`)가 맞다. 1라운드 1a와 같은 산식(잔차 252창, 최소 짝 126, `feat_macro_exposure` 확장)으로 넣는다. `feat_macro_exposure`의 `FORMULA_VERSION`이 바뀌므로 새 config.
- 착수 조건: 2단계 국면 중 하나 이상이 Phase C 2라운드에서 `screen_pass`, 또는 모델 E5에서 국면 변수의 기여가 확인될 때.

---

## 5. 하지 않는 것

- 매크로 level·Δ·YoY를 Phase B family로 등록(1라운드 결정 유지).
- 국내 원천의 사후 개정(vintage) 처리 설계 — 보류 목록(경기지수·산업생산·경상수지)은 그대로 보류.
- 기존 국면 정의 변경·재검정.
- 유료·비공개 원천(CDS, DRAM 가격, PMI).
