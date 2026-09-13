# 02. 피쳐 집합과 전처리

- 작성일: 2026-09-07
- 근거: `docs/dev/20260830_feature_summary/01_feature_inventory.md`(컬럼 목록), `08_quality_summary_gaps.md` §4(horizon별 배치), `06_macro_regime.md` §5(interaction)
- 코드 위치(신규): `research/models/_02_updown_prob/features.py` — 피쳐 집합은 **코드에 컬럼 목록으로 고정**한다. YAML로 두면 실행 중에 바뀔 수 있다.

---

## 1. 피쳐 집합 — 사전등록

### 1.1 FS0 — baseline 40 (기존 모델과 같다)

`feat_price` 15 + `feat_flow` 15 + `feat_fin_pit` 10. `run_grade_a_acceptance_gate.py`의 `BASELINE_COLS`와 글자 하나까지 같아야 한다(테스트로 고정).

| 마트 | 컬럼 |
|---|---|
| `feat_price` | `px_ret_1d`, `px_ret_5d`, `px_ret_20d`, `px_ret_60d`, `px_mom_20_60`, `px_vol_20d`, `px_vol_60d`, `px_high_low_range_20d`, `px_turnover`, `px_turnover_ma20`, `px_amihud_20d`, `px_gap_vs_ma20`, `px_dist_52w_high`, `px_is_halted`, `px_halt_ratio_20d` |
| `feat_flow` | `flow_foreign_netbuy_sum_5d`, `_20d`, `flow_inst_netbuy_sum_5d`, `_20d`, `flow_indiv_netbuy_sum_5d`, `_20d`, `flow_foreign_holding_chg_5d`, `_20d`, `flow_short_balance_chg_20d`, `flow_foreign_netbuy_z_20d`, `flow_inst_netbuy_z_20d`, `flow_short_avg_price`, `flow_short_selling_volume`, `flow_short_selling_value`, `flow_short_balance_qty` |
| `feat_fin_pit` | `fin_roa`, `fin_roe`, `fin_operating_margin`, `fin_debt_to_equity`, `fin_equity_ratio`, `fin_ocf_to_assets`, `fin_cash_ratio`, `fin_asset_turnover`, `fin_is_negative_equity`, `fin_has_fs` |

FS0의 역할은 **비교 기준**이다. 이 40개는 단변량 검정을 거치지 않았고, `fin_pit`의 PIT 규칙(+90/+45일 보수 지연)은 검증 트랙과 다르다. 그래도 바꾸지 않는다 — 바꾸면 기존 게이트 결과와 이어지지 않는다.

### 1.2 FS1 — 검증 통과 피쳐 추가 (56)

FS0 + T1 후보 5 + T2 후보 14에서 중복 3을 뺀 11.

| 출처 | 추가 컬럼 | 뺀 것과 이유 |
|---|---|---|
| T1 Grade A | `px_reversal_5d`, `px_maxret_20d`, `px_idio_vol_60d`, `flow_individual_netbuy_to_volume_5d`, `flow_individual_netbuy_to_volume_20d` | `px_amihud_20d`·`px_near_52w_high`는 FS0에 이미 있다(후자는 `px_dist_52w_high`와 같은 산식) |
| T2 screen_pass 14 | `fin_log_mcap`, `fin_value_z`, `fin_gross_profitability`, `ev_amendment_ratio_1y`, `ev_filing_burst_60d`, `ev_net_share_issuance_yoy`, `ev_payout_yield`, `own_major_filing_60d`, `own_major_stake_chg`, `hc_employee_growth_yoy`, `hc_revenue_per_employee` | **`mcap_krx_log`**(`fin_log_mcap`과 같은 개념. `fin_log_mcap`은 시간 placebo를 실제로 통과한 유일한 Phase B A등급이라 그쪽을 남긴다), **`own_amendment_ratio_1y`**(`ev_amendment_ratio_1y`의 부분집합), **`own_major_stake`**(`own_major_stake_chg`가 모든 지표에서 낫다) |

기대 부호(단조 제약 변형 E4b에 쓴다): `px_reversal_5d +`, `px_maxret_20d −`, `px_idio_vol_60d −`, `flow_individual_* +`, `fin_log_mcap −`, `fin_value_z +`, `fin_gross_profitability +`, `ev_amendment_ratio_1y −`, `ev_filing_burst_60d −`(관측), `ev_net_share_issuance_yoy −`, `ev_payout_yield +`, `own_major_filing_60d −`(관측), `own_major_stake_chg +`(관측), `hc_* +`(관측). 부호는 라벨이 "상승"이므로 IC 부호와 같다.

### 1.3 FS1h — horizon별 부분집합 (E2 비교용)

`08_quality_summary_gaps.md` §4의 배치를 그대로 코드로 옮긴다. FS1에서 h에 맞지 않는 것을 뺀다.

| h | FS1에서 빼는 것 | 남는 수 |
|---|---|---|
| 5 | `fin_*` 3, `ev_amendment/payout/issuance`, `own_*`, `hc_*` (사전등록 60·120) | 40 + 6 = 46 |
| 20 | `fin_value_z`, `fin_gross_profitability`, `ev_payout_yield`, `ev_net_share_issuance_yoy`, `hc_*`, `own_major_stake_chg` | 40 + 9 = 49 |
| 60 | `px_reversal_5d`, `flow_individual_netbuy_to_volume_5d` (half-life 5~10) | 40 + 14 = 54 |
| 120 | `px_reversal_5d`, `px_maxret_20d`, `flow_individual_*` 2, `ev_filing_burst_60d`, `own_major_filing_60d` | 40 + 10 = **50** |

**h120 합계 정정 (2026-09-09).** 이 행은 원래 `40 + 9 = 49`로 적혀 있었다. 빼는 컬럼 6개를 세면
FS1 추가 16개 중 10개가 남으므로 50이 맞다. 49를 맞추려면 컬럼 하나를 더 골라 빼야 하는데 그것은
개수에 맞춰 피쳐를 고르는 것이라 §5의 규율에 어긋난다. **컬럼 목록이 정본이고 합계 표기가 오타였다**로
확정했다(`features.py`의 `FS1H_DROPPED`가 이 목록이다).

FS1과 FS1h 중 어느 쪽이 나은지는 E2가 답한다. 트리 모델이 관련 없는 피쳐를 스스로 무시하면 FS1이, 가설 수 증가가 과적합으로 이어지면 FS1h가 이긴다.

### 1.4 FS2 — 매크로·국면·interaction 추가 (68)

FS1 + 12.

| 종류 | 컬럼 | 출처 마트 | 비고 |
|---|---|---|---|
| exposure 베타 4 | `macro_beta_vix`, `macro_beta_wti`, `macro_beta_sp500_lag`, `px_market_beta` | `feat_macro_exposure` | `usdkrw`·`kr10y`·`rawbeta`·`semibeta`는 넣지 않는다(신호 없음·시장 베타 변장) |
| 국면 이진 4 | `rg_vix_up`, `rg_vix_high`, `rg_market_up`, `rg_liq_high` | `dim_regime_daily`(신규, §3) | 날짜 상수. 순위 라벨에서는 단독 정보 0이지만 트리의 분기 조건으로 산다 |
| interaction 4 | `ix_reversal_vixhigh = rank(px_reversal_5d) × rg_vix_high` · `ix_tshock_liqhigh = rank(px_turnover_shock) × rg_liq_high` · `ix_foreign_liqhigh = rank(flow_foreign_netbuy_to_volume_20d) × rg_liq_high` · `ix_mktbeta_mktup = rank(px_market_beta) × rg_market_up` | 빌더(§3) | Phase C 통과 4쌍. `rank`는 `(date, market)` 내 백분위(0~1), 국면이 0인 날은 0 |

interaction의 재료 `px_turnover_shock`·`flow_foreign_netbuy_to_volume_20d`는 단변량에서 D등급(반대 부호 안정)이다. **재료 자체는 FS2에 넣지 않고 interaction만 넣는다.** 재료를 넣고 싶으면 부호를 뒤집은 별도 변형(E4c)이다.

### 1.5 FS3 — 병행 스트림에서 오는 것 (E5, **컬럼 확정 2026-09-09**)

`20260907_additional_feature/`의 **F-HS-1**이 `screen_pass`한 신규 11 컬럼이다. 출처는 config
`3ca949e6a2275706c172da639469b77477f2c48430f8681ec462b17ebde6e4ae`, snapshot `2026-09-08`의
A→B→AB run이고 기록은 `20260907_additional_feature/results/f_hs1_run_20260909.md`다. FS2 68 + 11 = **79**.

| 등급 | 컬럼 | 마트 | 기대 부호 | 검정 horizon | 최대 \|IC\| |
|---|---|---|---|---|---|
| A | `rel_peer_dispersion_20d` | `feat_relation_stat` | − | 20, 40, 60 | 0.0615 |
| A | `rel_own_minus_peer_20d` | `feat_relation_stat` | − | 5, 10, 20 | 0.0564 |
| A | `rel_peer_mom_20d` | `feat_relation_stat` | − | 20, 40, 60 | 0.0227 |
| A | `rel_peer_bigcap_lag_ret_5d` | `feat_relation_stat` | + | 10, 20 | 0.0065 |
| B | `fin_net_debt_to_mcap` | `feat_fin_risk` | + | 60, 120 | 0.0644 |
| B | `fin_ext_finance_to_assets` | `feat_fin_risk` | − | 60 | 0.0487 |
| B | `fin_interest_coverage` | `feat_fin_risk` | + | 60 | 0.0384 |
| B | `fin_lifecycle_stage` | `feat_fin_risk` | + | 60 | 0.0150 |
| B | `fin_debt_to_assets` | `feat_fin_risk` | − | 60 | 0.0130 |
| B | `fin_profit_turn` | `feat_fin_risk` | + | 20 | 0.0068 |
| B | `fin_lifecycle_transition` | `feat_fin_risk` | − | 20, 60 | 0.0059 |

부호는 F-HS-1이 **첫 run 뒤 고정한 값**이고 이후 바꾸지 않는다(양방향으로 등록했던 6건 포함, 뒤집힌
family는 없었다). 라벨이 "상승"이므로 IC 부호와 같다 — §1.2와 같은 규칙이다.

**h 배치.** 채택 config가 FS1h면 FS3도 위 표의 검정 horizon 밖 h에서는 뺀다(§1.3과 같은 규칙).
FS1(전체 투입)이면 11개를 네 h에 그대로 넣는다. 이 규칙을 지금 적어 두는 이유는 E5 결과를 보고
컬럼을 골라내는 것이 §5가 금지한 것이기 때문이다.

**넣지 않는 것.** `rel_peer_corr_mean`(조건 변수 후보로만 넘어왔다), `rel_peer_mom_20d_k10`·`_k40`
(K 민감도 진단), `rel_peer_n_*`, `rel_peer_ksic_agree`, `rel_peer_month_end`, `rel_own_ret_20d`,
`fin_lifecycle_transition_up`·`_down`,
`fin_lifecycle_prev_*`, `fin_interest_coverage_capped`, `fs_basis_used`, 모든 `_lag1`(§2의 flow 규칙과
달리 이 둘은 `native_t`가 정본이다). event cohort 2 family(`fin_dividend_initiation`,
`fin_negative_equity_exit`)는 `fin_risk_event` 마트가 없어 F-HS-1에서 `blocked_exploratory`로 얼었다 —
FS3에 없다. F-3(업종 관계)·F-5 파생 family는 F-HS-2 뒤 별건이다.

**카드에서 그대로 넘어오는 경고.**

- `fin_interest_coverage`를 `financial_risk`의 독립적 발견으로 읽지 않는다. (a) `feat_fin_scan_daily`의
  `fin_operating_profitability`와 ρ = 0.87이다(분자가 같은 영업이익). 그 컬럼은 FS0~FS2에 없으므로 모델
  집합 안의 중복은 아니다. (b) `operating_income`의 `mapping_fallback_ratio` = 0.9121 — 값의 91%가 우선
  규칙이 아닌 fallback으로 해소된다.
- `fin_net_debt_to_mcap`의 `+`는 value 축과 겹친다. `fin_book_to_market`과 ρ = 0.52(분모인 시총 공유).
  FS1의 `fin_value_z`는 B/M을 최대 네 성분 중 하나로 담으므로 이 경고는 FS1 안에서도 유효하다. 합성 뒤의
  ρ은 측정한 적이 없다.
- `fin_debt_to_assets`는 FS0의 `fin_debt_to_equity`·`fin_equity_ratio`와 같은 레버리지 개념이다. 분모와
  PIT 규칙이 다르다(`fin_pit`의 +90/+45 보수 지연 vs `fin_risk`의 strict vintage). FS0를 건드리지 않으므로
  셋을 같이 넣고, 겹침은 E4d식 ablation이 필요하면 그때 사전등록한다.
- `fin_*` 7개는 **등급 B 상한**이다. 누락이 아니라 측정 결과다 — `revision_ratio` 0.1014~0.1015로 문턱
  0.10을 막 넘겼다.
- `fin_lifecycle_stage`의 Decline 축은 **영구 한계**다. OpenDART가 상폐 법인에 재무 엔드포인트 둘 다
  status 013을 돌려주므로(F-9.3) 상폐 재무는 수집이 불가능하고, 상폐 커버리지 37%가 상한이다.

**E5 선행 조건 — 08-23 snapshot의 마트가 옛 산식이다** (2026-09-09 측정).

E0~E4 공통 snapshot `2026-08-23`에도 `feat_relation_stat`·`feat_fin_risk`가 있지만 **v1 빌드**다
(`sql_hash`가 09-08과 다르고 `analysis_config_hash`는 `None`). 검정된 것은 `relation_stat_v2`(primary
5개의 `_lag1` 추가, 파티션 경계·365일 공백 규칙)와 `fin_risk_v2`(F-5.0의 XBRL fallback 16규칙)다.
`stock_metric_vintage_fact`도 08-23에서는 F-5.0 이전이고, 그 규칙이 fin_risk 5 family의 유효 시작을
2~4년 앞당긴 것이다.

따라서 E5는 08-23 snapshot에서 아래 넷을 현재 코드로 재빌드한 뒤 돈다.

```
stock_metric_vintage_fact → fin_quarterly_metric_vintage → feat_fin_risk
dim_peer_monthly → feat_relation_stat
```

**FS0~FS2 컬럼은 바뀌지 않는다.** F-5.0이 규칙을 더한 4 metric(`total_liabilities`,
`cash_and_cash_equivalents`, `operating_income`, `interest_paid`)을 기존 `feat_fin_scan_daily`가 읽지
않고, 그 마트의 `sql_hash`가 그대로이므로 캐시가 유지된다(격리 lake 측정 + 2026-09-09 라운드에서
공유 셀 `|ΔIC| = 0`으로 재확인). E5만 snapshot 2026-09-08에서 돌리는 대안은 택하지 않는다 — E0~E4와
snapshot이 갈리면 "FS3는 추가만 한다"는 비교가 깨진다.

---

## 2. 시점(PIT) 규칙 — 집합 전체에 적용

| 마트 | 규칙 | 이번 모델의 처리 |
|---|---|---|
| `feat_price` | 당일 종가 = 관측 시점(`native_t`) | 그대로 |
| `feat_flow` | 검증 정본은 **`lag1`**(당일 집계 사용 가능 여부 미검증) | **기본 `lag1`**: FS0의 flow 15개는 `_lag1` 컬럼이 없으므로 빌더에서 `LAG(·,1) OVER (ticker, market ORDER BY trade_date)`를 valid session 위에서 만든다. FS1의 `flow_individual_netbuy_to_volume_*`는 `_lag1` 컬럼이 마트에 있다. E3에서 당일 값 변형과 비교 (D-6 확정 2026-09-07) |
| `feat_fin_pit` | `stock_metric_fact` + period_end+90일(연)/+45일(분기) | FS0 그대로. 병행 스트림 F-9가 strict vintage 재작성을 끝내면 FS0'로 E3에서 비교 |
| `feat_fin_scan_daily`, `feat_event_scan_daily` | strict PIT vintage | 그대로 |
| `feat_filing_activity` | 접수일 다음 거래일 노출 | 그대로 |
| `feat_periodic_extras` | `*_available_from` | 그대로. `hc_*`·`own_major_stake_chg`의 `available_from` 이전은 NULL |
| `feat_macro_exposure` | 국내 요인 베타는 한 세션 늦게 갱신, 해외는 당일 | 그대로 |
| `dim_regime_daily` | 국내 계열 `next_krx_session`(t−1 정보), VIX는 NY t−1 | KRX 세션 격자에서 계산 |

**빌더가 마트 값을 시점 방향으로 옮기는 것은 flow `lag1` 하나뿐**이고, 그것도 검증 정본을 따르는 것이다. 다른 마트는 만든 그대로 쓴다.

---

## 3. 국면 마트와 interaction 빌더 (M-2)

### 3.1 `dim_regime_daily`

`research/analysis/horizon_scan_phase_c_regimes.py`가 Phase C용으로 만든 국면 시계열을 **마트로 저장**한다(`research/output/horizon_scan/phase_c_regimes/`에 이미 산출물이 있다).

- grain `(trade_date)`. 컬럼 `rg_vix_up`, `rg_vix_high`, `rg_market_up`, `rg_liq_high`(이진), 같은 이름의 `_z`(연속, 진단용), `rg_term_steep`·`rg_kosdaq_rel_up`·`rg_krw_weak_20d`(exploratory, 모델에는 안 넣는다).
- 격자는 **KRX 세션**(`label_scan`의 distinct `trade_date`). fact 격자(2014~2023 평일)에서 만들면 안 된다.
- 정의는 `03_stage1b_conditional_ic_phase_c.md` §2.3 그대로. 바꾸지 않는다.
- 유효 시작 2015-06-16(252세션 warm-up). 그 전은 NULL → `*_isna`.
- 구현: `research/etl/features/regime.py`에 `materialize_regime(con, config)`. Phase C 모듈의 함수를 import해 같은 값을 내는지 테스트로 고정한다.

### 3.2 interaction

```python
# _02/features.py
INTERACTIONS = (
    ("ix_reversal_vixhigh", "px_reversal_5d", "rg_vix_high"),
    ("ix_tshock_liqhigh", "px_turnover_shock", "rg_liq_high"),
    ("ix_foreign_liqhigh", "flow_foreign_netbuy_to_volume_20d_lag1", "rg_liq_high"),
    ("ix_mktbeta_mktup", "px_market_beta", "rg_market_up"),
)
# 값 = pct_rank(x) within (trade_date, market) × regime  — regime이 0이면 0, x가 NULL이면 NULL
```

- `pct_rank`는 §4의 `rank` 전처리와 같은 함수를 쓴다. 두 곳에서 다른 순위 정의가 생기면 안 된다.
- 국면이 0인 날 interaction이 전부 0이 되는 것은 의도다. 트리는 `rg_*`와 `ix_*`를 함께 보고 "국면 1일 때의 x 순위"를 읽는다.
- 국면 × 특성의 **증분성**(특성 단독 대비)은 E2에서 FS1 → FS2 비교로만 본다. 쌍별 ablation은 E4d다.

---

## 4. 전처리 (M-3)

### 4.1 `rank` 프로파일 추가

`research/etl/preprocess.PreprocessConfig.profile`에 `"rank"`를 추가한다.

| 프로파일 | 변환 | 결측 | 쓰는 모델 |
|---|---|---|---|
| `linear`(기존) | fold train에서 winsor(1%, 99%) → z-score | median impute + `*_isna` | Ridge |
| `tree`(기존) | winsor만 | NaN 유지 + `*_isna` | HGB(`_01`) |
| **`rank`(신규)** | **`(trade_date, market)` 내 백분위 [0,1]** (동점 average) | 0.5 impute + `*_isna`(트리는 NaN 유지 가능) | HGB 분류기·로지스틱·interaction |

- rank는 날짜별로 계산되므로 **fold train에서 fit할 통계량이 없다.** look-ahead가 구조적으로 없다. 그래서 `FittedPreprocess`에 `rank` 분기는 상태 없는 변환으로 넣는다.
- 이진·플래그 컬럼(`px_is_halted`, `fin_is_negative_equity`, `fin_has_fs`, `rg_*`)은 rank하지 않는다. `_cast_bool_features` 목록에 `rg_*`를 추가한다.
- 건수형(`ev_filing_burst_60d`, `own_major_filing_60d`)은 동점이 많다. average rank가 동점을 같은 값으로 두므로 상수 횡단면(I13)에서도 0.5가 되어 안전하다. 유효 시작일 이전은 NULL로 두어야 한다(마트가 이미 그렇게 한다).
- 20종목 미만 횡단면은 Horizon Scan이 버렸다. 전처리에서는 버리지 않고 rank만 매긴다. 표본 규칙은 유니버스 필터가 담당한다.

### 4.2 결측 플래그

`add_isna_flags`가 모든 피쳐에 `*_isna`를 붙인다. FS2에서 68 → design 136 컬럼. 커버리지가 낮은 피쳐(`ev_net_share_issuance_yoy` 0.48, `hc_*`, `own_major_stake_chg`)의 `_isna`는 "어떤 회사인가"의 정보이자 "어느 시대인가"의 정보다. E4에서 `_isna`를 뺀 변형(E4e)으로 시대 학습 여부를 본다.

### 4.3 하지 않는 것

- 업종 중립화(업종 코드 없음. 병행 스트림 F-1 뒤).
- 피쳐 간 상관 제거·PCA. 트리 모델은 필요 없고, 로지스틱은 L2로 다룬다.
- 타깃 인코딩·시차 변수 추가. 시차는 마트 정의에 이미 있다.

---

## 5. 유니버스와 패널 조립

- `dim_universe_daily`(`research/etl/universe.py`) 기본 `UniverseFilter`: warm-up 60행 중 유효 40, 60일 평균 거래대금 ≥ 1억원, 라벨 존재. **D-5 확정(2026-09-07)대로 기본 필터를 쓴다.**
- 조립은 `_01.build_dataset.assemble_panel`의 as-of LEFT JOIN 패턴을 따른다. `_GROUP_VIEW`를 확장해 `fin_scan`(`feat_fin_scan_daily`), `mcap`(`feat_market_cap`), `evs`(`feat_event_scan_daily`), `filing`(`feat_filing_activity`), `pex`(`feat_periodic_extras`), `mx`(`feat_macro_exposure`), `rg`(`dim_regime_daily`)를 추가한다. `cf`(`feat_common`)는 매핑만 두고 FS에 넣지 않는다.
- 마트는 `snapshot_date=2026-08-23`·`source=sj2_remote` 것을 쓴다. `feat_fin_scan_daily`·`feat_filing_activity`·`feat_periodic_extras`·`feat_macro_exposure`는 Horizon Scan A0 경로가 만든 것이 이미 있다. **`compute_all --features`로 다시 만들면 A0 계약 해시가 어긋난다**(`00_status.md` §5-1 주의) — 있는 것을 읽기만 한다.
- 기간: `period_start = 2015-01-02`(기존과 같음), **`period_end = 2025-07-31`**(D-2). h=120 라벨이 있는 마지막 formation은 2025-02 초이고, 그 뒤 formation은 purge된다.

---

## 6. 검증 항목 (테스트)

- FS0 컬럼 목록 == `run_grade_a_acceptance_gate.BASELINE_COLS`.
- FS1 ⊃ FS0, FS2 ⊃ FS1, FS1h ⊂ FS1(h마다).
- 모든 FS 컬럼이 매핑된 마트 뷰에 실제로 존재한다(snapshot 2026-08-23 smoke).
- `rank` 변환: 날짜×시장 안에서 min 0·max 1, 동점 average, NULL 유지.
- interaction: 국면 0인 날 0, 국면 1인 날 `pct_rank(x)`와 같다, x NULL이면 NULL.
- `dim_regime_daily` == Phase C 산출물(`phase_c_regimes/`)의 이진 값, 세션 격자 일치.
- flow `lag1` 빌더: 거래정지 세션을 건너뛴 직전 유효 세션 값.
