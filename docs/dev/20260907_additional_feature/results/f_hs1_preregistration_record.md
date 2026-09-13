# F-2 관계 피쳐 · F-4 재무위험/생애주기/전이 사전등록 — 2026-09-09

- config: `research/analysis/horizon_scan_expansion_202609.yaml`
- config hash: `3ca949e6a2275706c172da639469b77477f2c48430f8681ec462b17ebde6e4ae`
- base config: `horizon_scan_macro_20260829.yaml`, `236d0d35…`
  (그 base는 `horizon_scan_expansion_20260827.yaml` `889c3e83…`, 그 base는 `horizon_scan_config.yaml` `ab0de634…`)
- 등록 시점: **이 13 family의 label·IC·p-value 계산 전**
- 설계 문서: `docs/dev/20260907_additional_feature/` (`02`, `03`), 측정 근거 `results/f2_relation_stat_verification.md` §6 ·
  `results/f4_fin_risk_verification.md` §8
- snapshot: `2026-09-08` / source `sj2_remote`
- 커밋: `e704e08`

## 결론

기존 41 family와 Phase A 75개 가설은 바꾸지 않는다. 이 층이 더하는 것은 하나다.

**Phase B에 13 family × 66 candidate cell.** Phase B는 102 → **168**, 결합 BH 모집단은 75 + `m_b_ready`다.

세 앞선 층의 hash는 그대로다(`ab0de634` / `889c3e83` / `236d0d35`). 발행된 모든 Phase A/B/AB/C run이 그 셋 중
하나에 묶여 있으므로, 그것을 움직이면 run이 자기 계약에서 떨어져 나간다. 테스트가 세 값을 직접 박아 둔다.

전 family `fdr_include: false`다. Phase A의 75개 모집단은 움직이지 않고, 결합 BH는 `m_b_ready`를 통해서만 커진다 —
앞선 세 층이 한 것과 같다.

Phase C는 열지 않는다. 두 리포트가 적은 국면 쌍 후보 6개(`rel_peer_mom × liq_high`, `rel_peer_mom × market_up`,
`rel_own_minus_peer × vix_high`, `rel_peer_bigcap_lag × liq_high`, `fin_debt_to_assets × credit_wide`,
`fin_lifecycle_stage × market_up`)는 F-HS-C2로 넘기고 `phase_c.pairs`에서 일부러 뺐다.

## 신규 Phase B registry — F-2 통계적 peer 관계 (4)

공통: `phase: B`, `fdr_family: relation`, `role: phase_b_blocked`, `fdr_include: false`,
`include_bucket_primary: true`, `official_feature_variant: native_t`,
`readiness_dependencies: [dim_peer_monthly, feat_relation_stat, label_scan]`.

| family | primary | secondary | 부호 | primary horizon | cell |
|---|---|---|---|---|---|
| `rel_peer_mom` | `rel_peer_mom_20d` | `rel_peer_mom_60d` | 양방향 | [20, 40, 60] | 6 |
| `rel_own_minus_peer` | `rel_own_minus_peer_20d` | — | `−` | [5, 10, 20] | 6 |
| `rel_peer_bigcap_lag` | `rel_peer_bigcap_lag_ret_5d` | — | `+` | [5, 10, 20] | 6 |
| `rel_peer_dispersion` | `rel_peer_dispersion_20d` | — | 양방향 | [20, 40, 60] | 6 |

검정하지 않는 컬럼: `rel_peer_corr_mean`(모델에 조건 변수로만 넘긴다), `rel_peer_ksic_agree`,
`rel_peer_mom_20d_k10`, `rel_peer_mom_20d_k40`, `rel_peer_n_*`, `rel_peer_month_end`, `rel_own_ret_20d`.
K는 20으로 고정하고 `_k10`·`_k40`은 민감도 진단으로만 남는다.

## 신규 Phase B registry — F-4 재무위험·생애주기·전이 (9)

공통: `phase: B`, `role: phase_b_blocked`, `fdr_include: false`, `official_feature_variant: native_t`.
연속형 7개는 `readiness_dependencies: [feat_fin_risk, label_scan]`.

| family | primary | fdr_family | 부호 | horizon / cell 축 | cell |
|---|---|---|---|---|---|
| `fin_debt_to_assets` | `fin_debt_to_assets` | `financial_risk` | 양방향 | [60, 120] | 4 |
| `fin_net_debt_to_mcap` | `fin_net_debt_to_mcap` | `financial_risk` | 양방향 | [60, 120] | 4 |
| `fin_interest_coverage` | `fin_interest_coverage` | `financial_risk` | `+` | [60, 120] | 4 |
| `fin_ext_finance` | `fin_ext_finance_to_assets` | `financial_risk` | `−` | [60, 120] | 4 |
| `fin_lifecycle_stage` | `fin_lifecycle_stage` | `lifecycle` | 양방향 | [60, 120] | 4 |
| `fin_lifecycle_transition` | `fin_lifecycle_transition` | `lifecycle` | 양방향 | [20, 60] | 4 |
| `fin_profit_turn` | `fin_profit_turn` | `transition` | `+` | [20, 60, 120] | 6 |
| `fin_dividend_initiation` | `fin_dividend_initiation` | `transition` | `+` | event bucket | 6 |
| `fin_negative_equity_exit` | `fin_negative_equity_exit` | `transition` | `+` | event bucket | 6 |

검정하지 않는 컬럼: `fin_lifecycle_transition_up`·`_down`(방향 진단), `fin_lifecycle_prev_stage`,
`fin_lifecycle_prev_aligned`, `fin_interest_coverage_capped`, `fs_basis_used`, `*_available_from`,
`*_fin_age_days`, 그리고 모든 `_lag1`(기존 variant 축).

### 양방향 6건

`rel_peer_mom`, `rel_peer_dispersion`, `fin_debt_to_assets`, `fin_net_debt_to_mcap`,
`fin_lifecycle_stage`, `fin_lifecycle_transition`. **첫 run 뒤 관측 부호를 고정하고 이후 바꾸지 않는다.**
테스트가 양방향 집합과 부호가 박힌 7건을 둘 다 직접 박아 둔다 — 나중에 조용히 채워 넣는 것이
사전등록이 막으려는 실패이기 때문이다.

### event cohort 두 건

`03` §1.3이 결과 전에 고정한 규칙은 "연 이벤트 종목 비율 5% 미만이면 event cohort"다. `f4` §6 측정값이
`fin_dividend_initiation` 0.0397, `fin_negative_equity_exit` 0.0040이므로 둘 다 event cohort다.

event 경로의 cell 축은 `phase_b.event_buckets`이고 이 값은 고정 프로토콜 값이다. 그래서 두 family의 cell은
**0–60**이고, 설계 초안이 적은 [60, 120]은 연속형으로 읽을 때의 값이라 event 경로에서는 표현되지 않는다.

**두 family는 이번 라운드에서 `blocked_exploratory`로 얼어붙는다.** `run_phase_b_event_scan`은
`fin_sue_event`의 cohort grain(`(ticker, original_rcept_no, event_formation_date, market)`)에 묶여 있고
그 view에는 이 두 컬럼이 없다. 대응하는 `fin_risk_event` 마트는 없다. 의존성에 `fin_risk_event`를 적어 둔 것이
바로 이것을 기록한 것이고, SUE용 스캔에 잘못 넘겨지는 것을 막는다. 이게 옳은 outcome-blind 상태다.

## 표본 시작 — 기록이지 강제가 아니다

스캔 기계에는 전역 `sample.start`(2014-06-01) 하나만 있고 family별 시작은 없다. 아래 값은 `preregistration.notes`에
적어 둔 기록이고, 각 cell은 자기 `effective_sample_start`를 보고하며 얇은 cell은 `stats.min_dates_per_cell`(60)이
걸러낸다.

snapshot 2026-09-08에서 잰 연도별 커버리지 0.5 규칙으로 정했다.

| 시작 | family |
|---|---|
| 2016 | relation 4개 (2016 커버리지 0.8545), `fin_debt_to_assets` 0.545, `fin_net_debt_to_mcap` 0.521 |
| 2017 | `fin_negative_equity_exit` 0.750 |
| 2018 | `fin_ext_finance` 0.652, `fin_lifecycle_stage` 0.647, `fin_lifecycle_transition` 0.603, `fin_profit_turn` 0.635 |
| 2019 | `fin_interest_coverage` 0.551, `fin_dividend_initiation` 0.533 |

F-5.0(`metric_rules` XBRL fallback 확장)이 5개 family를 2~4년 앞으로 당긴 결과다. 격리 lake에서 잰 값과
실제 snapshot 재측정값이 소수 셋째 자리까지 일치했다.

## 카드에 남기는 경고

`f4` §7의 중복 경고 2건을 그대로 옮긴다.

- `fin_interest_coverage` × `feat_fin_scan_daily.fin_operating_profitability` ρ = 0.87. 분자가 같은 영업이익이라
  사실상 같은 축이다. **`fin_interest_coverage`의 발견을 `financial_risk`의 독립적 발견으로 읽지 않는다.**
- `fin_net_debt_to_mcap` × `fin_book_to_market` ρ = 0.52. 분모(시총)를 공유한다.

`fin_lifecycle_stage`의 Decline 축과 부실 판정은 계속 보류하고, 이것은 **영구 한계**다. F-9.3이 확인한 대로
OpenDART는 상폐 법인에 대해 두 재무 엔드포인트 모두 status 013을 돌려주므로 빠진 상폐 재무는 아예 수집이
불가능하다. 생존편향은 수집으로 닫을 수 없다.

## source quality — 누락이 아니라 측정으로 등급 상한

13 family 전부 `FAMILY_METRIC_DEPENDENCIES`에 등록했다. 여기 없는 family는 verdict 자체가 없고
`source_quality_allows_grade_a(None)`이 False라서 **측정이 아니라 누락으로 B 상한**이 걸린다.

- relation 4개 → `frozenset()`. 가격·PIT 시총·`dart_corp_master` 업종코드만 읽고 metric 레이어를 거치지
  않는다(`fin_log_mcap`과 같은 이유).
- `fin_dividend_initiation` → `frozenset()`. `dps`를 `dart_shareholder_return_raw`에서 바로 읽는다
  (`ev_net_share_issuance_yoy`와 같은 경우).
- 나머지 7개는 `fin_risk.py`의 산식에서 그대로 뽑았다. `net_income`이 전부에 들어가는 이유는 `fin_risk`가
  CFS/OFS basis를 그 값으로 고르기 때문이고, 이는 기존 `fin_*` family와 같은 규칙이다.

## 이 층이 드러낸 것 두 가지

### 1. `feat_relation_stat`에 `_lag1`이 없었다 → `relation_stat_v2`

`variant_columns`는 family마다 `lag1` 매핑을 **필수로** 요구하는데 validator는 키 존재만 본다.
등록된 다른 모든 마트(`feat_market_cap`, `feat_filing_activity`, `feat_periodic_extras`,
`feat_macro_exposure`, `feat_fin_risk`)는 자기 primary마다 실제 `_lag1` 컬럼을 갖고 있지만
`feat_relation_stat`은 하나도 없었다. 그대로 두면 **없는 컬럼 위에 계약을 얼리는** 것이 된다.

`relation_stat_v2`가 `PRIMARY_COLUMNS` 5개의 `_lag1`을 추가한다. peer 규칙도 컬럼 산식도 바꾸지 않았다 —
`dim_peer_monthly`는 11,292,902행 / 232 월말로 비트 단위 동일하게 재생성됐고, `feat_relation_stat`도
7,053,322행 그대로이며 F-2 리포트의 모든 수치가 바뀌지 않았다(diff는 타임스탬프와 버전 문자열 2줄).

두 가지를 고쳐야 했다.

- **파티션 경계.** 이 마트는 연 단위로 쓰이고 SQL은 window보다 WHERE를 먼저 적용한다. 자기 연도만 걸러 읽는
  part는 매년 첫 세션의 `lag1`을 NULL로 만들면서도 완전한 마트처럼 보인다. 이제 각 part는 **자기 연도보다
  한 해 넓게 읽고 자기 연도만 내보낸다.**
- **장기 공백.** 전체 이력 LAG와 비교하니 121행이 남았다. 직전 유효 세션이 1년 넘게 떨어진 종목
  (거래정지 후 재개, 최소 432일 최대 2,870일)이다. 그걸 "한 세션 실행 지연"이라고 부르는 것은 거짓이고,
  guard 없이 두면 값이 **파티션 경계가 어디 떨어졌는지에만 의존**했다. `LAG1_MAX_GAP_DAYS = 365`로 규칙을
  명시했다. 365일 안의 선행 세션은 항상 part 안에 있으므로 part가 비파티션 계약을 정확히 재현한다 —
  재빌드 후 NULL 패턴 차이 0, 값 차이 1e-9 초과 0, 최대 잔차 1.4e-17(이미 문서화된 DuckDB 병렬 AVG의
  마지막 비트 비결정성).

### 2. Phase B 코드 배선

`register_phase_b_marts`에 `dim_peer_monthly` → `feat_relation_stat`, 그리고 `fin_quarterly_metric_vintage`
아래에 `feat_fin_risk`를 넣었다. `daily_marts`에는 `feat_relation_stat`·`feat_fin_risk`만 넣는다 —
`dim_peer_monthly`는 월별 grain이라 일별 panel에 조인하면 안 되고, readiness 의존성 이름으로만 쓰인다.
`_FORMULA_VERSION_BY_DEPENDENCY`에 두 마트의 `FORMULA_VERSION`을 등록해 §1.3 fingerprint가 비지 않게 했다.

## 이번 층이 하지 않는 것

- **일반 coverage 진단.** `FEATURE_COVERAGE_SPECS`와 `build_feature_coverage_sql`은 `feat_fin_scan_daily`·
  `feat_event_scan_daily` 두 마트만 받는다. `feat_market_cap`·`feat_filing_activity`·`feat_periodic_extras`·
  `feat_macro_exposure`도 들어가 있지 않은 기존 한계이고, 세 번째 마트를 넣으려면 그 SQL에 view 인자를
  더하는 별도 작업이 필요하다. 이 13 family의 연도별 커버리지·노출 지연·분포는 두 검증 리포트가 이미 낸다.
- **F-5 파생 family**(`fin_current_ratio`, `fin_borrowings_to_mcap`, `fin_altman_z`, `fin_rnd_to_sales`,
  `fin_bm_intangible_adj`). 정의가 태그 커버리지 PoC 결과에 달려 있어 다음 config다(`03` §3).

## 실행

```bash
# A0 — 스캔 자기 입력 마트. compute-all --features로 대신할 수 없다(계약 해시가 다르다)
uv run python -m research.etl.horizon_scan_inputs --source sj2_remote --force \
  --config research/analysis/horizon_scan_expansion_202609.yaml

uv run python -m research.analysis.horizon_scan --phase A --source sj2_remote \
  --config research/analysis/horizon_scan_expansion_202609.yaml
uv run python -m research.analysis.horizon_scan --phase B --source sj2_remote \
  --config research/analysis/horizon_scan_expansion_202609.yaml
uv run python -m research.analysis.horizon_scan --phase AB \
  --config research/analysis/horizon_scan_expansion_202609.yaml \
  --phase-a-run-dir <phase=A run> --phase-b-run-dir <phase=B run>
```

비교 기준은 snapshot 2026-08-23 / config `236d0d35` 계보다 — A `20260830T085718-efd35e70`,
B `20260830T100518-efd35e70`, AB `20260830T122850-efd35e70`. `m_ab` 177(75 + 102 ready),
primary discovery 103, `screen_pass` 53, 등급 A=23·B=30·C=40·D=9.

Phase B ready cell이 늘면 결합 BH 모집단이 커지고 문턱은 **더 엄격해질 수만 있다.** 그래서
"기존 discovery 변화 0"은 가정이 아니라 확인해야 하는 것이다. 실행 결과는 별도 문서에 남긴다.
