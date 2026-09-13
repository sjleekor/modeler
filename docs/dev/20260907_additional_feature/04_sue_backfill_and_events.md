# 04. `fin_sue` XBRL 백필 — F-6 · 구조화 이벤트 수집 — F-7

- 작성일: 2026-09-07
- 근거: `22_fin_sue.md`(표본 0, "아직 못 쟀다"), `09_w4_filing_text.md` §6(DS005가 텍스트 없이 이벤트를 준다), `10_work_breakdown.md` N5(elestock 보류 결정과 그 위험).

---

## 1. F-6 — `fin_sue` 표본 살리기

### 1.1 왜 표본이 0인가

`fin_sue`는 표준화 실적 서프라이즈(seasonal random walk: 이번 분기 EPS − 4분기 전 EPS를 과거 8분기 차이의 표준편차로 나눈 값)이고, **접수일 기준** event-time으로 잰다. 8분기 이력을 요구하는데 XBRL이 원본 접수 기준으로 연속해서 있는 법인·구간이 적어 `effective_start`가 2025-05-02, coverage 0.00으로 나왔다(6 cell `insufficient`). 서프라이즈 축(T2)은 **한 번도 측정되지 않았다.**

### 1.2 대상 선정 — 역산

1. `dart_filing_receipt_raw`에서 정기보고서(사업·반기·분기) **원본 접수**(정정 제외) 목록을 뽑는다. 2015~2025, `report_nm`·`rcept_no`·`corp_code`.
2. `dart_xbrl_document`에 그 `rcept_no`의 XBRL이 있는지 대조한다. 없는 접수가 백필 대상 후보다.
3. `fin_sue`가 요구하는 것은 **분기 EPS 연속 8개 + 이번 분기**다(F-6.1에서 코드로 확인: `comparative_eps`가 같은 보고서의 `value_lag_4q`라 보고서 하나가 `seasonal_change` 하나를 만든다 — 13분기가 아니다). 법인별로 "연속 9분기가 되기 위해 빠진 접수"만 대상으로 좁힌다. **아홉 분기 중 하나라도 제출되지 않은 창은 제외한다** — 받아도 완성되지 않는다.
4. 규모 측정 산출물: `results/sue_backfill_targets_2026MM.md` — 법인 수, 접수 수, 연도 분포, 예상 호출 수. **측정 전에는 규모를 말하지 않는다.** 참고로 정기보고서 정정은 2022~2025 5,041건이었다(`08` §4.3).

### 1.3 실행

```bash
# 대상 파일 생성 (신규 스크립트)
uv run python -m research.analysis.sue_backfill_targets --snapshot-date 2026-09-08 --source sj2_remote --out targets/sue_xbrl_targets.jsonl
# 백필 (기존 CLI)
uv run krx-collector dart backfill-xbrl-receipts --targets-file targets/sue_xbrl_targets.csv --rate-limit-seconds 0.2
```

- 대상 파일은 **JSON lines**다(`ticker`·`corp_code`·`bsns_year`·`reprt_code`·`rcept_no`) — 위 `.csv`는 F-6.1에서 정정했다.
- `dart backfill-xbrl-receipts`는 `(corp, filing, receipt)` 목록을 받아 XBRL을 받는다. `--force`는 쓰지 않는다(있는 문서는 skip).
- 다중 키 rotation, 일 한도 도달 시 exit 75 → 다음 날 재개(skip-if-present). 대상이 수만 건이면 slice ledger를 쓰는 편이 안전하다 — 규모 측정 뒤 결정.
- prod에서 돌린다(raw는 prod Postgres). `opendart` lock 공유, 04:00 체인·23:30 filings와 겹치지 않는 시간대.

### 1.4 백필 뒤

1. `db sync-remote` → raw export → `bin/parquet-compute-all.sh`(vintage·`fin_scan`·`fin_sue_event` 재빌드). **다른 fin family의 값이 바뀔 수 있다**(XBRL fallback이 더 채워진다) — I7 때처럼 `FIN_FEATURE_FORMULA_VERSION`을 올리지는 않지만(정의 불변) 입력이 바뀐 것이라 새 snapshot으로 A0 → A → B → AB를 다시 돈다.
2. `event_coverage.parquet`에서 `fin_sue`의 `effective_start`가 2016~2017로 들어왔는지 확인.
3. **같은 사전등록**(event bucket 6개, 기대 부호 `+`, `min_events` 규칙)으로 다시 판정한다. 격자·부호를 바꾸지 않는다(`22` §11).

### 1.5 완료 기준

- 대상 측정 문서 1회, 백필 완료(`ingestion_runs` success/partial, 오류 목록), coverage 0.00 → 실측값, `fin_sue` 6 cell이 `insufficient`에서 판정으로 바뀜(결과가 D여도 완료).

**진행 (2026-09-10 기준)**

| 완료 기준 | 상태 |
|---|---|
| 대상 측정 문서 1회 | ✅ F-6.1 → [`results/sue_backfill_targets_202609.md`](results/sue_backfill_targets_202609.md). 22,700 접수 / 2,807 법인 |
| 백필 완료 | ✅ F-6.2 → [`results/f6_2_sue_backfill_20260910.md`](results/f6_2_sue_backfill_20260910.md). 2026-09-10 02:24, **47 run 전부 `success`**, 오류 0, `partial` 0 |
| coverage 실측 | ⏳ F-6.3. 접수·XBRL 가용성 기준으로는 산출 가능 분기 33,141 → **75,293**(×2.27)이지만, 이것은 **상한**이고 vintage 층은 재빌드해야 안다 |
| 6 cell 재판정 | ⏳ F-6.3 |

§1.2의 "연속 9분기"는 F-6.2에서 코드로 확인됐다: `comparative_eps`가 같은 보고서의
`value_lag_4q`라 보고서 하나가 `seasonal_change` 하나를 만들고, `history_count >= 8`에
자기 자신을 더해 9다. 비교값이 4분기 전 보고서에서 왔다면 13개가 필요했다.

---

## 2. F-7 — DS005 주요사항보고서 구조화 이벤트

### 2.1 왜

- 현재 이벤트 축(C4)은 `feat_event_scan_daily`(발행·배당)와 `feat_filing_activity`(건수·정정)뿐이다. **무엇이 일어났는지**(유상증자 결정, 자사주 취득, 합병)는 `report_nm` 텍스트 분류로만 가능하고, 그것도 `ev_material_event_flag`로 미뤄 뒀다(N5-7).
- OpenDART DS005(주요사항보고서 주요정보)는 이벤트별 **구조화 필드**를 준다. 텍스트 파싱이 없다(`09_w4` §6).

### 2.2 PoC (D-F3)

| 단계 | 내용 |
|---|---|
| 1 | OpenDART 개발가이드에서 DS005 endpoint 목록을 확정한다(유상증자 결정·무상증자 결정·유무상증자 결정·감자 결정·전환사채 발행 결정·신주인수권부사채·교환사채·자기주식 취득 결정·자기주식 처분 결정·합병 결정·분할 결정·영업양수도·타법인 주식 양수도 등). **이름·인자·응답을 실호출로 확인**한다 — 문서상 이름을 이 계획에 미리 적지 않는다 |
| 2 | 인자가 `corp_code` + 기간(`bgn_de`/`end_de`)인지, 기간 상한이 있는지(N5의 elestock은 2년 롤링이었다) 확인 |
| 3 | 법인 20개 × 2015~2025로 볼륨·필드·`rcept_no` 존재 여부 측정. `rcept_no`가 있으면 `dart_filing_receipt_raw`와 조인해 PIT 노출일을 접수 다음 세션으로 정한다 |
| 4 | 산출물 `poc/ds005_events.md` — endpoint 6종 채택 여부, 예상 호출 수(법인 3,959 × endpoint × 연도 분할?), 스키마 초안 |

### 2.3 수집 (PoC 통과 시)

- raw 테이블 `dart_major_event_raw`: `(corp_code, event_type, rcept_no, rcept_dt, fields JSONB, run_id)`. PK `(event_type, rcept_no)`. 7곳 등록.
- 어댑터는 `opendart_common` 다중 키 실행기 + N6과 같은 **statement enum 하나**로 endpoint를 받는다(5개가 인자·응답이 같았던 N6 패턴).
- 백필 2015~2025 + 정기 증분(매일 23:30 filings 체인 뒤, 최근 14일 lookback). 대상은 `--universe-scope historical`.
- 규모가 N6(8.4만 호출)급이면 slice ledger.

### 2.4 피쳐 — `feat_event_decision`

| family | primary | 정의 | 부호 | horizon |
|---|---|---|---|---|
| `ev_capital_raise_decision` | `ev_capraise_60d` | 60세션 안 유상증자·CB/BW/EB 발행 결정 건수(0/1/2+) | `−` | [5, 10, 20, 60] |
| `ev_buyback_decision` | `ev_buyback_60d` | 60세션 안 자기주식 취득 결정(처분은 별도 secondary) | `+` | [5, 10, 20, 60] |
| `ev_restructure_decision` | `ev_restructure_120d` | 120세션 안 합병·분할·영업양수도 결정 | 양방향 | [20, 60, 120] |
| `ev_capital_reduction` | `ev_capreduce_120d` | 감자 결정 | `−` | [20, 60, 120] |

- 노출은 접수 다음 세션. 건수형이라 0 과다·상수 횡단면 초기 구간 주의(I13). 이벤트 빈도에 따라 event cohort 스캔.
- `ev_net_share_issuance_yoy`(결과: 발행주식수 변화)와 개념이 겹친다. 결정 시점 vs 결과 시점의 차이가 정보다 — 상관 진단을 카드에.

---

## 3. `elestock` 정기 수집 — D-F2

- N5-6은 "수량을 포기하고 빈도로 간다"로 결정했고(`own_insider_filing_*`가 그 결과), N5-8 `elestock` 정기 수집은 **보류**였다. 보류의 비용은 "지연 1개월 = 과거 1개월 영구 손실"(응답이 최근 2년 롤링)이다.
- **확정(2026-09-07): 지금 시작한다.** 월 1회 `--universe-scope current` 2,657 법인 호출(약 9분). raw `dart_insider_holding_raw`(임원·주요주주 소유상황 — 보고자·변동 수량·단가·사유). 피쳐(`ins_net_buy_90d`, `ins_buy_count_90d`)는 24개월이 쌓인 2028년 이후에나 검정하지만, 시작하지 않으면 그 시점이 계속 밀린다.
- 비용이 작고 PIT가 깨끗하다(보고 의무 시점 명확). 7곳 등록 + 월 Cronicle 이벤트(`06` §5 `sdc_monthly_insider_holding`).
- 첫 실행은 F-7.1 DS005 PoC와 같은 주에 한다. 수집만 시작하고 피쳐 정의는 24개월 뒤 별도 사전등록이다.

---

## 4. 순서와 의존

| 순서 | 작업 | 선행 |
|---|---|---|
| 1 | F-6 대상 측정(로컬 lake만으로 가능) | — |
| 2 | F-7 PoC(호출 수백 건) | — |
| 3 | `elestock` 스키마·어댑터·월 이벤트 (D-F2 확정) | — |
| 4 | F-6 백필 실행(prod, 며칠) | 1 |
| 5 | F-7 raw 수집·백필 | 2, D-F3 |
| 6 | 마트 재빌드 → `fin_sue` 재판정(기존 config), `feat_event_decision` → F-HS 다음 config | 4, 5 |

F-6은 기존 사전등록으로 재판정하므로 F-HS 새 config에 **넣지 않는다**(같은 family를 두 모집단에 넣지 않는다). 다만 재판정을 위한 A→B→AB 재실행은 F-HS 실행과 **같은 snapshot으로 묶어** 한 번에 돈다.
