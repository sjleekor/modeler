# 06. 인프라·PIT 통일 — F-9

- 작성일: 2026-09-07
- 목적: 두 스트림이 공통으로 밟는 바닥을 고른다. (1) 모델 baseline 재무 피쳐의 PIT 규칙을 검증 트랙과 맞추고, (2) 국면·롤링 창 계산을 흔드는 `holidays_krx.csv` 공백을 메우고, (3) 상폐 법인 커버리지를 채워 생존편향을 줄이고, (4) 새 수집을 운영에 얹는다.

---

## 1. `fin_pit` strict vintage 재작성 — FS0'

### 1.1 문제

| 마트 | 원천 | PIT | 쓰는 곳 |
|---|---|---|---|
| `feat_fin_pit` (baseline `fin_*` 10) | `stock_metric_fact` | `period_end + 90일(연) / +45일(분기)` 보수 지연. 실제 접수일 무시 | 모델 baseline 40 |
| `feat_fin_scan_daily` (검증 `fin_*` 5) | `stock_metric_vintage_fact` | strict PIT(접수 다음 세션, 같은 날 후보는 최신 회계기간) | Horizon Scan·T2 |

같은 `fin_` 접두어에 두 시점 규칙이 있다. 보수 지연은 늦게 노출해 look-ahead는 없지만, 실제 공시 뒤 최대 두 달을 버린다. 모델에 둘을 섞으면 같은 회계 정보가 두 시점에 들어간다(`08_quality_summary_gaps.md` §2.10).

### 1.2 작업

- 새 마트 `feat_fin_ratios_pit`(`research/etl/features/fin_ratios_pit.py`): `fin_pit.py`의 10개 비율 정의를 **그대로**, 입력만 `fin_scan.py`의 vintage 선택·`base_ok` 함수로 바꾼다. 컬럼 이름은 `fin_roa_pit` 등 접미 `_pit`로 구분한다(같은 이름으로 두면 어느 규칙인지 알 수 없다).
- `feat_fin_pit`는 **삭제하지 않는다.** 기존 게이트 결과 재현 경로다.
- 검증: 두 마트의 값 차이 분포(같은 종목·날짜), 노출 시점 차이 히스토그램(접수일 vs +90/+45), 커버리지 차이.
- 모델 스트림 E3-b가 FS0(`fin_pit`) vs FS0'(`fin_ratios_pit`)를 비교한다. 이 마트가 준비되는 시점을 모델 스트림에 알린다.

---

## 2. `holidays_krx.csv` 2014~2023 보강

### 2.1 문제

`docs/holidays_krx.csv`가 2024~2026만 담고 있어 `common_feature_daily_fact`의 `feature_date` 격자가 2014~2023년에 **평일**(휴장 평일 연 13~17개 포함, 전 세션 값 복사)이다. fact 격자에서 `t−20`·`t−252`를 세면 창 길이가 2024년 전후로 다르다. 국면 계산은 이 때문에 KRX 세션 격자로 우회했다(`03_stage1b` §2.1). 우회는 되지만 fact를 직접 롤링하는 모든 소비자가 같은 함정에 빠진다.

### 2.2 원천 — 새 수집 없이 된다

`daily_market_cap`(KRX Open API, 2014-06-02~)은 **휴장일에 행이 0**이다(pykrx와 반대, `poc/krx_open_api.md`). 그러므로 2014-06-02 이후의 KRX 휴장 평일 = `daily_market_cap`에 행이 없는 평일이다. N1-8에서 완료 못 한 300 슬라이스가 정확히 이것이었다.

- 2014-06-02 이전(2014-01~05)은 `daily_ohlcv`(naver)로 같은 방법을 쓴다(거래정지·상폐가 아니라 **전 종목**이 비는 날).
- 산출: `research/analysis/derive_krx_holidays.py` → 2014~2023 휴장일 목록 → `docs/holidays_krx.csv` append. 임시공휴일·대체공휴일이 그대로 잡힌다.

### 2.3 영향과 절차

- `infra/calendar/`가 이 CSV를 읽는다. 바뀌면 (a) `common_build` 마트의 fact 격자가 2014~2023에서 세션 격자로 바뀌고, (b) `tests/unit/golden/*.json` 중 fact 관련 golden이 바뀌고, (c) `00_survey/00` §4의 상관 재현값이 움직인다.
- 절차: CSV 보강 → golden 재생성(값 변화를 diff로 기록) → `common_build` 버전 표기 → 새 snapshot에서 fact 재빌드 → 국면 재계산(세션 격자와 일치해야 한다 — 일치하면 우회가 필요 없어진다) → 문서 갱신.
- **시점**: F-HS 재실행과 같은 snapshot 갱신에 묶는다. 중간에 하면 계보가 갈린다.

---

## 3. 상폐 법인 커버리지 — S-1 잔여 · S-2 · S-3 (D-F4 확정 2026-09-07: 지금 실행한다)

| 항목 | 상태 | 작업 | 왜 지금 |
|---|---|---|---|
| S-1 잔여 | filings 3,490 법인 완료. financials·share-info·xbrl은 **약 1,300 상폐 법인 미수집**(2,608/3,959) | `dart sync-financials/sync-share-info/sync-xbrl --universe-scope historical --include-delisted`, 2015~2025. OpenDART만. slice ledger | F-4 Decline·부실 축, `fin_sue` 표본, PIT 유니버스의 선행 |
| S-2 상폐 종목 가격 | `universe backfill-master`(v0.9.8) 준비됨, 실행 대기. `daily_market_cap`이 상폐 종목을 세션별로 준다 | `prices backfill`을 복구된 master로 실행(naver 원천, 상폐일까지) | `available` 표본이 실제 상폐 종목을 포함해야 생존편향 검사가 뜻이 있다 |
| S-3 상폐 시점 확정 | 미착수 | `daily_market_cap` 마지막 세션 = 상폐 직전 세션. `stock_master` DELISTED 행에 `delisted_date` 채움 | 라벨 계산에서 상폐 종목의 forward 수익률 처리(마지막 가격 이후 −100%? 제외?) 정책 결정의 입력 |

- S-4 측정(2026-08-27)은 현재 편향이 작다고 봤다(부호 반전 0, IC 차이 중앙값 0.0006). 그래도 Decline·부실 family는 편향에 가장 민감한 축이라 **F-4 판정 전에 S-1 잔여를 끝낸다.**
- 규모: 약 1,300 법인 × 연 11 × 3 API ≈ 4.3만 호출(재무·주식수·XBRL). 키 2개면 며칠. exit 75 재개.
- 상폐 종목의 라벨 정책은 모델 스트림에 영향이 있다(현재는 `daily_ohlcv`에 없어 자연히 빠진다). S-2 뒤에는 "상폐 직전 h일 라벨"이 생긴다 — `label_scan`·`label_daily`가 어떻게 다루는지 확인해 문서화한다(`07_labels_sample_validation.md` 갱신).

---

## 4. PIT 유니버스 마트 (N3-7 안 1)

- `dim_universe_daily`가 `daily_ohlcv`에서 만들어진다(`universe.py` `price_view="daily_ohlcv"`). N3 §3.5 판정은 **일별 유니버스의 정본을 `daily_market_cap` 행**으로 봤다(KONEX 구간 배제, 상폐 종목 포함, 같은 세션에서 최대 324종목 더 많다).
- 작업: `materialize_universe(price_view="daily_market_cap")` variant → `dim_universe_daily_krx`. 기존 유니버스는 유지(재현 경로). 두 유니버스의 종목 집합 차이를 날짜별로 기록.
- 소비: Horizon Scan `available` 표본과 모델 학습 유니버스가 이걸 쓰는 것은 **새 config·새 실험 단계에서만**. 기존 결과를 다시 만들지 않는다.
- Cronicle `SDC Backfill N3 Universe Snapshots` 이벤트는 삭제 대상(2019~2025 공백은 판정, N3-7).

---

## 5. 운영에 얹는 것 — Cronicle

| 이벤트 | 주기 | 명령 | lock | 문서 |
|---|---|---|---|---|
| `sdc_monthly_corp_profile_history` | 매월 1일 05:30 | `dart sync-corp-profile --universe-scope historical --refresh` + history append | `opendart` | `01` §2.4 |
| `sdc_monthly_insider_holding` (D-F2 확정) | 매월 1일 06:00 | `dart sync-insider-holding --universe-scope current` | `opendart` | `04` §3 |
| `sdc_daily_major_events`(F-7 시) | 매일 23:45 | `dart sync-major-events --lookback-days 14` | `opendart` | `04` §2 |
| 기존 20:30 common | 유지 | ECOS/FRED 새 시리즈 자동 포함 | — | `05` §1 |
| 사람 작업 | 매월 초 | KRX 업종분류 현황 CSV 다운로드 → `docs/dev/…/industry_csv/YYYYMM.csv`, 기준일 대조 | — | `01` §3 |

- freshness 게이트에 월 단위 예산(45 영업일) 항목을 추가한다.
- `sdc_backfill_*` 일회성 이벤트는 끝나면 삭제(기존 관행).

---

## 6. 등록·DoD 체크리스트 (새 raw 테이블마다)

`02_data_expansion_plan/01_implementation_checklist.md` §1·§7 그대로.

- [ ] 7곳 등록: `sql/postgres_ddl.sql`, `infra/db_postgres/remote_sync.py`(`SYNC_TABLE_SPECS`·full-refresh 목록), `service/profiling/catalog.py`, `tools/raw-parquet-exporter/config/export_tables.toml`, `bin/raw-parquet-export-all.sh`, `research/etl/config.py RAW_TABLES`, `docs/database.md`
- [ ] 개수 단언 테스트 갱신(`test_research_config`, `test_remote_db_sync`, `test_profiling`)
- [ ] `db init` → 테이블·인덱스 생성, `db sync-remote --full-refresh` → 미러링, export → lake 파티션, `LakeConfig.table_glob` 동작
- [ ] idempotent(재실행 시 skip 증가·upsert 0), 부분 실패 `partial` + exit 0, 다중 키·exit 75
- [ ] `ingestion_runs` RunType 추가, 문서에 수집 결과 관측 요약(행 수·기간·커버리지·결측)
- [ ] 스크래핑 금지 확인(원천이 OpenDART·KIS·Open API·FRED·ECOS·FDR 익명 중 하나)

---

## 7. Horizon Scan 재실행 규칙 (F-HS)

- **묶어서 한 번.** F-2·F-4(·F-7)가 마트로 준비되면 새 overlay config(`horizon_scan_expansion_2026MM.yaml`, `extends` 체인)를 사전등록하고 A0 재빌드 → A → B → AB → (C 2라운드)를 새 snapshot으로 돈다. 약 2시간 + Phase C.
- F-6(`fin_sue`)·S-1 잔여·holidays 보강은 **같은 snapshot**에 묶는다. 입력이 바뀌므로 기존 family의 값도 움직일 수 있다 — canonical과의 exact match 검사(단계 0 규약)로 차이를 기록한다.
- F-3(업종 관계)은 D-F1 판정 뒤 별도 config.
- 실행 중 모델 실험 run과 겹치지 않게 한다(장비 공유).
