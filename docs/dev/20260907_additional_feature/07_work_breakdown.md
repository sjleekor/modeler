# 07. 작업 분해 — 체크리스트·순서·규모

- 작성일: 2026-09-07
- 이 문서가 진행 상태를 추적한다. 체크는 **끝나고 검증됐을 때** 채운다. 왜 하는가는 `01`~`06`에 있다.
- 모델 스트림 접점: FS3 인계(F-HS `screen_pass` 뒤), FS0' 통보(F-9.1 뒤).
- **현재 위치 (2026-09-10 02:30):** 항목 38/64 완료. F-1·F-2·F-4·F-5 완료, F-6은 백필까지 끝나고 재판정만 남았다. F-3은 2026-12 D-F1 대기, F-7·F-8은 미착수. prod는 v0.13.0이고 다음 행동은 **F-6.3 재빌드**다 — 그 한 번이 F-5의 새 metric 5개와 `fin_risk_v3`, `fin_sue` 재판정을 같이 처리한다.

---

## 0. 상태표

| 패키지 | 내용 | 새 수집 | 상태 | 다음 행동 |
|---|---|---|---|---|
| F-1 | 업종 `induty_code` 버저닝 | OpenDART 월 3,959 | F-1.1~F-1.7 완료(**F-1.5·F-1.7 2026-09-09**) | F-1.8 KIS PoC / F-1.9 D-F1(2026-12) |
| F-2 | 통계적 peer 관계 피쳐 | 없음 | **완료**(`relation_stat_v2`, 2026-09-09) | **F-HS-1에서 19 cell 등급 A `screen_pass`.** 양방향 2건 부호 `−`로 확정 |
| F-3 | 업종 관계 피쳐 | 없음(F-1 선행) | 대기(D-F1) — 선행 마트는 준비됨 | 비-seed 3쌍(2026-12-01 스냅샷) 뒤 판정 |
| F-4 | 재무위험·생애주기·전이 | 없음 | **완료**(`fin_risk_v2`, 2026-09-08) | **F-HS-1에서 13 cell `screen_pass`, 전부 등급 B 상한**(측정된 `revision` 경고). 양방향 4건 부호 확정 |
| F-5 | `metric_rules` 확장 | 없음(매핑) | **F-5.0~F-5.4 완료**(`fin_risk_v3`, 2026-09-09) | 커버리지 리포트는 다음 F-HS config A0 |
| F-6 | `fin_sue` XBRL 백필 | OpenDART 22,700 호출 **완료** | **F-6.1·F-6.2 완료**(2026-09-10 02:24) | F-6.3 재빌드 → 같은 사전등록으로 재판정 |
| F-7 | DS005 이벤트·elestock | 있음 | 미착수 (D-F2 확정: `elestock` 시작 / D-F3: PoC 뒤 6종) | DS005 PoC + `elestock` 스키마 |
| F-8 | 매크로 2단계 시리즈 | 시리즈 정의 | 미착수 (D-F5 확정: 2단계 먼저) | ECOS item_code 확정 |
| F-9 | `fin_pit` strict·휴장일·상폐·유니버스·Cronicle | 일부 | F-9.2(**배포 완료**)·**F-9.3 종결**·F-9.7·**F-9.9~F-9.11**·F-9.12·F-9.13·F-9.14 완료 | F-9.1 / F-9.4~F-9.6 / F-9.8 |
| F-HS | 새 config 사전등록·A→B→AB→C | — | **F-HS-1 완료**(`3ca949e6`, 2026-09-09), FS3 인계 완료 | F-HS-C2(F-8 선행) / F-HS-2 |

---

## 1. 순서

### 지나온 것 (2026-09-07 ~ 09-10)

```
09-07        계획 수립, D-F1~D-F5 확정
09-08        F-2 peer·F-4 fin_risk 마트  ‖  F-1 스키마·시드·첫 prod 스냅샷
             F-9.2 휴장일 CSV(46→201)  ‖  F-9.3 S-1 잔여 종결(수확 0)  ‖  F-5.0
09-09        F-HS-1 A0→A→B→AB (8.7시간) → FS3 11 컬럼 모델 인계
             F-1.5 마트·F-1.7 리포트  ‖  F-5.1~F-5.4 (catalog 29→34, fin_risk_v3)
             F-9.9~F-9.11 게이트 3건  ‖  F-6.1 대상 22,700 확정
             v0.13.0 릴리즈·배포 → F-9.2 prod 반영, K-5 죽은 fdr 경로 교체
09-10 02:24  F-6.2 백필 완주 (22,700 호출 / 2시간 20분 / 오류 0)
```

### 남은 것

```
바로 가능    F-6.3 재빌드·재판정(F-5.4 커버리지 리포트를 같이 낸다)
             F-7.1 DS005 PoC  ‖  F-8.1 ECOS item_code  ‖  F-9.1 fin_ratios_pit
             F-9.4~F-9.6 상폐·PIT 유니버스  ‖  F-9.8 freshness 월 예산
F-8 뒤       F-8.5 국면 사전 계산 → F-HS-C2 Phase C 2라운드
PoC 뒤       F-7.2~F-7.5 수집 → F-HS-2
2026-12      D-F1 판정(비-seed 3쌍) → F-3 마트 → F-HS-2
```

- `‖`는 병행 가능. 병목이던 **prod OpenDART 호출**은 F-6.2로 풀렸다. 남은 키
  예산 소비는 **F-7**뿐이다(PoC 뒤 규모 확정).
- 장비: F-2 peer 계산·마트 빌드와 Horizon Scan·모델 run이 겹치지 않게.
- **재빌드가 하나로 묶인다.** F-6.3의 새 snapshot 재빌드가 F-5.2~F-5.4의 새 metric
  5개와 `fin_risk_v3`를 같이 반영한다. 따로 돌리지 않는다.

---

## 2. 체크리스트

### F-1 업종 PIT (`01`)

- [x] **F-1.1** DDL `dart_corp_profile_history` + 7곳 등록 + `docs/database.md`. `db init` → 테이블 + 인덱스 3개 생성 확인(local `mydb`)
- [x] **F-1.2** 서비스: `sync-corp-profile` 결과를 history에 append(`(corp_code, observed_month)` skip). `RunType` 추가(`dart_corp_profile_history_seed`), 결과 카운터 `history_rows_appended`
      — 옵션 이름은 `--refresh`가 아니라 기존 `--force`다(이 저장소에 `--refresh`는 없다)
      — 함께 고친 것: `CompanyProfileResult`가 다른 OpenDART 결과와 달리 `all_rate_limited`를 들고 있어 `is_opendart_daily_limit_exhausted`가 이 경로에서 **한 번도 걸리지 않았다.** 키 소진 시 법인당 3회 재시도하며 3,959건을 끝까지 돌았고 exit 75가 안 났다. `exhaustion_reason`으로 통일
- [x] **F-1.3** 시드: `dart_corp_master` 현재 행 → `observed_month`, `is_seed=true`. CLI `dart seed-corp-profile-history --observed-month`
      — local `mydb`에서 실행 확인: 1회차 700행, 2회차 0행(idempotent), `observed_at`은 corp master의 `profile_fetched_at`(2026-08-15~18) 유지
- [x] **F-1.4** 첫 실행(prod) **완료 2026-09-08** → [`results/f1_4_first_snapshot_20260908.md`](results/f1_4_first_snapshot_20260908.md)
      — v0.12.0 릴리즈·배포 → `db init`(4분 33초) → 시드 2026-08-01 **3,959행** → 첫 수집 스냅샷 2026-09 **3,959/3,959/3,959, 오류 0, 19분 27초, `status=success`**
      — 시드 재실행 0행으로 prod `ON CONFLICT DO NOTHING` 확인
      — 월 호출 실측 **20분**(3.27 req/s). §3의 13분 추정은 호출당 0.2s만 계산한 값이라 틀렸다
      — 첫 변경률(판정 아님): 코드 0.0505%, **그룹 0.0253%**(문턱 0.3%). 코드 2건 중 1건은 업종 변경이 아니라 정밀도 변경(262→26293)이다
      — 부수 확인: `corp_cls`→`E` 6건 중 5건이 `stock_master` 상폐와 일치하고 `last_seen_date`가 두 스냅샷 사이에 들어간다. 6번째는 `stock_master`에 행이 없는 KONEX 종목이다. `corp_cls`가 `is_active`보다 신선하다
- [x] **F-1.5 완료 2026-09-09** 마트 `dim_industry_pit_daily`(`research/etl/features/industry_pit.py`, `industry_pit_v1`) + 유닛 15개. snapshot 2026-09-08에서 **7,053,322행 / 2007-06-05~2026-09-07 / 빌드 2.3초** — `feat_relation_stat`과 같은 격자다
      — 구간 셋: **PIT 37,703행**(2026-08-18~09-07, 2,544 종목) / 소급 6,650,222행(3,019 종목) / **관측 없음 365,397행**(189 종목). 마지막 것이 새 컬럼 상태다 — `ind_is_backcast`를 `NULL`로 두어 "소급"과 "한 번도 프로필에 없던 종목"을 가른다. 불변식 `(ind_observed_at IS NULL) = (ind_is_backcast IS NULL)`
      — **노출 규칙을 `01` §2.3의 `≤ t`에서 "다음 세션"으로 바꿨다.** 그대로 두면 안전하지 않다: seed는 23:10, 2026-09 스냅샷은 00:14에 관측됐다. 같은 날 노출은 앞의 경우 look-ahead고 뒤는 아닌데 행에는 구분이 없다. `00` §4.3의 국내 관측 규칙(관측일 다음 세션)이 두 경우 다 맞다. 비용은 1세션 지연
      — **변경 플래그는 seed 전이를 세지 않는다.** seed 행은 corp master 현재값 복사본이라 변경 시점을 잡을 수 없다(DDL 주석과 같은 이유). 그래서 `ind_changed_recent_252`가 처음 non-NULL이 될 수 있는 달은 **2026-11**이고, 그 전에는 `FALSE`가 아니라 `NULL`이다. 감지 가능 여부는 누적으로 센다 — 창 안으로만 세면 플래그를 세운 변경이 창 밖으로 밀리는 순간 NULL로 되돌아간다
      — 스펙 5컬럼에 **`ind_group_changed_recent_252`를 더했다.** D-F1 문턱이 그룹 기준이고, 코드 비교만으로는 F-1.4가 측정한 정밀도 변경(262→26293)이 실제 업종 변경과 섞인다
      — fold-up은 `observed_month`마다 다시 푼다(멤버십이 움직인다). 상폐 법인도 세는 쪽을 택했다 — 살아 있는 종목만 세면 라벨 맵이 2026 유니버스로 풀려 2015 세션 대부분이 `OTHER`가 된다. 그쪽이 더 큰 왜곡이다
      — `compute_all --with-features`에 붙였다. 2026-08-23처럼 `dart_corp_profile_history`가 없는 snapshot은 경고 후 skip한다(빈 이력은 skip이 아니라 **실패**다 — 월 수집이 깨진 것이므로)
- [x] **F-1.6** Cronicle 월 이벤트 등록 완료 2026-09-08: `sdc_monthly_corp_profile_history`, 매월 1일 **05:30 KST**, `max_children=1`, `timeout=5400`, **`catch_up=1`**(1일에 호스트가 내려가 있어도 같은 달로 채운다)
      — `DART_PROFILE_FORCE=1`이 **필수**다. 없으면 skip-if-present가 전 법인을 건너뛰고 history에 아무것도 안 들어간다 — 옛 일회성 이벤트를 월 스냅샷으로 못 쓴 이유가 이것이다
      — 05:30을 고른 근거: 04:00 `sdc_daily_opendart_corp`이 실측 **2~4초**에 끝난다. 다음 opendart 잡은 23:30이라 20분 런과 겹치지 않는다
      — 중단된 달을 다시 돌리면 **빈 곳만 채운다**(이미 행이 있는 법인은 DO NOTHING)
      — freshness 월 예산 항목은 아직 안 넣었다 → F-9.8
- [x] **F-1.7 스크립트 완료 2026-09-09** `research/analysis/industry_change_report.py` + 유닛 9개 → [`results/industry_change_202609.md`](results/industry_change_202609.md)
      — **F-1.4의 손계산과 자리까지 일치한다**: 코드 0.0505%(2건), 그룹 0.0253%(1건), 분모 3,959. 독립 검산이 된 셈이다
      — 변경 방향은 **47 → 46 1건**(소매 → 도매). 정밀도/실제 분리: 코드 2건 = 정밀도 1 + 실제 1
      — fold-up은 마트와 **같은 호출**(`compute_industry_observations`)을 쓴다. 리포트와 피쳐가 그룹을 다르게 볼 수 없다
      — **D-F1 규칙을 한 곳 보수적으로 고쳤다(사전등록 수정, 근거 기록).** `01` §4.2는 "3개월 스냅샷(2026-09·10·11)"이라고 적었는데 그 셋의 첫 쌍(08→09)은 seed 쌍이다. 위 F-1.5의 이유로 시점을 잡을 수 없으므로 판정 평균에서 뺀다. 결과를 보고 고친 것이 아니라는 근거 둘: (a) 사유가 데이터 속성(`is_seed`)이고, (b) 빠지는 쌍이 0.0253%로 **가장 낮아** 제외하면 평균이 **올라간다** — 문턱 통과가 쉬워지는 방향이 아니다. 일정은 그대로다: 비-seed 쌍 09→10·10→11·11→12가 **2026-12-01 스냅샷**이면 셋이 되므로 판정은 여전히 2026-12
      - [ ] **첫 판정용 회차는 2026-10 스냅샷 뒤.** 지금 회차(seed 쌍)는 판정에 쓰지 않는다
- [ ] **F-1.8** KIS 종목 기본정보 업종 필드 PoC(`poc/kis_stock_info_industry.md`) → L2 채택 여부
- [ ] **F-1.9** D-F1 판정(2026-12, 3개월) — 문턱 0.3% 고정

### F-2 통계적 peer (`02` §1)

- [x] **F-2.1** `resid_ret` CTE를 `trading_panel`/`price.py`에서 함수로 공유(`feat_price` SQL 불변 확인 — A0 해시). 이미 `trading_panel.build_market_model_sql`로 빠져 있어 `relation_stat`이 그대로 재사용한다. `feat_price` 해시 두 개를 `test_relation_stat.py`에서 다시 고정
- [x] **F-2.2** `dim_peer_monthly`: 월말 252세션 잔차 상관, 유효 189, K=20, numpy → parquet. PIT 테스트. 저장은 K=40(진단 `_k10`·`_k40`을 rank 필터로 얻는다), 컬럼명은 `rank` → `peer_rank`(DuckDB 예약어 회피)
- [x] **F-2.3** `feat_relation_stat`: 5 primary + 진단 `_k10/_k40`, `rel_peer_corr_mean`, `rel_peer_ksic_agree`. **연도별** 파티션 materialize(`mart.materialize_in_parts`) — 월별보다 파트 수가 적고 한 파트가 여전히 수백만 행이라 조인 working set이 작다
- [x] **F-2.4** 마트 검증 리포트: 커버리지·분포·자기상관·규모 축 상관 → [`results/f2_relation_stat_verification.md`](results/f2_relation_stat_verification.md)
- [x] **F-2.5** 사전등록 항목 4 family(부호·horizon·fdr_family `relation`) 확정 → 같은 리포트 §6. YAML은 F-HS-1에서 만든다

### F-3 업종 관계 (`02` §2) — D-F1 뒤

- [ ] **F-3.1** `feat_relation_ind` 6컬럼 + `ind_group_id`. `require_pit` 처리
- [ ] **F-3.2** `feat_fin_scan_daily_ind` 컬럼을 family로 등록(표지 포함)
- [ ] **F-3.3** F-HS-2 config

### F-4 재무위험·생애주기·전이 (`03`)

- [x] **F-4.1** `fin_scan.py`의 vintage 선택·`base_ok`·`available_from`을 `features/fin_vintage.py`로 공유. `feat_fin_scan_daily` SQL 해시 2개 불변 확인(`9656de5e…`/`2615e77d…` — 2026-08-23 snapshot의 `_cache_metadata.json`과 일치)
- [x] **F-4.2** `feat_fin_risk` 9 family(`fin_risk_v1`), TTM·직전 vintage·`dps` 노출 규칙. 7,211,785행
      — 조인 형태를 바꿔야 했다: metric 10 × basis 2 = 20개의 equality+range 조인은 7.2M 세션 × 1.17M 인터벌에서 **완료되지 않았다**(20분 무출력). ASOF 20개도 마찬가지. interval을 먼저 wide로 피벗해 **ASOF 1개**로 읽으면 100초에 끝난다. `feat_fin_scan_daily`는 range 조인을 그대로 둔다(SQL 텍스트가 A0 캐시 키). 두 형태가 같은 행을 고른다는 것을 엇갈린 접수일 fixture로 테스트
- [x] **F-4.3** 생애주기 8조합 매핑 테스트(8개 전수 parametrize), 전이 이벤트 빈도 리포트 → **`fin_lifecycle_transition`·`fin_profit_turn` = continuous**(연평균 0.69/0.27, 문턱 0.05), 나머지 둘은 event cohort 유지
- [x] **F-4.4** 상관 진단 → **경고 2건**: `fin_interest_coverage` × `fin_operating_profitability` ρ=0.88(분자 공유), `fin_net_debt_to_mcap` × `fin_book_to_market` ρ=0.51(분모 공유). 상폐 커버리지 9.2%(`stock_master` 기준)
      — **2026-09-08 갱신:** F-9.3이 종결돼 Decline 판정 보류는 "백필 대기"가 아니라 **영구 한계**다. 상폐 법인 재무 37%가 상한이다(F-9.3 결과 §5.1). 카드 문구를 그렇게 적는다
- [x] **F-4.5** 사전등록 9 family 확정 → [`results/f4_fin_risk_verification.md`](results/f4_fin_risk_verification.md) §8
- [x] **F-4.6 해소 2026-09-08** → [`poc/metric_rules_ext.md`](poc/metric_rules_ext.md). XBRL fallback 16규칙 추가 → `fin_risk_v2`
      — 다섯 family가 **2~4년 앞당겨졌다**: `fin_net_debt_to_mcap` 2020→**2016**, `fin_ext_finance`·`fin_lifecycle_stage`·`fin_lifecycle_transition` 2021→**2018**, `fin_interest_coverage` 2021→**2019**
      — 핵심은 철자였다. DART가 2019년경 taxonomy 접두를 바꿔서 2018년 이전을 덮는 건 `ifrs_` 쪽이다(`ifrs-full_Liabilities` ≤2018 364행 vs `ifrs_Liabilities` 112,827행). `ifrs-full_`만 붙였다면 아무것도 안 늘었다
      — 덧붙이기만 했다: `feat_fin_scan_daily` 입력 9 metric은 행 수 +0이고 **값이 다른 행 0개**. 유닛 1,609개 통과

### F-5 `metric_rules` 확장 (`03` §3)

- [x] **F-5.1 완료 2026-09-09** 태그 커버리지 PoC → [`poc/metric_rules_ext.md`](poc/metric_rules_ext.md) §6~§9. 수집 0, lake snapshot 2026-09-08만 읽었다. 판정 문턱은 F-5.0 §0의 0.5를 그대로 썼다(새로 고른 값이 아니다)
      — **또 철자였다.** 단기차입금은 두 철자(`ifrs-full_`/`ifrs_`)만으로 2015년 커버리지가 **0.003**이다. 세 번째 철자 `dart_ShortTermBorrowings`를 붙이면 같은 해가 **0.823** — 규칙 하나에 260배다
      — 통과 셋: `current_assets` 0.994 / `current_liabilities` 0.993 / (**추가**) `retained_earnings` 0.959, 전부 2015년부터. instant 값이라 TTM 4분기가 필요 없고 분기별 법인-연도 수가 참조 metric과 사실상 같다
      — **`retained_earnings`는 원래 목록에 없었다.** `03` §3이 `fin_altman_z`를 "완전판이 되면"으로 조건부로 뒀는데 그 조건을 정하는 값이라 같이 쟀다. 결과: **Altman Z 다섯 항이 다 있다**(WC·RE·EBIT·MVE·Sales)
      — **`borrowings`는 단일 metric으로 불가하다.** 매핑 규칙 모델이 `(metric, corp, period, basis)`마다 하나만 고르므로 합산을 표현할 수 없다. 성분 둘(`borrowings_short_term` 0.823 / `borrowings_long_term` 0.731)을 등록하고 합은 `feat_fin_risk`에서 한다. 합산 유효성은 확인했다 — 10,012 법인-연도에서 `단기+장기 > 부채총계`가 **0건**
      — `borrowings_current_portion` **탈락**(최대 0.466). 빼면 이자부부채를 중위 **10.2%**, 90분위 **59.7%** 과소계상하므로 카드에 적는다. 문턱을 낮추는 쪽은 택하지 않았다 — 0.4 구간 법인이 대형주에 쏠려 분자 정의가 종목 크기에 따라 달라진다
      — **`rnd_expense` 탈락.** XBRL은 2023년부터만(447 법인), 2025년에도 0.136이다. FS 쪽은 `-표준계정코드 미사용-`/'연구개발비' 5,639행 **203 법인**뿐이다. `dart_Capitalised...` 계열은 바이오 임상단계 **자본화** 개발비 주석이지 경상연구개발비가 아니다
      — 결과로 **F-5.4가 5 family → 3**이 된다: `fin_rnd_to_sales`·`fin_bm_intangible_adj`(Peters-Taylor는 R&D 이력이 재료)를 만들지 않는다
- [x] **F-5.0 완료 2026-09-08** 기존 4 metric에 XBRL fallback 16규칙 추가 → [`poc/metric_rules_ext.md`](poc/metric_rules_ext.md)
      — 판정 규칙 두 개를 **측정 전에 커밋**했다(`05a1b3e`). 처음 제안한 "행 수 50%"는 틀린 양이라 폐기 — TTM은 연속 4분기를 요구하므로 분기의 50%를 무작위로 가지면 사용 가능 확률이 6%다
      — vintage 행 증가: `interest_paid` +38,957 / `investing_cash_flow` +51,668 / `financing_cash_flow` +49,446 / `cash_and_cash_equivalents` +50,335
      — **canonical 마트는 안 건드렸다.** 별도 lake(`data_lake_f50/`, raw·A0 마트는 심볼릭 링크)에서 측정했다. F-HS-1은 `07` §7대로 새 snapshot에서 돈다
- [x] **F-5.2 완료 2026-09-09** catalog 5 + 규칙 26개 추가 → **catalog 29 → 34, 규칙 129 → 155**. 유닛 9개(`test_metric_rules_ext.py`)
      — 기존 29 metric **golden 불변 확인**: 전체 유닛 스위트 통과. 새 규칙의 `rule_code`가 기존 것과 하나도 겹치지 않고, 29 metric 어느 것도 새 규칙을 받지 않는다는 것을 불변식으로 박았다
      — fallback 우선순위 규약도 테스트로 박았다: 새 metric의 XBRL fallback은 전부 statement 규칙보다 **낮은 우선순위**다(공백만 메우고 보고된 값을 덮지 않는다)
- [x] **F-5.3 완료 2026-09-09** vintage 3,301,030행을 in-memory로 재빌드해 측정 → [`poc/metric_rules_ext.md`](poc/metric_rules_ext.md) §10~§12. **canonical snapshot은 안 건드렸다** — F-HS-1이 발행한 2026-09-08 `feature_mart`를 아예 만지지 않는다(F-5.0의 심볼릭 링크 lake보다 간단하다)
      — 다섯 개 다 문턱을 넘어 **표본 시작은 전부 2015**다. `current_assets`·`current_liabilities`는 0.998~1.001로 참조 metric과 같은 급이다
      — **역산 비율이 참조 metric과 똑같다**: 2015~2018은 99% 이상이 XBRL fallback인데 `total_assets`·`total_liabilities`가 이미 그렇게 작동해 왔다(statement 규칙이 `ifrs-full_` 철자만 매핑하고 2018년 이전 FS 행은 `ifrs_`를 쓴다). 신규 metric이 더 역산에 기대는 것이 **아니다**
      — 기록할 한계 둘. (a) `retained_earnings`의 2015~2016은 raw 0.959인데 **vintage 0.72**다(나머지 넷은 raw와 vintage가 거의 같다). 0.5는 넘으므로 표본 시작은 2015이지만 그 두 해의 횡단면이 72%임을 카드에 적는다. (b) **차입금은 최근으로 갈수록 나빠진다** — 단기 0.826(2015) → 0.552(2025), 장기 0.714 → **0.495**(2025년 문턱 아래). 방향이 보통과 반대다
      — (b)의 원인은 2023년 이후 taxonomy로 보인다. `ifrs-full_Borrowings`(1,118 법인)와 `ifrs-full_BorrowingsInterestRate`(1,095 법인)가 2023년부터만 있고 둘 다 차입금 명세 주석 모양이다. **성분에 붙이지 않았다** — 이름이 합계이고 커버리지가 0.40이라 붙이면 어떤 법인은 단기만, 어떤 법인은 총액을 담아 분자의 뜻이 법인에 따라 달라진다. 행 수로는 판정할 수 없다
      — `borrowings_long_term`만 2015~2022 역산이 0%다. 주 규칙 `dart_LongTermBorrowingsGross`가 statement 쪽에서 2015~2026을 고르게 덮는다(87,019행 / 2,384 법인) — 철자 문제가 없는 유일한 신규 metric이다
- [x] **F-5.4 마트 완료 2026-09-09** `feat_fin_risk`에 파생 family **3개** 추가 → `fin_risk_v3`. 유닛 19개(`test_fin_risk_derived.py`). ~~`fin_rnd_to_sales`·`fin_bm_intangible_adj`~~는 F-5.1에서 재료가 없어 취소됐다
      — `fin_current_ratio` = 유동자산 / 유동부채. 분모 0은 NULL이다
      — `fin_borrowings_to_mcap` = (단기 + 장기) / 시가총액. 성분 하나가 없으면 **NULL이고 0으로 안 채운다** — 없는 쪽을 0으로 보면 공시가 부실한 법인이 부채가 적은 것으로 보이고 그 오차가 공시 품질과 상관된다. 감사용으로 `fin_borrowings_total`(원 단위)을 같이 낸다
      — **`fin_altman_z`는 완전판이다**(다섯 항). 가중치는 Altman(1968) 표 1 그대로이고 이 저장소가 다시 맞출 것이 아니다(`LIFECYCLE_STAGES`와 같은 규칙). 다섯 항을 각각 컬럼으로 내고, **하나라도 NULL이면 점수는 NULL이다** — 부분합은 매출 라인이 없는 법인을 진짜로 낮게 나온 법인과 같은 분포에 놓는다
      — EBIT는 `operating_income`으로 근사한다. 진짜 EBIT는 이자·법인세를 되돌려 더해야 하고 canonical에 그 항목이 없다. 근사임을 적었고 조용히 같다고 하지 않았다
      — **`financial_quarters.INSTANT_METRICS`에 다섯 개를 넣어야 했다.** `metric_rules`에만 등록해도 `_ALL_QUARTERED_METRICS`가 B-3 입력을 걸러서 `feat_fin_risk`까지 오지 않는다. F-5.2에서 놓친 연결이다
      - [ ] **커버리지·분포 리포트는 다음 F-HS config의 A0에서 낸다.** `fin_risk_v3`로 올려서 마트를 지금 다시 만들면 F-HS-1이 발행한 2026-09-08 `feature_mart`를 덮는다 — F-5.0이 지킨 규칙("canonical 마트는 안 건드린다")대로 미룬다. 그 config에서 어차피 재빌드된다

### F-6 `fin_sue` 백필 (`04` §1)

- [x] **F-6.1 완료 2026-09-09** `research/analysis/sue_backfill_targets.py` + 유닛 12개 → [`results/sue_backfill_targets_202609.md`](results/sue_backfill_targets_202609.md). 수집 0
      — **필요한 것은 13분기가 아니라 연속 9분기다.** `sue_event.py`의 `comparative_eps`가 `value_lag_4q` — **같은 보고서 안의 비교 컬럼**이라(`comparative_policy='as_was_lag4q'`) 보고서 하나가 `seasonal_change` 하나를 만든다. `history_count >= 8` + 자기 자신 = 9. 비교값이 4분기 전 보고서에서 왔다면 13개였다
      — 규모: 원본 정기보고서 **110,687건 / 3,407 법인**, XBRL 없는 접수 25,370건, 그중 **대상 22,700건 / 2,807 법인**(10.5% 제외)
      — **좁히기의 핵심은 "채울 수 없는 창"이다**(7,041개). 아홉 분기 중 하나라도 애초에 제출되지 않았으면 아무리 받아도 완성되지 않으므로 그 안의 결측은 대상이 아니다. 창 판정: 완성 가능 48,810 / 이미 완성 33,141 / 채울 수 없음 7,041
      — 연도 분포는 **2015년이 지배적**이다(대상 6,267건). 2015년은 원본 접수 8,036건 중 6,791건이 XBRL이 없다(84.5%). 2016~2023은 연 1,400~1,900건, 2024~2025는 650~1,130건
      — 호출 **약 22,700회**. 키 2개면 2일. slice ledger를 **쓴다**(여러 날에 걸친다). F-9.3과 달리 회수 불가 대상이 반복되는 문제는 없다 — 대상은 제출된 보고서이고 없는 것은 XBRL 문서뿐이다
      — **정정:** `04` §1.3의 `--out targets/sue_xbrl_targets.csv`는 확장자가 틀렸다. `backfill-xbrl-receipts`가 파싱하는 것은 **JSON lines**다(`cli/app.py`). 대상 파일은 저장소에 넣지 않는다 — 같은 snapshot에서 그대로 재생성된다
- [x] **F-6.2 완료 2026-09-10 02:24 KST** → [`results/f6_2_sue_backfill_20260910.md`](results/f6_2_sue_backfill_20260910.md). **00:04:17~02:24:19 (2시간 20분), 47 run 전부 `success`, 오류 0, exit 75 없음**
      — 들어온 것: `dart_xbrl_document` 95,720 → **111,385**(+15,665), ledger `fnlttXbrl` no_data **7,036**. 합이 22,701로 대상 22,700을 전부 설명한다. **대상의 31%는 OpenDART에 XBRL이 아예 없다** — ledger가 TTL 없이 은퇴시켰다
      — **`fin_sue` 표본이 2.27배가 됐다**(접수·XBRL 가용성 기준): SUE 산출 가능 분기 33,141 → **75,293**, 9분기 구간 보유 법인 2,395 → **2,743**
      — **구간 수는 3,465 → 2,895로 줄었다.** 이것이 핵심 신호다 — 공백을 메우면 끊어진 짧은 구간이 **하나로 이어진다**. 구간이 기여하는 분기는 `길이 − 8`이라 이어붙이기가 2.27배를 만든다. F-6.1이 "완성 가능한 창 안의 결측만" 고른 규칙이 의도대로 작동했다는 증거다
      — **이 숫자는 상한이다.** 잰 것은 접수·문서 가용성이고 그것이 F-6.2가 바꿀 수 있는 유일한 입력이다. 백필 전에도 산출 가능 분기가 33,141이었는데 coverage가 0.00이었으므로 vintage 층(`captured_vintage_status`·`is_revision`·`controlling_net_income`·주식수)에서 더 걸린다. 실측은 F-6.3
      — 디스크: `dart_xbrl_fact_raw` 136 → **159 GB**, DB 221 → **244 GB**(+23 GB). 추정 32 GB보다 적은 것은 31%가 no_data였기 때문이다
      — **04:00까지 끝내라는 요구로 두 값을 조정했다.** blackout 02:30 → **03:40**(02:30은 작업 단위 35분인 S-1의 값이고 이 드라이버는 4분이라 여유가 대상의 20배였다), rate limit 0.2 → **0.1**(S-1의 기본값이고 F-9.3이 그 값으로 131,292 요청·실패 0). 실측 **약 2.7 req/s**, 청크당 183초. 추정 2h22m 대비 실측 2h20m
      — **03:40 가드는 쓰이지 않았다** — 02:24에 끝나 창이 닫히기 전이고 lock도 그때 풀려 04:00 체인은 무경합이다. 전날 23:30 filings도 무영향(23:30:00~23:43:34, code=0, 814.3s)
      — 야간 무인 실행을 위해 **루프를 실패에 견디게 고쳤다**: 전에는 청크 하나가 죽으면 루프가 끝나 아무도 못 띄웠다. 연속 3회까지 견디고 120초 뒤 재시도한다
- [ ] **F-6.3** 새 snapshot 재빌드 → `effective_start`·coverage 확인 → 기존 사전등록으로 재판정(F-HS-1과 같은 snapshot)

### F-7 구조화 이벤트 (`04` §2·§3)

- [ ] **F-7.1** DS005 PoC: endpoint 목록·인자·응답·볼륨 실측 → `poc/ds005_events.md`, D-F3 확정
- [ ] **F-7.2** `dart_major_event_raw` DDL + 7곳 + 어댑터(statement enum) + 서비스·CLI + 테스트
- [ ] **F-7.3** 백필 2015~2025 + 일 증분 이벤트(23:45)
- [ ] **F-7.4** `feat_event_decision` 4 family → F-HS-2
- [ ] **F-7.5** `elestock` 월 수집 시작(`dart_insider_holding_raw`) — D-F2 확정 2026-09-07. DDL + 7곳 + 어댑터 + 월 이벤트, 첫 실행 기록

### F-8 매크로 2·3단계 (`05`)

- [ ] **F-8.1** ECOS item_code 실호출 확정(`817Y002` AA-/BBB-/CD91, `722Y001`, `513Y001`) → `poc/ecos_stage2_items.md`
- [ ] **F-8.2** 시리즈 10 + 파생 feature 추가(`common_features.py`), golden 불변 확인, `common seed`
- [ ] **F-8.3** 백필(`common sync --sources ecos,fred`), readiness·freshness 통과
- [ ] **F-8.4** inactive 시리즈 정리(주석·`active=False` 유지)
- [ ] **F-8.5** 국면 R7~R11 사전 계산(점유율·지속·G2) → 기록
- [ ] **F-8.6** Phase C 2라운드 사전등록(Q1~Q6 + X3) — F-4 완료 뒤 → 실행
- [ ] F-8.7 3단계 어댑터: 관세청 수출·SOX → `macro_beta_export`·`macro_beta_sox` — D-F5 확정: **F-HS-C2에서 2단계 국면이 하나 이상 `screen_pass`한 뒤에만** 착수

### F-9 인프라·PIT (`06`)

- [ ] **F-9.1** `feat_fin_ratios_pit`(`_pit` 접미) + `fin_pit` 대비 차이 리포트 → 모델 스트림 통보(E3-b)
- [x] **F-9.2 완료 2026-09-08** `holidays_krx.csv` 46 → 201행. 거래일 4,972 → **4,828**로 반영 확인
      — 가드 두 개가 첫 실행에서 걸렸다: 교차검증이 **기존 CSV에 빠진 실제 휴장일 4건**을 찾았고(국군의 날 임시공휴일 2024-10-01, 설 임시공휴일 2025-01-27, 대선일 2025-06-03, KRX 연말휴장 2025-12-31) — 2014~2023을 채우면서 2024~2026도 고쳤다. 주말 항목 4건은 비교에서 제외
      — 연속 결측 가드가 `2026-08-24~12-31`(94일, 데이터 밖)을 잡아 범위를 마지막 세션으로 clamp했고, `2017-10-02~10-09`도 잡혔지만 **진짜**였다(임시공휴일로 이어진 추석 10일 연휴) → 문턱 6 → 8
      — golden 재생성은 불필요했다. 유닛 픽스처가 합성 날짜라 겹치지 않는다
- [x] **F-9.3** S-1 잔여 **종결 2026-09-08 — 더 받을 것이 없다** → [`results/f9_3_s1_remainder_20260908.md`](results/f9_3_s1_remainder_20260908.md)
      — 176 유닛 전수 완주(5시간 45분), 실패 0, 요청 **131,292건**, 저장 **0행**. 커버리지 변화 없음(재무 3,093 / 주식수 3,430 / XBRL 3,093)
      — 원인: 상장 법인은 이미 99.7% 완결이고 공백은 전부 상폐(E 1,230중 455=37%)·코넥스(N 105중 19)에 있다. 남은 866중 356은 정기보고서를 냈는데도 OpenDART가 `013 조회된 데이타가 없습니다`로 답한다
      — PoC(양성 대조 포함): `fnlttSinglAcnt`(주요계정)도 같은 013. 삼성전자는 000/229행 → 키·요청 형태는 정상. **재무제표 API가 현재 규제 대상 법인만 서빙하는 것**으로 보인다
      — **생존편향은 수집으로 닫히지 않는다.** F-4의 "S-1 잔여 뒤 판정"은 **영구 한계**로 문구를 바꿔야 한다(§5.1)
      — Cronicle 이벤트는 삭제했다. 드라이버 스크립트는 남긴다. 이유: skip-if-present가 *저장된* 행만 건너뛰므로 회수 불가 861개는 매 실행 13.1만 요청을 반복한다(스크립트에 "no-op가 된다"고 적은 것은 틀렸다)
      — `06` §3 정정: 누락은 1,300이 아니라 866, 호출은 4.3만이 아니라 13.1만, `--include-delisted`는 없는 플래그
- [ ] **F-9.4** S-2: `universe backfill-master` → 상폐 종목 `prices backfill`
- [ ] **F-9.5** S-3: `delisted_date` 확정, 상폐 종목 라벨 정책 문서화
- [ ] **F-9.6** `dim_universe_daily_krx`(`daily_market_cap` 기준) + 집합 차이 리포트
- [x] **F-9.7 완료** Cronicle 정리 — 2026-09-08 시작, **2026-09-09 마무리**
      — [x] 월 corp-profile 이벤트 등록(F-1.6과 같은 항목)
      — [x] `emsugdoe907`("SDC Backfill DART Corp Profile (one-time)") **삭제**. 2026-08-15에 1회 1,300.6초 실행 code=0으로 목적 달성, `--force`가 없어 월 스냅샷으로는 못 쓴다
      — [x] `sdc_backfill_s1_remainder` **삭제**(F-9.3 종결, 이유는 그쪽 §4)
      — [x] `emsugmrjp0a`("SDC Backfill N3 Universe Snapshots (one-time)") **삭제 2026-09-09** (`{"code":0}`). `06` §4가 삭제 대상으로 지목했다
            지우기 전 실측: 마지막 실행 2026-08-16 09:06 `jmsv1p53q0n`, 3,699초 뒤 **수동 abort**(code=1) — KRX가 이 호스트를 막은 날이다. 그 앞 2026-08-15 23:53 `jmsuhx5wl0d`는 code=0으로 2014~2018을 넣었다. **남은 데이터 공백은 F-9.14**이고 스크립트를 되살리는 방식이 아니다(pykrx 경로)
      — [x] `emr0r4xgb0h`("SDC Common Backfill 2015 (one-time)") **삭제 2026-09-09** (`{"code":0}`)
            지우기 전 실측: 마지막 실행 2026-07-04 06:08 `jmr5fdw6c0e`가 **32,174초(8시간 56분) code=0으로 완주**했다. 그 앞 세 번(`jmr0r6nir0i`·`jmr23rbc20y`·`jmr4nuu1n01`)은 code=1이다. 백필은 그때 끝나 있었고 남은 것이 이벤트 삭제뿐이었다
      — 같은 조회에서 확인한 것: 전체 20 이벤트, active job 0, `sdc_backfill_s1_remainder`와 `emsugdoe907`은 실제로 없다(삭제 반영됨), `sdc_monthly_corp_profile_history`는 `timing={1일 05:30}`·`catch_up=1`·`max_children=1`·`timeout=5400`으로 등록돼 있다. `manual`로 보이는 일별 이벤트 7개는 전부 chain으로 걸려 있다(`opendart_corp`→`financials`→`share_info`→`xbrl`, `fdr_universe`→`pykrx_prices`→`krx_flows`→`krx_common`, `ecos_common_daily`→`ecos_common_macro`)
      — 월 insider(F-7.5)·일 major-events(F-7)는 수집기가 아직 없어 해당 없음
      — **삭제 뒤 확인(2026-09-09 19:18):** 전체 이벤트 20 → **18**, 삭제 대상 0개 남음. `timing=false`로 남은 7개는 **전부 chain 대상**이므로(`opendart_corp`→`financials`→`share_info`→`xbrl`, `fdr_universe`→`pykrx_prices`→`krx_flows`→`krx_common`, `ecos_common_daily`→`ecos_common_macro`) 고아 수동 이벤트는 없다. 같은 시각 active job은 정상 스케줄 잡 하나(`sdc_kis_flows_trial`, 19:00 시작)뿐이었다
      — 삭제는 `delete_event` API로 했고 둘 다 `{"code":0}`. 지우기 전에 두 이벤트의 정의(`params.script` 포함)를 받아 뒀고, 없어지는 정보는 F-9.14에 옮겨 적었다
- [ ] **F-9.8** freshness 월 예산 항목
- [x] **F-9.9 완료 2026-09-09** readiness 게이트가 기본값에서 통과 가능해졌다. 원인 둘을 다 고쳤고 유닛 13개(`test_readiness_window.py`)
      — **(a) 격자가 데이터 끝을 넘어 뻗었다.** `available_from_date`는 *예고된* 공개일이라 월별 ECOS 시리즈가 미래 날짜를 정당하게 들고 있다 — snapshot 2026-09-08에서 `macro_cpi`·`macro_consumer_sentiment`가 **2026-09-21**인데 마지막 가격 세션은 09-07이다. raw max를 그대로 쓰면 격자가 10세션 더 뻗고 모든 일별 시리즈가 아직 오지 않은 세션의 결측을 뒤집어쓴다. `_common_feature_calendars`가 이제 마지막 가격 세션으로 clamp한다(`ohlcv_view` 인자가 이미 있었는데 안 쓰이고 있었다). 가격이 관측보다 앞서도 **격자를 늘리지는 않는다** — clamp는 상한 보정이다
      — **(b) 파생의 워밍업을 결측으로 셌다.** 판정 시작점을 첫 *fact*가 아니라 **첫 non-NULL 값**으로 옮겼다. YoY는 입력 첫날부터 행을 내지만 12개월이 차기 전에는 값을 못 담는다(`macro_cpi_yoy_latest` 253개). 앞머리 NULL 구간은 구멍이 아니라 워밍업이다
      — **구간 안의 결함은 그대로 잡힌다.** 창을 옮기면 `coverage_report`의 전체 격자 카운트를 재사용할 수 없어(앞머리 NULL도 실제 fact다) `_window_counts`로 창 안에서 다시 센다. 구간 내부 NULL·결측 날짜·PIT 위반은 전부 여전히 blocker다 — 2017-10-10 추석 휴장이 보이는 이유가 그것이다
      — 잘려나간 워밍업은 `warmup_null_count`로 **보고한다.** YoY가 12개월치를 갖는 것은 정상이고, 없어야 할 피쳐에 큰 값이 찍히면 그건 진짜 발견이다
      — `coverage_report`는 안 건드렸다. golden(`golden/common_feature_reports.json`)에 얼려 있고 별개 리포트다

- [x] **F-9.10 완료 2026-09-09** `run_spec.json`의 `command_line`이 실제 인자를 버렸다. `command_line = ["horizon_scan", *(argv or [])]`인데 `__main__` 경로는 `argv=None`으로 들어와 `parse_args(None)`이 `sys.argv`를 직접 읽으므로 기록에는 `['horizon_scan']`만 남았다. 2026-08-30 run 넷이 다 그렇다
      — 고친 방법: `effective_argv = list(argv) if argv is not None else sys.argv[1:]`를 한 번 정하고 **파싱과 기록에 같은 리스트**를 쓴다. 같은 버그가 `horizon_scan_phase_c.main`에도 있어 같이 고쳤다
      — `argv=[]`(진짜 인자 없음)과 `argv=None`(인자를 sys.argv에서 읽음)이 `or` 때문에 구분되지 않았던 것이 원인이다. 유닛 4개로 두 호출 형태를 다 박았다
      — **라운드 경계에서 넣었다**(F-HS-1 완료 뒤). 이후 run의 `run_spec` 내용이 바뀌므로 A/B 사이에 넣으면 안 되는 변경이다. `config_hash`는 그대로라 발행된 run의 계보는 영향이 없다

- [x] **F-9.11 완료 2026-09-09** 문턱 근처 셀에 Monte Carlo 구간을 같이 보고한다. 유닛 10개(`test_temporal_placebo_marginal.py`)
      — **판정 규칙은 안 바꿨다.** `temporal_p_max`·`temporal_long_cell_repeats`는 얼린 config에 있고 결과를 보고 그것을 움직이는 것이 사전등록이 막으려는 바로 그것이다. replicate를 늘리는 쪽은 config 변경이라 새 층이 된다. 바뀐 것은 **노이즈 안에서 갈린 판정에 표시가 붙는다**는 것뿐이다
      — 새 출력 셋: `p_temporal_nw_se`(추정치 자체의 오차), `temporal_marginal_band`, `temporal_null_marginal`. 카드와 placebo 절이 `(marginal: within 2 MC SE of p_max, ±0.060)`을 붙인다
      — **밴드는 관측된 p̂이 아니라 `p_max`에서 잰다.** p̂에서 재면 p가 작은 곳에서 밴드가 좁아져 측정된 flip 하나를 놓친다 — `mcap_krx_log|cum|0|120`의 0.0495는 0.10에서 0.0505 떨어져 있는데 p̂ 기준 2-SE는 0.0434다. `p_max=0.10`·100 replicate 기준 밴드는 0.060이고 구간 0.04~0.16이 **측정된 flip 둘을 다 담는다**
      — 답하는 질문이 다르기 때문이다: "재추출이 이 셀을 문턱 반대편으로 보낼 수 있나"는 `p_max` 근방에 대한 진술이다
- [x] **F-9.12 완료 2026-09-09** `register_phase_b_marts`가 계약 불일치를 재빌드로 처리한다. `--force` 플래그를 붙이는 쪽은 택하지 않았다 — 사람이 미리 알아야 하고, 이미 맞는 마트까지 전부 다시 만든다. 지금은 **불일치한 것만** 다시 만든다
      — `research/etl/mart.py`에 `StaleMartContract(RuntimeError)`를 두고 9개 raise 자리를 옮겼다. 메시지 문자열로 구분하면 문구가 바뀔 때 조용히 안 걸리기 때문이다. `RuntimeError` 하위라 기존 `except RuntimeError`와 `pytest.raises(RuntimeError, match=...)`는 그대로 동작한다
      — `register_mart_view`의 raise도 같은 타입으로 바꿨지만 **동작은 안 바꿨다.** A0 마트를 bind할 때 불일치하면 A0와 Phase B의 config가 다르다는 뜻이므로 죽는 것이 맞다
      — 진짜 산식 변경도 여전히 재빌드로 처리된다(그게 맞다). 재빌드는 공짜가 아니라서 warning으로 남긴다
      — **`force=True` 중간에 끊으면 마트가 빈다.** `materialize`가 쓰기 전에 `rmtree`하기 때문이다. 2026-09-09에 검증 중 `feat_fin_scan_daily`를 이렇게 날려 다시 만들었다. 발행된 run 산출물은 별도 디렉터리라 무사했다

- [x] **F-9.14 완료 2026-09-09** 월말 PIT 유니버스 스냅샷 2019~2026-08을 KRX Open API로 채웠다
      — 최종: `KRX_OPENAPI_BACKFILL` **92개**(2019-01-31~2026-08-31) + `PYKRX_BACKFILL` 60개(2014-01-31~2018-12-31) = **152개**. 2014~2025 전 연도가 정확히 12개월, 2026은 8개월(1~8월). 종목 수는 1,912(2014) → 2,788(2025)로 단조 증가한다
      — 본 실행 **20:01:14~20:03:51 (2분 37초)**, `--end 2026-08-31`: 시도 147 / 스킵 55 / 저장 82 / 항목 208,669 / 오류 10. 보정 10건을 더해 92가 됐다
      — Open API의 과거 깊이는 문제가 아니었다. 실행 전에 `daily_market_cap` 3,010일이 **전량 `KRX_OPENAPI`**로 2014-06-02까지 있는 것을 확인했고(옛 N3 실패는 API 한계가 아니라 pykrx 경로가 막힌 것), 실제로 월말당 약 1.7초·시장당 1콜로 받아졌다
      — **오류 10건의 원인은 prod 배포본의 휴장일 CSV가 46행이라는 것이다.** F-9.2의 201행이 아직 배포되지 않았다(로그 첫 줄 `Loaded 46 holidays`). 옛 캘린더는 월말을 하루~이틀 뒤로 잡고 그 날은 실제 휴장이라 Open API가 빈 응답을 준다 → `refusing to record an empty universe`. 실행 전에 두 CSV의 월말을 비교해 **어긋나는 10개 달을 예측했고 실패 목록이 정확히 일치했다**: 2019-12(31→30), 2020-04(30→29), 2020-09(30→29), 2020-12(31→30), 2021-12(31→30), 2022-01(31→28), 2022-12(30→29), 2023-09(29→27), 2023-12(29→28), 2025-12(31→30)
      — 보정은 릴리즈 없이 됐다. `month_end_trading_days`가 `window_end = min(end, last_day)`로 clamp하므로 `--start <월>-01 --end <실제 마지막 세션>`을 주면 옛 캘린더에서도 그 날을 고른다. 옛 CSV가 새 CSV의 **부분집합**임을 먼저 확인해 이 방법이 안전함을 보장했다
      — **드라이버 루프에서 두 번 헛돌았다.** `while read ... <<HEREDOC` 안에서 `docker compose run`이 **stdin을 물려받아 heredoc을 전부 읽어버려** 다음 `read`가 EOF를 본다. 그래서 1·2차 실행이 각각 한 달만 하고 정상 종료했다(`failures=0`으로 끝나 조용하다). for 루프 + 자식 `< /dev/null`로 고쳐 3차에서 8건이 다 돌았다. **앞으로 prod 드라이버 루프는 이 형태로 쓴다**
      — 이 사이 **20:16에 호스트가 재부팅**됐다(정상 shutdown, 192.168.0.23에서. 이 작업과 무관하다). `/tmp`가 비워져 1·2차 로그가 사라졌으므로 3차는 로그를 `/home/whi/`에 뒀다. 재부팅 뒤 `sdc-postgres` healthy, Cronicle(pid 1464) `get_schedule` code=0·18 이벤트로 정상 복귀를 확인했다
      — 남은 두 가지(별 항목):
        - [x] **F-9.2 배포 완료 2026-09-09** v0.13.0 릴리즈(`43838ae`) → 이미지 빌드 성공(1분 3초) → `deploy_to_sj2.sh` → `docker compose pull collector`. compose 이미지 태그 `ghcr.io/sjleekor/sdc:v0.13.0` 확인
              — **검증은 F-9.14가 지목한 열 달의 월말로 했다.** 배포된 이미지에서 2019-12→12-30, 2020-04→04-29, 2020-09→09-29, 2020-12→12-30, 2021-12→12-30, 2022-01→01-28, 2022-12→12-29, 2023-09→09-27, 2023-12→12-28, 2025-12→12-30 — **열 개 전부 새 캘린더 값**이다. 거래일 수도 로컬과 일치한다(2014-01-01~2026-09-07 기준 3,112)
              — 같이 배포된 것: **K-5**(`universe sync --source krx-openapi`, lock domain `fdr` → `krx_marketdata`). prod 래퍼에서 확인했다 — 죽은 fdr 경로를 쓰던 18:30 체인 머리가 이번 배포로 살아난다
              — `sdc-postgres` healthy. collector는 persistent 서비스가 아니라 Cronicle이 `docker compose run --rm`으로 띄우므로 다음 잡부터 새 태그를 쓴다
        - **정정(2026-09-09):** 어제 이 자리에 "월 1회 이벤트가 없어서 필요하다"고 적었는데 **틀렸다.** `docs/operations.md` §41이 `0 5 1 * *` 줄 바로 위에 **"prod에서는 돌리지 않습니다 — daily_market_cap이 일별 PIT 유니버스를 대신합니다(04_w1_pit_universe.md §3.6). 감사·교차검증이 필요할 때만 씁니다"**라고 적어 둔다. 즉 스케줄 없음이 설계 결정이고 공백이 아니다. 월말 스냅샷은 감사·교차검증 자산이며 일별 PIT 유니버스는 F-9.6(`dim_universe_daily_krx`)이 맡는다. **이벤트를 만들 이유가 없다**
        - [ ] (참고, 별건) **lock domain이 바뀌었다.** K-5 커밋 `89444fb`이 `universe-sync.sh`를 `fdr` → `krx_marketdata`로 옮겨 18:30 일일 체인 머리가 이 domain을 쓴다. 체인은 단계마다 lock을 따로 잡고 놓으므로 **18:30~19:00에 긴 백필을 끼워 넣으면 다음 단계가 900초 대기 후 실패한다**(`SDC_LOCK_CONFLICT_MODE=fail`). 다음에 이 백필을 감사 목적으로 돌릴 때는 18:30~19:00과 20:00(`sdc_daily_market_cap`)을 피한다. 오늘 실행(20:01)은 market_cap이 20:00:09에 끝난 뒤라 무경합이었다
- [ ] **F-9.14 배경**(2026-09-09 발견) **월말 PIT 유니버스 스냅샷이 2014~2018만 있다.** prod `stock_master_snapshot`은 `PYKRX_BACKFILL` 60행(2014-01-31~2018-12-31, 월말 12 × 5년)과 `FDR` 83행(2026-04-10~09-07, 일별 수집)뿐이다. **2019-01~2025-12의 84개 월말이 비어 있다** — `emsugmrjp0a` 이벤트가 2026-08-16 실행에서 2014~2018을 넣고 2019년부터 전부 실패했고(KRX가 이 호스트를 막은 날), 그 뒤 다시 돌지 않았다
      — 지금은 메울 수 있다: `universe backfill-snapshots`의 기본 source가 KRX Open API(K-4)이고 `deploy/prod/bin/universe-backfill-snapshots.sh`가 그 경로를 쓴다(`docs/operations.md` §583이 pykrx 문을 닫힌 것으로 적는다). 옛 이벤트 스크립트의 연도별 `docker compose run` 쪼개기·45초 sleep은 **pykrx 로그인 수명 때문에 있던 것이라 Open API에서는 필요 없다** — 그래서 그 스크립트를 되살리지 않고 새로 짠다
      — 이 공백은 F-9.6(`dim_universe_daily_krx` 집합 차이)과 F-1의 소급 판정에 걸린다. 2019~2025 PIT 유니버스가 없으면 그 구간 유니버스는 현재 상태의 소급이다
- [x] **F-9.13 완료 2026-09-09** 09-08 snapshot의 `feat_fin_scan_daily`가 **0바이트로 남아 있었다** — `part-000000.parquet` 0B, `_cache_metadata.json` 없음. F-9.12가 적은 `force=True` 중간 중단 자국이고, 그때 "다시 만들었다"고 적은 것과 실제 디스크가 달랐다. 같은 snapshot의 다른 마트 20개는 정상이었다(`feat_fin_risk` 77MB, `feat_relation_stat` 555MB)
      — 고친 방법: Phase B run과 **같은 경로**를 그대로 호출했다(`register_phase_b_marts(con, lake)`, `force` 없이). F-9.12의 새 동작이 설계대로 걸렸다 — 로그가 `mart cache metadata is missing for 'feat_fin_scan_daily' … rebuilding under this run's contract`를 남기고 **그 하나만** 다시 만들었고, 계약이 맞는 12개는 등록만 됐다
      — 결과: **7,242,208행 / 2007-06-05~2026-09-07 / 208MB, 26분**(1,558초). 13 마트 available. `sql_hash`가 `9656de5eb96b9b14`로 08-23 빌드와 **같다** — F-4.1이 박아 둔 값이고, 산식이 그대로라는 뜻이라 발행된 run과 같은 정의다. `analysis_config_hash`는 이제 `3ca949e6`로 찍혔다
      — 34분은 13개 전량이 아니라 사실상 이 한 마트 값이었다(`register_phase_b_marts` docstring이 "most of it feat_fin_scan_daily"라고 적은 대로)

### F-HS Horizon Scan 사전등록·실행

- [x] **F-HS-1** overlay config `horizon_scan_expansion_202609.yaml`, hash `3ca949e6`, `registered_at: 2026-09-09`, 커밋 `e704e08`. F-2 4 + F-4 9 family = 66 cell, Phase B 102 → 168. 전 family `fdr_include: false`이라 Phase A 75개는 그대로고 앞선 세 층 hash(`ab0de634`/`889c3e83`/`236d0d35`)도 그대로다(테스트가 박아 둔다). 기록 → `results/f_hs1_preregistration_record.md`
      — 이 층이 드러낸 것 둘. (a) `feat_relation_stat`에 `_lag1`이 없어 **없는 컬럼에 계약을 얼릴 상황**이었다 → `relation_stat_v2`. 산식 불변(peer 11,292,902행·마트 7,053,322행·리포트 수치 전부 동일)이고, 연 단위 part가 매년 첫 세션 lag1을 NULL로 만드는 문제와 1년 넘는 거래 공백 121행을 `LAG1_MAX_GAP_DAYS=365`로 정리했다. (b) event cohort 2건은 `run_phase_b_event_scan`이 `fin_sue_event` grain에 묶여 있어 `fin_risk_event`를 의존성에 적어 `blocked_exploratory`로 얼린다
- [x] **F-HS-1 실행** snapshot 2026-09-08 → A0 → A `20260909T032631-eb4867bb` → B `20260909T055405-eb4867bb` → AB `20260909T121215-eb4867bb`. **기존 discovery 변화 0** (공유 177 셀, 얻음 0·잃음 0), **단계 0 exact match** (`daily_ic` 892,420행 양방향 0, 실 스캔 412 셀 `|ΔIC|=0.0`). `m_ab` 177 → **231**, discovery 103 → **147**, `screen_pass` 53 → **85**, 등급 A=43·B=42·C=55·D=16. 기록 → `results/f_hs1_run_20260909.md`
      — 짚어둘 것 셋. (a) 공유 셀 `screen_pass` 2건이 뒤집혔는데 둘 다 `p_temporal_nw`가 0.10을 넘나든 것이고 나머지 통계는 비트 동일하다(→ F-9.11). (b) F-4 7 family가 전부 등급 B 상한인데 **누락이 아니라 측정**이다 — `revision_ratio` 0.1014~0.1015, `fin_interest_coverage`는 `operating_income` fallback 0.9121. (c) Phase B가 마트 계약 스탬프 때문에 한 번 죽었다(→ F-9.12)
- [x] **F-HS-1 결과 문서** `results/f_hs1_run_20260909.md`. **FS3 신규 11 컬럼** — A: `rel_peer_dispersion_20d`·`rel_own_minus_peer_20d`·`rel_peer_mom_20d`·`rel_peer_bigcap_lag_ret_5d`, B: `fin_net_debt_to_mcap`·`fin_ext_finance_to_assets`·`fin_interest_coverage`·`fin_lifecycle_stage`·`fin_debt_to_assets`·`fin_profit_turn`·`fin_lifecycle_transition`. 레지스트리 전체는 29 컬럼
- [x] **F-HS-1 → 모델 `05` E5 기록 완료 2026-09-09.** `20260907_model_experiment/02` §1.5에 11 컬럼·마트·기대 부호·검정 horizon·최대 `|IC|`, `05` §1·§2 E5에 run 12(4 h × 시드 3)·선택 규칙. `fin_interest_coverage`는 경고 둘을 같이 넘겼다 — `fin_operating_profitability`와 ρ=0.87(분자 공유), `operating_income` fallback 0.9121
      — 기록하면서 **E5 선행 조건 하나가 드러났다**: E0~E4 공통 snapshot 2026-08-23의 `feat_relation_stat`·`feat_fin_risk`는 **v1 빌드**이고(`sql_hash` 다름, `analysis_config_hash=None`) `stock_metric_vintage_fact`도 F-5.0 이전이다. E5는 08-23에서 `stock_metric_vintage_fact`→`fin_quarterly_metric_vintage`→`feat_fin_risk`, `dim_peer_monthly`→`feat_relation_stat`를 재빌드한 뒤 돈다. FS0~FS2는 안 바뀐다(`feat_fin_scan_daily`가 F-5.0의 4 metric을 읽지 않고 `sql_hash`가 불변이라 캐시 유지)
      — h 배치 규칙을 같이 박았다: 채택 config가 FS1h면 FS3도 검정 horizon 밖 h에서 뺀다. 결과를 보고 컬럼을 고르는 것을 막는다
- [ ] **F-HS-C2** Phase C 2라운드(F-8.6)
- [ ] **F-HS-2** F-3·F-5·F-7 family(D-F1·PoC 결과 뒤)

---

## 3. 규모 추정

| 항목 | 값 | 근거 |
|---|---|---|
| F-1 월 호출 | **약 20분** (3.27 req/s 실측, 2026-09-08) | N2-7b 1,301초와 같은 급. 0.2s × 3,959 = 13분은 응답 지연을 뺀 값이라 틀렸다 |
| F-1 마트 | **2.3초** (7.05M행) | 2026-09-09 실측. 월 관측이 7,918건뿐이라 ASOF 조인 하나로 끝난다 |
| F-2 peer 계산 | 월 2,800² 상관 × 130개월, 수 분 | numpy |
| F-2 마트 | 7M행 × peer 20 조인 → 월별 파티션 | DuckDB |
| F-4 마트 | `feat_fin_scan_daily`(7.2M행)와 같은 규모 | 같은 vintage 입력 |
| F-6 백필 | **실측 22,700 호출 / 2시간 20분 / 오류 0**(2026-09-10 02:24 종결). 약 2.7 req/s, 청크당 183초. 문서 +15,665, no_data 7,036(31%) | F-6.2 결과 §1·§3 |
| F-6 디스크 | **+23 GB** (`dart_xbrl_fact_raw` 136 → 159 GB, DB 221 → 244 GB) | 추정 32 GB보다 적다 — 31%가 no_data였다 |
| F-5 vintage 측정 | in-memory 3,301,030행, 수 분 | canonical 마트를 안 건드리는 경로 |
| F-7 백필 | PoC 뒤. N6급이면 8만 호출 | N6 83,700 |
| F-8 백필 | 시리즈 10 × 12년 일별, 수 분 | ECOS/FRED |
| ~~S-1 잔여~~ | **실측 131,292 요청 / 5시간 45분 / 저장 0행**(2026-09-08 종결). 옛 추정 "1,300 법인 ≈ 4.3만 호출"은 대상이 866이고 보고서·fs_div 축을 빼먹어 둘 다 틀렸다 | F-9.3 결과 §1~2 |
| F-HS-1 실행 | **A0 8분 + A 105분 + B 378분 + AB 1초 미만 ≈ 8.7시간** (마트 재빌드 34분 별도) | 2026-09-09 실측. 08-23은 A 62분 + B 142분 ≈ 3.4시간이었다 — 옛 표의 "A 60분 + B 45분"은 B가 틀렸다 |
| F-HS 결합 permutation | replicate당 **148초** (08-23은 37초) | 각 replicate가 결합 모집단 전체를 다시 스캔한다. `m_ab` 177 → 231이라 4배. family를 더 늘리면 선형 이상으로 늘어난다 |
| F-HS Phase A permutation | 48 → **89.5분** | 가설이 늘어서가 아니다. seed가 `config_hash`를 받아 null을 새로 뽑는다(`real_scan`은 7.6 → 7.7분으로 동일) |

---

## 4. DoD (패키지 공통)

- 마트: `FORMULA_VERSION` 명시, PIT 규칙·유효 시작일 docstring, 커버리지·분포 리포트, 단위 테스트(PIT·NULL·정의), 기존 마트 SQL 불변(A0 해시).
- raw: `06` §6 체크리스트 전부.
- 사전등록: 부호·horizon·fdr_family·스캔 경로가 결과를 보기 전에 YAML에 고정, hash 기록.
- 문서: 이 디렉터리 `results/`에 PoC·측정·실행 기록. 상태표(§0) 갱신.

---

## 5. 개정 이력

| 날짜 | 내용 |
|---|---|
| 2026-09-07 | 최초 작성. F-1~F-9·F-HS 정의, 결정 D-F1~D-F5 제시 |
| 2026-09-07 | **D-F1~D-F5 권고안 채택 확정.** D-F1은 판정 규칙(3개월 변경률, 문턱 0.3%)이 확정된 것이고 판정 자체는 2026-12. `elestock` 시작, S-1 잔여 실행, DS005 6종 PoC, 3단계는 2단계 국면 통과 뒤 |
| 2026-09-08 | F-2·F-4 마트 완성, F-1 prod 첫 스냅샷, F-9.2 휴장일 CSV 201행, F-9.3 S-1 잔여 **종결**(생존편향은 수집으로 안 닫힌다 — 영구 한계로 문구 변경), F-5.0 XBRL fallback |
| 2026-09-09 | F-HS-1 실행·FS3 인계. F-1.5·F-1.7, F-5.1~F-5.4, F-9.9~F-9.11, F-6.1. **v0.13.0 릴리즈·배포**(F-9.2 prod 반영 + K-5) |
| 2026-09-09 | **사전등록 규칙 2건 정정, 둘 다 보수적 방향.** (a) D-F1 평균에서 seed 쌍 제외 — 빠지는 쌍이 가장 낮아 평균이 **올라간다**(통과가 쉬워지는 방향이 아니다), 판정 시점 2026-12은 불변. (b) F-9.11은 규칙을 **안 바꾸고** MC 밴드 표시만 추가 |
| 2026-09-10 | **F-6.2 백필 완주**(02:24). `fin_sue` 산출 가능 분기 33,141 → **75,293**(×2.27). 남은 것은 F-6.3 재빌드·재판정 |
| 2026-09-10 | 추정 오류 3건을 실측이 정정: F-6.1의 "키 2개·2일"(실제 **키 9개**), "slice ledger 사용 여부 결정 필요"(**이미 코드에 있었다**), F-5.4 이전 `metric_rules` 등록만으로 충분하다는 가정(**`INSTANT_METRICS` 등록이 별도로 필요**) |
