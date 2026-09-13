# 01. 업종 분류 PIT — F-1

- 작성일: 2026-09-07
- 문제: 업종 코드가 **현재 시점 값**으로만 있다(`dart_corp_master.induty_code`, N2 2026-08-15 전량 수집, 결측 0). 과거 업종 변경 이력이 없어 PIT가 아니고, 업종 중립화·업종 관계 피쳐를 alpha로 쓸 수 없다. KRX 지수 구성종목(N4)은 Open API에 없고, KRX 업종분류 ↔ KSIC crosswalk는 종목 36.4%만 1:1이라 실패로 판정됐다(`poc/n7_n4_alternatives.md` §2.4).
- 결론: **과거는 살릴 수 없다.** 앞으로의 이력을 지금부터 쌓고(§2), 과거 구간은 소급값으로 두되 누수 크기를 재서(§4) 쓸 수 있는 범위를 정한다. 통계적 peer(§5, `02`)가 PIT 관계 축의 즉시 대안이다.

---

## 1. 세 층 구조

| 층 | 무엇 | PIT | 이력 | 비용 | 상태 |
|---|---|---|---|---|---|
| **L1 `induty_code` 버저닝** | DART 기업개황 `induty_code`(KSIC)를 월 1회 다시 받아 스냅샷으로 쌓는다 | ✅ 스냅샷 시점부터 | **2026-09부터** | OpenDART 월 3,959호출(약 13분) | 이 문서 §2 |
| L2 전방 스냅샷(보조) | KIS 종목 기본정보의 업종 필드, KRX 업종분류 현황 CSV(사람 다운로드) | ✅ 스냅샷 시점부터 | 2026-09부터 | KIS 월 2,800호출(1 req/s, 약 47분) / 사람 월 1회 | PoC 필요 §3 |
| **L3 통계적 peer** | 수익률 상관으로 만든 peer 집합 | ✅ 구조적 | **2015~** | 계산만 | `02` §1 |
| L0 소급 | 현재 `induty_code`를 과거 전체에 적용 | ✗ | 2014~ | 0 | 있음(`feat_fin_scan_daily_ind`), **진단 전용** |

L1이 본선이다. L2는 L1과 다른 분류 체계(KRX 업종·지수 업종)를 준다는 점에서 **대조·보완**이지 대체가 아니다. L3는 업종과 무관하게 지금 되는 관계 축이다.

---

## 2. L1 — `induty_code` 버저닝

### 2.1 원천과 명령

- API: OpenDART `company.json`(기업개황). 이미 `dart sync-corp-profile`이 받는다. 옵션 `--refresh`(이미 프로필이 있는 법인도 다시 받는다), `--universe-scope historical`(ticker를 가진 적 있는 3,959 법인), `--rate-limit-seconds`(기본 0.2).
- 현재 동작: `dart_corp_master`에 **upsert**하므로 덮어써서 이력이 없다(`profile_fetched_at`이 skip 키).
- 바꾸는 것: 수집 결과를 **이력 테이블에도 append**한다. `dart_corp_master`는 그대로 최신값을 유지한다(다른 소비자가 있다).

### 2.2 스키마 — `dart_corp_profile_history` (raw, Postgres)

| 컬럼 | 타입 | 뜻 |
|---|---|---|
| `corp_code` | TEXT | PK 일부 |
| `observed_month` | DATE (그 달 1일) | PK 일부. skip-if-present 키 |
| `observed_at` | TIMESTAMPTZ | 실제 응답 시각 |
| `ticker`, `corp_cls`, `induty_code`, `est_dt`, `acc_mt`, `corp_name`, `stock_name` | | 응답 그대로 |
| `profile_raw` | JSONB | 응답 원문 |
| `run_id` | | `ingestion_runs` 참조 |

- **한 법인·한 달에 한 행.** 같은 달에 두 번 돌면 skip. `ON CONFLICT (corp_code, observed_month) DO NOTHING`.
- 첫 스냅샷(2026-09)은 `dart_corp_master`의 현재 행을 `observed_month = 2026-09-01`, `observed_at = profile_fetched_at`으로 **복사해 시드**한다. 그 전 이력은 없다는 사실을 `is_seed=true`로 남긴다.
- 7곳 등록 + `docs/database.md`. exporter 전략은 `snapshot`(작다, 월 3,959행).

### 2.3 마트 — `dim_industry_pit_daily`

`research/etl/features/industry_pit.py`. grain `(trade_date, ticker, market)`.

| 컬럼 | 정의 |
|---|---|
| `ind_ksic_code` | `date(observed_at)` **다음** 유효 세션부터 노출되는 마지막 `induty_code`. ASOF join |
| `ind_group` | `definitions/industry_groups.py`의 fold-up(KSIC 2자리 접두 → 그룹 크기 20 미만이면 섹션 → 그래도 미달이면 `OTHER`). `observed_month`마다 다시 푼다 |
| `ind_observed_at` | 그 값이 관측된 시각 |
| `ind_is_backcast` | **t가 첫 스냅샷 노출 세션보다 앞이면 true** — 이 행은 현재 코드 소급이다. 관측이 아예 없는 종목은 `NULL` |
| `ind_changed_recent_252` | 지난 252세션 안에 코드가 바뀐 적이 있으면 true. 감지 불가 구간은 `NULL` |
| `ind_group_changed_recent_252` | 같은 것을 `ind_group` 기준으로 (F-1.5에서 추가) |

**구현 정정 (F-1.5, 2026-09-09).** 위 표의 노출 규칙은 처음 `≤ t`(관측일 당일부터)로 적었는데
그대로 두면 안전하지 않다. seed 행은 23:10 KST, 2026-09 스냅샷은 00:14 KST에 관측됐다 —
같은 날 노출은 앞의 경우 look-ahead고 뒤는 아닌데 행에는 그 구분이 없다. `00_overview.md`
§4.3의 국내 관측 규칙(관측일 다음 세션)이 두 경우 다 맞으므로 그쪽으로 고쳤다. 비용은
값이 1세션 늦게 보이는 것뿐이다.

`ind_changed_recent_252`는 **seed 전이를 세지 않는다.** seed 행은 `dart_corp_master`의 현재
상태 복사본이라 변경 시점을 잡을 수 없다(§2.2의 `is_seed` 설명과 같은 이유). 그래서 두
비-seed 관측이 쌓이기 전에는 `FALSE`가 아니라 `NULL`이고, 처음 non-NULL이 될 수 있는 달은
2026-11이다.

- 그룹 fold-up 규칙은 **N2-6·V2에서 고정된 것**(`MIN_GROUP_SIZE=20`, 2자리 → 섹션 → `OTHER`)을 그대로 쓴다. 깊이를 새로 고르지 않는다.
- 소급 구간(`ind_is_backcast=true`)의 값은 마트에 두되 **소비자가 표지를 보고 결정**한다. Horizon Scan family와 모델 FS3는 §4의 판정 전까지 backcast 행을 NULL로 취급한다.

### 2.4 운영

- Cronicle 월 1회(매월 1일 05:30 KST 예시 크론이 있다. prod 이벤트 존재 여부 확인 후 없으면 등록): `dart sync-corp-profile --universe-scope historical --refresh` + 이력 append. `opendart` lock 공유(04:00 체인 뒤).
- 규모: 3,959호출 × 0.2s ≈ 13분. 일 한도(키당 10,000)에 여유. 상폐 법인도 받는다(`corp_cls` 변화 = 상폐 diff 검증 수단, N2 V3).
- freshness: `dart_corp_profile_history`의 최신 `observed_month`가 이번 달인지 월 단위 게이트를 `ops freshness-report`에 추가(월 45 영업일 예산).

---

## 3. L2 — 전방 스냅샷 (PoC 뒤 결정)

| 원천 | 필드 | 확인할 것 | 가치 |
|---|---|---|---|
| **KIS 종목 기본정보**(`search-stock-info` 계열) | 표준산업분류 코드·명, 지수업종 대/중/소분류명 (문서 미확인 — **PoC에서 응답 필드를 실측**) | 필드 존재, 전종목 응답, 1 req/s 한도에서 월 2,800호출 가능 여부, 분류 체계가 KRX 업종인지 | KRX 업종 축을 자동으로 받는 유일한 길일 수 있다. KIS 약관은 비공개 개인 연구 범위 안(K-6c) |
| KRX 업종분류 현황 CSV(`MDC0201020506`) | 종목코드·업종명·시장구분 등 8열 | 사람이 월 1회 다운로드. **파일명 날짜 ≠ 기준일**(2026-08-18 파일이 08-14 종가) → 기준일은 Open API 종가 대조로 확정 | 검증·대조용. 자동화 불가 |

- L2는 L1의 대체가 아니다(분류 체계가 다르다). 용도는 (a) KRX 업종 기반 관계 피쳐(`02` §2)에 **KRX 분류가 더 맞는 경우**(지수 업종 모멘텀), (b) L1 변경의 교차 확인.
- PoC 산출물: `poc/kis_stock_info_industry_2026MM.md` — 응답 필드, 커버리지, 분류 목록, KSIC 2자리와의 교차표.
- PoC가 통과하면 `kis_stock_industry_snapshot` raw(월별)로 등록. 통과하지 않으면 L2는 사람 CSV만 남기고 자동화하지 않는다.

---

## 4. 소급 누수 측정과 사용 정책 — D-F1 (판정 규칙 확정 2026-09-07, 판정은 2026-12)

### 4.1 무엇을 재나

"업종이 실제로 바뀐 기업이 몇 개인가"(N4 목적 A)는 아직 아무도 못 잰 값이다. L1이 그 답을 **앞으로만** 준다.

| 지표 | 정의 | 언제 |
|---|---|---|
| 월별 변경률 | `observed_month` m에서 m−1과 `induty_code`가 다른 법인 수 / 양쪽에 있는 법인 수 | 매월 |
| 그룹 변경률 | 같은 것을 `ind_group` 기준으로 | 매월 |
| 누적 변경률 | 첫 스냅샷 대비 코드가 한 번이라도 바뀐 법인 비율 | 3·6·12개월 |
| 변경 방향 | 어느 그룹 → 어느 그룹 | 6개월 |

### 4.2 판정 규칙 (결과를 보기 전에 고정)

- **3개월 스냅샷(2026-09·10·11)에서 그룹 변경률 월평균 < 0.3%**(연 약 3.5%)이면: 소급 업종의 누수는 "연 3~4% 법인의 그룹이 과거에 잘못 붙어 있다"는 크기다. **F-3 업종 관계 피쳐를 소급 업종으로 만들어 Horizon Scan 후보와 모델 FS3에 올리는 것을 허용**하되, 카드에 "업종은 2026-09 이전 소급(연 x% 누수)"을 적고 `ind_is_backcast` 표지를 피쳐로 같이 넘긴다.
- **0.3% 이상**이면: 소급 업종은 정규화·진단 전용을 유지한다. 업종 관계 피쳐는 L1 이력이 12개월 이상 쌓인 뒤 PIT 구간에서만 검정한다(2027-09 이후).
- 판정은 3개월 뒤 한 번, 12개월 뒤 한 번 다시 본다. 문턱 0.3%는 여기서 고정하고 바꾸지 않는다.

**평균에 넣는 쌍 정정 (F-1.7, 2026-09-09).** 위에 적은 세 스냅샷(2026-09·10·11)의 첫 쌍은
`2026-08(seed) → 2026-09`다. seed 쌍은 변경 시점을 잡을 수 없으므로(§2.3 정정) 평균에서
뺀다. 결과를 보고 고친 것이 아니라는 근거는 둘이다. 사유가 데이터 속성이고, 빠지는 쌍이
0.0253%로 관측된 것 중 **가장 낮아** 제외하면 평균이 올라간다 — 문턱 통과가 쉬워지는
방향이 아니다. 필요한 것은 비-seed 쌍 셋(09→10, 10→11, 11→12)이고 그것은 **2026-12-01
스냅샷**이면 모이므로 판정 시점 2026-12는 그대로다. 문턱 0.3%도 그대로다.

### 4.3 왜 이 문턱인가

- N2 V3에서 `corp_cls='E'`(상폐)와 `status='DELISTED'`가 정확히 일치했듯 DART 기업개황은 갱신이 규칙적이다. 업종 변경은 사업 전환·합병 뒤 정기보고서 시점에 반영되므로 연 수 % 이하일 가능성이 크다. 그렇다면 소급 누수는 `fin_gross_profitability`의 역산값 94.4%나 `mcap_krx_log`의 tradable 유지율 0.56 같은 기존 한계와 같은 급이다.
- 반대로 변경률이 크면 "사업 전환 기업"에서 미래 정보가 새고, 그 기업들이 바로 관계 피쳐가 가장 크게 움직이는 집단이라 편향이 결과를 좌우한다.

---

## 5. L3 — 통계적 peer

정의·계산은 `02_relationship_features.md` §1. 여기서는 L1과의 관계만 적는다.

- 통계적 peer는 **업종을 대체하지 않는다.** 상관 기반 집단은 규모·스타일·시장 공통 요인으로도 묶인다. 그래서 `02`는 시장 잔차 수익률로 상관을 잰다.
- L1 이력이 쌓이면 **peer 집합과 KSIC 그룹의 일치율**(같은 peer 집합 안의 KSIC 2자리 동일 비율)을 진단으로 낸다. 일치율이 높으면 통계적 peer가 업종의 PIT 대용으로 쓸 만하다는 뜻이고, 낮으면 두 축이 다른 정보를 본다는 뜻이다. 둘 다 결과다.

---

## 6. 산출물과 완료 기준

| 산출물 | 위치 |
|---|---|
| DDL `dart_corp_profile_history` + 등록 7곳 | `sql/postgres_ddl.sql`, `remote_sync.py`, `profiling/catalog.py`, `export_tables.toml`, `raw-parquet-export-all.sh`, `research/etl/config.py`, `docs/database.md` |
| 서비스 변경 | `service/sync_dart_corp_profile.py`(가정 이름) — append 경로, 시드 명령 `dart seed-corp-profile-history` |
| 마트 | `research/etl/features/industry_pit.py` → `dim_industry_pit_daily` |
| 변경률 리포트 | `research/analysis/industry_change_report.py` → `docs/dev/20260907_additional_feature/results/industry_change_YYYYMM.md` |
| Cronicle | 월 1회 이벤트(`sdc_monthly_corp_profile_history`) |
| KIS PoC | `poc/kis_stock_info_industry.md` |

완료 기준: (1) 첫 스냅샷 시드 + 2026-09·10 두 달 append 확인, (2) 마트가 `ind_is_backcast`를 정확히 가르는지 테스트, (3) 변경률 리포트 첫 회, (4) `db sync-remote`·export·lake 경로 확인, (5) 유닛 테스트·ruff 통과.

---

## 7. 하지 않는 것

- KRX MDC 자동 요청으로 업종 받기(약관 위반).
- 과거 업종 복원(사업보고서 텍스트 파싱 등). 비용 대비 근거가 없다.
- 업종 그룹 깊이 재선택. `industry_groups.py`의 규칙을 그대로 쓴다.
- 소급 업종을 표지 없이 마트에 넣기.
