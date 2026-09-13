# 00. 추가 피쳐·수집 계획 — 개요

- 작성일: 2026-09-07
- 목표: 모델 실험(`docs/dev/20260907_model_experiment/`)이 쓸 **새 피쳐 축**을 만든다. 우선은 사용자가 요구한 **종목 간 관계 피쳐**, 그 선행 조건인 **PIT 업종 분류**, 그리고 새 수집 없이 canonical 지표만으로 되는 **재무위험·생애주기·상태 전이**, `fin_sue` 표본을 살리는 **XBRL 백필**, 국면 변수를 넓히는 **매크로 2·3단계**다.
- 입력 문서: `11_feature_taxonomy.md`(빈 축 진단과 우선순위), `02_data_expansion_plan/`(N1~N8 수집 이력, K 묶음, S 묶음), `20260829_macro_features/00_survey/02`(매크로 단계표), `20260830_feature_summary/08` §3(공백 우선순위).
- 이 스트림은 모델 실험과 **병행**한다. 접점은 §5 하나다.

---

## 1. 한 장 요약

| 작업 패키지 | 내용 | 새 수집 | 모델에 넘기는 것 | 문서 |
|---|---|---|---|---|
| **F-1** 업종 PIT | `induty_code` 버저닝(월별 스냅샷) + 업종 마트. 과거는 현재 코드 소급(진단 전용) | OpenDART 재호출(월 3,959건) | `dim_industry_pit_daily` | `01` |
| **F-2** 통계적 관계 피쳐 | 수익률 상관 기반 peer 집합(월별 재추정) → peer 모멘텀·잔차·분산·대형주 리드래그 | **없음** | `feat_relation_stat` | `02` |
| **F-3** 업종 관계 피쳐 | 업종 모멘텀·업종 순매수 쏠림·업종 중립 variant·업종 더미 | 없음(F-1 선행) | `feat_relation_ind`, `_ind` 마트 | `02` |
| **F-4** 재무위험·생애주기·전이 | 레버리지 4·생애주기 2·전이 3 family, strict PIT vintage | **없음** | `feat_fin_risk` | `03` |
| **F-5** `metric_rules` 확장 | 유동자산·유동부채·차입금·R&D 매핑 → Piotroski·Altman·무형 조정 B/M | 없음(매핑) | `feat_fin_risk` 확장 | `03` |
| **F-6** `fin_sue` XBRL 백필 | 8분기 EPS 이력에 필요한 receipt-targeted XBRL 재수집 | OpenDART(규모는 측정 뒤) | `fin_sue` 표본 | `04` |
| **F-7** 구조화 이벤트 | DS005 주요사항보고서(유상증자·자사주·합병 결정 등) 신규 raw + 이벤트 플래그. `elestock` 정기 수집 결정 | **있음** | `feat_event_decision` | `04` |
| **F-8** 매크로 2·3단계 | ECOS 회사채·기준금리·ESI, FRED 신용·달러·breakeven·EPU, 관세청 수출, SOX → 신용 국면 | 시리즈 정의 + 어댑터 2 | `common_feature_daily_fact` 확장, 국면 `credit_wide` | `05` |
| **F-9** 인프라·PIT 통일 | `fin_pit` strict vintage 재작성, `holidays_krx.csv` 2014~2023 보강, 상폐 법인 S-1 잔여·S-2, PIT 유니버스 마트, Cronicle 이벤트 | 일부(상폐 백필) | FS0', 격자 정합 | `06` |
| **F-HS** Horizon Scan 사전등록 | 위 새 family를 **새 config 하나**로 묶어 A→B→AB(→C) 검정 | — | `screen_pass` 목록 = FS3 | `07` |

**새 수집이 전혀 없는 것(F-2·F-4)이 먼저다.** 그 둘만으로 관계 축(C5)과 재무위험·전이 축(C2·T3)이 처음 생긴다. F-1은 앞으로만 쌓이는 이력이라 지금 시작해야 시간이 벌린다.

---

## 2. 왜 이 순서인가

`11_feature_taxonomy.md` §10의 우선순위(업종 → 생애주기·레버리지 → 전이·exposure → 관계 → 공시·내부자 → `metric_rules` → 계절성)를 이번 목표(관계 피쳐, 모델 투입)에 맞춰 다시 배열했다.

1. **관계 피쳐는 업종 없이도 절반은 된다.** 통계적 peer(F-2)는 수익률 상관만 쓰므로 PIT가 구조적으로 보장되고 지금 만들 수 있다. 개별 모멘텀이 한국에서 반대 부호로 나온 만큼 집계(peer·업종) 수준 모멘텀의 기대값이 오히려 크다(`11` §7.1).
2. **업종은 과거를 살릴 수 없다.** N4가 닫히고(KRX 지수 구성종목 미제공, KSIC↔KRX crosswalk 36%만 1:1) 남은 길은 `induty_code`를 지금부터 버저닝하는 것뿐이다. 과거 구간은 현재 코드 소급이라 PIT가 아니다. 그래서 F-1은 (a) 지금 시작해 이력을 쌓고, (b) 소급 구간의 누수 크기를 **월별 변경률로 측정**해 모델 사용 가능 여부를 정한다(`01` §4).
3. **재무위험·생애주기는 비용 0인데 축이 비어 있다.** Barra의 leverage, Dickinson의 생애주기가 canonical 29 metric으로 바로 된다. 단 Decline·부실 축은 상폐 법인 커버리지(S-1 잔여·S-2)가 선행이다.
4. **`fin_sue`는 "못 잰 것"이다.** 서프라이즈 축(T2)이 한 번도 측정되지 않았다. 백필 대상만 고르면 기존 사전등록으로 다시 돈다.
5. **매크로 2단계는 국면 변수 확장이다.** Phase C에서 유동성·VIX 수준·시장 상태 국면이 통했다. 신용 스프레드 국면이 그 다음 후보이고, F-4의 레버리지 피쳐와 짝(레버리지 × 신용 국면)이 된다.

---

## 3. 의존 관계

```
F-2 통계적 peer ─────────────────────────────┐
F-4 재무위험·생애주기 ──┬── S-1 잔여·S-2(F-9) ──┤
F-5 metric_rules 확장 ──┘                      ├──► F-HS 새 config 사전등록 → A→B→AB(→C) ──► FS3 (모델 E5)
F-1 업종 버저당 ──► 3~6개월 변경률 측정 ──► F-3 업종 관계 피쳐 ──┤
F-6 SUE 백필 ──► 마트 재빌드 ──► 기존 사전등록으로 재검정 ──────┤
F-7 DS005 PoC ──► raw 수집 ──► 이벤트 플래그 ────────────────────┘
F-8 매크로 2단계 ──► 시리즈 백필·readiness ──► 국면 credit_wide ──► Phase C 2라운드
F-9 fin_pit strict 재작성 ──────────────────────────────────────► 모델 E3-b (FS0')
F-9 holidays csv ──► golden 갱신 ──► fact 격자 정합(모든 국면 계산에 이득)
```

- **F-HS는 묶어서 한 번.** A→B→AB가 약 1시간 44분, Phase C까지 약 2시간이다. family가 생길 때마다 돌리지 않고 F-2·F-4(·F-5·F-7)가 모이면 한 config로 사전등록한다. F-3은 F-1의 변경률 판정 뒤 별도 config다.
- 모델 스트림(E0~E4)은 이 스트림과 무관하게 끝난다. FS3는 E5로만 들어간다.

---

## 4. 규율 — 이 저장소의 수집·검정 규칙

1. **raw는 Postgres, 파생은 DuckDB 마트.** 새 raw 테이블은 ports & adapters로 수집기를 만들고(`01_implementation_checklist.md` §2), 피쳐는 `research/etl/features/*.py` 마트로만 만든다. 피쳐를 Postgres에 넣지 않는다.
2. **7곳 등록.** 새 raw 테이블은 DDL·`remote_sync`·profiling catalog·`export_tables.toml`·`raw-parquet-export-all.sh`·`research/etl/config.py RAW_TABLES`·`docs/database.md`에 등록한다(`01_implementation_checklist.md` §1). 빠지면 lake에 안 온다.
3. **PIT.** 노출 시점은 접수일 다음 KRX 세션(공시), 관측일 다음 세션(국내 시계열), NY t−1(해외), strict vintage(재무)다. 현재 값의 과거 소급은 **진단 전용**으로만 쓰고 마트 컬럼에 `is_backcast` 같은 표지를 남긴다.
4. **KRX 스크래핑 금지.** KRX 데이터는 Open API(승인 16 endpoint), KIS, 사람이 내려받는 화면 CSV로만 받는다(`docs/operations.md` "KRX 접근 제한"). pykrx·MDC 자동 요청은 `ALLOW_KRX_SCRAPING=false` 그대로 둔다.
5. **OpenDART 키 예산.** 다중 키 rotation, 일 한도 도달 시 exit 75 재개. 대규모 백필은 slice ledger(`collection_slice_state`)를 쓴다.
6. **새 family는 새 config.** 기존 `config_hash`(`ab0de634`·`889c3e83`·`236d0d35`)의 YAML을 고치지 않는다. 부호·창·업종 깊이는 결과를 보기 전에 고정한다. 여러 정의를 만들어 좋은 것을 고르지 않는다.
7. **holdout(2025-08-01~)은 열지 않는다.** 이 스트림의 검정도 같은 경계다.
8. **수집과 검정을 섞지 않는다.** 수집이 끝났다고 바로 검정하지 않고 F-HS로 묶는다. 재실행 비용을 기억한다.

---

## 5. 모델 스트림과의 접점

| 항목 | 규칙 |
|---|---|
| 마트 계약 | grain `(trade_date, ticker, market)`, valid session만, 컬럼 접두 `rel_`·`fin_`·`ev_`·`ind_`. docstring에 PIT 규칙·유효 시작일·`FORMULA_VERSION` |
| 넘기는 것 | F-HS `screen_pass` 컬럼 목록 + 기대·관측 부호 + 사전등록 horizon → 모델 `features.py` FS3 |
| 넘기지 않는 것 | 소급 업종으로 만든 `_ind` variant(진단 전용), 검정 안 한 컬럼 |
| 시점 | FS3 목록이 확정되면 모델 스트림 `05_experiment_matrix.md` E5에 적는다. E0~E4는 기다리지 않는다 |
| F-9 FS0' | `feat_fin_ratios_pit`(strict vintage 10 비율)이 준비되면 모델 E3-b에 통보 |

---

## 6. 결정 — 확정 (2026-09-07)

**2026-09-07 사용자가 아래 다섯 항목을 권고안 그대로 채택했다.** D-F1은 "측정 뒤 문턱 0.3%로 판정한다"는 규칙 자체가 확정된 것이고, 판정은 2026-12에 한다.

| id | 결정 | **확정** |
|---|---|---|
| **D-F1** | 소급 업종(현재 KSIC를 과거에 적용)을 모델 피쳐에 쓸 것인가 | **변경률 측정 뒤 결정.** 첫 3개월 스냅샷에서 월별 `induty_code` 변경 법인 비율이 0.3% 미만이면 소급 누수를 한계로 명시하고 F-3을 모델 후보로 허용. 넘으면 정규화·진단 전용 (`01` §4) |
| **D-F2** | `elestock`(임원·주요주주 소유상황) 정기 수집을 지금 시작할 것인가 | **시작.** 응답이 최근 2년 롤링이라 한 달 늦으면 한 달이 영구히 사라진다. 월 2,657건으로 비용이 작다. 피쳐는 나중에 (`04` §3) |
| **D-F3** | DS005 주요사항보고서 수집 범위(이벤트 종류) | PoC로 endpoint별 응답·볼륨을 재고 **유상증자·무상증자·자기주식 취득/처분·합병·분할·전환사채** 6종으로 시작 (`04` §2) |
| **D-F4** | 상폐 법인 S-1 잔여(financials·share-info·xbrl 약 1,300 법인) 백필을 지금 돌릴 것인가 | **돌린다.** F-4의 Decline·부실 축과 F-9 PIT 유니버스의 선행이다. OpenDART만 쓴다 (`06` §3) |
| **D-F5** | 관세청 수출·SOX 어댑터(3단계)를 이번 범위에 넣을 것인가 | **2단계(시리즈 정의만) 먼저, 3단계는 2단계 국면이 Phase C에서 통하면** (`05` §4) |

---

## 7. 문서 지도

| 파일 | 내용 |
|---|---|
| **`00_overview.md`** | 이 문서 |
| [`01_industry_pit.md`](01_industry_pit.md) | 업종 PIT 세 층(버저닝·전방 스냅샷·통계적 peer), 소급 정책, 변경률 측정, 마트·등록·이벤트 |
| [`02_relationship_features.md`](02_relationship_features.md) | 통계적 peer 피쳐(지금)·업종 관계 피쳐(F-1 뒤) 정의, PIT, 계산 설계, 사전등록 초안 |
| [`03_financial_risk_lifecycle_transition.md`](03_financial_risk_lifecycle_transition.md) | 레버리지·생애주기·전이 family 정의, strict vintage, `metric_rules` 확장, 생존편향 선행 조건 |
| [`04_sue_backfill_and_events.md`](04_sue_backfill_and_events.md) | `fin_sue` 백필 대상 선정·실행, DS005 구조화 이벤트 PoC·수집, `elestock` 결정 |
| [`05_macro_stage2_3.md`](05_macro_stage2_3.md) | ECOS·FRED 시리즈 추가, 신용 국면, 관세청·SOX 어댑터, Phase C 2라운드 후보 |
| [`06_infra_pit_unification.md`](06_infra_pit_unification.md) | `fin_pit` strict 재작성, 휴장일 CSV 보강, 상폐 커버리지, PIT 유니버스 마트, Cronicle, 등록 체크리스트 |
| [`07_work_breakdown.md`](07_work_breakdown.md) | F-1~F-9·F-HS 체크리스트, 의존·순서·규모·DoD, 상태표 |
