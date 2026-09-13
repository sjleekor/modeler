# prod 실행 준비 — F-1.4(업종 첫 스냅샷) · F-9.3(S-1 잔여 백필)

- 작성: 2026-09-07
- 상태(**2026-09-09 갱신**): **둘 다 끝났다.** 이 문서는 실행 전 준비 기록으로 남긴다.
  - F-1(§1) 2026-09-08 실행 완료 → [`f1_4_first_snapshot_20260908.md`](f1_4_first_snapshot_20260908.md)
  - F-9.3(§2) 2026-09-08 실행·**종결**(더 받을 것이 없다) → [`f9_3_s1_remainder_20260908.md`](f9_3_s1_remainder_20260908.md)
- 실측은 모두 prod `krx_data`(sj2) 읽기 전용 조회다.

---

## 0. 두 작업 공통 — prod에 코드가 먼저 가야 한다

`dart_corp_profile_history`는 **prod에 아직 없다**(확인: `information_schema.tables` 0행).
F-1.4는 물론이고 `dart sync-corp-profile`을 그냥 돌려도 append 단계에서 실패한다.
그래서 순서가 이렇게 된다.

1. 릴리즈 — `sdc-release` 스킬(버전 범프 → lint+unit → 커밋 → `vX.Y.Z` 태그 → GitHub 이미지 빌드 → sj2 compose 이미지 태그 갱신)
2. `deploy/prod/bin/db-init.sh` — `dart_corp_profile_history` + 인덱스 3개 생성
3. 그 다음에 아래 F-1.4 / F-9.3

수동으로 하지 않는다(`01_implementation_checklist.md` §5.1).

---

## 1. F-1.4 — 업종 이력 첫 스냅샷

### 1.1 실측 — prod의 현재 상태

| 항목 | 값 |
|---|---|
| ticker를 가진 적 있는 법인 | 3,959 |
| `profile_fetched_at`이 있는 법인 | 3,959 (100%) |
| `induty_code`가 있는 법인 | 3,959 (100%) |
| `profile_fetched_at` 범위 | **2026-08-15 하루** (min = max) |
| `dart_corp_profile_history` | 테이블 없음 |
| Cronicle 이벤트 | `emsugdoe907` "SDC Backfill DART Corp Profile (one-time)", `enabled=1`, **`timing=false`(수동 전용)** — 자동으로 돌지 않는다 |

### 1.2 확인이 필요한 결정 — 시드를 몇 월로 찍는가

`01` §2.2는 첫 스냅샷을 `observed_month = 2026-09-01`로 시드하라고 적었다. 그런데 prod의
값이 **실제로 관측된 시각은 2026-08-15**이다. 시드 행의 `observed_at`은 그 값을 그대로
가져가므로, 월만 2026-09로 찍으면 "9월에 관측한 값"이라는 표시와 실제 관측 시각이 어긋난다.

| 안 | 시드 월 | 첫 *수집* 스냅샷 | OpenDART 호출 | D-F1 판정(3개월) |
|---|---|---|---|---|
| **A (문서 그대로)** | 2026-09 | 2026-10 | 지금 0건 | 2027-01 |
| **B (권고)** | 2026-08 | **2026-09 = 지금** | 3,959건 ≈ 13분 | **2026-12** (계획대로) |

B를 권한다. 이유 셋.

1. `observed_at`이 2026-08-15이므로 2026-08이 사실과 맞다.
2. 시드 월과 첫 수집 월이 갈리므로 **첫 변경 구간이 바로 생긴다.** A는 같은 달에 시드와
   수집이 겹쳐 append가 no-op가 되고(같은 달 `DO NOTHING`), 첫 구간이 10월까지 밀린다.
3. D-F1 판정 시점이 계획(2026-12)대로 유지된다. A는 한 달 밀린다.

비용은 3,959 호출 한 번이다 — **실측 19분 27초**(3.27 req/s). 13분 추정은 응답 지연을 뺀 값이라 틀렸다.

### 1.3 명령 — 안 B

```bash
# (1) 시드: 2026-08로. API 호출 없음, 수 초.  [실행됨 2026-09-08 → 3,959행]
ssh whi@sj2-server 'cd /home/whi/apps/sdc && \
  DART_PROFILE_HISTORY_MONTH=2026-08-01 ./bin/dart-seed-corp-profile-history.sh'
# 기대: "Rows inserted: 3959", is_seed=true, observed_at=2026-08-15

# (2) 첫 수집 스냅샷: 2026-09. --force로 skip-if-present를 끈다.  [실행됨 2026-09-08 → 3,959, 19분 27초]
#     opendart lock을 잡으므로 04:00 체인과 겹치지 않는 시간에.
ssh whi@sj2-server 'cd /home/whi/apps/sdc && \
  DART_PROFILE_UNIVERSE_SCOPE=historical DART_PROFILE_FORCE=1 \
  ./bin/dart-sync-corp-profile.sh'
# 기대: Requests attempted 3959 / Rows upserted 3959 / History appended 3959
#       exit 0. exit 75면 키 소진 — 다음 날 같은 명령으로 재개(같은 달이므로
#       이미 들어간 행은 DO NOTHING, 남은 것만 채워진다)
```

**F-1.4 완료 판정** (`07` 원문: "3,959 법인, 오류 0, 행 수 = 법인 수"):

```sql
select observed_month, count(*), count(*) filter (where is_seed) as seeded,
       count(induty_code) as with_industry, min(observed_at), max(observed_at)
from dart_corp_profile_history group by 1 order by 1;
-- 기대: 2026-08-01 → 3959행 전부 is_seed
--       2026-09-01 → 3959행 전부 is_seed=false, observed_at = 실행일
```

첫 변경률(F-1.7)은 이 두 달을 비교해 낼 수 있다. 다만 **시드 월은 관측 구간이 아니다**
(`observed_at`이 8-15 하루에 몰려 있고 그 이전 이력이 없다) — 8월→9월 변경은 "8-15 이후
어느 시점"이고, 진짜 한 달 구간은 9월→10월부터다. 리포트에 그대로 적는다.

### 1.4 함께 정리할 것 (권고, 별건)

`emsugdoe907`은 일회성 백필 이벤트이고 목적을 이미 달성했다(3,959 전량 2026-08-15).
`--force`가 없어 지금 돌려도 전부 skip하고 history에도 아무것도 안 넣는다. 이 저장소
관례대로 삭제 대상이고(F-9.7), 그 자리에 월 1회 이벤트(F-1.6)를 만든다. **둘 다 이번
범위 밖이고 별도 승인이 필요하다.**

---

## 2. F-9.3 — S-1 잔여 백필

> **실행 결과(2026-09-08): 176 유닛 전수 완주, 요청 131,292건, 저장 0행, 커버리지 변화 없음.**
> 남은 공백은 수집 누락이 아니라 OpenDART가 상폐 법인에 재무 엔드포인트 두 개 모두 status 013을
> 돌려주는 구조적 한계다. 생존편향은 수집으로 닫히지 않는다 →
> [`f9_3_s1_remainder_20260908.md`](f9_3_s1_remainder_20260908.md).
> 아래 §2.1~§2.4는 **실행 전 예측**이고, §2.4의 호출 상한 13.7만은 실측 13.1만으로 거의 맞았다.

### 2.1 실측 — 무엇이 빠져 있나

ticker를 가진 적 있는 3,959 법인 기준:

| raw | 있는 법인 | 빠진 법인 |
|---|---|---|
| `dart_financial_statement_raw` | 3,093 | **866** |
| `dart_share_count_raw` | 3,430 | **529** |
| `dart_xbrl_document` | 3,093 | **866** |

빠진 866을 `stock_master` 상태로 쪼개면:

| `stock_master` 상태 | 법인 | 재무 없음 |
|---|---|---|
| 행 자체가 없음 | 920 | **764** |
| `DELISTED` | 431 | 97 |
| `ACTIVE` | 2,624 | 5 |

**중요한 발견.** 빠진 866 중 764는 `stock_master`에 **행이 아예 없는** 법인이다.
`dart_corp_master`에는 ticker가 있는데(corpCode.xml이 상폐 후에도 `stock_code`를 유지한다)
유니버스 sync가 그 종목을 본 적이 없다 — 수집 시작 전에 이미 상폐된 종목들이다.
`--universe-scope historical`은 `dart_corp_master`를 보므로 이 764를 **대상에 포함한다.**
`06` §3이 "약 1,300 법인"이라고 본 것보다 재무 기준으로는 적고(866), 대신 `06` §4의
"상폐 종목 가격"(F-9.4, S-2)에서 복구할 대상은 그 920에 가깝다.

### 2.2 문서 정정

`06` §3은 `--include-delisted` 플래그를 쓰라고 적었다. **그런 플래그는 없다.**
`sync-financials` / `sync-share-info` / `sync-xbrl`이 노출하는 것은
`--universe-scope {current,historical}`이고, `historical`이 `get_dart_corp_master(
active_only=False, include_delisted=True)`로 내려간다(`01_implementation_checklist.md` §1.3).

그리고 **`bin/dart-backfill-all-years.sh`는 `--universe-scope`를 전혀 넘기지 않았다** —
CLI 기본값 `current`로 돌았다. S-1 갭이 생긴 직접적인 이유가 이것이다. 이번 커밋에서
`SDC_DART_BACKFILL_UNIVERSE_SCOPE` 환경변수를 추가했고, **기본값은 `current` 그대로**다
(기존 스케줄 잡이 업그레이드만으로 동작이 바뀌지 않게).

### 2.3 명령

```bash
# 3단계(financials / share-info / xbrl)를 2015~2025 연도별로. filings는 이미 끝났으므로 끈다.
ssh whi@sj2-server 'cd /home/whi/apps/sdc && \
  SDC_DART_BACKFILL_UNIVERSE_SCOPE=historical \
  SDC_DART_BACKFILL_START_YEAR=2015 \
  SDC_DART_BACKFILL_END_YEAR=2025 \
  SDC_DART_BACKFILL_FILINGS=0 \
  nohup ./bin/dart-backfill-all-years.sh > /tmp/s1-remainder-$(date +%F).log 2>&1 &'
```

### 2.4 규모와 운영 주의

- 호출 수 상한: 재무 866 × 11년 × 4보고서 × 2 fs_div ≈ 7.6만, 주식수 529 × 11 × 4 ≈ 2.3만,
  XBRL 866 × 11 × 4 ≈ 3.8만. **상한 합계 약 13.7만.** `06` §3의 4.3만 추정보다 크다
  (그 추정은 법인 × 연 × API 3개만 곱했고 보고서·fs_div 축을 빼먹었다).
  실제로는 대부분이 조기 상폐 법인이라 `no_data` 응답이 많고, skip-if-present와 negative
  cache가 재실행 비용을 줄인다 — 그래도 **며칠 걸린다고 보는 게 맞다.**
- exit 75(키 전량 소진)로 끊기면 같은 명령으로 재개한다. 저장된 raw는 건너뛴다.
- **키 예산 충돌.** `07` §1이 정한 순서는 S-1 잔여 → F-6 → F-7이다. 이 백필이 도는 동안
  04:00 OpenDART 체인이 같은 키를 쓴다. 백필을 `opendart` lock 없이 `nohup`으로 띄우면
  둘이 겹친다 — Cronicle 일회성 이벤트로 만들어 lock을 공유하는 쪽이 안전하다.
  **이 판단은 사용자 확인이 필요하다.**
- 끝난 뒤: 새 snapshot으로 `feat_fin_risk` 재빌드 → §4 상폐 커버리지 재측정 →
  Decline·부실 축 판정 해제(F-4의 보류 해제).

---

## 3. 확인받을 것 정리 — **네 건 모두 2026-09-08에 결정·실행됐다**

| | 물어본 것 | 결정과 결과 |
|---|---|---|
| 1 | F-1.4를 안 B(시드 2026-08 + 지금 2026-09 수집)로 갈지 | **안 B.** 시드 3,959행 + 첫 수집 스냅샷 3,959행(19분 27초, 오류 0) |
| 2 | 그 앞에 릴리즈 + `db init`을 돌려도 되는지 | **돌렸다.** v0.12.0 배포 → `db init` 4분 33초 |
| 3 | F-9.3을 `nohup`으로 띄울지 Cronicle 일회성 이벤트로 만들지 | **Cronicle 이벤트**(키 lock 공유). job `jmtrpfc991a`, 5시간 45분, exit 0. 이벤트는 종결 뒤 삭제했다 |
| 4 | `emsugdoe907` 삭제와 월 1회 이벤트 등록을 같이 할지 | **같이 했다.** `emsugdoe907` 삭제 + `sdc_monthly_corp_profile_history` 등록(1일 05:30, `catch_up=1`) |
