# F-1.4 실행 기록 — 업종 이력 시드 + 첫 수집 스냅샷

- 실행: 2026-09-08 00:09 ~ 00:35 KST
- 릴리즈: **v0.12.0** (`458857d`), 이미지 `ghcr.io/sjleekor/sdc:v0.12.0`
- 결정: **안 B 채택**(시드 2026-08, 첫 수집 2026-09). 사용자 승인 2026-09-08
- 결과: **완료. 오류 0.**

---

## 1. 실행한 것

| # | 단계 | 결과 |
|---|---|---|
| 1 | `main` fast-forward 머지 | 5커밋 |
| 2 | 릴리즈 v0.11.5 → **v0.12.0** | 태그 푸시, CI 이미지 빌드 1m11s 성공 |
| 3 | GHCR 확인 | `v0.12.0` linux/amd64 `sha256:d491acb7…` |
| 4 | `deploy/deploy_to_sj2.sh` | compose 태그 v0.12.0, `dart-seed-corp-profile-history.sh` 신규 동기화 |
| 5 | 이미지 pull + `compose config` | OK, postgres healthy(3일 up) |
| 6 | 배포 이미지에서 새 CLI 확인 | `dart seed-corp-profile-history --help` 정상 |
| 7 | `bin/db-init.sh` | **4분 33초.** `dart_corp_profile_history` + 컬럼 15개 + 인덱스 3개 생성 |
| 8 | 시드 `--observed-month 2026-08-01` | **3,959행** |
| 9 | 시드 재실행(같은 달) | **0행** — `ON CONFLICT DO NOTHING` prod 확인 |
| 10 | 첫 수집 스냅샷(2026-09) | **19분 27초**, `status=success` |

`db init`이 4분 33초 걸린 이유는 새 테이블이 아니라 `dart_xbrl_fact_raw`(136 GB)를 포함한
기존 인덱스 전수 검증이다. 모든 문장이 `IF NOT EXISTS`라서 실제로 만든 것은 새 테이블과
인덱스 3개뿐이다.

### 첫 수집 스냅샷 카운터

```
Requests attempted: 3959
Requests skipped:   0        (--force로 skip-if-present 해제)
Rows upserted:      3959     (dart_corp_master)
No data:            0
History appended:   3959     (dart_corp_profile_history)
```

`ingestion_runs`: `status=success`, `duration=00:19:26.956`, `error_summary=NULL`.

실측 속도 **3.27 req/s**. 런북에 적었던 13분 추정은 호출당 0.2s만 계산한 값이라 틀렸다.
N2-7b 실측(1,301초)과 같은 급이고, **월 20분**으로 보는 게 맞다.

---

## 2. 저장된 상태

| `observed_month` | 행 | 법인 | `is_seed` | `induty_code` 있음 | `observed_at` |
|---|---|---|---|---|---|
| 2026-08-01 | 3,959 | 3,959 | **3,959** | 3,959 | 2026-08-15 (전부 동일) |
| 2026-09-01 | 3,959 | 3,959 | 0 | 3,959 | 2026-09-08 (전부 동일) |

시드 행의 `observed_at`이 `now()`가 아니라 corp master의 `profile_fetched_at`(2026-08-15)로
남았다 — 의도한 동작이다. **그래서 시드 월은 관측 구간이 아니다.**

---

## 3. 첫 변경률 — 데이터 한 점, 판정 아님

`01` §4.1의 정의대로 두 달을 비교했다.

| 지표 | 값 | D-F1 문턱 |
|---|---|---|
| `induty_code` 변경 | 2 / 3,959 = **0.0505%** | — |
| **`ind_group` 변경**(`resolve_groups` fold-up) | 1 / 3,959 = **0.0253%** | 0.3% |
| `corp_cls` 변경 | 6 / 3,959 = 0.1516% | — |
| `ticker` 변경 | 0 | — |
| 그룹 수 | 37 → 37 | — |

### 3.1 무엇이 바뀌었나

| ticker | 법인 | 8월 코드 | 9월 코드 | 그룹 |
|---|---|---|---|---|
| 054410 | 케이피티유 | 474 | 467 | **47 → 46 (실제 그룹 변경)** |
| 355690 | 에이텀 | 262 | 26293 | 26 → 26 (변경 없음) |

**`induty_code` 변경 2건 중 1건은 업종 변경이 아니라 코드 정밀도 변경이다.** 에이텀은
3자리에서 5자리로 세분화됐을 뿐 2자리 접두는 그대로다. 그래서 코드 변경률(0.0505%)과
그룹 변경률(0.0253%)이 갈린다. **D-F1이 보는 것은 그룹 변경률**이므로 이 구분이 중요하다.

### 3.2 이 숫자를 판정에 쓰면 안 되는 이유

- 8월 행은 시드다. `observed_at`이 2026-08-15 하루에 몰려 있고 그 이전 이력이 없어서,
  8→9월 변경은 "8-15 이후 어느 시점"이고 구간 길이가 24일이다. 한 달이 아니다.
- D-F1은 **3개월 스냅샷(2026-09·10·11)의 월평균**으로 2026-12에 판정한다. 진짜 한 달
  구간은 **2026-09 → 2026-10**부터 시작한다.
- 즉 0.0253%는 "문턱 0.3%를 크게 밑돈다"는 방향만 시사한다. 판정은 그대로 2026-12다.

---

## 4. 부수 확인 — `corp_cls`가 상폐를 먼저 잡는다

`corp_cls`가 `E`로 바뀐 6건을 `stock_master`와 대조했다.

| ticker | 법인 | 8월 → 9월 | `stock_master.status` | `last_seen_date` | `dart_corp_master.is_active` |
|---|---|---|---|---|---|
| 082640 | 동양생명보험 | Y → E | DELISTED | 2026-08-28 | t |
| 096610 | 알에프세미 | K → E | DELISTED | 2026-09-02 | t |
| 269620 | 시스웍 | K → E | DELISTED | 2026-08-24 | t |
| 299900 | 위지윅스튜디오 | K → E | DELISTED | 2026-08-17 | t |
| 327610 | 펨토바이오메드 | N → E | **행 자체가 없음** | — | f |
| 471050 | 대신밸런스제17호스팩 | K → E | DELISTED | 2026-08-31 | t |

세 가지가 나온다.

1. **6건 중 5건이 `stock_master`의 상폐 판정과 일치하고, `last_seen_date`가 두 스냅샷
   사이 구간(08-17 ~ 09-02)에 정확히 들어간다.** N2 V3이 본 `corp_cls='E'` ↔
   `status='DELISTED'` 일치가 *증분*에서도 성립한다는 뜻이다. DDL 주석에 적어둔 용도가
   그대로 확인됐다.
2. **6번째(펨토바이오메드, KONEX)는 `stock_master`에 행이 없다.** 유니버스 sync가 추적한
   적이 없는 종목의 상폐를 이 이력이 잡았다. F-9.4(S-2, `universe backfill-master`)가
   메워야 할 집합의 성격을 보여준다.
3. **`dart_corp_master.is_active`는 5건 모두 아직 `t`다.** `is_active`는 `dart sync-corp`가
   갱신하므로, `corp_cls`가 `is_active`보다 **신선하다.**

---

## 5. 다음

- **2026-10 스냅샷**이 첫 진짜 한 달 구간을 만든다. F-1.6(월 1회 Cronicle 이벤트)이
  선행이고, 지금은 등록돼 있지 않다 — `emsugdoe907`은 `timing=false`(수동 전용)이고
  `--force`도 없어서 지금 돌려도 전부 skip한다.
- F-1.7 변경률 리포트 스크립트(`research/analysis/industry_change_report.py`)는 아직 없다.
  이 문서 §3의 숫자는 임시 SQL + `resolve_groups`로 직접 계산했다.
- F-1.5 마트 `dim_industry_pit_daily`는 미착수. `ind_is_backcast` 경계는 **2026-08-15**
  (시드의 `observed_at`)가 된다.
