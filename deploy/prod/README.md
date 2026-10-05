# 일일 브리핑 운영 초안

이 디렉터리는 실행 인터페이스와 **비활성** Cronicle 제안입니다. `config.example.json`과 `release.example.json`의 필수 경로·hash는 아직 비어 있으므로 그대로 실행할 수 없습니다. 실제 KR/US 입력과 `stock_reports` 저장소 설정(Private, 협업자 없음)을 확인하기 전에는 production 일정이나 게시를 시작하지 않습니다.

## 고정 release와 경로

검토한 소스 파일만 별도 release 디렉터리에 복사하고 hash를 고정합니다. 가변 `source/` checkout이나 전체 repository를 runtime에 직접 연결하지 않습니다. Release manifest는 세 모델의 entrypoint, bundle manifest, code inventory와 hash를 묶습니다. frozen release에는 `collector` Python package도 포함해야 합니다. `runtime-verified-sj2-20260930.json`은 sj2 **격리 시험 venv**에서 확인한 Python·패키지 버전입니다. 운영 image가 같은 구성이라는 증거는 아닙니다. `model-cards.json`의 최종 SHA-256은 `f45f4326adcdbeb6e2c8799de3c69b3e9c0c37935d1b76a2d90c5f598d09e385`이며, 운영 config의 `model_cards_path`와 `model_cards_sha256`은 이 파일의 고정 release 사본을 가리켜야 합니다. 카드의 `publication.allowed`는 자체 작성 설명문에만 적용됩니다. report의 `publication` 필드는 지우지 않았지만, private 저장소에 본인만 보는 게시는 이 gate와 별개의 경로입니다(아래 `stock_reports` 절).

Serving data root 아래 `prepared/kr`, `prepared/us`에는 읽기 전용 native 산출물, `prepared/selections`에는 D별 불변 selection, `runs/`에는 report·coordinator state, `private-projections/`에는 내부 렌더 결과를 둡니다. `selection_root`는 `prepared_root` 아래에 있어야 합니다. 모델 프로세스는 collector raw 테이블을 바꾸거나 holdout label을 읽지 않습니다. 학습과 일일 추론은 분리합니다.

## Cronicle entrypoint

실제 호출은 운영 config를 고정하고 아래 세 stage를 사용합니다. `--report-date`를 생략하면 KST 현재 날짜가 D입니다. `--fixture-now`는 `synthetic_fixture=true`인 격리 release에서만 허용합니다.

```sh
python -m modeler.serving.daily_wrapper select  --config CONFIG
python -m modeler.serving.daily_wrapper run     --config CONFIG
python -m modeler.serving.daily_wrapper monitor --config CONFIG --attempt 0
```

`select`는 D 09:30에 KR 이전 완료 세션 K와 US U/E/A를 확인하고 native completion·달력·source policy를 고정합니다. 원천 E는 실제 A에서 역으로 추정하지 않습니다. US E/lag 정책이 없으면 US를 unavailable로 닫고 KR은 독립적으로 선택할 수 있습니다. KR 휴장일에는 skip 상태를 남기고 report를 만들지 않습니다. Native 완료 marker, hash, cutoff 증거가 없거나 늦으면 해당 모델을 선택하지 않습니다.

`run`은 D 10:00 이후에 저장된 selection을 확인하고 infer → render → publish 순서로 실행합니다. 세 모델 모두 unavailable이어도 해당 D의 단위를 만들고 입력 실패를 exit code 1로 알립니다. `publisher_enabled=false`이면 내부 report와 비공개 화면(`private_projection_root`)은 만들지만 게시 단계는 `publication_withheld`(이유 `publisher_disabled`)입니다. KIS 개장 관측은 `modeler.serving.opening_prepare`의 별도 불변 artifact를 씁니다. 원천 시각이 확인되지 않은 관측은 장중 확인으로 올리지 않습니다.

`opening_snapshot_root`, `opening_output_root`, `opening_max_age_seconds`가 모두 설정돼 있으면 `run`이 이미 캡처한 `report_date=D/slot-*.json` 중 수신 시각이 D 10:00 이하인 snapshot만 mapper에 전달합니다. subprocess는 인자를 분리해 `shell=False`로 실행하고 timeout은 90초입니다. slot이 없거나 mapper가 실패하면 opening만 `unavailable`로 두고 KR·US 추론은 계속합니다. 예시 config에서는 이 세 값을 비워 둡니다. 실제 캡처·원천 시각·공개 권한 검증이 끝났다는 뜻은 아닙니다.

`monitor --attempt 0`, `1`, `2`, `3`은 각각 10:15, 10:17, 10:22, 10:32 KST 제안입니다. 저장된 D report와 publisher journal을 확인하고, 실패한 게시는 **동기화 단계만** 다시 돕니다. 추론도 렌더도 다시 하지 않습니다. `external_verification_enabled=true`이면 `git ls-remote`로 원격 `main`이 publisher가 push한 커밋을 포함하는지 보고, 포함해야 `verified`입니다. 꺼져 있으면 `remote_pending`으로 남습니다. 자세한 상태는 아래 `stock_reports` 절에 있습니다.

`cronicle-events.disabled.json`의 이벤트는 전부 `enabled=false`이며 `config_path=null`입니다. 실제 Cronicle category·target·plugin·wrapper 경로는 운영 설정에서 별도 확인해야 합니다. 이 파일은 기존 이벤트를 바꾸지 않습니다. `publisher_enabled=false`, `external_verification_enabled=false`인 예시 config도 실제 게시를 허가하지 않습니다. deploy key(쓰기 권한), 저장소 설정(Private, 협업자 없음), checkout이 확인돼야 게시 경로를 열 수 있습니다.

## 격리 검증 범위

최종 모델 설명 카드가 들어간 격리 evidence는 `ops_real_adapter_e2e_20260930_final_cards/evidence.json`입니다. 실제 세 bundle을 **synthetic feature·fixture clock**으로 호출했고 세 모델 결과, 실패 0건, 전체 `partial`을 확인했습니다. `select`, `infer`, `render`가 성공했고 `publish`는 `publication_withheld`였습니다. 당시 공개 projection도 `synthetic_fixture=true`였습니다(그 경로는 지금 없습니다). 이 결과는 실제 최신 거래일 추론, 공개 Pages 게시, Cronicle 활성화 또는 5일 전진 운영의 증거가 아닙니다.

Cronicle API key는 별도 관리자 절차로 교체했습니다. 비밀 없는 독립 검증에서 이전 key 비활성, 새 key 작동·같은 권한, secret 파일 일치와 mode `0600`을 확인했습니다. 검증 때 조회한 일정은 22행이며 변경 요청은 0건입니다. 이전 응답에서 `api_key` 필드가 보였던 21건과 일정 22행은 다른 집계입니다. 이 검증으로 전후 일정 내용 전체가 같다고 주장하지 않습니다.

수정된 US incremental snapshot의 진단용 native feature는 root 독립 검증에서 22개 원천 파일 SHA와 native hash가 맞았습니다. 4,096행 가운데 모델 대상 3,406행을 독립 Pandas 106열 design으로 다시 예측했으며 LightGBM·Ridge 모두 순위가 전부 같았습니다. 최대 예측값 차이는 각각 0과 `1.1102230246251565e-16`입니다. 수정 전 snapshot의 같은 feature key와 53개 값도 같았습니다. 이 검증은 이전 membership 오류가 기존 모델 예측 오류로 이어졌다는 뜻이 아닙니다. 반면 raw와 frozen 입력의 재무 피쳐 11개는 아직 다릅니다. `diagnostic_only=true`, `serving_eligible=false`, 게시 보류를 유지합니다.

## KR prepare 휴장일 CSV

`kr_live_prepare`는 collector 패키지 데이터 `holidays_krx.csv`로 거래일을 계산합니다. frozen release는 `*.py`만 복사하므로 release 트리에서 돌릴 때는 CSV를 `--holidays-csv`로 넘겨야 합니다. CSV를 찾지 못하면 주말만 제외하고 계속하지 않고 멈춥니다. 사용한 달력은 prepare manifest의 `holiday_calendar`에 남습니다. 이 CSV는 2014년부터 2026-12-31까지만 있습니다. 2027년 이후 D는 CSV를 갱신하기 전까지 prepare가 멈춥니다. 2026년 안에 갱신하십시오. frozen release의 추론 경로(`adapters`→`kr_serving`·`us_daily`)는 이 CSV를 쓰지 않습니다.

`kr_live_prepare --profile`은 `full`(기본, mart 12개 전부)과 `serving`(모델 피쳐와 품질 컬럼에 필요한 것만)을 고릅니다. `serving`은 vintage mart 둘을 만들지 않고 `feat_fin_scan_daily`를 요청한 컬럼(`fin_log_mcap`)만 담은 projection으로 만들며, 달력도 쓰지 않으므로 CSV 갱신에 막히지 않습니다. 고른 profile은 `_SUCCESS.json`의 `profile`과 `build_profile.json`에 남고, `kr_prepare`가 marker와 디스크의 mart가 맞는지 확인합니다. `full`에서 접수일이 달력 밖인 공시는 `--max-beyond-calendar-rows`(기본 500행)까지는 건수만 `_SUCCESS.json`의 `beyond_calendar`에 남기고 경고합니다. 넘으면 빌드가 멈춥니다.

## report provenance 키

- `bundle_manifest_sha256`, `prepared_manifest_sha256`은 KR·US 모두 pinned **파일**의 SHA-256입니다.
- KR 내부 계산 hash는 별도 키 `bundle_manifest_content_sha256`, `native_prepare_manifest_content_sha256`에 있습니다.

## 최신 격리 검증 증거 (2026-09-30)

synthetic 입력으로 실제 세 bundle을 호출한 `release_nav` E2E입니다. 실제 시장 데이터나 게시 증거가 아닙니다. 전체 `partial`, 실패 0건, `publication_withheld`이며 push는 하지 않았습니다.

- 경로: `/home/whi/tmp/daily_market_briefing_20260929/ops_real_adapter_e2e_20260930_release_nav/` (`opus_independent_verification.json` 101개 항목 통과, 화면 캡처는 `opus_visual_qa/`)
- 종합: `/home/whi/tmp/daily_market_briefing_20260929/root_opus_20260930/opus_final_verification_20260930.json`
- 상세와 남은 일은 `my/milestones/common/20260929_daily_market_briefing/01_implementation/readiness.md`에 있습니다.

## 고정 release 만들기

고정 release는 `python -m modeler.serving.release_build`로 만듭니다. `--source-manifest`(`sha256sum` 형식 검토 목록)에 없거나 hash가 다른 소스는 복사하지 않고 멈춥니다. `--model-cards`, `--runtime-manifest`, `--uv-lock`도 `modeler_root` 아래 파일이어야 하고 같은 목록과 hash가 맞아야 합니다.

출력 경로가 이미 있으면 멈춥니다. 같은 부모의 임시 폴더에서 만든 뒤 rename하고, 끝에 `_release_jobs`로 자체 검증합니다.

`code_sha256`은 release의 절대경로를 포함하므로 만든 뒤에 release를 옮기거나 복사할 수 없습니다.

패키지 데이터는 allowlist(`holidays_krx.csv`)만 복사하고 `release.json`의 `data_files`에 경로와 sha를 적습니다. `_release_jobs`는 `data_files`가 있으면 release root 안 상대경로, symlink 아님, 파일 존재, sha256 일치, 경로 중복 없음을 검증합니다. 키가 없는 기존 fixture release는 그대로 통과합니다.

## `stock_reports` 게시 (reports publisher)

Pages 공개 경로는 없앴습니다. `publish` 단계는 private 저장소 `sjleekor/stock_reports`의 `main`에 markdown 단위를 push합니다. 올리는 내용과 범위는 [`02_publish_to_stock_reports.md`](../../../my/milestones/common/20261005_daily_briefing_reports/00_candidate_plan/02_publish_to_stock_reports.md)가 정합니다. 올리는 것은 모델별 상위 N(기본 100)의 순위·코드·이름과 순위용 점수(소수 4자리), KR 품질 사유, 기준일, 신선도, 실패 사유, release·sha256입니다. 전체 순위, 원시 피쳐, 서버 경로, 비밀값은 올리지 않습니다. **이 경로는 저장소를 본인만 본다는 전제입니다.** 협업자를 추가하거나 공개로 바꾸면 올린 내용의 판단이 무효이므로, publisher 설정의 `audience`는 `owner_only`가 아니면 멈춥니다.

### 구성 요소

| 구성 요소 | 위치 | 하는 일 |
|---|---|---|
| 렌더러 | `src/modeler/reporting/markdown.py` (release 안) | report를 단위 폴더(`README.md`·`market-sector.md`·`kr-stocks.md`·`us-stocks.md`·`data-status.md`)로 렌더하고, 인덱스 자동 구간을 다시 쓰고, 트리를 검증합니다. 표준 라이브러리만 쓰고, 템플릿은 `.py` 문자열이라 release `DATA_FILES`를 늘리지 않습니다. **release 검토 목록(`SOURCE_MANIFEST`)에 이 파일을 넣어야 release 빌드가 통과합니다.** |
| publisher | `deploy/reports/publish_reports.py` (release 밖, `serving/publisher/`로 복사, sha256 고정) | 로컬 단계와 동기화 단계 |
| 검증기 | `deploy/reports/validate_reports.py` (같은 곳) | 경로 allowlist, front matter, 크기, 상대 링크, 과거 단위 보호, 서버 경로·비밀값, symlink |

coordinator가 publisher를 `PYTHONPATH=<release>/src`로 부르므로 렌더와 검증은 고정된 release 코드가 합니다.

### 두 단계

`publish` 단계는 `publish_reports.py run`을 한 번 부릅니다. 이 안에서 로컬 단계가 먼저 끝나므로 GitHub에 닿지 않는 날에도 단위와 journal이 남습니다.

| 단계 | 하는 일 | 실패하면 |
|---|---|---|
| L1 | `runs/D/report-D.json`의 sha256과 `run-state.json`의 invocation id가 이번 실행의 것인지 확인합니다 | `rejected` |
| L2 | `runs/D/markdown/unit/`에 단위 파일 다섯 개를 렌더합니다. 같은 입력을 다시 돌려 내용이 같으면 파일을 그대로 둡니다(`generated_at` 고정) | `rejected` |
| L3 | 단위 수준 검증(front matter, 1MB, 단위 안 링크, 서버 경로·호스트명·키 모양 문자열) | `rejected`, `markdown/rejected.json` |
| L4 | journal에 "동기화 대기"를 적습니다 | |
| S1 | flock으로 동시 실행을 막습니다 | 바로 종료(`locked`) |
| S2 | checkout을 확인합니다. 깨끗함, branch `main`, remote URL 정확 일치, checkout 자체의 작성자 설정(`user.name`·`user.email`, 전역 설정에 기대지 않음) | 멈춤. journal 그대로 |
| S3 | fetch합니다 | 연결 실패면 `sync_pending`으로 끝. journal 그대로 |
| S4 | 커밋은 했지만 push하지 못한 단위가 있으면 먼저 처리합니다. 원격이 앞서 나갔으면 로컬 커밋을 버리고(`reset --hard origin/main`) 단위를 "동기화 대기"로 되돌려 다시 얹습니다. journal에 없는 로컬 커밋이 있으면 멈춥니다 | `failed` |
| S5 | 밀린 날짜부터 차례로 원격 최신 위에 단위를 얹습니다. 원격에 같은 단위가 있고 `generated_at`·`revision`·정정 줄을 뺀 내용이 같으면 **새 커밋만 만들지 않습니다**(그래도 push 대기 커밋은 계속 처리합니다) | |
| S6 | 인덱스(루트·종류·월 README의 자동 구간)를 다시 만들고, 없는 모델 카드만 만들고, 이번 커밋이 바꾸는 파일을 검증합니다 | `rejected`. 작업 트리를 되돌림 |
| S7 | 파일을 하나씩 지정해 `git add`하고 커밋합니다. 메시지는 `daily-briefing 2026-10-07` | |
| S8 | force 없이 push합니다. non-fast-forward면 S3부터 다시, 최대 3회 | `push_pending`. 로컬 커밋과 journal이 남음 |

monitor의 재시도는 로컬 단계가 이미 끝났다면 `publish_reports.py sync`만 돕니다(끝나지 않았으면 `run`을 다시 돕니다). 그래도 안 되면 밀린 단위는 다음 날 실행이 날짜 순서대로 먼저 올립니다.

### 정정과 과거 단위

- 같은 D를 다시 돌려 내용이 같으면 커밋하지 않습니다(`unchanged`).
- 내용이 다르면 `correction_required`(종료 코드 30)로 멈추고 아무것도 올리지 않습니다. 고치려면 사람이 `sync --correct <단위> --reason <사유>`를 돌립니다. `revision`을 하나 올리고, 요약 맨 위에 정정 줄(사유, 이전 판 커밋)을 넣고, 커밋 메시지는 `daily-briefing 2026-10-07 r2: <사유>`입니다. 이전 판은 git 이력에 남습니다.
- 이미 올린 단위를 바꾸거나 지우는 커밋은 검증기가 거부합니다(`--correct`한 그 단위만 예외). 루트 README의 자동 구간 밖을 고치는 것, `CONVENTIONS.md`·`reference/glossary.md`를 고치는 것, 남의 모델 카드를 만들거나 바꾸는 것도 거부합니다.

```sh
# 이미 올린 2026-10-07 단위를 고칩니다(local 단계로 새 판이 journal에 대기 중이어야 합니다).
python publish_reports.py sync --config runs/2026-10-07/reports-publisher-config.json \
    --correct 2026-10-07 --reason "KR 점수 재계산"
```

### 종료 코드와 상태

| 종료 코드 | 결과 `status` | 뜻 |
|---|---|---|
| 0 | `published` · `unchanged` · `nothing_to_do` · `local_ready` | 원격 `main`에 단위가 있습니다(또는 할 일이 없습니다) |
| 1 | `failed` | checkout·설정·journal 오류. journal은 그대로입니다 |
| 10 | `rejected` | 검증 실패. 단위는 `runs/D/markdown/`에 있고 journal에 사유가 있습니다 |
| 20 | `sync_pending` | fetch가 안 됨(원격에 닿지 않음). 단위와 journal이 남았습니다 |
| 21 | `push_pending` | 커밋은 했지만 push하지 못함. 같은 내용으로 다시 돌리면 같은 커밋을 push합니다 |
| 30 | `correction_required` | 원격에 같은 단위가 다른 내용으로 있음 |
| 75 | `locked` | 다른 publisher 실행이 잠금을 쥐고 있음 |

결과는 stdout에 JSON 한 줄로 나옵니다. coordinator는 종료 코드 0이면 `coordinator-publication.json`의 `status`를 `published`로, 아니면 `publisher_failed`로 적고 `publisher_status`·`publisher_exit_code`·`local_done`을 같이 남깁니다. 성공하면 `reports_commit`(원격 `main`의 커밋)도 적습니다.

### ops 설정 키

Pages용 `base_path`·`projection_root`·`previous_projection_dir`·`publisher_script(_sha256)`·`publisher_config`·`site_checkout`·`actions_repository`·`actions_workflow`·`public_manifest_url`은 없앴습니다. `publisher_enabled`가 `true`일 때 아래 키를 모두 검증합니다. `false`이면 검증하지 않고 `publish`는 `publication_withheld`로 끝납니다.

| 키 | 값 |
|---|---|
| `publisher_enabled` | 게시를 켤지. 명시해야 합니다 |
| `external_verification_enabled` | monitor가 `git ls-remote`로 원격을 볼지 |
| `reports_repository` | `sjleekor/stock_reports`(다르면 멈춤) |
| `reports_audience` | `owner_only`(다르면 멈춤) |
| `reports_branch` | `main` |
| `reports_remote_url` | `git@github.com:sjleekor/stock_reports.git`. publisher가 checkout의 remote URL과 정확히 같은지 봅니다 |
| `reports_checkout` | 깨끗한 checkout 경로. 예: `/home/whi/apps/market-briefing/reports-checkout`. 작성자는 이 checkout의 git 설정(`stock-reports-bot`)입니다 |
| `reports_publisher`, `reports_publisher_sha256` | `serving/publisher/publish_reports.py`와 그 sha256 |
| `reports_top_n` | 모델별로 올릴 상위 개수(1\~500, 기본 100) |

checkout 옆(같은 부모 디렉터리)에 숨김 파일 둘이 생깁니다. `.reports-checkout.reports-publish.lock`(flock)과 `.reports-checkout.reports-publish-journal.json`(0600, 처리할 단위가 없으면 지움)입니다. journal은 단위별로 `sync_pending`·`push_pending`·`rejected`·`correction_required`와 단위 해시, 커밋 sha를 적습니다. checkout 안에는 아무것도 만들지 않습니다.

날짜별 입력은 coordinator가 `run_root/<D>/reports-publisher-config.json`에 원자적으로 씁니다(저장소·checkout·release id·report sha256·invocation id·모델 카드 경로·top_n). 비공개 화면 경로는 여기 넣지 않습니다. 합성 fixture release에서만 `--allow-synthetic`을 붙입니다.

### monitor 상태

| `status` | 뜻 |
|---|---|
| `verified` | 원격 `main`이 publisher가 push한 커밋을 포함합니다. 그 뒤에 사용자가 직접 커밋을 올려 head가 달라졌어도, 그 커밋이 새 head의 조상이면 포함으로 봅니다 |
| `remote_missing_commit` | 원격 `main`에 그 커밋이 없습니다(되돌려졌거나 push가 사라짐) |
| `remote_pending` | 검증을 끄고 있거나(`external_verification_enabled=false`) 원격에 닿지 않아 확인하지 못했습니다 |
| `publisher_failed` | 게시가 아직 안 됐습니다. 10:15 이후 시도마다 동기화 단계만 다시 돕니다 |

axes는 `input`, `inference`, `rights`(참고용), `publisher`, `remote`입니다. 사용자의 후속 커밋 확인은 개인 ref `refs/monitor/main`으로 받아 publisher가 쓰는 `refs/remotes/origin/main`을 건드리지 않습니다.

## 비공개 리포트 보기

서버에서 본인만 보는 HTML 화면입니다. 설정 키 `private_projection_root`(선택, null이면 만들지 않음)에 만듭니다. `stock_reports`에 올라가는 markdown 단위와 별개이고, **이 디렉터리 자체는 어떤 저장소에도 복사하지 않습니다.**

- `render` 단계가 `private_projection_root/<D>`에 비공개 화면을 씁니다. 이 경로가 `run_root`나 `reports_checkout`과 같거나 서로 안쪽이면 설정 검증에서 멈춥니다.
- 출력은 `index.html`(최근 날짜), `archive/index.html`(날짜별 목록), `reports/<날짜>/index.html`, `assets/private.css`, `private-manifest.json`(`private: true`), `PRIVATE_DO_NOT_PUBLISH.txt`입니다. 링크가 모두 상대 경로라 어느 디렉터리에서 열어도 됩니다.
- 모델별 상위 100개(전체 개수 표시)와 점수가 나옵니다. 점수는 순위용이고 확률이 아닙니다. 모든 페이지 맨 위에 "비공개 — 게시 금지"와 `noindex` 메타가 있습니다.
- `coordinator-render.json`의 `private_projection_dir`, `private_manifest_sha256`에 출력 경로와 manifest hash가 남습니다.
- `publish` 단계는 이 출력을 publisher에 넘기지 않습니다. 검증기는 이 출력(HTML, 마커, 매니페스트)을 허용 경로 밖 파일로 보고 거부합니다. 이 디렉터리를 `reports_checkout`에 복사하지 마십시오. 마커 파일(`PRIVATE_DO_NOT_PUBLISH.txt`)이 이 금지를 적고 있습니다.

보는 방법은 SSH 터널입니다. sj2에서 해당 날짜 디렉터리를 loopback으로만 서빙합니다.

```
ssh -L 8765:127.0.0.1:8765 whi@sj2-server \
  "python3 -m http.server --bind 127.0.0.1 8765 --directory <private_projection_root>/<D>"
```

터널을 연 채로 브라우저에서 `http://127.0.0.1:8765/`를 엽니다. 이전 날짜도 `archive/index.html`에서 볼 수 있습니다.

## US 재무 feature 규칙 A와 점수 동등성 gate

US 서빙은 재무 11개(`bm`, `ep_ttm`, `cfp_ttm`, `sp_ttm`, `roa_ttm`, `roe_ttm`, `gpa`, `opm_ttm`, `asset_growth`, `accruals`, `net_issuance`)와 같은 fact 선택을 타는 `buyback_yield`에 규칙 A(`det_a`)를 씁니다. 기존 규칙(`legacy`)은 같은 `filed` 행과 q\_end 동률을 입력 순서가 정해서, 행 순서만 바꿔도 값이 달라졌습니다. 연구 경로는 인자를 안 넘기므로 `legacy` 그대로입니다. 서빙 상수는 `SERVING_FINANCIAL_RULE = "det_a"`이고 native manifest에 `financial_selection_rule`, `financial_code_hash`로 남습니다.

규칙 A는 같은 `filed`에서 `(accn, 정정본 여부, unit이 USD·shares인지, val)` 오름차순의 마지막 행을 고르고, q\_end 동률은 `start` 내림차순, `q_val` 오름차순으로 정합니다. unit은 거르지 않습니다. frozen 값과 셀 단위로 같지는 않습니다(재무 11개에서 62셀 안팎 차이). 그래서 parity 상태는 `passed`가 아니라 `score_equivalent`입니다.

gate는 `python -m modeler.serving.us_daily parity-check --rule det_a --output <json> --lake-root <us lake> --frozen-root <frozen_inputs> --bundle-root <bundles>`입니다. frozen source map으로 golden 6일을 다시 만들고 label은 읽지 않습니다. 기준은 `modeler.serving.orchestration.US_PARITY_CRITERIA` 상수이고 evidence JSON에도 적힙니다.

| 검사 | 기준 |
|---|---|
| 규칙 대상 12개 결정성 (fact 순서 2회 섞기 x panel 범위 2가지) | 차이 0셀 |
| 나머지 42개 feature | frozen과 셀 단위 동일 |
| 두 모델 각각, 날짜별 최소 Spearman | 0.9999 이상 |
| 상위 50·100 겹침 | 1.0 |
| p99 순위 이동 | 1 이하 |
| frozen 상위 100 안 최대 순위 이동 | 1 이하 |

2026-09-30 실행 결과는 `score_equivalent`입니다. evidence는 `/home/whi/tmp/daily_market_briefing_20260929/us_parity_det_a_20260930/parity_det_a_20260930.json`(sha256 `b0c8b41db3978f1569e171d3bf67e901a5ef88644fcfadb7308a920f7952e058`)입니다. 코드가 바뀌면 `financial_code_hash`가 달라져 evidence를 다시 만들어야 합니다.

서빙이 열리는 조건은 모두 맞아야 합니다.

- native manifest의 `raw_feature_parity_status`가 `score_equivalent`이고 `diagnostic_only=false`, `serving_eligible=true`.
- `raw_feature_parity_evidence`의 sha256이 `raw_feature_parity_evidence_sha256`과 같다.
- evidence의 `status`가 `score_equivalent`이고 `rule`이 manifest의 `financial_selection_rule`(`det_a`)과 같다. `financial_code_hash`도 같다.
- evidence 안의 수치가 위 기준을 실제로 넘는다. 저장된 `status` 글자는 믿지 않고 수치에서 다시 계산합니다.

evidence가 없거나 `failed`이거나 규칙이 다르거나 `legacy`면 지금처럼 차단됩니다(`select`, `infer`, `daily_inputs`, `run_daily` 모두 `us_native_block_reason`으로 같은 판정). 만드는 명령은 `prepare --as-of <A> --raw-feature-parity-status score_equivalent --raw-feature-parity-evidence <json>`이고 `--diagnostic-only`와 같이 쓸 수 없습니다. 운영 활성화는 이 절의 범위 밖입니다.

## US E 표와 native prepare 운영 연결 (제안, 활성화 전)

**E 표.** `python -m modeler.serving.us_expected build --start D --end D --output <json>`이 `us-expected-source.v1` 파일을 만듭니다. 정책 초안 1(잠정)을 그대로 구현했습니다. `E(D)`는 D-1 15:00 KST(`sdc_daily_us` 시작)에 이미 닫힌 마지막 XNYS 세션이고, 닫힘 시각은 날짜별 정규장·조기폐장 close를 뉴욕 시간대로 계산합니다. 한국 거래일만 넣고 KR 휴장일은 뺍니다. 달력은 `calendar_sources`의 NYSE(2026\~2027)·KRX(2026) manifest입니다. 범위가 달력 밖이면 멈춥니다. `market_lag_limit_sessions`는 2입니다. 출력 파일에 규칙 이름, 생성 시각, 달력 manifest sha256, 수집 시각 가정(15:00 KST, derive 15:30)을 적습니다.

`daily_inputs.select`는 `reviewed_status`가 `confirmed`인 파일만 읽습니다. 그래서 생성기는 기본값 `unreviewed`로 쓰고, 이 값이면 US는 `unavailable`로 닫힙니다. 정책을 사람이 확인한 뒤에만 `--reviewed-status confirmed --review-note "<누가 언제>"`로 다시 만드십시오. 같은 이름의 파일은 덮어쓰지 않습니다. 새 날짜 이름으로 만들어 겹치는 날짜 값이 이전 파일과 같은지 비교한 뒤 config의 `us_expected_source`를 바꿉니다. `select`는 D별로 이 파일의 sha를 고정하므로, D 09:30\~10:35에는 바꾸지 마십시오. 갱신은 월 1회 08:30 KST 제안(`market_briefing_us_expected_refresh`)입니다. KR 달력은 2026-12-31까지라 2027년 표는 휴장일 CSV를 갱신한 뒤에만 만들 수 있습니다.

`python -m modeler.serving.us_expected session --report-date X`는 X 하나의 E를 찍습니다(KR 필터 없음). prepare가 쓸 A를 이 명령으로 구합니다. 실행일 T 17:30에는 `X = T+1`입니다. 데이터에서 A를 거꾸로 추정하지 않습니다.

2026-10-01\~12-31 표(62일)를 `/home/whi/tmp/daily_market_briefing_20260929/us_expected_20260930/us-expected-source_20261001_20261231.json`에 만들어 두었습니다(`unreviewed`, 어떤 config에도 넣지 않았습니다).

**native prepare 제안.** `cronicle-events.disabled.json`의 `market_briefing_prepare_us`를 매일 17:30 KST로 고쳤습니다. `--diagnostic-only`를 빼고 `--raw-feature-parity-status score_equivalent`와 고정 evidence(sha256 `5d434c4e…`)를 씁니다. 16:30 universe incremental 뒤 시각인 이유는 격리 실측이 2분 6\~9초(최대 RSS 4.4GB, 2코어)라 incremental이 길어져도 여유가 있기 때문입니다. prepare는 collector 컨테이너 밖에서 돌아 `us` flock을 공유할 수 없습니다. 같은 날 derive와 incremental이 code 0으로 끝났는지를 선행조건으로 보십시오. 도중에 snapshot이 바뀌면 prepare가 스스로 멈춥니다. 쓰기 위치는 `SERVING_STOCK_ROOT/us/output/us_scoring_daily_v1/prepared`뿐이고, `SERVING_STOCK_ROOT/us/raw`·`derived`는 운영 lake로 가는 링크입니다. `prepared_root/us`는 그 `prepared` 디렉터리를 가리키게 합니다. 격리 시험에서 이 링크 구조로 prepare가 돌고 `daily_inputs._candidates`가 native를 찾는 것까지 확인했습니다. `select` 전체 경로는 이 구조로 시험하지 않았습니다.

격리 실데이터 결과(score_date 2026-09-25): native가 `serving_eligible=true`, `diagnostic_only=false`, `financial_selection_rule=det_a`이고 `us_native_block_reason`이 `None`을 돌려줍니다. 이전 진단 native(`prep_id=b30587f6f946d626`)와 key 4,096개가 같습니다. 재무 12개는 49,152셀 중 4셀만 다르고, 나머지 feature는 `div_yield` 11셀(차이 7e-18, 부동소수 합산 순서)만 다릅니다. 두 모델 순위는 전부 같습니다. 같은 입력으로 prepare를 두 번 돌리면 `div_yield`의 같은 종류 잡음 때문에 `feature_hash`와 `prep_id`가 달라질 수 있습니다. 점수에는 영향이 없지만 prep_id 재현성이 필요하면 `payout.py`의 합산 순서를 고정해야 합니다.


## div_yield 결정성 수정과 evidence v2 (2026-09-30)

같은 입력으로 prepare를 두 번 돌리면 `div_yield` 합산 순서 잡음(약 7e-18) 때문에 feature hash와 `prep_id`가 달라졌습니다. 원인은 `payout._dividend_ttm`의 `group_by().sum()`이 입력 순서대로 더한 것이고, `selection_rule=det_a`일 때만 합산 전에 `date, symbol, ex_date, amount`로 정렬하도록 고쳤습니다. legacy 연구 경로는 그대로입니다. 두 번 실행한 feature sha256이 같아졌습니다(`e030662b…`).

재무 코드 hash가 `b00bd5ec…`로 바뀌어 gate를 다시 돌렸습니다. **현행 evidence는 `/home/whi/tmp/daily_market_briefing_20260929/us_parity_det_a_20260930/parity_det_a_v2_20260930.json`(sha256 `5d434c4e1f670d2ef88361c90549df912673ee593dcef7f20597b5aa3cb301c8`)**, `status=score_equivalent`입니다. LightGBM 203행·최소 Spearman 0.9999943, Ridge 542행·0.99999987, 두 모델 상위 50·100 겹침 1.0입니다. 이전 v1 evidence(`b0c8b41d…`)는 이전 재무 코드용 기록이고 운영에 쓰지 않습니다. v1에서 Ridge가 548행이던 것은 이 합산 잡음 때문이었고, v2는 주 세션 독립 계산(542행)과 같습니다. 새 evidence로 만든 격리 native `prep_id=9a9f4cd9257bf27d`는 `serving_eligible=true`입니다. 이전 코드로 만든 native는 `code_hash`가 현재 release와 달라 select·infer에서 거부됩니다.

## 운영 serving 설치와 wrapper

운영 serving root는 `provision_serving.py`가 한 번에 만듭니다. 표준 라이브러리만 써서 시스템 `python3`로 돌립니다. release는 자기 절대 경로에 묶이므로 같은 release id나 이미 `config/ops.json`이 있는 root에는 다시 실행되지 않고 멈춥니다. 중간에 실패하면 새 root를 지우고 다시 하거나 새 release id와 새 root를 쓰십시오. 먼저 `--dry-run`으로 입력 검사와 만들 경로만 확인합니다. dry-run은 아무것도 쓰지 않습니다.

```
B=/home/whi/tmp/daily_market_briefing_20260929
python3 $B/source/modeler/deploy/prod/provision_serving.py \
  --serving-root /home/whi/apps/market-briefing/serving --release-id r20261001 \
  --modeler-root $B/source/modeler --collector-root $B/source/collector \
  --source-manifest <고정한 SOURCE_MANIFEST.sha256> \
  --kr-bundle $B/kr_serving/artifacts/kr_daily_h20_v1 \
  --us-lightgbm-bundle $B/us_serving_verified_20260930/bundles_variant_v1/lightgbm \
  --us-ridge-bundle $B/us_serving_verified_20260930/bundles_variant_v1/ridge \
  --venv-source $B/venv \
  --us-expected-source $B/us_expected_20260930/us-expected-source_20261001_20261231.confirmed.json \
  --us-expected-sha256 54450ffd0ea135fd93f58ca420e47425818a198ef90fda276067df29c26ddf91 \
  --parity-evidence $B/us_parity_det_a_20260930/parity_det_a_v2_20260930.json \
  --parity-evidence-sha256 5d434c4e1f670d2ef88361c90549df912673ee593dcef7f20597b5aa3cb301c8 \
  --reports-checkout /home/whi/apps/market-briefing/reports-checkout \
  --us-lake /home/whi/data/stock_data/us
```

이 스크립트는 lake를 읽기 전용 링크로만 걸고 `select`·`run`·`monitor`를 실행하지 않습니다. 끝에서 `daily_coordinator._config`, `daily_inputs._release_jobs`, `runtime_contract` 검증과 prepared root 구조 확인을 하고 요약 JSON을 찍습니다.

| 경로 (serving root 아래) | 내용 | 권한 |
|---|---|---|
| `venv/` | 검증된 venv의 `cp -a` 사본 (이미 있으면 python·패키지 버전만 검증) | 복사 그대로 |
| `releases/<id>/` | `release_build`로 만든 non-synthetic release. 만든 뒤 읽기 전용 | 디렉터리 0550, 파일 0440 |
| `config/` | `ops.json`(0640), `pins.json`, E 표, parity evidence, `calendars/`(KR 2026, US 2026\~2027) | 디렉터리 0750, 나머지 0440 |
| `publisher/` | `publish_reports.py`, `validate_reports.py` 사본 (검토한 manifest의 sha와 같아야 함) | 0750, 파일 0440 |
| `stock_data/us/raw`, `derived` | 운영 lake로 가는 symlink (읽기만) | |
| `stock_data/us/output/` | 실제 디렉터리. prepare가 쓰는 유일한 곳 | 0750 |
| `prepared/kr/` | 빈 디렉터리 (KR 입력이 없으면 selector가 `unavailable`로 닫음) | 0750 |
| `prepared/us` | `stock_data/us/output/us_scoring_daily_v1/prepared`로 가는 symlink | |
| `prepared/selections/` | D 선택 결과. prepared root 안에 있어야 함 | 0750 |
| `runs/`, `locks/` | coordinator 상태 | 0700 |
| `logs/` | 로그 | 0750 |
| `private-projection/` | 비공개 리포트. 게시하지 않음 | 0700 |

권한은 모두 소유자 전용입니다. Cronicle이 같은 사용자로 돌고, 비공개 리포트와 실행 상태를 다른 계정에 열 이유가 없기 때문입니다. release를 읽기 전용으로 둔 것은 frozen 코드를 실수로 고치지 못하게 하려는 것입니다. 지우거나 바꿔야 하면 `chmod -R u+w`를 먼저 하십시오. `ops.json`만 0640이라 나중에 publisher를 켤 때 고칠 수 있습니다.

`ops.json`은 `publisher_enabled=false`, `external_verification_enabled=false`로 만듭니다. `opening_*`는 null, `private_projection_root`는 `private-projection`입니다. `reports_*` 키는 채우되(`reports_repository=sjleekor/stock_reports`, `reports_audience=owner_only`, `reports_branch=main`, `reports_remote_url`은 SSH URL, `reports_checkout`은 `--reports-checkout` 값, `reports_publisher(_sha256)`은 `serving/publisher/publish_reports.py`) 게시는 켜지 않습니다. `--reports-checkout`은 아직 없어도 되지만 serving root 안에 있으면 안 됩니다. `pins.json`에 E 표, evidence, publisher 스크립트의 sha를 남깁니다. KR 달력은 2026-12-31까지라 2027년 전에 휴장일 CSV와 달력을 갱신해야 합니다.

### wrapper 셋

- `bin/briefing-stage.sh <select|run|monitor> [--attempt N]`: `SERVING_ROOT`(기본 `/home/whi/apps/market-briefing/serving`)의 `config/ops.json`에서 release root와 python을 읽습니다. release root에서 `PYTHONPATH=<release>/src`로 `python -m modeler.serving.daily_wrapper <stage> --config <ops.json>`을 실행합니다. `--attempt`는 `monitor`에서만 쓰고 0\~3입니다. 종료 코드는 wrapper의 것을 그대로 돌려주고, 인자 오류는 2, serving root 문제는 10입니다. 로그는 stdout·stderr로 나가 Cronicle이 받습니다.
- `bin/us-prepare.sh [--run-date YYYY-MM-DD]`: 실행일 T(KST)에 `us_expected session --report-date T+1`로 A를 구하고, `pins.json`의 evidence sha를 확인합니다. lake가 A를 아직 덮지 못하면 멈추지 않고 **A′**(`us_daily resolve-session`: A 이하의 가장 최근 XNYS 세션 가운데 `prices_daily`와 `universe_daily`가 둘 다 덮는 세션)로 바꿔 만듭니다. `STOCK_DATA_ROOT=<serving>/stock_data`, `POLARS_MAX_THREADS=2`, `taskset -c 0,1`, `timeout 1800`으로 `us_daily prepare --as-of A′ --raw-feature-parity-status score_equivalent --raw-feature-parity-evidence <config 사본>`을 돌립니다. 같은 A′가 이미 서빙 가능 상태(`score_equivalent`, `serving_eligible`, 현재 코드 hash, evidence sha 일치)로 있으면 성공으로 건너뜁니다. 종료 코드는 10(설정, `flock` 없음 포함), 11(evidence sha 불일치), 12(A 계산 실패), 20(lake가 완결 세션을 하나도 덮지 못하거나 lake를 못 읽음), 21(prepare는 끝났는데 서빙 가능한 native가 없음)이고, 그 밖에는 prepare의 코드입니다(124는 timeout). A′가 A보다 이르면 `WARNING`을 남기고 종료 코드는 0입니다. evidence 경로가 `prep_hash`에 들어가므로 config 사본 경로를 옮기지 마십시오.
- `bin/kr-prepare.sh [--report-date YYYY-MM-DD]`: 기다리지 않고 KR native를 만듭니다. 아래 "기다리지 않는 실행"에 있습니다.

### 첫 실제 실행 순서

1. 10-01 17:30 `us-prepare.sh`: A는 2026-09-30입니다. 16:30 `sdc_daily_us_universe_incremental`이 끝난 뒤여야 합니다. 2026-09-30 실측에서 운영 lake의 `universe_daily`는 2026-09-22 snapshot(최대 날짜 09-18)이고 incremental 완료 marker가 없어, prepare가 `US universe_daily requires a completed incremental membership snapshot`으로 멈춥니다. incremental이 먼저 한 번 돌아야 합니다.
2. 10-02 09:30 `briefing-stage.sh select`: D 선택이 잠깁니다. 이 시각 전에는 실행하지 마십시오.
3. 10-02 10:00 `briefing-stage.sh run`. 이어서 10:15·10:17·10:22·10:32에 `monitor`(`--attempt` 0\~3)입니다. publisher를 켜기 전에는 `publication_withheld`로 끝나는 것이 정상입니다.

## 기다리지 않는 실행 (R3 · 2026-10-06, 서버에는 아직 반영하지 않음)

원칙은 "막지 말고 표시한다"입니다. 정확성 조건(입력 완료 시각 ≤ D 09:30, 해시·marker·schema, US 서빙 허용 수치 판정, 달력 범위 밖, marker 없는 snapshot)은 그대로 막고, 신선도는 막지 않고 리포트에 적습니다. 계획은 `my/milestones/common/20261005_daily_briefing_reports/00_candidate_plan/04_run_without_waiting.md`입니다. 이 절의 변경은 release가 바뀌므로 US native를 다시 prepare해야 합니다(`code_tree_hash`가 `us_daily.py`를 포함합니다).

### KR prepare: 대기 없이 기준일을 고릅니다

`kr-prepare.sh`는 D 07:30까지 기다리지 않습니다. 순서는 다음과 같습니다.

| 단계 | 하는 일 |
|---|---|
| 0. lock | `$SERVING_ROOT/locks/kr-prepare.lock`에 `flock -n`을 잡고 보존 정리 끝까지 쥡니다. chain 실행과 fallback 타이머 실행이 같은 파일을 씁니다. 못 잡으면 `locked`를 남기고 종료 코드 0으로 끝납니다. lock은 프로세스가 끝나면(죽어도) 풀립니다 |
| 1. 게이트 | `kr-export-wait-ready.sh --deadline-seconds 0`을 한 번 부릅니다. 판정(`ready`·`not_yet`·`blocked`, 오류)과 DART run 기록은 증거에만 남기고 막지 않습니다. `KR_PREPARE_GATE=off`면 부르지 않습니다 |
| 2. raw export | 그 시점 DB를 단일 snapshot으로 내보냅니다 |
| 3. 기준일 선택 | `modeler.serving.kr_reference`가 **내보낸 snapshot 자체에** 완결성 검사를 돌립니다. 가격 종목 수가 직전 세션의 97% 이상이고 수급 그룹(해외보유·투자자·공매도)이 그 세션까지 오면 완결입니다. K가 완결이면 `complete_K`, 아니면 K 아래 5세션까지 가장 최근의 완결 세션 K′를 `fallback_K_prime`으로, 없으면 `none`으로 정합니다 |
| 4. mart | 기준일 R(K 또는 K′)로 `kr_live_prepare`를 돌립니다. export에 R 뒤 행이 있으면 `--cut-to-asof`로 R 뒤 행(가격·수급·DART 접수일)을 가립니다. 가린 결과는 R일에 내보냈다면 나왔을 입력과 같습니다(fixture로 확인). R이 K이고 뒤 행이 없으면 가리지 않아 지금까지의 경로와 같습니다 |
| 5. native | `kr_prepare`가 `prepared/kr/score_date=R/prep_id=D`를 만듭니다. `--reference-evidence`로 1\~3단계의 증거를 manifest에 넣고, 표시용 종목명을 같은 snapshot의 raw `stock_master`에서 붙입니다(모델 입력 아님) |
| 6. 보존 정리 | 지금까지와 같습니다 |

증거는 `$SERVING_ROOT/logs/kr-prepare/D=<D>/reference-selection.json`(스키마 `kr-reference-selection.v1`)에 있고 native manifest의 `reference_selection`에도 들어갑니다. `prep_id=D` 아래 어떤 `score_date`든 `completion.json`이 있으면 그날은 끝난 것으로 보고 바로 종료합니다. **한 번 K′로 끝난 날 나중에 K가 완결돼도 같은 날 다시 만들지 않습니다**(snapshot이 D 하나이기 때문입니다).

종료 코드: 0 완료·이미 완료·`locked`·휴장일, 2 사용법, 10 설정 또는 `flock` 없음, 12 달력 범위 밖, **32 `none`**(완결 세션이 없어 prepared를 만들지 않았습니다. selector가 이전 prepared를 stale로 씁니다), 143·130·129 TERM·INT·HUP으로 멈춤, 그 밖에는 단계의 코드입니다(124는 timeout). 예전의 30(게이트 시간 초과)·31(게이트 blocked)은 없어졌습니다.

### 중단 신호와 프로세스 그룹

`kr-prepare.sh`와 `us-prepare.sh`는 단계마다 하위 명령을 별도 프로세스 그룹으로 띄우고(`set -m`) 스크립트는 `wait`합니다. TERM·INT·HUP을 받으면 그 그룹 전체를 TERM, 20초 뒤 KILL로 끝내고 종료 코드 143·130·129로 나옵니다. 2026-10-06 04:29 Cronicle이 메모리 제한으로 job을 중단했을 때 wrapper만 죽고 `nice`·`taskset` 아래 Python이 빌드를 끝낸 문제를 막으려는 것입니다. `us-prepare.sh`도 같은 구조에 `locks/us-prepare.lock`을 더했습니다. `briefing-stage.sh`는 `exec`으로 Python wrapper로 바뀌어 셸 층이 없습니다. 대신 Python 쪽(`daily_wrapper`, `daily_coordinator.run_group`)이 runner·opening·publisher를 별도 프로세스 그룹으로 띄우고, timeout이나 TERM·INT·HUP에 그 그룹 전체를 끝냅니다. 서버에서 쓰는 `flock`은 util-linux 것입니다(맥에는 없어 시험은 `fcntl.flock` shim을 씁니다).

### select·run·envelope

- **KR 이전 prepared를 stale로**: select가 `score_date` ≤ K인 가장 최근 prepared를 고릅니다. K보다 이르면 `stale`이고 지연 세션 수(K와의 KR 세션 차이)를 `lag_sessions`로 적습니다. K보다 늦은 것은 쓰지 않고, 달력이 그 날짜를 몰라 지연을 셀 수 없으면 `unavailable`입니다.
- **US 지연은 표시**: `market_lag_limit_sessions`(ops의 E 표)와 도착 지연 2세션 중단은 더 이상 막지 않고 `stale`로 표시합니다. 지연은 U와의 세션 차이(`market_lag`)입니다.
- **5세션 상한(Q5)**: 지연이 5세션을 넘으면 report의 순위를 비우고 `quality.rankings_withheld`에 사유를 적습니다. 섹션 상태는 `stale`입니다(`freshness.MAX_STALE_SESSIONS`, 렌더러의 `LAG_SUPPRESS`와 같은 값).
- **자동 select**: `daily_wrapper run`이 selection이 없으면 같은 규칙으로 직접 고릅니다(D 10:00 이후일 때만, 아니면 실패). selection에는 `selected_at`, `selection_mode`(`scheduled`·`run_fallback`)를 적고, 입력별 `producer_completed_at`을 따로 둡니다(`completed_at`은 없어졌습니다). infer는 입력 완료 ≤ cutoff를 모든 모드에서 확인하고, `selected_at`이 D 10:00 뒤인 selection은 `run_fallback`일 때만 받습니다. 옛 selection(`completed_at`만 있음)은 `scheduled`로 읽습니다.
- **envelope 상태**: 전 섹션 `ok`면 `ok`, 낸 섹션(`ok`·`partial`·`stale`)이 하나도 없으면 `failed`, 나머지는 `partial`입니다. 공개 게이트와 opening은 상태에 넣지 않습니다. KR adapter가 늘 `partial`을 내므로 KR이 들어간 날은 `ok`가 나오지 않습니다(결정 필요).
- **runner timeout·예외**: infer가 이번 실행의 report를 받지 못하면(timeout, 비정상 종료, selection 실패, report 없음) coordinator가 실패 report를 만들어 render·publish로 넘깁니다. 모든 모델 섹션이 `failed`이고 `failure`에 단계·원인 클래스·timeout 여부가 있습니다. 이번 실행의 `run-state.json`도 같이 씁니다. 같은 D에 이전 실행의 report가 있으면 `report-D.superseded-<sha12>.json`으로 옮겨 두고 쓰지 않습니다. 이 경우 publisher는 내용이 달라 정정을 요구합니다(종료 30).
- report `quality`의 새 키: `selection_mode`, `selected_at`, `producer_completed_at`, `lag_sessions`, `status_before_stale`, `rankings_withheld`. KR은 `reference_verdict`, `reference_k`, `reference_date`, `reference_lag_sessions`, `k_ticker_count`, `k_ticker_ratio`, `k_previous_session`, `export_gate_verdict`, `dart_chain_ended_at`도 있고 `provenance.reference_selection`에 증거 전체가 있습니다.

### US universe revision 불일치

새 US 세션이 없는 날에도 derive-daily가 `prices_daily`를 새로 쓰고 universe incremental은 건너뛰어, 표별 최신 snapshot과 universe completion이 기록한 입력 revision이 달라졌습니다(2026-10-05 A=10-02 실측). `us_daily.prepare`는 이제 universe 입력 네 표(`prices_daily`, `listing_snapshots`, `filings_sub`, `midas_security_daily`)를 universe completion이 기록한 revision으로 고정하고 나머지 표는 최신을 씁니다. `verify_universe_completion`은 그대로입니다(고정한 snapshot의 해시가 바뀌면 계속 멈춥니다). 데이터가 같아 look-ahead는 없고, universe보다 새 prices는 universe를 다시 만들 때까지 못 씁니다. 그 경우 A′가 두 표가 같이 덮는 세션으로 내려갑니다. 근본 수정(collector derive-daily가 새 행이 없으면 새 snapshot을 쓰지 않음)은 collector 몫입니다.
