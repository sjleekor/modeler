# 일일 브리핑 운영 초안

이 디렉터리는 실행 인터페이스와 **비활성** Cronicle 제안입니다. `config.example.json`과 `release.example.json`의 필수 경로·hash는 아직 비어 있으므로 그대로 실행할 수 없습니다. 실제 KR/US 입력, 공개 권한, Pages 저장소가 확인되지 않아 production 일정이나 게시를 시작하지 않았습니다.

## 고정 release와 경로

검토한 소스 파일만 별도 release 디렉터리에 복사하고 hash를 고정합니다. 가변 `source/` checkout이나 전체 repository를 runtime에 직접 연결하지 않습니다. Release manifest는 세 모델의 entrypoint, bundle manifest, code inventory와 hash를 묶습니다. frozen release에는 `collector` Python package도 포함해야 합니다. `runtime-verified-sj2-20260930.json`은 sj2 **격리 시험 venv**에서 확인한 Python·패키지 버전입니다. 운영 image가 같은 구성이라는 증거는 아닙니다. `model-cards.json`의 최종 SHA-256은 `f45f4326adcdbeb6e2c8799de3c69b3e9c0c37935d1b76a2d90c5f598d09e385`이며, 운영 config의 `model_cards_path`와 `model_cards_sha256`은 이 파일의 고정 release 사본을 가리켜야 합니다. 카드의 `publication.allowed`는 자체 작성 설명문에만 적용됩니다. 순위·시세의 공개 상태는 별도 gate가 정합니다.

Serving data root 아래 `prepared/kr`, `prepared/us`에는 읽기 전용 native 산출물, `prepared/selections`에는 D별 불변 selection, `runs/`에는 report·coordinator state, `private-projections/`에는 내부 렌더 결과를 둡니다. `selection_root`는 `prepared_root` 아래에 있어야 합니다. 모델 프로세스는 collector raw 테이블을 바꾸거나 holdout label을 읽지 않습니다. 학습과 일일 추론은 분리합니다.

## Cronicle entrypoint

실제 호출은 운영 config를 고정하고 아래 세 stage를 사용합니다. `--report-date`를 생략하면 KST 현재 날짜가 D입니다. `--fixture-now`는 `synthetic_fixture=true`인 격리 release에서만 허용합니다.

```sh
python -m modeler.serving.daily_wrapper select  --config CONFIG
python -m modeler.serving.daily_wrapper run     --config CONFIG
python -m modeler.serving.daily_wrapper monitor --config CONFIG --attempt 0
```

`select`는 D 09:30에 KR 이전 완료 세션 K와 US U/E/A를 확인하고 native completion·달력·source policy를 고정합니다. 원천 E는 실제 A에서 역으로 추정하지 않습니다. US E/lag 정책이 없으면 US를 unavailable로 닫고 KR은 독립적으로 선택할 수 있습니다. KR 휴장일에는 skip 상태를 남기고 report를 만들지 않습니다. Native 완료 marker, hash, cutoff 증거가 없거나 늦으면 해당 모델을 선택하지 않습니다.

`run`은 D 10:00 이후에 저장된 selection을 확인하고 infer → render → publish 순서로 실행합니다. 세 모델 모두 unavailable이어도 해당 D의 상태 페이지를 만들고 입력 실패를 exit code 1로 알립니다. 순위 공개 권한이 `unresolved`이면 내부 report와 projection은 생성할 수 있지만 공개 단계는 `publication_withheld`입니다. KIS 개장 관측은 `modeler.serving.opening_prepare`의 별도 불변 artifact를 씁니다. 원천 시각이 확인되지 않은 관측은 장중 확인으로 올리지 않습니다.

`opening_snapshot_root`, `opening_output_root`, `opening_max_age_seconds`가 모두 설정돼 있으면 `run`이 이미 캡처한 `report_date=D/slot-*.json` 중 수신 시각이 D 10:00 이하인 snapshot만 mapper에 전달합니다. subprocess는 인자를 분리해 `shell=False`로 실행하고 timeout은 90초입니다. slot이 없거나 mapper가 실패하면 opening만 `unavailable`로 두고 KR·US 추론은 계속합니다. 예시 config에서는 이 세 값을 비워 둡니다. 실제 캡처·원천 시각·공개 권한 검증이 끝났다는 뜻은 아닙니다.

`monitor --attempt 0`, `1`, `2`, `3`은 각각 10:15, 10:17, 10:22, 10:32 KST 제안입니다. 저장된 D report와 publisher journal을 확인하고, 실패한 게시만 같은 commit으로 재시도합니다. 추론을 다시 하지 않습니다. Actions 성공과 공개 URL의 정확한 site manifest 일치가 확인돼야 `verified`입니다. 외부 HTTPS 확인은 대상 URL과 운영 네트워크 경로가 정해질 때까지 비활성입니다. HTTP client는 운영 환경의 proxy 설정을 그대로 따라야 합니다.

`cronicle-events.disabled.json`의 이벤트는 전부 `enabled=false`이며 `config_path=null`입니다. 실제 Cronicle category·target·plugin·wrapper 경로는 운영 설정에서 별도 확인해야 합니다. 이 파일은 기존 이벤트를 바꾸지 않습니다. `publisher_enabled=false`, `external_verification_enabled=false`인 예시 config도 실제 Pages 게시를 허가하지 않습니다. Actions 전체 commit SHA, 저장소·branch·Pages 설정·쓰기 권한과 공개 데이터 권리가 확인돼야 게시 경로를 열 수 있습니다.

## 격리 검증 범위

최종 모델 설명 카드가 들어간 격리 evidence는 `ops_real_adapter_e2e_20260930_final_cards/evidence.json`입니다. 실제 세 bundle을 **synthetic feature·fixture clock**으로 호출했고 세 모델 결과, 실패 0건, 전체 `partial`을 확인했습니다. `select`, `infer`, `render`가 성공했고 `publish`는 `publication_withheld`였습니다. Site도 `synthetic_fixture=true`입니다. 이 결과는 실제 최신 거래일 추론, 공개 Pages 게시, Cronicle 활성화 또는 5일 전진 운영의 증거가 아닙니다.

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

## 날짜별 게시 설정과 이전 projection

둘째 날부터 게시가 멈추지 않도록 coordinator가 날짜마다 두 값을 정합니다.

- `publisher_config`는 **base 설정**입니다. `projection_dir`가 없거나 null이어야 하고, 값이 있으면 publish가 멈춥니다. `publish` 단계가 `run_root/<D>/pages-publisher-config.json`을 원자적으로 쓰고(base 내용 + 그날 render의 `projection_dir`), publisher는 이 파일로 부릅니다. monitor 재시도도 같은 파일을 씁니다. 경로와 sha256은 `coordinator-publication.json`에 남습니다.
- `previous_projection_dir`는 운영에서 **null**로 둡니다. null이면 `render`가 `projection_root` 아래에서 이름이 `YYYY-MM-DD`이고 D보다 앞선 디렉터리 중, `site-manifest.json`의 `latest_report_date`가 이름과 같은 가장 최근 것을 고릅니다. symlink면 멈춥니다. 고른 projection의 `synthetic_fixture`가 이번 실행과 다르면 멈춥니다. 없으면(첫날) 이전 없이 render합니다. 경로를 직접 적으면 그 경로를 씁니다(시험·이관용).
- 사용한 이전 경로와 그 `site-manifest.json`의 sha256(없으면 null)은 `coordinator-render.json`의 `previous_projection_dir`, `previous_site_manifest_sha256`에 남습니다.

## 비공개 리포트 보기

순위·점수·개장 관측 값은 공개 Pages에 싣지 않습니다. 본인만 보는 비공개 화면은 설정 키 `private_projection_root`(선택, null이면 만들지 않음)에 만듭니다.

- `render` 단계가 공개 projection을 쓴 다음 `private_projection_root/<D>`에 비공개 화면을 씁니다. 이 경로가 `projection_root`나 `site_checkout`과 같거나 서로 안쪽이면 설정 검증에서 멈춥니다.
- 출력은 `index.html`(최근 날짜), `archive/index.html`(날짜별 목록), `reports/<날짜>/index.html`, `assets/private.css`, `private-manifest.json`(`private: true`), `PRIVATE_DO_NOT_PUBLISH.txt`입니다. 링크가 모두 상대 경로라 어느 디렉터리에서 열어도 됩니다.
- 모델별 상위 100개(전체 개수 표시)와 점수가 나옵니다. 점수는 순위용이고 확률이 아닙니다. 모든 페이지 맨 위에 "비공개 — 게시 금지"와 `noindex` 메타가 있습니다.
- `coordinator-render.json`의 `private_projection_dir`, `private_manifest_sha256`에 출력 경로와 manifest hash가 남습니다.
- `publish` 단계는 이 출력을 publisher에 넘기지 않습니다. 공개 검증기는 이 출력을 허용 목록 밖 파일로 보고 거부합니다. 이 디렉터리를 `site_checkout`이나 Pages 저장소에 복사하지 마십시오.

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
  --pages-config /home/whi/apps/market-briefing/config/pages-config.base.json \
  --site-checkout /home/whi/apps/market-briefing/site-checkout \
  --us-lake /home/whi/data/stock_data/us
```

이 스크립트는 lake를 읽기 전용 링크로만 걸고 `select`·`run`·`monitor`를 실행하지 않습니다. 끝에서 `daily_coordinator._config`, `daily_inputs._release_jobs`, `runtime_contract` 검증과 prepared root 구조 확인을 하고 요약 JSON을 찍습니다.

| 경로 (serving root 아래) | 내용 | 권한 |
|---|---|---|
| `venv/` | 검증된 venv의 `cp -a` 사본 (이미 있으면 python·패키지 버전만 검증) | 복사 그대로 |
| `releases/<id>/` | `release_build`로 만든 non-synthetic release. 만든 뒤 읽기 전용 | 디렉터리 0550, 파일 0440 |
| `config/` | `ops.json`(0640), `pins.json`, E 표, parity evidence, `calendars/`(KR 2026, US 2026\~2027) | 디렉터리 0750, 나머지 0440 |
| `publisher/` | `publish_site.py`, `validate_public_site.py` 사본 (검토한 manifest의 sha와 같아야 함) | 0750, 파일 0440 |
| `stock_data/us/raw`, `derived` | 운영 lake로 가는 symlink (읽기만) | |
| `stock_data/us/output/` | 실제 디렉터리. prepare가 쓰는 유일한 곳 | 0750 |
| `prepared/kr/` | 빈 디렉터리 (KR 입력이 없으면 selector가 `unavailable`로 닫음) | 0750 |
| `prepared/us` | `stock_data/us/output/us_scoring_daily_v1/prepared`로 가는 symlink | |
| `prepared/selections/` | D 선택 결과. prepared root 안에 있어야 함 | 0750 |
| `runs/`, `locks/` | coordinator 상태 | 0700 |
| `projection/`, `logs/` | 공개 projection, 로그 | 0750 |
| `private-projection/` | 비공개 리포트. 게시하지 않음 | 0700 |

권한은 모두 소유자 전용입니다. Cronicle이 같은 사용자로 돌고, 비공개 리포트와 실행 상태를 다른 계정에 열 이유가 없기 때문입니다. release를 읽기 전용으로 둔 것은 frozen 코드를 실수로 고치지 못하게 하려는 것입니다. 지우거나 바꿔야 하면 `chmod -R u+w`를 먼저 하십시오. `ops.json`만 0640이라 나중에 publisher를 켤 때 고칠 수 있습니다.

`ops.json`은 `publisher_enabled=false`, `external_verification_enabled=false`로 만듭니다. `opening_*`는 null, `private_projection_root`는 `private-projection`, `base_path`는 `/market-briefing/`입니다. `publisher_config`와 `site_checkout`은 Pages 기본 설정과 checkout을 그대로 가리키고 복사하지 않습니다. `actions_repository=sjleekor/market-briefing`, `actions_workflow=pages.yml`, `public_manifest_url=https://sjleekor.github.io/market-briefing/site-manifest.json`입니다. `pins.json`에 E 표, evidence, publisher 스크립트의 sha를 남깁니다. KR 달력은 2026-12-31까지라 2027년 전에 휴장일 CSV와 달력을 갱신해야 합니다.

### wrapper 둘

- `bin/briefing-stage.sh <select|run|monitor> [--attempt N]`: `SERVING_ROOT`(기본 `/home/whi/apps/market-briefing/serving`)의 `config/ops.json`에서 release root와 python을 읽습니다. release root에서 `PYTHONPATH=<release>/src`로 `python -m modeler.serving.daily_wrapper <stage> --config <ops.json>`을 실행합니다. `--attempt`는 `monitor`에서만 쓰고 0\~3입니다. 종료 코드는 wrapper의 것을 그대로 돌려주고, 인자 오류는 2, serving root 문제는 10입니다. 로그는 stdout·stderr로 나가 Cronicle이 받습니다.
- `bin/us-prepare.sh [--run-date YYYY-MM-DD]`: 실행일 T(KST)에 `us_expected session --report-date T+1`로 A를 구하고, `pins.json`의 evidence sha를 확인한 뒤 `STOCK_DATA_ROOT=<serving>/stock_data`, `POLARS_MAX_THREADS=2`, `taskset -c 0,1`, `timeout 1800`으로 `us_daily prepare --as-of A --raw-feature-parity-status score_equivalent --raw-feature-parity-evidence <config 사본>`을 돌립니다. 이미 같은 A가 서빙 가능 상태(`score_equivalent`, `serving_eligible`, 현재 코드 hash, evidence sha 일치)로 있으면 성공으로 건너뜁니다. 종료 코드는 10(설정), 11(evidence sha 불일치), 12(A 계산 실패), 20(A 데이터가 lake에 아직 없음: `prices_daily`·`universe_daily` 최대 날짜 < A 또는 lake를 못 읽음), 21(prepare는 끝났는데 서빙 가능한 native가 없음)이고, 그 밖에는 prepare의 코드입니다(124는 timeout). evidence 경로가 `prep_hash`에 들어가므로 config 사본 경로를 옮기지 마십시오.

### 첫 실제 실행 순서

1. 10-01 17:30 `us-prepare.sh`: A는 2026-09-30입니다. 16:30 `sdc_daily_us_universe_incremental`이 끝난 뒤여야 합니다. 2026-09-30 실측에서 운영 lake의 `universe_daily`는 2026-09-22 snapshot(최대 날짜 09-18)이고 incremental 완료 marker가 없어, prepare가 `US universe_daily requires a completed incremental membership snapshot`으로 멈춥니다. incremental이 먼저 한 번 돌아야 합니다.
2. 10-02 09:30 `briefing-stage.sh select`: D 선택이 잠깁니다. 이 시각 전에는 실행하지 마십시오.
3. 10-02 10:00 `briefing-stage.sh run`. 이어서 10:15·10:17·10:22·10:32에 `monitor`(`--attempt` 0\~3)입니다. publisher가 꺼져 있어 `publication_withheld`로 끝나는 것이 정상입니다.
