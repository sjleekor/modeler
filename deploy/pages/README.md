# GitHub Pages 전달 템플릿

이 디렉터리는 Pages 전용 저장소에 공개 site를 전달하고 검증하기 위한 검토용 템플릿입니다. 대상 저장소 `sjleekor/market-briefing`, `site` 브랜치, Pages 설정은 아직 확인하지 않았습니다. 이 파일만으로 저장소나 workflow를 활성화하지 않습니다.

서버는 `SiteBuilder`가 완성한 정적 출력 디렉터리만 `public/`로 전달합니다. 입력 JSON, 원천 데이터, 점수, 내부 provenance, 모델 파일과 실행 로그를 전달하지 않습니다. 공개 전용 `public_projection`의 전체 날짜·revision 이력을 한 번에 검증합니다. JSON 키는 공개 스키마 allowlist와 대조하고, HTML/CSS 경로는 허용 파일 목록에 있는지 확인합니다. 심볼릭 링크, 예상 밖 파일, 외부 링크, 지나치게 큰 파일, manifest에 없는 날짜와 현재 revision 누락은 거부합니다.

## 서버 wrapper 설정

운영자는 서버에서 `config.example.json`을 복사해 접근 권한이 제한된 설정 파일을 만들고, 모든 경로와 저장소 식별을 실제 환경에 맞게 확인해야 합니다. `repository_confirmed`는 전용 저장소 이름과 Pages 설정을 사람이 확인한 뒤에만 `true`로 바꾸십시오. 예제 설정은 의도적으로 `false`이며 wrapper는 이 값으로 중단합니다. `base_path`는 저장소의 Pages project path와 SiteBuilder 설정에 맞춰야 합니다.

사전 조건은 전용 저장소의 깨끗한 checkout이 `site` 브랜치에 있고, `origin`의 fetch/push URL이 확인된 저장소를 가리키며, 서버 생성기가 별도 디렉터리에 공개 projection을 완성한 상태입니다. wrapper는 checkout 잠금을 잡고, origin URL을 확인하고, 원격 `site` 이후에만 fast-forward하며, 공개 projection만 `public/` 아래에서 교체합니다. 기존 report date와 immutable revision, 해시 경로 CSS 및 model card 파일이 빠지거나 달라지면 중단합니다. `git add` 범위도 `public/`로 한정합니다. branch force push는 하지 않습니다.

commit 성공 뒤 push가 실패하면 wrapper는 checkout 바깥의 sibling journal에 remote URL, branch, parent commit, publisher commit/tree hash, `public/` tree hash와 synthetic opt-in을 기록합니다. 다음 실행은 remote tip이 기록된 parent와 같고 local HEAD가 기록된 동일 public-only commit일 때만 그 commit을 다시 push합니다. 이 재시도는 저장된 public site commit을 사용하므로 재추론이 필요하지 않습니다. journal이 없거나 다른 local ahead commit, `public/` 밖 변경, remote branch 이동, synthetic opt-in 누락이 있으면 중단합니다. commit 직후 journal 저장 전에 프로세스가 멈춰도 자동 복구하지 않고 수동 확인이 필요합니다. 성공하면 journal을 지웁니다. journal은 checkout이나 공개 projection 안에 만들지 않습니다.

일일 coordinator를 쓰는 운영에서는 이 파일을 `projection_dir`가 없는 base 설정으로 두십시오. coordinator가 날짜별로 `projection_dir`를 채운 설정을 `run_root/<D>/pages-publisher-config.json`에 만들어 publisher에 넘깁니다. 이 publisher는 완성된 설정의 키가 예제와 정확히 같아야 받습니다.

실행 명령은 명시 설정과 게시 인자를 모두 요구합니다.

```sh
python3 /path/to/publish_site.py --config /restricted/path/pages-config.json --publish
```

합성 fixture site를 실제 Pages에 공개하는 특별한 경우에는 위 명령에 `--allow-synthetic`도 붙여야 합니다. 이 선택은 공개 배포를 막지 않으므로 fixture 표기가 사이트에 남는지 먼저 확인하십시오. 평소 게시에는 이 인자를 쓰지 않습니다.

격리된 sj2-server bare Git remote fixture에서 push 실패 뒤 같은 publisher commit 재시도 성공과 journal 없는 임의 local ahead commit 거부를 검증했습니다. 두 테스트가 통과했습니다. 원격 로그는 `/home/whi/tmp/daily_market_briefing_20260929/reporting_ops/logs/pages_publisher_retry_fixture.log`이며 SHA-256은 `5c0e5479bf2508f5f86527ef49743d95561dfa018b5caf1286ec8e831f81020c`입니다. 실제 GitHub origin, credential, 공개 저장소 전송, 운영 commit/push는 확인하거나 실행하지 않았습니다. 운영 전에는 credential helper와 권한 범위를 따로 확인해야 합니다. 인증 정보는 설정 JSON이나 URL에 넣지 마십시오.

## Actions workflow template

`workflows/pages.yml.template`은 활성 workflow가 아닙니다. 대상 저장소의 코드 리뷰 경로에서 `.github/workflows/pages.yml`로 옮기기 전에 검토한 40자리 action commit SHA로 모든 `REPLACE_WITH_REVIEWED_40_HEX_COMMIT_SHA` 값을 바꾸고, `main`에 validator를 검토 반영해야 합니다. repository variable `PAGES_BASE_PATH`에는 실제 project base path를 설정합니다. workflow는 `public/`만 검사하고 Pages artifact로 배포합니다. 데이터 수집·추론·사이트 생성은 실행하지 않습니다.

push 검증은 직전 site commit과 비교해 날짜, immutable revision, versioned CSS 및 model card의 보존 여부를 확인합니다. synthetic fixture는 기본적으로 거부합니다. `workflow_dispatch`에서 `allow_synthetic`를 명시적으로 켜면 해당 fixture를 검증하고 공개 Pages에 배포합니다. 이 선택은 비공개 검토만 뜻하지 않으므로 실제 시장 자료와 혼동되지 않게 fixture 표기를 유지해야 합니다. 자동 배포 경로에서는 synthetic opt-in을 켜지 않습니다.

## 검토 상태

저장소 이름, repository visibility, Pages source/environment, action SHA, GitHub 권한, credential helper, 원천별 공개 허용 판정은 확인되지 않았습니다. 이 템플릿은 게시 승인을 뜻하지 않습니다. wrapper fixture는 격리 서버에서만 시험하며 GitHub Pages는 활성화하지 않습니다.
