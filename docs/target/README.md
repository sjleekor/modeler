# 모델 타깃 / ETL 설계 인덱스

`docs/target/`는 주가예측 모델별 **예측 대상 + ETL 설계**를 모은다. 소스는 exporter가 출력한 Parquet lake이며(PostgreSQL 직접 조회 금지), 여러 모델이 공통 ETL 계층을 공유한다.

## 공통 (모든 모델 적용)

| 문서 | 내용 |
|---|---|
| [`00_shared_etl_platform.md`](./00_shared_etl_platform.md) | **멀티모델 ETL 플랫폼** — 공통 feature mart 계층, 라벨/전처리/분할 라이브러리, 재현성 매니페스트, 새 모델 추가 체크리스트 |

## 모델 목록

| ID | 디렉토리 | 예측 대상 | 상태 |
|---|---|---|---|
| 01 | [`01_20_access_return_rank/`](./01_20_access_return_rank/) | 20영업일 시장 대비 초과수익률 랭킹 (Ridge/ElasticNet) | [구현 계획](./01_20_access_return_rank/etl_03_implementation_plan.md) 수립 · **마일스톤 A(P0~P6)+B(P7)+C(P8) 구현 완료** (`research/` 패키지: 풀 멀티모달 px+flow+fin+cf+ev, end-to-end Rank IC + 그룹 ablation) · rcept_dt PIT 정밀화만 보류 |

## 새 모델 추가 시

[`00_shared_etl_platform.md`](./00_shared_etl_platform.md) §8 체크리스트를 따른다. 요지: 새 모델 = 피처 그룹 선택 + 라벨 1개 정의 + 유니버스/분할 지정 (공통 mart·라이브러리 재사용, 중복 구현 금지).
