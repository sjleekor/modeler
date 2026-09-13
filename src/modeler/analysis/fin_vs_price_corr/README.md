# 재무지표 vs 주가 forward return 상관분석

로컬 전용 EDA 작업입니다. 배포 파이프라인에는 포함하지 않고, 스냅샷 parquet를 읽어 재무 PIT 마트와 forward-return label을 결합합니다.

## 실행

```bash
uv run python research/analysis/fin_vs_price_corr/run_fin_price_correlation.py \
  --snapshot-date 2026-06-19 \
  --report-date 20260627
```

입력은 `data_lake/.../snapshot_date=2026-06-19`이고, 출력은 `reports/analysis/fin_vs_price_corr/20260627/`에 생성됩니다. `label_daily` 마트가 없으면 `daily_ohlcv`에서 한 번 생성합니다.

## 파일

- `queries/00_setup_views.sql`: DuckDB parquet view 등록.
- `queries/01_pit_join_marts.sql`: 경로 A, `feat_fin_pit` x `label_daily` 결합.
- `queries/02_pit_join_raw_account.sql`: 경로 B, raw DART 계정 PIT 정렬 템플릿.
- `run_fin_price_correlation.py`: heatmap, 상관표, 산점도, 분위수, 단순 회귀, 그룹 안정성 산출.
- `notebooks/01_fin_price_correlation.ipynb`: 동일 스크립트 실행 및 결과 확인용 노트북.

## 결과 요약

2026-06-19 스냅샷으로 실행 완료했습니다.

- 분석 base: 5,024,637행, 2,606종목, 2015-04-14 ~ 2026-05-20. 종료일이 스냅샷일보다 이른 것은 20거래일 forward label 생성 때문입니다.
- primary target: `raw_label_20d`(시장 대비 20거래일 초과수익).
- `raw_label_20d` 기준 최상위 비율 지표: `fin_cash_ratio`(Spearman -0.0340, Pearson -0.0002), `fin_equity_ratio`(Spearman -0.0300), `fin_debt_to_equity`(Spearman -0.0281).
- 상위 3개 지표의 단순 선형회귀 R^2 최대값은 0.000016입니다.

결론: 현재 스냅샷에서 표준 재무비율 단독으로는 forward excess return에 대한 선형 설명력이 매우 약합니다. Spearman 기준으로 약한 순위 신호와 분위수별 구간 차이는 보이지만 단조성이 안정적이지 않아, 바로 모델 feature로 승격하기보다는 연도/시장/섹터/시총 통제 후 재검증이 필요합니다.
