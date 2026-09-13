# access_return_rank — 매수 후보 Top-k 선정 결과

> 피쳐 조합: **px+flow+fin** / 모델: hgb (max_iter=200, learning_rate=0.01)
> k=100 / 데이터셋: `/Users/whishaw/wss_p/stock_data_collector/data/datasets/01_20_access_return_rank/snapshot_date=2026-06-19`

## 1. 모델 요약

- walk-forward valid mean Rank IC: **0.1394**
- holdout Rank IC: **0.0689** (ICIR 0.453, top-decile spread 0.0325)

## 2. holdout 백테스트 (per-date Top-k)

- holdout 일자 수: 120, 출력: `data/predictions/01_20_access_return_rank/holdout/topk_holdout.csv`
- 라벨 확정 구간 Top-100 실현 20일 초과수익 평균: **0.1282** (적중률 0.341, 107일 / 10528행)

## 3. 최신일 매수 후보 (2026-06-10, 라벨 미확정)

- 출력: `data/predictions/01_20_access_return_rank/latest/topk_2026-06-10.csv` (총 100종목)

| rank | ticker | market | 종목명 | 시총(억) | 매출(억) | 영업이익(억) | pred |
|---|---|---|---|---|---|---|---|
| 1 | 290670 | KOSDAQ | 대보마그네틱 | 492 | - | - | 0.6041 |
| 2 | 175140 | KOSDAQ | 휴먼테크놀로지 | 571 | - | - | 0.6031 |
| 3 | 006050 | KOSDAQ | 국영지앤엠 | 238 | - | - | 0.6014 |
| 4 | 300120 | KOSDAQ | 라온피플 | 123 | - | - | 0.6008 |
| 5 | 455180 | KOSDAQ | 케이지에이 | 234 | - | - | 0.6007 |
| 6 | 458350 | KOSDAQ | 에스팀 | 311 | - | - | 0.5991 |
| 7 | 084180 | KOSDAQ | 수성웹툰 | 115 | - | - | 0.5987 |
| 8 | 191410 | KOSDAQ | 육일씨엔에쓰 | 173 | - | - | 0.5977 |
| 9 | 348080 | KOSDAQ | 큐라티스 | 344 | - | - | 0.5975 |
| 10 | 407400 | KOSDAQ | 꿈비 | 346 | - | - | 0.5947 |
| 11 | 148780 | KOSDAQ | 비큐AI | 199 | - | - | 0.5926 |
| 12 | 101000 | KOSDAQ | KS인더스트리 | 177 | 61 | -11 | 0.5913 |
| 13 | 377220 | KOSDAQ | 프롬바이오 | 193 | - | - | 0.5888 |
| 14 | 435570 | KOSDAQ | 에르코스 | 474 | - | - | 0.5887 |
| 15 | 101970 | KOSDAQ | 우양에이치씨 | 1,268 | - | - | 0.5882 |
| 16 | 288620 | KOSDAQ | 에스프리즘 | 479 | - | - | 0.5879 |
| 17 | 115310 | KOSDAQ | 인포바인 | 244 | - | - | 0.5875 |
| 18 | 291650 | KOSDAQ | 압타머사이언스 | 350 | - | - | 0.5874 |
| 19 | 464500 | KOSDAQ | 아이언디바이스 | 302 | - | - | 0.5860 |
| 20 | 215090 | KOSDAQ | 솔디펜스 | 378 | - | - | 0.5859 |

> 상위 20종목만 표기. 전체 목록은 위 CSV/Parquet 참조.
