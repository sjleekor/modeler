# ETL 설계 — Ridge / ElasticNet 학습·테스트 데이터셋

> ⚠️ **데이터 소스 (필독)**: 본 문서의 SQL은 **PostgreSQL을 직접 조회하지 않는다.** 실제 ETL은 exporter가 출력한 **Parquet lake(`data_lake/`)** 를 DuckDB/Polars로 읽으며, 아래 SQL은 그 Parquet 위에서 실행할 **논리 명세**다. 데이터 흐름·접근 원칙·Parquet에 없는 테이블(재무·공통 피처)의 exporter 확장은 [`etl_01_parquet_data_flow_plan.md`](./etl_01_parquet_data_flow_plan.md) §0.5(불변 규칙)·§2·§5를 따른다. (`DISTINCT ON` 등 PG 전용 문법은 DuckDB 등가물로 이식.)

- 작성 일시: 2026-06-18
- 위치: `docs/target/01_20_access_return_rank/`
- 예측 대상: [`prediction_target_20d_excess_return_rank.md`](./prediction_target_20d_excess_return_rank.md)
  — **20영업일 시장 대비 초과수익률 랭킹**(메인), 5d/60d 초과수익률·20d 변동성(보조)
- 피처 근거: [`docs/features/table_stat_20260528/feature_profile_summary_for_model_selection.md`](../../features/table_stat_20260528/feature_profile_summary_for_model_selection.md)
- 모델: **Ridge / ElasticNet** (선형, L2 / L1+L2 정규화)

---

## 0. 모델 특성이 ETL에 거는 제약 (가장 먼저)

Ridge/ElasticNet은 선형 회귀 + 정규화 모델이다. 트리계열과 달리 ETL이 **반드시** 보장해야 하는 전처리 요구가 있다. 본 설계의 모든 결정은 이 5가지에서 출발한다.

| # | 선형모델 특성 | ETL이 해야 할 일 |
|---|---|---|
| L1 | **결측(NaN)을 직접 못 먹는다** | 모든 피처를 결측 없는 실수 행렬로 만들어야 함 → impute + `*_isna` 플래그 동반 |
| L2 | **피처 스케일에 민감** (정규화 패널티가 스케일에 비례) | 피처를 표준화해야 함 → **일자별(cross-sectional) z-score** 권장 |
| L3 | **이상치(fat-tail)에 매우 취약** | winsorize + log/signed-log 변환 필수 (시세 가격·거래대금·재무 금액·수급 극단치) |
| L4 | **다중공선성에 약함** (Ridge는 완화, ElasticNet L1이 선택) | 강상관 피처군 정리 + ElasticNet에 변수선택 위임, 상관 진단 산출물 포함 |
| L5 | **비선형·상호작용을 못 잡는다** | 비율·스프레드·모멘텀 등 **도메인 파생 피처를 ETL 단에서 미리 생성** |

추가 공통 제약:

- **PIT(Point-in-Time) 누설 금지**: 모든 피처는 예측 기준일 `t` 이전 정보만 사용. 재무는 공시 시점, 공통 피처는 `asof_available_date` 기준.
- **라벨 누설 금지**: 라벨은 `t+20`까지의 미래를 쓰므로, 학습/검증 분할 시 **embargo(=label horizon)** 을 둔다.
- **일자별 표준화·랭킹**: 라벨과 피처 모두 "같은 날짜 종목 횡단" 기준으로 정규화 → regime/시장방향 착시 제거.

---

## 1. 데이터 범위 / 종목 유니버스

### 1.1 기간

```text
학습 시작: 2015-01-02   (krx_security_flow_raw ↔ daily_ohlcv join 100% 구간)
전체 종료: 2026-06-10   (sync_checkpoints cursor = 로컬 최신 거래일)
```

- 2007~2013은 `daily_ohlcv`가 단일 종목이라 횡단 학습 불가 → 제외.
- 공통 피처(`common_feature_*`)는 일별 raw가 **2025-12-15**부터라 2015~2026 학습 패널 대부분에서 결측 → **1차 모델에서는 공통 피처를 핵심 피처로 쓰지 않고**, 가용 구간에서만 "시장 국면 보정 변수"로 선택 결합(§4.5).

### 1.2 유니버스 (거래일 t 기준 PIT)

`stock_master_snapshot` + `stock_master_snapshot_items`로 PIT universe를 잡는 것이 이상적이나, 스냅샷 이력이 2026-04~06에만 존재(21개, 부분 스냅샷 1건 포함). 따라서 1차에서는 **`daily_ohlcv` 자체를 유니버스**로 사용하고 아래 필터를 적용한다.

거래일 `t`에 종목이 학습 대상이 되려면:

```text
1) daily_ohlcv에 t 행 존재 AND is_halted(t) = false      -- 거래정지일 제외
2) 최근 60거래일 중 유효 거래일 >= 40                       -- 신규상장/장기정지 제외 (warm-up)
3) t 시점 60일 평균 거래대금 >= 하한(예: 1억원)             -- 초저유동성 제외 (선형모델 노이즈 방지)
4) t+20 거래일 close 존재                                   -- 라벨 산출 가능 (학습셋에 한함)
```

- `is_halted := (open=0 AND high=0 AND low=0)` — 프로파일에서 확인된 pykrx 정지일 규약(1.6%).
- 조건 4는 **학습/검증 셋에만** 적용. 운영 추론 시에는 미래가 없으므로 적용하지 않음.

---

## 2. 라벨 생성 (target ETL)

### 2.1 시장 벤치마크 수익률

별도 시장지수 테이블 의존을 피하고, **유니버스 내 시장별 동일가중 평균수익률**을 벤치마크로 사용한다(robust, PIT-safe, 데이터 자급).

```sql
-- 종목별 20영업일 forward 수익률 (거래일 인덱스 기준)
WITH px AS (
  SELECT trade_date, ticker, market, close,
         ROW_NUMBER() OVER (PARTITION BY ticker, market ORDER BY trade_date) AS d_idx
  FROM daily_ohlcv
  WHERE NOT (open=0 AND high=0 AND low=0)        -- 정지일 close 왜곡 방지
),
fwd AS (
  SELECT a.trade_date, a.ticker, a.market,
         a.close AS close_t,
         f.close AS close_t20,
         f.close::numeric / NULLIF(a.close,0) - 1 AS fwd_ret_20d
  FROM px a
  JOIN px f
    ON f.ticker = a.ticker AND f.market = a.market
   AND f.d_idx  = a.d_idx + 20                    -- 정확히 20거래일 후
),
mkt AS (   -- 시장별 동일가중 벤치마크 (해당 일자 횡단 평균)
  SELECT trade_date, market, AVG(fwd_ret_20d) AS mkt_ret_20d
  FROM fwd GROUP BY trade_date, market
)
SELECT f.trade_date, f.ticker, f.market,
       f.fwd_ret_20d,
       m.mkt_ret_20d,
       f.fwd_ret_20d - m.mkt_ret_20d AS excess_ret_20d
FROM fwd f JOIN mkt m USING (trade_date, market);
```

> 거래일 인덱스(`d_idx`)로 `t+20`을 잡아 캘린더 결측/정지일에도 정확히 20거래일을 보장한다. 시장지수 종가가 별도로 확보되면 `mkt_ret_20d`를 KOSPI/KOSDAQ 지수 수익률로 교체 가능(설계상 1줄 교체).

### 2.2 라벨 3종 (회귀 / 랭킹 / 분류)

```text
A. 회귀:   y_reg_20d   = winsorize_by_date(excess_ret_20d, p=[0.005, 0.995])
B. 랭킹:   y_rank_20d  = percent_rank(excess_ret_20d) WITHIN 같은 (trade_date, market)   ∈ [0,1]
C. 분류:   y_cls_20d   = 1 if y_rank_20d>=0.8 ; -1 if y_rank_20d<=0.2 ; else 0
```

- **Ridge/ElasticNet 메인 타깃 = `y_rank_20d`** (또는 일자별 z-score한 `y_reg_20d`). 랭킹/표준화 타깃이 선형모델에서 regime에 robust.
- 회귀 학습 시에도 타깃을 **일자별로 표준화**(zscore_by_date)하면 날짜 간 스케일 차이를 제거할 수 있음.

### 2.3 보조 라벨

```text
y_rank_5d   : 위와 동일하되 d_idx+5
y_rank_60d  : 위와 동일하되 d_idx+60
y_vol_20d   : t+1..t+20 일간수익률의 표준편차 (realized volatility)  → 별도 회귀
y_mdd_20d   : t+1..t+20 누적수익 경로의 최대낙폭                      → 별도 회귀
```

각 horizon은 독립 데이터셋으로 산출하되 동일 피처 행렬을 공유한다.

---

## 3. 피처 ETL — 소스별 설계

모든 피처는 거래일 `t`(=`as_of_date`)에 정렬되며, `(t, ticker, market)`이 피처 행의 그레인이다.

### 3.1 시세 피처 (`daily_ohlcv`) — 항상 가용

가격 레벨이 아니라 **수익률/모멘텀/변동성/유동성**으로 변환(L3, L5).

```text
ret_1d, ret_5d, ret_20d, ret_60d        : close 기반 누적 수익률 (log-return 권장)
mom_20_60                                : ret_20d - ret_60d (모멘텀 가속)
vol_20d, vol_60d                         : 일간수익률 표준편차
high_low_range_20d                       : (max(high)-min(low))/close 변동폭
turnover = close * volume                : numeric 캐스팅(오버플로 방지), log1p
turnover_z_20d                           : turnover의 20일 z-score (유동성 시그널)
amihud_20d                               : |ret_1d| / turnover 평균 (비유동성)
gap_vs_ma20 = close/ma20 - 1             : 이동평균 대비 위치
dist_52w_high = close/max(close,252) - 1 : 신고가 거리
is_halted, halt_ratio_20d                : 정지일 플래그/비율
```

- 가격·거래대금은 **per-date winsorize → log/signed-log → 일자별 z-score** 순으로 정규화(L2/L3).
- warm-up: `ret_60d`/`vol_60d`는 60거래일 이력 필요 → 유니버스 조건 2와 정합.

### 3.2 수급/공매도 피처 (`krx_security_flow_raw`) — 핵심

먼저 **KRX 우선 dedupe**(KRX/PYKRX 중복분 제거, 값 충돌 0).

> **수치(lake 실측, 2026-06-19)**: raw 76.5M행 = KRX 55,908,238 + PYKRX 20,628,334. `(trade_date,ticker,market,metric_code)` 기준 KRX 우선 dedupe 후 **55,918,702 distinct**([`etl_01_parquet_data_flow_plan.md`](./etl_01_parquet_data_flow_plan.md) §4.2 / [`etl_02_engine_comparison.md`](./etl_02_engine_comparison.md) §3 Q1). 과거 요약(feature profile)의 "중복 ~19.9M / distinct ~49.8M"은 백필 전 추정이며, 위 실측이 authoritative.

```sql
WITH flow_dedup AS (
  SELECT DISTINCT ON (trade_date, ticker, market, metric_code)
         trade_date, ticker, market, metric_code, value
  FROM krx_security_flow_raw
  ORDER BY trade_date, ticker, market, metric_code,
           CASE source WHEN 'KRX' THEN 0 ELSE 1 END     -- KRX 우선
)
SELECT * FROM flow_dedup;
```

metric_code 7종을 wide pivot 후 파생:

```text
-- 순매수(개인/기관/외국인): 누적·z-score (절대 주식수 아님 → 비율/표준화)
foreign_netbuy_sum_5d, _20d              : 외국인 순매수 누적
inst_netbuy_sum_5d, _20d                 : 기관 순매수 누적
indiv_netbuy_sum_5d, _20d                : 개인 순매수 누적
netbuy_z_20d (각 주체)                    : 거래량 대비 표준화 순매수
-- 외국인 보유: 변화율 (레벨 아님)
foreign_holding_chg_5d, _20d             : 외국인 지분 변화
-- 공매도: 비중·비율 (레벨 극단치 회피, L3)
short_volume_ratio = short_selling_volume / NULLIF(volume,0)
short_value_ratio  = short_selling_value  / NULLIF(close*volume,0)
short_balance_chg_20d                    : 대차잔고 변화
short_intensity_z_20d                    : short_volume_ratio의 20일 z-score
```

- 커버리지 비대칭 주의: 공매도 잔고는 **2016-06-30~**, 종목 커버리지 최저 → 해당 피처는 결측이 많음 → impute+`*_isna` 플래그(L1) 필수.
- 순매수 3주체 합은 0이 아님(기타법인 미포함) → 합계 항등식으로 파생 피처 만들지 말 것.

### 3.3 재무 피처 (`stock_metric_fact`) — PIT 결합

`stock_metric_fact`는 `(ticker, metric_code, bsns_year, reprt_code)` 그레인, `period_end` 보유. **공시일 컬럼이 없으므로** 보수적 PIT 래그를 적용한다.

```text
available_from := period_end + INTERVAL '90 days'    -- 사업보고서 공시 지연 보수적 가정
```

> 정확도를 높이려면 `dart_financial_statement_raw.rcept_no` → 접수일(`rcept_dt`)을 조인해 실제 공시일을 쓰는 것이 이상적(후속 개선). 1차는 +90일(분기보고서는 +45일) 보수 래그로 시작.

거래일 `t`에는 **`available_from <= t`인 가장 최근 보고서**의 값만 사용(as-of join):

```sql
-- 종목 t 시점에 가용한 최신 metric value (PIT as-of)
SELECT DISTINCT ON (u.trade_date, u.ticker, f.metric_code)
       u.trade_date, u.ticker, f.metric_code, f.value_numeric
FROM universe u
JOIN stock_metric_fact f
  ON f.ticker = u.ticker
 AND (f.period_end + INTERVAL '90 days') <= u.trade_date
ORDER BY u.trade_date, u.ticker, f.metric_code, f.period_end DESC;
```

파생(레벨 금액 직접 사용 금지 → 비율·성장률, L3/L5):

```text
-- core (커버리지 2,100+ 종목, 안정): BS/CF/shares 기반
roa            = net_income / total_assets         (가용 시)
debt_to_equity = total_liabilities / NULLIF(total_equity,0)
equity_ratio   = total_equity / total_assets
ocf_to_assets  = operating_cash_flow / total_assets
cash_ratio     = cash_and_cash_equivalents / total_assets
asset_growth_yoy = total_assets / lag(total_assets, 1yr) - 1
-- 밸류(시세 결합): market_cap = close * issued_shares
per_proxy, pbr_proxy = market_cap / (net_income | total_equity)
-- IS 5종(revenue/cogs/operating_income/net_income/sga): 매핑 희소(120~200종목)
--   → 결측 많음. impute+isna 플래그로만 포함하거나 1차에서 제외 옵션
```

- 자본잠식 12종: `total_equity<=0` → 비율 무한대/음수 → **clip + `is_negative_equity` 플래그**.
- `has_fs` 플래그: FS 없는 ~499 corp(SC/SR만 존재) 구분.

### 3.4 이벤트 피처 (`dart_shareholder_return_raw`, `dart_share_count_raw`) — 선택(2차)

희소·비표준(`stock_knd` 50+, `se` 139종)이라 1차에서는 **저빈도 요약 피처**만:

```text
has_dividend_flag, dividend_yield_proxy   (배당)
treasury_buy_flag_1y                      (자기주식 취득 여부)
shares_outstanding_chg_yoy                (발행주식수 변화 — se='합계' 행 앵커)
```

정규화 사전(보통주/우선주, se canonical 매핑) 정비 전까지는 플래그 위주로만 사용.

### 3.5 시장·거시 공통 피처 (`common_feature_daily_fact`) — 보조(국면 보정)

`(feature_date, feature_code)` 그레인, **`asof_available_date`로 PIT 보장**. 단 일별 이력 2025-12-15~ → 2015~2026 학습 구간 대부분 결측.

```text
1차: 학습 패널 전체에 broadcast하되, 결측 구간은 0 + cf_isna 플래그.
     실질적으로는 2025-12-15 이후 검증 fold에서만 신호.
사용 피처(가용 구간): kospi_ret_20d, kosdaq_ret_20d, rate_kr_gov3y/10y level,
     term_spread, usdkrw_ret, vix류 global_risk level
결합 키: feature_date = t (종목 무관, 전 종목 동일값 broadcast)
PIT: WHERE asof_available_date <= t 인 행만 (룩어헤드 0 검증됨)
```

- 1차 모델 성능에는 거의 기여 못 함(이력 짧음) → **장기 백필 후 3차에서 본격 사용**. 설계상 조인 경로만 확보.

---

## 4. 피처 행렬 조립 & 선형모델 전처리

### 4.1 조립 (as-of join 파이프라인)

```text
base = universe(t, ticker, market)                       -- §1.2
   ├─ LEFT JOIN price_features      ON (t, ticker, market)
   ├─ LEFT JOIN flow_features       ON (t, ticker, market)   -- dedup 후
   ├─ LEFT JOIN fin_features        ON (t, ticker) PIT as-of  -- §3.3
   ├─ LEFT JOIN event_features      ON (t, ticker) PIT as-of  -- 선택
   ├─ LEFT JOIN common_features     ON (feature_date=t)       -- broadcast, 보조
   └─ LEFT JOIN labels(y_*)         ON (t, ticker, market)    -- 학습셋만
```

### 4.2 결측 처리 (L1)

```text
1) 각 수치 피처 x 에 대해 x_isna = (x IS NULL) 플래그 컬럼 생성
2) impute:
   - 비율/모멘텀/수급 z-score → 0 (중립값)
   - 레벨성 피처 → 일자별 cross-sectional median
3) impute 후 NaN/Inf 잔존 0 보장 (선형모델 학습 전 assert)
```

### 4.3 이상치·스케일 (L2, L3) — **반드시 일자별(per-date)로**

```text
순서: per-date winsorize([0.01, 0.99])  →  log/signed-log(금액·거래대금·수급)
      →  per-date z-score (cross-sectional standardize)
```

- **per-date 표준화가 핵심**: 같은 거래일 종목들 사이의 상대 위치만 학습 → 시장 전체 등락(regime) 제거, 라벨(초과수익률 랭킹)과 정합. 전체기간 글로벌 스케일러는 사용 금지.
- 표준화 통계량(mean/std)은 **학습 fold에서만 fit**, 검증/테스트엔 transform만(룩어헤드 방지). 단 per-date 표준화는 같은 날 내부 통계라 fold 누설이 적음 → 양쪽 다 허용 가능하되 fit 범위 문서화.

### 4.4 다중공선성 (L4)

```text
- 학습 fold에서 |corr|>0.95 피처쌍 진단 리포트 산출 → 중복 피처 1개 제거
- Ridge: 공선성 완화(계수 분산↓). ElasticNet: l1_ratio로 변수선택 위임
- VIF 상위 피처 로깅 (산출물)
```

### 4.5 최종 피처 그룹 토글

```text
F_price  : §3.1   (항상)
F_flow   : §3.2   (항상, 핵심)
F_fin    : §3.3   (2차부터 / core 비율 우선, IS 5종 토글)
F_event  : §3.4   (선택)
F_common : §3.5   (보조, 가용 구간만)
```

ablation을 위해 그룹 단위 on/off 가능하게 컬럼 prefix(`px_`, `flow_`, `fin_`, `ev_`, `cf_`)로 네이밍.

---

## 5. 학습 / 검증 분할 (Walk-Forward + Embargo)

라벨이 `t+20` 미래를 보므로 시계열 누설 방지가 필수.

```text
Purged Walk-Forward (expanding window):
  fold k:  train = [start, T_k]            검증 = [T_k + embargo + 1, T_k + valid_len]
  embargo  = 20 거래일 (= label horizon)   -- train 끝과 valid 시작 사이 공백
  purge    = train 끝 직전 20거래일 라벨 제거 (라벨이 valid 구간을 침범하므로)

예시 (연 단위 expanding):
  fold1: train 2015~2018  | embargo 20d | valid 2019
  fold2: train 2015~2019  | embargo 20d | valid 2020
  ...
  foldN: train 2015~2024  | embargo 20d | valid 2025
  holdout(최종 테스트): 2026-01 ~ 2026-06-10 (모델 선정 후 1회만)
```

- 하이퍼파라미터(`alpha`, `l1_ratio`)는 walk-forward 검증 평균 지표로 선택.
- 표준화/winsorize/impute 통계량은 **각 fold의 train 구간에서만 fit**.

---

## 6. 평가 지표 (선형회귀 RMSE 아님 — 랭킹 중심)

```text
주지표:  Rank IC (Spearman, 일자별 산출 후 평균)
         Top-decile mean excess return
         Top-minus-bottom spread (Q5 - Q1)
보조:    Hit ratio of top 20%
         ICIR = mean(Rank IC) / std(Rank IC)
전략성:  Portfolio turnover, strategy MDD, Sharpe (상위 분위 롱)
회귀참고: per-date 표준화 타깃 RMSE / R² (모델 sanity check용)
```

일자별 Rank IC 분포(평균·표준편차·t-stat)를 fold별로 리포트한다.

---

## 7. 산출물 (테이블 / 파일)

ETL 결과는 재현 가능한 중간 테이블 또는 parquet로 물질화한다.

```text
feat_daily_panel        (t, ticker, market, 모든 피처 + *_isna)        -- 표준화 전 raw 피처
feat_daily_panel_std    (t, ticker, market, per-date 표준화 피처)       -- 모델 입력
label_daily             (t, ticker, market, y_reg_20d, y_rank_20d, y_cls_20d, y_*_5d/60d, y_vol_20d, y_mdd_20d)
universe_daily          (t, ticker, market, 필터 통과 플래그)
split_folds             (fold_id, role, date_start, date_end)
corr_report / vif_report (피처 진단, fold별)
```

- 위치 제안: `data/datasets/01_20_access_return_rank/` (parquet). 단 실제 소스/엔진/계층 구조는 `etl_01`(Parquet lake)·`00_shared_etl_platform.md`(mart/dataset 분리)를 따른다.
- 컬럼 명세·dtype·결측규칙을 `feature_dictionary.md`로 별도 관리.

---

## 8. 실행 파이프라인 요약

```text
Step 1  universe_daily   생성   (§1.2)
Step 2  label_daily      생성   (§2: 거래일 인덱스 forward join, winsor/rank)
Step 3  price_features   생성   (§3.1)
Step 4  flow_features    생성   (§3.2: KRX dedupe → pivot → 파생)
Step 5  fin_features     생성   (§3.3: period_end+90d PIT as-of)
Step 6  (event/common 선택) 생성 (§3.4, §3.5)
Step 7  feat_daily_panel 조립   (§4.1 as-of join + §4.2 impute/isna)
Step 8  per-date winsor/log/zscore → feat_daily_panel_std  (§4.3, fold-aware)
Step 9  split_folds      생성   (§5)
Step 10 Ridge/ElasticNet 학습·검증 (§6 지표), 하이퍼파라미터 선택
Step 11 holdout 1회 평가 → 모델 확정
```

---

## 9. 1차 구현 시 주의 체크리스트

- [ ] `t+20`는 캘린더가 아니라 **거래일 인덱스** 기준인가 (정지일·휴일 흡수)
- [ ] 라벨/피처 모두 **일자별 횡단** 처리인가 (글로벌 통계 미사용)
- [ ] `krx_security_flow_raw` **KRX 우선 dedupe** 적용했나
- [ ] 재무 피처 **PIT 래그(+90d/+45d)** 또는 실제 공시일 조인했나
- [ ] 모든 모델 입력 컬럼 **NaN/Inf 0** 보장 + `*_isna` 동반했나
- [ ] winsorize → log → zscore 통계량을 **fold train에서만 fit** 했나
- [ ] walk-forward **embargo/purge = 20거래일** 적용했나
- [ ] 평가는 **Rank IC / Top-decile**가 주지표인가 (RMSE는 보조)
- [ ] 자본잠식·초저유동성·정지일 종목 처리 규칙 적용했나
- [ ] 공통 피처는 가용 구간 외 **결측+플래그**로만 들어갔나 (1차 비핵심)
