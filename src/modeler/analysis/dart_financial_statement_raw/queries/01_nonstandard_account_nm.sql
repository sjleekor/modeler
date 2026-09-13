-- 01_nonstandard_account_nm.sql
-- C9 후속: account_id = '-표준계정코드 미사용-' 인 행들이 실제로 어떤 계정(account_nm)인지.
-- 프로파일 리포트의 C9 는 5% TABLESAMPLE 이었으나, 아래는 parquet 전수 집계다.
--
-- ============================================================
-- fs_raw 컬럼 목록 (dart_financial_statement_raw parquet 기준)
-- ============================================================
--  컬럼명               타입                      설명
-- ------------------------------------------------------------
--  raw_id               bigint                    PK (PostgreSQL 시퀀스)
--  corp_code            varchar                   OpenDART 고유 기업코드 (8자리)
--  ticker               varchar                   종목코드 (상장사만 존재, 비상장은 NULL)
--  bsns_year            bigint                    사업연도 (예: 2023) ※ hive 파티션 컬럼
--  reprt_code           bigint                    보고서 코드 (11011=사업보고서, 11012=반기, 11013=1분기, 11014=3분기) ※ hive 파티션 컬럼
--  fs_div               varchar                   재무제표 구분 (OFS=개별, CFS=연결)
--  sj_div               varchar                   재무제표 종류 코드 (BS=재무상태표, IS=손익계산서, CIS=포괄손익, CF=현금흐름, SCE=자본변동)
--  sj_nm                varchar                   재무제표 종류명 (한글, 예: '재무상태표')
--  account_id           varchar                   계정과목 표준코드 (미사용 시 '-표준계정코드 미사용-')
--  account_nm           varchar                   계정과목명 (한글, 예: '유동자산')
--  account_detail       varchar                   계정과목 세부 구분 (일부 행만 존재)
--  thstrm_nm            varchar                   당기 기간명 (예: '제 55 기 3분기')
--  thstrm_amount        decimal                   당기 금액 (원 단위)
--  thstrm_add_amount    decimal                   당기 누적 금액 (분기 보고서의 연초 누적)
--  frmtrm_nm            varchar                   전기 기간명
--  frmtrm_amount        decimal                   전기 금액
--  frmtrm_q_nm          varchar                   전기 동일 분기 기간명
--  frmtrm_q_amount      decimal                   전기 동일 분기 금액
--  frmtrm_add_amount    decimal                   전기 누적 금액
--  bfefrmtrm_nm         varchar                   전전기 기간명
--  bfefrmtrm_amount     decimal                   전전기 금액
--  ord                  bigint                    재무제표 내 계정 표시 순서
--  currency             varchar                   통화 코드 (예: 'KRW', 'USD')
--  rcept_no             varchar                   OpenDART 접수번호 (14자리)
--  source               varchar                   데이터 출처 식별자 (hive 파티션 컬럼)
--  fetched_at           timestamptz               OpenDART API 수집 일시 (KST)
--  raw_payload          varchar                   OpenDART API 원본 JSON 응답 전문
--  schema_version       bigint                    raw 테이블 스키마 버전
--  snapshot_date        date                      parquet 스냅샷 생성일 (hive 파티션 컬럼)
-- ============================================================
SELECT account_nm,
       sj_div,
       COUNT(*)              AS row_cnt,
    COUNT(DISTINCT corp_code) AS corps,
    MIN(bsns_year)            AS first_year,
    MAX(bsns_year)            AS last_year
FROM fs_raw
WHERE account_id = '-표준계정코드 미사용-'
GROUP BY account_nm, sj_div
ORDER BY row_cnt DESC
FETCH FIRST 50 ROWS ONLY;



