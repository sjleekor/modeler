#!/usr/bin/env bash
# research/analysis/dart_financial_statement_raw/run.sh
#
# queries/ 안의 .sql 을 순서대로 실행하고, 결과를 reports/ 아래 날짜 폴더에
# markdown(.md, 보기용)과 csv(.csv, 데이터용)로 떨군다.
# 00_setup_views.sql 은 VIEW 생성 전용이라 결과 파일을 만들지 않는다.
#
# 사용법:  bash research/analysis/dart_financial_statement_raw/run.sh
set -euo pipefail

cd "$(dirname "$0")/../../.."                                   # repo 루트로 이동
QDIR="research/analysis/dart_financial_statement_raw/queries"
DB="reports/analysis/dart_fs.duckdb"                            # 영구 DB (VIEW 보존). reports/ 는 gitignore.
OUT="reports/analysis/dart_financial_statement_raw/$(date +%F)"
mkdir -p "$(dirname "$DB")" "$OUT"

# VIEW 정의 (parquet glob → fs_raw). 매 실행마다 최신 정의로 갱신.
duckdb "$DB" < "$QDIR/00_setup_views.sql"

for q in "$QDIR"/[0-9]*.sql; do
    name=$(basename "$q" .sql)
    case "$name" in 00_setup_views|00b_setup_views_abs) continue ;; esac
    duckdb "$DB" -markdown < "$q" > "$OUT/$name.md"
    duckdb "$DB" -csv      < "$q" > "$OUT/$name.csv"
    echo "✓ $name → $OUT/$name.{md,csv}"
done

echo
echo "결과: $OUT"
echo "인터랙티브 탐색:  duckdb $DB   (fs_raw VIEW 가 이미 생성되어 있음)"
