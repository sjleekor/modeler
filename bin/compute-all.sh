#!/usr/bin/env bash
set -euo pipefail

# 마트 재계산 단계 (원본 stock_data_collector/bin/parquet-compute-all.sh의 4~6단계).
#
# 원래 스크립트는 수집(1~3: db sync-remote → raw parquet export → freshness gate)과
# 모델링(4~6: derived mart 재계산 → coverage/readiness gate → feat_*/labels)을
# 이어서 돌렸다. 저장소가 갈라지며 1~3은 collector/의 일, 4~6은 이 스크립트다.
# raw parquet export까지 먼저 끝낸 뒤(collector/bin/raw-parquet-export-all.sh 등)
# 이 스크립트를 돌린다.
#
# 실제 계산은 `python -m modeler.etl.compute_all` 한 줄이다. 옵션은 그쪽 것을 그대로 쓴다.

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
app_dir="${MODELER_APP_DIR:-$(cd "$script_dir/.." && pwd)}"
cd "$app_dir"

_wants_help=0
for _arg in "$@"; do
  case "$_arg" in -h|--help) _wants_help=1 ;; esac
done
if [ "$_wants_help" -eq 0 ]; then
  : "${STOCK_DATA_ROOT:?STOCK_DATA_ROOT가 없습니다. .envrc를 확인하십시오 (direnv allow).}"
fi

if [ "$_wants_help" -eq 1 ]; then
  cat <<'EOF'
Usage: bin/compute-all.sh [options]

derived mart(stock_metric_fact/common_feature_daily_fact)를 재계산하고
freshness/coverage/readiness gate를 통과시킨다. --features를 주면
feat_*/labels 마트까지 이어서 만든다.

raw parquet export가 먼저 끝나 있어야 한다 (collector/의 일).

옵션은 modeler.etl.compute_all의 것을 그대로 전달한다:
  --snapshot-date YYYY-MM-DD
  --source NAME                 (예: local_mydb, sj2_remote)
  --from-step freshness|marts|reports|features
  --end YYYY-MM-DD
  --required-coverage-ratio R
  --threads N
  --memory-limit SIZE
  --features
  -h, --help
EOF
  exit 0
fi

exec uv run python -m modeler.etl.compute_all "$@"
