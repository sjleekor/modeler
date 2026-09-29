"""시장·섹터 레이어 2 설정: 사전등록 후보 값을 한 곳에 모은다.

**이 값들은 첫 실제 평가 전에 사용자가 동결한다. 튜닝하지 않는다.** ``run.py``가 전체
설정과 ``config_hash()``를 manifest에 남긴다. 값을 바꾸면 해시가 바뀌므로 다른 실험이다.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import date
from typing import Any


def _train_start() -> dict[str, date]:
    return {"US": date(2011, 1, 3), "KR": date(2010, 1, 4)}


def _first_test_year() -> dict[str, int]:
    return {"US": 2016, "KR": 2015}


@dataclass(frozen=True)
class MsConfig:
    # 라벨
    horizon_sessions: int = 60
    loss_threshold: float = -0.08
    # 피쳐 warm-up: 자산 패널의 처음 이만큼의 세션은 피쳐가 안 채워져 학습·평가에서 뺀다.
    max_lookback_sessions: int = 252
    # 분할 (last_test_year는 마지막으로 만기된 라벨의 연도로 데이터에서 정한다)
    train_start: dict[str, date] = field(default_factory=_train_start)
    first_test_year: dict[str, int] = field(default_factory=_first_test_year)
    # 날짜 블록 부트스트랩
    block_sessions: int = 60
    bootstrap_resamples: int = 1000
    bootstrap_seed: int = 0
    block_sensitivity: tuple[int, ...] = (20, 120)
    # 표본 하한
    min_train_rows: int = 500
    min_train_events_per_asset: int = 30
    # 주 모델
    ridge_alpha: float = 10.0
    logit_C: float = 0.1
    logit_max_iter: int = 2000
    # LightGBM: 타깃당 고정 후보 한 개. 탐색·early stopping 없음.
    lgbm_num_leaves: int = 7
    lgbm_max_depth: int = 3
    lgbm_n_estimators: int = 300
    lgbm_learning_rate: float = 0.03
    lgbm_min_child_samples: int = 200
    lgbm_subsample: float = 0.8
    lgbm_subsample_freq: int = 1
    lgbm_colsample_bytree: float = 0.8
    lgbm_reg_lambda: float = 10.0
    lgbm_random_state: int = 0
    # 실행 재현용 (성능 설정이 아니다): 스레드 하나, 결정적 히스토그램
    lgbm_n_jobs: int = 1
    lgbm_deterministic: bool = True
    # baseline
    baseline_smoothing_k: int = 60
    # 현금
    cash_staleness_days: int = 7
    # 점수 변환
    opportunity_reference_min_oof: int = 250
    # 가중치: 날짜마다 시장별 총 가중치 1, 시장 안에서는 자산 균등
    market_weight_equal: bool = True
    # 거시 입력이 이 일수보다 오래되면 그 피쳐는 null (stale 입력을 조용히 쓰지 않는다)
    macro_staleness_days: int = 10
    # 평가 부가 설정
    crisis_top_n: int = 3
    reliability_bins: int = 10
    reliability_bins_per_asset: int = 5

    def as_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["train_start"] = {k: v.isoformat() for k, v in self.train_start.items()}
        d["block_sensitivity"] = list(self.block_sensitivity)
        return d

    def config_hash(self) -> str:
        payload = json.dumps(self.as_dict(), sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(payload.encode()).hexdigest()

    def feature_config(self) -> dict[str, Any]:
        """피쳐 데이터셋에 영향을 주는 값만."""
        return {
            "max_lookback_sessions": self.max_lookback_sessions,
            "macro_staleness_days": self.macro_staleness_days,
        }

    def feature_config_hash(self) -> str:
        return hashlib.sha256(
            json.dumps(self.feature_config(), sort_keys=True).encode()
        ).hexdigest()

    def lgbm_params(self) -> dict[str, Any]:
        return {
            "num_leaves": self.lgbm_num_leaves,
            "max_depth": self.lgbm_max_depth,
            "n_estimators": self.lgbm_n_estimators,
            "learning_rate": self.lgbm_learning_rate,
            "min_child_samples": self.lgbm_min_child_samples,
            "subsample": self.lgbm_subsample,
            "subsample_freq": self.lgbm_subsample_freq,
            "colsample_bytree": self.lgbm_colsample_bytree,
            "reg_lambda": self.lgbm_reg_lambda,
            "random_state": self.lgbm_random_state,
            "n_jobs": self.lgbm_n_jobs,
            "deterministic": self.lgbm_deterministic,
            "force_row_wise": True,
            "verbose": -1,
        }
