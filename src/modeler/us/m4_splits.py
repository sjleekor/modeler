"""M4 walk-forward fold 경계 — ``05_validation_protocol.md`` §2.

``modeler.etl.splits.walk_forward_splits``는 순수 거래일 인덱스를 등분해
fold를 만든다. M4가 필요한 것은 그것과 다르다 — **달력에 못박힌 split 월**
(2020-04·2021-04·2022-04·2023-04·2024-04)로 fold를 나누고, 각 fold의 검증은
정확히 12개 리밸런스(달)여야 한다. ``walk_forward_splits``의 균등분할은 이
요구를 만족하지 못해 이 모듈을 따로 둔다(``06`` 지시 "안 맞으면 미국용으로
따로 짜되 왜 안 맞는지 적어라").

**purge와 embargo가 왜 달 하나로 합쳐지는가.** 리밸런스가 월 1회이고
``h = purge = embargo = 21`` 거래일(≈ 한 달)이므로, ``split_k`` 자신이 속한
달 하나를 학습·검증 양쪽에서 빼면 두 조건이 동시에 만족된다:

* **purge** (학습 마지막 라벨이 검증 첫 피쳐 시점과 겹치지 않게): 학습의
  마지막 리밸런스가 ``split_k`` 바로 전 달이면, 그 라벨(h21 뒤 만기)은
  ``split_k`` 달 안에서 닫혀 검증 첫 달(``split_k`` 다음 달) 피쳐 시점보다
  앞선다.
* **embargo** (검증 첫 라벨이 학습 마지막 피쳐와 겹치지 않게): 검증 첫
  리밸런스가 ``split_k`` 다음 달이면, 그 피쳐 시점은 학습 마지막 리밸런스
  (``split_k`` 전 달)로부터 한 달 이상 떨어져 있다.

이 해석은 ``05`` §2 표의 구체적 예("마지막 검증: 2024-05 ~ 2025-04")와
정확히 맞는다 — split_k+21거래일을 두 번(purge·embargo 각각) 더하면 검증이
2024-06에 시작해야 하는데, 문서는 2024-05를 명시했다. 즉 이 프로토콜은
"``split_k`` 달 자체를 gap으로 비운다" 하나로 두 조건을 같이 채우는
설계다.

**early stopping 검증** (``05`` §2 "학습 구간의 마지막 12개월")은 각 fold의
``train`` 안에서 또 나눈다 — 테스트 fold(``valid``)는 보지 않는다.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date

#: fold 경계 — ``05_validation_protocol.md`` §2 사전등록값. 5개, expanding.
FOLD_SPLIT_MONTHS: tuple[date, ...] = (
    date(2020, 4, 1),
    date(2021, 4, 1),
    date(2022, 4, 1),
    date(2023, 4, 1),
    date(2024, 4, 1),
)

#: 검증 fold 길이(월) — ``05`` §2.
VALID_MONTHS = 12

#: early stopping용으로 학습 구간 뒤에서 떼어내는 길이(월) — ``05`` §2.
ES_VALID_MONTHS = 12

#: purge = embargo = 21 거래일(``05`` §2). 리밸런스가 월 1회라 이 상수 자체는
#: fold 경계 계산에 쓰이지 않는다 — ``split_k`` 달을 gap으로 비우는 것으로
#: 대체된다(모듈 docstring 참고). 재현 로그·manifest에 값을 남기려고 둔다.
PURGE_EMBARGO_TRADING_DAYS = 21


def add_months(d: date, n: int) -> date:
    """``d``가 속한 달의 1일 기준으로 ``n``개월 더한 달의 1일."""
    total = d.year * 12 + (d.month - 1) + n
    year, month0 = divmod(total, 12)
    return date(year, month0 + 1, 1)


@dataclass(frozen=True)
class WfFold:
    """M4 walk-forward fold 하나. 날짜는 그 fold에 실제로 들어가는 리밸런스
    날짜(월별 첫 거래일) 리스트 — 인덱스가 아니라 값 그 자체다."""

    fold_id: int
    split_k: date
    train_dates: tuple[date, ...]
    valid_dates: tuple[date, ...]
    es_train_dates: tuple[date, ...]
    es_valid_dates: tuple[date, ...]

    @property
    def train_start(self) -> date:
        return self.train_dates[0]

    @property
    def train_end(self) -> date:
        return self.train_dates[-1]

    @property
    def valid_start(self) -> date:
        return self.valid_dates[0]

    @property
    def valid_end(self) -> date:
        return self.valid_dates[-1]


def build_wf_folds(
    dates: Sequence[date],
    *,
    splits: tuple[date, ...] = FOLD_SPLIT_MONTHS,
    valid_months: int = VALID_MONTHS,
    es_valid_months: int = ES_VALID_MONTHS,
    dev_end: date | None = None,
) -> list[WfFold]:
    """``dates``(정렬된 유일 리밸런스 날짜)로 M4 fold 5개를 만든다.

    ``dev_end``를 주면 ``dates``에 그 뒤 날짜가 있을 때 예외를 던진다 —
    holdout이 fold 구성에 섞이지 않았는지 확인하는 방어선이다(``scan.py``의
    ``assert_dev_window``와 같은 관례). 검증 fold 길이가 ``valid_months``와
    다르면(가용 날짜 부족 등) 예외를 던진다 — 조용히 짧은 fold를 만들지
    않는다.
    """
    ordered = sorted(dates)
    if len(ordered) != len(set(ordered)):
        raise ValueError("dates에 중복이 있습니다")
    if dev_end is not None and ordered and ordered[-1] > dev_end:
        raise ValueError(f"holdout 날짜가 섞였습니다: max(dates)={ordered[-1]} > dev_end={dev_end}")

    folds: list[WfFold] = []
    for fold_id, split_k in enumerate(splits, start=1):
        train_dates = tuple(d for d in ordered if d < split_k)
        valid_start = add_months(split_k, 1)
        valid_end_exclusive = add_months(split_k, 1 + valid_months)
        valid_dates = tuple(d for d in ordered if valid_start <= d < valid_end_exclusive)

        if not train_dates:
            raise ValueError(f"fold {fold_id}(split_k={split_k}): 학습 날짜가 없습니다")
        if len(valid_dates) != valid_months:
            raise ValueError(
                f"fold {fold_id}(split_k={split_k}): 검증 리밸런스가 {len(valid_dates)}개입니다"
                f" (기대 {valid_months}개) — 가용 날짜를 확인하십시오"
            )
        if len(train_dates) <= es_valid_months:
            raise ValueError(
                f"fold {fold_id}(split_k={split_k}): 학습 날짜({len(train_dates)}개)가"
                f" early-stopping 검증({es_valid_months}개월)보다 적거나 같습니다"
            )

        es_valid_dates = train_dates[-es_valid_months:]
        es_train_dates = train_dates[:-es_valid_months]

        folds.append(
            WfFold(
                fold_id=fold_id,
                split_k=split_k,
                train_dates=train_dates,
                valid_dates=valid_dates,
                es_train_dates=es_train_dates,
                es_valid_dates=es_valid_dates,
            )
        )
    return folds
