"""``modeler.us.m4_splits`` 단위 테스트 — 전부 합성 날짜다.

M4 walk-forward fold 경계(purge·embargo에 해당하는 gap 월, early-stopping
분할, holdout 날짜 벽)를 검사한다(``06_execution_steps.md`` M4 §7 지시).
"""

from __future__ import annotations

from datetime import date

import pytest

from modeler.us.m4_splits import (
    ES_VALID_MONTHS,
    FOLD_SPLIT_MONTHS,
    VALID_MONTHS,
    add_months,
    build_wf_folds,
)


def _month_starts(start: date, end: date) -> list[date]:
    """``start``부터 ``end``까지(포함) 매달 1일 — 실제 레이크를 읽지 않는
    합성 리밸런스 날짜(월별 첫 거래일의 대역)."""
    dates = []
    cur = start
    while cur <= end:
        dates.append(cur)
        cur = add_months(cur, 1)
    return dates


#: ``05_validation_protocol.md`` §1의 개발 구간과 같은 범위(2018-09~2025-06).
DEV_RANGE_DATES = _month_starts(date(2018, 9, 1), date(2025, 6, 1))


# --- 1. add_months -------------------------------------------------------------


def test_add_months_rolls_over_year() -> None:
    assert add_months(date(2024, 11, 15), 2) == date(2025, 1, 1)


def test_add_months_zero_is_month_start() -> None:
    assert add_months(date(2024, 3, 17), 0) == date(2024, 3, 1)


def test_add_months_negative() -> None:
    assert add_months(date(2020, 1, 1), -1) == date(2019, 12, 1)


# --- 2. fold 크기 — 05 §2가 명시한 구체적 예 ------------------------------------


def test_five_folds_built_from_dev_range() -> None:
    folds = build_wf_folds(DEV_RANGE_DATES)
    assert len(folds) == 5
    assert [f.split_k for f in folds] == list(FOLD_SPLIT_MONTHS)


def test_train_months_grow_as_documented() -> None:
    """``05`` §2: "첫 학습 19개월", expanding window로 계속 늘어난다."""
    folds = build_wf_folds(DEV_RANGE_DATES)
    train_months = [len(f.train_dates) for f in folds]
    assert train_months == [19, 31, 43, 55, 67]
    assert train_months == sorted(train_months)  # expanding: 단조증가


def test_valid_is_always_exactly_12_months() -> None:
    folds = build_wf_folds(DEV_RANGE_DATES)
    for f in folds:
        assert len(f.valid_dates) == VALID_MONTHS == 12


def test_last_fold_matches_documented_example() -> None:
    """``05`` §2: "마지막 검증: 2024-05 ~ 2025-04 리밸런스"."""
    folds = build_wf_folds(DEV_RANGE_DATES)
    last = folds[-1]
    assert last.split_k == date(2024, 4, 1)
    assert last.valid_start == date(2024, 5, 1)
    assert last.valid_end == date(2025, 4, 1)


# --- 3. purge/embargo gap — split_k 달 자체가 비어야 한다 ------------------------


def test_split_k_month_itself_is_excluded_from_train_and_valid() -> None:
    """purge=embargo=21거래일(≈1개월)이 ``split_k`` 달 하나로 합쳐진다는 이
    모듈의 핵심 주장 — ``split_k``가 train에도 valid에도 없어야 한다."""
    folds = build_wf_folds(DEV_RANGE_DATES)
    for f in folds:
        assert f.split_k not in f.train_dates
        assert f.split_k not in f.valid_dates


def test_train_ends_before_split_k_and_valid_starts_after() -> None:
    folds = build_wf_folds(DEV_RANGE_DATES)
    for f in folds:
        assert f.train_end < f.split_k
        assert f.valid_start > f.split_k
        # 정확히 한 달짜리 gap이다 — 그 이상 비지 않는다.
        assert add_months(f.train_end, 1) == f.split_k
        assert add_months(f.split_k, 1) == f.valid_start


def test_no_overlap_between_train_and_valid() -> None:
    folds = build_wf_folds(DEV_RANGE_DATES)
    for f in folds:
        assert set(f.train_dates).isdisjoint(f.valid_dates)


def test_validation_windows_across_folds_are_contiguous_and_disjoint() -> None:
    """fold별 검증 구간이 서로 겹치지 않고, 이어 붙이면 60개월이 된다."""
    folds = build_wf_folds(DEV_RANGE_DATES)
    all_valid: list[date] = []
    for f in folds:
        all_valid.extend(f.valid_dates)
    assert len(all_valid) == len(set(all_valid)) == 60


# --- 4. early-stopping 분할 — 05 §2 "학습 구간의 마지막 12개월" ------------------


def test_es_valid_is_last_12_months_of_train() -> None:
    folds = build_wf_folds(DEV_RANGE_DATES)
    for f in folds:
        assert len(f.es_valid_dates) == ES_VALID_MONTHS == 12
        assert f.es_valid_dates == f.train_dates[-12:]


def test_es_train_plus_es_valid_reconstructs_train() -> None:
    folds = build_wf_folds(DEV_RANGE_DATES)
    for f in folds:
        assert f.es_train_dates + f.es_valid_dates == f.train_dates


def test_es_valid_never_touches_test_fold() -> None:
    """early-stopping 검증이 테스트 fold(``valid_dates``)를 보면 안 된다
    (``05`` §2 "테스트 fold를 안 본다")."""
    folds = build_wf_folds(DEV_RANGE_DATES)
    for f in folds:
        assert set(f.es_valid_dates).isdisjoint(f.valid_dates)
        assert set(f.es_train_dates).isdisjoint(f.valid_dates)


def test_first_fold_es_train_is_seven_months() -> None:
    """fold1: train 19개월 - es_valid 12개월 = es_train 7개월."""
    folds = build_wf_folds(DEV_RANGE_DATES)
    assert len(folds[0].es_train_dates) == 7


# --- 5. holdout 날짜 벽 --------------------------------------------------------


def test_dev_end_guard_rejects_holdout_dates() -> None:
    leaked = [*DEV_RANGE_DATES, date(2025, 7, 1)]  # holdout — 2025-06-30 뒤
    with pytest.raises(ValueError, match="holdout"):
        build_wf_folds(leaked, dev_end=date(2025, 6, 30))


def test_dev_end_guard_passes_when_no_leak() -> None:
    build_wf_folds(DEV_RANGE_DATES, dev_end=date(2025, 6, 30))  # 예외 없이 통과


def test_no_dev_end_means_no_guard() -> None:
    leaked = [*DEV_RANGE_DATES, date(2026, 1, 1)]
    build_wf_folds(leaked)  # dev_end 안 주면 걸러내지 않는다 — 예외 없음


# --- 6. 방어적 검사 -------------------------------------------------------------


def test_duplicate_dates_raise() -> None:
    with pytest.raises(ValueError, match="중복"):
        build_wf_folds([*DEV_RANGE_DATES, DEV_RANGE_DATES[0]])


def test_insufficient_valid_dates_raise() -> None:
    """검증 fold가 12개월을 못 채우면(가용 날짜 부족) 조용히 짧은 fold를
    만들지 않고 예외를 던진다."""
    short_range = _month_starts(date(2018, 9, 1), date(2024, 6, 1))  # 마지막 fold가 짤린다
    with pytest.raises(ValueError, match="검증 리밸런스"):
        build_wf_folds(short_range)


def test_too_few_train_dates_for_es_split_raises() -> None:
    with pytest.raises(ValueError):
        build_wf_folds(
            _month_starts(date(2019, 10, 1), date(2025, 6, 1)),
            splits=(date(2020, 4, 1),),
        )
