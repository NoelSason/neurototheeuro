"""Splits must be deterministic per subject, stratified, and leak-free."""
import numpy as np
import pytest

from n2e.experiment import splits


def test_target_test_split_deterministic_and_no_leakage():
    y = np.array([0, 1] * 50)
    pool1, test1 = splits.target_test_split(y, 0.3, "sub-10")
    pool2, test2 = splits.target_test_split(y, 0.3, "sub-10")
    assert np.array_equal(test1, test2)  # identical across calls (keyed by subject)
    assert len(test1) == 30
    splits.assert_no_leakage(pool1, test1)
    _, test_other = splits.target_test_split(y, 0.3, "sub-11")
    assert not np.array_equal(test1, test_other)  # different subject -> different split


def test_draw_budget_is_stratified_and_sized():
    y = np.array([0] * 70 + [1] * 30)
    pool = np.arange(100)
    small = splits.draw_budget(pool, y, 10, seed=0)
    assert set(y[small]) == {0, 1}  # >=1 per class even at tiny budget
    assert len(splits.draw_budget(pool, y, 100, seed=0)) == 100


def test_study_pool_deterministic_excludes_target():
    decodable = [f"sub-{i}" for i in range(10)]
    a = splits.study_pool_subjects(decodable, "sub-3", 4)
    b = splits.study_pool_subjects(decodable, "sub-3", 4)
    assert a == b and "sub-3" not in a and len(a) == 4


def test_leakage_guard_raises():
    with pytest.raises(AssertionError):
        splits.assert_no_leakage(np.array([1, 2, 3]), np.array([3, 4]))
