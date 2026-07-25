import numpy as np
import pytest

from DeepMTP._split_utils import (
    absolute_split_sizes,
    partition_ordered_ids,
    validate_split_ratio,
)


def test_validate_split_ratio_supports_optional_train_and_validation():
    assert validate_split_ratio(
        {"test": 0.2},
        parameter_name="split",
        require_train=False,
        require_validation=False,
    ) == {"train": 0.8, "test": 0.2}


def test_validate_split_ratio_requires_consistent_supplied_train_ratio():
    with pytest.raises(ValueError, match="sum to 1"):
        validate_split_ratio(
            {"train": 0.7, "val": 0.2, "test": 0.2},
        )


def test_absolute_split_sizes_use_original_population():
    assert absolute_split_sizes(
        10,
        {"train": 0.5, "val": 0.2, "test": 0.3},
        parameter_name="split",
        population_name="ID",
    ) == (2, 3)


def test_partition_ordered_ids_preserves_order_and_absolute_sizes():
    train_ids, validation_ids, test_ids = partition_ordered_ids(
        np.arange(10),
        {"train": 0.5, "val": 0.2, "test": 0.3},
        parameter_name="split",
    )

    np.testing.assert_array_equal(train_ids, np.arange(5))
    np.testing.assert_array_equal(validation_ids, np.arange(5, 7))
    np.testing.assert_array_equal(test_ids, np.arange(7, 10))
