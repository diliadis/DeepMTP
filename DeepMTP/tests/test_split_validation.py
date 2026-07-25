import pandas as pd
import pytest

from DeepMTP.utils.tests import (
    SplitValidationError,
    check_dp_results,
    check_mc_results,
    check_mlc_results,
    check_mtl_results,
    check_mtr_results,
)


def make_split(
    instances,
    targets,
    *,
    feature_instances=None,
    feature_targets=None,
):
    instance_features = (
        list(dict.fromkeys(instances))
        if feature_instances is None
        else feature_instances
    )
    target_features = (
        list(dict.fromkeys(targets)) if feature_targets is None else feature_targets
    )
    return {
        "X_instance": {"data": pd.DataFrame({"id": instance_features})},
        "X_target": {"data": pd.DataFrame({"id": target_features})},
        "y": {
            "data": pd.DataFrame(
                {
                    "instance_id": instances,
                    "target_id": targets,
                }
            )
        },
    }


@pytest.mark.parametrize(
    "validator",
    [check_mlc_results, check_mtr_results, check_mtl_results],
)
def test_instance_partition_validators_accept_consistent_disjoint_splits(
    validator,
    capsys,
):
    validator(
        make_split([0, 0], [0, 1]),
        make_split([1, 1], [0, 1]),
        make_split([2, 2], [0, 1]),
    )

    assert "Done" in capsys.readouterr().out


def test_instance_partition_rejects_missing_features():
    train = make_split([0], [0], feature_instances=[99])

    with pytest.raises(SplitValidationError, match="missing features"):
        check_mlc_results(
            train,
            make_split([1], [0]),
            make_split([2], [0]),
        )


def test_instance_partition_rejects_ids_shared_between_splits():
    with pytest.raises(SplitValidationError, match="overlap"):
        check_mlc_results(
            make_split([0], [0]),
            make_split([0], [1]),
            make_split([2], [0]),
        )


def test_dyadic_prediction_rejects_target_feature_mismatch():
    with pytest.raises(SplitValidationError, match="X_target IDs"):
        check_dp_results(
            make_split([0], [0], feature_targets=[99]),
            make_split([1], [1]),
            make_split([2], [2]),
        )


def test_dyadic_prediction_accepts_matching_feature_coverage():
    check_dp_results(
        make_split([0, 1], [0, 1]),
        make_split([0, 2], [1, 2]),
        make_split([1, 2], [0, 2]),
    )


def test_matrix_completion_rejects_shared_interaction_pairs():
    with pytest.raises(SplitValidationError, match="Instance-target pairs overlap"):
        check_mc_results(
            make_split([0], [0]),
            make_split([0], [0]),
            make_split([0], [1]),
        )


def test_matrix_completion_accepts_unique_interaction_pairs():
    check_mc_results(
        make_split([0], [0]),
        make_split([0], [1]),
        make_split([1], [0]),
    )
