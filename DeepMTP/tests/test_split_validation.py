import ast
import inspect

import pandas as pd
import pytest

import DeepMTP.utils.tests as split_validation_module
from DeepMTP import NullDataProgressObserver
from DeepMTP.utils.tests import (
    SplitValidationError,
    check_dp_results,
    check_mc_results,
    check_mlc_results,
    check_mtl_results,
    check_mtr_results,
)


class RecordingDataProgress:
    def __init__(self):
        self.events = []

    def __bool__(self):
        return False

    def on_event(self, event):
        self.events.append(event)


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

    label = validator.__name__.removeprefix("check_").removesuffix("_results").upper()
    assert capsys.readouterr().out == (
        f"Checking if {label} splitting results are valid... Done\n"
    )


def test_instance_partition_rejects_missing_features():
    train = make_split([0], [0], feature_instances=[99])

    with pytest.raises(SplitValidationError, match="missing features"):
        check_mlc_results(
            train,
            make_split([1], [0]),
            make_split([2], [0]),
        )


def test_failed_validation_emits_started_but_not_completed():
    progress = RecordingDataProgress()

    with pytest.raises(SplitValidationError, match="missing features"):
        check_mlc_results(
            make_split([0], [0], feature_instances=[99]),
            make_split([1], [0]),
            make_split([2], [0]),
            progress=progress,
        )

    assert [event.message for event in progress.events] == [
        "Checking if MLC splitting results are valid... "
    ]
    assert progress.events[0].kind == "operation_started"
    assert progress.events[0].subject == "MLC"


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


def test_dyadic_prediction_accepts_matching_feature_coverage(capsys):
    check_dp_results(
        make_split([0, 1], [0, 1]),
        make_split([0, 2], [1, 2]),
        make_split([1, 2], [0, 2]),
    )

    assert capsys.readouterr().out == (
        "Checking if DP splitting results are valid... Done\n"
    )


def test_matrix_completion_rejects_shared_interaction_pairs():
    with pytest.raises(SplitValidationError, match="Instance-target pairs overlap"):
        check_mc_results(
            make_split([0], [0]),
            make_split([0], [0]),
            make_split([0], [1]),
        )


def test_matrix_completion_accepts_unique_interaction_pairs(capsys):
    check_mc_results(
        make_split([0], [0]),
        make_split([0], [1]),
        make_split([1], [0]),
    )

    assert capsys.readouterr().out == (
        "Checking if MC splitting results are valid... Done\n"
    )


def test_injected_progress_captures_validation_without_stdout(capsys):
    progress = RecordingDataProgress()

    check_dp_results(
        make_split([0, 1], [0, 1]),
        make_split([0, 2], [1, 2]),
        make_split([1, 2], [0, 2]),
        progress=progress,
    )

    assert [event.message for event in progress.events] == [
        "Checking if DP splitting results are valid... ",
        "Done",
    ]
    assert progress.events[0].end == ""
    assert [event.kind for event in progress.events] == [
        "operation_started",
        "operation_completed",
    ]
    assert [event.subject for event in progress.events] == ["DP", "DP"]
    assert capsys.readouterr().out == ""


def test_null_progress_suppresses_validation_output(capsys):
    check_mc_results(
        make_split([0], [0]),
        make_split([0], [1]),
        make_split([1], [0]),
        progress=NullDataProgressObserver(),
    )

    assert capsys.readouterr().out == ""


def test_split_validators_have_no_direct_presentation_output():
    tree = ast.parse(inspect.getsource(split_validation_module))
    direct_prints = [
        node.lineno
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "print"
    ]

    assert direct_prints == []
