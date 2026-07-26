import ast
import inspect

import pytest

from DeepMTP.utils.model_utils import EarlyStopping


def test_early_stopping():

    # test 1
    early = EarlyStopping(
        True, patience=2, delta=0.0, metric_to_track="loss", verbose=True
    )
    for i, d in enumerate([0.6, 0.5, 0.4, 0.2, 0.1]):
        if not early.early_stop_flag:
            val_results = {"val_loss": d}
            early(
                val_results,
                None,
                i,
            )
        else:
            break

    assert early.get_best_epoch() == 4
    assert early.get_best_score() == -0.1

    # test 2
    early = EarlyStopping(
        True, patience=2, delta=0.0, metric_to_track="loss", verbose=True
    )
    for i, d in enumerate([0.6, 0.7, 0.8, 0.2, 0.1]):
        if not early.early_stop_flag:
            val_results = {"val_loss": d}
            early(
                val_results,
                None,
                i,
            )
        else:
            break

    assert early.get_best_epoch() == 0
    assert early.get_best_score() == -0.6

    # test 3
    early = EarlyStopping(
        True, patience=2, delta=0.0, metric_to_track="loss", verbose=True
    )
    for i, d in enumerate([0.6, 0.7, 0.5, 0.4, 0.9, 0.1]):
        if not early.early_stop_flag:
            val_results = {"val_loss": d}
            early(
                val_results,
                None,
                i,
            )
        else:
            break

    assert early.get_best_epoch() == 5
    assert early.get_best_score() == -0.1

    # test 4
    early = EarlyStopping(
        True, patience=3, delta=0.1, metric_to_track="loss", verbose=True
    )
    for i, d in enumerate([0.9, 0.8, 0.5, 0.4]):
        if not early.early_stop_flag:
            val_results = {"val_loss": d}
            early(
                val_results,
                None,
                i,
            )
        else:
            break

    assert early.get_best_epoch() == 3
    assert early.get_best_score() == -0.4

    # test 5
    early = EarlyStopping(
        True, patience=3, delta=0.11, metric_to_track="loss", verbose=True
    )
    for i, d in enumerate([0.9, 0.8, 0.6, 0.5, 0.5, 0.4]):
        if not early.early_stop_flag:
            val_results = {"val_loss": d}
            early(
                val_results,
                None,
                i,
            )
        else:
            break

    assert early.get_best_epoch() == 5
    assert early.get_best_score() == -0.4

    # test 6
    early = EarlyStopping(
        True, patience=2, delta=0.11, metric_to_track="loss", verbose=True
    )
    for i, d in enumerate([0.9, 0.8, 0.6, 0.5, 0.5, 0.4]):
        if not early.early_stop_flag:
            val_results = {"val_loss": d}
            early(
                val_results,
                None,
                i,
            )
        else:
            break

    assert early.get_best_epoch() == 4
    assert early.get_best_score() == -0.5


def test_early_stopping_rejects_unknown_metric():
    with pytest.raises(AttributeError, match="Invalid metric name"):
        EarlyStopping(True, metric_to_track="not_a_metric")


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        ({"patience": -1}, "patience"),
        ({"patience": True}, "patience"),
        ({"delta": -0.1}, "delta"),
        ({"delta": float("nan")}, "delta"),
    ],
)
def test_early_stopping_rejects_invalid_configuration(arguments, message):
    with pytest.raises(ValueError, match=message):
        EarlyStopping(True, **arguments)


def test_early_stopping_rejects_missing_or_non_finite_metric():
    early = EarlyStopping(True)

    with pytest.raises(KeyError, match="val_loss"):
        early({}, None, 0)

    with pytest.raises(ValueError, match="finite"):
        early({"val_loss": float("nan")}, None, 0)


def test_early_stopping_keeps_a_metric_snapshot():
    early = EarlyStopping(True)
    performance = {"val_loss": 0.5}

    early(performance, None, 0)
    performance["val_loss"] = 1.0

    assert early.get_best_performance_results() == {"val_loss": 0.5}


def test_early_stopping_reports_structured_state_transitions():
    early = EarlyStopping(True, patience=2)

    initialized = early({"val_loss": 0.5}, None, 0)
    waiting = early({"val_loss": 0.6}, None, 1)
    stopped = early({"val_loss": 0.7}, None, 2)
    improved = early({"val_loss": 0.4}, None, 3)

    assert initialized.status == "initialized"
    assert initialized.counter == 0
    assert initialized.best_epoch == 0
    assert not initialized.should_stop

    assert waiting.status == "patience_incremented"
    assert waiting.counter == 1
    assert waiting.patience == 2
    assert waiting.best_epoch == 0
    assert not waiting.should_stop

    assert stopped.status == "patience_incremented"
    assert stopped.counter == 2
    assert stopped.should_stop

    assert improved.status == "improved"
    assert improved.counter == 0
    assert improved.best_epoch == 3


def test_early_stopping_update_preserves_established_best_epoch_within_delta():
    early = EarlyStopping(True, patience=2, delta=0.1)

    early({"val_loss": 0.5}, None, 0)
    update = early({"val_loss": 0.45}, None, 1)

    assert update.status == "patience_incremented"
    assert update.best_epoch == 0
    assert early.get_best_epoch() == 1


def test_early_stopping_has_no_direct_presentation_output():
    tree = ast.parse(inspect.getsource(EarlyStopping))
    direct_prints = [
        node.lineno
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "print"
    ]

    assert direct_prints == []
