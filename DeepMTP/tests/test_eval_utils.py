import ast
import inspect
import math

import numpy as np
import pytest
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    hamming_loss,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)
from sklearn.preprocessing import StandardScaler

import DeepMTP.utils.eval_utils as eval_utils_module
from DeepMTP.utils.eval_utils import (
    get_epilepsy_specific_metrics,
    get_performance_results,
)


class RecordingEvaluationProgress:
    def __init__(self):
        self.events = []

    def on_event(self, event):
        self.events.append(event)


mode = [
    {
        "problem_mode": "classification",
        "metrics": ["accuracy", "hamming_loss", "f1_score", "recall", "precision"],
        "averaging": ["macro", "micro", "instance"],
        "format": "numpy",
        "validation_setting": "B",
        "scaler": None,
    },
    {
        "problem_mode": "classification",
        "metrics": ["accuracy", "hamming_loss", "f1_score", "recall", "precision"],
        "averaging": ["macro"],
        "format": "list",
        "validation_setting": "B",
        "scaler": None,
    },
    {
        "problem_mode": "classification",
        "metrics": ["accuracy", "hamming_loss", "f1_score", "recall", "precision"],
        "averaging": ["micro"],
        "format": "numpy",
        "validation_setting": "B",
        "scaler": None,
    },
    {
        "problem_mode": "classification",
        "metrics": ["accuracy", "hamming_loss", "f1_score", "recall", "precision"],
        "averaging": ["instance"],
        "format": "numpy",
        "validation_setting": "B",
        "scaler": None,
    },
    {
        "problem_mode": "regression",
        "metrics": ["RMSE", "MSE", "MAE", "R2"],
        "averaging": ["macro", "micro", "instance"],
        "format": "numpy",
        "validation_setting": "B",
        "scaler": None,
    },
    {
        "problem_mode": "regression",
        "metrics": ["RMSE", "MSE", "MAE", "R2"],
        "averaging": ["macro"],
        "format": "numpy",
        "validation_setting": "B",
        "scaler": None,
    },
    {
        "problem_mode": "regression",
        "metrics": ["RMSE", "MSE", "MAE", "R2"],
        "averaging": ["micro"],
        "format": "numpy",
        "validation_setting": "A",
        "scaler": None,
    },
    {
        "problem_mode": "regression",
        "metrics": ["RMSE", "MSE", "MAE", "R2"],
        "averaging": ["micro"],
        "format": "numpy",
        "validation_setting": "A",
        "scaler": "standard",
    },
    {
        "problem_mode": "regression",
        "metrics": ["RMSE", "MSE", "MAE", "R2"],
        "averaging": ["instance"],
        "format": "numpy",
        "validation_setting": "B",
        "scaler": None,
    },
]


@pytest.mark.parametrize("mode", mode)
def test_get_performance_results(mode):
    if mode["problem_mode"] == "classification":
        y_true = np.random.rand(1000, 10)
        y_true[y_true > 0.5] = 1
        y_true[y_true <= 0.5] = 0

        y_pred = np.random.rand(1000, 10)
        y_pred[y_pred > 0.5] = 1
        y_pred[y_pred <= 0.5] = 0

        # calculating macro accuracy
        accuracy_per_column = []
        hamming_loss_per_column = []
        f1_score_per_column = []
        recall_per_column = []
        precision_per_column = []
        for j in range(y_true.shape[1]):
            accuracy_per_column.append(accuracy_score(y_true[:, j], y_pred[:, j]))
            hamming_loss_per_column.append(hamming_loss(y_true[:, j], y_pred[:, j]))
            f1_score_per_column.append(
                f1_score(y_true[:, j], y_pred[:, j], zero_division=0)
            )
            recall_per_column.append(
                recall_score(y_true[:, j], y_pred[:, j], zero_division=0)
            )
            precision_per_column.append(
                precision_score(y_true[:, j], y_pred[:, j], zero_division=0)
            )

        macro_accuracy = np.mean(accuracy_per_column)
        macro_hamming_loss = np.mean(hamming_loss_per_column)
        macro_f1_score = np.mean(f1_score_per_column)
        macro_recall = np.mean(recall_per_column)
        macro_precision = np.mean(precision_per_column)

        micro_accuracy = accuracy_score(y_true.flatten(), y_pred.flatten())
        micro_hamming_loss = hamming_loss(y_true.flatten(), y_pred.flatten())
        micro_f1_score = f1_score(y_true.flatten(), y_pred.flatten(), zero_division=0)
        micro_recall = recall_score(y_true.flatten(), y_pred.flatten(), zero_division=0)
        micro_precision = precision_score(
            y_true.flatten(), y_pred.flatten(), zero_division=0
        )

        accuracy_per_row = []
        hamming_loss_per_row = []
        f1_score_per_row = []
        recall_per_row = []
        precision_per_row = []
        for j in range(y_true.shape[0]):
            accuracy_per_row.append(accuracy_score(y_true[j, :], y_pred[j, :]))
            hamming_loss_per_row.append(hamming_loss(y_true[j, :], y_pred[j, :]))
            f1_score_per_row.append(
                f1_score(y_true[j, :], y_pred[j, :], zero_division=0)
            )
            recall_per_row.append(
                recall_score(y_true[j, :], y_pred[j, :], zero_division=0)
            )
            precision_per_row.append(
                precision_score(y_true[j, :], y_pred[j, :], zero_division=0)
            )

        instance_accuracy = np.mean(accuracy_per_row)
        instance_hamming_loss = np.mean(hamming_loss_per_row)
        instance_f1_score = np.mean(f1_score_per_row)
        instance_recall = np.mean(recall_per_row)
        instance_precision = np.mean(precision_per_row)

        instances_arr, targets_arr, true_values_arr, pred_values_arr = [], [], [], []
        for i in range(y_true.shape[0]):
            for j in range(y_true.shape[1]):
                instances_arr.append(i)
                targets_arr.append(j)
                true_values_arr.append(y_true[i, j])
                pred_values_arr.append(y_pred[i, j])

        if mode["format"] == "numpy":
            instances_arr = np.array(instances_arr)
            targets_arr = np.array(targets_arr)
            true_values_arr = np.array(true_values_arr)
            pred_values_arr = np.array(pred_values_arr)

        results = get_performance_results(
            "train",
            0,
            instances_arr,
            targets_arr,
            true_values_arr,
            pred_values_arr,
            mode["validation_setting"],
            mode["problem_mode"],
            mode["metrics"],
            mode["averaging"],
            slices_arr=None,
            verbose=False,
            per_target_verbose=False,
            per_instance_verbose=False,
            top_k=None,
            return_detailed_macro=False,
            train_true_value=None,
            scaler_per_target=None,
        )
        if "accuracy" in mode["metrics"]:
            if "macro" in mode["averaging"]:
                # assert math.isclose(results['train_accuracy_macro'], results['train_accuracy_micro'])
                assert math.isclose(results["train_accuracy_macro"], macro_accuracy)
            if "micro" in mode["averaging"]:
                assert math.isclose(results["train_accuracy_micro"], micro_accuracy)
            if "instance" in mode["averaging"]:
                assert math.isclose(
                    results["train_accuracy_instance"], instance_accuracy, abs_tol=0.01
                )

        if "hamming_loss" in mode["metrics"]:
            if "macro" in mode["averaging"]:
                # assert math.isclose(results['train_hamming_loss_macro'], results['train_hamming_loss_micro'])
                assert math.isclose(
                    results["train_hamming_loss_macro"], macro_hamming_loss
                )
            if "micro" in mode["averaging"]:
                assert math.isclose(
                    results["train_hamming_loss_micro"], micro_hamming_loss
                )
            if "instance" in mode["averaging"]:
                assert math.isclose(
                    results["train_hamming_loss_instance"],
                    instance_hamming_loss,
                    abs_tol=0.01,
                )

        if "f1_score" in mode["metrics"]:
            if "macro" in mode["averaging"]:
                assert math.isclose(results["train_f1_score_macro"], macro_f1_score)
            if "micro" in mode["averaging"]:
                assert math.isclose(results["train_f1_score_micro"], micro_f1_score)
            if "instance" in mode["averaging"]:
                assert math.isclose(
                    results["train_f1_score_instance"], instance_f1_score, abs_tol=0.01
                )

        if "recall" in mode["metrics"]:
            if "macro" in mode["averaging"]:
                assert math.isclose(results["train_recall_macro"], macro_recall)
            if "micro" in mode["averaging"]:
                assert math.isclose(results["train_recall_micro"], micro_recall)
            if "instance" in mode["averaging"]:
                assert math.isclose(
                    results["train_recall_instance"], instance_recall, abs_tol=0.01
                )

        if "precision" in mode["metrics"]:
            if "macro" in mode["averaging"]:
                assert math.isclose(results["train_precision_macro"], macro_precision)
            if "micro" in mode["averaging"]:
                assert math.isclose(results["train_precision_micro"], micro_precision)
            if "instance" in mode["averaging"]:
                assert math.isclose(
                    results["train_precision_instance"],
                    instance_precision,
                    abs_tol=0.01,
                )

    else:
        y_true = np.random.rand(1000, 10)
        y_pred = np.random.rand(1000, 10)

        # calculating macro metrics
        RMSE_per_column = []
        MSE_per_column = []
        MAE_score_per_column = []
        R2_per_column = []
        for j in range(y_true.shape[1]):
            RMSE_per_column.append(
                np.sqrt(mean_squared_error(y_true[:, j], y_pred[:, j]))
            )
            MSE_per_column.append(mean_squared_error(y_true[:, j], y_pred[:, j]))
            MAE_score_per_column.append(mean_absolute_error(y_true[:, j], y_pred[:, j]))
            R2_per_column.append(r2_score(y_true[:, j], y_pred[:, j]))

        macro_RMSE = np.mean(RMSE_per_column)
        macro_MSE = np.mean(MSE_per_column)
        macro_MAE = np.mean(MAE_score_per_column)
        macro_R2 = np.mean(R2_per_column)

        # calculating instance metrics
        RMSE_per_row = []
        MSE_per_row = []
        MAE_score_per_row = []
        R2_per_row = []
        for j in range(y_true.shape[0]):
            RMSE_per_row.append(np.sqrt(mean_squared_error(y_true[j, :], y_pred[j, :])))
            MSE_per_row.append(mean_squared_error(y_true[j, :], y_pred[j, :]))
            MAE_score_per_row.append(mean_absolute_error(y_true[j, :], y_pred[j, :]))
            R2_per_row.append(r2_score(y_true[j, :], y_pred[j, :]))

        instance_RMSE = np.mean(RMSE_per_row)
        instance_MSE = np.mean(MSE_per_row)
        instance_MAE = np.mean(MAE_score_per_row)
        instance_R2 = np.mean(R2_per_row)

        # calculating metrics
        micro_RMSE = np.sqrt(mean_squared_error(y_true.flatten(), y_pred.flatten()))
        micro_MSE = mean_squared_error(y_true.flatten(), y_pred.flatten())
        micro_MAE = mean_absolute_error(y_true.flatten(), y_pred.flatten())
        micro_R2 = r2_score(y_true.flatten(), y_pred.flatten())

        instances_arr, targets_arr, true_values_arr, pred_values_arr = [], [], [], []
        for i in range(y_true.shape[0]):
            for j in range(y_true.shape[1]):
                instances_arr.append(i)
                targets_arr.append(j)
                true_values_arr.append(y_true[i, j])
                pred_values_arr.append(y_pred[i, j])

        if mode["format"] == "numpy":
            instances_arr = np.array(instances_arr)
            targets_arr = np.array(targets_arr)
            true_values_arr = np.array(true_values_arr)
            pred_values_arr = np.array(pred_values_arr)

        true_values_arr_scaled = None
        pred_values_arr_scaled = None
        scaler = None

        if (
            mode["scaler"] is not None
            and mode["scaler"].lower() == "standard"
            and mode["validation_setting"] == "A"
        ):
            scaler = StandardScaler()
            scaler.fit(true_values_arr.flatten().reshape(-1, 1))
            true_values_arr_scaled = scaler.transform(
                true_values_arr.reshape(-1, 1)
            ).flatten()
            pred_values_arr_scaled = scaler.transform(
                pred_values_arr.reshape(-1, 1)
            ).flatten()

        results = get_performance_results(
            "train",
            0,
            instances_arr,
            targets_arr,
            true_values_arr if mode["scaler"] is None else true_values_arr_scaled,
            pred_values_arr if mode["scaler"] is None else pred_values_arr_scaled,
            mode["validation_setting"],
            mode["problem_mode"],
            mode["metrics"],
            mode["averaging"],
            slices_arr=None,
            verbose=False,
            per_target_verbose=False,
            per_instance_verbose=False,
            top_k=None,
            return_detailed_macro=False,
            train_true_value=None,
            scaler_per_target=scaler,
        )

        if "MSE" in mode["metrics"]:
            if "macro" in mode["averaging"]:
                # assert math.isclose(results['train_MSE_macro'], results['train_MSE_micro'])
                assert math.isclose(results["train_MSE_macro"], macro_MSE)
            if "micro" in mode["averaging"]:
                assert math.isclose(results["train_MSE_micro"], micro_MSE)
            if "instance" in mode["averaging"]:
                assert math.isclose(
                    results["train_MSE_instance"], instance_MSE, abs_tol=0.01
                )

        if "MAE" in mode["metrics"]:
            if "macro" in mode["averaging"]:
                # assert math.isclose(results['train_MAE_macro'], results['train_MAE_micro'])
                assert math.isclose(results["train_MAE_macro"], macro_MAE)
            if "micro" in mode["averaging"]:
                assert math.isclose(results["train_MAE_micro"], micro_MAE)
            if "instance" in mode["averaging"]:
                assert math.isclose(
                    results["train_MAE_instance"], instance_MAE, abs_tol=0.01
                )

        if "R2" in mode["metrics"]:
            if "macro" in mode["averaging"]:
                assert math.isclose(results["train_R2_macro"], macro_R2)
            if "micro" in mode["averaging"]:
                assert math.isclose(results["train_R2_micro"], micro_R2)
            if "instance" in mode["averaging"]:
                assert math.isclose(
                    results["train_R2_instance"], instance_R2, abs_tol=0.01
                )

        if "RMSE" in mode["metrics"]:
            if "macro" in mode["averaging"]:
                assert math.isclose(results["train_RMSE_macro"], macro_RMSE)
            if "micro" in mode["averaging"]:
                assert math.isclose(results["train_RMSE_micro"], micro_RMSE)
            if "instance" in mode["averaging"]:
                assert math.isclose(
                    results["train_RMSE_instance"], instance_RMSE, abs_tol=0.01
                )


def test_single_class_micro_confusion_matrix_has_all_cells():
    results = get_performance_results(
        mode="test",
        epoch_id=0,
        instances_arr=np.array([0, 1]),
        targets_arr=np.array([0, 0]),
        true_values_arr=np.array([0, 0]),
        pred_values_arr=np.array([0.1, 0.2]),
        validation_setting="B",
        problem_mode="classification",
        metrics=["accuracy"],
        averaging=["micro"],
    )

    assert results["test_tn_micro"] == 2
    assert results["test_fp_micro"] == 0
    assert results["test_fn_micro"] == 0
    assert results["test_tp_micro"] == 0


@pytest.mark.parametrize("averaging", ["micro", "macro"])
def test_single_class_curve_metrics_are_reported_as_undefined(averaging):
    results = get_performance_results(
        mode="test",
        epoch_id=0,
        instances_arr=np.array([0, 1]),
        targets_arr=np.array([0, 1]),
        true_values_arr=np.array([0, 0]),
        pred_values_arr=np.array([0.1, 0.2]),
        validation_setting="B",
        problem_mode="classification",
        metrics=["auroc", "aupr"],
        averaging=[averaging],
        return_detailed_macro=averaging == "macro",
    )

    assert math.isnan(results[f"test_auroc_{averaging}"])
    assert math.isnan(results[f"test_aupr_{averaging}"])
    if averaging == "macro":
        assert math.isnan(results["test_auroc_target_0"])
        assert math.isnan(results["test_aupr_target_1"])


def test_singleton_macro_groups_are_evaluated():
    results = get_performance_results(
        mode="test",
        epoch_id=0,
        instances_arr=np.array([0, 1]),
        targets_arr=np.array([0, 1]),
        true_values_arr=np.array([0, 1]),
        pred_values_arr=np.array([0.1, 0.9]),
        validation_setting="B",
        problem_mode="classification",
        metrics=["accuracy"],
        averaging=["macro"],
        top_k=1,
        return_detailed_macro=True,
    )

    assert results["test_accuracy_target_0"] == 1.0
    assert results["test_accuracy_target_1"] == 1.0
    assert results["test_accuracy_macro"] == 1.0
    assert results["test_top_1_accuracy_target_0"] == 1.0
    assert results["test_top_1_accuracy_macro"] == 1.0


def test_macro_details_are_opt_in():
    results = get_performance_results(
        mode="test",
        epoch_id=0,
        instances_arr=np.array([0, 1, 0, 1]),
        targets_arr=np.array([0, 0, 1, 1]),
        true_values_arr=np.array([0, 1, 0, 1]),
        pred_values_arr=np.array([0.1, 0.9, 0.2, 0.8]),
        validation_setting="B",
        problem_mode="classification",
        metrics=["accuracy"],
        averaging=["macro"],
        top_k=1,
    )

    assert results["test_accuracy_macro"] == 1.0
    assert results["test_top_1_accuracy_macro"] == 1.0
    assert not any("_target_" in key for key in results)


def test_macro_rrmse_uses_each_targets_training_mean():
    results = get_performance_results(
        mode="test",
        epoch_id=0,
        instances_arr=np.array([0, 1, 0, 1]),
        targets_arr=np.array([0, 0, 1, 1]),
        true_values_arr=np.array([2.0, 4.0, 10.0, 14.0]),
        pred_values_arr=np.array([1.0, 5.0, 8.0, 16.0]),
        validation_setting="B",
        problem_mode="regression",
        metrics=["RRMSE"],
        averaging=["macro"],
        return_detailed_macro=True,
        train_true_value={0: 3.0, 1: 12.0},
    )

    assert results["test_RRMSE_target_0"] == pytest.approx(1.0)
    assert results["test_RRMSE_target_1"] == pytest.approx(1.0)
    assert results["test_RRMSE_macro"] == pytest.approx(1.0)


def test_rrmse_requires_complete_target_training_means():
    arguments = {
        "mode": "test",
        "epoch_id": 0,
        "instances_arr": np.array([0, 1, 0, 1]),
        "targets_arr": np.array([0, 0, 1, 1]),
        "true_values_arr": np.array([2.0, 4.0, 10.0, 14.0]),
        "pred_values_arr": np.array([1.0, 5.0, 8.0, 16.0]),
        "validation_setting": "B",
        "problem_mode": "regression",
        "metrics": ["RRMSE"],
        "averaging": ["macro"],
    }

    with pytest.raises(ValueError, match="requires train_true_value"):
        get_performance_results(**arguments)

    with pytest.raises(ValueError, match="missing target IDs.*1"):
        get_performance_results(
            **arguments,
            train_true_value={0: 3.0},
        )


def test_rrmse_rejects_averaging_without_matching_training_baselines():
    with pytest.raises(ValueError, match="requires macro averaging"):
        get_performance_results(
            mode="test",
            epoch_id=0,
            instances_arr=np.array([0, 1]),
            targets_arr=np.array([0, 0]),
            true_values_arr=np.array([2.0, 4.0]),
            pred_values_arr=np.array([1.0, 5.0]),
            validation_setting="B",
            problem_mode="regression",
            metrics=["RRMSE"],
            averaging=["micro"],
            train_true_value={0: 3.0},
        )


def test_target_specific_scalers_restore_original_values():
    targets = np.array([0, 0, 1, 1])
    true_values = np.array([10.0, 12.0, 100.0, 104.0])
    predicted_values = true_values + 1.0
    scalers = {
        0: StandardScaler().fit(true_values[targets == 0].reshape(-1, 1)),
        1: StandardScaler().fit(true_values[targets == 1].reshape(-1, 1)),
    }
    scaled_true_values = np.array(
        [
            scalers[target].transform([[value]])[0, 0]
            for target, value in zip(targets, true_values, strict=True)
        ]
    )
    scaled_predicted_values = np.array(
        [
            scalers[target].transform([[value]])[0, 0]
            for target, value in zip(targets, predicted_values, strict=True)
        ]
    )

    results = get_performance_results(
        mode="test",
        epoch_id=0,
        instances_arr=np.array([0, 1, 0, 1]),
        targets_arr=targets,
        true_values_arr=scaled_true_values,
        pred_values_arr=scaled_predicted_values,
        validation_setting="B",
        problem_mode="regression",
        metrics=["MSE"],
        averaging=["micro"],
        scaler_per_target=scalers,
    )

    assert results["test_MSE_micro"] == pytest.approx(1.0)


def test_target_specific_scalers_must_cover_every_target():
    scaler = StandardScaler().fit(np.array([[10.0], [12.0]]))

    with pytest.raises(ValueError, match="scaler_per_target.*missing target IDs.*1"):
        get_performance_results(
            mode="test",
            epoch_id=0,
            instances_arr=np.array([0, 1]),
            targets_arr=np.array([0, 1]),
            true_values_arr=np.array([0.0, 0.0]),
            pred_values_arr=np.array([0.0, 0.0]),
            validation_setting="B",
            problem_mode="regression",
            metrics=["MSE"],
            averaging=["micro"],
            scaler_per_target={0: scaler},
        )


def test_inverse_score_scaling_rejects_classification():
    scaler = StandardScaler().fit(np.array([[0.0], [1.0]]))

    with pytest.raises(ValueError, match="only for regression"):
        get_performance_results(
            mode="test",
            epoch_id=0,
            instances_arr=np.array([0, 1]),
            targets_arr=np.array([0, 0]),
            true_values_arr=np.array([0, 1]),
            pred_values_arr=np.array([0.1, 0.9]),
            validation_setting="A",
            problem_mode="classification",
            metrics=["accuracy"],
            averaging=["micro"],
            scaler_per_target=scaler,
        )


def test_inverse_score_scaling_requires_transformer_protocol():
    with pytest.raises(TypeError, match=r"scaler_per_target must provide"):
        get_performance_results(
            mode="test",
            epoch_id=0,
            instances_arr=np.array([0, 1]),
            targets_arr=np.array([0, 0]),
            true_values_arr=np.array([1.0, 2.0]),
            pred_values_arr=np.array([1.0, 2.0]),
            validation_setting="A",
            problem_mode="regression",
            metrics=["MSE"],
            averaging=["micro"],
            scaler_per_target=object(),
        )


def test_verbose_evaluation_uses_modern_numpy_nan_api(capsys):
    results = get_performance_results(
        mode="test",
        epoch_id=0,
        instances_arr=[0, 1],
        targets_arr=[0, 0],
        true_values_arr=[1.0, 2.0],
        pred_values_arr=[1.0, 2.0],
        validation_setting="A",
        problem_mode="regression",
        metrics=["MSE"],
        averaging=["micro"],
        verbose=True,
    )

    assert results["test_MSE_micro"] == 0.0
    assert "non zero predictions" in capsys.readouterr().out


def test_evaluation_routes_verbose_details_to_injected_progress(capsys):
    progress = RecordingEvaluationProgress()

    results = get_performance_results(
        mode="test",
        epoch_id=0,
        instances_arr=[0, 0, 1, 1],
        targets_arr=[0, 1, 0, 1],
        true_values_arr=[1.0, 2.0, 3.0, 4.0],
        pred_values_arr=[1.0, 2.0, 3.0, 4.0],
        validation_setting="B",
        problem_mode="regression",
        metrics=["MSE"],
        averaging=["micro", "macro", "instance"],
        verbose=True,
        per_target_verbose=True,
        per_instance_verbose=True,
        top_k=1,
        progress=progress,
    )

    event_kinds = [event.kind for event in progress.events]
    assert results["test_MSE_micro"] == 0.0
    assert results["test_MSE_macro"] == 0.0
    assert results["test_MSE_instance"] == 0.0
    assert event_kinds == [
        "input_inspected",
        "evaluation_started",
        "metric_calculated",
        "group_results_calculated",
        "aggregate_results_calculated",
        "group_results_calculated",
        "aggregate_results_calculated",
        "group_results_calculated",
        "aggregate_results_calculated",
        "group_results_calculated",
        "aggregate_results_calculated",
        "evaluation_completed",
    ]
    target_top_k_event = next(
        event
        for event in progress.events
        if event.kind == "group_results_calculated"
        and event.group == "target"
        and event.top_k == 1
    )
    assert target_top_k_event.results is not None
    assert capsys.readouterr().out == ""


def test_evaluation_routes_single_class_diagnostic_to_progress(capsys):
    progress = RecordingEvaluationProgress()

    results = get_performance_results(
        mode="val",
        epoch_id=0,
        instances_arr=[0, 1],
        targets_arr=[4, 4],
        true_values_arr=[1.0, 1.0],
        pred_values_arr=[0.8, 0.9],
        validation_setting="B",
        problem_mode="classification",
        metrics=["auroc"],
        averaging=["macro"],
        verbose=True,
        progress=progress,
    )

    diagnostic = next(
        event for event in progress.events if event.kind == "single_class_group"
    )
    assert math.isnan(results["val_auroc_macro"])
    assert diagnostic.group_id == 4
    assert diagnostic.unique_true_values == 1
    assert capsys.readouterr().out == ""


def test_per_target_verbose_keeps_legacy_console_output(capsys):
    get_performance_results(
        mode="test",
        epoch_id=0,
        instances_arr=[0, 1],
        targets_arr=[0, 0],
        true_values_arr=[1.0, 2.0],
        pred_values_arr=[1.0, 2.0],
        validation_setting="B",
        problem_mode="regression",
        metrics=["MSE"],
        averaging=["macro"],
        per_target_verbose=True,
    )

    output = capsys.readouterr().out
    assert "Per target results:" in output
    assert "==========" not in output


def test_eval_utils_has_no_direct_presentation_output():
    tree = ast.parse(inspect.getsource(eval_utils_module))
    direct_prints = [
        node.lineno
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "print"
    ]

    assert direct_prints == []


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"validation_setting": "Z"}, "validation_setting"),
        ({"problem_mode": "forecasting"}, "problem_mode"),
        ({"averaging": ["weighted"]}, "averaging"),
        ({"averaging": ["macro"]}, "validation setting A"),
        ({"top_k": 0}, "top_k"),
        ({"top_k": 1}, "requires macro or instance averaging"),
        ({"return_detailed_macro": True}, "requires macro averaging"),
        ({"metrics": ["BOGUS"]}, "Unsupported regression metrics"),
    ],
)
def test_evaluation_rejects_invalid_boundary_values(override, message):
    arguments = {
        "mode": "test",
        "epoch_id": 0,
        "instances_arr": [0],
        "targets_arr": [0],
        "true_values_arr": [1.0],
        "pred_values_arr": [1.0],
        "validation_setting": "A",
        "problem_mode": "regression",
        "metrics": ["MSE"],
        "averaging": ["micro"],
    }
    arguments.update(override)

    with pytest.raises(ValueError, match=message):
        get_performance_results(**arguments)


def test_evaluation_rejects_unimplemented_slice_ids():
    with pytest.raises(NotImplementedError, match="Slice-aware evaluation"):
        get_performance_results(
            mode="test",
            epoch_id=0,
            instances_arr=[0],
            targets_arr=[0],
            true_values_arr=[1.0],
            pred_values_arr=[1.0],
            validation_setting="A",
            problem_mode="regression",
            metrics=["MSE"],
            averaging=["micro"],
            slices_arr=[0],
        )


def test_ranking_ndcg_accepts_one_query_as_flat_values():
    results = get_performance_results(
        mode="test",
        epoch_id=0,
        instances_arr=[0, 0, 0],
        targets_arr=[0, 1, 2],
        true_values_arr=[3.0, 2.0, 1.0],
        pred_values_arr=[0.9, 0.2, 0.1],
        validation_setting="A",
        problem_mode="ranking",
        metrics=["ndcg"],
        averaging=["micro"],
    )

    assert results["test_ndcg_micro"] == pytest.approx(1.0)


def test_singleton_ranking_group_reports_undefined_ndcg():
    results = get_performance_results(
        mode="test",
        epoch_id=0,
        instances_arr=[0],
        targets_arr=[0],
        true_values_arr=[1.0],
        pred_values_arr=[0.9],
        validation_setting="A",
        problem_mode="ranking",
        metrics=["ndcg"],
        averaging=["micro"],
    )

    assert math.isnan(results["test_ndcg_micro"])


def test_epilepsy_metrics_count_single_sample_episode_overlap():
    results = get_epilepsy_specific_metrics(
        y_true=[0, 1, 0],
        y_pred=[0, 1, 0],
        metrics=["sensitivity"],
    )

    assert results["sensitivity_1"] == 1.0
    assert results["sensitivity_10"] == 1.0
    assert results["sensitivity_60"] == 1.0


def test_epilepsy_metrics_group_episodes_and_false_alarm_windows():
    results = get_epilepsy_specific_metrics(
        y_true=[0, 1, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0],
        y_pred=[1, 0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 1],
    )

    for window in (1, 10, 60):
        assert results[f"sensitivity_{window}"] == pytest.approx(2 / 3)
    assert results["false_alarm_rate_per_hour_1"] == 1200
    assert results["false_alarm_rate_per_hour_10"] == 600
    assert results["false_alarm_rate_per_hour_60"] == 300
    assert results["positive_predictive_value_1"] == pytest.approx(1 / 3)
    assert results["positive_predictive_value_10"] == 0.5
    assert results["positive_predictive_value_60"] == pytest.approx(2 / 3)
    assert results["f1_score_1"] == pytest.approx(4 / 9)
    assert results["f1_score_10"] == pytest.approx(4 / 7)
    assert results["f1_score_60"] == pytest.approx(2 / 3)
