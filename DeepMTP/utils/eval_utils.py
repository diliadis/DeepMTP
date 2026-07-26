from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Protocol, TypeAlias

import more_itertools as mit
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    auc,
    confusion_matrix,
    f1_score,
    hamming_loss,
    mean_absolute_error,
    mean_squared_error,
    ndcg_score,
    precision_recall_curve,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)

ArrayValues: TypeAlias = Sequence[Any]
MetricResults: TypeAlias = dict[str, float]
TrainingMeans: TypeAlias = Mapping[Any, float]


class InverseTransformer(Protocol):
    """Minimal scaler interface needed by the evaluation utilities."""

    def inverse_transform(self, values: Any) -> Any:
        """Return values transformed back to their original scale."""


ScalerInput: TypeAlias = InverseTransformer | Mapping[Any, InverseTransformer] | None
EPILEPSY_METRICS = frozenset(
    {
        "sensitivity",
        "false_alarm_rate_per_hour",
        "positive_predictive_value",
        "f1_score_epilepsy_version",
    }
)
SUPPORTED_METRICS = {
    "classification": frozenset(
        {
            "accuracy",
            "aupr",
            "auroc",
            "f1_score",
            "hamming_loss",
            "precision",
            "recall",
        }
    )
    | EPILEPSY_METRICS,
    "regression": frozenset({"MAE", "MSE", "R2", "RMSE", "RRMSE"}),
    "ranking": frozenset({"ndcg"}),
}


def _inverse_transform_scalar(
    scaler: InverseTransformer,
    value: Any,
) -> float:
    transformed = np.asarray(
        scaler.inverse_transform([[float(value)]]),
        dtype=float,
    ).reshape(-1)
    if transformed.size != 1:
        raise ValueError("A scalar inverse transformation must return one value.")
    return float(transformed[0])


def _inverse_transform_values(
    scaler: InverseTransformer,
    values: pd.Series,
) -> list[float]:
    transformed = np.asarray(
        scaler.inverse_transform(values.to_numpy().reshape(-1, 1)),
        dtype=float,
    ).reshape(-1)
    if transformed.size != len(values):
        raise ValueError(
            "The inverse transformer returned a different number of values."
        )
    return [float(value) for value in transformed]


def instance_wise_true_inverse_transformation(
    row: pd.Series,
    scaler_per_instance: Mapping[Any, InverseTransformer],
) -> float:
    """Inverse-transform one true value with its instance-specific scaler."""
    return _inverse_transform_scalar(
        scaler_per_instance[row["instance_id"]],
        row["true_values"],
    )


def instance_wise_pred_inverse_transformation(
    row: pd.Series,
    scaler_per_instance: Mapping[Any, InverseTransformer],
) -> float:
    """Inverse-transform one prediction with its instance-specific scaler."""
    return _inverse_transform_scalar(
        scaler_per_instance[row["instance_id"]],
        row["pred_values"],
    )


def target_wise_true_inverse_transformation(
    row: pd.Series,
    scaler_per_target: Mapping[Any, InverseTransformer],
) -> float:
    """Inverse-transform one true value with its target-specific scaler."""
    return _inverse_transform_scalar(
        scaler_per_target[row["target_id"]],
        row["true_values"],
    )


def target_wise_pred_inverse_transformation(
    row: pd.Series,
    scaler_per_target: Mapping[Any, InverseTransformer],
) -> float:
    """Inverse-transform one prediction with its target-specific scaler."""
    return _inverse_transform_scalar(
        scaler_per_target[row["target_id"]],
        row["pred_values"],
    )


def _validate_evaluation_inputs(
    instances: ArrayValues,
    targets: ArrayValues,
    true_values: ArrayValues,
    predicted_values: ArrayValues,
    validation_setting: str,
    problem_mode: str,
    metrics: Sequence[str],
    averaging: Sequence[str],
    slices: ArrayValues | None,
    return_detailed_macro: bool,
    top_k: int | None,
    train_true_value: TrainingMeans | None,
    scaler_per_target: ScalerInput,
) -> None:
    lengths = {
        len(instances),
        len(targets),
        len(true_values),
        len(predicted_values),
    }
    if len(lengths) != 1:
        raise ValueError(
            "instances, targets, true values, and predictions must have equal length."
        )
    if len(true_values) == 0:
        raise ValueError("Evaluation requires at least one prediction.")
    if slices is not None:
        raise NotImplementedError("Slice-aware evaluation is not implemented.")
    if validation_setting not in {"A", "B", "C", "D"}:
        raise ValueError(
            "validation_setting must be one of 'A', 'B', 'C', or 'D'; "
            f"received {validation_setting!r}."
        )
    if problem_mode not in {"classification", "regression", "ranking"}:
        raise ValueError(
            "problem_mode must be 'classification', 'regression', or 'ranking'; "
            f"received {problem_mode!r}."
        )
    _validate_metrics(problem_mode, metrics)
    unknown_averaging = set(averaging).difference({"micro", "macro", "instance"})
    if unknown_averaging:
        raise ValueError(
            f"Unsupported metric averaging schemes: {sorted(unknown_averaging)}."
        )
    if validation_setting == "A" and list(averaging) != ["micro"]:
        raise ValueError("averaging must be ['micro'] for validation setting A.")
    if not isinstance(return_detailed_macro, bool):
        raise TypeError("return_detailed_macro must be a boolean.")
    if return_detailed_macro and "macro" not in averaging:
        raise ValueError("return_detailed_macro requires macro averaging.")
    if "RRMSE" in metrics:
        if validation_setting == "A" or list(averaging) != ["macro"]:
            raise ValueError(
                "RRMSE requires macro averaging in validation settings B, C, or D."
            )
        if not isinstance(train_true_value, Mapping):
            raise ValueError(
                "RRMSE requires train_true_value with one training mean per target."
            )
        _validate_target_mapping(
            train_true_value,
            targets,
            parameter_name="train_true_value",
            require_finite_values=True,
        )
    if scaler_per_target is not None:
        if problem_mode != "regression":
            raise ValueError("Inverse score scaling is supported only for regression.")
        if validation_setting == "A":
            if isinstance(scaler_per_target, Mapping):
                raise TypeError(
                    "Setting A expects one scaler, not a mapping of target scalers."
                )
            _validate_inverse_transformer(
                scaler_per_target,
                parameter_name="scaler_per_target",
            )
        else:
            if not isinstance(scaler_per_target, Mapping):
                raise TypeError(
                    f"Setting {validation_setting} expects a mapping of target scalers."
                )
            _validate_target_mapping(
                scaler_per_target,
                targets,
                parameter_name="scaler_per_target",
                require_inverse_transformers=True,
            )
    if top_k is not None and (
        not isinstance(top_k, int) or isinstance(top_k, bool) or top_k <= 0
    ):
        raise ValueError("top_k must be a positive integer when provided.")


def get_performance_results(
    mode: str,
    epoch_id: int,
    instances_arr: ArrayValues,
    targets_arr: ArrayValues,
    true_values_arr: ArrayValues,
    pred_values_arr: ArrayValues,
    validation_setting: str,
    problem_mode: str,
    metrics: Sequence[str],
    averaging: Sequence[str],
    slices_arr: ArrayValues | None = None,
    verbose: bool = False,
    per_target_verbose: bool = False,
    per_instance_verbose: bool = False,
    top_k: int | None = None,
    return_detailed_macro: bool = False,
    train_true_value: TrainingMeans | None = None,
    scaler_per_target: ScalerInput = None,
) -> MetricResults:
    """Calculates all the metrics using different averaging schemes

    Args:
        mode (str): The mode during which the calculation of metrics is requested. Possible values are train, val, test.
        epoch_id (int): The id of the current epoch.
        instances_arr (numpy.array): An array with the instance ids. This is used when instance averaging is needed.
        targets_arr (numpy.array): An array with the target ids. This is used when macro-averaging is needed.
        true_values_arr (numpy.array): An array with the true values.
        pred_values_arr (numpy.array): An array with the predicted values.
        validation_setting (str): The validation setting of the current problem. This is used to determine if some averaging methods make sense to be calculated given the validation setting.
        problem_mode (str): The type of task of the given problem. Possible values are classification and regression.
        metrics (list): The performance metrics that will be calculated.
        averaging (list): The averaging strategy that will be used to calculate the metric.
        slices_arr (np.array, optional): An array with the slice ids. Slice-aware evaluation is not implemented and non-None values are rejected. Defaults to None.
        verbose (bool, optional): Whether or not to print useful info in the terminal. Defaults to False.
        per_target_verbose (bool, optional): Whether or not to print useful info per target in the terminal. Defaults to False.
        per_instance_verbose (bool, optional): Whether or not to print useful info per instance in the terminal. Defaults to False.
        top_k (_type_, optional): The number of top performing instances or targets that are used to calculate metrics. Defaults to None.
        return_detailed_macro (bool, optional): Whether to include per-target metrics in addition to macro averages. Requires macro averaging. Defaults to False.
        train_true_value (_type_, optional): The training mean for every target, required when calculating macro RRMSE. Defaults to None.
        scaler_per_target (_type_, optional): One global scaler for setting A or a scaler for every target in settings B-D. Supported for regression scores only. Defaults to None.

    Raises:
        AttributeError: only when the problem setting is A, but a macro or instance wise averaging is requested

    Returns:
        dict: A dictionary with key:value pairs of metric_name: metric_value
    """
    _validate_evaluation_inputs(
        instances_arr,
        targets_arr,
        true_values_arr,
        pred_values_arr,
        validation_setting,
        problem_mode,
        metrics,
        averaging,
        slices_arr,
        return_detailed_macro,
        top_k,
        train_true_value,
        scaler_per_target,
    )
    final_result: MetricResults = {}
    if mode != "":
        mode += "_"

    if verbose:  # pragma: no cover
        print(
            "There are "
            + str(np.count_nonzero(np.array(pred_values_arr)))
            + " non zero predictions"
        )
        if np.isnan(np.asarray(true_values_arr, dtype=float)).any():
            print("NaN value in true_values_arr")
        if np.isnan(np.asarray(pred_values_arr, dtype=float)).any():
            print("NaN value in pred_values_arr")

    # package all arrays into a dataframe (will have to check if this implementation is faster than the numpy version)
    df = pd.DataFrame(
        {
            "instance_id": instances_arr,
            "target_id": targets_arr,
            "true_values": true_values_arr,
            "pred_values": pred_values_arr,
        }
    )

    if verbose:
        print("========== " + str(mode) + " ==========")  # pragma: no cover

    # for Setting A, the only averating option that makes sense is the micro version.
    if validation_setting == "A":
        # In setting A one scaler covers the entire score matrix.
        if scaler_per_target is not None:
            if isinstance(scaler_per_target, Mapping):
                raise TypeError(
                    "Setting A expects one scaler, not a mapping of target scalers."
                )
            scaler = scaler_per_target
            df["true_values"] = _inverse_transform_values(
                scaler,
                df["true_values"],
            )
            df["pred_values"] = _inverse_transform_values(
                scaler,
                df["pred_values"],
            )
            if verbose:  # pragma: no cover
                print("These are the unscaled values: ")
                print(
                    "Unscaled True_values: "
                    + str(df[["true_values", "pred_values"]].head())
                )

        results = base_evaluator(df, problem_mode, metrics, -1, verbose=verbose)
        # iterate the metric_name, metric_value pairs
        for metric_name, metric_val in results.items():
            final_result.update({mode + metric_name + "_micro": metric_val})
            if verbose:  # pragma: no cover
                print(metric_name + "_micro: " + str(metric_val))

    # for Setting B, micro, macro and instance-wise averaging version should be available. In settings B,C and probably D, you have a scaler per target.
    elif validation_setting in ["B", "C", "D"]:
        if scaler_per_target is not None:
            if not isinstance(scaler_per_target, Mapping):
                raise TypeError(
                    f"Setting {validation_setting} expects a mapping of target scalers."
                )
            df["true_values"] = df.apply(
                target_wise_true_inverse_transformation,
                scaler_per_target=scaler_per_target,
                axis=1,
            )
            df["pred_values"] = df.apply(
                target_wise_pred_inverse_transformation,
                scaler_per_target=scaler_per_target,
                axis=1,
            )
            if verbose:  # pragma: no cover
                print(
                    "Unscaled True_values: "
                    + str(df[["true_values", "pred_values"]].head())
                )

        if "micro" in averaging:
            results = base_evaluator(df, problem_mode, metrics, -1, verbose=verbose)
            # iterate the metric_name, metric_value pairs
            for metric_name, metric_val in results.items():
                final_result.update({mode + metric_name + "_micro": metric_val})
                if verbose:  # pragma: no cover
                    print(metric_name + "_micro: " + str(metric_val))

        if "macro" in averaging:
            results_per_target = df.groupby("target_id").apply(
                base_evaluator,
                problem_mode=problem_mode,
                metrics=metrics,
                idx=None,
                train_true_value=train_true_value,
                verbose=verbose,
            )
            if return_detailed_macro:
                for target_i, row in results_per_target.items():
                    final_result.update(
                        {
                            mode + k + "_target_" + str(target_i): v
                            for k, v in row.items()
                        }
                    )
            if per_target_verbose:  # pragma: no cover
                print("Per target results: " + str(results_per_target))

            macro_results_per_metric = results_per_target.apply(pd.Series).mean()
            final_result.update(
                {
                    mode + k + "_macro": v
                    for k, v in macro_results_per_metric.to_dict().items()
                }
            )
            if verbose:  # pragma: no cover
                print("Macro results: " + str(macro_results_per_metric))

            if top_k is not None:
                top_k_results_per_target = (
                    df.sort_values(
                        ["target_id", "pred_values"], ascending=[True, False]
                    )
                    .groupby("target_id")
                    .head(top_k)
                    .reset_index(drop=True)
                    .groupby("target_id")
                    .apply(
                        base_evaluator,
                        problem_mode=problem_mode,
                        metrics=metrics,
                        idx=None,
                        train_true_value=train_true_value,
                        verbose=verbose,
                    )
                )
                if return_detailed_macro:
                    for target_i, row in top_k_results_per_target.items():
                        final_result.update(
                            {
                                mode
                                + "top_"
                                + str(top_k)
                                + "_"
                                + k
                                + "_target_"
                                + str(target_i): v
                                for k, v in row.items()
                            }
                        )
                if per_target_verbose:  # pragma: no cover
                    print(
                        "Per target top_"
                        + str(top_k)
                        + "results: "
                        + str(top_k_results_per_target)
                    )

                top_k_macro_results_per_metric = top_k_results_per_target.apply(
                    pd.Series
                ).mean()
                final_result.update(
                    {
                        mode + "top_" + str(top_k) + "_" + k + "_macro": v
                        for k, v in top_k_macro_results_per_metric.to_dict().items()
                    }
                )
                if verbose:  # pragma: no cover
                    print("Macro results: " + str(top_k_macro_results_per_metric))

        if "instance" in averaging:
            results_per_instance = df.groupby("instance_id").apply(
                base_evaluator,
                problem_mode=problem_mode,
                metrics=metrics,
                idx=None,
                verbose=verbose,
            )
            # iterate over the instances
            for instance_i, row in results_per_instance.items():
                final_result.update(
                    {
                        mode + k + "_instance_" + str(instance_i): v
                        for k, v in row.items()
                    }
                )
            if per_instance_verbose:  # pragma: no cover
                print("Per instance results: " + str(results_per_instance))

            instance_results_per_metric = results_per_instance.apply(pd.Series).mean()
            final_result.update(
                {
                    mode + k + "_instance": v
                    for k, v in instance_results_per_metric.to_dict().items()
                }
            )
            if verbose:  # pragma: no cover
                print("Instance results: " + str(instance_results_per_metric))

            if top_k is not None:
                top_k_results_per_instance = (
                    df.sort_values(
                        ["instance_id", "pred_values"], ascending=[True, False]
                    )
                    .groupby("instance_id")
                    .head(top_k)
                    .reset_index(drop=True)
                    .groupby("instance_id")
                    .apply(
                        base_evaluator,
                        problem_mode=problem_mode,
                        metrics=metrics,
                        idx=None,
                        verbose=verbose,
                    )
                )
                for instance_i, row in top_k_results_per_instance.items():
                    final_result.update(
                        {
                            mode
                            + "top_"
                            + str(top_k)
                            + "_"
                            + k
                            + "_instance_"
                            + str(instance_i): v
                            for k, v in row.items()
                        }
                    )
                if per_instance_verbose:  # pragma: no cover
                    print(
                        "Per instance top_"
                        + str(top_k)
                        + "results: "
                        + str(top_k_results_per_instance)
                    )

                top_k_instance_results_per_metric = top_k_results_per_instance.apply(
                    pd.Series
                ).mean()
                final_result.update(
                    {
                        mode + "top_" + str(top_k) + "_" + k + "_instance": v
                        for k, v in top_k_instance_results_per_metric.to_dict().items()
                    }
                )
                if verbose:  # pragma: no cover
                    print("Instance results: " + str(top_k_instance_results_per_metric))

    if verbose:  # pragma: no cover
        print("==================================")
        print("")

    return final_result


# with micro-averaging you loose the notion of multiple-targets. You just simplify the problem and assume you are working with just one target. The train_true_value variable is only used for the calculation of the relative root mean squared error RRMSE.
def base_evaluator(
    rows: pd.DataFrame,
    problem_mode: str,
    metrics: Sequence[str],
    idx: Any,
    train_true_value: TrainingMeans | None = None,
    verbose: bool = False,
    threshold: float = 0.5,
) -> MetricResults:
    """The function that actually calculates the different metrics

    Args:
        true_values_arr (numpy.array): An array with the true values.
        pred_values_arr (numpy.array): An array with the predicted values.
        problem_mode (str): The type of task of the given problem. Possible values are classification and regression.
        metrics (list): The performance metrics that will be calculated.
        idx (int): The id of the instance of target.
        train_true_value (numpy.array, optional): The true values per target. This is used when calculating the RRMSE score. Defaults to None.
        threshold (float): The threshold used to binarize the predictions
    Returns:
        dict: a dictionary with the results per metric
    """
    if not 0 <= threshold <= 1:
        raise ValueError("threshold must be between 0 and 1.")
    if problem_mode not in {"classification", "regression", "ranking"}:
        raise ValueError(
            "problem_mode must be 'classification', 'regression', or 'ranking'; "
            f"received {problem_mode!r}."
        )
    _validate_metrics(problem_mode, metrics)

    results: MetricResults = {}

    true_values, pred_values = rows["true_values"], rows["pred_values"]

    if idx is None:
        idx = getattr(rows, "name", None)

    training_mean: float | None = None
    if train_true_value is not None and idx not in (None, -1):
        training_mean = float(train_true_value[idx])

    if problem_mode == "regression":
        if "RMSE" in metrics:
            # ``squared`` was removed from mean_squared_error in newer
            # scikit-learn releases. The explicit square root works across the
            # old and modern APIs.
            results["RMSE"] = float(
                np.sqrt(mean_squared_error(true_values, pred_values))
            )
        if "MSE" in metrics:
            results["MSE"] = float(mean_squared_error(true_values, pred_values))
        if "MAE" in metrics:
            results["MAE"] = float(mean_absolute_error(true_values, pred_values))
        if "R2" in metrics:
            results["R2"] = float(r2_score(true_values, pred_values))
        if "RRMSE" in metrics:
            if training_mean is None:
                raise ValueError(
                    "RRMSE requires a training mean for each evaluated group."
                )
            numerator = float(np.sqrt(np.mean(np.square(true_values - pred_values))))
            denominator = float(
                np.sqrt(
                    np.mean(
                        np.square(
                            true_values
                            - [training_mean for _ in range(len(pred_values))]
                        )
                    )
                )
            )
            results["RRMSE"] = (
                float("nan") if denominator == 0 else numerator / denominator
            )

    elif problem_mode == "classification":
        bin_values = (pred_values >= threshold).astype(int)

        if idx == -1:
            (
                results["tn"],
                results["fp"],
                results["fn"],
                results["tp"],
            ) = [
                float(value)
                for value in confusion_matrix(
                    true_values,
                    bin_values,
                    labels=[0, 1],
                ).ravel()
            ]

        if "accuracy" in metrics:
            results["accuracy"] = float(accuracy_score(true_values, bin_values))
        if "recall" in metrics:
            results["recall"] = float(
                recall_score(true_values, bin_values, zero_division=0)
            )
        if "precision" in metrics:
            results["precision"] = float(
                precision_score(true_values, bin_values, zero_division=0)
            )
        if "f1_score" in metrics:
            results["f1_score"] = float(
                f1_score(true_values, bin_values, zero_division=0)
            )
        if "hamming_loss" in metrics:
            results["hamming_loss"] = float(hamming_loss(true_values, bin_values))

        if true_values.nunique() > 1:
            if "auroc" in metrics:
                results["auroc"] = float(roc_auc_score(true_values, pred_values))
            if "aupr" in metrics:
                precision, recall, _thresholds = precision_recall_curve(
                    true_values, pred_values
                )
                results["aupr"] = float(auc(recall, precision))
        else:
            if "auroc" in metrics:
                results["auroc"] = float("nan")
            if "aupr" in metrics:
                results["aupr"] = float("nan")
            if verbose:  # pragma: no cover
                print(
                    "Warning: instance"
                    + str(idx)
                    + " has "
                    + str(true_values.nunique())
                    + " unique true values"
                )
        selected_epilepsy_metrics = set(metrics).intersection(EPILEPSY_METRICS)
        if selected_epilepsy_metrics:
            results.update(
                get_epilepsy_specific_metrics(
                    true_values,
                    bin_values,
                    metrics=sorted(selected_epilepsy_metrics),
                )
            )

    elif problem_mode == "ranking":
        if "ndcg" in metrics:
            if len(true_values) < 2:
                results["ndcg"] = float("nan")
            else:
                results["ndcg"] = float(
                    ndcg_score(
                        np.asarray(true_values, dtype=float).reshape(1, -1),
                        np.asarray(pred_values, dtype=float).reshape(1, -1),
                    )
                )

    return results


def get_epilepsy_specific_metrics(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    metrics: Sequence[str] | None = None,
) -> MetricResults:  # pragma: no cover
    """Function that calculates specific metrics used in Epilepsy prediction tasks

    Args:
        y_true (numpy.array): An array with the true values.
        y_pred (_type_): An array with the predicted values.
        metrics (list, optional): A list with metric names that have to be calculated. Defaults to [ 'sensitivity', 'false_alarm_rate_per_hour', 'positive_predictive_value', 'f1_score_epilepsy_version', ].

    Returns:
        dict: a dictionary with the results per metric
    """
    selected_metrics = (
        [
            "sensitivity",
            "false_alarm_rate_per_hour",
            "positive_predictive_value",
            "f1_score_epilepsy_version",
        ]
        if metrics is None
        else list(metrics)
    )
    unknown_metrics = set(selected_metrics).difference(EPILEPSY_METRICS)
    if unknown_metrics:
        raise ValueError(f"Unsupported epilepsy metrics: {sorted(unknown_metrics)}.")

    true_array = np.asarray(y_true)
    predicted_array = np.asarray(y_pred)
    if true_array.ndim != 1 or predicted_array.ndim != 1:
        raise ValueError("Epilepsy metrics require one-dimensional inputs.")
    if len(true_array) != len(predicted_array):
        raise ValueError("y_true and y_pred must have equal length.")
    if len(true_array) == 0:
        raise ValueError("Epilepsy metrics require at least one observation.")

    results: MetricResults = {}
    # define consecutive epileptic moments
    true_episodes = [
        list(group) for group in mit.consecutive_groups(np.where(true_array == 1)[0])
    ]

    TP_OVLP = 0
    FN_OVLP = 0
    for i in range(len(true_episodes)):
        episode_start = true_episodes[i][0]
        episode_end = true_episodes[i][-1] + 1
        if np.sum(predicted_array[episode_start:episode_end]) > 0:
            TP_OVLP += 1
        else:
            FN_OVLP += 1

    # new implementation of calculating FPs
    for FP_window in [1, 10, 60]:
        FP = 0
        suffix = "_" + str(FP_window)

        # calculate FPs per sample (stored as True, False values in an arry)
        FPs = np.array(
            [
                (true_array[i] == 0) and (predicted_array[i] == 1)
                for i in range(len(predicted_array))
            ]
        )

        # create chunks for consequtive observations using a user-defined window size
        FP_chunks = list(mit.chunked(FPs, FP_window))
        for c in FP_chunks:
            if np.array(c).any():
                FP += 1

        # print('FP: '+str(FP))

        if "sensitivity" in selected_metrics:
            results["sensitivity" + suffix] = float(
                TP_OVLP / (TP_OVLP + FN_OVLP) if (TP_OVLP + FN_OVLP) != 0 else 0
            )
        if "false_alarm_rate_per_hour" in selected_metrics:
            results["false_alarm_rate_per_hour" + suffix] = float(
                FP / (len(true_array) / 3600)
            )
        if "positive_predictive_value" in selected_metrics:
            results["positive_predictive_value" + suffix] = float(
                TP_OVLP / (TP_OVLP + FP) if (TP_OVLP + FP) != 0 else 0
            )
        if "f1_score_epilepsy_version" in selected_metrics:
            results["f1_score" + suffix] = float(
                (2 * TP_OVLP / (2 * TP_OVLP + FN_OVLP + FP))
                if (2 * TP_OVLP + FN_OVLP + FP)
                else 0
            )
    return results


def _validate_metrics(problem_mode: str, metrics: Sequence[str]) -> None:
    unknown_metrics = set(metrics).difference(SUPPORTED_METRICS[problem_mode])
    if unknown_metrics:
        raise ValueError(
            f"Unsupported {problem_mode} metrics: {sorted(unknown_metrics)}."
        )


def _validate_target_mapping(
    values: Mapping[Any, Any],
    targets: ArrayValues,
    *,
    parameter_name: str,
    require_finite_values: bool = False,
    require_inverse_transformers: bool = False,
) -> None:
    target_ids = set(targets)
    missing_targets = target_ids.difference(values)
    if missing_targets:
        raise ValueError(
            f"{parameter_name} is missing target IDs: "
            f"{sorted(missing_targets, key=str)!r}."
        )
    if require_finite_values:
        for target_id in target_ids:
            try:
                value = float(values[target_id])
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"{parameter_name}[{target_id!r}] must be numeric."
                ) from exc
            if not np.isfinite(value):
                raise ValueError(f"{parameter_name}[{target_id!r}] must be finite.")
    if require_inverse_transformers:
        for target_id in target_ids:
            _validate_inverse_transformer(
                values[target_id],
                parameter_name=f"{parameter_name}[{target_id!r}]",
            )


def _validate_inverse_transformer(
    value: Any,
    *,
    parameter_name: str,
) -> None:
    if not callable(getattr(value, "inverse_transform", None)):
        raise TypeError(f"{parameter_name} must provide inverse_transform().")
