from sklearn.metrics import (
    f1_score,
    accuracy_score,
    hamming_loss,
    precision_score,
    recall_score,
    roc_auc_score,
    multilabel_confusion_matrix,
    average_precision_score,
    precision_recall_curve,
    auc,
    ndcg_score,
    mean_squared_error,
    confusion_matrix,
    r2_score,
    average_precision_score,
    mean_absolute_error,
    f1_score,
    ndcg_score,
)
import numpy as np
import pandas as pd
import more_itertools as mit


def instance_wise_true_inverse_transformation(
    row, scaler_per_instance
):  # add check for cases where there is only one unique value true value or predicted value
    return scaler_per_instance[row["instance_id"]].inverse_transform(row["true_value"])


def instance_wise_pred_inverse_transformation(row, scaler_per_instance):
    return scaler_per_instance[row["instance_id"]].inverse_transform(row["pred_value"])


def target_wise_true_inverse_transformation(row, scaler_per_target):
    return scaler_per_target[row["target_id"]].inverse_transform(row["true_value"])


def target_wise_pred_inverse_transformation(row, scaler_per_target):
    return scaler_per_target[row["target_id"]].inverse_transform(row["pred_value"])


def get_performance_results(
    mode,
    epoch_id,
    instances_arr,
    targets_arr,
    true_values_arr,
    pred_values_arr,
    validation_setting,
    problem_mode,
    metrics,
    averaging,
    slices_arr=None,
    verbose=False,
    per_target_verbose=False,
    per_instance_verbose=False,
    top_k=None,
    return_detailed_macro=False,
    train_true_value=None,
    scaler_per_target=None,
):
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
        slices_arr (np.array, optional): An array with the slice ids. This can be used when the problem has dyadic features, or in the case of tensor completion. Not currently implemented. Defaults to None.
        verbose (bool, optional): Whether or not to print useful info in the terminal. Defaults to False.
        per_target_verbose (bool, optional): Whether or not to print useful info per target in the terminal. Defaults to False.
        per_instance_verbose (bool, optional): Whether or not to print useful info per instance in the terminal. Defaults to False.
        top_k (_type_, optional): The number of top performing instances or targets that are used to calculate metrics. This is used for ranking tasks. Not currently implemented. Defaults to None.
        return_detailed_macro (bool, optional): Whether or not to return the per-target metrics. Defaults to False.
        train_true_value (_type_, optional): The true values per target. This is used when calculating the RRMSE score. Not currently implemented. Defaults to None.
        scaler_per_target (_type_, optional): The scaler of every target. This is used when the target values have been previously scaled. Not currently implemented. Defaults to None.

    Raises:
        AttributeError: only when the problem setting is A, but a macro or instance wise averaging is requested

    Returns:
        dict: A dictionary with key:value pairs of metric_name: metric_value
    """
    print("CALCULATING STUFF USING DATAFRAMES!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    final_result = {}
    if mode != "":
        mode += "_"

    if verbose:  # pragma: no cover
        print(
            "There are "
            + str(np.count_nonzero(np.array(pred_values_arr)))
            + " non zero predictions"
        )
        if np.NaN in true_values_arr:
            print("NaN value in true_values_arr")
        if np.NaN in pred_values_arr:
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
        # check if values are scaled and if so, inverse_transform them. In setting A you have a single scaler for the entire score matrix.
        if scaler_per_target is not None:
            df["true_values"] = scaler_per_target.inverse_transform(
                df["true_values"].values.reshape(-1, 1)
            )
            df["pred_values"] = scaler_per_target.inverse_transform(
                df["pred_values"].values.reshape(-1, 1)
            )
            if verbose:  # pragma: no cover
                print("These are the unscaled values: ")
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

        else:
            raise AttributeError(
                "Only the micro-averaging option is compatible with setting A"
            )

    # for Setting B, micro, macro and instance-wise averaging version should be available. In settings B,C and probably D, you have a scaler per target.
    elif validation_setting in ["B", "C", "D"]:
        if scaler_per_target is not None:
            # check if values are scaled and if so, inverse_transform them
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
                base_evaluator, problem_mode, metrics, train_true_value, verbose
            )
            # iterate over the targets
            for target_i, row in results_per_target.items():
                final_result.update(
                    {mode + k + "_target_" + str(target_i): v for k, v in row.items()}
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
                        base_evaluator, problem_mode, metrics, train_true_value, verbose
                    )
                )
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
                base_evaluator, problem_mode, metrics, None, verbose
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
                    .apply(base_evaluator, problem_mode, metrics, None, verbose)
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
def base_evaluator(  # pragma: no cover
    rows,
    problem_mode,
    metrics,
    idx,
    train_true_value=None,
    verbose=False,
    threshold=0.5,
):
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
    results = {}

    true_values, pred_values = rows["true_values"], rows["pred_values"]

    if len(rows) == 1:
        train_true_value = train_true_value[rows["target_id"]]

    if problem_mode == "regression":
        if "RMSE" in metrics:
            results["RMSE"] = np.sqrt(np.mean(np.square(true_values - pred_values)))
        if "MSE" in metrics:
            results["MSE"] = np.mean(np.square(true_values - pred_values))
        if "MAE" in metrics:
            results["MAE"] = np.mean(np.abs(true_values - pred_values))
        if "R2" in metrics:
            results["R2"] = r2_score(true_values, pred_values)
        if "RRMSE" in metrics:  # pragma: no cover
            if train_true_value is None:
                results["RRMSE"] = np.nan
                # raise Exception(
                #     'Requested RRMSE without suplying the average target values from the training set'
                # )
            else:
                results["RRMSE"] = np.sqrt(
                    np.mean(np.square(true_values - pred_values))
                ) / np.sqrt(
                    np.mean(
                        np.square(
                            true_values
                            - [train_true_value for i in range(len(pred_values))]
                        )
                    )
                )

    elif problem_mode == "classification":
        bin_values = (pred_values >= threshold).astype(int)

        if idx == -1:
            (
                results["tn"],
                results["fp"],
                results["fn"],
                results["tp"],
            ) = confusion_matrix(true_values, bin_values).ravel()

        if "accuracy" in metrics:
            results["accuracy"] = accuracy_score(true_values, bin_values)
        if "recall" in metrics:
            results["recall"] = recall_score(true_values, bin_values, zero_division=0)
        if "precision" in metrics:
            results["precision"] = precision_score(
                true_values, bin_values, zero_division=0
            )
        if "f1_score" in metrics:
            results["f1_score"] = f1_score(true_values, bin_values, zero_division=0)
        if "hamming_loss" in metrics:
            results["hamming_loss"] = hamming_loss(true_values, bin_values)

        if true_values.nunique() > 1:
            if "auroc" in metrics:
                results["auroc"] = roc_auc_score(true_values, pred_values)
            if "aupr" in metrics:
                precision, recall, thresholds = precision_recall_curve(
                    true_values, pred_values
                )
                results["aupr"] = auc(recall, precision)
        else:
            if verbose:  # pragma: no cover
                print(
                    "Warning: instance"
                    + str(idx)
                    + " has "
                    + str(true_values.nunique())
                    + " unique true values"
                )
        if set(metrics).intersection(
            set(
                [
                    "sensitivity",
                    "false_alarm_rate_per_hour",
                    "positive_predictive_value",
                    "f1_score_epilepsy_version",
                ]
            )
        ):
            results.update(
                get_epilepsy_specific_metrics(
                    true_values,
                    bin_values,
                    metrics=list(
                        set(metrics).intersection(
                            set(
                                [
                                    "sensitivity",
                                    "false_alarm_rate_per_hour",
                                    "positive_predictive_value",
                                    "f1_score_epilepsy_version",
                                ]
                            )
                        )
                    ),
                )
            )

    elif problem_mode == "ranking":
        if "ndcg" in metrics:
            results["ndcg"] = ndcg_score(true_values, pred_values)

    return results


def get_epilepsy_specific_metrics(
    y_true,
    y_pred,
    metrics=[
        "sensitivity",
        "false_alarm_rate_per_hour",
        "positive_predictive_value",
        "f1_score_epilepsy_version",
    ],
):  # pragma: no cover
    """Function that calculates specific metrics used in Epilepsy prediction tasks

    Args:
        y_true (numpy.array): An array with the true values.
        y_pred (_type_): An array with the predicted values.
        metrics (list, optional): A list with metric names that have to be calculated. Defaults to [ 'sensitivity', 'false_alarm_rate_per_hour', 'positive_predictive_value', 'f1_score_epilepsy_version', ].

    Returns:
        dict: a dictionary with the results per metric
    """
    results = {}
    # define consecutive epileptic moments
    true_episodes = [
        list(group) for group in mit.consecutive_groups(np.where(y_true == 1)[0])
    ]
    predicted_episodes = [
        list(group) for group in mit.consecutive_groups(np.where(y_pred == 1)[0])
    ]

    TP_OVLP = 0
    FN_OVLP = 0
    for i in range(len(true_episodes)):
        if np.sum(y_pred[true_episodes[i][0] : true_episodes[i][-1]]) > 0:
            TP_OVLP += 1
        else:
            FN_OVLP += 1

    # new implementation of calculating FPs
    for FP_window in [1, 10, 60]:
        FP = 0
        suffix = "_" + str(FP_window)

        # calculate FPs per sample (stored as True, False values in an arry)
        FPs = np.array(
            [((y_true[i] == 0) and (y_pred[i] == 1)) for i in range(len(y_pred))]
        )

        # create chunks for consequtive observations using a user-defined window size
        FP_chunks = list(mit.chunked(FPs, FP_window))
        for c in FP_chunks:
            if np.array(c).any():
                FP += 1

        # print('FP: '+str(FP))

        if "sensitivity" in metrics:
            results["sensitivity" + suffix] = (
                TP_OVLP / (TP_OVLP + FN_OVLP) if (TP_OVLP + FN_OVLP) != 0 else 0
            )
        if "false_alarm_rate_per_hour" in metrics:
            results["false_alarm_rate_per_hour" + suffix] = FP / (len(y_true) / 3600)
        if "positive_predictive_value" in metrics:
            results["positive_predictive_value" + suffix] = (
                TP_OVLP / (TP_OVLP + FP) if (TP_OVLP + FP) != 0 else 0
            )
        if "f1_score_epilepsy_version" in metrics:
            results["f1_score" + suffix] = (
                (2 * TP_OVLP / (2 * TP_OVLP + FN_OVLP + FP))
                if (2 * TP_OVLP + FN_OVLP + FP)
                else 0
            )
    return results
