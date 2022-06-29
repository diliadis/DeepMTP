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
)
import numpy as np
import more_itertools as mit


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
    '''Calculates all the metrics using different averaging schemes

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
    '''

    final_result = {}
    if mode != '':
        mode += '_'

    if verbose:
        print(
            "There are "
            + str(np.count_nonzero(np.array(pred_values_arr)))
            + " non zero predictions"
        )
        if np.NaN in true_values_arr:
            print("NaN value in true_values_arr")
        if np.NaN in pred_values_arr:
            print("NaN value in pred_values_arr")

    if isinstance(true_values_arr, list):
        true_values_arr = np.array(true_values_arr)
    if isinstance(pred_values_arr, list):
        pred_values_arr = np.array(pred_values_arr).flatten()
    if isinstance(instances_arr, list):
        instances_arr = np.array(instances_arr)
    if isinstance(targets_arr, list):
        targets_arr = np.array(targets_arr)

    if verbose:
        print("train_true_value: " + str(train_true_value))
        print("True_values length: " + str(len(true_values_arr)))
        print("Predicted_values length: " + str(len(pred_values_arr)))
        print("instances_arr: " + str(instances_arr[:10]))
        print("targets_arr: " + str(targets_arr[:10]))
        print("True_values: " + str(true_values_arr[:10]))
        print("Predicted_values: " + str(pred_values_arr[:10]))

    values_per_metric = {m: [] for m in metrics}

    if verbose: print("========== " + str(mode) + " ==========")

    # for Setting A, the only averating option that makes sense is the micro version.
    if validation_setting == "A":
        # check if values are scaled and if so, inverse_transform them. In setting A you have a single scaler for the entire score matrix.
        if scaler_per_target is not None:
            true_values_arr = scaler_per_target.inverse_transform(
                np.reshape(true_values_arr, (-1, 1))
            ).flatten()
            pred_values_arr = scaler_per_target.inverse_transform(
                np.reshape(pred_values_arr, (-1, 1))
            ).flatten()
            print("These are the unscaled values: ")
            print("Unscaled True_values: " + str(true_values_arr[:10]))
            print("Unscaled Predicted_values: " + str(pred_values_arr[:10]))

        if "micro" in averaging:
            results = base_evaluator(
                true_values_arr, pred_values_arr, problem_mode, metrics, -1
            )
            # iterate the metric_name, metric_value pairs
            for metric_name, metric_val in results.items():
                final_result.update({mode + metric_name + "_micro": metric_val})
                if verbose:
                    print(metric_name + "_micro: " + str(metric_val))

        else:
            raise AttributeError('Only the micro-averaging option is compatible with setting A')

    # for Setting B, micro, macro and instance-wise averaging version should be available. In settings B,C and probably D, you have a scaler per target.
    elif validation_setting in ['B', 'C', 'D']:

        if scaler_per_target is not None:
            # check if values are scaled and if so, inverse_transform them

            index_arr_per_target = {}
            for target_id in np.unique(targets_arr):
                index_arr_per_target[target_id]  = np.where(np.array(targets_arr)==target_id)[0]

            for target_i, idxs in index_arr_per_target.items():
                true_values_arr[idxs] = (
                    scaler_per_target[target_i]
                    .inverse_transform(np.reshape(true_values_arr[idxs], (-1, 1)))
                    .flatten()
                )
                pred_values_arr[idxs] = (
                    scaler_per_target[target_i]
                    .inverse_transform(np.reshape(pred_values_arr[idxs], (-1, 1)))
                    .flatten()
                )

            if verbose:
                print("Unscaled True_values: " + str(true_values_arr[:10]))
                print("Unscaled Predicted_values: " + str(pred_values_arr[:10]))

        if "micro" in averaging:
            results = base_evaluator(
                true_values_arr, pred_values_arr, problem_mode, metrics, -1
            )
            # iterate the metric_name, metric_value pairs
            for metric_name, metric_val in results.items():
                final_result.update({mode + metric_name + "_micro": metric_val})
                if verbose:
                    print(metric_name + "_micro: " + str(metric_val))

        if "macro" in averaging:

            index_arr_per_target = {}
            for target_id in np.unique(targets_arr):
                index_arr_per_target[target_id]  = np.where(np.array(targets_arr)==target_id)[0]

            # iterate over the targets
            for target_i, idxs in index_arr_per_target.items():
                if (len(np.unique(true_values_arr[idxs])) > 1) or (len(np.unique(pred_values_arr[idxs])) > 1):

                    results = base_evaluator(
                        true_values_arr[idxs],
                        pred_values_arr[idxs],
                        problem_mode,
                        metrics,
                        target_i,
                        train_true_value=None if train_true_value is None else train_true_value[target_i],
                    )

                    # this prints and/or logs the performance metrics per target
                    for metric_name, metric_val in results.items():
                        if metric_name not in values_per_metric:
                            values_per_metric[metric_name] = []
                        values_per_metric[metric_name].append(metric_val)
                        final_result.update(
                            {
                                mode
                                + metric_name
                                + "_target_"
                                + str(target_i): metric_val
                            }
                        )
                        if per_target_verbose:
                            print(
                                metric_name
                                + "_target_"
                                + str(target_i)
                                + ": "
                                + str(metric_val)
                            )
                else:
                    if verbose:
                        print(
                            "Warning: Target"
                            + str(target_i)
                            + " has "
                            + str(len(np.unique(true_values_arr[idxs])))
                            + " unique true values and "
                            + str(len(np.unique(pred_values_arr[idxs])))
                            + " unique predictions"
                        )

            for metric_name in values_per_metric.keys():
                avg_val = np.mean(values_per_metric[metric_name])
                final_result.update({mode + metric_name + "_macro": avg_val})
                if verbose:
                    print(metric_name + "_macro: " + str(avg_val))

        if "instance" in averaging:

            index_arr_per_instance = {}
            for instance_i in np.unique(instances_arr):
                index_arr_per_instance[instance_i]  = np.where(np.array(instances_arr)==instance_i)[0]

            # iterate over the instances
            for instance_i, idxs in index_arr_per_instance.items():
                if (len(np.unique(true_values_arr[idxs])) > 1) and (len(np.unique(pred_values_arr[idxs])) > 1):

                    results = base_evaluator(
                        true_values_arr[idxs],
                        pred_values_arr[idxs],
                        problem_mode,
                        metrics,
                        instance_i,
                        train_true_value=None,
                    )

                    # this prints and/or logs the performance metrics per instance
                    for metric_name, metric_val in results.items():
                        if metric_name not in values_per_metric:
                            values_per_metric[metric_name] = []
                        values_per_metric[metric_name].append(metric_val)
                        final_result.update(
                            {
                                mode
                                + metric_name
                                + "_instance_"
                                + str(target_i): metric_val
                            }
                        )
                        if per_instance_verbose:
                            print(
                                metric_name
                                + "_instance_"
                                + str(instance_i)
                                + ": "
                                + str(metric_val)
                            )

                else:
                    if verbose:
                        print(
                            "Warning: Instance"
                            + str(instance_i)
                            + " has "
                            + str(len(np.unique(true_values_arr[idxs])))
                            + " unique true values and "
                            + str(len(np.unique(pred_values_arr[idxs])))
                            + " unique predictions"
                        )

            for metric_name in metrics:
                avg_val = np.mean(values_per_metric[metric_name])
                final_result.update({mode + metric_name + "_instance": avg_val})
                if verbose:
                    print(metric_name + "_instance: " + str(avg_val))

    if verbose:
        print("==================================")
        print("")

    return final_result


# with micro-averaging you loose the notion of multiple-targets. You just simplify the problem and assume you are working with just one target. The train_true_value variable is only used for the calculation of the relative root mean squared error RRMSE.
def base_evaluator(
    true_values_arr, pred_values_arr, problem_mode, metrics, idx, train_true_value=None
):
    '''The function that actually calculates the different metrics

    Args:
        true_values_arr (numpy.array): An array with the true values.
        pred_values_arr (numpy.array): An array with the predicted values.
        problem_mode (str): The type of task of the given problem. Possible values are classification and regression. 
        metrics (list): The performance metrics that will be calculated.
        idx (int): The id of the instance of target.
        train_true_value (numpy.array, optional): The true values per target. This is used when calculating the RRMSE score. Defaults to None.

    Returns:
        dict: a dictionary with the results per metric
    '''
    results = {}

    if problem_mode == "regression":

        if "RMSE" in metrics:
            results["RMSE"] = np.sqrt(np.mean(np.square(true_values_arr - pred_values_arr)))
        if "MSE" in metrics:
            results["MSE"] = np.mean(np.square(true_values_arr - pred_values_arr))
        if "MAE" in metrics:
            results["MAE"] = np.mean(np.abs(true_values_arr - pred_values_arr))
        if "R2" in metrics:
            results["R2"] = r2_score(true_values_arr, pred_values_arr)
        if "RRMSE" in metrics:
            if train_true_value is None:
                results["RRMSE"] = np.nan
                # raise Exception(
                #     "Requested RRMSE without suplying the average target values from the training set"
                # )
            else:
                results["RRMSE"] = np.sqrt(
                    np.mean(np.square(true_values_arr - pred_values_arr))
                ) / np.sqrt(
                    np.mean(
                        np.square(
                            true_values_arr
                            - [train_true_value for i in range(len(pred_values_arr))]
                        )
                    )
                )

    elif problem_mode == "classification":
        bin_arr = np.where(pred_values_arr > 0.5, 1, 0)

        if idx == -1:
            (
                results["tn"],
                results["fp"],
                results["fn"],
                results["tp"],
            ) = confusion_matrix(true_values_arr, bin_arr).ravel()

        if "accuracy" in metrics:
            results["accuracy"] = accuracy_score(true_values_arr, bin_arr)
        if "recall" in metrics:
            results["recall"] = recall_score(true_values_arr, bin_arr, zero_division=0)
        if "precision" in metrics:
            results["precision"] = precision_score(
                true_values_arr, bin_arr, zero_division=0
            )
        if "f1_score" in metrics:
            results["f1_score"] = f1_score(true_values_arr, bin_arr, zero_division=0)
        if "hamming_loss" in metrics:
            results["hamming_loss"] = hamming_loss(true_values_arr, bin_arr)
        if len(np.unique(true_values_arr)) > 1:
            if "auroc" in metrics:
                results["auroc"] = roc_auc_score(true_values_arr, pred_values_arr)
            if "aupr" in metrics:
                precision, recall, thresholds = precision_recall_curve(
                    true_values_arr, pred_values_arr
                )
                results["aupr"] = auc(recall, precision)
        else:
            print(
                "Warning: instance"
                + str(idx)
                + " has "
                + str(len(np.unique(true_values_arr)))
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
                    true_values_arr,
                    bin_arr,
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
):
    """Function that calculates specific metrics used in Epilepsy prediction tasks

    Args:
        y_true (numpy.array): An array with the true values.
        y_pred (_type_): An array with the predicted values.
        metrics (list, optional): A list with metric names that have to be calculated. Defaults to [ "sensitivity", "false_alarm_rate_per_hour", "positive_predictive_value", "f1_score_epilepsy_version", ].

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
