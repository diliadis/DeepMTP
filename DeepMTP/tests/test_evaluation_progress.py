from DeepMTP import (
    ConsoleEvaluationProgressObserver,
    EvaluationProgressEvent,
    NullEvaluationProgressObserver,
)


def test_console_evaluation_progress_renders_structured_details(capsys):
    observer = ConsoleEvaluationProgressObserver()

    observer.on_event(
        EvaluationProgressEvent(
            "input_inspected",
            nonzero_predictions=3,
            true_values_have_nan=True,
            predicted_values_have_nan=True,
        )
    )
    observer.on_event(EvaluationProgressEvent("evaluation_started", mode="test_"))
    observer.on_event(
        EvaluationProgressEvent(
            "values_unscaled",
            values_preview="preview",
            announce_unscaled_values=True,
        )
    )
    observer.on_event(
        EvaluationProgressEvent(
            "metric_calculated",
            metric_name="MSE",
            metric_value=0.25,
            averaging="micro",
        )
    )
    observer.on_event(
        EvaluationProgressEvent(
            "group_results_calculated",
            group="target",
            top_k=2,
            results="target rows",
        )
    )
    observer.on_event(
        EvaluationProgressEvent(
            "aggregate_results_calculated",
            averaging="macro",
            results="macro row",
        )
    )
    observer.on_event(
        EvaluationProgressEvent(
            "single_class_group",
            group_id=4,
            unique_true_values=1,
        )
    )
    observer.on_event(EvaluationProgressEvent("evaluation_completed"))

    output = capsys.readouterr().out
    assert "There are 3 non zero predictions" in output
    assert "NaN value in true_values_arr" in output
    assert "NaN value in pred_values_arr" in output
    assert "========== test_ ==========" in output
    assert "These are the unscaled values" in output
    assert "Unscaled True_values: preview" in output
    assert "MSE_micro: 0.25" in output
    assert "Per target top_2 results: target rows" in output
    assert "Macro results: macro row" in output
    assert "Warning: instance4 has 1 unique true values" in output
    assert "==================================" in output


def test_null_evaluation_progress_is_silent(capsys):
    observer = NullEvaluationProgressObserver()

    observer.on_event(
        EvaluationProgressEvent(
            "input_inspected",
            nonzero_predictions=3,
            true_values_have_nan=True,
        )
    )
    observer.on_event(
        EvaluationProgressEvent(
            "single_class_group",
            group_id=4,
            unique_true_values=1,
        )
    )

    assert capsys.readouterr().out == ""
