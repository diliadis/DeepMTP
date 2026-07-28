import pytest

from DeepMTP import DeepMTPConfig, EpochOutput, EvaluationPolicy, Evaluator


class RecordingEvaluationProgress:
    def __init__(self):
        self.events = []

    def on_event(self, event):
        self.events.append(event)


def make_config(**overrides):
    values = {
        "validation_setting": "A",
        "problem_mode": "classification",
        "compute_mode": "cpu",
        "metrics": ["accuracy"],
        "metrics_average": ["micro"],
        "instance_branch_architecture": "MLP",
        "instance_branch_input_dim": 1,
        "target_branch_architecture": "MLP",
        "target_branch_input_dim": 1,
    }
    values.update(overrides)
    return DeepMTPConfig(**values)


def test_evaluation_policy_covers_test_interval_metric_and_last_epoch():
    policy = EvaluationPolicy(
        evaluate_train=True,
        evaluate_val=False,
        eval_every_n_epochs=3,
        num_epochs=5,
        metric_to_optimize_early_stopping="loss",
    )

    assert policy.should_evaluate("test", epoch=1)
    assert not policy.should_evaluate("val", epoch=1)
    assert policy.should_evaluate("val", epoch=4)
    assert policy.should_evaluate_training(epoch=0)
    assert not policy.should_evaluate_training(epoch=1)
    assert policy.should_evaluate_training(epoch=4)

    metric_policy = EvaluationPolicy(
        evaluate_train=False,
        evaluate_val=False,
        eval_every_n_epochs=10,
        num_epochs=100,
        metric_to_optimize_early_stopping="accuracy_micro",
    )
    assert metric_policy.should_evaluate("val", epoch=7)
    assert not metric_policy.should_evaluate_training(epoch=0)


def test_evaluation_policy_uses_best_epoch_metric_without_early_stopping():
    policy = EvaluationPolicy.from_config(
        make_config(
            evaluate_val=False,
            eval_every_n_epochs=10,
            num_epochs=100,
            use_early_stopping=False,
            metric_to_optimize_early_stopping="loss",
            metric_to_optimize_best_epoch_selection="accuracy_micro",
        )
    )

    assert policy.metric_to_track == "accuracy_micro"
    assert policy.should_evaluate("val", epoch=7)


def test_evaluator_calculates_metrics_from_epoch_output():
    evaluator = Evaluator(make_config())
    output = EpochOutput(
        true_values=[0.0, 1.0, 1.0, 0.0],
        predicted_values=[0.1, 0.9, 0.8, 0.2],
        instance_ids=[0, 1, 2, 3],
        target_ids=[0, 0, 0, 0],
    )

    results = evaluator.evaluate("val", epoch=0, output=output)

    assert results["val_accuracy_micro"] == pytest.approx(1.0)


def test_evaluator_uses_injected_progress_without_stdout(capsys):
    progress = RecordingEvaluationProgress()
    evaluator = Evaluator(
        make_config(results_verbose=True),
        progress=progress,
    )
    output = EpochOutput(
        true_values=[0.0, 1.0],
        predicted_values=[0.1, 0.9],
        instance_ids=[0, 1],
        target_ids=[0, 0],
    )

    results = evaluator.evaluate("val", epoch=0, output=output)

    assert results["val_accuracy_micro"] == pytest.approx(1.0)
    assert [event.kind for event in progress.events] == [
        "input_inspected",
        "evaluation_started",
        "metric_calculated",
        "metric_calculated",
        "metric_calculated",
        "metric_calculated",
        "metric_calculated",
        "evaluation_completed",
    ]
    assert capsys.readouterr().out == ""


def test_evaluator_honors_per_target_result_option():
    output = EpochOutput(
        true_values=[0.0, 1.0, 0.0, 1.0],
        predicted_values=[0.1, 0.9, 0.2, 0.8],
        instance_ids=[0, 1, 0, 1],
        target_ids=[0, 0, 1, 1],
    )
    summary_evaluator = Evaluator(
        make_config(validation_setting="B", metrics_average=["macro"])
    )
    detailed_evaluator = Evaluator(
        make_config(
            validation_setting="B",
            metrics_average=["macro"],
            return_results_per_target=True,
        )
    )

    summary = summary_evaluator.evaluate("val", epoch=0, output=output)
    detailed = detailed_evaluator.evaluate("val", epoch=0, output=output)

    assert summary["val_accuracy_macro"] == pytest.approx(1.0)
    assert not any("_target_" in key for key in summary)
    assert detailed["val_accuracy_target_0"] == pytest.approx(1.0)
    assert detailed["val_accuracy_target_1"] == pytest.approx(1.0)


def test_evaluator_returns_no_metrics_when_none_are_configured():
    evaluator = Evaluator(make_config(metrics=[], metrics_average=[]))

    assert evaluator.evaluate("test", 0, EpochOutput()) == {}


def test_prediction_frame_has_stable_columns_and_values():
    output = EpochOutput(
        true_values=[1.0],
        predicted_values=[0.75],
        instance_ids=[4],
        target_ids=[9],
    )

    predictions = Evaluator.predictions_frame(output)

    assert list(predictions.columns) == [
        "instance_id",
        "target_id",
        "true_values",
        "predicted_values",
    ]
    assert predictions.iloc[0].to_dict() == {
        "instance_id": 4.0,
        "target_id": 9.0,
        "true_values": 1.0,
        "predicted_values": 0.75,
    }
