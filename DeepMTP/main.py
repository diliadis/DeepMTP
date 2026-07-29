from __future__ import annotations

import copy
from collections.abc import Iterable, Mapping
from os import PathLike
from typing import Any, Literal, TypeAlias, cast, overload

import pandas as pd
import torch
from torch import nn

from DeepMTP.config import DeepMTPConfig
from DeepMTP.data.loading import DataLoaderFactory
from DeepMTP.models import (
    BranchFactory,
    ModelFactory,
    TwoBranchDotProductModel,
    TwoBranchKroneckerModel,
    TwoBranchMLPModel,
)
from DeepMTP.runtime.evaluation import Evaluator
from DeepMTP.runtime.persistence import ExperimentStore
from DeepMTP.runtime.presentation import (
    ConsoleProgressObserver,
    NullProgressObserver,
    ProgressEvent,
    ProgressObserver,
    RunHistory,
    build_progress_observer,
)
from DeepMTP.runtime.reporting import ExperimentReporter, build_reporter
from DeepMTP.runtime.training import TensorBatch, TrainingEngine
from DeepMTP.utils.model_utils import EarlyStopping

__all__ = [
    "DeepMTP",
    "TwoBranchDotProductModel",
    "TwoBranchKroneckerModel",
    "TwoBranchMLPModel",
]

ConfigInput: TypeAlias = DeepMTPConfig | Mapping[str, Any]
DatasetInput: TypeAlias = Mapping[str, Any]
CheckpointPath: TypeAlias = str | PathLike[str]
MetricResults: TypeAlias = dict[str, float]
PredictionResults: TypeAlias = tuple[MetricResults, pd.DataFrame]
TrainingState: TypeAlias = Literal["ready", "running", "completed", "failed"]


class DeepMTP:
    """Implements the training and inference logic of the DeepMTP framework."""

    def __init__(
        self,
        config: ConfigInput,
        instance_branch_model: BranchFactory | None = None,
        target_branch_model: BranchFactory | None = None,
        checkpoint_dir: CheckpointPath | None = None,
        reporter: ExperimentReporter | None = None,
        progress: ProgressObserver | None = None,
    ) -> None:
        self.checkpoint_dict: dict[str, Any] | None = None
        self.reporter: ExperimentReporter | None = reporter
        self._progress_was_injected = progress is not None
        self._training_state: TrainingState = "ready"

        if isinstance(config, DeepMTPConfig):
            config_overrides = config.to_dict(include_defaults=True)
        elif isinstance(config, Mapping):
            config_overrides = dict(config)
        else:
            raise TypeError(
                "config must be a mapping or DeepMTPConfig, "
                f"not {type(config).__name__}."
            )
        setup_progress = (
            progress
            if progress is not None
            else build_progress_observer(bool(config_overrides.get("verbose", False)))
        )

        # Load a checkpoint file, if it exists
        if checkpoint_dir is not None:
            setup_progress.on_event(
                ProgressEvent(
                    "checkpoint_loading_started",
                    message=str(checkpoint_dir),
                )
            )
            self.checkpoint_dict = ExperimentStore.load_checkpoint(
                checkpoint_dir, map_location="cpu"
            )
            setup_progress.on_event(
                ProgressEvent(
                    "checkpoint_loading_completed",
                    message=str(checkpoint_dir),
                )
            )

            # when a checkpoint is loaded, it's config can (or sometimes has to) be modified
            checkpoint_config = self.checkpoint_dict["config"]
            if isinstance(checkpoint_config, DeepMTPConfig):
                checkpoint_config = checkpoint_config.to_dict(include_defaults=True)
            if not isinstance(checkpoint_config, Mapping):
                raise ValueError(
                    "DeepMTP checkpoint config must be a mapping or DeepMTPConfig."
                )
            merged_config = dict(checkpoint_config)
            merged_config.update(config_overrides)
        else:
            merged_config = config_overrides

        self.configuration = DeepMTPConfig.from_mapping(merged_config)
        self.progress = (
            progress
            if progress is not None
            else build_progress_observer(self.configuration.verbose)
        )

        # initialize the device that will be used of training-inference
        self.device = torch.device(
            self.configuration.compute_mode
            if (torch.cuda.is_available() and "cuda" in self.configuration.compute_mode)
            else "cpu"
        )
        self.progress.on_event(
            ProgressEvent("device_selected", message=str(self.device))
        )
        self.data_loader_factory = DataLoaderFactory(
            self.configuration,
            self.device,
        )

        self.experiment_store = ExperimentStore.create(
            self.configuration.results_path,
            self.configuration.experiment_name,
        )
        # Keep the historical string attribute for downstream HPO code.
        self.experiment_dir = str(self.experiment_store.experiment_dir)
        self.configuration.set_runtime_value("experiment_dir", self.experiment_dir)
        # ``config`` remains a dictionary snapshot for public, reporting, and
        # checkpoint compatibility. Internal behavior uses ``configuration``.
        self.config = self.configuration.to_dict(include_defaults=True)

        model_bundle = ModelFactory(self.configuration).build(
            instance_branch_factory=instance_branch_model,
            target_branch_factory=target_branch_model,
        )
        self.instance_branch_model = model_bundle.instance_branch
        self.target_branch_model = model_bundle.target_branch
        self.deepMTP_model = model_bundle.model

        self.deepMTP_model.to(self.device)

        # define the optimizer
        self.optimizer = torch.optim.Adam(
            self.deepMTP_model.parameters(),
            lr=self.configuration.learning_rate,
            weight_decay=self.configuration.decay,
        )

        if self.checkpoint_dict is not None:
            # apply the loaded checkpoint
            self.progress.on_event(ProgressEvent("checkpoint_weights_started"))
            self.deepMTP_model.load_state_dict(self.checkpoint_dict["model_state_dict"])
            # update the device of the optimizer
            self.optimizer.load_state_dict(self.checkpoint_dict["optimizer_state_dict"])
            for state in self.optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.to(self.device)
            self.progress.on_event(ProgressEvent("checkpoint_weights_completed"))

        # initialize the loss function that will be used. BCE for classification tasks and MSE for regression tasks.
        problem_mode = self.configuration.problem_mode
        if problem_mode is None:
            raise RuntimeError(
                "Trainer received an incomplete validated problem configuration."
            )
        if problem_mode == "classification":
            self.criterion = nn.BCELoss(reduction="mean")
        else:
            self.criterion = nn.MSELoss(reduction="mean")

        self.training_engine = TrainingEngine(
            model=self.deepMTP_model,
            criterion=self.criterion,
            device=self.device,
            problem_mode=problem_mode,
            optimizer=self.optimizer,
        )
        self.evaluator = Evaluator(
            self.configuration,
            progress=self.progress if self._progress_was_injected else None,
        )

        """
            the user can specify wether or not they want to use early stopping.
            In both cases we create the early stopping object so that we can track the best epoch, model etc. 
        """
        self.early_stopping = EarlyStopping(
            use_early_stopping=self.configuration.use_early_stopping,
            patience=self.configuration.patience,
            delta=self.configuration.delta,
            verbose=self.configuration.verbose,
            metric_to_track=(
                self.configuration.metric_to_optimize_early_stopping
                if self.configuration.use_early_stopping
                else self.configuration.metric_to_optimize_best_epoch_selection
            ),
        )

    @overload
    def inference(
        self,
        model: nn.Module,
        dataloader: Iterable[TensorBatch],
        mode: str,
        epoch: int = 0,
        return_predictions: Literal[False] = False,
        verbose: bool = False,
    ) -> MetricResults: ...

    @overload
    def inference(
        self,
        model: nn.Module,
        dataloader: Iterable[TensorBatch],
        mode: str,
        epoch: int = 0,
        return_predictions: Literal[True] = True,
        verbose: bool = False,
    ) -> PredictionResults: ...

    def inference(
        self,
        model: nn.Module,
        dataloader: Iterable[TensorBatch],
        mode: str,
        epoch: int = 0,
        return_predictions: bool = False,
        verbose: bool = False,
    ) -> MetricResults | PredictionResults:
        should_evaluate = self.evaluator.policy.should_evaluate(mode, epoch)
        progress = self._inference_progress(verbose)

        engine = self.training_engine
        if model is not engine.model:
            problem_mode = self.configuration.problem_mode
            if problem_mode is None:
                raise RuntimeError(
                    "Trainer received an incomplete validated problem configuration."
                )
            engine = TrainingEngine(
                model=model,
                criterion=self.criterion,
                device=self.device,
                problem_mode=problem_mode,
            )
        epoch_output = engine.evaluate_epoch(
            dataloader,
            calculate_loss=mode != "test",
            collect_predictions=should_evaluate or return_predictions,
        )
        results: MetricResults = {}

        if should_evaluate and self.configuration.metrics:
            progress.on_event(ProgressEvent("metrics_started", epoch=epoch, mode=mode))
            results = self.evaluator.evaluate(mode, epoch, epoch_output)
            progress.on_event(
                ProgressEvent("metrics_completed", epoch=epoch, mode=mode)
            )

        if mode != "test":
            results[mode + "_loss"] = epoch_output.mean_loss
        if return_predictions:
            return results, self.evaluator.predictions_frame(epoch_output)
        return results

    def _inference_progress(self, verbose: bool) -> ProgressObserver:
        if (
            verbose
            and not self._progress_was_injected
            and isinstance(self.progress, NullProgressObserver)
        ):
            return ConsoleProgressObserver()
        return self.progress

    def train(
        self,
        train_data: DatasetInput,
        val_data: DatasetInput,
        test_data: DatasetInput,
        verbose: bool = False,
    ) -> MetricResults:
        training_state = getattr(self, "_training_state", "ready")
        if training_state != "ready":
            raise RuntimeError(
                "A DeepMTP instance supports one training run. "
                "Create a new instance to train again."
            )

        reporter = self.reporter
        if reporter is None:
            reporter = build_reporter(
                self.configuration,
                self.experiment_dir,
            )
            self.reporter = reporter
        try:
            reporter.start(self.config, self.deepMTP_model)
            self._training_state = "running"
            try:
                self.progress.on_event(ProgressEvent("training_started"))
                results = self._train(
                    train_data,
                    val_data,
                    test_data,
                    reporter=reporter,
                    verbose=verbose,
                )
            except Exception as exc:
                self._training_state = "failed"
                self.progress.on_event(
                    ProgressEvent("training_failed", message=str(exc))
                )
                raise

            self._training_state = "completed"
            self.progress.on_event(
                ProgressEvent(
                    "training_completed",
                    metrics=results,
                    best_epoch=self.early_stopping.get_best_epoch(),
                )
            )
            return results
        finally:
            reporter.close()

    def _train(
        self,
        train_data: DatasetInput,
        val_data: DatasetInput,
        test_data: DatasetInput,
        *,
        reporter: ExperimentReporter,
        verbose: bool = False,
    ) -> MetricResults:
        self.deepMTP_model.train()
        effective_verbose = verbose or self.configuration.verbose

        dataloaders = self.data_loader_factory.for_training(
            train_data,
            val_data,
            test_data,
        )
        if isinstance(self.configuration, DeepMTPConfig):
            self.config = self.configuration.to_dict(include_defaults=True)
        train_dataloader = dataloaders.train
        val_dataloader = dataloaders.validation
        test_dataloader = dataloaders.test

        run_history = RunHistory(
            self.configuration.metrics,
            self.configuration.metrics_average,
        )

        # iterate over epochs
        for epoch in range(self.configuration.num_epochs):
            train_results = {}

            self.progress.on_event(ProgressEvent("epoch_started", epoch=epoch))
            epoch_output = self.training_engine.train_epoch(
                train_dataloader,
                collect_predictions=self.configuration.evaluate_train,
            )

            # calculate performance metrics on the training set
            if (
                self.evaluator.policy.should_evaluate_training(epoch)
                and self.configuration.metrics
            ):
                train_results = self.evaluator.evaluate("train", epoch, epoch_output)

            mean_loss = epoch_output.mean_loss
            run_history.add_training(
                epoch,
                mean_loss,
                train_results,
            )
            # update the loggers
            results_to_log = {
                "train_" + m + "_" + av: train_results["train_" + m + "_" + av]
                for m in self.configuration.metrics
                for av in self.configuration.metrics_average
                if "train_" + m + "_" + av in train_results
            }
            results_to_log["train_loss"] = mean_loss
            reporter.log_metrics(results_to_log, step=epoch)

            self.progress.on_event(
                ProgressEvent(
                    "epoch_completed",
                    epoch=epoch,
                    metrics=results_to_log,
                )
            )

            # generate predictions for the validation set. This needs to be done in order to early stop and to select between configurations
            self.progress.on_event(ProgressEvent("validation_started", epoch=epoch))
            val_results = self.inference(
                self.deepMTP_model,
                val_dataloader,
                "val",
                epoch,
                verbose=effective_verbose,
            )
            # update the loggers
            results_to_log = {"val_loss": val_results["val_loss"]}
            results_to_log.update(
                {
                    "val_" + m + "_" + av: val_results["val_" + m + "_" + av]
                    for m in self.configuration.metrics
                    for av in self.configuration.metrics_average
                    if "val_" + m + "_" + av in val_results
                }
            )
            reporter.log_metrics(results_to_log, step=epoch)

            # update early stopping and keep track of best model
            early_stopping_update = self.early_stopping(
                val_results,
                copy.deepcopy(self.deepMTP_model),
                epoch,
                optimizer_state_dict=copy.deepcopy(self.optimizer.state_dict()),
            )
            if early_stopping_update.status == "patience_incremented":
                self.progress.on_event(
                    ProgressEvent(
                        "early_stopping_counter_updated",
                        epoch=epoch,
                        early_stopping_counter=early_stopping_update.counter,
                        early_stopping_patience=early_stopping_update.patience,
                        best_epoch=early_stopping_update.best_epoch,
                    )
                )
            run_history.add_validation(
                epoch,
                val_results["val_loss"],
                val_results,
                early_stopping_counter=self.early_stopping.counter,
                early_stopping_patience=self.early_stopping.patience,
            )
            self.progress.on_event(
                ProgressEvent(
                    "validation_completed",
                    epoch=epoch,
                    metrics=val_results,
                    early_stopping_counter=self.early_stopping.counter,
                    early_stopping_patience=self.early_stopping.patience,
                )
            )
            should_stop = (
                self.early_stopping.early_stop_flag
                and self.configuration.use_early_stopping
            )
            if should_stop:
                self.progress.on_event(
                    ProgressEvent("early_stopping_triggered", epoch=epoch)
                )
                break

        best_model = self.early_stopping.get_best_model()
        if not isinstance(best_model, nn.Module):
            raise RuntimeError("Training completed without selecting a best model.")
        self.deepMTP_model = best_model
        self.training_engine.replace_model(self.deepMTP_model)

        # log the performance validation results of the best model
        raw_validation_results = self.early_stopping.get_best_performance_results()
        if not isinstance(raw_validation_results, Mapping):
            raise RuntimeError(
                "Training completed without validation performance results."
            )
        val_best_performance_results = cast(
            MetricResults,
            dict(raw_validation_results),
        )
        results_to_log = {
            "best_val_" + m + "_" + av: val_best_performance_results[
                "val_" + m + "_" + av
            ]
            for m in self.configuration.metrics
            for av in self.configuration.metrics_average
            if "val_" + m + "_" + av in val_best_performance_results
        }
        results_to_log["best_val_loss"] = val_best_performance_results["val_loss"]
        reporter.log_metrics(results_to_log)
        # training is done (either completed all epochs or early stopping kicked in). Now testing starts using the best model
        self.progress.on_event(ProgressEvent("testing_started"))
        test_results = self.inference(
            self.deepMTP_model,
            test_dataloader,
            "test",
            epoch,
            verbose=effective_verbose,
        )
        best_epoch = self.early_stopping.get_best_epoch()
        if not isinstance(best_epoch, int):
            raise RuntimeError("Training completed without selecting a best epoch.")
        run_history.add_test(
            best_epoch,
            test_results,
        )
        # update the loggers
        results_to_log = {
            "test_" + m + "_" + av: test_results["test_" + m + "_" + av]
            for m in self.configuration.metrics
            for av in self.configuration.metrics_average
        }
        reporter.log_metrics(results_to_log)
        self.progress.on_event(
            ProgressEvent(
                "testing_completed",
                metrics=test_results,
                best_epoch=best_epoch,
            )
        )

        summaries = run_history.render()
        self.progress.on_summary(
            summaries,
            include_train=self.configuration.evaluate_train,
            include_validation=self.configuration.evaluate_val,
        )

        self.experiment_store.save_summary(
            summaries.train,
            summaries.validation,
            summaries.test,
        )

        # save the best model
        if self.configuration.save_model:
            self.save_model(verbose=effective_verbose)

        self.experiment_store.save_config(self.config)

        return val_best_performance_results

    @overload
    def predict(
        self,
        data: DatasetInput,
        return_predictions: Literal[False] = False,
        verbose: bool = False,
    ) -> MetricResults: ...

    @overload
    def predict(
        self,
        data: DatasetInput,
        return_predictions: Literal[True] = True,
        verbose: bool = False,
    ) -> PredictionResults: ...

    def predict(
        self,
        data: DatasetInput,
        return_predictions: bool = False,
        verbose: bool = False,
    ) -> MetricResults | PredictionResults:
        self.deepMTP_model.to(self.device)
        dataloader = self.data_loader_factory.for_prediction(data)
        if return_predictions:
            return self.inference(
                self.deepMTP_model,
                dataloader,
                "",
                0,
                return_predictions=True,
                verbose=verbose,
            )
        return self.inference(
            self.deepMTP_model,
            dataloader,
            "",
            0,
            return_predictions=False,
            verbose=verbose,
        )

    def save_model(self, verbose: bool = False) -> None:
        best_model = self.early_stopping.get_best_model()
        best_optimizer_state = self.early_stopping.get_best_optimizer_state_dict()
        if not isinstance(best_model, nn.Module) or not isinstance(
            best_optimizer_state,
            Mapping,
        ):
            raise RuntimeError("Cannot save a model before training has completed.")
        progress = (
            self.progress
            if self._progress_was_injected
            else build_progress_observer(verbose)
        )
        progress.on_event(ProgressEvent("checkpoint_saving_started"))
        checkpoint_config = dict(self.config)
        checkpoint_config["use_tensorboard_logger"] = False
        checkpoint_path = self.experiment_store.save_checkpoint(
            model_state_dict=best_model.state_dict(),
            optimizer_state_dict=best_optimizer_state,
            config=checkpoint_config,
        )
        progress.on_event(
            ProgressEvent(
                "checkpoint_saving_completed",
                message=str(checkpoint_path),
            )
        )


def initialize_mode(config: ConfigInput) -> DeepMTP:  # pragma: no cover
    return DeepMTP(config)
