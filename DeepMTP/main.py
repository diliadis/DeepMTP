import copy
from collections.abc import Mapping

import torch
from torch import nn

from DeepMTP.config import DeepMTPConfig
from DeepMTP.data_loading import DataLoaderFactory
from DeepMTP.evaluation import Evaluator
from DeepMTP.model_factory import (
    ModelFactory,
    TwoBranchDotProductModel,
    TwoBranchKroneckerModel,
    TwoBranchMLPModel,
)
from DeepMTP.persistence import ExperimentStore
from DeepMTP.presentation import (
    ConsoleProgressObserver,
    NullProgressObserver,
    ProgressEvent,
    RunHistory,
    build_progress_observer,
)
from DeepMTP.reporting import build_reporter
from DeepMTP.training import TrainingEngine
from DeepMTP.utils.model_utils import EarlyStopping

__all__ = [
    "DeepMTP",
    "TwoBranchDotProductModel",
    "TwoBranchKroneckerModel",
    "TwoBranchMLPModel",
]


class DeepMTP:  # pragma: no cover
    """Implements the training and inference logic of the DeepMTP framework."""

    def __init__(
        self,
        config,
        instance_branch_model=None,
        target_branch_model=None,
        checkpoint_dir=None,
        reporter=None,
        progress=None,
    ):  # pragma: no cover
        self.checkpoint_dict = None
        self.reporter = reporter
        self._progress_was_injected = progress is not None

        if isinstance(config, DeepMTPConfig):
            config_overrides = config.to_dict()
        elif isinstance(config, Mapping):
            config_overrides = dict(config)
        else:
            raise TypeError(
                "config must be a mapping or DeepMTPConfig, "
                f"not {type(config).__name__}."
            )

        # Load a checkpoint file, if it exists
        if checkpoint_dir is not None:
            if config_overrides.get("verbose", False):
                print("Loading checkpoint from " + str(checkpoint_dir) + "... ", "")
            self.checkpoint_dict = ExperimentStore.load_checkpoint(
                checkpoint_dir, map_location="cpu"
            )
            if config_overrides.get("verbose", False):
                print("Done")

            # when a checkpoint is loaded, it's config can (or sometimes has to) be modified
            checkpoint_config = self.checkpoint_dict["config"]
            if isinstance(checkpoint_config, DeepMTPConfig):
                checkpoint_config = checkpoint_config.to_dict()
            merged_config = dict(checkpoint_config)
            merged_config.update(config_overrides)
        else:
            merged_config = config_overrides

        self.configuration = DeepMTPConfig.from_mapping(merged_config)
        # ``config`` remains a dictionary for public and checkpoint
        # compatibility while new code can use the typed ``configuration``.
        self.config = self.configuration.to_dict()
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
        if self.config["verbose"]:
            print("Selected device: " + str(self.device))
        self.data_loader_factory = DataLoaderFactory(
            self.configuration,
            self.device,
        )

        self.experiment_store = ExperimentStore.create(
            self.config["results_path"],
            self.config["experiment_name"],
        )
        # Keep the historical string attribute for downstream HPO code.
        self.experiment_dir = str(self.experiment_store.experiment_dir)
        self.config["experiment_dir"] = self.experiment_dir
        self.configuration.set_runtime_value("experiment_dir", self.experiment_dir)

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
            lr=self.config["learning_rate"],
            weight_decay=self.config["decay"],
        )

        if self.checkpoint_dict is not None:
            # apply the loaded checkpoint
            if self.config["verbose"]:
                print("Applying saved weights... ", end="")
            self.deepMTP_model.load_state_dict(self.checkpoint_dict["model_state_dict"])
            # update the device of the optimizer
            self.optimizer.load_state_dict(self.checkpoint_dict["optimizer_state_dict"])
            for state in self.optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.to(self.device)
            if self.config["verbose"]:
                print("Done")

        # initialize the loss function that will be used. BCE for classification tasks and MSE for regression tasks.
        if self.config["problem_mode"] == "classification":
            self.criterion = nn.BCELoss(reduction="mean")
        elif self.config["problem_mode"] == "regression":
            self.criterion = nn.MSELoss(reduction="mean")
        else:
            raise AttributeError(
                "Invalid problem model: " + str(self.config["problem_mode"])
            )

        self.training_engine = TrainingEngine(
            model=self.deepMTP_model,
            criterion=self.criterion,
            device=self.device,
            problem_mode=self.config["problem_mode"],
            optimizer=self.optimizer,
        )
        self.evaluator = Evaluator(self.configuration)

        """
            the user can specify wether or not they want to use early stopping.
            In both cases we create the early stopping object so that we can track the best epoch, model etc. 
        """
        self.early_stopping = EarlyStopping(
            use_early_stopping=self.config["use_early_stopping"],
            patience=self.config["patience"],
            delta=self.config["delta"],
            verbose=self.config["verbose"],
            metric_to_track=self.config["metric_to_optimize_early_stopping"]
            if self.config["use_early_stopping"]
            else self.config["metric_to_optimize_best_epoch_selection"],
        )

    def inference(
        self, model, dataloader, mode, epoch=0, return_predictions=False, verbose=False
    ):  # pragma: no cover
        should_evaluate = self.evaluator.policy.should_evaluate(mode, epoch)
        progress = self._inference_progress(verbose)

        engine = self.training_engine
        if model is not engine.model:
            engine = TrainingEngine(
                model=model,
                criterion=self.criterion,
                device=self.device,
                problem_mode=self.config["problem_mode"],
            )
        epoch_output = engine.evaluate_epoch(
            dataloader,
            calculate_loss=mode != "test",
            collect_predictions=should_evaluate or return_predictions,
        )
        results = {}

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

    def _inference_progress(self, verbose):
        if (
            verbose
            and not self._progress_was_injected
            and isinstance(self.progress, NullProgressObserver)
        ):
            return ConsoleProgressObserver()
        return self.progress

    def train(self, train_data, val_data, test_data, verbose=False):  # pragma: no cover
        if self.reporter is None:
            self.reporter = build_reporter(
                self.config,
                self.experiment_dir,
            )
        self.reporter.start(self.config, self.deepMTP_model)
        try:
            return self._train(train_data, val_data, test_data, verbose=verbose)
        finally:
            self.reporter.close()

    def _train(self, train_data, val_data, test_data, verbose=False):
        self.deepMTP_model.train()

        dataloaders = self.data_loader_factory.for_training(
            train_data,
            val_data,
            test_data,
        )
        train_dataloader = dataloaders.train
        val_dataloader = dataloaders.validation
        test_dataloader = dataloaders.test

        self.progress.on_event(ProgressEvent("training_started"))
        run_history = RunHistory(
            self.configuration.metrics,
            self.configuration.metrics_average,
        )

        #########################    Profiler    ############################
        """        
        prof = torch.profiler.profile(
                schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=2),
                on_trace_ready=torch.profiler.tensorboard_trace_handler('./log/profiler'),
                record_shapes=True,
                with_stack=True)
        prof.start()
        """

        # iterate over epochs
        for epoch in range(self.config["num_epochs"]):
            train_results = {}

            self.progress.on_event(ProgressEvent("epoch_started", epoch=epoch))
            epoch_output = self.training_engine.train_epoch(
                train_dataloader,
                collect_predictions=self.config["evaluate_train"],
            )

            # calculate performance metrics on the training set
            if (
                self.evaluator.policy.should_evaluate_training(epoch)
                and self.configuration.metrics
            ):
                train_results = self.evaluator.evaluate("train", epoch, epoch_output)

            #########################    Profiler    ############################
            # prof.stop()

            mean_loss = epoch_output.mean_loss
            run_history.add_training(
                epoch,
                mean_loss,
                train_results,
            )
            # update the loggers
            results_to_log = {
                "train_" + m + "_" + av: train_results["train_" + m + "_" + av]
                for m in self.config["metrics"]
                for av in self.config["metrics_average"]
                if "train_" + m + "_" + av in train_results
            }
            results_to_log["train_loss"] = mean_loss
            self.reporter.log_metrics(results_to_log, step=epoch)

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
                verbose=self.config["verbose"],
            )
            # update the loggers
            results_to_log = {"val_loss": val_results["val_loss"]}
            results_to_log.update(
                {
                    "val_" + m + "_" + av: val_results["val_" + m + "_" + av]
                    for m in self.config["metrics"]
                    for av in self.config["metrics_average"]
                    if "val_" + m + "_" + av in val_results
                }
            )
            self.reporter.log_metrics(results_to_log, step=epoch)

            # update early stopping and keep track of best model
            self.early_stopping(
                val_results,
                copy.deepcopy(self.deepMTP_model),
                epoch,
                optimizer_state_dict=copy.deepcopy(self.optimizer.state_dict()),
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
                and self.config["use_early_stopping"]
            )
            if should_stop:
                self.progress.on_event(
                    ProgressEvent("early_stopping_triggered", epoch=epoch)
                )
                break

        self.deepMTP_model = self.early_stopping.get_best_model()
        self.training_engine.replace_model(self.deepMTP_model)

        # log the performance validation results of the best model
        val_best_performance_results = (
            self.early_stopping.get_best_performance_results()
        )
        results_to_log = {
            "best_val_" + m + "_" + av: val_best_performance_results[
                "val_" + m + "_" + av
            ]
            for m in self.config["metrics"]
            for av in self.config["metrics_average"]
            if "val_" + m + "_" + av in val_best_performance_results
        }
        results_to_log["best_val_loss"] = val_best_performance_results["val_loss"]
        self.reporter.log_metrics(results_to_log)
        # training is done (either completed all epochs or early stopping kicked in). Now testing starts using the best model
        self.progress.on_event(ProgressEvent("testing_started"))
        test_results = self.inference(
            self.deepMTP_model,
            test_dataloader,
            "test",
            epoch,
            verbose=self.config["verbose"],
        )
        run_history.add_test(
            self.early_stopping.get_best_epoch(),
            test_results,
        )
        # update the loggers
        results_to_log = {
            "test_" + m + "_" + av: test_results["test_" + m + "_" + av]
            for m in self.config["metrics"]
            for av in self.config["metrics_average"]
        }
        self.reporter.log_metrics(results_to_log)
        self.progress.on_event(
            ProgressEvent(
                "testing_completed",
                metrics=test_results,
                best_epoch=self.early_stopping.get_best_epoch(),
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
        if self.config["save_model"]:
            self.save_model(verbose=self.config["verbose"])

        self.experiment_store.save_config(self.config)

        return val_best_performance_results

    def predict(
        self, data, return_predictions=False, verbose=False
    ):  # pragma: no cover
        self.deepMTP_model.to(self.device)
        dataloader = self.data_loader_factory.for_prediction(data)
        return self.inference(
            self.deepMTP_model,
            dataloader,
            "",
            0,
            return_predictions=return_predictions,
            verbose=verbose,
        )

    def save_model(self, verbose=False):  # pragma: no cover
        self.config["use_tensorboard_logger"] = False
        self.configuration.set_runtime_value("use_tensorboard_logger", False)
        if verbose:
            print("Saving the best model... ", end="")
        self.experiment_store.save_checkpoint(
            model_state_dict=self.early_stopping.get_best_model().state_dict(),
            optimizer_state_dict=self.early_stopping.get_best_optimizer_state_dict(),
            config=self.config,
        )
        if verbose:
            print("Done")


def initialize_mode(config):  # pragma: no cover
    return DeepMTP(config)
