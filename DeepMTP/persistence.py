"""Experiment directory and artifact persistence."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
import inspect
import json
import os
from pathlib import Path
import pickle
import tempfile
from typing import Any

import torch

from DeepMTP.config import DeepMTPConfig


CHECKPOINT_KEYS = frozenset(
    {"model_state_dict", "optimizer_state_dict", "config"}
)


@dataclass(frozen=True)
class ExperimentStore:
    """Owns the filesystem side effects associated with one experiment."""

    experiment_dir: Path

    @classmethod
    def create(
        cls,
        results_path: str | os.PathLike[str],
        experiment_name: str | None = None,
        *,
        timestamp: str | None = None,
    ) -> ExperimentStore:
        """Create a unique experiment directory below ``results_path``."""
        results_dir = Path(results_path)
        results_dir.mkdir(parents=True, exist_ok=True)

        if experiment_name is not None:
            directory_name = experiment_name
        elif timestamp is not None:
            directory_name = timestamp
        else:
            directory_name = datetime.now().strftime("%d_%m_%Y__%H_%M_%S")
        cls._validate_directory_name(directory_name)

        candidate = results_dir / directory_name
        suffix = 0
        while True:
            try:
                candidate.mkdir()
            except FileExistsError:
                suffix += 1
                candidate = results_dir / f"{directory_name}_{suffix}"
            else:
                return cls(candidate)

    def save_summary(
        self,
        train_summary: str,
        validation_summary: str,
        test_summary: str,
    ) -> Path:
        summary = "\n".join(
            [train_summary, validation_summary, test_summary]
        )
        summary_path = self.experiment_dir / "summary.txt"
        _atomic_write_text(summary_path, summary)
        return summary_path

    def save_config(
        self, config: Mapping[str, Any] | DeepMTPConfig
    ) -> Path:
        """Save JSON-compatible configs as JSON and all others as pickle."""
        if isinstance(config, DeepMTPConfig):
            config = config.to_dict()
        elif not isinstance(config, Mapping):
            raise TypeError(
                "config must be a mapping or DeepMTPConfig, "
                f"not {type(config).__name__}."
            )

        config_mapping = dict(config)
        try:
            serialized_config = json.dumps(config_mapping, indent=4)
        except (TypeError, ValueError):
            config_path = self.experiment_dir / "config.pkl"
            _atomic_write_bytes(config_path, pickle.dumps(config_mapping))
            (self.experiment_dir / "config.json").unlink(missing_ok=True)
        else:
            config_path = self.experiment_dir / "config.json"
            _atomic_write_text(config_path, serialized_config)
            (self.experiment_dir / "config.pkl").unlink(missing_ok=True)
        return config_path

    def save_checkpoint(
        self,
        *,
        model_state_dict: Mapping[str, Any],
        optimizer_state_dict: Mapping[str, Any],
        config: Mapping[str, Any] | DeepMTPConfig,
    ) -> Path:
        """Atomically save a checkpoint using the historical payload format."""
        if isinstance(config, DeepMTPConfig):
            config = config.to_dict()
        checkpoint = {
            "model_state_dict": dict(model_state_dict),
            "optimizer_state_dict": dict(optimizer_state_dict),
            "config": dict(config),
        }
        checkpoint_path = self.experiment_dir / "model.pt"
        _atomic_torch_save(checkpoint_path, checkpoint)
        return checkpoint_path

    @staticmethod
    def load_checkpoint(
        checkpoint_path: str | os.PathLike[str],
        *,
        map_location: str | torch.device = "cpu",
    ) -> dict[str, Any]:
        """Load a trusted DeepMTP checkpoint onto an explicit device."""
        path = Path(checkpoint_path)
        if not path.is_file():
            raise FileNotFoundError(f"Checkpoint does not exist: {path}")

        load_arguments: dict[str, Any] = {"map_location": map_location}
        if "weights_only" in inspect.signature(torch.load).parameters:
            # DeepMTP checkpoints contain configuration metadata in addition
            # to tensor state dictionaries. Only load checkpoints you trust.
            load_arguments["weights_only"] = False
        checkpoint = torch.load(path, **load_arguments)
        if not isinstance(checkpoint, dict):
            raise ValueError("DeepMTP checkpoint must contain a dictionary.")

        missing_keys = CHECKPOINT_KEYS.difference(checkpoint)
        if missing_keys:
            raise ValueError(
                "DeepMTP checkpoint is missing required keys: "
                f"{sorted(missing_keys)}."
            )
        return checkpoint

    @staticmethod
    def _validate_directory_name(directory_name: str) -> None:
        if (
            not isinstance(directory_name, str)
            or not directory_name
            or Path(directory_name).name != directory_name
            or directory_name in {".", ".."}
        ):
            raise ValueError(
                "experiment_name must be a non-empty directory name without "
                f"path separators; received {directory_name!r}."
            )


def _atomic_write_text(path: Path, content: str) -> None:
    _atomic_write_bytes(path, content.encode("utf-8"))


def _atomic_write_bytes(path: Path, content: bytes) -> None:
    temporary_path = _temporary_path(path)
    try:
        temporary_path.write_bytes(content)
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _atomic_torch_save(path: Path, content: Any) -> None:
    temporary_path = _temporary_path(path)
    try:
        torch.save(content, temporary_path)
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _temporary_path(destination: Path) -> Path:
    file_descriptor, temporary_name = tempfile.mkstemp(
        dir=destination.parent,
        prefix=f".{destination.name}.",
        suffix=".tmp",
    )
    os.close(file_descriptor)
    return Path(temporary_name)
