"""Versioned checkpoint provenance and reproducibility metadata."""

from __future__ import annotations

import json
import os
import platform
import subprocess
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import torch

from DeepMTP.config import DeepMTPConfig

CHECKPOINT_FORMAT_VERSION = 1

__all__ = [
    "CHECKPOINT_FORMAT_VERSION",
    "build_checkpoint_metadata",
    "validate_checkpoint_metadata",
]


@dataclass(frozen=True)
class _CheckpointSettings:
    """Typed metadata inputs resolved at the compatibility boundary."""

    compute_mode: object
    random_seed: object
    validation_setting: object
    data_preparation_state: object
    instance_tabular_preprocessing_state: object
    target_tabular_preprocessing_state: object

    @classmethod
    def from_config(
        cls,
        config: DeepMTPConfig | Mapping[str, Any],
    ) -> _CheckpointSettings:
        if isinstance(config, DeepMTPConfig):
            return cls(
                compute_mode=config.compute_mode,
                random_seed=config.random_seed,
                validation_setting=config.validation_setting,
                data_preparation_state=config.data_preparation_state,
                instance_tabular_preprocessing_state=(
                    config.instance_tabular_preprocessing_state
                ),
                target_tabular_preprocessing_state=(
                    config.target_tabular_preprocessing_state
                ),
            )
        values = dict(config)
        return cls(
            compute_mode=values.get("compute_mode"),
            random_seed=values.get("random_seed"),
            validation_setting=values.get("validation_setting"),
            data_preparation_state=values.get("data_preparation_state"),
            instance_tabular_preprocessing_state=values.get(
                "instance_tabular_preprocessing_state"
            ),
            target_tabular_preprocessing_state=values.get(
                "target_tabular_preprocessing_state"
            ),
        )


def build_checkpoint_metadata(
    config: DeepMTPConfig | Mapping[str, Any],
    *,
    effective_device: str,
    training: Mapping[str, Any] | None = None,
    tracking: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build one JSON-compatible manifest for local and remote artifacts."""
    settings = _CheckpointSettings.from_config(config)
    metadata = {
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "preprocessing": {
            "data_preparation_state": settings.data_preparation_state,
            "instance_tabular_preprocessing_state": (
                settings.instance_tabular_preprocessing_state
            ),
            "target_tabular_preprocessing_state": (
                settings.target_tabular_preprocessing_state
            ),
        },
        "training": {
            "model_seed": settings.random_seed,
            "validation_setting": settings.validation_setting,
            **dict(training or {}),
        },
        "environment": _environment_metadata(
            requested_device=settings.compute_mode,
            effective_device=effective_device,
        ),
        "tracking": dict(tracking or {}),
    }
    return validate_checkpoint_metadata(metadata)


def validate_checkpoint_metadata(value: object) -> dict[str, Any]:
    """Validate a checkpoint manifest without accepting future formats."""
    if not isinstance(value, Mapping):
        raise ValueError("DeepMTP checkpoint metadata must be a mapping.")
    format_version = value.get("format_version")
    if format_version != CHECKPOINT_FORMAT_VERSION:
        raise ValueError(
            "Unsupported DeepMTP checkpoint metadata format version "
            f"{format_version!r}; expected {CHECKPOINT_FORMAT_VERSION}."
        )

    normalized = dict(value)
    for section in ("preprocessing", "training", "environment", "tracking"):
        section_value = normalized.get(section)
        if not isinstance(section_value, Mapping):
            raise ValueError(
                f"DeepMTP checkpoint metadata section {section!r} must be a mapping."
            )
        normalized[section] = dict(section_value)
    try:
        json.dumps(normalized, allow_nan=False)
    except (TypeError, ValueError) as error:
        raise ValueError(
            "DeepMTP checkpoint metadata must contain only finite "
            "JSON-compatible values."
        ) from error
    return normalized


def _environment_metadata(
    *,
    requested_device: object,
    effective_device: str,
) -> dict[str, Any]:
    git_metadata = _git_metadata()
    return {
        "deepmtp_version": _distribution_version("DeepMTP"),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "requested_device": requested_device,
        "effective_device": effective_device,
        "default_dtype": str(torch.get_default_dtype()).removeprefix("torch."),
        "python_hash_seed": os.environ.get("PYTHONHASHSEED"),
        "torch_runtime": {
            "cuda_version": torch.version.cuda,
            "cudnn_version": torch.backends.cudnn.version(),
            "deterministic_algorithms": (torch.are_deterministic_algorithms_enabled()),
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
            "cudnn_deterministic": torch.backends.cudnn.deterministic,
            "accelerator_name": _accelerator_name(effective_device),
        },
        "dependencies": {
            package: _distribution_version(package)
            for package in (
                "numpy",
                "pandas",
                "scikit-learn",
                "scipy",
                "torch",
                "torch-geometric",
                "wandb",
            )
        },
        "git": git_metadata,
    }


def _distribution_version(package: str) -> str | None:
    try:
        return version(package)
    except PackageNotFoundError:
        return None


def _git_metadata() -> dict[str, Any] | None:
    repository_root = Path(__file__).resolve().parents[2]
    try:
        revision = subprocess.run(
            ["git", "-C", str(repository_root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=2,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "-C", str(repository_root), "status", "--porcelain"],
            check=True,
            capture_output=True,
            text=True,
            timeout=2,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    return {
        "commit": revision,
        "dirty": bool(status.strip()),
    }


def _accelerator_name(effective_device: str) -> str | None:
    if not effective_device.startswith("cuda") or not torch.cuda.is_available():
        return None
    try:
        return torch.cuda.get_device_name(torch.device(effective_device))
    except (AssertionError, RuntimeError, ValueError):
        return None
