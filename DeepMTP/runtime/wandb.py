"""W&B Artifact download helpers kept behind the optional tracking extra."""

from __future__ import annotations

import importlib
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

from DeepMTP.runtime.foundation_checkpoints import (
    FOUNDATION_ADAPTER_CHECKPOINT_NAME,
)
from DeepMTP.runtime.reporting import OptionalIntegrationError

__all__ = ["WandBCheckpoint", "download_wandb_checkpoint"]


@dataclass(frozen=True)
class WandBCheckpoint:
    """A downloaded W&B model artifact containing one DeepMTP checkpoint."""

    artifact_reference: str
    checkpoint_path: Path
    download_root: Path
    artifact_name: str | None
    artifact_version: str | None
    artifact_digest: str | None
    checkpoint_format: str = "full"


def download_wandb_checkpoint(
    artifact_reference: str,
    *,
    download_dir: str | os.PathLike[str] | None = None,
) -> WandBCheckpoint:
    """Download and validate one DeepMTP model artifact through the Public API."""
    if not isinstance(artifact_reference, str) or not artifact_reference.strip():
        raise ValueError("artifact_reference must be a non-empty string.")
    try:
        wandb = importlib.import_module("wandb")
    except (ImportError, ModuleNotFoundError) as error:
        raise OptionalIntegrationError(
            "Downloading a W&B checkpoint requires the optional tracking "
            "dependency. Install DeepMTP[tracking]."
        ) from error

    root = (
        Path(tempfile.mkdtemp(prefix="deepmtp-wandb-"))
        if download_dir is None
        else Path(download_dir)
    )
    root.mkdir(parents=True, exist_ok=True)
    artifact = wandb.Api().artifact(artifact_reference)
    downloaded_root = Path(artifact.download(root=str(root)))
    checkpoint_paths = sorted(
        (
            *downloaded_root.rglob("model.pt"),
            *downloaded_root.rglob(FOUNDATION_ADAPTER_CHECKPOINT_NAME),
        )
    )
    if not checkpoint_paths:
        raise ValueError(
            f"W&B artifact {artifact_reference!r} contains no supported DeepMTP "
            "checkpoint (model.pt or foundation_adapter.zip)."
        )
    if len(checkpoint_paths) > 1:
        raise ValueError(
            f"W&B artifact {artifact_reference!r} contains multiple supported "
            "DeepMTP checkpoints."
        )
    checkpoint_path = checkpoint_paths[0]
    return WandBCheckpoint(
        artifact_reference=artifact_reference,
        checkpoint_path=checkpoint_path,
        download_root=downloaded_root,
        artifact_name=getattr(artifact, "name", None),
        artifact_version=getattr(artifact, "version", None),
        artifact_digest=getattr(artifact, "digest", None),
        checkpoint_format=(
            "foundation_adapter"
            if checkpoint_path.name == FOUNDATION_ADAPTER_CHECKPOINT_NAME
            else "full"
        ),
    )
