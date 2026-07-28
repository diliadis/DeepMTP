"""Core trainer configured with Streamlit progress rendering."""

from __future__ import annotations

from collections.abc import Mapping
from os import PathLike
from typing import Any

from DeepMTP.config import DeepMTPConfig
from DeepMTP.integrations.streamlit.progress import StreamlitProgressObserver
from DeepMTP.main import DeepMTP as CoreDeepMTP
from DeepMTP.models import BranchFactory
from DeepMTP.presentation import ProgressObserver
from DeepMTP.reporting import ExperimentReporter

__all__ = ["DeepMTP", "initialize_mode"]


class DeepMTP(CoreDeepMTP):  # pragma: no cover
    """Core DeepMTP trainer configured with Streamlit progress rendering."""

    def __init__(
        self,
        config: DeepMTPConfig | Mapping[str, Any],
        instance_branch_model: BranchFactory | None = None,
        target_branch_model: BranchFactory | None = None,
        checkpoint_dir: str | PathLike[str] | None = None,
        reporter: ExperimentReporter | None = None,
        progress: ProgressObserver | None = None,
    ) -> None:
        streamlit_progress = (
            progress if progress is not None else StreamlitProgressObserver()
        )
        super().__init__(
            config,
            instance_branch_model=instance_branch_model,
            target_branch_model=target_branch_model,
            checkpoint_dir=checkpoint_dir,
            reporter=reporter,
            progress=streamlit_progress,
        )
        if isinstance(streamlit_progress, StreamlitProgressObserver):
            streamlit_progress.configure(
                self.configuration,
                experiment_dir=self.experiment_dir,
            )


def initialize_mode(
    config: DeepMTPConfig | Mapping[str, Any],
) -> DeepMTP:  # pragma: no cover
    return DeepMTP(config)
