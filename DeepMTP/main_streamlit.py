"""Backward-compatible Streamlit adapter for the core trainer."""

from __future__ import annotations

from DeepMTP.main import DeepMTP as CoreDeepMTP
from DeepMTP.streamlit_adapter import StreamlitProgressObserver


class DeepMTP(CoreDeepMTP):  # pragma: no cover
    """Core DeepMTP trainer configured with Streamlit progress rendering."""

    def __init__(
        self,
        config,
        instance_branch_model=None,
        target_branch_model=None,
        checkpoint_dir=None,
        reporter=None,
        progress=None,
    ):
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


def initialize_mode(config):  # pragma: no cover
    return DeepMTP(config)
