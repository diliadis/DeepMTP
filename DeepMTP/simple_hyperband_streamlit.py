"""Backward-compatible Streamlit adapter for Hyperband."""

from typing import Any

from DeepMTP.hpo.hyperband import HyperBand as CoreHyperBand
from DeepMTP.hpo.progress import HPOProgressObserver
from DeepMTP.streamlit_adapter import StreamlitHPOProgressObserver


class HyperBand(CoreHyperBand):  # pragma: no cover
    """Core Hyperband configured with Streamlit progress rendering."""

    def __init__(
        self,
        *args: Any,
        progress: HPOProgressObserver | None = None,
        **kwargs: Any,
    ) -> None:
        streamlit_progress = (
            progress if progress is not None else StreamlitHPOProgressObserver()
        )
        kwargs["progress"] = streamlit_progress
        super().__init__(*args, **kwargs)
