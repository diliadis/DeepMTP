"""Backward-compatible Streamlit adapter for Hyperband."""

from DeepMTP.simple_hyperband import HyperBand as CoreHyperBand
from DeepMTP.streamlit_adapter import StreamlitHPOProgressObserver


class HyperBand(CoreHyperBand):  # pragma: no cover
    """Core Hyperband configured with Streamlit progress rendering."""

    def __init__(self, *args, progress=None, **kwargs):
        streamlit_progress = (
            progress
            if progress is not None
            else StreamlitHPOProgressObserver()
        )
        super().__init__(
            *args,
            progress=streamlit_progress,
            **kwargs,
        )
