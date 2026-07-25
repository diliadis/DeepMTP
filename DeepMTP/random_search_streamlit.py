"""Backward-compatible Streamlit adapter for random search."""

from DeepMTP.random_search import RandomSearch as CoreRandomSearch
from DeepMTP.streamlit_adapter import StreamlitHPOProgressObserver


class RandomSearch(CoreRandomSearch):  # pragma: no cover
    """Core random search configured with Streamlit progress rendering."""

    def __init__(self, *args, progress=None, **kwargs):
        streamlit_progress = (
            progress if progress is not None else StreamlitHPOProgressObserver()
        )
        kwargs["progress"] = streamlit_progress
        super().__init__(*args, **kwargs)
