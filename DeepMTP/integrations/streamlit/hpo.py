"""Hyperparameter optimizers configured with Streamlit progress rendering."""

from typing import Any

from DeepMTP.hpo.hyperband import HyperBand as CoreHyperBand
from DeepMTP.hpo.progress import HPOProgressObserver
from DeepMTP.hpo.random_search import RandomSearch as CoreRandomSearch
from DeepMTP.integrations.streamlit.progress import StreamlitHPOProgressObserver

__all__ = ["HyperBand", "RandomSearch"]


class RandomSearch(CoreRandomSearch):  # pragma: no cover
    """Core random search configured with Streamlit progress rendering."""

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
