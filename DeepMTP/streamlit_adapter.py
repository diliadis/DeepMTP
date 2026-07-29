"""Compatibility facade for :mod:`DeepMTP.integrations.streamlit.progress`."""

from DeepMTP.integrations.streamlit.progress import (
    StreamlitAPI,
    StreamlitHPOProgressObserver,
    StreamlitProgressObserver,
)

__all__ = [
    "StreamlitAPI",
    "StreamlitHPOProgressObserver",
    "StreamlitProgressObserver",
]
