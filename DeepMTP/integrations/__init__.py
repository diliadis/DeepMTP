"""Optional integrations around the framework's core runtime."""

from DeepMTP.integrations.huggingface import (
    FoundationModelSpec,
    HuggingFaceSequenceEncoder,
    TokenizedFoundationSequences,
    tokenize_foundation_sequences,
)

__all__ = [
    "FoundationModelSpec",
    "HuggingFaceSequenceEncoder",
    "TokenizedFoundationSequences",
    "tokenize_foundation_sequences",
]
