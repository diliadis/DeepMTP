"""Optional integrations around the framework's core runtime."""

from DeepMTP.integrations.huggingface import (
    FoundationLoRASpec,
    FoundationModelSpec,
    HuggingFaceSequenceEncoder,
    TokenizedFoundationSequences,
    tokenize_foundation_sequences,
)

__all__ = [
    "FoundationLoRASpec",
    "FoundationModelSpec",
    "HuggingFaceSequenceEncoder",
    "TokenizedFoundationSequences",
    "tokenize_foundation_sequences",
]
