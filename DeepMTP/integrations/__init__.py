"""Optional integrations around the framework's core runtime."""

from DeepMTP.integrations.huggingface import (
    FoundationCacheSpec,
    FoundationLoRASpec,
    FoundationModelSpec,
    HuggingFaceSequenceEncoder,
    TokenizedFoundationSequences,
    tokenize_foundation_sequences,
)

__all__ = [
    "FoundationCacheSpec",
    "FoundationLoRASpec",
    "FoundationModelSpec",
    "HuggingFaceSequenceEncoder",
    "TokenizedFoundationSequences",
    "tokenize_foundation_sequences",
]
