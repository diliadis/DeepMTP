"""Compare uncached, first-cache, and warm-cache foundation encoding latency."""

from __future__ import annotations

import argparse
import statistics
import time
from pathlib import Path

import torch

from DeepMTP import (
    FoundationCacheSpec,
    FoundationModelSpec,
    HuggingFaceSequenceEncoder,
    tokenize_foundation_sequences,
)
from DeepMTP.data import SequenceInput
from DeepMTP.data.sequences import collate_sequence_rows

_DEFAULT_VALUES = (
    "DeepMTP caches repeated entity representations.",
    "Foundation backbones can dominate training time.",
    "The downstream projection remains trainable.",
    "Cache identities protect reproducibility.",
)


def _elapsed(operation) -> float:
    started = time.perf_counter()
    operation()
    return time.perf_counter() - started


def _median_seconds(operation, *, iterations: int) -> float:
    return statistics.median(_elapsed(operation) for _ in range(iterations))


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def benchmark(
    *,
    model_name: str,
    revision: str | None,
    values: tuple[str, ...],
    repeats: int,
    iterations: int,
    device_name: str,
    cache_directory: str | None,
    local_files_only: bool,
) -> None:
    """Run a repeated-entity benchmark with one compatible AutoModel."""

    device = torch.device(device_name)
    base_spec = FoundationModelSpec(
        model_name=model_name,
        revision=revision,
        train_mode="frozen",
        local_files_only=local_files_only,
    )
    tokenized = tokenize_foundation_sequences(values, base_spec)
    rows = tuple(row for row in tokenized for _ in range(repeats))
    token_ids, attention_mask, lengths = collate_sequence_rows(
        rows,
        padding_idx=tokenized.padding_idx,
    )
    batch = SequenceInput(
        token_ids=token_ids,
        attention_mask=attention_mask,
        lengths=lengths,
        padding_idx=tokenized.padding_idx,
    ).to(device)

    uncached = HuggingFaceSequenceEncoder(
        spec=tokenized.spec,
        vocabulary_size=tokenized.vocabulary_size,
        output_dim=32,
        padding_idx=tokenized.padding_idx,
    ).to(device)
    cached = HuggingFaceSequenceEncoder(
        spec=FoundationModelSpec.from_config(
            {
                **tokenized.spec.to_dict(),
                "cache": FoundationCacheSpec(
                    directory=cache_directory,
                    memory_capacity=max(1, len(values)),
                ).to_dict(),
            }
        ),
        vocabulary_size=tokenized.vocabulary_size,
        output_dim=32,
        padding_idx=tokenized.padding_idx,
    ).to(device)
    uncached.eval()
    cached.eval()

    def run(encoder: HuggingFaceSequenceEncoder) -> None:
        with torch.no_grad():
            encoder(batch)
        _synchronize(device)

    run(uncached)
    uncached_seconds = _median_seconds(
        lambda: run(uncached),
        iterations=iterations,
    )
    cached.clear_cache(memory=True, disk=False)
    first_cache_seconds = _elapsed(lambda: run(cached))
    warm_seconds = _median_seconds(
        lambda: run(cached),
        iterations=iterations,
    )

    print(f"model: {model_name}")
    print(f"device: {device}; entities: {len(rows)} ({len(values)} unique)")
    print(f"uncached median: {uncached_seconds:.4f} seconds")
    print(f"first cache pass: {first_cache_seconds:.4f} seconds")
    print(f"warm cache median: {warm_seconds:.4f} seconds")
    print(f"warm speedup: {uncached_seconds / warm_seconds:.2f}x")
    print(f"cache stats: {cached.cache_stats}")


def _positive_integer(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--revision")
    parser.add_argument("--value", action="append", dest="values")
    parser.add_argument("--repeats", type=_positive_integer, default=32)
    parser.add_argument("--iterations", type=_positive_integer, default=5)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--cache-directory", type=Path)
    parser.add_argument("--local-files-only", action="store_true")
    arguments = parser.parse_args()
    benchmark(
        model_name=arguments.model_name,
        revision=arguments.revision,
        values=tuple(arguments.values or _DEFAULT_VALUES),
        repeats=arguments.repeats,
        iterations=arguments.iterations,
        device_name=arguments.device,
        cache_directory=(
            None
            if arguments.cache_directory is None
            else str(arguments.cache_directory)
        ),
        local_files_only=arguments.local_files_only,
    )


if __name__ == "__main__":
    main()
