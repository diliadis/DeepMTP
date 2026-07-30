"""Compare dense and sparse first-projection memory and CPU throughput."""

from __future__ import annotations

import argparse
import statistics
import time

import torch
from torch import nn

from DeepMTP.models import SparseMLP


def _tensor_storage_bytes(values: torch.Tensor) -> int:
    if values.layout == torch.sparse_coo:
        values = values.coalesce()
        return (
            values.indices().numel() * values.indices().element_size()
            + values.values().numel() * values.values().element_size()
        )
    return values.numel() * values.element_size()


def _median_forward_seconds(
    operation,
    *,
    warmup: int,
    iterations: int,
) -> float:
    with torch.no_grad():
        for _ in range(warmup):
            operation()
        durations = []
        for _ in range(iterations):
            started = time.perf_counter()
            operation()
            durations.append(time.perf_counter() - started)
    return statistics.median(durations)


def benchmark(
    *,
    batch_size: int,
    input_dim: int,
    output_dim: int,
    density: float,
    warmup: int,
    iterations: int,
    seed: int,
) -> None:
    """Run one reproducible CPU comparison and print a compact report."""
    generator = torch.Generator().manual_seed(seed)
    requested_nnz = max(1, round(batch_size * input_dim * density))
    indices = torch.stack(
        (
            torch.randint(batch_size, (requested_nnz,), generator=generator),
            torch.randint(input_dim, (requested_nnz,), generator=generator),
        )
    )
    sparse_values = torch.sparse_coo_tensor(
        indices,
        torch.randn(requested_nnz, generator=generator),
        size=(batch_size, input_dim),
    ).coalesce()
    dense_values = sparse_values.to_dense()

    dense_projection = nn.Linear(input_dim, output_dim)
    sparse_projection = SparseMLP(
        {},
        input_dim=input_dim,
        output_dim=output_dim,
        nodes_per_layer=output_dim,
        num_layers=0,
        dropout_rate=0.0,
        batch_norm=False,
        skip_last_act=True,
    )
    sparse_projection.input_projection.load_state_dict(dense_projection.state_dict())

    dense_seconds = _median_forward_seconds(
        lambda: dense_projection(dense_values),
        warmup=warmup,
        iterations=iterations,
    )
    sparse_seconds = _median_forward_seconds(
        lambda: sparse_projection(sparse_values),
        warmup=warmup,
        iterations=iterations,
    )
    dense_bytes = _tensor_storage_bytes(dense_values)
    sparse_bytes = _tensor_storage_bytes(sparse_values)

    print(f"shape: {batch_size} x {input_dim}; output: {output_dim}")
    print(
        f"stored nonzeros: {sparse_values._nnz()} "
        f"({sparse_values._nnz() / dense_values.numel():.4%})"
    )
    print(f"dense input storage: {dense_bytes / 2**20:.2f} MiB")
    print(f"sparse input storage: {sparse_bytes / 2**20:.2f} MiB")
    print(f"input storage reduction: {dense_bytes / sparse_bytes:.2f}x")
    print(f"dense median throughput: {batch_size / dense_seconds:,.0f} rows/second")
    print(f"sparse median throughput: {batch_size / sparse_seconds:,.0f} rows/second")


def _positive_integer(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def _density(value: str) -> float:
    parsed = float(value)
    if not 0 < parsed <= 1:
        raise argparse.ArgumentTypeError("density must be in (0, 1]")
    return parsed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=_positive_integer, default=512)
    parser.add_argument("--input-dim", type=_positive_integer, default=50_000)
    parser.add_argument("--output-dim", type=_positive_integer, default=128)
    parser.add_argument("--density", type=_density, default=0.001)
    parser.add_argument("--warmup", type=_positive_integer, default=3)
    parser.add_argument("--iterations", type=_positive_integer, default=10)
    parser.add_argument("--seed", type=int, default=7)
    arguments = parser.parse_args()
    benchmark(**vars(arguments))


if __name__ == "__main__":
    main()
