"""Validated memory and disk caches for frozen foundation-model outputs."""

from __future__ import annotations

import hashlib
import inspect
import json
import os
import pickle
import tempfile
from collections import OrderedDict
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

__all__ = [
    "FoundationCacheSpec",
    "FoundationOutputCache",
]

_CACHE_FORMAT_VERSION = 1


@dataclass(frozen=True)
class FoundationCacheSpec:
    """Serializable settings for pooled frozen-backbone output caching.

    Supplying this specification enables caching. ``directory=None`` creates
    a process-local memory cache. A directory adds persistent entries that can
    be shared by later runs with exactly the same foundation-model identity.
    """

    directory: str | None = None
    memory_capacity: int = 1024

    def __post_init__(self) -> None:
        if self.directory is not None and (
            not isinstance(self.directory, str) or not self.directory.strip()
        ):
            raise ValueError("Foundation cache directory must be a non-empty string.")
        if (
            isinstance(self.memory_capacity, bool)
            or not isinstance(self.memory_capacity, int)
            or self.memory_capacity < 0
        ):
            raise ValueError(
                "Foundation cache memory_capacity must be a non-negative integer."
            )
        if self.directory is None and self.memory_capacity == 0:
            raise ValueError(
                "Foundation cache requires a positive memory_capacity or a directory."
            )

    @classmethod
    def from_config(
        cls,
        value: FoundationCacheSpec | Mapping[str, Any],
    ) -> FoundationCacheSpec:
        """Validate a public mapping as cache settings."""

        if isinstance(value, cls):
            return value
        if not isinstance(value, Mapping):
            raise TypeError("cache must be a mapping.")
        unknown = set(value).difference(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(
                "cache contains unknown settings: "
                f"{sorted(str(name) for name in unknown)}."
            )
        return cls(**dict(value))

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-compatible cache settings."""

        return {
            "directory": self.directory,
            "memory_capacity": self.memory_capacity,
        }


class FoundationOutputCache:
    """LRU and optional persistent cache for one pooled-output identity."""

    def __init__(
        self,
        spec: FoundationCacheSpec,
        *,
        identity: Mapping[str, Any],
        hidden_size: int,
        dtype: torch.dtype,
    ) -> None:
        if (
            isinstance(hidden_size, bool)
            or not isinstance(hidden_size, int)
            or hidden_size <= 0
        ):
            raise ValueError("Foundation cache hidden_size must be positive.")
        if not isinstance(dtype, torch.dtype):
            raise TypeError("Foundation cache dtype must be a torch.dtype.")

        self.spec = spec
        self.identity = _json_mapping(identity, name="foundation cache identity")
        self.hidden_size = hidden_size
        self.dtype = dtype
        self.dtype_name = str(dtype).removeprefix("torch.")
        self.identity_digest = _json_digest(
            {
                "format_version": _CACHE_FORMAT_VERSION,
                "identity": self.identity,
                "hidden_size": hidden_size,
                "dtype": self.dtype_name,
            }
        )
        self._memory: OrderedDict[str, torch.Tensor] = OrderedDict()
        self._memory_hits = 0
        self._disk_hits = 0
        self._misses = 0
        self._disk_writes = 0
        self._namespace_directory: Path | None = None
        self._entries_directory: Path | None = None

        if spec.directory is not None:
            namespace_directory = (
                Path(spec.directory).expanduser() / self.identity_digest
            )
            entries_directory = namespace_directory / "entries"
            try:
                entries_directory.mkdir(parents=True, exist_ok=True)
            except OSError as error:
                raise ValueError(
                    "Foundation cache directory cannot be created or accessed: "
                    f"{namespace_directory}."
                ) from error
            self._namespace_directory = namespace_directory
            self._entries_directory = entries_directory
            self._prepare_manifest()

    @property
    def namespace_directory(self) -> Path | None:
        """Resolved identity-specific persistent directory, when enabled."""

        return self._namespace_directory

    def __deepcopy__(self, memo: dict[int, Any]) -> FoundationOutputCache:
        """Copy cache configuration without duplicating derived tensor entries."""

        copied = type(self)(
            self.spec,
            identity=self.identity,
            hidden_size=self.hidden_size,
            dtype=self.dtype,
        )
        memo[id(self)] = copied
        return copied

    def key_for_tokens(self, token_ids: torch.Tensor) -> str:
        """Build a stable content key from one unpadded token sequence."""

        if not isinstance(token_ids, torch.Tensor) or token_ids.ndim != 1:
            raise TypeError(
                "Foundation cache token IDs must be a one-dimensional tensor."
            )
        if token_ids.numel() == 0:
            raise ValueError("Foundation cache token sequences cannot be empty.")
        if token_ids.dtype not in {torch.int32, torch.int64}:
            raise TypeError("Foundation cache token IDs must use an integer dtype.")
        normalized = np.asarray(
            token_ids.detach().to(device="cpu", dtype=torch.int64).tolist(),
            dtype="<i8",
        )
        digest = hashlib.sha256()
        digest.update(len(normalized).to_bytes(8, byteorder="little", signed=False))
        digest.update(normalized.tobytes())
        return digest.hexdigest()

    def get(self, key: str) -> torch.Tensor | None:
        """Return one validated CPU tensor, or ``None`` for a cache miss."""

        value = self._memory.get(key)
        if value is not None:
            self._memory.move_to_end(key)
            self._memory_hits += 1
            return value

        path = self._entry_path(key)
        if path is not None and path.is_file():
            value = self._read_entry(path, key)
            self._remember(key, value)
            self._disk_hits += 1
            return value

        self._misses += 1
        return None

    def put(self, key: str, value: torch.Tensor) -> None:
        """Validate and store one pooled CPU representation."""

        normalized = self._validate_tensor(value, key=key).detach().cpu().clone()
        self._remember(key, normalized)
        path = self._entry_path(key)
        if path is None:
            return
        if path.exists():
            self._read_entry(path, key)
            return
        _atomic_torch_save(
            path,
            {
                "format_version": _CACHE_FORMAT_VERSION,
                "identity_digest": self.identity_digest,
                "key": key,
                "tensor": normalized,
            },
        )
        self._disk_writes += 1

    def stats(self) -> dict[str, Any]:
        """Return JSON-compatible hit, miss, storage, and identity statistics."""

        return {
            "enabled": True,
            "identity": self.identity_digest,
            "directory": (
                None
                if self._namespace_directory is None
                else str(self._namespace_directory)
            ),
            "memory_capacity": self.spec.memory_capacity,
            "memory_entries": len(self._memory),
            "memory_hits": self._memory_hits,
            "disk_hits": self._disk_hits,
            "misses": self._misses,
            "disk_writes": self._disk_writes,
        }

    def clear(self, *, memory: bool = True, disk: bool = False) -> None:
        """Clear memory entries and, when explicit, this namespace's disk entries."""

        if not isinstance(memory, bool) or not isinstance(disk, bool):
            raise TypeError("Foundation cache clear flags must be booleans.")
        if memory:
            self._memory.clear()
        if disk and self._entries_directory is not None:
            try:
                for path in self._entries_directory.iterdir():
                    if path.is_file() and path.suffix == ".pt":
                        path.unlink()
            except OSError as error:
                raise ValueError(
                    "Foundation cache disk entries could not be cleared from "
                    f"{self._entries_directory}."
                ) from error

    def _prepare_manifest(self) -> None:
        if self._namespace_directory is None:  # pragma: no cover - constructor guard
            return
        manifest_path = self._namespace_directory / "manifest.json"
        expected = {
            "format_version": _CACHE_FORMAT_VERSION,
            "identity_digest": self.identity_digest,
            "identity": self.identity,
            "hidden_size": self.hidden_size,
            "dtype": self.dtype_name,
        }
        if manifest_path.exists():
            try:
                observed = json.loads(manifest_path.read_text(encoding="utf-8"))
            except (OSError, UnicodeError, json.JSONDecodeError) as error:
                raise ValueError(
                    f"Foundation cache manifest is unreadable: {manifest_path}."
                ) from error
            if observed != expected:
                raise ValueError(
                    "Foundation cache manifest is stale or incompatible with the "
                    f"requested model identity: {manifest_path}."
                )
            return
        _atomic_write_text(
            manifest_path,
            json.dumps(expected, indent=2, sort_keys=True, allow_nan=False),
        )

    def _entry_path(self, key: str) -> Path | None:
        if (
            not isinstance(key, str)
            or len(key) != 64
            or any(character not in "0123456789abcdef" for character in key)
        ):
            raise ValueError("Foundation cache keys must be SHA-256 hex digests.")
        return (
            None
            if self._entries_directory is None
            else self._entries_directory / f"{key}.pt"
        )

    def _read_entry(self, path: Path, key: str) -> torch.Tensor:
        load_arguments: dict[str, Any] = {"map_location": "cpu"}
        if "weights_only" in inspect.signature(torch.load).parameters:
            load_arguments["weights_only"] = True
        try:
            payload = torch.load(path, **load_arguments)
        except (
            OSError,
            RuntimeError,
            EOFError,
            ValueError,
            pickle.UnpicklingError,
        ) as error:
            raise ValueError(
                f"Foundation cache entry is unreadable: {path}."
            ) from error
        if not isinstance(payload, Mapping):
            raise ValueError(f"Foundation cache entry must contain a mapping: {path}.")
        if (
            payload.get("format_version") != _CACHE_FORMAT_VERSION
            or payload.get("identity_digest") != self.identity_digest
            or payload.get("key") != key
        ):
            raise ValueError(
                f"Foundation cache entry is stale or incompatible: {path}."
            )
        value = payload.get("tensor")
        if not isinstance(value, torch.Tensor):
            raise ValueError(f"Foundation cache entry has no tensor: {path}.")
        return self._validate_tensor(value, key=key)

    def _validate_tensor(self, value: torch.Tensor, *, key: str) -> torch.Tensor:
        if value.device.type != "cpu":
            value = value.detach().cpu()
        if value.shape != (self.hidden_size,):
            raise ValueError(
                "Foundation cache entry has an incompatible pooled shape for "
                f"{key}: expected ({self.hidden_size},), received {tuple(value.shape)}."
            )
        if value.dtype != self.dtype:
            raise ValueError(
                "Foundation cache entry has an incompatible dtype for "
                f"{key}: expected {self.dtype_name}, received "
                f"{str(value.dtype).removeprefix('torch.')}."
            )
        if not torch.isfinite(value).all():
            raise ValueError(
                f"Foundation cache entry contains non-finite values for {key}."
            )
        return value

    def _remember(self, key: str, value: torch.Tensor) -> None:
        if self.spec.memory_capacity == 0:
            return
        self._memory[key] = value
        self._memory.move_to_end(key)
        while len(self._memory) > self.spec.memory_capacity:
            self._memory.popitem(last=False)


def _json_mapping(value: Mapping[str, Any], *, name: str) -> dict[str, Any]:
    normalized = dict(value)
    try:
        serialized = json.dumps(normalized, sort_keys=True, allow_nan=False)
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"{name} must contain only finite JSON-compatible values."
        ) from error
    restored = json.loads(serialized)
    if not isinstance(restored, dict):  # pragma: no cover - Mapping invariant
        raise ValueError(f"{name} must be a mapping.")
    return restored


def _json_digest(value: Mapping[str, Any]) -> str:
    serialized = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()


def _atomic_write_text(path: Path, value: str) -> None:
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temporary.write(value)
            temporary.flush()
            os.fsync(temporary.fileno())
            temporary_path = Path(temporary.name)
        os.replace(temporary_path, path)
    except OSError as error:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise ValueError(
            f"Foundation cache file could not be written: {path}."
        ) from error


def _atomic_torch_save(path: Path, value: Mapping[str, Any]) -> None:
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            torch.save(dict(value), temporary)
            temporary.flush()
            os.fsync(temporary.fileno())
        os.replace(temporary_path, path)
    except (OSError, RuntimeError) as error:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise ValueError(
            f"Foundation cache entry could not be written: {path}."
        ) from error
