"""Compact, versioned checkpoints for PEFT foundation-model branches."""

from __future__ import annotations

import hashlib
import inspect
import io
import json
import os
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path, PurePosixPath
from typing import Any, cast
from zipfile import ZIP_DEFLATED, BadZipFile, ZipFile

import torch
from torch import nn

from DeepMTP.config import DeepMTPConfig
from DeepMTP.integrations.huggingface import (
    FoundationModelSpec,
    HuggingFaceSequenceEncoder,
)
from DeepMTP.runtime.metadata import validate_checkpoint_metadata

FOUNDATION_ADAPTER_CHECKPOINT_FORMAT_VERSION = 1
FOUNDATION_ADAPTER_CHECKPOINT_NAME = "foundation_adapter.zip"
FOUNDATION_ADAPTER_MANIFEST_NAME = "manifest.json"
FOUNDATION_ADAPTER_TASK_STATE_NAME = "task_state.pt"

__all__ = [
    "FOUNDATION_ADAPTER_CHECKPOINT_FORMAT_VERSION",
    "FOUNDATION_ADAPTER_CHECKPOINT_NAME",
    "FoundationAdapterCheckpoint",
    "FoundationAdapterExport",
    "load_foundation_adapter_checkpoint",
    "restore_foundation_adapter_checkpoint",
    "save_foundation_adapter_checkpoint",
]

_ARTIFACT_KIND = "deepmtp.foundation_adapter"
_ADAPTER_CONFIG_NAME = "adapter_config.json"
_ADAPTER_WEIGHTS_NAME = "adapter_model.safetensors"


@dataclass(frozen=True)
class _LoadedFoundationComponent:
    module_path: str
    foundation_model: dict[str, Any]
    train_mode: str
    adapter_state_dict: Mapping[str, torch.Tensor] | None


@dataclass(frozen=True)
class FoundationAdapterCheckpoint:
    """Validated compact checkpoint ready to restore into a built model."""

    checkpoint_path: Path
    config: dict[str, Any]
    metadata: dict[str, Any]
    task_state_dict: Mapping[str, Any]
    components: tuple[_LoadedFoundationComponent, ...]
    manifest: dict[str, Any]


@dataclass(frozen=True)
class FoundationAdapterExport:
    """A newly written compact checkpoint and its JSON manifest."""

    checkpoint_path: Path
    manifest: dict[str, Any]

    @property
    def artifact_metadata(self) -> dict[str, Any]:
        """Small W&B-safe summary retaining exact foundation identities."""

        return {
            "format_version": self.manifest["format_version"],
            "artifact_kind": self.manifest["artifact_kind"],
            "foundation_components": [
                {
                    "module_path": component["module_path"],
                    "train_mode": component["train_mode"],
                    "foundation_model": component["foundation_model"],
                }
                for component in self.manifest["foundation_components"]
            ],
            "checkpoint_metadata": self.manifest["checkpoint_metadata"],
        }


def save_foundation_adapter_checkpoint(
    checkpoint_path: str | os.PathLike[str],
    *,
    model: nn.Module,
    config: DeepMTPConfig | Mapping[str, Any],
    metadata: Mapping[str, Any],
) -> FoundationAdapterExport:
    """Save task weights and native PEFT adapters without base-model weights."""

    if not isinstance(model, nn.Module):
        raise TypeError("model must be a torch.nn.Module.")
    destination = Path(checkpoint_path)
    if destination.name != FOUNDATION_ADAPTER_CHECKPOINT_NAME:
        raise ValueError(
            "Foundation adapter checkpoints must use the filename "
            f"{FOUNDATION_ADAPTER_CHECKPOINT_NAME!r}."
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    normalized_config = (
        config.to_dict(include_defaults=True)
        if isinstance(config, DeepMTPConfig)
        else dict(config)
    )
    _require_json_mapping("config", normalized_config)
    normalized_metadata = validate_checkpoint_metadata(metadata)
    encoders = _foundation_encoders(model)
    if not encoders:
        raise ValueError(
            "checkpoint_format='foundation_adapter' requires at least one "
            "foundation-model encoder."
        )
    full_paths = [
        module_path
        for module_path, encoder in encoders.items()
        if encoder.spec.train_mode == "full"
    ]
    if full_paths:
        raise ValueError(
            "Compact foundation adapter checkpoints cannot omit fully fine-tuned "
            f"base weights; full-mode encoders: {full_paths}. Use "
            "checkpoint_format='full'."
        )
    if not any(encoder.spec.train_mode == "lora" for encoder in encoders.values()):
        raise ValueError(
            "checkpoint_format='foundation_adapter' requires at least one "
            "train_mode='lora' foundation encoder."
        )

    excluded_prefixes = tuple(f"{module_path}.backbone." for module_path in encoders)
    task_state = {
        name: _state_value_to_cpu(value)
        for name, value in model.state_dict().items()
        if not name.startswith(excluded_prefixes)
    }

    with tempfile.TemporaryDirectory(
        dir=destination.parent,
        prefix=f".{destination.stem}.",
    ) as temporary_directory:
        bundle_root = Path(temporary_directory) / "bundle"
        bundle_root.mkdir()
        task_state_path = bundle_root / FOUNDATION_ADAPTER_TASK_STATE_NAME
        torch.save(task_state, task_state_path)

        component_manifests = []
        for index, (module_path, encoder) in enumerate(encoders.items()):
            identity = _foundation_identity(encoder.spec, module_path=module_path)
            component_manifest: dict[str, Any] = {
                "module_path": module_path,
                "train_mode": encoder.spec.train_mode,
                "foundation_model": identity,
                "files": {},
            }
            if encoder.spec.train_mode == "lora":
                adapter_directory = bundle_root / "adapters" / f"{index:03d}"
                _save_native_peft_adapter(encoder, adapter_directory)
                adapter_config_path = adapter_directory / _ADAPTER_CONFIG_NAME
                adapter_weights_path = adapter_directory / _ADAPTER_WEIGHTS_NAME
                for required_path in (adapter_config_path, adapter_weights_path):
                    if not required_path.is_file():
                        raise ValueError(
                            "PEFT adapter export did not create required file "
                            f"{required_path.name!r} for {module_path!r}."
                        )
                relative_directory = adapter_directory.relative_to(bundle_root)
                component_manifest["adapter_directory"] = relative_directory.as_posix()
                component_manifest["files"] = {
                    relative_path.as_posix(): _sha256(bundle_root / relative_path)
                    for relative_path in (
                        relative_directory / _ADAPTER_CONFIG_NAME,
                        relative_directory / _ADAPTER_WEIGHTS_NAME,
                    )
                }
            component_manifests.append(component_manifest)

        manifest = {
            "artifact_kind": _ARTIFACT_KIND,
            "format_version": FOUNDATION_ADAPTER_CHECKPOINT_FORMAT_VERSION,
            "config": normalized_config,
            "checkpoint_metadata": normalized_metadata,
            "task_state": {
                "path": FOUNDATION_ADAPTER_TASK_STATE_NAME,
                "sha256": _sha256(task_state_path),
            },
            "foundation_components": component_manifests,
        }
        manifest_path = bundle_root / FOUNDATION_ADAPTER_MANIFEST_NAME
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False),
            encoding="utf-8",
        )

        temporary_zip = _temporary_path(destination)
        try:
            with ZipFile(temporary_zip, mode="w", compression=ZIP_DEFLATED) as archive:
                for path in sorted(bundle_root.rglob("*")):
                    if path.is_file():
                        archive.write(path, path.relative_to(bundle_root).as_posix())
            os.replace(temporary_zip, destination)
        finally:
            temporary_zip.unlink(missing_ok=True)
    return FoundationAdapterExport(destination, manifest)


def load_foundation_adapter_checkpoint(
    checkpoint_path: str | os.PathLike[str],
    *,
    map_location: str | torch.device = "cpu",
) -> FoundationAdapterCheckpoint:
    """Read and validate a compact foundation adapter checkpoint."""

    path = Path(checkpoint_path)
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {path}")
    try:
        with ZipFile(path) as archive:
            names = _validated_archive_names(archive)
            if FOUNDATION_ADAPTER_MANIFEST_NAME not in names:
                raise ValueError(
                    "Foundation adapter checkpoint contains no manifest.json."
                )
            manifest = _load_json_bytes(
                "manifest",
                archive.read(FOUNDATION_ADAPTER_MANIFEST_NAME),
            )
            _validate_manifest_header(manifest)
            config = _mapping_field(manifest, "config")
            metadata = validate_checkpoint_metadata(manifest.get("checkpoint_metadata"))
            task_info = _mapping_field(manifest, "task_state")
            task_path = _safe_member_path(task_info.get("path"), "task_state.path")
            task_bytes = _verified_member(
                archive,
                names,
                task_path,
                task_info.get("sha256"),
            )
            task_state = _load_torch_state(task_bytes, map_location=map_location)
            components = _load_components(archive, names, manifest)
    except BadZipFile as error:
        raise ValueError(
            f"Foundation adapter checkpoint is not a valid ZIP archive: {path}"
        ) from error
    return FoundationAdapterCheckpoint(
        checkpoint_path=path,
        config=config,
        metadata=metadata,
        task_state_dict=task_state,
        components=components,
        manifest=manifest,
    )


def restore_foundation_adapter_checkpoint(
    model: nn.Module,
    checkpoint: FoundationAdapterCheckpoint,
) -> None:
    """Restore validated task and LoRA state into a newly built DeepMTP model."""

    encoders = _foundation_encoders(model)
    expected_paths = {component.module_path for component in checkpoint.components}
    if set(encoders) != expected_paths:
        raise ValueError(
            "Foundation adapter checkpoint components do not match the built model: "
            f"checkpoint={sorted(expected_paths)}, model={sorted(encoders)}."
        )

    excluded_prefixes = tuple(f"{module_path}.backbone." for module_path in encoders)
    incompatible = model.load_state_dict(checkpoint.task_state_dict, strict=False)
    if incompatible.unexpected_keys:
        raise ValueError(
            "Foundation adapter task state contains unexpected parameters: "
            f"{sorted(incompatible.unexpected_keys)}."
        )
    missing_task_parameters = [
        name
        for name in incompatible.missing_keys
        if not name.startswith(excluded_prefixes)
    ]
    if missing_task_parameters:
        raise ValueError(
            "Foundation adapter task state is missing non-backbone parameters: "
            f"{sorted(missing_task_parameters)}."
        )

    peft = None
    for component in checkpoint.components:
        encoder = encoders[component.module_path]
        actual_identity = _foundation_identity(
            encoder.spec,
            module_path=component.module_path,
        )
        if actual_identity != component.foundation_model:
            raise ValueError(
                "Foundation model identity does not match the adapter checkpoint "
                f"for {component.module_path!r}."
            )
        if component.train_mode == "frozen":
            continue
        if component.adapter_state_dict is None:
            raise ValueError(
                f"LoRA component {component.module_path!r} has no adapter weights."
            )
        if peft is None:
            peft = _peft_module()
        outcome = peft.set_peft_model_state_dict(
            encoder.backbone,
            dict(component.adapter_state_dict),
            adapter_name="default",
        )
        unexpected_keys = tuple(getattr(outcome, "unexpected_keys", ()))
        if unexpected_keys:
            raise ValueError(
                f"PEFT reported unexpected adapter parameters for "
                f"{component.module_path!r}: {sorted(unexpected_keys)}."
            )


def _foundation_encoders(model: nn.Module) -> dict[str, HuggingFaceSequenceEncoder]:
    return {
        name: module
        for name, module in model.named_modules()
        if name and isinstance(module, HuggingFaceSequenceEncoder)
    }


def _foundation_identity(
    spec: FoundationModelSpec,
    *,
    module_path: str,
) -> dict[str, Any]:
    model_is_local = Path(spec.model_name).is_dir()
    tokenizer_name = cast(str, spec.tokenizer_name)
    tokenizer_is_local = Path(tokenizer_name).is_dir()
    if not model_is_local and spec.resolved_model_revision is None:
        raise ValueError(
            f"Foundation encoder {module_path!r} requires a resolved_model_revision "
            "before compact checkpoint export or restoration. Pin and reload the "
            "Hub model revision."
        )
    if not tokenizer_is_local and spec.resolved_tokenizer_revision is None:
        raise ValueError(
            f"Foundation encoder {module_path!r} requires a "
            "resolved_tokenizer_revision before compact checkpoint export or "
            "restoration. Pin and retokenize with the Hub tokenizer revision."
        )
    return {
        "spec": spec.to_dict(),
        "base_model": {
            "name": spec.model_name,
            "resolved_revision": spec.resolved_model_revision,
            "source": "local" if model_is_local else "hub",
        },
        "tokenizer": {
            "name": tokenizer_name,
            "resolved_revision": spec.resolved_tokenizer_revision,
            "source": "local" if tokenizer_is_local else "hub",
        },
    }


def _save_native_peft_adapter(
    encoder: HuggingFaceSequenceEncoder,
    directory: Path,
) -> None:
    save_pretrained = getattr(encoder.backbone, "save_pretrained", None)
    if not callable(save_pretrained):
        raise TypeError("LoRA foundation backbone does not expose save_pretrained().")
    directory.mkdir(parents=True)
    save_pretrained(
        str(directory),
        safe_serialization=True,
        selected_adapters=["default"],
        save_embedding_layers=False,
    )


def _load_components(
    archive: ZipFile,
    names: frozenset[str],
    manifest: Mapping[str, Any],
) -> tuple[_LoadedFoundationComponent, ...]:
    raw_components = manifest.get("foundation_components")
    if not isinstance(raw_components, list) or not raw_components:
        raise ValueError(
            "Foundation adapter manifest must contain foundation_components."
        )
    components = []
    seen_paths: set[str] = set()
    for raw_component in raw_components:
        if not isinstance(raw_component, Mapping):
            raise ValueError("Every foundation component must be a mapping.")
        module_path = raw_component.get("module_path")
        if (
            not isinstance(module_path, str)
            or not module_path
            or module_path in seen_paths
        ):
            raise ValueError(
                "Foundation component module_path values must be unique non-empty "
                "strings."
            )
        seen_paths.add(module_path)
        train_mode = raw_component.get("train_mode")
        if train_mode not in {"frozen", "lora"}:
            raise ValueError(
                f"Foundation component {module_path!r} has unsupported train_mode "
                f"{train_mode!r}."
            )
        foundation_model = _mapping_field(raw_component, "foundation_model")
        foundation_spec = _mapping_field(foundation_model, "spec")
        if foundation_spec.get("train_mode") != train_mode:
            raise ValueError(
                f"Foundation component {module_path!r} has inconsistent "
                "train_mode values."
            )
        adapter_state = None
        if train_mode == "lora":
            adapter_directory = _safe_member_path(
                raw_component.get("adapter_directory"),
                f"foundation component {module_path!r} adapter_directory",
            )
            files = _mapping_field(raw_component, "files")
            config_path = f"{adapter_directory}/{_ADAPTER_CONFIG_NAME}"
            weights_path = f"{adapter_directory}/{_ADAPTER_WEIGHTS_NAME}"
            config_bytes = _verified_member(
                archive,
                names,
                config_path,
                files.get(config_path),
            )
            _load_json_bytes(f"adapter config for {module_path}", config_bytes)
            weights_bytes = _verified_member(
                archive,
                names,
                weights_path,
                files.get(weights_path),
            )
            safetensors = import_module("safetensors.torch")
            loaded = safetensors.load(weights_bytes)
            if not isinstance(loaded, Mapping) or not loaded:
                raise ValueError(
                    f"PEFT adapter weights for {module_path!r} are empty or invalid."
                )
            adapter_state = dict(loaded)
        elif raw_component.get("adapter_directory") is not None:
            raise ValueError(
                f"Frozen foundation component {module_path!r} must not declare "
                "an adapter directory."
            )
        components.append(
            _LoadedFoundationComponent(
                module_path=module_path,
                foundation_model=foundation_model,
                train_mode=cast(str, train_mode),
                adapter_state_dict=adapter_state,
            )
        )
    return tuple(components)


def _validated_archive_names(archive: ZipFile) -> frozenset[str]:
    names = [member.filename for member in archive.infolist() if not member.is_dir()]
    if len(names) != len(set(names)):
        raise ValueError("Foundation adapter checkpoint contains duplicate paths.")
    for name in names:
        _safe_member_path(name, "archive member")
    return frozenset(names)


def _validate_manifest_header(manifest: Mapping[str, Any]) -> None:
    if manifest.get("artifact_kind") != _ARTIFACT_KIND:
        raise ValueError("File is not a DeepMTP foundation adapter checkpoint.")
    version = manifest.get("format_version")
    if version != FOUNDATION_ADAPTER_CHECKPOINT_FORMAT_VERSION:
        raise ValueError(
            "Unsupported foundation adapter checkpoint format version "
            f"{version!r}; expected "
            f"{FOUNDATION_ADAPTER_CHECKPOINT_FORMAT_VERSION}."
        )


def _mapping_field(value: Mapping[str, Any], name: str) -> dict[str, Any]:
    field_value = value.get(name)
    if not isinstance(field_value, Mapping):
        raise ValueError(
            f"Foundation adapter manifest field {name!r} must be a mapping."
        )
    return dict(field_value)


def _require_json_mapping(name: str, value: Mapping[str, Any]) -> None:
    try:
        json.dumps(value, allow_nan=False)
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"Foundation adapter checkpoint {name} must be finite and JSON-compatible."
        ) from error


def _load_json_bytes(name: str, content: bytes) -> dict[str, Any]:
    try:
        value = json.loads(content.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"Foundation adapter {name} is not valid JSON.") from error
    if not isinstance(value, Mapping):
        raise ValueError(f"Foundation adapter {name} must contain a JSON object.")
    return dict(value)


def _safe_member_path(value: object, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty relative path.")
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or path.as_posix() != value:
        raise ValueError(f"{name} must be a safe normalized relative path.")
    return value


def _verified_member(
    archive: ZipFile,
    names: frozenset[str],
    path: str,
    expected_sha256: object,
) -> bytes:
    if path not in names:
        raise ValueError(f"Foundation adapter checkpoint is missing {path!r}.")
    if not isinstance(expected_sha256, str) or len(expected_sha256) != 64:
        raise ValueError(
            f"Foundation adapter checkpoint has no valid hash for {path!r}."
        )
    content = archive.read(path)
    if hashlib.sha256(content).hexdigest() != expected_sha256:
        raise ValueError(
            f"Foundation adapter checkpoint integrity check failed for {path!r}."
        )
    return content


def _load_torch_state(
    content: bytes,
    *,
    map_location: str | torch.device,
) -> Mapping[str, Any]:
    arguments: dict[str, Any] = {"map_location": map_location}
    if "weights_only" in inspect.signature(torch.load).parameters:
        arguments["weights_only"] = True
    state = torch.load(io.BytesIO(content), **arguments)
    if not isinstance(state, Mapping):
        raise ValueError("Foundation adapter task_state.pt must contain a mapping.")
    return dict(state)


def _state_value_to_cpu(value: Any) -> Any:
    return value.detach().cpu() if isinstance(value, torch.Tensor) else value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _temporary_path(destination: Path) -> Path:
    descriptor, name = tempfile.mkstemp(
        dir=destination.parent,
        prefix=f".{destination.name}.",
        suffix=".tmp",
    )
    os.close(descriptor)
    return Path(name)


def _peft_module() -> Any:
    try:
        return import_module("peft")
    except ModuleNotFoundError as error:  # pragma: no cover - guarded by model build
        if error.name != "peft":
            raise
        raise ModuleNotFoundError(
            "Foundation adapter restoration requires PEFT. Install it with: "
            'pip install "DeepMTP[foundation]"'
        ) from error
