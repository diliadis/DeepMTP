"""Verify one pinned Hugging Face checkpoint through the DeepMTP adapter."""

from __future__ import annotations

import argparse
import gc
import json
import re
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import torch

from DeepMTP import (
    FoundationLoRASpec,
    FoundationModelSpec,
    HuggingFaceSequenceEncoder,
    tokenize_foundation_sequences,
)
from DeepMTP.data import SequenceInput
from DeepMTP.data.sequences import collate_sequence_rows

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST_PATH = _PROJECT_ROOT / "scripts/foundation_compatibility_models.json"
_FULL_COMMIT_PATTERN = re.compile(r"^[0-9a-f]{40}$")


class CompatibilityManifestError(ValueError):
    """Raised when the reference-checkpoint manifest is malformed."""


@dataclass(frozen=True)
class ReferenceModel:
    """One immutable reference checkpoint and its compatibility inputs."""

    key: str
    domain: str
    model_name: str
    revision: str
    samples: tuple[str, ...]
    max_length: int
    pooling: str
    lora_target_modules: tuple[str, ...]


def _required_string(value: Mapping[str, Any], name: str) -> str:
    result = value.get(name)
    if not isinstance(result, str) or not result.strip():
        raise CompatibilityManifestError(f"{name} must be a non-empty string.")
    return result


def _string_tuple(value: Mapping[str, Any], name: str) -> tuple[str, ...]:
    result = value.get(name)
    if isinstance(result, (str, bytes)) or not isinstance(result, list):
        raise CompatibilityManifestError(f"{name} must be a non-empty string list.")
    normalized = tuple(result)
    if not normalized or any(
        not isinstance(item, str) or not item.strip() for item in normalized
    ):
        raise CompatibilityManifestError(f"{name} must be a non-empty string list.")
    return normalized


def _parse_reference(value: object) -> ReferenceModel:
    if not isinstance(value, Mapping):
        raise CompatibilityManifestError("Each model entry must be an object.")
    expected = {
        "key",
        "domain",
        "model_name",
        "revision",
        "samples",
        "max_length",
        "pooling",
        "lora_target_modules",
    }
    unknown = set(value).difference(expected)
    missing = expected.difference(value)
    if missing:
        raise CompatibilityManifestError(
            f"Model entry is missing fields: {sorted(missing)}."
        )
    if unknown:
        raise CompatibilityManifestError(
            f"Model entry contains unknown fields: {sorted(unknown)}."
        )

    revision = _required_string(value, "revision")
    if _FULL_COMMIT_PATTERN.fullmatch(revision) is None:
        raise CompatibilityManifestError(
            "revision must be a full 40-character lowercase commit SHA."
        )
    max_length = value.get("max_length")
    if (
        isinstance(max_length, bool)
        or not isinstance(max_length, int)
        or max_length <= 0
    ):
        raise CompatibilityManifestError("max_length must be a positive integer.")
    pooling = _required_string(value, "pooling")
    if pooling not in {"cls", "masked_mean", "pooler"}:
        raise CompatibilityManifestError(
            "pooling must be 'cls', 'masked_mean', or 'pooler'."
        )
    return ReferenceModel(
        key=_required_string(value, "key"),
        domain=_required_string(value, "domain"),
        model_name=_required_string(value, "model_name"),
        revision=revision,
        samples=_string_tuple(value, "samples"),
        max_length=max_length,
        pooling=pooling,
        lora_target_modules=_string_tuple(value, "lora_target_modules"),
    )


def load_reference_models(
    path: Path = DEFAULT_MANIFEST_PATH,
) -> dict[str, ReferenceModel]:
    """Load and validate the version-controlled compatibility manifest."""

    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise CompatibilityManifestError(
            f"Could not read compatibility manifest {path}: {error}"
        ) from error
    if not isinstance(value, Mapping):
        raise CompatibilityManifestError("Compatibility manifest must be an object.")
    if value.get("schema_version") != 1:
        raise CompatibilityManifestError(
            "Compatibility manifest schema_version must be 1."
        )
    entries = value.get("models")
    if not isinstance(entries, list) or not entries:
        raise CompatibilityManifestError(
            "Compatibility manifest models must be a non-empty list."
        )
    references = [_parse_reference(entry) for entry in entries]
    keys = [reference.key for reference in references]
    if len(keys) != len(set(keys)):
        raise CompatibilityManifestError("Compatibility model keys must be unique.")
    names = [reference.model_name for reference in references]
    if len(names) != len(set(names)):
        raise CompatibilityManifestError(
            "Compatibility model_name values must be unique."
        )
    return {reference.key: reference for reference in references}


def _package_version(package_name: str) -> str:
    try:
        return version(package_name)
    except PackageNotFoundError:
        return "not-installed"


def _environment() -> dict[str, str]:
    return {
        "python": sys.version.split()[0],
        "deepmtp": _package_version("DeepMTP"),
        "torch": torch.__version__,
        "transformers": _package_version("transformers"),
        "peft": _package_version("peft"),
        "huggingface_hub": _package_version("huggingface-hub"),
    }


def _batch(tokenized: Any) -> SequenceInput:
    token_ids, attention_mask, lengths = collate_sequence_rows(
        tokenized,
        padding_idx=tokenized.padding_idx,
    )
    return SequenceInput(
        token_ids=token_ids,
        attention_mask=attention_mask,
        lengths=lengths,
        padding_idx=tokenized.padding_idx,
    )


def _assert_revision(label: str, resolved: str | None, expected: str) -> None:
    if resolved != expected:
        raise ValueError(
            f"{label} revision {resolved!r}; expected pinned revision {expected!r}."
        )


def _run_encoder(
    *,
    reference: ReferenceModel,
    tokenized: Any,
    batch: SequenceInput,
    train_mode: str,
    output_dim: int,
) -> tuple[dict[str, Any], HuggingFaceSequenceEncoder]:
    lora = (
        FoundationLoRASpec(
            rank=2,
            alpha=4,
            target_modules=reference.lora_target_modules,
        )
        if train_mode == "lora"
        else None
    )
    spec = FoundationModelSpec.from_config(
        {
            **tokenized.spec.to_dict(),
            "train_mode": train_mode,
            "lora": None if lora is None else lora.to_dict(),
        }
    )
    encoder = HuggingFaceSequenceEncoder(
        spec=spec,
        vocabulary_size=tokenized.vocabulary_size,
        output_dim=output_dim,
        padding_idx=tokenized.padding_idx,
    )
    encoder.eval()
    with torch.no_grad():
        output = encoder(batch)
        repeated_output = encoder(batch)
    if tuple(output.shape) != (len(reference.samples), output_dim):
        raise ValueError(
            f"{train_mode} output shape was {tuple(output.shape)}; expected "
            f"{(len(reference.samples), output_dim)}."
        )
    if not bool(torch.isfinite(output).all()):
        raise ValueError(f"{train_mode} output contains non-finite values.")
    if not torch.allclose(output, repeated_output, rtol=1e-5, atol=1e-6):
        raise ValueError(f"{train_mode} evaluation output is not repeatable.")
    _assert_revision(
        "Model",
        encoder.spec.resolved_model_revision,
        reference.revision,
    )

    backbone_trainable = [
        name
        for name, parameter in encoder.backbone.named_parameters()
        if parameter.requires_grad
    ]
    if train_mode == "frozen" and backbone_trainable:
        raise ValueError("Frozen backbone unexpectedly has trainable parameters.")
    if train_mode == "lora" and (
        not backbone_trainable
        or any("lora_" not in name for name in backbone_trainable)
    ):
        raise ValueError(
            "LoRA backbone trainable parameters are missing or include non-LoRA "
            "parameters."
        )
    projection_trainable = sum(
        parameter.numel()
        for parameter in encoder.projection.parameters()
        if parameter.requires_grad
    )
    if projection_trainable <= 0:
        raise ValueError("The DeepMTP output projection must remain trainable.")

    return (
        {
            "output_shape": list(output.shape),
            "output_dtype": str(output.dtype),
            "finite": True,
            "repeatable": True,
            "hidden_size": encoder.hidden_size,
            "model_padding_idx": getattr(
                encoder.backbone.config,
                "pad_token_id",
                None,
            ),
            "resolved_model_revision": encoder.spec.resolved_model_revision,
            "backbone_trainable_parameters": sum(
                parameter.numel()
                for parameter in encoder.backbone.parameters()
                if parameter.requires_grad
            ),
            "projection_trainable_parameters": projection_trainable,
        },
        encoder,
    )


def check_reference_model(
    reference: ReferenceModel,
    *,
    output_dim: int = 8,
) -> dict[str, Any]:
    """Run frozen and LoRA compatibility checks for one reference model."""

    if (
        isinstance(output_dim, bool)
        or not isinstance(output_dim, int)
        or output_dim <= 0
    ):
        raise ValueError("output_dim must be a positive integer.")
    started = time.perf_counter()
    base_spec = FoundationModelSpec(
        model_name=reference.model_name,
        revision=reference.revision,
        pooling=reference.pooling,
        train_mode="frozen",
        trust_remote_code=False,
        max_length=reference.max_length,
        truncation=False,
    )
    tokenized = tokenize_foundation_sequences(reference.samples, base_spec)
    _assert_revision(
        "Tokenizer",
        tokenized.spec.tokenizer_load_revision,
        reference.revision,
    )
    batch = _batch(tokenized)
    frozen_result, frozen_encoder = _run_encoder(
        reference=reference,
        tokenized=tokenized,
        batch=batch,
        train_mode="frozen",
        output_dim=output_dim,
    )
    del frozen_encoder
    gc.collect()
    lora_result, lora_encoder = _run_encoder(
        reference=reference,
        tokenized=tokenized,
        batch=batch,
        train_mode="lora",
        output_dim=output_dim,
    )
    del lora_encoder
    gc.collect()

    return {
        "schema_version": 1,
        "status": "passed",
        "checked_at": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "reference": {
            "key": reference.key,
            "domain": reference.domain,
            "model_name": reference.model_name,
            "revision": reference.revision,
            "pooling": reference.pooling,
            "max_length": reference.max_length,
            "lora_target_modules": list(reference.lora_target_modules),
        },
        "environment": _environment(),
        "tokenization": {
            "rows": len(tokenized),
            "lengths": [int(row.shape[0]) for row in tokenized],
            "vocabulary_size": tokenized.vocabulary_size,
            "padding_idx": tokenized.padding_idx,
            "requested_tokenizer_revision": tokenized.spec.tokenizer_revision,
            "resolved_tokenizer_revision": (tokenized.spec.resolved_tokenizer_revision),
        },
        "modes": {"frozen": frozen_result, "lora": lora_result},
    }


def _error_category(error: Exception) -> str:
    if isinstance(error, CompatibilityManifestError):
        return "manifest"
    if isinstance(error, ModuleNotFoundError):
        return "environment"
    if isinstance(error, (ConnectionError, TimeoutError, OSError)):
        return "hub_or_download"
    if isinstance(error, (TypeError, ValueError)):
        return "compatibility_contract"
    return "runtime"


def _failure_report(model_key: str, error: Exception) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "status": "failed",
        "checked_at": datetime.now(timezone.utc).isoformat(),
        "reference": {"key": model_key},
        "environment": _environment(),
        "error": {
            "category": _error_category(error),
            "type": type(error).__name__,
            "message": str(error),
        },
    }


def write_report(report: Mapping[str, Any], output_path: Path) -> None:
    """Write a deterministic, human-readable JSON compatibility report."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Run one manifest entry and emit a machine-readable report."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Reference model key.")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_MANIFEST_PATH,
        help="Compatibility manifest path.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="JSON output path; defaults to foundation-compatibility-<model>.json.",
    )
    parser.add_argument("--output-dim", type=int, default=8)
    arguments = parser.parse_args(argv)
    output_path = arguments.output or Path(
        f"foundation-compatibility-{arguments.model}.json"
    )

    try:
        references = load_reference_models(arguments.manifest)
        try:
            reference = references[arguments.model]
        except KeyError as error:
            available = ", ".join(sorted(references))
            raise CompatibilityManifestError(
                f"Unknown model key {arguments.model!r}; choose one of: {available}."
            ) from error
        report = check_reference_model(reference, output_dim=arguments.output_dim)
    except Exception as error:
        report = _failure_report(arguments.model, error)
        write_report(report, output_path)
        print(
            f"Foundation compatibility check failed: {error}. Report: {output_path}",
            file=sys.stderr,
        )
        return 1

    write_report(report, output_path)
    print(
        f"Verified {reference.model_name} at {reference.revision}. "
        f"Report: {output_path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
