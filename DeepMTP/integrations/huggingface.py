"""Optional Hugging Face tokenization and sequence-encoder integration."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from contextlib import nullcontext
from dataclasses import dataclass, replace
from importlib import import_module
from typing import Any, ClassVar, Literal, TypeAlias, cast

import numpy as np
import torch
from torch import nn

from DeepMTP.data.batches import BranchInputKind, SequenceInput
from DeepMTP.data.sequences import normalize_sequence_rows
from DeepMTP.integrations.foundation_cache import (
    FoundationCacheSpec,
    FoundationOutputCache,
)

__all__ = [
    "FoundationCacheSpec",
    "FoundationLoRASpec",
    "FoundationModelSpec",
    "HuggingFaceSequenceEncoder",
    "TokenizedFoundationSequences",
    "tokenize_foundation_sequences",
]

FoundationPooling: TypeAlias = Literal["cls", "masked_mean", "pooler"]
FoundationTrainMode: TypeAlias = Literal["frozen", "full", "lora"]


def _optional_dependency_error() -> ModuleNotFoundError:
    return ModuleNotFoundError(
        "Foundation-model integration requires the optional Transformers "
        'dependency. Install it with: pip install "DeepMTP[foundation]"'
    )


def _transformers_module() -> Any:
    try:
        return import_module("transformers")
    except ModuleNotFoundError as error:
        if error.name != "transformers":
            raise
        raise _optional_dependency_error() from error


def _peft_module() -> Any:
    try:
        return import_module("peft")
    except ModuleNotFoundError as error:
        if error.name != "peft":
            raise
        raise ModuleNotFoundError(
            "LoRA foundation-model training requires the optional PEFT "
            'dependency. Install it with: pip install "DeepMTP[foundation]"'
        ) from error


def _non_empty_optional_string(name: str, value: object) -> None:
    if value is not None and (not isinstance(value, str) or not value.strip()):
        raise ValueError(f"{name} must be a non-empty string or None.")


@dataclass(frozen=True)
class FoundationLoRASpec:
    """Serializable LoRA settings supported by the foundation adapter."""

    target_modules: tuple[str, ...]
    rank: int = 8
    alpha: int = 16
    dropout: float = 0.0

    def __post_init__(self) -> None:
        for name in ("rank", "alpha"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"LoRA {name} must be a positive integer.")
        if (
            isinstance(self.dropout, bool)
            or not isinstance(self.dropout, (int, float))
            or not 0 <= self.dropout < 1
        ):
            raise ValueError("LoRA dropout must be in the interval [0, 1).")
        if isinstance(self.target_modules, (str, bytes)) or not isinstance(
            self.target_modules,
            (list, tuple),
        ):
            raise ValueError("LoRA target_modules must be a list or tuple of names.")
        target_modules = tuple(
            target.strip() if isinstance(target, str) else target
            for target in self.target_modules
        )
        if not target_modules or any(
            not isinstance(target, str) or not target for target in target_modules
        ):
            raise ValueError(
                "LoRA target_modules must contain at least one non-empty name."
            )
        if len(target_modules) != len(set(target_modules)):
            raise ValueError("LoRA target_modules must not contain duplicates.")

        object.__setattr__(self, "dropout", float(self.dropout))
        object.__setattr__(self, "target_modules", target_modules)

    @classmethod
    def from_config(
        cls,
        value: FoundationLoRASpec | Mapping[str, Any],
    ) -> FoundationLoRASpec:
        """Validate a public mapping as a LoRA specification."""

        if isinstance(value, cls):
            return value
        if not isinstance(value, Mapping):
            raise TypeError("lora must be a mapping.")
        unknown = set(value).difference(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(
                "lora contains unknown settings: "
                f"{sorted(str(name) for name in unknown)}."
            )
        return cls(**dict(value))

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-compatible LoRA settings."""

        return {
            "rank": self.rank,
            "alpha": self.alpha,
            "dropout": self.dropout,
            "target_modules": list(self.target_modules),
        }


@dataclass(frozen=True)
class FoundationModelSpec:
    """Serializable model, tokenizer, pooling, and training settings."""

    model_name: str
    revision: str | None = None
    tokenizer_name: str | None = None
    tokenizer_revision: str | None = None
    pooling: FoundationPooling | str = "masked_mean"
    train_mode: FoundationTrainMode | str = "frozen"
    trust_remote_code: bool = False
    local_files_only: bool = False
    max_length: int | None = None
    truncation: bool = False
    exclude_special_tokens_from_pooling: bool = True
    pooling_excluded_token_ids: tuple[int, ...] = ()
    resolved_model_revision: str | None = None
    resolved_tokenizer_revision: str | None = None
    lora: FoundationLoRASpec | Mapping[str, Any] | None = None
    cache: FoundationCacheSpec | Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.model_name, str) or not self.model_name.strip():
            raise ValueError("model_name must be a non-empty string.")
        for name in (
            "revision",
            "tokenizer_name",
            "tokenizer_revision",
            "resolved_model_revision",
            "resolved_tokenizer_revision",
        ):
            _non_empty_optional_string(name, getattr(self, name))

        pooling = (
            self.pooling.lower() if isinstance(self.pooling, str) else self.pooling
        )
        if pooling not in {"cls", "masked_mean", "pooler"}:
            raise ValueError(
                "pooling must be 'cls', 'masked_mean', or 'pooler'; "
                f"received {self.pooling!r}."
            )
        train_mode = (
            self.train_mode.lower()
            if isinstance(self.train_mode, str)
            else self.train_mode
        )
        if train_mode not in {"frozen", "full", "lora"}:
            raise ValueError(
                "train_mode must be 'frozen', 'full', or 'lora'; "
                f"received {self.train_mode!r}."
            )
        lora = (
            FoundationLoRASpec.from_config(self.lora) if self.lora is not None else None
        )
        cache = (
            FoundationCacheSpec.from_config(self.cache)
            if self.cache is not None
            else None
        )
        if train_mode == "lora" and lora is None:
            raise ValueError("train_mode='lora' requires explicit lora settings.")
        if train_mode != "lora" and lora is not None:
            raise ValueError("lora settings require train_mode='lora'.")
        if cache is not None and train_mode != "frozen":
            raise ValueError(
                "Foundation output caching requires train_mode='frozen' because "
                "fine-tuned backbone representations change during training."
            )
        for name in (
            "trust_remote_code",
            "local_files_only",
            "truncation",
            "exclude_special_tokens_from_pooling",
        ):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"{name} must be a boolean.")
        excluded_token_ids = self.pooling_excluded_token_ids
        if not isinstance(excluded_token_ids, (list, tuple)) or any(
            isinstance(token_id, bool) or not isinstance(token_id, int) or token_id < 0
            for token_id in excluded_token_ids
        ):
            raise ValueError(
                "pooling_excluded_token_ids must contain non-negative integers."
            )
        if len(excluded_token_ids) != len(set(excluded_token_ids)):
            raise ValueError("pooling_excluded_token_ids must not contain duplicates.")
        if self.max_length is not None and (
            isinstance(self.max_length, bool)
            or not isinstance(self.max_length, int)
            or self.max_length <= 0
        ):
            raise ValueError("max_length must be a positive integer or None.")
        if self.truncation and self.max_length is None:
            raise ValueError("truncation=True requires an explicit max_length.")

        object.__setattr__(self, "pooling", pooling)
        object.__setattr__(self, "train_mode", train_mode)
        object.__setattr__(self, "lora", lora)
        object.__setattr__(self, "cache", cache)
        object.__setattr__(
            self,
            "pooling_excluded_token_ids",
            tuple(excluded_token_ids),
        )
        if self.tokenizer_name is None:
            object.__setattr__(self, "tokenizer_name", self.model_name)
        if self.tokenizer_revision is None:
            object.__setattr__(self, "tokenizer_revision", self.revision)

    @classmethod
    def from_config(
        cls,
        value: FoundationModelSpec | Mapping[str, Any] | None,
    ) -> FoundationModelSpec:
        """Validate a public mapping as a foundation-model specification."""

        if isinstance(value, cls):
            return value
        if not isinstance(value, Mapping):
            raise TypeError("foundation_model must be a mapping.")
        unknown = set(value).difference(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(
                "foundation_model contains unknown settings: "
                f"{sorted(str(name) for name in unknown)}."
            )
        return cls(**dict(value))

    @property
    def model_load_revision(self) -> str | None:
        """Immutable resolved revision when known, otherwise the requested one."""

        return self.resolved_model_revision or self.revision

    @property
    def tokenizer_load_revision(self) -> str | None:
        """Immutable tokenizer revision when known, otherwise the requested one."""

        return self.resolved_tokenizer_revision or self.tokenizer_revision

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-compatible settings for config, W&B, and checkpoints."""

        return {
            "model_name": self.model_name,
            "revision": self.revision,
            "tokenizer_name": self.tokenizer_name,
            "tokenizer_revision": self.tokenizer_revision,
            "pooling": self.pooling,
            "train_mode": self.train_mode,
            "lora": (
                self.lora.to_dict()
                if isinstance(self.lora, FoundationLoRASpec)
                else None
            ),
            "cache": (
                self.cache.to_dict()
                if isinstance(self.cache, FoundationCacheSpec)
                else None
            ),
            "trust_remote_code": self.trust_remote_code,
            "local_files_only": self.local_files_only,
            "max_length": self.max_length,
            "truncation": self.truncation,
            "exclude_special_tokens_from_pooling": (
                self.exclude_special_tokens_from_pooling
            ),
            "pooling_excluded_token_ids": list(self.pooling_excluded_token_ids),
            "resolved_model_revision": self.resolved_model_revision,
            "resolved_tokenizer_revision": self.resolved_tokenizer_revision,
        }


@dataclass(frozen=True)
class TokenizedFoundationSequences(Sequence[np.ndarray]):
    """Unpadded token rows plus their reproducible tokenizer settings."""

    rows: tuple[np.ndarray, ...]
    vocabulary_size: int
    padding_idx: int
    spec: FoundationModelSpec

    def __post_init__(self) -> None:
        if not self.rows:
            raise ValueError("Tokenized foundation sequences cannot be empty.")
        if (
            isinstance(self.vocabulary_size, bool)
            or not isinstance(self.vocabulary_size, int)
            or self.vocabulary_size <= 0
        ):
            raise ValueError("vocabulary_size must be a positive integer.")
        if (
            isinstance(self.padding_idx, bool)
            or not isinstance(self.padding_idx, int)
            or not 0 <= self.padding_idx < self.vocabulary_size
        ):
            raise ValueError(
                "padding_idx must be a non-negative integer smaller than "
                "vocabulary_size."
            )

    def __getitem__(self, index: int | slice) -> np.ndarray | Sequence[np.ndarray]:
        return self.rows[index]

    def __len__(self) -> int:
        return len(self.rows)


def _load_huggingface_tokenizer(spec: FoundationModelSpec) -> Any:
    transformers = _transformers_module()
    kwargs: dict[str, Any] = {
        "trust_remote_code": spec.trust_remote_code,
        "local_files_only": spec.local_files_only,
    }
    if spec.tokenizer_load_revision is not None:
        kwargs["revision"] = spec.tokenizer_load_revision
    return transformers.AutoTokenizer.from_pretrained(
        cast(str, spec.tokenizer_name),
        **kwargs,
    )


def _resolved_revision(value: object, fallback: str | None) -> str | None:
    init_kwargs = getattr(value, "init_kwargs", None)
    if isinstance(init_kwargs, Mapping):
        revision = init_kwargs.get("_commit_hash")
        if isinstance(revision, str) and revision:
            return revision
    config = getattr(value, "config", None)
    revision = getattr(config, "_commit_hash", None)
    return revision if isinstance(revision, str) and revision else fallback


def tokenize_foundation_sequences(
    values: Sequence[str],
    spec: FoundationModelSpec | Mapping[str, Any],
) -> TokenizedFoundationSequences:
    """Tokenize raw protein, SMILES, or text rows without batch padding."""

    normalized_spec = FoundationModelSpec.from_config(spec)
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise TypeError("Foundation sequences must be a sequence of strings.")
    raw_values = list(values)
    if not raw_values:
        raise ValueError("Foundation sequences must contain at least one row.")
    if any(not isinstance(value, str) or not value for value in raw_values):
        raise ValueError("Every foundation sequence must be a non-empty string.")

    tokenizer = _load_huggingface_tokenizer(normalized_spec)
    padding_idx = getattr(tokenizer, "pad_token_id", None)
    if isinstance(padding_idx, bool) or not isinstance(padding_idx, int):
        raise ValueError(
            "The selected foundation tokenizer must declare an integer pad_token_id."
        )
    try:
        vocabulary_size = len(tokenizer)
    except TypeError as error:
        raise ValueError(
            "The selected foundation tokenizer must declare its vocabulary size."
        ) from error

    tokenizer_kwargs: dict[str, Any] = {
        "add_special_tokens": True,
        "padding": False,
        "truncation": normalized_spec.truncation,
        "return_attention_mask": False,
        "return_token_type_ids": False,
    }
    if normalized_spec.max_length is not None:
        tokenizer_kwargs["max_length"] = normalized_spec.max_length
    encoded = tokenizer(raw_values, **tokenizer_kwargs)
    if not isinstance(encoded, Mapping) or "input_ids" not in encoded:
        raise TypeError(
            "The selected foundation tokenizer must return an input_ids mapping."
        )
    token_rows, _ = normalize_sequence_rows(
        encoded["input_ids"],
        entity_name="foundation",
        vocabulary_size=vocabulary_size,
        padding_idx=padding_idx,
    )
    if (
        not normalized_spec.truncation
        and normalized_spec.max_length is not None
        and any(len(row) > normalized_spec.max_length for row in token_rows)
    ):
        longest = max(len(row) for row in token_rows)
        raise ValueError(
            "A tokenized foundation sequence exceeds max_length="
            f"{normalized_spec.max_length}; observed {longest}. Enable truncation "
            "explicitly or choose a larger limit."
        )

    resolved_tokenizer_revision = _resolved_revision(
        tokenizer,
        normalized_spec.resolved_tokenizer_revision,
    )
    tokenizer_excluded_token_ids = tuple(
        dict.fromkeys(
            token_id
            for name in (
                "cls_token_id",
                "sep_token_id",
                "bos_token_id",
                "eos_token_id",
            )
            if isinstance((token_id := getattr(tokenizer, name, None)), int)
            and not isinstance(token_id, bool)
            and token_id != padding_idx
        )
    )
    excluded_token_ids = tuple(
        dict.fromkeys(
            (
                *normalized_spec.pooling_excluded_token_ids,
                *tokenizer_excluded_token_ids,
            )
        )
    )
    return TokenizedFoundationSequences(
        rows=tuple(token_rows),
        vocabulary_size=vocabulary_size,
        padding_idx=padding_idx,
        spec=replace(
            normalized_spec,
            pooling_excluded_token_ids=excluded_token_ids,
            resolved_tokenizer_revision=resolved_tokenizer_revision,
        ),
    )


def _load_huggingface_model(spec: FoundationModelSpec) -> nn.Module:
    transformers = _transformers_module()
    kwargs: dict[str, Any] = {
        "trust_remote_code": spec.trust_remote_code,
        "local_files_only": spec.local_files_only,
    }
    if spec.model_load_revision is not None:
        kwargs["revision"] = spec.model_load_revision
    model = transformers.AutoModel.from_pretrained(spec.model_name, **kwargs)
    if not isinstance(model, nn.Module):
        raise TypeError("AutoModel.from_pretrained() must return torch.nn.Module.")
    return model


def _positive_config_integer(config: object, *names: str) -> int:
    for name in names:
        value = getattr(config, name, None)
        if isinstance(value, int) and not isinstance(value, bool) and value > 0:
            return value
    raise ValueError(
        "The selected foundation model must declare a positive hidden size "
        f"through one of: {', '.join(names)}."
    )


def _model_vocabulary_size(model: nn.Module) -> int | None:
    get_embeddings = getattr(model, "get_input_embeddings", None)
    if callable(get_embeddings):
        embeddings = get_embeddings()
        size = getattr(embeddings, "num_embeddings", None)
        if isinstance(size, int) and not isinstance(size, bool) and size > 0:
            return size
    size = getattr(getattr(model, "config", None), "vocab_size", None)
    return (
        size
        if isinstance(size, int) and not isinstance(size, bool) and size > 0
        else None
    )


def _model_output(outputs: object, name: str) -> object:
    if isinstance(outputs, Mapping):
        return outputs.get(name)
    return getattr(outputs, name, None)


def _module_dtype(module: nn.Module) -> torch.dtype:
    for parameter in module.parameters():
        if torch.is_floating_point(parameter):
            return parameter.dtype
    for buffer in module.buffers():
        if torch.is_floating_point(buffer):
            return buffer.dtype
    return torch.get_default_dtype()


def _foundation_cache_identity(
    spec: FoundationModelSpec,
    *,
    vocabulary_size: int,
    padding_idx: int,
    hidden_size: int,
) -> dict[str, Any]:
    return {
        "model_name": spec.model_name,
        "model_revision": spec.resolved_model_revision,
        "tokenizer_name": spec.tokenizer_name,
        "tokenizer_revision": spec.resolved_tokenizer_revision,
        "pooling": spec.pooling,
        "max_length": spec.max_length,
        "truncation": spec.truncation,
        "exclude_special_tokens_from_pooling": (
            spec.exclude_special_tokens_from_pooling
        ),
        "pooling_excluded_token_ids": list(spec.pooling_excluded_token_ids),
        "vocabulary_size": vocabulary_size,
        "padding_idx": padding_idx,
        "hidden_size": hidden_size,
    }


def _validate_lora_target_modules(
    model: nn.Module,
    spec: FoundationLoRASpec,
) -> None:
    module_names = tuple(name for name, _module in model.named_modules() if name)
    missing = [
        target
        for target in spec.target_modules
        if not any(
            name == target or name.endswith(f".{target}") for name in module_names
        )
    ]
    if not missing:
        return

    leaf_suffixes = sorted(
        {
            name.rsplit(".", maxsplit=1)[-1]
            for name, module in model.named_modules()
            if name and not any(module.children())
        }
    )
    examples = ", ".join(leaf_suffixes[:20]) or "<none>"
    raise ValueError(
        f"LoRA target_modules did not match the loaded model: {missing}. "
        f"Available leaf-module suffixes include: {examples}."
    )


def _apply_lora(model: nn.Module, spec: FoundationLoRASpec) -> nn.Module:
    peft = _peft_module()
    model.requires_grad_(False)
    config = peft.LoraConfig(
        task_type=peft.TaskType.FEATURE_EXTRACTION,
        inference_mode=False,
        r=spec.rank,
        lora_alpha=spec.alpha,
        lora_dropout=spec.dropout,
        target_modules=list(spec.target_modules),
        bias="none",
    )
    adapted = peft.get_peft_model(model, config)
    if not isinstance(adapted, nn.Module):
        raise TypeError("PEFT get_peft_model() must return torch.nn.Module.")
    if not any(parameter.requires_grad for parameter in adapted.parameters()):
        raise ValueError("PEFT did not create any trainable LoRA parameters.")
    return adapted


class HuggingFaceSequenceEncoder(nn.Module):
    """Pool and project one Hugging Face encoder as a DeepMTP branch."""

    input_kind: ClassVar[BranchInputKind] = "sequence"

    def __init__(
        self,
        *,
        spec: FoundationModelSpec | Mapping[str, Any],
        vocabulary_size: int,
        output_dim: int,
        padding_idx: int,
    ) -> None:
        super().__init__()
        for name, value in {
            "vocabulary_size": vocabulary_size,
            "output_dim": output_dim,
        }.items():
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer.")
        if (
            isinstance(padding_idx, bool)
            or not isinstance(padding_idx, int)
            or not 0 <= padding_idx < vocabulary_size
        ):
            raise ValueError(
                "padding_idx must be a non-negative integer smaller than "
                "vocabulary_size."
            )

        normalized_spec = FoundationModelSpec.from_config(spec)
        backbone = _load_huggingface_model(normalized_spec)
        model_vocabulary_size = _model_vocabulary_size(backbone)
        if (
            model_vocabulary_size is not None
            and model_vocabulary_size != vocabulary_size
        ):
            raise ValueError(
                "Foundation model vocabulary size does not match the configured "
                f"SEQUENCE vocabulary: expected {model_vocabulary_size}, "
                f"received {vocabulary_size}."
            )
        model_padding_idx = getattr(
            getattr(backbone, "config", None),
            "pad_token_id",
            None,
        )
        if model_padding_idx is not None and model_padding_idx != padding_idx:
            raise ValueError(
                "Foundation model pad_token_id does not match the configured "
                f"SEQUENCE padding_idx: expected {model_padding_idx}, "
                f"received {padding_idx}."
            )

        hidden_size = _positive_config_integer(
            getattr(backbone, "config", None),
            "hidden_size",
            "d_model",
            "dim",
        )
        resolved_model_revision = _resolved_revision(
            backbone,
            normalized_spec.resolved_model_revision,
        )
        if normalized_spec.train_mode == "lora":
            lora_spec = cast(FoundationLoRASpec, normalized_spec.lora)
            _validate_lora_target_modules(backbone, lora_spec)
            backbone = _apply_lora(backbone, lora_spec)
        self.spec = replace(
            normalized_spec,
            resolved_model_revision=resolved_model_revision,
        )
        self.vocabulary_size = vocabulary_size
        self.padding_idx = padding_idx
        self.output_dim = output_dim
        self.hidden_size = hidden_size
        self.backbone = backbone
        self.projection = nn.Linear(hidden_size, output_dim)
        self._output_cache: FoundationOutputCache | None = None

        if self.spec.train_mode == "frozen":
            self.backbone.requires_grad_(False)
            self.backbone.eval()
        if isinstance(self.spec.cache, FoundationCacheSpec):
            if self.spec.cache.directory is not None:
                missing_revisions = [
                    name
                    for name, value in {
                        "resolved_model_revision": self.spec.resolved_model_revision,
                        "resolved_tokenizer_revision": (
                            self.spec.resolved_tokenizer_revision
                        ),
                    }.items()
                    if value is None
                ]
                if missing_revisions:
                    raise ValueError(
                        "Persistent foundation output caching requires exact "
                        "resolved revisions; missing "
                        f"{', '.join(missing_revisions)}. Tokenize with "
                        "tokenize_foundation_sequences() and use an immutable model "
                        "revision before configuring a disk cache."
                    )
            self._output_cache = FoundationOutputCache(
                self.spec.cache,
                identity=_foundation_cache_identity(
                    self.spec,
                    vocabulary_size=vocabulary_size,
                    padding_idx=padding_idx,
                    hidden_size=hidden_size,
                ),
                hidden_size=hidden_size,
                dtype=_module_dtype(self.backbone),
            )

    def train(self, mode: bool = True) -> HuggingFaceSequenceEncoder:
        """Keep a frozen backbone deterministic while training its projection."""

        super().train(mode)
        if self.spec.train_mode == "frozen":
            self.backbone.eval()
        return self

    def _pool(self, outputs: object, values: SequenceInput) -> torch.Tensor:
        if self.spec.pooling == "pooler":
            pooled = _model_output(outputs, "pooler_output")
            if not isinstance(pooled, torch.Tensor):
                raise ValueError(
                    "pooling='pooler' requires the foundation model to return "
                    "pooler_output."
                )
        else:
            hidden = _model_output(outputs, "last_hidden_state")
            if not isinstance(hidden, torch.Tensor) or hidden.ndim != 3:
                raise ValueError(
                    "The foundation model must return last_hidden_state with "
                    "shape [batch, sequence_length, hidden_size]."
                )
            if hidden.shape[:2] != values.token_ids.shape:
                raise ValueError(
                    "Foundation model last_hidden_state does not match the input "
                    "batch and sequence dimensions."
                )
            if self.spec.pooling == "cls":
                pooled = hidden[:, 0]
            else:
                pooling_mask = values.attention_mask
                if (
                    self.spec.exclude_special_tokens_from_pooling
                    and self.spec.pooling_excluded_token_ids
                ):
                    excluded = torch.zeros_like(pooling_mask)
                    for token_id in self.spec.pooling_excluded_token_ids:
                        excluded |= values.token_ids == token_id
                    pooling_mask = pooling_mask & ~excluded
                if not pooling_mask.any(dim=1).all():
                    raise ValueError(
                        "masked_mean pooling has no non-special token for at "
                        "least one sequence."
                    )
                mask = pooling_mask.unsqueeze(-1).to(hidden.dtype)
                pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1)

        if pooled.ndim != 2 or pooled.shape != (
            values.batch_size,
            self.hidden_size,
        ):
            raise ValueError(
                "Foundation model pooling must produce shape "
                f"[batch, {self.hidden_size}]; received {tuple(pooled.shape)}."
            )
        return pooled

    @property
    def cache_stats(self) -> dict[str, Any]:
        """JSON-compatible cache statistics for diagnostics and reporting."""

        if self._output_cache is None:
            return {"enabled": False}
        return self._output_cache.stats()

    def clear_cache(self, *, memory: bool = True, disk: bool = False) -> None:
        """Clear this encoder's derived memory and optionally disk entries."""

        if self._output_cache is not None:
            self._output_cache.clear(memory=memory, disk=disk)

    def _run_frozen_backbone(self, values: SequenceInput) -> torch.Tensor:
        with torch.no_grad():
            outputs = self.backbone(
                input_ids=values.token_ids.long(),
                attention_mask=values.attention_mask.long(),
            )
            return self._pool(outputs, values)

    def _cached_pool(self, values: SequenceInput) -> torch.Tensor:
        cache = self._output_cache
        if cache is None:  # pragma: no cover - caller invariant
            raise RuntimeError("Foundation output cache is not configured.")

        lengths = values.lengths.detach().cpu().tolist()
        token_ids_cpu = values.token_ids.detach().cpu()
        keys: list[str] = []
        key_to_index: dict[str, int] = {}
        pooled_by_key: dict[str, torch.Tensor] = {}
        for index, raw_length in enumerate(lengths):
            length = int(raw_length)
            key = cache.key_for_tokens(token_ids_cpu[index, :length])
            keys.append(key)
            if key in key_to_index or key in pooled_by_key:
                continue
            cached = cache.get(key)
            if cached is None:
                key_to_index[key] = index
            else:
                pooled_by_key[key] = cached

        if key_to_index:
            miss_indices = torch.tensor(
                list(key_to_index.values()),
                device=values.token_ids.device,
                dtype=torch.long,
            )
            miss_lengths = values.lengths.index_select(0, miss_indices)
            miss_width = int(miss_lengths.max())
            missing_values = SequenceInput(
                token_ids=values.token_ids.index_select(0, miss_indices)[
                    :, :miss_width
                ],
                attention_mask=values.attention_mask.index_select(0, miss_indices)[
                    :, :miss_width
                ],
                lengths=miss_lengths,
                padding_idx=values.padding_idx,
            )
            missing_pooled = self._run_frozen_backbone(missing_values)
            for key, pooled in zip(
                key_to_index,
                missing_pooled,
                strict=True,
            ):
                cpu_pooled = pooled.detach().cpu()
                cache.put(key, cpu_pooled)
                pooled_by_key[key] = cpu_pooled

        return torch.stack([pooled_by_key[key] for key in keys]).to(
            device=values.token_ids.device,
        )

    def forward(self, values: SequenceInput) -> torch.Tensor:
        if not isinstance(values, SequenceInput):
            raise TypeError("HuggingFaceSequenceEncoder expects a SequenceInput.")
        if values.padding_idx != self.padding_idx:
            raise ValueError(
                "Sequence input padding_idx does not match the foundation encoder: "
                f"expected {self.padding_idx}, received {values.padding_idx}."
            )
        token_ids = values.token_ids.long()
        if token_ids.numel() and (
            int(token_ids.min()) < 0 or int(token_ids.max()) >= self.vocabulary_size
        ):
            raise ValueError(
                "Sequence token IDs contain an index outside the configured "
                f"vocabulary [0, {self.vocabulary_size})."
            )

        if self._output_cache is not None:
            pooled = self._cached_pool(values)
        else:
            context = (
                torch.no_grad() if self.spec.train_mode == "frozen" else nullcontext()
            )
            with context:
                outputs = self.backbone(
                    input_ids=token_ids,
                    attention_mask=values.attention_mask.long(),
                )
                pooled = self._pool(outputs, values)
        return self.projection(pooled)
