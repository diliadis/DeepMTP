"""Semantic and terminal-friendly descriptions of DeepMTP architectures."""

from __future__ import annotations

import importlib.util
import sys
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from io import StringIO
from os import PathLike
from pathlib import Path
from typing import Any, Literal, TextIO

from torch import nn

from DeepMTP.config import DeepMTPConfig
from DeepMTP.integrations.huggingface import FoundationLoRASpec, FoundationModelSpec

__all__ = [
    "ArchitectureBranch",
    "ArchitectureComponent",
    "ArchitectureDescription",
    "ArchitectureFusion",
    "ParameterStatistics",
    "describe_architecture",
    "export_architecture_svg",
    "format_architecture",
    "print_architecture",
]

RenderStyle = Literal["auto", "plain", "rich"]


@dataclass(frozen=True)
class ArchitectureComponent:
    """One independently encoded component inside a composite branch."""

    name: str
    architecture: str
    input_dimension: int
    output_dimension: int
    optional: bool
    details: tuple[str, ...] = ()


@dataclass(frozen=True)
class ArchitectureBranch:
    """A normalized instance or target branch description."""

    name: Literal["instance", "target"]
    architecture: str
    input_dimension: int | None
    output_dimension: int
    details: tuple[str, ...] = ()
    components: tuple[ArchitectureComponent, ...] = ()


@dataclass(frozen=True)
class ArchitectureFusion:
    """The operation that combines encoded instance and target branches."""

    architecture: str
    input_dimension: int
    output_dimension: int
    details: tuple[str, ...] = ()


@dataclass(frozen=True)
class ParameterStatistics:
    """Exact aggregate parameter counts from an instantiated model."""

    total: int
    trainable: int
    frozen: int

    @property
    def trainable_fraction(self) -> float:
        """Fraction of parameters whose gradients are enabled."""
        return self.trainable / self.total if self.total else 0.0


@dataclass(frozen=True)
class ArchitectureDescription:
    """Renderer-independent representation of a configured DeepMTP model."""

    validation_setting: str
    task: str
    loss: str
    instance_branch: ArchitectureBranch
    target_branch: ArchitectureBranch
    fusion: ArchitectureFusion
    parameters: ParameterStatistics | None = None
    warnings: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible representation for tooling and exports."""
        return asdict(self)


def _expanded_nodes(nodes: list[int] | int, layers: int | None) -> tuple[int, ...]:
    if isinstance(nodes, list):
        return tuple(nodes)
    return tuple(nodes for _ in range(layers or 0))


def _dimensions_text(dimensions: tuple[int, ...]) -> str:
    return " → ".join(f"{dimension:,}" for dimension in dimensions)


def _validate_width(width: int) -> None:
    if isinstance(width, bool) or not isinstance(width, int) or width < 40:
        raise ValueError("width must be an integer greater than or equal to 40.")


def _foundation_details(raw_spec: Mapping[str, Any]) -> tuple[str, ...]:
    spec = FoundationModelSpec.from_config(raw_spec)
    revision = spec.model_load_revision or "unpinned"
    details = [
        f"model: {spec.model_name} @ {revision}",
        f"pooling: {spec.pooling}",
        f"training: {spec.train_mode}",
    ]
    if spec.train_mode == "lora" and spec.lora is not None:
        lora = FoundationLoRASpec.from_config(spec.lora)
        targets = ", ".join(lora.target_modules)
        details.append(
            f"LoRA: rank {lora.rank}, alpha {lora.alpha:g}, targets [{targets}]"
        )
    return tuple(details)


def _component_description(
    name: str,
    component: Mapping[str, Any],
) -> ArchitectureComponent:
    architecture = str(component["architecture"])
    input_dimension = int(component["input_dim"])
    output_dimension = int(component["output_dim"])
    details: list[str] = []

    if architecture in {"MLP", "SPARSE"}:
        hidden = _expanded_nodes(
            component["nodes_per_layer"],
            component.get("num_layers"),
        )
        details.append(
            "layers: " + _dimensions_text((input_dimension, *hidden, output_dimension))
        )
        if architecture == "SPARSE":
            details.append("sparse first projection")
    elif architecture == "EMBEDDING":
        details.append(f"{input_dimension:,} IDs → {output_dimension:,}")
    elif architecture == "GRAPH":
        graph_kind = "GINE" if component["use_edge_features"] else "GIN"
        details.extend(
            (
                f"{graph_kind}: {component['num_layers']} message-passing layers",
                f"{component['pooling']} pooling",
            )
        )
    elif architecture == "SEQUENCE":
        encoder = str(component["encoder"])
        details.append(
            f"{encoder}: vocabulary {input_dimension:,}, output {output_dimension:,}"
        )
        if encoder == "foundation":
            details.extend(_foundation_details(component["foundation_model"]))
        else:
            details.append(
                f"embedding {component['embedding_dim']:,}, "
                f"{component['num_layers']} layer(s)"
            )
            if encoder == "transformer":
                details.append(f"{component['transformer_num_heads']} attention heads")
            elif encoder == "conv1d":
                details.append(f"kernel size {component['conv_kernel_size']}")

    return ArchitectureComponent(
        name=name,
        architecture=architecture,
        input_dimension=input_dimension,
        output_dimension=output_dimension,
        optional=bool(component["optional"]),
        details=tuple(details),
    )


def _branch_output_dimension(config: DeepMTPConfig, branch: str) -> int:
    architecture = getattr(config, f"{branch}_branch_architecture")
    if architecture == "COMPOSITE":
        components = getattr(config, f"{branch}_branch_composite_components")
        if components is None:  # pragma: no cover - validated invariant
            raise RuntimeError(f"{branch} composite branch has no components.")
        return sum(int(component["output_dim"]) for component in components.values())
    if config.general_architecture_version == "dot_product":
        return config.embedding_size
    if architecture == "EMBEDDING":
        return config.embedding_size
    if architecture == "SEQUENCE":
        return int(getattr(config, f"{branch}_branch_sequence_output_dim"))
    if architecture == "GRAPH":
        return int(getattr(config, f"{branch}_branch_graph_output_dim"))
    nodes = getattr(config, f"{branch}_branch_nodes_per_layer")
    expanded = _expanded_nodes(nodes, getattr(config, f"{branch}_branch_layers"))
    return expanded[-1]


def _branch_description(config: DeepMTPConfig, branch: str) -> ArchitectureBranch:
    architecture = getattr(config, f"{branch}_branch_architecture")
    output_dimension = _branch_output_dimension(config, branch)
    input_dimension = getattr(config, f"{branch}_branch_input_dim")
    details: list[str] = []
    components: tuple[ArchitectureComponent, ...] = ()

    if architecture in {"MLP", "SPARSE", "TABULAR", "CUSTOM"}:
        hidden = _expanded_nodes(
            getattr(config, f"{branch}_branch_nodes_per_layer"),
            getattr(config, f"{branch}_branch_layers"),
        )
        if input_dimension is not None:
            details.append(
                "layers: "
                + _dimensions_text((input_dimension, *hidden, output_dimension))
            )
        if architecture == "SPARSE":
            details.append("sparse first projection")
        elif architecture == "TABULAR":
            schema = getattr(config, f"{branch}_branch_tabular_schema")
            if isinstance(schema, Mapping):
                numeric = len(schema.get("numeric_columns", ()))
                categorical = len(schema.get("categorical_columns", ()))
                details.append(
                    f"features: {numeric} numeric, {categorical} categorical"
                )
        elif architecture == "CUSTOM":
            details.append("output contract verified when the model is built")
    elif architecture == "CONV":
        conv_name = getattr(config, f"{branch}_branch_conv_architecture")
        conv_version = getattr(config, f"{branch}_branch_conv_architecture_version")
        pretrained = getattr(config, f"{branch}_branch_conv_pretrained")
        details.extend(
            (
                f"backbone: {conv_name}/{conv_version}",
                "pretrained" if pretrained else "random initialization",
                "train from: "
                + getattr(
                    config,
                    f"{branch}_branch_conv_architecture_last_layer_trained",
                ),
            )
        )
    elif architecture == "EMBEDDING":
        if input_dimension is not None:
            details.append(f"{input_dimension:,} IDs → {output_dimension:,}")
    elif architecture == "GRAPH":
        uses_edges = getattr(config, f"{branch}_branch_graph_use_edge_features")
        graph_kind = "GINE" if uses_edges else "GIN"
        details.extend(
            (
                f"{graph_kind}: "
                f"{getattr(config, f'{branch}_branch_graph_num_layers')} "
                "message-passing layers",
                f"{getattr(config, f'{branch}_branch_graph_pooling')} pooling",
                f"hidden width: "
                f"{getattr(config, f'{branch}_branch_graph_hidden_dim'):,}",
            )
        )
    elif architecture == "SEQUENCE":
        encoder = getattr(config, f"{branch}_branch_sequence_encoder")
        details.append(f"encoder: {encoder}")
        if input_dimension is not None:
            details.append(f"vocabulary: {input_dimension:,}")
        if encoder == "foundation":
            details.extend(
                _foundation_details(
                    getattr(config, f"{branch}_branch_foundation_model")
                )
            )
        else:
            details.append(
                f"embedding: "
                f"{getattr(config, f'{branch}_branch_sequence_embedding_dim'):,}"
            )
            details.append(
                f"layers: {getattr(config, f'{branch}_branch_sequence_num_layers')}"
            )
            if encoder == "transformer":
                details.append(
                    f"attention heads: "
                    f"{getattr(config, f'{branch}_branch_sequence_transformer_num_heads')}"
                )
            elif encoder == "conv1d":
                details.append(
                    f"kernel size: "
                    f"{getattr(config, f'{branch}_branch_sequence_conv_kernel_size')}"
                )
    elif architecture == "COMPOSITE":
        raw_components = getattr(
            config,
            f"{branch}_branch_composite_components",
        )
        if raw_components is None:  # pragma: no cover - validated invariant
            raise RuntimeError(f"{branch} composite branch has no components.")
        components = tuple(
            _component_description(name, component)
            for name, component in raw_components.items()
        )
        fusion = getattr(config, f"{branch}_branch_composite_fusion")
        details.append(f"component fusion: {fusion}")
        if fusion == "attention":
            details.extend(
                (
                    f"shared attention width: "
                    f"{getattr(config, f'{branch}_branch_composite_attention_dim'):,}",
                    f"attention heads: "
                    f"{getattr(config, f'{branch}_branch_composite_attention_num_heads')}",
                )
            )
        dropout = getattr(config, f"{branch}_branch_composite_modality_dropout")
        if dropout:
            details.append(f"modality dropout: {dropout:g}")

    return ArchitectureBranch(
        name=branch,  # type: ignore[arg-type]
        architecture=architecture,
        input_dimension=input_dimension,
        output_dimension=output_dimension,
        details=tuple(details),
        components=components,
    )


def _fusion_description(
    config: DeepMTPConfig,
    instance_width: int,
    target_width: int,
) -> ArchitectureFusion:
    architecture = config.general_architecture_version
    details: list[str] = []
    if architecture == "dot_product":
        input_dimension = instance_width
        details.append(f"elementwise product and sum at width {instance_width:,}")
        if config.output_dimension > 1:
            details.append(
                f"class projection: {instance_width:,} → {config.output_dimension:,}"
            )
    elif architecture == "mlp":
        input_dimension = instance_width + target_width
        hidden = _expanded_nodes(
            config.comb_mlp_nodes_per_layer, config.comb_mlp_layers
        )
        details.append(
            "layers: "
            + _dimensions_text((input_dimension, *hidden, config.output_dimension))
        )
    else:
        input_dimension = instance_width * target_width
        details.append(
            f"Kronecker width: {instance_width:,} × {target_width:,} "
            f"= {input_dimension:,}"
        )
        details.append(f"linear projection → {config.output_dimension:,}")
    return ArchitectureFusion(
        architecture=architecture,
        input_dimension=input_dimension,
        output_dimension=config.output_dimension,
        details=tuple(details),
    )


def _parameter_statistics(model: object) -> ParameterStatistics:
    if isinstance(model, nn.Module):
        module = model
    else:
        candidate = getattr(model, "model", None)
        if not isinstance(candidate, nn.Module):
            candidate = getattr(model, "deepMTP_model", None)
        if not isinstance(candidate, nn.Module):
            raise TypeError(
                "model must be a torch.nn.Module, ModelBundle, or DeepMTP trainer."
            )
        module = candidate
    parameters = tuple(module.parameters())
    total = sum(parameter.numel() for parameter in parameters)
    trainable = sum(
        parameter.numel() for parameter in parameters if parameter.requires_grad
    )
    return ParameterStatistics(
        total=total,
        trainable=trainable,
        frozen=total - trainable,
    )


def _task_label(config: DeepMTPConfig) -> str:
    if config.problem_mode == "regression":
        return "regression"
    if config.classification_mode == "multiclass":
        return f"multiclass classification ({config.output_dimension} classes)"
    return "binary classification"


def describe_architecture(
    config: DeepMTPConfig | Mapping[str, Any],
    *,
    model: object | None = None,
) -> ArchitectureDescription:
    """Describe a validated configuration without instantiating its model."""
    normalized = DeepMTPConfig.from_mapping(config)
    instance_branch = _branch_description(normalized, "instance")
    target_branch = _branch_description(normalized, "target")
    warnings: list[str] = []
    if (
        normalized.general_architecture_version == "dot_product"
        and instance_branch.output_dimension != target_branch.output_dimension
    ):
        warnings.append(
            "Dot-product fusion requires equal branch widths, but the configured "
            f"branches produce {instance_branch.output_dimension} and "
            f"{target_branch.output_dimension}."
        )
    return ArchitectureDescription(
        validation_setting=str(normalized.validation_setting),
        task=_task_label(normalized),
        loss=str(normalized.loss),
        instance_branch=instance_branch,
        target_branch=target_branch,
        fusion=_fusion_description(
            normalized,
            instance_branch.output_dimension,
            target_branch.output_dimension,
        ),
        parameters=_parameter_statistics(model) if model is not None else None,
        warnings=tuple(warnings),
    )


def _plain_branch_lines(branch: ArchitectureBranch) -> list[str]:
    lines = [
        f"{branch.name.title()} branch [{branch.architecture}] "
        f"→ {branch.output_dimension:,}"
    ]
    for detail in branch.details:
        lines.append(f"  · {detail}")
    for component in branch.components:
        availability = "optional" if component.optional else "required"
        lines.append(
            f"  ├─ {component.name} [{component.architecture}] "
            f"→ {component.output_dimension:,} · {availability}"
        )
        for detail in component.details:
            lines.append(f"  │  · {detail}")
    return lines


def _format_plain(description: ArchitectureDescription) -> str:
    lines = [
        "DeepMTP architecture",
        f"Task: {description.task} · validation {description.validation_setting}",
        f"Loss: {description.loss}",
        "",
        *_plain_branch_lines(description.instance_branch),
        "",
        *_plain_branch_lines(description.target_branch),
        "",
        f"Pair fusion [{description.fusion.architecture}] "
        f"→ {description.fusion.output_dimension:,}",
    ]
    lines.extend(f"  · {detail}" for detail in description.fusion.details)
    if description.parameters is not None:
        statistics = description.parameters
        lines.extend(
            (
                "",
                "Parameters",
                f"  · total: {statistics.total:,}",
                f"  · trainable: {statistics.trainable:,} "
                f"({statistics.trainable_fraction:.1%})",
                f"  · frozen: {statistics.frozen:,}",
            )
        )
    if description.warnings:
        lines.extend(("", "Warnings"))
        lines.extend(f"  ! {warning}" for warning in description.warnings)
    return "\n".join(lines) + "\n"


def _require_rich() -> None:
    if importlib.util.find_spec("rich") is None:
        raise ModuleNotFoundError(
            "Rich architecture rendering requires the optional visualization "
            "dependencies. Install them with `pip install 'DeepMTP[visualization]'`."
        )


def _rich_branch_tree(branch: ArchitectureBranch, color: str) -> Any:
    from rich.markup import escape
    from rich.tree import Tree

    tree = Tree(
        f"[bold {color}]{escape(branch.architecture)}[/] "
        f"[bold]→ {branch.output_dimension:,}[/]",
        guide_style=color,
    )
    for detail in branch.details:
        tree.add(_rich_detail(detail))
    for component in branch.components:
        state = "[yellow]optional[/]" if component.optional else "[green]required[/]"
        component_tree = tree.add(
            f"[bold]{escape(component.name)}[/] "
            f"[{color}]{escape(component.architecture)}[/] "
            f"→ {component.output_dimension:,} · {state}"
        )
        for detail in component.details:
            component_tree.add(_rich_detail(detail))
    return tree


def _rich_detail(detail: str) -> str:
    from rich.markup import escape

    escaped = escape(detail)
    if detail.startswith("training: lora") or detail.startswith("LoRA:"):
        return f"[green]{escaped}[/]"
    if detail.startswith("training: full"):
        return f"[bold green]{escaped}[/]"
    if detail.startswith("training: frozen"):
        return f"[dim]{escaped}[/]"
    if "@ unpinned" in detail:
        return f"[yellow]{escaped}[/]"
    return f"[dim]{escaped}[/]"


def _parameter_bar(statistics: ParameterStatistics) -> str:
    width = 24
    filled = round(statistics.trainable_fraction * width)
    return "█" * filled + "░" * (width - filled)


def _rich_renderable(description: ArchitectureDescription, width: int) -> Any:
    from rich import box
    from rich.align import Align
    from rich.console import Group
    from rich.markup import escape
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text

    metadata = Table.grid(expand=True)
    metadata.add_column(justify="left", ratio=1)
    metadata.add_column(justify="center", width=1)
    metadata.add_column(justify="center", ratio=1)
    metadata.add_column(justify="center", width=1)
    metadata.add_column(justify="right", ratio=1)
    metadata.add_row(
        f"[bold]Task[/] {escape(description.task)}",
        "[dim]│[/]",
        f"[bold]Validation[/] {escape(description.validation_setting)}",
        "[dim]│[/]",
        f"[bold]Loss[/] {escape(description.loss)}",
    )
    header = Panel(
        metadata,
        title="[bold bright_white]DeepMTP Architecture[/]",
        border_style="bright_blue",
        box=box.ROUNDED,
    )

    instance_panel = Panel(
        _rich_branch_tree(description.instance_branch, "cyan"),
        title="[bold cyan]INSTANCE BRANCH[/]",
        border_style="cyan",
        box=box.ROUNDED,
    )
    target_panel = Panel(
        _rich_branch_tree(description.target_branch, "magenta"),
        title="[bold magenta]TARGET BRANCH[/]",
        border_style="magenta",
        box=box.ROUNDED,
    )
    if width >= 100:
        branches = Table.grid(expand=True, padding=(0, 1))
        branches.add_column(ratio=1)
        branches.add_column(ratio=1)
        branches.add_row(instance_panel, target_panel)
    else:
        branches = Group(instance_panel, target_panel)

    fusion_body = Table.grid(expand=True)
    fusion_body.add_column(justify="center")
    fusion_body.add_row(
        f"[bold orange3]{escape(description.fusion.architecture.upper())}[/] "
        f"[bold]→ {description.fusion.output_dimension:,} output(s)[/]"
    )
    for detail in description.fusion.details:
        fusion_body.add_row(f"[dim]{escape(detail)}[/]")
    fusion_panel = Align.center(
        Panel(
            fusion_body,
            title="[bold orange3]PAIR FUSION[/]",
            border_style="orange3",
            box=box.ROUNDED,
            width=min(76, max(42, width - 4)),
        )
    )

    sections: list[Any] = [
        header,
        branches,
        Align.center(Text("╰───────────────┬───────────────╯", style="dim")),
        fusion_panel,
    ]
    if description.parameters is not None:
        statistics = description.parameters
        parameter_table = Table.grid(expand=True)
        parameter_table.add_column(justify="left")
        parameter_table.add_column(justify="center")
        parameter_table.add_column(justify="right")
        parameter_table.add_row(
            f"[green]Trainable[/] {statistics.trainable:,} "
            f"({statistics.trainable_fraction:.1%})",
            f"[green]{_parameter_bar(statistics)}[/]",
            f"[dim]Frozen {statistics.frozen:,}[/] · Total {statistics.total:,}",
        )
        sections.append(
            Panel(
                parameter_table,
                title="[bold green]PARAMETERS[/]",
                border_style="green",
                box=box.ROUNDED,
            )
        )
    if description.warnings:
        warning_text = "\n".join(
            f"• {escape(warning)}" for warning in description.warnings
        )
        sections.append(
            Panel(
                warning_text,
                title="[bold red]CONFIGURATION WARNINGS[/]",
                border_style="red",
                box=box.ROUNDED,
            )
        )
    return Group(*sections)


def format_architecture(
    config: DeepMTPConfig | Mapping[str, Any] | ArchitectureDescription,
    *,
    model: object | None = None,
    style: Literal["plain", "rich"] = "plain",
    width: int = 120,
    color: bool = False,
) -> str:
    """Render an architecture to a string without writing to the terminal."""
    _validate_width(width)
    description = (
        config
        if isinstance(config, ArchitectureDescription)
        else describe_architecture(config, model=model)
    )
    if style == "plain":
        return _format_plain(description)
    if style != "rich":
        raise ValueError("style must be 'plain' or 'rich'.")
    _require_rich()
    from rich.console import Console

    output = StringIO()
    console = Console(
        file=output,
        width=width,
        color_system="truecolor" if color else None,
        force_terminal=color,
    )
    console.print(_rich_renderable(description, width))
    return output.getvalue()


def print_architecture(
    config: DeepMTPConfig | Mapping[str, Any] | ArchitectureDescription,
    *,
    model: object | None = None,
    style: RenderStyle = "auto",
    width: int | None = None,
    file: TextIO | None = None,
) -> ArchitectureDescription:
    """Print a responsive architecture diagram and return its description."""
    if width is not None:
        _validate_width(width)
    description = (
        config
        if isinstance(config, ArchitectureDescription)
        else describe_architecture(config, model=model)
    )
    if style not in {"auto", "plain", "rich"}:
        raise ValueError("style must be 'auto', 'plain', or 'rich'.")

    rich_available = importlib.util.find_spec("rich") is not None
    if style == "rich" and not rich_available:
        _require_rich()
    if style == "plain" or not rich_available:
        (file or sys.stdout).write(_format_plain(description))
        return description

    from rich.console import Console

    console = Console(file=file, width=width)
    if style == "auto" and not (console.is_terminal or console.is_jupyter):
        (file or sys.stdout).write(_format_plain(description))
        return description
    render_width = width or console.width
    console.print(_rich_renderable(description, render_width))
    return description


def export_architecture_svg(
    config: DeepMTPConfig | Mapping[str, Any] | ArchitectureDescription,
    destination: str | PathLike[str],
    *,
    model: object | None = None,
    width: int = 140,
) -> Path:
    """Export a Rich architecture rendering as an SVG file."""
    _validate_width(width)
    _require_rich()
    from rich.console import Console

    description = (
        config
        if isinstance(config, ArchitectureDescription)
        else describe_architecture(config, model=model)
    )
    destination_path = Path(destination)
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    console = Console(
        file=StringIO(),
        record=True,
        width=width,
        force_terminal=True,
        color_system="truecolor",
    )
    console.print(_rich_renderable(description, width))
    console.save_svg(str(destination_path), title="DeepMTP Architecture")
    return destination_path
