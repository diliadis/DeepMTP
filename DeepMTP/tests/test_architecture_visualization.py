import importlib.util
import json
from io import StringIO

import pytest
from torch import nn

from DeepMTP import (
    DeepMTPConfig,
    describe_architecture,
    export_architecture_svg,
    format_architecture,
    print_architecture,
)
from DeepMTP.cli import main as cli_main


def make_config(**overrides):
    values = {
        "validation_setting": "B",
        "problem_mode": "classification",
        "general_architecture_version": "dot_product",
        "instance_branch_architecture": "MLP",
        "instance_branch_input_dim": 10,
        "instance_branch_nodes_per_layer": [12],
        "target_branch_architecture": "MLP",
        "target_branch_input_dim": 4,
        "target_branch_nodes_per_layer": [12],
        "embedding_size": 8,
    }
    values.update(overrides)
    return DeepMTPConfig(**values)


def test_description_captures_two_branch_semantics_without_building_model():
    description = describe_architecture(make_config())

    assert description.validation_setting == "B"
    assert description.task == "binary classification"
    assert description.loss == "binary_cross_entropy_with_logits"
    assert description.instance_branch.architecture == "MLP"
    assert description.instance_branch.output_dimension == 8
    assert description.instance_branch.details == ("layers: 10 → 12 → 8",)
    assert description.target_branch.output_dimension == 8
    assert description.fusion.architecture == "dot_product"
    assert description.fusion.input_dimension == 8
    assert description.fusion.output_dimension == 1
    assert description.parameters is None
    assert description.warnings == ()


def test_description_expands_composite_foundation_and_attention_settings():
    config = make_config(
        general_architecture_version="mlp",
        instance_branch_architecture="COMPOSITE",
        instance_branch_composite_fusion="attention",
        instance_branch_composite_attention_dim=32,
        instance_branch_composite_attention_num_heads=4,
        instance_branch_composite_modality_dropout=0.2,
        instance_branch_composite_components={
            "descriptors": {
                "architecture": "MLP",
                "input_dim": 128,
                "output_dim": 16,
            },
            "protein": {
                "architecture": "SEQUENCE",
                "input_dim": 33,
                "output_dim": 16,
                "embedding_dim": 16,
                "optional": True,
                "encoder": "foundation",
                "foundation_model": {
                    "model_name": "example/protein-model",
                    "revision": "abc123",
                    "train_mode": "lora",
                    "lora": {
                        "rank": 8,
                        "alpha": 16,
                        "dropout": 0.1,
                        "target_modules": ["query", "value"],
                    },
                },
            },
        },
        target_branch_nodes_per_layer=[24],
        comb_mlp_nodes_per_layer=[32, 16],
    )

    description = describe_architecture(config)

    branch = description.instance_branch
    assert branch.output_dimension == 32
    assert branch.details == (
        "component fusion: attention",
        "shared attention width: 32",
        "attention heads: 4",
        "modality dropout: 0.2",
    )
    assert [component.name for component in branch.components] == [
        "descriptors",
        "protein",
    ]
    protein = branch.components[1]
    assert protein.optional is True
    assert "model: example/protein-model @ abc123" in protein.details
    assert "training: lora" in protein.details
    assert "LoRA: rank 8, alpha 16, targets [query, value]" in protein.details
    assert description.fusion.input_dimension == 56
    assert description.fusion.details == ("layers: 56 → 32 → 16 → 1",)


def test_description_shows_foundation_cache_storage():
    config = make_config(
        instance_branch_architecture="SEQUENCE",
        instance_branch_input_dim=33,
        instance_branch_sequence_encoder="foundation",
        instance_branch_sequence_output_dim=8,
        instance_branch_foundation_model={
            "model_name": "example/protein-model",
            "train_mode": "frozen",
            "cache": {
                "directory": "./foundation-cache",
                "memory_capacity": 128,
            },
        },
    )

    description = describe_architecture(config)

    assert "cache: memory + disk, 128 memory entries" in (
        description.instance_branch.details
    )


def test_dot_product_width_mismatch_is_visible_before_model_construction():
    config = make_config(
        instance_branch_architecture="COMPOSITE",
        instance_branch_composite_components={
            "dense": {
                "architecture": "MLP",
                "input_dim": 10,
                "output_dim": 5,
            },
            "sparse": {
                "architecture": "SPARSE",
                "input_dim": 100,
                "output_dim": 7,
            },
        },
    )

    description = describe_architecture(config)

    assert description.instance_branch.output_dimension == 12
    assert description.target_branch.output_dimension == 8
    assert description.warnings == (
        "Dot-product fusion requires equal branch widths, but the configured "
        "branches produce 12 and 8.",
    )
    assert "Warnings" in format_architecture(description)


def test_parameter_statistics_are_exact_for_an_existing_model():
    model = nn.Linear(2, 3)
    model.bias.requires_grad_(False)

    description = describe_architecture(make_config(), model=model)

    assert description.parameters is not None
    assert description.parameters.total == 9
    assert description.parameters.trainable == 6
    assert description.parameters.frozen == 3
    assert description.parameters.trainable_fraction == pytest.approx(2 / 3)


def test_parameter_statistics_accept_model_bundle_and_trainer_shapes():
    module = nn.Linear(2, 1)
    bundle = type("Bundle", (), {"model": module})()
    trainer = type("Trainer", (), {"deepMTP_model": module})()

    assert describe_architecture(make_config(), model=bundle).parameters is not None
    assert describe_architecture(make_config(), model=trainer).parameters is not None

    with pytest.raises(TypeError, match="ModelBundle, or DeepMTP trainer"):
        describe_architecture(make_config(), model=object())


def test_plain_and_rich_rendering_include_semantic_sections():
    description = describe_architecture(make_config())

    plain = format_architecture(description, style="plain")
    rich_wide = format_architecture(description, style="rich", width=120)
    rich_narrow = format_architecture(description, style="rich", width=72)

    assert "Instance branch [MLP] → 8" in plain
    assert "Pair fusion [dot_product] → 1" in plain
    for rendered in (rich_wide, rich_narrow):
        assert "DeepMTP Architecture" in rendered
        assert "INSTANCE BRANCH" in rendered
        assert "TARGET BRANCH" in rendered
        assert "PAIR FUSION" in rendered

    with pytest.raises(ValueError, match="greater than or equal to 40"):
        format_architecture(description, style="rich", width=39)


def test_auto_rendering_falls_back_to_plain_for_redirected_output():
    output = StringIO()

    returned = print_architecture(make_config(), file=output)

    assert returned.instance_branch.architecture == "MLP"
    assert output.getvalue().startswith("DeepMTP architecture\n")


def test_rich_rendering_has_an_actionable_missing_dependency_error(monkeypatch):
    original_find_spec = importlib.util.find_spec

    def find_spec(name):
        if name == "rich":
            return None
        return original_find_spec(name)

    monkeypatch.setattr("DeepMTP.architecture.importlib.util.find_spec", find_spec)

    with pytest.raises(ModuleNotFoundError, match=r"DeepMTP\[visualization\]"):
        format_architecture(make_config(), style="rich")


def test_svg_export_creates_parent_directories(tmp_path):
    destination = tmp_path / "nested" / "architecture.svg"

    returned = export_architecture_svg(make_config(), destination)

    assert returned == destination
    svg = destination.read_text(encoding="utf-8")
    assert '<svg class="rich-terminal"' in svg
    assert "Generated with Rich" in svg


def test_cli_supports_plain_json_module_and_svg_outputs(tmp_path, capsys):
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(make_config().to_dict(include_defaults=True)),
        encoding="utf-8",
    )

    assert cli_main(["architecture", str(config_path), "--format", "plain"]) == 0
    assert "Pair fusion [dot_product]" in capsys.readouterr().out

    assert cli_main(["architecture", str(config_path), "--format", "json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["instance_branch"]["architecture"] == "MLP"

    svg_path = tmp_path / "cli.svg"
    assert (
        cli_main(
            [
                "architecture",
                str(config_path),
                "--format",
                "plain",
                "--export-svg",
                str(svg_path),
            ]
        )
        == 0
    )
    capsys.readouterr()
    assert svg_path.is_file()


def test_cli_reports_invalid_configuration_files(tmp_path, capsys):
    config_path = tmp_path / "config.json"
    config_path.write_text("[]", encoding="utf-8")

    with pytest.raises(SystemExit, match="2"):
        cli_main(["architecture", str(config_path)])

    assert "configuration root must be a JSON object" in capsys.readouterr().err
