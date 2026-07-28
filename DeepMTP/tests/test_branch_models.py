from types import ModuleType

import pytest
import torch
from torch import nn

import DeepMTP.branch_models as branch_models
from DeepMTP.branch_models import MLP, ConvNet


class FakeResNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = nn.Linear(1, 1)
        self.layer1 = nn.Linear(1, 1)
        self.layer2 = nn.Linear(1, 1)
        self.layer3 = nn.Linear(1, 1)
        self.layer4 = nn.Linear(1, 1)
        self.avgpool = nn.Linear(1, 1)
        self.fc = nn.Linear(1, 1)


class FakeVGG(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Linear(1, 1)
        self.classifier = nn.Linear(1, 1)


def make_fake_resnet_models(version):
    models = ModuleType("fake_torchvision_models")
    setattr(models, version, lambda **_kwargs: FakeResNet())
    return models


test_MLP_data = [
    {
        "input_dim": 10,
        "output_dim": 15,
        "nodes_per_layer": 100,
        "num_layers": 2,
        "dropout": 0,
        "batch_norm": False,
    },
    {
        "input_dim": 10,
        "output_dim": 15,
        "nodes_per_layer": 123,
        "num_layers": 2,
        "dropout": 0,
        "batch_norm": False,
    },
    {
        "input_dim": 10,
        "output_dim": 15,
        "nodes_per_layer": 100,
        "num_layers": 2,
        "dropout": 0.5,
        "batch_norm": False,
    },
    {
        "input_dim": 10,
        "output_dim": 15,
        "nodes_per_layer": 123,
        "num_layers": 2,
        "dropout": 0,
        "batch_norm": True,
    },
    {
        "input_dim": 10,
        "output_dim": 15,
        "nodes_per_layer": 123,
        "num_layers": 2,
        "dropout": 0.5,
        "batch_norm": True,
    },
    {
        "input_dim": 10,
        "output_dim": 15,
        "nodes_per_layer": [100, 200, 300],
        "num_layers": 2,
        "dropout": 0,
        "batch_norm": False,
    },
]


@pytest.mark.parametrize("test_MLP_data", test_MLP_data)
def test_MLP(test_MLP_data):
    calculated_nodes_per_layer = []

    model = MLP(
        {},
        input_dim=test_MLP_data["input_dim"],
        output_dim=test_MLP_data["output_dim"],
        nodes_per_layer=test_MLP_data["nodes_per_layer"],
        num_layers=test_MLP_data["num_layers"],
        dropout_rate=test_MLP_data["dropout"],
        batch_norm=test_MLP_data["batch_norm"],
    )
    linear_layers = [layer for layer in model[0] if isinstance(layer, nn.Linear)]

    calculated_nodes_per_layer.append(test_MLP_data["input_dim"])
    if not isinstance(test_MLP_data["nodes_per_layer"], list):
        calculated_nodes_per_layer.extend(
            [test_MLP_data["nodes_per_layer"]] * test_MLP_data["num_layers"]
        )
    else:
        calculated_nodes_per_layer.extend(test_MLP_data["nodes_per_layer"])
    calculated_nodes_per_layer.append(test_MLP_data["output_dim"])

    for idx, layer in enumerate(linear_layers):
        assert layer.in_features == calculated_nodes_per_layer[idx]
        assert layer.out_features == calculated_nodes_per_layer[idx + 1]


def test_convnet_explains_how_to_install_image_dependencies(monkeypatch):
    def missing_torchvision(_module_name):
        raise ImportError("torchvision is not installed")

    monkeypatch.setattr(branch_models.importlib, "import_module", missing_torchvision)

    with pytest.raises(ModuleNotFoundError, match=r"DeepMTP\[image\]"):
        ConvNet(
            config={},
            input_dim=10,
            output_dim=15,
            conv_architecture="resnet",
            conv_architecture_version="resnet18",
            conv_architecture_last_trained_layer="last",
            conv_architecture_dense_layers=1,
        )


@pytest.mark.parametrize(
    ("architecture", "version", "factory_name", "weights_name"),
    [
        ("resnet", "resnet18", "resnet18", "ResNet18_Weights"),
        ("resnet", "resnet101", "resnet101", "ResNet101_Weights"),
        ("VGG", "unused", "vgg11_bn", "VGG11_BN_Weights"),
    ],
)
@pytest.mark.parametrize("pretrained", [False, True])
def test_torchvision_backbone_uses_explicit_weights_api(
    architecture,
    version,
    factory_name,
    weights_name,
    pretrained,
):
    models = ModuleType("fake_torchvision_models")
    backbone = nn.Identity()
    default_weights = object()
    calls = []

    class Weights:
        DEFAULT = default_weights

    def build_backbone(**kwargs):
        calls.append(kwargs)
        return backbone

    setattr(models, factory_name, build_backbone)
    if pretrained:
        setattr(models, weights_name, Weights)

    result = branch_models._build_torchvision_backbone(
        models,
        architecture=architecture,
        version=version,
        pretrained=pretrained,
    )

    assert result is backbone
    assert calls == [{"weights": default_weights if pretrained else None}]


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"input_dim": 0}, "input_dim"),
        ({"output_dim": 0}, "output_dim"),
        ({"nodes_per_layer": []}, "nodes_per_layer"),
        ({"nodes_per_layer": 0}, "nodes_per_layer"),
        ({"nodes_per_layer": 4, "num_layers": None}, "num_layers"),
        ({"dropout_rate": 1.0}, "dropout_rate"),
    ],
)
def test_mlp_rejects_invalid_dimensions_and_dropout(overrides, message):
    arguments = {
        "config": {},
        "input_dim": 2,
        "output_dim": 1,
        "nodes_per_layer": [4],
        "num_layers": 1,
        "dropout_rate": 0.0,
        "batch_norm": False,
    }
    arguments.update(overrides)

    with pytest.raises(ValueError, match=message):
        MLP(**arguments)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"conv_architecture": "alexnet"}, "Unknown name"),
        ({"conv_architecture_version": "resnet50"}, "supported resnet"),
        ({"conv_architecture_last_trained_layer": "classifier"}, "not a valid"),
        ({"conv_architecture_dense_layers": 3}, "Only 1 or 2"),
        ({"conv_architecture_dense_layers": True}, "Only 1 or 2"),
        ({"pretrained": "yes"}, "pretrained must be a boolean"),
    ],
)
def test_convnet_rejects_invalid_configuration_before_import(
    monkeypatch,
    overrides,
    message,
):
    def unexpected_import(_module_name):
        raise AssertionError("invalid configuration must not import torchvision")

    monkeypatch.setattr(branch_models.importlib, "import_module", unexpected_import)

    arguments = {
        "config": {},
        "input_dim": 10,
        "output_dim": 15,
        "conv_architecture": "resnet",
        "conv_architecture_version": "resnet18",
        "conv_architecture_last_trained_layer": "last",
        "conv_architecture_dense_layers": 1,
    }
    arguments.update(overrides)

    with pytest.raises(AttributeError, match=message):
        ConvNet(**arguments)


@pytest.mark.parametrize(
    ("earliest_trainable_layer", "trainable_backbone_modules"),
    [
        ("last", set()),
        ("layer4", {"avgpool", "layer4"}),
        ("layer3", {"avgpool", "layer4", "layer3"}),
        ("layer2", {"avgpool", "layer4", "layer3", "layer2"}),
        (
            "layer1",
            {"avgpool", "layer4", "layer3", "layer2", "layer1"},
        ),
    ],
)
def test_convnet_enables_expected_resnet_blocks_offline(
    monkeypatch,
    earliest_trainable_layer,
    trainable_backbone_modules,
):
    monkeypatch.setattr(
        branch_models,
        "_load_torchvision_models",
        lambda: make_fake_resnet_models("resnet18"),
    )

    model = ConvNet(
        config={},
        input_dim=10,
        output_dim=7,
        conv_architecture="resnet",
        conv_architecture_version="resnet18",
        conv_architecture_last_trained_layer=earliest_trainable_layer,
        conv_architecture_dense_layers=1,
        pretrained=False,
    )

    backbone = model.predictor[0]
    for module_name in ("stem", "layer1", "layer2", "layer3", "layer4", "avgpool"):
        parameters = list(getattr(backbone, module_name).parameters())
        assert parameters
        assert all(parameter.requires_grad for parameter in parameters) == (
            module_name in trainable_backbone_modules
        )
    assert all(parameter.requires_grad for parameter in backbone.fc.parameters())


@pytest.mark.parametrize(
    ("version", "dense_layers", "expected_dimensions"),
    [
        ("resnet18", 1, [(512, 7)]),
        ("resnet101", 2, [(2048, 2048), (2048, 7)]),
    ],
)
def test_convnet_builds_version_specific_classifier_offline(
    monkeypatch,
    version,
    dense_layers,
    expected_dimensions,
):
    monkeypatch.setattr(
        branch_models,
        "_load_torchvision_models",
        lambda: make_fake_resnet_models(version),
    )

    model = ConvNet(
        config={},
        input_dim=10,
        output_dim=7,
        conv_architecture="resnet",
        conv_architecture_version=version,
        conv_architecture_last_trained_layer="last",
        conv_architecture_dense_layers=dense_layers,
        pretrained=False,
    )

    linear_layers = [
        layer for layer in model.predictor[0].fc if isinstance(layer, nn.Linear)
    ]
    assert [
        (layer.in_features, layer.out_features) for layer in linear_layers
    ] == expected_dimensions


def test_convnet_builds_frozen_vgg_with_trainable_classifier_offline(monkeypatch):
    models = ModuleType("fake_torchvision_models")
    models.vgg11_bn = lambda **_kwargs: FakeVGG()
    monkeypatch.setattr(
        branch_models,
        "_load_torchvision_models",
        lambda: models,
    )

    with torch.device("meta"):
        model = ConvNet(
            config={},
            input_dim=10,
            output_dim=7,
            conv_architecture="VGG",
            conv_architecture_version="unused",
            conv_architecture_last_trained_layer="unused",
            conv_architecture_dense_layers=99,
            pretrained=False,
        )

    backbone = model.predictor[0]
    linear_layers = [
        layer for layer in backbone.classifier if isinstance(layer, nn.Linear)
    ]
    assert [(layer.in_features, layer.out_features) for layer in linear_layers] == [
        (25088, 4096),
        (4096, 4096),
        (4096, 7),
    ]
    assert not any(
        parameter.requires_grad for parameter in backbone.features.parameters()
    )
    assert all(
        parameter.requires_grad for parameter in backbone.classifier.parameters()
    )
