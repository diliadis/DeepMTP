import pytest

from DeepMTP._registry import FactoryRegistry


def test_factory_registry_registers_and_resolves_factories():
    registry = FactoryRegistry("example")

    def factory():
        return "value"

    registry.register("first", factory)

    assert registry.names == ("first",)
    assert registry.resolve("first") is factory
    assert "first" in registry
    assert list(registry) == ["first"]
    assert len(registry) == 1


def test_factory_registry_requires_explicit_replacement():
    registry = FactoryRegistry("example", {"first": lambda: "original"})

    def replacement():
        return "replacement"

    with pytest.raises(ValueError, match="already registered"):
        registry.register("first", replacement)

    registry.register("first", replacement, replace=True)

    assert registry.resolve("first") is replacement


@pytest.mark.parametrize("name", ["", "   ", None])
def test_factory_registry_rejects_invalid_names(name):
    registry = FactoryRegistry("example")

    with pytest.raises(ValueError, match="non-empty string"):
        registry.register(name, lambda: None)


def test_factory_registry_rejects_non_callable_factories():
    registry = FactoryRegistry("example")

    with pytest.raises(TypeError, match="factory must be callable"):
        registry.register("invalid", object())
