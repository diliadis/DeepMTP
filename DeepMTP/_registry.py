"""Small typed factory registry shared by model and runtime components."""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping
from typing import Generic, TypeVar

FactoryT = TypeVar("FactoryT", bound=Callable[..., object])


class FactoryRegistry(Generic[FactoryT]):
    """Resolve named component factories with focused registration errors."""

    def __init__(
        self,
        component_name: str,
        factories: Mapping[str, FactoryT] | None = None,
    ) -> None:
        if not isinstance(component_name, str) or not component_name.strip():
            raise ValueError("component_name must be a non-empty string.")
        self.component_name = component_name
        self._factories: dict[str, FactoryT] = {}
        for name, factory in (factories or {}).items():
            self.register(name, factory)

    @property
    def names(self) -> tuple[str, ...]:
        """Return registered names in deterministic order."""
        return tuple(sorted(self._factories))

    def register(
        self,
        name: str,
        factory: FactoryT,
        *,
        replace: bool = False,
    ) -> None:
        """Register a named factory, rejecting accidental replacement."""
        if not isinstance(name, str) or not name.strip():
            raise ValueError(f"{self.component_name} name must be a non-empty string.")
        if not callable(factory):
            raise TypeError(f"{self.component_name} factory must be callable.")
        if name in self._factories and not replace:
            raise ValueError(
                f"{self.component_name} {name!r} is already registered. "
                "Pass replace=True to replace it explicitly."
            )
        self._factories[name] = factory

    def resolve(self, name: str) -> FactoryT:
        """Resolve one factory or report all available names."""
        try:
            return self._factories[name]
        except KeyError:
            available = ", ".join(repr(item) for item in self.names) or "<none>"
            raise KeyError(
                f"Unknown {self.component_name} {name!r}. Available: {available}."
            ) from None

    def __contains__(self, name: object) -> bool:
        return name in self._factories

    def __iter__(self) -> Iterator[str]:
        return iter(self.names)

    def __len__(self) -> int:
        return len(self._factories)
