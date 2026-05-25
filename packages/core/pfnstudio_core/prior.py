"""Prior abstraction.

A Prior is a synthetic data generator — the defining mechanic of a PFN.

Promptable priors (see :mod:`pfnstudio_core.axes`) extend this by
declaring *steerable axes* the trained brain learns to honor at
inference. Subclasses declare which axes they support via the
``axes`` class attribute and accept a ``tag`` argument in ``sample``.

Back-compat invariant: priors that don't declare ``axes`` ignore the
``tag`` argument entirely. Old call sites — ``prior.sample(seed=42)``
without a ``tag`` — keep working unchanged.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, ClassVar

from pydantic import BaseModel, Field


class PriorParameter(BaseModel):
    type: str
    range: list[float] | None = None
    choices: list[Any] | None = None
    default: Any = None
    description: str | None = None


class PriorOutputVariable(BaseModel):
    name: str
    type: str
    shape: str | None = None
    description: str | None = None


class PriorOutputs(BaseModel):
    variables: list[PriorOutputVariable]


class PriorAxisRef(BaseModel):
    """Reference to a registered axis from prior.yaml. The actual Axis
    instance is resolved at load time via the global axis registry."""

    name: str
    # Optional per-prior override of the axis's default unknown_mass.
    unknown_mass: float | None = None


class PriorSpec(BaseModel):
    """Typed view of prior.yaml. Validated against schemas/prior.schema.json by the CLI."""

    id: str
    name: str
    version: str
    kind: str
    description: str | None = None
    parameters: dict[str, PriorParameter] = Field(default_factory=dict)
    outputs: PriorOutputs
    citations: list[str] = Field(default_factory=list)
    implementation: str = "prior.py"
    # Promptable axes declared by this prior. Empty list (default) =
    # non-promptable; sample() ignores any tag argument.
    axes: list[PriorAxisRef] = Field(default_factory=list)


class Prior(ABC):
    """Subclass and decorate with @register_prior("<id>") to bind to a PriorSpec.

    To opt into promptable priors:
    1. List supported axis names in the class attribute ``axes``.
    2. Accept ``tag: dict[str, Any] | None = None`` in ``sample()``.
    3. When tag is not None, honor each axis whose value is not UNKNOWN.

    Priors that don't override ``axes`` get unchanged behavior — the
    ``tag`` argument is accepted but ignored by the base class.
    """

    spec: PriorSpec
    # Axis names this prior supports. Empty by default — non-promptable.
    # Listed names must be registered with ``register_axis`` before
    # ``sample()`` is called.
    axes: ClassVar[list[str]] = []

    @abstractmethod
    def sample(
        self,
        *,
        seed: int,
        tag: dict[str, Any] | None = None,
        **params: Any,
    ) -> dict[str, Any]:
        """Return one sample as a dict keyed by ``outputs.variables[*].name``.

        ``tag`` is an optional dict mapping axis name → value. Values
        of :data:`axes.UNKNOWN` (or axis names not in the tag at all)
        mean "skip this axis." Subclasses that don't declare axes
        should accept and ignore this argument.
        """

    def sample_batch(
        self,
        *,
        batch_size: int,
        seed: int,
        tag: dict[str, Any] | None = None,
        **params: Any,
    ) -> list[dict[str, Any]]:
        return [self.sample(seed=seed + i, tag=tag, **params) for i in range(batch_size)]
