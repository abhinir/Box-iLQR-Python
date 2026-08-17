"""Configuration options for Box-iLQR."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TypeAlias

import numpy as np

from boxilqr.problem import Array


ScalarOrArray: TypeAlias = float | Array


def _validate_positive(
    value: ScalarOrArray,
    name: str,
) -> ScalarOrArray:
    """Validate a positive scalar or one-dimensional array."""

    array = np.asarray(value, dtype=float)

    if array.ndim > 1:
        raise ValueError(f"{name} must be a scalar or one-dimensional array.")

    if array.size == 0:
        raise ValueError(f"{name} cannot be empty.")

    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values.")

    if np.any(array <= 0.0):
        raise ValueError(f"{name} must be strictly positive.")

    if array.ndim == 0:
        return float(array)

    return array.copy()


def _validate_reduction_factor(
    value: ScalarOrArray,
    name: str,
) -> ScalarOrArray:
    """Validate a barrier-reduction factor in the interval (0, 1)."""

    array = np.asarray(value, dtype=float)

    if array.ndim > 1:
        raise ValueError(f"{name} must be a scalar or one-dimensional array.")

    if array.size == 0:
        raise ValueError(f"{name} cannot be empty.")

    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values.")

    if np.any(array <= 0.0) or np.any(array >= 1.0):
        raise ValueError(f"{name} must lie strictly between zero and one.")

    if array.ndim == 0:
        return float(array)

    return array.copy()


@dataclass(frozen=True)
class ILQROptions:
    """Options for the inner iLQR solver."""

    max_iterations: int = 200

    relative_cost_tolerance: float = 1.0e-7
    feedforward_tolerance: float = 1.0e-6

    max_line_search_steps: int = 15
    line_search_decay: float = 0.5
    acceptance_ratio: float = 1.0e-4

    regularization_initial: float = 1.0e-6
    regularization_minimum: float = 1.0e-9
    regularization_maximum: float = 1.0e12
    regularization_increase: float = 10.0
    regularization_decrease: float = 0.5

    def __post_init__(self) -> None:
        if self.max_iterations <= 0:
            raise ValueError("max_iterations must be positive.")

        if self.relative_cost_tolerance <= 0.0:
            raise ValueError("relative_cost_tolerance must be positive.")

        if self.feedforward_tolerance <= 0.0:
            raise ValueError("feedforward_tolerance must be positive.")

        if self.max_line_search_steps <= 0:
            raise ValueError("max_line_search_steps must be positive.")

        if not 0.0 < self.line_search_decay < 1.0:
            raise ValueError(
                "line_search_decay must lie strictly between zero and one."
            )

        if not 0.0 <= self.acceptance_ratio < 1.0:
            raise ValueError(
                "acceptance_ratio must lie in the interval [0, 1)."
            )

        if self.regularization_minimum <= 0.0:
            raise ValueError("regularization_minimum must be positive.")

        if self.regularization_initial <= 0.0:
            raise ValueError("regularization_initial must be positive.")

        if self.regularization_maximum <= 0.0:
            raise ValueError("regularization_maximum must be positive.")

        if not (
            self.regularization_minimum
            <= self.regularization_initial
            <= self.regularization_maximum
        ):
            raise ValueError(
                "The initial regularization must lie between its "
                "minimum and maximum values."
            )

        if self.regularization_increase <= 1.0:
            raise ValueError(
                "regularization_increase must be greater than one."
            )

        if not 0.0 < self.regularization_decrease < 1.0:
            raise ValueError(
                "regularization_decrease must lie strictly between "
                "zero and one."
            )


@dataclass(frozen=True)
class BarrierOptions:
    """Options for the outer barrier-continuation loop."""

    initial_state_barrier: ScalarOrArray = 1.0
    initial_control_barrier: ScalarOrArray = 1.0

    state_reduction_factor: ScalarOrArray = 0.2
    control_reduction_factor: ScalarOrArray = 0.2

    reduction_update_rate: float = 1.5
    tolerance: float = 1.0e-6

    max_outer_iterations: int = 100
    max_failed_reductions: int = 25

    minimum_margin: float = 1.0e-12

    def __post_init__(self) -> None:
        state_barrier = _validate_positive(
            self.initial_state_barrier,
            "initial_state_barrier",
        )

        control_barrier = _validate_positive(
            self.initial_control_barrier,
            "initial_control_barrier",
        )

        state_reduction = _validate_reduction_factor(
            self.state_reduction_factor,
            "state_reduction_factor",
        )

        control_reduction = _validate_reduction_factor(
            self.control_reduction_factor,
            "control_reduction_factor",
        )

        object.__setattr__(
            self,
            "initial_state_barrier",
            state_barrier,
        )
        object.__setattr__(
            self,
            "initial_control_barrier",
            control_barrier,
        )
        object.__setattr__(
            self,
            "state_reduction_factor",
            state_reduction,
        )
        object.__setattr__(
            self,
            "control_reduction_factor",
            control_reduction,
        )

        if self.reduction_update_rate <= 1.0:
            raise ValueError(
                "reduction_update_rate must be greater than one."
            )

        if self.tolerance <= 0.0:
            raise ValueError("Barrier tolerance must be positive.")

        if self.max_outer_iterations <= 0:
            raise ValueError("max_outer_iterations must be positive.")

        if self.max_failed_reductions <= 0:
            raise ValueError("max_failed_reductions must be positive.")

        if self.minimum_margin <= 0.0:
            raise ValueError("minimum_margin must be positive.")


@dataclass(frozen=True)
class BoxILQROptions:
    """Complete configuration for the Box-iLQR solver."""

    ilqr: ILQROptions = field(default_factory=ILQROptions)
    barrier: BarrierOptions = field(default_factory=BarrierOptions)

    verbose: bool = True
    store_iteration_history: bool = True
