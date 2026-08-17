"""Logarithmic barriers for componentwise box constraints."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from boxilqr.options import ScalarOrArray
from boxilqr.problem import Array, Bounds


class InfeasiblePointError(ValueError):
    """Raised when a point is outside the strict barrier domain."""

    def __init__(
        self,
        variable_name: str,
        index: int,
        side: str,
        margin: float,
        minimum_margin: float,
    ) -> None:
        self.variable_name = variable_name
        self.index = index
        self.side = side
        self.margin = margin
        self.minimum_margin = minimum_margin

        message = (
            f"{variable_name}[{index}] violates the strict {side} "
            f"barrier margin. Received margin {margin:.6e}; "
            f"required margin greater than {minimum_margin:.6e}."
        )

        super().__init__(message)


@dataclass(frozen=True)
class BarrierEvaluation:
    """Value and derivatives of a componentwise box barrier."""

    value: float
    gradient: Array
    hessian_diagonal: Array

    @property
    def hessian(self) -> Array:
        """Return the full diagonal barrier Hessian."""

        return np.diag(self.hessian_diagonal)


def expand_barrier_parameters(
    parameters: ScalarOrArray,
    dimension: int,
    name: str = "barrier_parameters",
) -> Array:
    """Convert scalar or vector barrier parameters into a vector."""

    array = np.asarray(parameters, dtype=float)

    if array.ndim == 0:
        expanded = np.full(
            dimension,
            float(array),
            dtype=float,
        )
    elif array.shape == (dimension,):
        expanded = array.copy()
    else:
        raise ValueError(
            f"{name} must be a scalar or have shape ({dimension},), "
            f"but received shape {array.shape}."
        )

    if not np.all(np.isfinite(expanded)):
        raise ValueError(f"{name} must contain only finite values.")

    if np.any(expanded <= 0.0):
        raise ValueError(f"{name} must be strictly positive.")

    return expanded


def evaluate_box_barrier(
    value: Array,
    bounds: Bounds,
    parameters: ScalarOrArray,
    *,
    minimum_margin: float = 1.0e-12,
    variable_name: str = "value",
) -> BarrierEvaluation:
    """Evaluate a logarithmic barrier and its first two derivatives.

    The barrier for component ``i`` is

        -mu_i log(value_i - lower_i)

    when the lower bound is finite, plus

        -mu_i log(upper_i - value_i)

    when the upper bound is finite.

    Unbounded sides make no contribution.
    """

    value = np.asarray(value, dtype=float)

    if value.shape != (bounds.dimension,):
        raise ValueError(
            f"{variable_name} must have shape ({bounds.dimension},), "
            f"but received {value.shape}."
        )

    if not np.all(np.isfinite(value)):
        raise ValueError(
            f"{variable_name} must contain only finite values."
        )

    minimum_margin = float(minimum_margin)

    if not np.isfinite(minimum_margin):
        raise ValueError("minimum_margin must be finite.")

    if minimum_margin < 0.0:
        raise ValueError("minimum_margin cannot be negative.")

    barrier_parameters = expand_barrier_parameters(
        parameters,
        bounds.dimension,
    )

    lower_is_finite = bounds.lower_is_finite
    upper_is_finite = bounds.upper_is_finite

    two_sided = lower_is_finite & upper_is_finite
    widths = bounds.upper[two_sided] - bounds.lower[two_sided]

    if np.any(widths <= 2.0 * minimum_margin):
        raise ValueError(
            "minimum_margin is too large for at least one "
            "two-sided constraint."
        )

    lower_margin = value - bounds.lower
    upper_margin = bounds.upper - value

    for index in np.flatnonzero(lower_is_finite):
        if lower_margin[index] <= minimum_margin:
            raise InfeasiblePointError(
                variable_name=variable_name,
                index=int(index),
                side="lower",
                margin=float(lower_margin[index]),
                minimum_margin=minimum_margin,
            )

    for index in np.flatnonzero(upper_is_finite):
        if upper_margin[index] <= minimum_margin:
            raise InfeasiblePointError(
                variable_name=variable_name,
                index=int(index),
                side="upper",
                margin=float(upper_margin[index]),
                minimum_margin=minimum_margin,
            )

    barrier_value = 0.0
    gradient = np.zeros(bounds.dimension, dtype=float)
    hessian_diagonal = np.zeros(bounds.dimension, dtype=float)

    lower_indices = np.flatnonzero(lower_is_finite)

    if lower_indices.size > 0:
        lower_distance = lower_margin[lower_indices]
        lower_parameters = barrier_parameters[lower_indices]

        barrier_value -= np.sum(
            lower_parameters * np.log(lower_distance)
        )

        gradient[lower_indices] -= (
            lower_parameters / lower_distance
        )

        hessian_diagonal[lower_indices] += (
            lower_parameters / lower_distance**2
        )

    upper_indices = np.flatnonzero(upper_is_finite)

    if upper_indices.size > 0:
        upper_distance = upper_margin[upper_indices]
        upper_parameters = barrier_parameters[upper_indices]

        barrier_value -= np.sum(
            upper_parameters * np.log(upper_distance)
        )

        gradient[upper_indices] += (
            upper_parameters / upper_distance
        )

        hessian_diagonal[upper_indices] += (
            upper_parameters / upper_distance**2
        )

    return BarrierEvaluation(
        value=float(barrier_value),
        gradient=gradient,
        hessian_diagonal=hessian_diagonal,
    )
