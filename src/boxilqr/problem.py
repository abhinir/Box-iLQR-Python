"""Problem definitions for Box-iLQR."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, TypeAlias

import numpy as np
from numpy.typing import NDArray


Array: TypeAlias = NDArray[np.float64]

Dynamics: TypeAlias = Callable[[Array, Array, int], Array]
RunningCost: TypeAlias = Callable[[Array, Array, int], float]
TerminalCost: TypeAlias = Callable[[Array], float]


@dataclass(frozen=True)
class Bounds:
    """Componentwise lower and upper bounds."""

    lower: Array
    upper: Array

    def __post_init__(self) -> None:
        lower = np.asarray(self.lower, dtype=float)
        upper = np.asarray(self.upper, dtype=float)

        if lower.ndim != 1 or upper.ndim != 1:
            raise ValueError("Bounds must be one-dimensional arrays.")

        if lower.shape != upper.shape:
            raise ValueError(
                "Lower and upper bounds must have the same shape."
            )

        if lower.size == 0:
            raise ValueError("Bounds cannot be empty.")

        if np.any(np.isnan(lower)) or np.any(np.isnan(upper)):
            raise ValueError("Bounds cannot contain NaN values.")

        if np.any(lower >= upper):
            raise ValueError(
                "Every lower bound must be strictly less than its upper bound."
            )

        object.__setattr__(self, "lower", lower.copy())
        object.__setattr__(self, "upper", upper.copy())

    @property
    def dimension(self) -> int:
        """Number of bounded components."""

        return self.lower.size

    @property
    def lower_is_finite(self) -> NDArray[np.bool_]:
        """Components with finite lower bounds."""

        return np.isfinite(self.lower)

    @property
    def upper_is_finite(self) -> NDArray[np.bool_]:
        """Components with finite upper bounds."""

        return np.isfinite(self.upper)

    @property
    def has_constraints(self) -> bool:
        """Whether at least one finite constraint is present."""

        return bool(
            np.any(self.lower_is_finite) or np.any(self.upper_is_finite)
        )

    def contains(self, value: Array, *, strict: bool = True) -> bool:
        """Check whether a vector satisfies the bounds."""

        value = np.asarray(value, dtype=float)

        if value.shape != (self.dimension,):
            raise ValueError(
                f"Expected a vector with shape ({self.dimension},), "
                f"but received {value.shape}."
            )

        if not np.all(np.isfinite(value)):
            return False

        if strict:
            lower_satisfied = value > self.lower
            upper_satisfied = value < self.upper
        else:
            lower_satisfied = value >= self.lower
            upper_satisfied = value <= self.upper

        return bool(np.all(lower_satisfied & upper_satisfied))

    def project(self, value: Array) -> Array:
        """Project a vector onto the closed box."""

        value = np.asarray(value, dtype=float)

        if value.shape != (self.dimension,):
            raise ValueError(
                f"Expected a vector with shape ({self.dimension},), "
                f"but received {value.shape}."
            )

        return np.clip(value, self.lower, self.upper)


BoundsInput: TypeAlias = Bounds | tuple[Array, Array] | None


def _prepare_bounds(
    bounds: BoundsInput,
    expected_dimension: int,
    name: str,
) -> Bounds | None:
    """Convert a bounds tuple into a validated Bounds object."""

    if bounds is None:
        return None

    if isinstance(bounds, Bounds):
        prepared = bounds
    else:
        if not isinstance(bounds, tuple) or len(bounds) != 2:
            raise TypeError(
                f"{name} must be a Bounds object, a "
                "(lower, upper) tuple, or None."
            )

        prepared = Bounds(
            lower=bounds[0],
            upper=bounds[1],
        )

    if prepared.dimension != expected_dimension:
        raise ValueError(
            f"{name} has dimension {prepared.dimension}, but the "
            f"expected dimension is {expected_dimension}."
        )

    return prepared


@dataclass(frozen=True)
class BoxILQRProblem:
    """Definition of a finite-horizon optimal-control problem."""

    dynamics: Dynamics
    running_cost: RunningCost
    terminal_cost: TerminalCost
    horizon: int
    state_dim: int
    control_dim: int
    state_bounds: BoundsInput = None
    control_bounds: BoundsInput = None
    name: str = "unnamed_problem"

    def __post_init__(self) -> None:
        if not callable(self.dynamics):
            raise TypeError("dynamics must be callable.")

        if not callable(self.running_cost):
            raise TypeError("running_cost must be callable.")

        if not callable(self.terminal_cost):
            raise TypeError("terminal_cost must be callable.")

        if not isinstance(self.horizon, int) or self.horizon <= 0:
            raise ValueError("horizon must be a positive integer.")

        if not isinstance(self.state_dim, int) or self.state_dim <= 0:
            raise ValueError("state_dim must be a positive integer.")

        if not isinstance(self.control_dim, int) or self.control_dim <= 0:
            raise ValueError("control_dim must be a positive integer.")

        prepared_state_bounds = _prepare_bounds(
            self.state_bounds,
            self.state_dim,
            "state_bounds",
        )

        prepared_control_bounds = _prepare_bounds(
            self.control_bounds,
            self.control_dim,
            "control_bounds",
        )

        object.__setattr__(
            self,
            "state_bounds",
            prepared_state_bounds,
        )
        object.__setattr__(
            self,
            "control_bounds",
            prepared_control_bounds,
        )

    def validate_state(self, state: Array) -> Array:
        """Validate and return a state vector."""

        state = np.asarray(state, dtype=float)

        if state.shape != (self.state_dim,):
            raise ValueError(
                f"State must have shape ({self.state_dim},), "
                f"but received {state.shape}."
            )

        if not np.all(np.isfinite(state)):
            raise ValueError("State must contain only finite values.")

        return state

    def validate_control(self, control: Array) -> Array:
        """Validate and return a control vector."""

        control = np.asarray(control, dtype=float)

        if control.shape != (self.control_dim,):
            raise ValueError(
                f"Control must have shape ({self.control_dim},), "
                f"but received {control.shape}."
            )

        if not np.all(np.isfinite(control)):
            raise ValueError("Control must contain only finite values.")

        return control

    def step(self, state: Array, control: Array, k: int) -> Array:
        """Evaluate the discrete dynamics."""

        state = self.validate_state(state)
        control = self.validate_control(control)

        if not isinstance(k, (int, np.integer)):
            raise TypeError("Time index k must be an integer.")

        if k < 0 or k >= self.horizon:
            raise ValueError(
                f"Time index k must satisfy 0 <= k < {self.horizon}."
            )

        next_state = self.dynamics(state, control, int(k))
        next_state = self.validate_state(next_state)

        return next_state

    def evaluate_running_cost(
        self,
        state: Array,
        control: Array,
        k: int,
    ) -> float:
        """Evaluate the running cost."""

        state = self.validate_state(state)
        control = self.validate_control(control)

        value = np.asarray(
            self.running_cost(state, control, k),
            dtype=float,
        )

        if value.ndim != 0:
            raise ValueError("running_cost must return a scalar.")

        scalar_value = float(value)

        if not np.isfinite(scalar_value):
            raise ValueError("running_cost returned a non-finite value.")

        return scalar_value

    def evaluate_terminal_cost(self, state: Array) -> float:
        """Evaluate the terminal cost."""

        state = self.validate_state(state)

        value = np.asarray(
            self.terminal_cost(state),
            dtype=float,
        )

        if value.ndim != 0:
            raise ValueError("terminal_cost must return a scalar.")

        scalar_value = float(value)

        if not np.isfinite(scalar_value):
            raise ValueError("terminal_cost returned a non-finite value.")

        return scalar_value

    def is_state_feasible(
        self,
        state: Array,
        *,
        strict: bool = True,
    ) -> bool:
        """Check whether a state satisfies its box constraints."""

        state = self.validate_state(state)

        if self.state_bounds is None:
            return True

        return self.state_bounds.contains(state, strict=strict)

    def is_control_feasible(
        self,
        control: Array,
        *,
        strict: bool = True,
    ) -> bool:
        """Check whether a control satisfies its box constraints."""

        control = self.validate_control(control)

        if self.control_bounds is None:
            return True

        return self.control_bounds.contains(control, strict=strict)
