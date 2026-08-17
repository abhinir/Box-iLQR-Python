"""Result structures returned by Box-iLQR."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np

from boxilqr.problem import Array, Bounds


class SolverStatus(str, Enum):
    """Possible Box-iLQR termination conditions."""

    CONVERGED = "converged"
    MAX_ITERATIONS = "maximum_iterations_reached"
    LINE_SEARCH_FAILED = "line_search_failed"
    REGULARIZATION_LIMIT = "regularization_limit_reached"
    INFEASIBLE_INITIAL_TRAJECTORY = "infeasible_initial_trajectory"
    BARRIER_CONTINUATION_FAILED = "barrier_continuation_failed"
    NUMERICAL_FAILURE = "numerical_failure"


@dataclass(frozen=True)
class IterationRecord:
    """Diagnostic information from one inner iLQR iteration."""

    outer_iteration: int
    inner_iteration: int
    cost: float
    regularization: float
    accepted: bool

    cost_change: float | None = None
    expected_reduction: float | None = None
    step_size: float | None = None
    feedforward_norm: float | None = None

    state_barrier: Array | None = None
    control_barrier: Array | None = None


@dataclass(frozen=True)
class BoxILQRSolution:
    """Nominal trajectory and feedback policy returned by Box-iLQR."""

    states: Array
    controls: Array
    feedback_gains: Array
    feedforward_terms: Array

    cost: float
    status: SolverStatus
    message: str

    augmented_cost: float | None = None
    barrier_cost: float | None = None

    history: tuple[IterationRecord, ...] = ()

    state_barrier: Array | None = None
    control_barrier: Array | None = None

    def __post_init__(self) -> None:
        states = np.asarray(self.states, dtype=float)
        controls = np.asarray(self.controls, dtype=float)
        feedback_gains = np.asarray(self.feedback_gains, dtype=float)
        feedforward_terms = np.asarray(
            self.feedforward_terms,
            dtype=float,
        )

        if states.ndim != 2:
            raise ValueError(
                "states must have shape (horizon + 1, state_dim)."
            )

        if controls.ndim != 2:
            raise ValueError(
                "controls must have shape (horizon, control_dim)."
            )

        if feedback_gains.ndim != 3:
            raise ValueError(
                "feedback_gains must have shape "
                "(horizon, control_dim, state_dim)."
            )

        if feedforward_terms.ndim != 2:
            raise ValueError(
                "feedforward_terms must have shape "
                "(horizon, control_dim)."
            )

        horizon = controls.shape[0]
        state_dim = states.shape[1]
        control_dim = controls.shape[1]

        if states.shape[0] != horizon + 1:
            raise ValueError(
                "The state trajectory must contain one more point "
                "than the control trajectory."
            )

        expected_gain_shape = (
            horizon,
            control_dim,
            state_dim,
        )

        if feedback_gains.shape != expected_gain_shape:
            raise ValueError(
                f"feedback_gains must have shape "
                f"{expected_gain_shape}, but received "
                f"{feedback_gains.shape}."
            )

        if feedforward_terms.shape != controls.shape:
            raise ValueError(
                "feedforward_terms must have the same shape as controls."
            )

        if not np.all(np.isfinite(states)):
            raise ValueError("states contains non-finite values.")

        if not np.all(np.isfinite(controls)):
            raise ValueError("controls contains non-finite values.")

        if not np.all(np.isfinite(feedback_gains)):
            raise ValueError("feedback_gains contains non-finite values.")

        if not np.all(np.isfinite(feedforward_terms)):
            raise ValueError("feedforward_terms contains non-finite values.")

        object.__setattr__(self, "states", states.copy())
        object.__setattr__(self, "controls", controls.copy())
        object.__setattr__(
            self,
            "feedback_gains",
            feedback_gains.copy(),
        )
        object.__setattr__(
            self,
            "feedforward_terms",
            feedforward_terms.copy(),
        )
        object.__setattr__(self, "cost", float(self.cost))
        if not np.isfinite(self.cost):
            raise ValueError("cost must be finite.")

        if self.augmented_cost is not None:
            augmented_cost = float(self.augmented_cost)

            if not np.isfinite(augmented_cost):
                raise ValueError(
                    "augmented_cost must be finite."
                )

            object.__setattr__(
                self,
                "augmented_cost",
                augmented_cost,
            )

        if self.barrier_cost is not None:
            barrier_cost = float(self.barrier_cost)

            if not np.isfinite(barrier_cost):
                raise ValueError(
                    "barrier_cost must be finite."
                )

            object.__setattr__(
                self,
                "barrier_cost",
                barrier_cost,
            )
        object.__setattr__(self, "history", tuple(self.history))

        if self.state_barrier is not None:
            object.__setattr__(
                self,
                "state_barrier",
                np.asarray(self.state_barrier, dtype=float).copy(),
            )

        if self.control_barrier is not None:
            object.__setattr__(
                self,
                "control_barrier",
                np.asarray(self.control_barrier, dtype=float).copy(),
            )
    
    
    @property
    def original_cost(self) -> float:
        """Objective value without logarithmic barriers."""

        return self.cost
        
    @property
    def success(self) -> bool:
        """Whether Box-iLQR converged successfully."""

        return self.status is SolverStatus.CONVERGED

    @property
    def horizon(self) -> int:
        """Number of control intervals."""

        return self.controls.shape[0]

    @property
    def state_dim(self) -> int:
        """Dimension of the state vector."""

        return self.states.shape[1]

    @property
    def control_dim(self) -> int:
        """Dimension of the control vector."""

        return self.controls.shape[1]

    def feedback_control(
        self,
        state: Array,
        k: int,
        control_bounds: Bounds | None = None,
    ) -> Array:
        """Evaluate the time-varying feedback policy.

        The policy is

            u_k = u_nominal_k + K_k (x_k - x_nominal_k).

        If control_bounds is supplied, the result is projected onto
        the admissible control box.
        """

        if k < 0 or k >= self.horizon:
            raise ValueError(
                f"k must satisfy 0 <= k < {self.horizon}."
            )

        state = np.asarray(state, dtype=float)

        if state.shape != (self.state_dim,):
            raise ValueError(
                f"state must have shape ({self.state_dim},), "
                f"but received {state.shape}."
            )

        state_error = state - self.states[k]

        control = (
            self.controls[k]
            + self.feedback_gains[k] @ state_error
        )

        if control_bounds is not None:
            control = control_bounds.project(control)

        return control
