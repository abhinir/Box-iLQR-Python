"""The backward pass for Box-iLQR."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from boxilqr.barriers import (
    evaluate_box_barrier,
    expand_barrier_parameters,
)
from boxilqr.derivatives import (
    differentiate_dynamics,
    differentiate_running_cost,
    differentiate_terminal_cost,
)
from boxilqr.options import ScalarOrArray
from boxilqr.problem import Array, Bounds, BoxILQRProblem


class BackwardPassError(RuntimeError):
    """Raised when a valid backward pass cannot be completed."""

    def __init__(
        self,
        message: str,
        time_index: int | None = None,
    ) -> None:
        self.time_index = time_index

        if time_index is not None:
            message = f"{message} Time index: {time_index}."

        super().__init__(message)


@dataclass(frozen=True)
class BackwardPassResult:
    """Feedforward and feedback terms from a backward pass."""

    feedforward_terms: Array
    feedback_gains: Array

    expected_linear_change: float
    expected_quadratic_change: float

    max_feedforward_norm: float
    minimum_regularized_eigenvalue: float
    regularization: float

    def predicted_cost_change(self, step_size: float) -> float:
        """Predicted change in cost for a line-search step."""

        step_size = float(step_size)

        return (
            step_size * self.expected_linear_change
            + step_size**2 * self.expected_quadratic_change
        )

    def predicted_reduction(self, step_size: float) -> float:
        """Positive predicted cost reduction."""

        return -self.predicted_cost_change(step_size)


def _prepare_parameters(
    bounds: Bounds | None,
    parameters: ScalarOrArray | None,
    dimension: int,
    name: str,
) -> Array | None:
    """Prepare componentwise barrier parameters."""

    if bounds is None or not bounds.has_constraints:
        return None

    if parameters is None:
        raise ValueError(
            f"{name} must be provided because finite bounds are present."
        )

    return expand_barrier_parameters(
        parameters,
        dimension,
        name=name,
    )


def _cholesky_solve(
    factor: Array,
    right_hand_side: Array,
) -> Array:
    """Solve a positive-definite system from its Cholesky factor."""

    intermediate = np.linalg.solve(
        factor,
        right_hand_side,
    )

    return np.linalg.solve(
        factor.T,
        intermediate,
    )


def backward_pass(
    problem: BoxILQRProblem,
    states: Array,
    controls: Array,
    *,
    state_barrier: ScalarOrArray | None = None,
    control_barrier: ScalarOrArray | None = None,
    regularization: float = 1.0e-6,
    minimum_margin: float = 1.0e-12,
    jacobian_step: float = 1.0e-6,
    gradient_step: float = 1.0e-6,
    hessian_step: float = 1.0e-4,
) -> BackwardPassResult:
    """Perform one regularized Box-iLQR backward pass."""

    states = np.asarray(states, dtype=float)
    controls = np.asarray(controls, dtype=float)

    expected_state_shape = (
        problem.horizon + 1,
        problem.state_dim,
    )

    expected_control_shape = (
        problem.horizon,
        problem.control_dim,
    )

    if states.shape != expected_state_shape:
        raise ValueError(
            f"states must have shape {expected_state_shape}, "
            f"but received {states.shape}."
        )

    if controls.shape != expected_control_shape:
        raise ValueError(
            f"controls must have shape {expected_control_shape}, "
            f"but received {controls.shape}."
        )

    if not np.all(np.isfinite(states)):
        raise ValueError("states contains non-finite values.")

    if not np.all(np.isfinite(controls)):
        raise ValueError("controls contains non-finite values.")

    regularization = float(regularization)

    if not np.isfinite(regularization):
        raise ValueError("regularization must be finite.")

    if regularization < 0.0:
        raise ValueError("regularization cannot be negative.")

    minimum_margin = float(minimum_margin)

    if not np.isfinite(minimum_margin):
        raise ValueError("minimum_margin must be finite.")

    if minimum_margin < 0.0:
        raise ValueError("minimum_margin cannot be negative.")

    state_parameters = _prepare_parameters(
        problem.state_bounds,
        state_barrier,
        problem.state_dim,
        "state_barrier",
    )

    control_parameters = _prepare_parameters(
        problem.control_bounds,
        control_barrier,
        problem.control_dim,
        "control_barrier",
    )

    feedforward_terms = np.zeros(
        (problem.horizon, problem.control_dim),
        dtype=float,
    )

    feedback_gains = np.zeros(
        (
            problem.horizon,
            problem.control_dim,
            problem.state_dim,
        ),
        dtype=float,
    )

    terminal_derivatives = differentiate_terminal_cost(
        problem,
        states[-1],
        gradient_step=gradient_step,
        hessian_step=hessian_step,
    )

    value_gradient = terminal_derivatives.phix.copy()
    value_hessian = terminal_derivatives.phixx.copy()

    if state_parameters is not None:
        terminal_barrier = evaluate_box_barrier(
            value=states[-1],
            bounds=problem.state_bounds,
            parameters=state_parameters,
            minimum_margin=minimum_margin,
            variable_name="terminal state",
        )

        value_gradient += terminal_barrier.gradient
        value_hessian += terminal_barrier.hessian

    value_hessian = 0.5 * (
        value_hessian + value_hessian.T
    )

    expected_linear_change = 0.0
    expected_quadratic_change = 0.0

    maximum_feedforward_norm = 0.0
    minimum_regularized_eigenvalue = np.inf

    control_identity = np.eye(problem.control_dim)

    for k in reversed(range(problem.horizon)):
        state = states[k]
        control = controls[k]

        dynamics_derivatives = differentiate_dynamics(
            problem,
            state,
            control,
            k,
            relative_step=jacobian_step,
        )

        cost_derivatives = differentiate_running_cost(
            problem,
            state,
            control,
            k,
            gradient_step=gradient_step,
            hessian_step=hessian_step,
        )

        lx = cost_derivatives.lx.copy()
        lu = cost_derivatives.lu.copy()
        lxx = cost_derivatives.lxx.copy()
        luu = cost_derivatives.luu.copy()
        lux = cost_derivatives.lux.copy()

        if state_parameters is not None:
            state_barrier_evaluation = evaluate_box_barrier(
                value=state,
                bounds=problem.state_bounds,
                parameters=state_parameters,
                minimum_margin=minimum_margin,
                variable_name=f"state at time {k}",
            )

            lx += state_barrier_evaluation.gradient
            lxx += state_barrier_evaluation.hessian

        if control_parameters is not None:
            control_barrier_evaluation = evaluate_box_barrier(
                value=control,
                bounds=problem.control_bounds,
                parameters=control_parameters,
                minimum_margin=minimum_margin,
                variable_name=f"control at time {k}",
            )

            lu += control_barrier_evaluation.gradient
            luu += control_barrier_evaluation.hessian

        fx = dynamics_derivatives.fx
        fu = dynamics_derivatives.fu

        qx = lx + fx.T @ value_gradient
        qu = lu + fu.T @ value_gradient

        qxx = lxx + fx.T @ value_hessian @ fx
        quu = luu + fu.T @ value_hessian @ fu
        qux = lux + fu.T @ value_hessian @ fx

        qxx = 0.5 * (qxx + qxx.T)
        quu = 0.5 * (quu + quu.T)

        regularized_quu = (
            quu + regularization * control_identity
        )

        regularized_quu = 0.5 * (
            regularized_quu + regularized_quu.T
        )

        if not np.all(np.isfinite(regularized_quu)):
            raise BackwardPassError(
                "The regularized Q_uu matrix is non-finite.",
                time_index=k,
            )

        try:
            eigenvalues = np.linalg.eigvalsh(
                regularized_quu
            )
        except np.linalg.LinAlgError as error:
            raise BackwardPassError(
                "Could not calculate eigenvalues of Q_uu.",
                time_index=k,
            ) from error

        current_minimum_eigenvalue = float(
            np.min(eigenvalues)
        )

        minimum_regularized_eigenvalue = min(
            minimum_regularized_eigenvalue,
            current_minimum_eigenvalue,
        )

        try:
            cholesky_factor = np.linalg.cholesky(
                regularized_quu
            )
        except np.linalg.LinAlgError as error:
            raise BackwardPassError(
                "The regularized Q_uu matrix is not "
                "positive definite.",
                time_index=k,
            ) from error

        feedforward = -_cholesky_solve(
            cholesky_factor,
            qu,
        )

        feedback = -_cholesky_solve(
            cholesky_factor,
            qux,
        )

        if not np.all(np.isfinite(feedforward)):
            raise BackwardPassError(
                "The feedforward term is non-finite.",
                time_index=k,
            )

        if not np.all(np.isfinite(feedback)):
            raise BackwardPassError(
                "The feedback gain is non-finite.",
                time_index=k,
            )

        feedforward_terms[k] = feedforward
        feedback_gains[k] = feedback

        maximum_feedforward_norm = max(
            maximum_feedforward_norm,
            float(np.linalg.norm(feedforward, ord=np.inf)),
        )

        expected_linear_change += float(
            qu @ feedforward
        )

        expected_quadratic_change += float(
            0.5
            * feedforward
            @ regularized_quu
            @ feedforward
        )

        value_gradient = (
            qx
            + feedback.T @ quu @ feedforward
            + feedback.T @ qu
            + qux.T @ feedforward
        )

        value_hessian = (
            qxx
            + feedback.T @ quu @ feedback
            + feedback.T @ qux
            + qux.T @ feedback
        )

        value_hessian = 0.5 * (
            value_hessian + value_hessian.T
        )

        if not np.all(np.isfinite(value_gradient)):
            raise BackwardPassError(
                "The value gradient is non-finite.",
                time_index=k,
            )

        if not np.all(np.isfinite(value_hessian)):
            raise BackwardPassError(
                "The value Hessian is non-finite.",
                time_index=k,
            )

    if not np.isfinite(minimum_regularized_eigenvalue):
        minimum_regularized_eigenvalue = np.nan

    return BackwardPassResult(
        feedforward_terms=feedforward_terms,
        feedback_gains=feedback_gains,
        expected_linear_change=float(
            expected_linear_change
        ),
        expected_quadratic_change=float(
            expected_quadratic_change
        ),
        max_feedforward_norm=float(
            maximum_feedforward_norm
        ),
        minimum_regularized_eigenvalue=float(
            minimum_regularized_eigenvalue
        ),
        regularization=regularization,
    )
