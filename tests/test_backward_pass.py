"""Tests for the Box-iLQR backward pass."""

import numpy as np
import pytest

from boxilqr import (
    BackwardPassError,
    BoxILQRProblem,
    backward_pass,
    rollout,
)


def create_lqr_problem() -> tuple[
    BoxILQRProblem,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    state_matrix = np.array([
        [1.0, 0.1],
        [0.0, 1.0],
    ])

    control_matrix = np.array([
        [0.0],
        [0.1],
    ])

    state_cost = np.diag([1.0, 0.1])
    control_cost = np.array([[0.01]])
    terminal_cost_matrix = np.diag([10.0, 1.0])

    def dynamics(
        state: np.ndarray,
        control: np.ndarray,
        k: int,
    ) -> np.ndarray:
        del k
        return state_matrix @ state + control_matrix @ control

    def running_cost(
        state: np.ndarray,
        control: np.ndarray,
        k: int,
    ) -> float:
        del k

        return (
            0.5 * state @ state_cost @ state
            + 0.5 * control @ control_cost @ control
        )

    def terminal_cost(state: np.ndarray) -> float:
        return 0.5 * state @ terminal_cost_matrix @ state

    problem = BoxILQRProblem(
        dynamics=dynamics,
        running_cost=running_cost,
        terminal_cost=terminal_cost,
        horizon=3,
        state_dim=2,
        control_dim=1,
    )

    return (
        problem,
        state_matrix,
        control_matrix,
        state_cost,
        control_cost,
        terminal_cost_matrix,
    )


def test_last_feedback_gain_matches_lqr() -> None:
    (
        problem,
        state_matrix,
        control_matrix,
        _,
        control_cost,
        terminal_cost_matrix,
    ) = create_lqr_problem()

    initial_state = np.array([1.0, 0.0])
    controls = np.zeros((problem.horizon, problem.control_dim))

    trajectory = rollout(
        problem,
        initial_state,
        controls,
    )

    regularization = 1.0e-8

    result = backward_pass(
        problem,
        trajectory.states,
        trajectory.controls,
        regularization=regularization,
    )

    expected_quu = (
        control_cost
        + control_matrix.T
        @ terminal_cost_matrix
        @ control_matrix
        + regularization * np.eye(1)
    )

    expected_qux = (
        control_matrix.T
        @ terminal_cost_matrix
        @ state_matrix
    )

    expected_gain = -np.linalg.solve(
        expected_quu,
        expected_qux,
    )

    np.testing.assert_allclose(
        result.feedback_gains[-1],
        expected_gain,
        rtol=1.0e-4,
        atol=1.0e-4,
    )


def test_predicted_cost_change_is_negative() -> None:
    problem, *_ = create_lqr_problem()

    trajectory = rollout(
        problem,
        initial_state=np.array([1.0, 0.0]),
        controls=np.zeros((problem.horizon, 1)),
    )

    result = backward_pass(
        problem,
        trajectory.states,
        trajectory.controls,
        regularization=1.0e-6,
    )

    assert result.predicted_cost_change(1.0) < 0.0
    assert result.predicted_reduction(1.0) > 0.0
    assert result.max_feedforward_norm > 0.0


def test_control_barrier_reduces_feedback_gain() -> None:
    def dynamics(
        state: np.ndarray,
        control: np.ndarray,
        k: int,
    ) -> np.ndarray:
        del k
        return state + control

    def running_cost(
        state: np.ndarray,
        control: np.ndarray,
        k: int,
    ) -> float:
        del k
        return 0.05 * control @ control

    def terminal_cost(state: np.ndarray) -> float:
        return 5.0 * state @ state

    problem = BoxILQRProblem(
        dynamics=dynamics,
        running_cost=running_cost,
        terminal_cost=terminal_cost,
        horizon=1,
        state_dim=1,
        control_dim=1,
        control_bounds=(
            np.array([-1.0]),
            np.array([1.0]),
        ),
    )

    trajectory = rollout(
        problem,
        initial_state=np.array([0.2]),
        controls=np.array([[0.0]]),
        control_barrier=0.01,
    )

    small_barrier = backward_pass(
        problem,
        trajectory.states,
        trajectory.controls,
        control_barrier=0.01,
    )

    large_barrier = backward_pass(
        problem,
        trajectory.states,
        trajectory.controls,
        control_barrier=10.0,
    )

    assert (
        abs(large_barrier.feedback_gains[0, 0, 0])
        < abs(small_barrier.feedback_gains[0, 0, 0])
    )


def test_non_positive_definite_quu_is_reported() -> None:
    def dynamics(
        state: np.ndarray,
        control: np.ndarray,
        k: int,
    ) -> np.ndarray:
        del control, k
        return state

    def running_cost(
        state: np.ndarray,
        control: np.ndarray,
        k: int,
    ) -> float:
        del state, k
        return -(control @ control)

    problem = BoxILQRProblem(
        dynamics=dynamics,
        running_cost=running_cost,
        terminal_cost=lambda state: 0.0,
        horizon=1,
        state_dim=1,
        control_dim=1,
    )

    states = np.zeros((2, 1))
    controls = np.zeros((1, 1))

    with pytest.raises(BackwardPassError):
        backward_pass(
            problem,
            states,
            controls,
            regularization=0.1,
        )
