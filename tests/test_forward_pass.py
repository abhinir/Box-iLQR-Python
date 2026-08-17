"""Tests for the feasibility-preserving line search."""

import numpy as np

from boxilqr import (
    BackwardPassResult,
    BoxILQRProblem,
    ILQROptions,
    backward_pass,
    line_search,
    rollout,
)


def create_test_problem(
    horizon: int = 10,
) -> BoxILQRProblem:
    """Create a discrete double-integrator problem."""

    dt = 0.1

    state_matrix = np.array([
        [1.0, dt],
        [0.0, 1.0],
    ])

    control_matrix = np.array([
        [0.5 * dt**2],
        [dt],
    ])

    state_cost = np.diag([1.0, 0.1])
    control_cost = np.array([[0.01]])
    terminal_cost_matrix = np.diag([20.0, 2.0])

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
        return (
            0.5
            * state
            @ terminal_cost_matrix
            @ state
        )

    return BoxILQRProblem(
        dynamics=dynamics,
        running_cost=running_cost,
        terminal_cost=terminal_cost,
        horizon=horizon,
        state_dim=2,
        control_dim=1,
    )


def test_line_search_accepts_improving_trajectory() -> None:
    problem = create_test_problem()

    nominal = rollout(
        problem=problem,
        initial_state=np.array([1.0, 0.0]),
        controls=np.zeros((problem.horizon, 1)),
    )

    backward_result = backward_pass(
        problem,
        nominal.states,
        nominal.controls,
        regularization=1.0e-6,
    )

    result = line_search(
        problem=problem,
        nominal_rollout=nominal,
        backward_result=backward_result,
    )

    assert result.accepted
    assert result.rollout is not None
    assert result.rollout.total_cost < nominal.total_cost
    assert result.step_size is not None
    assert result.reduction_ratio is not None


def test_line_search_reduces_step_for_feasibility() -> None:
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
        del state, k
        return 0.5 * control @ control

    def terminal_cost(state: np.ndarray) -> float:
        terminal_error = state[0] - 1.0
        return 50.0 * terminal_error**2

    problem = BoxILQRProblem(
        dynamics=dynamics,
        running_cost=running_cost,
        terminal_cost=terminal_cost,
        horizon=1,
        state_dim=1,
        control_dim=1,
        control_bounds=(
            np.array([-0.2]),
            np.array([0.2]),
        ),
    )

    barrier_parameter = 1.0e-4

    nominal = rollout(
        problem=problem,
        initial_state=np.array([0.0]),
        controls=np.array([[0.0]]),
        control_barrier=barrier_parameter,
    )

    backward_result = backward_pass(
        problem,
        nominal.states,
        nominal.controls,
        control_barrier=barrier_parameter,
    )

    result = line_search(
        problem=problem,
        nominal_rollout=nominal,
        backward_result=backward_result,
        control_barrier=barrier_parameter,
        options=ILQROptions(
            max_line_search_steps=10,
            line_search_decay=0.5,
        ),
    )

    assert result.accepted
    assert result.rollout is not None
    assert result.step_size is not None
    assert result.step_size < 1.0

    assert any(
        attempt.feasible is False
        for attempt in result.attempts
    )

    assert np.all(
        np.abs(result.rollout.controls) < 0.2
    )


def test_line_search_reports_failure() -> None:
    def dynamics(
        state: np.ndarray,
        control: np.ndarray,
        k: int,
    ) -> np.ndarray:
        del k
        return state + control

    problem = BoxILQRProblem(
        dynamics=dynamics,
        running_cost=lambda state, control, k: (
            0.0
        ),
        terminal_cost=lambda state: state @ state,
        horizon=1,
        state_dim=1,
        control_dim=1,
    )

    nominal = rollout(
        problem=problem,
        initial_state=np.array([1.0]),
        controls=np.array([[0.0]]),
    )

    incorrect_direction = BackwardPassResult(
        feedforward_terms=np.array([[1.0]]),
        feedback_gains=np.zeros((1, 1, 1)),
        expected_linear_change=-1.0,
        expected_quadratic_change=0.5,
        max_feedforward_norm=1.0,
        minimum_regularized_eigenvalue=1.0,
        regularization=0.0,
    )

    result = line_search(
        problem=problem,
        nominal_rollout=nominal,
        backward_result=incorrect_direction,
        options=ILQROptions(
            max_line_search_steps=3,
        ),
    )

    assert not result.accepted
    assert result.rollout is None
    assert result.number_of_attempts == 3
