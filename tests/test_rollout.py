"""Tests for trajectory rollout."""

import numpy as np
import pytest

from boxilqr import (
    BoxILQRProblem,
    InfeasiblePointError,
    rollout,
    rollout_policy,
)


def create_double_integrator_problem() -> BoxILQRProblem:
    """Create a small discrete double-integrator problem."""

    def dynamics(
        state: np.ndarray,
        control: np.ndarray,
        k: int,
    ) -> np.ndarray:
        del k

        return np.array([
            state[0] + state[1],
            state[1] + control[0],
        ])

    def running_cost(
        state: np.ndarray,
        control: np.ndarray,
        k: int,
    ) -> float:
        del k
        return 0.5 * (
            state @ state
            + control @ control
        )

    def terminal_cost(state: np.ndarray) -> float:
        return 0.5 * state @ state

    return BoxILQRProblem(
        dynamics=dynamics,
        running_cost=running_cost,
        terminal_cost=terminal_cost,
        horizon=2,
        state_dim=2,
        control_dim=1,
    )


def test_open_loop_rollout() -> None:
    problem = create_double_integrator_problem()

    result = rollout(
        problem=problem,
        initial_state=np.array([0.0, 0.0]),
        controls=np.array([
            [1.0],
            [1.0],
        ]),
    )

    expected_states = np.array([
        [0.0, 0.0],
        [0.0, 1.0],
        [1.0, 2.0],
    ])

    np.testing.assert_allclose(
        result.states,
        expected_states,
    )

    np.testing.assert_allclose(
        result.original_cost,
        4.0,
    )

    np.testing.assert_allclose(
        result.barrier_cost,
        0.0,
    )


def test_rollout_with_barriers() -> None:
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
        return 0.5 * (
            state @ state
            + control @ control
        )

    def terminal_cost(state: np.ndarray) -> float:
        return state @ state

    problem = BoxILQRProblem(
        dynamics=dynamics,
        running_cost=running_cost,
        terminal_cost=terminal_cost,
        horizon=2,
        state_dim=1,
        control_dim=1,
        state_bounds=(
            np.array([-1.0]),
            np.array([1.0]),
        ),
        control_bounds=(
            np.array([-0.5]),
            np.array([0.5]),
        ),
    )

    result = rollout(
        problem=problem,
        initial_state=np.array([0.0]),
        controls=np.array([
            [0.2],
            [0.2],
        ]),
        state_barrier=0.1,
        control_barrier=0.2,
    )

    assert np.isfinite(result.total_cost)
    assert result.barrier_cost > 0.0

    assert problem.is_state_feasible(result.states[-1])
    assert problem.is_control_feasible(result.controls[0])


def test_infeasible_control_is_rejected() -> None:
    def dynamics(
        state: np.ndarray,
        control: np.ndarray,
        k: int,
    ) -> np.ndarray:
        del k
        return state + control

    problem = BoxILQRProblem(
        dynamics=dynamics,
        running_cost=lambda state, control, k: 0.0,
        terminal_cost=lambda state: 0.0,
        horizon=1,
        state_dim=1,
        control_dim=1,
        state_bounds=(
            np.array([-2.0]),
            np.array([2.0]),
        ),
        control_bounds=(
            np.array([-0.5]),
            np.array([0.5]),
        ),
    )

    with pytest.raises(InfeasiblePointError):
        rollout(
            problem=problem,
            initial_state=np.array([0.0]),
            controls=np.array([[0.5]]),
            state_barrier=0.1,
            control_barrier=0.1,
        )


def test_policy_rollout() -> None:
    problem = create_double_integrator_problem()

    def policy(
        state: np.ndarray,
        k: int,
    ) -> np.ndarray:
        del k
        return np.array([-0.5 * state[1]])

    result = rollout_policy(
        problem=problem,
        initial_state=np.array([0.0, 1.0]),
        policy=policy,
    )

    expected_controls = np.array([
        [-0.5],
        [-0.25],
    ])

    np.testing.assert_allclose(
        result.controls,
        expected_controls,
    )
