"""Tests for the optimal-control problem definition."""

import numpy as np
import pytest

from boxilqr import Bounds, BoxILQRProblem


def test_bounds_contains_and_projects() -> None:
    bounds = Bounds(
        lower=np.array([-1.0, -np.inf]),
        upper=np.array([1.0, np.inf]),
    )

    assert bounds.contains(np.array([0.0, 100.0]))
    assert not bounds.contains(np.array([1.0, 0.0]))
    assert bounds.contains(np.array([1.0, 0.0]), strict=False)

    projected = bounds.project(np.array([2.0, 100.0]))

    np.testing.assert_allclose(
        projected,
        np.array([1.0, 100.0]),
    )


def test_problem_evaluations() -> None:
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
        return 0.5 * (state @ state + control @ control)

    def terminal_cost(state: np.ndarray) -> float:
        return 10.0 * state @ state

    problem = BoxILQRProblem(
        dynamics=dynamics,
        running_cost=running_cost,
        terminal_cost=terminal_cost,
        horizon=10,
        state_dim=2,
        control_dim=2,
        state_bounds=(
            np.array([-5.0, -5.0]),
            np.array([5.0, 5.0]),
        ),
        control_bounds=(
            np.array([-1.0, -1.0]),
            np.array([1.0, 1.0]),
        ),
        name="test_problem",
    )

    state = np.array([0.5, -0.5])
    control = np.array([0.1, 0.2])

    next_state = problem.step(state, control, k=0)

    np.testing.assert_allclose(
        next_state,
        np.array([0.6, -0.3]),
    )

    assert problem.evaluate_running_cost(state, control, k=0) > 0.0
    assert problem.evaluate_terminal_cost(state) > 0.0
    assert problem.is_state_feasible(state)
    assert problem.is_control_feasible(control)


def test_invalid_bounds_are_rejected() -> None:
    with pytest.raises(ValueError):
        Bounds(
            lower=np.array([0.0]),
            upper=np.array([0.0]),
        )
