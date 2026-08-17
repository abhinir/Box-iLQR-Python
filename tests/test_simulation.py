"""Tests for open-loop and feedback simulation."""

import numpy as np

from boxilqr import (
    BoxILQRProblem,
    BoxILQRSolution,
    SolverStatus,
    simulate_closed_loop,
)


def create_scalar_problem(
    horizon: int,
    bounded: bool = False,
) -> BoxILQRProblem:
    control_bounds = None

    if bounded:
        control_bounds = (
            np.array([-0.1]),
            np.array([0.1]),
        )

    return BoxILQRProblem(
        dynamics=lambda state, control, k: (
            state + control
        ),
        running_cost=lambda state, control, k: 0.0,
        terminal_cost=lambda state: 0.0,
        horizon=horizon,
        state_dim=1,
        control_dim=1,
        control_bounds=control_bounds,
    )


def create_scalar_solution(
    horizon: int,
    feedback_gain: float,
) -> BoxILQRSolution:
    feedback_gains = np.full(
        (horizon, 1, 1),
        feedback_gain,
    )

    return BoxILQRSolution(
        states=np.zeros((horizon + 1, 1)),
        controls=np.zeros((horizon, 1)),
        feedback_gains=feedback_gains,
        feedforward_terms=np.zeros((horizon, 1)),
        cost=0.0,
        status=SolverStatus.CONVERGED,
        message="Test solution.",
    )


def test_open_loop_reproduces_nominal_trajectory() -> None:
    horizon = 5

    problem = create_scalar_problem(horizon)
    solution = create_scalar_solution(
        horizon,
        feedback_gain=-0.5,
    )

    result = simulate_closed_loop(
        problem,
        solution,
        feedback=False,
    )

    np.testing.assert_allclose(
        result.states,
        solution.states,
    )

    np.testing.assert_allclose(
        result.controls,
        solution.controls,
    )


def test_feedback_corrects_initial_error() -> None:
    horizon = 5

    problem = create_scalar_problem(horizon)
    solution = create_scalar_solution(
        horizon,
        feedback_gain=-0.5,
    )

    perturbed_initial_state = np.array([0.2])

    open_loop = simulate_closed_loop(
        problem,
        solution,
        initial_state=perturbed_initial_state,
        feedback=False,
    )

    feedback = simulate_closed_loop(
        problem,
        solution,
        initial_state=perturbed_initial_state,
        feedback=True,
    )

    assert (
        abs(feedback.states[-1, 0])
        < abs(open_loop.states[-1, 0])
    )


def test_applied_control_is_projected() -> None:
    problem = create_scalar_problem(
        horizon=1,
        bounded=True,
    )

    solution = create_scalar_solution(
        horizon=1,
        feedback_gain=-10.0,
    )

    result = simulate_closed_loop(
        problem,
        solution,
        initial_state=np.array([1.0]),
        feedback=True,
        project_controls=True,
    )

    np.testing.assert_allclose(
        result.commanded_controls[0],
        np.array([-10.0]),
    )

    np.testing.assert_allclose(
        result.applied_controls[0],
        np.array([-0.1]),
    )


def test_measurement_and_process_noise() -> None:
    problem = create_scalar_problem(horizon=1)
    solution = create_scalar_solution(
        horizon=1,
        feedback_gain=-1.0,
    )

    result = simulate_closed_loop(
        problem,
        solution,
        initial_state=np.array([0.0]),
        feedback=True,
        measurement_noise=lambda state, k: (
            np.array([0.1])
        ),
        process_noise=lambda state, control, k: (
            np.array([0.05])
        ),
    )

    np.testing.assert_allclose(
        result.measured_states[0],
        np.array([0.1]),
    )

    np.testing.assert_allclose(
        result.commanded_controls[0],
        np.array([-0.1]),
    )

    np.testing.assert_allclose(
        result.states[-1],
        np.array([-0.05]),
    )
