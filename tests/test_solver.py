"""Tests for the inner iLQR solver."""

import numpy as np

from boxilqr import (
    BoxILQRProblem,
    ILQROptions,
    SolverStatus,
    rollout,
    solve_ilqr_subproblem,
    BarrierOptions,
    BoxILQROptions,
    solve,
)


def create_lqr_problem(
    horizon: int = 25,
) -> BoxILQRProblem:
    """Create a finite-horizon linear-quadratic problem."""

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
    terminal_cost_matrix = np.diag([50.0, 5.0])

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


def test_inner_solver_converges_on_lqr() -> None:
    problem = create_lqr_problem()

    initial_state = np.array([1.0, 0.0])
    initial_controls = np.zeros(
        (problem.horizon, problem.control_dim)
    )

    initial_rollout = rollout(
        problem,
        initial_state,
        initial_controls,
    )

    result = solve_ilqr_subproblem(
        problem=problem,
        initial_state=initial_state,
        initial_controls=initial_controls,
        options=ILQROptions(
            max_iterations=50,
            relative_cost_tolerance=1.0e-8,
            feedforward_tolerance=1.0e-6,
        ),
    )

    assert result.success
    assert result.status is SolverStatus.CONVERGED
    assert result.rollout is not None
    assert result.backward_result is not None

    assert (
        result.rollout.total_cost
        < initial_rollout.total_cost
    )

    assert result.backward_result.feedback_gains.shape == (
        problem.horizon,
        problem.control_dim,
        problem.state_dim,
    )


def test_inner_solver_respects_control_bounds() -> None:
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
        return 0.05 * control @ control

    def terminal_cost(state: np.ndarray) -> float:
        error = state[0] - 1.0
        return 10.0 * error**2

    problem = BoxILQRProblem(
        dynamics=dynamics,
        running_cost=running_cost,
        terminal_cost=terminal_cost,
        horizon=5,
        state_dim=1,
        control_dim=1,
        control_bounds=(
            np.array([-0.3]),
            np.array([0.3]),
        ),
    )

    result = solve_ilqr_subproblem(
        problem=problem,
        initial_state=np.array([0.0]),
        initial_controls=np.zeros((5, 1)),
        control_barrier=0.05,
        options=ILQROptions(
            max_iterations=100,
            feedforward_tolerance=1.0e-5,
        ),
    )

    assert result.success
    assert result.rollout is not None

    assert np.all(result.rollout.controls < 0.3)
    assert np.all(result.rollout.controls > -0.3)


def test_infeasible_initial_control_is_reported() -> None:
    problem = BoxILQRProblem(
        dynamics=lambda state, control, k: (
            state + control
        ),
        running_cost=lambda state, control, k: 0.0,
        terminal_cost=lambda state: 0.0,
        horizon=1,
        state_dim=1,
        control_dim=1,
        control_bounds=(
            np.array([-0.2]),
            np.array([0.2]),
        ),
    )

    result = solve_ilqr_subproblem(
        problem=problem,
        initial_state=np.array([0.0]),
        initial_controls=np.array([[0.2]]),
        control_barrier=0.1,
    )

    assert not result.success

    assert (
        result.status
        is SolverStatus.INFEASIBLE_INITIAL_TRAJECTORY
    )

    assert result.failed_constraint_type == "control"
    assert result.failed_constraint_index == 0


def test_complete_solver_without_constraints() -> None:
    problem = create_lqr_problem()

    solution = solve(
        problem=problem,
        x0=np.array([1.0, 0.0]),
        u_initial=np.zeros(
            (problem.horizon, problem.control_dim)
        ),
        options=BoxILQROptions(
            ilqr=ILQROptions(
                max_iterations=50,
                feedforward_tolerance=1.0e-6,
            ),
            verbose=False,
        ),
    )

    assert solution.success
    assert solution.status is SolverStatus.CONVERGED

    assert solution.states.shape == (
        problem.horizon + 1,
        problem.state_dim,
    )

    assert solution.controls.shape == (
        problem.horizon,
        problem.control_dim,
    )


def test_complete_solver_with_control_continuation() -> None:
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
        return 0.05 * control @ control

    def terminal_cost(state: np.ndarray) -> float:
        error = state[0] - 1.0
        return 10.0 * error**2

    problem = BoxILQRProblem(
        dynamics=dynamics,
        running_cost=running_cost,
        terminal_cost=terminal_cost,
        horizon=5,
        state_dim=1,
        control_dim=1,
        control_bounds=(
            np.array([-0.3]),
            np.array([0.3]),
        ),
    )

    options = BoxILQROptions(
        ilqr=ILQROptions(
            max_iterations=100,
            feedforward_tolerance=1.0e-5,
        ),
        barrier=BarrierOptions(
            initial_control_barrier=0.1,
            control_reduction_factor=0.2,
            tolerance=1.0e-3,
            max_outer_iterations=20,
        ),
        verbose=False,
    )

    solution = solve(
        problem=problem,
        x0=np.array([0.0]),
        u_initial=np.zeros((5, 1)),
        options=options,
    )

    assert solution.success
    assert solution.control_barrier is not None

    assert (
        np.linalg.norm(solution.control_barrier)
        <= options.barrier.tolerance
    )

    assert np.all(solution.controls < 0.3)
    assert np.all(solution.controls > -0.3)
