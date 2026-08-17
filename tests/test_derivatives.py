"""Tests for finite-difference derivatives."""

import numpy as np

from boxilqr import (
    BoxILQRProblem,
    differentiate_dynamics,
    differentiate_running_cost,
    differentiate_terminal_cost,
    finite_difference_gradient,
    finite_difference_hessian,
    finite_difference_jacobian,
)


def test_vector_function_jacobian() -> None:
    def function(point: np.ndarray) -> np.ndarray:
        return np.array([
            point[0] ** 2 + 3.0 * point[1],
            np.sin(point[0]),
        ])

    point = np.array([0.4, -0.2])

    jacobian = finite_difference_jacobian(
        function,
        point,
    )

    expected = np.array([
        [0.8, 3.0],
        [np.cos(0.4), 0.0],
    ])

    np.testing.assert_allclose(
        jacobian,
        expected,
        rtol=1.0e-6,
        atol=1.0e-8,
    )


def test_quadratic_gradient_and_hessian() -> None:
    matrix = np.array([
        [4.0, 1.0],
        [1.0, 2.0],
    ])

    linear_term = np.array([0.5, -0.3])

    def function(point: np.ndarray) -> float:
        return (
            0.5 * point @ matrix @ point
            + linear_term @ point
        )

    point = np.array([0.3, -0.7])

    gradient = finite_difference_gradient(
        function,
        point,
    )

    hessian = finite_difference_hessian(
        function,
        point,
    )

    np.testing.assert_allclose(
        gradient,
        matrix @ point + linear_term,
        rtol=1.0e-6,
        atol=1.0e-8,
    )

    np.testing.assert_allclose(
        hessian,
        matrix,
        rtol=1.0e-5,
        atol=1.0e-6,
    )


def test_problem_derivatives() -> None:
    state_matrix = np.array([
        [1.0, 0.1],
        [0.0, 1.0],
    ])

    control_matrix = np.array([
        [0.0],
        [0.1],
    ])

    state_cost_matrix = np.diag([2.0, 4.0])
    control_cost_matrix = np.array([[3.0]])
    cross_vector = np.array([0.2, -0.4])

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
            0.5 * state @ state_cost_matrix @ state
            + 0.5 * control @ control_cost_matrix @ control
            + control[0] * cross_vector @ state
        )

    def terminal_cost(state: np.ndarray) -> float:
        return 5.0 * state @ state

    problem = BoxILQRProblem(
        dynamics=dynamics,
        running_cost=running_cost,
        terminal_cost=terminal_cost,
        horizon=10,
        state_dim=2,
        control_dim=1,
    )

    state = np.array([0.5, -0.25])
    control = np.array([0.1])

    dynamics_derivatives = differentiate_dynamics(
        problem,
        state,
        control,
        k=0,
    )

    cost_derivatives = differentiate_running_cost(
        problem,
        state,
        control,
        k=0,
    )

    terminal_derivatives = differentiate_terminal_cost(
        problem,
        state,
    )

    np.testing.assert_allclose(
        dynamics_derivatives.fx,
        state_matrix,
        rtol=1.0e-6,
        atol=1.0e-8,
    )

    np.testing.assert_allclose(
        dynamics_derivatives.fu,
        control_matrix,
        rtol=1.0e-6,
        atol=1.0e-8,
    )

    np.testing.assert_allclose(
        cost_derivatives.lxx,
        state_cost_matrix,
        rtol=1.0e-5,
        atol=1.0e-6,
    )

    np.testing.assert_allclose(
        cost_derivatives.luu,
        control_cost_matrix,
        rtol=1.0e-5,
        atol=1.0e-6,
    )

    np.testing.assert_allclose(
        cost_derivatives.lux,
        cross_vector.reshape(1, 2),
        rtol=1.0e-5,
        atol=1.0e-6,
    )

    np.testing.assert_allclose(
        terminal_derivatives.phix,
        10.0 * state,
        rtol=1.0e-6,
        atol=1.0e-8,
    )

    np.testing.assert_allclose(
        terminal_derivatives.phixx,
        10.0 * np.eye(2),
        rtol=1.0e-5,
        atol=1.0e-6,
    )
