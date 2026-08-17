"""Numerical derivatives used by Box-iLQR."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, TypeAlias

import numpy as np

from boxilqr.problem import Array, BoxILQRProblem


VectorFunction: TypeAlias = Callable[[Array], Array]
ScalarFunction: TypeAlias = Callable[[Array], float]
TwoArgumentScalarFunction: TypeAlias = Callable[[Array, Array], float]


@dataclass(frozen=True)
class DynamicsDerivatives:
    """Jacobians of the discrete dynamics."""

    fx: Array
    fu: Array


@dataclass(frozen=True)
class RunningCostDerivatives:
    """First- and second-order running-cost derivatives."""

    lx: Array
    lu: Array
    lxx: Array
    luu: Array
    lux: Array


@dataclass(frozen=True)
class TerminalCostDerivatives:
    """First- and second-order terminal-cost derivatives."""

    phix: Array
    phixx: Array


def _validate_point(point: Array, name: str) -> Array:
    """Validate a finite, one-dimensional point."""

    point = np.asarray(point, dtype=float)

    if point.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional.")

    if point.size == 0:
        raise ValueError(f"{name} cannot be empty.")

    if not np.all(np.isfinite(point)):
        raise ValueError(f"{name} must contain only finite values.")

    return point


def _validate_step(relative_step: float) -> float:
    """Validate a positive relative finite-difference step."""

    relative_step = float(relative_step)

    if not np.isfinite(relative_step):
        raise ValueError("relative_step must be finite.")

    if relative_step <= 0.0:
        raise ValueError("relative_step must be positive.")

    return relative_step


def _perturbation_sizes(
    point: Array,
    relative_step: float,
) -> Array:
    """Calculate scale-aware perturbation sizes."""

    return relative_step * np.maximum(1.0, np.abs(point))


def _evaluate_scalar(
    function: ScalarFunction,
    point: Array,
) -> float:
    """Evaluate and validate a scalar function."""

    value = np.asarray(function(point), dtype=float)

    if value.ndim != 0:
        raise ValueError("The differentiated function must return a scalar.")

    scalar_value = float(value)

    if not np.isfinite(scalar_value):
        raise ValueError(
            "The differentiated function returned a non-finite value."
        )

    return scalar_value


def finite_difference_jacobian(
    function: VectorFunction,
    point: Array,
    *,
    relative_step: float = 1.0e-6,
) -> Array:
    """Calculate a vector-function Jacobian using central differences."""

    point = _validate_point(point, "point")
    relative_step = _validate_step(relative_step)

    reference_output = np.asarray(function(point), dtype=float)

    if reference_output.ndim != 1:
        raise ValueError(
            "The differentiated vector function must return "
            "a one-dimensional array."
        )

    if not np.all(np.isfinite(reference_output)):
        raise ValueError(
            "The differentiated function returned non-finite values."
        )

    output_dimension = reference_output.size
    input_dimension = point.size

    jacobian = np.zeros(
        (output_dimension, input_dimension),
        dtype=float,
    )

    steps = _perturbation_sizes(point, relative_step)

    for index in range(input_dimension):
        perturbation = np.zeros(input_dimension, dtype=float)
        perturbation[index] = steps[index]

        output_plus = np.asarray(
            function(point + perturbation),
            dtype=float,
        )

        output_minus = np.asarray(
            function(point - perturbation),
            dtype=float,
        )

        if output_plus.shape != reference_output.shape:
            raise ValueError(
                "The vector-function output shape changed during "
                "finite differencing."
            )

        if output_minus.shape != reference_output.shape:
            raise ValueError(
                "The vector-function output shape changed during "
                "finite differencing."
            )

        jacobian[:, index] = (
            output_plus - output_minus
        ) / (2.0 * steps[index])

    if not np.all(np.isfinite(jacobian)):
        raise ValueError("The calculated Jacobian is non-finite.")

    return jacobian


def finite_difference_gradient(
    function: ScalarFunction,
    point: Array,
    *,
    relative_step: float = 1.0e-6,
) -> Array:
    """Calculate a scalar-function gradient using central differences."""

    point = _validate_point(point, "point")
    relative_step = _validate_step(relative_step)

    gradient = np.zeros(point.size, dtype=float)
    steps = _perturbation_sizes(point, relative_step)

    for index in range(point.size):
        perturbation = np.zeros(point.size, dtype=float)
        perturbation[index] = steps[index]

        value_plus = _evaluate_scalar(
            function,
            point + perturbation,
        )

        value_minus = _evaluate_scalar(
            function,
            point - perturbation,
        )

        gradient[index] = (
            value_plus - value_minus
        ) / (2.0 * steps[index])

    return gradient


def finite_difference_hessian(
    function: ScalarFunction,
    point: Array,
    *,
    relative_step: float = 1.0e-4,
) -> Array:
    """Calculate a scalar-function Hessian using central differences."""

    point = _validate_point(point, "point")
    relative_step = _validate_step(relative_step)

    dimension = point.size
    hessian = np.zeros((dimension, dimension), dtype=float)
    steps = _perturbation_sizes(point, relative_step)

    center_value = _evaluate_scalar(function, point)

    for row in range(dimension):
        row_perturbation = np.zeros(dimension, dtype=float)
        row_perturbation[row] = steps[row]

        value_plus = _evaluate_scalar(
            function,
            point + row_perturbation,
        )

        value_minus = _evaluate_scalar(
            function,
            point - row_perturbation,
        )

        hessian[row, row] = (
            value_plus
            - 2.0 * center_value
            + value_minus
        ) / steps[row] ** 2

        for column in range(row):
            column_perturbation = np.zeros(
                dimension,
                dtype=float,
            )
            column_perturbation[column] = steps[column]

            value_plus_plus = _evaluate_scalar(
                function,
                point + row_perturbation + column_perturbation,
            )

            value_plus_minus = _evaluate_scalar(
                function,
                point + row_perturbation - column_perturbation,
            )

            value_minus_plus = _evaluate_scalar(
                function,
                point - row_perturbation + column_perturbation,
            )

            value_minus_minus = _evaluate_scalar(
                function,
                point - row_perturbation - column_perturbation,
            )

            mixed_derivative = (
                value_plus_plus
                - value_plus_minus
                - value_minus_plus
                + value_minus_minus
            ) / (
                4.0
                * steps[row]
                * steps[column]
            )

            hessian[row, column] = mixed_derivative
            hessian[column, row] = mixed_derivative

    return 0.5 * (hessian + hessian.T)


def finite_difference_cross_hessian(
    function: TwoArgumentScalarFunction,
    first_point: Array,
    second_point: Array,
    *,
    relative_step: float = 1.0e-4,
) -> Array:
    """Calculate derivatives first with respect to the second argument.

    The returned matrix has shape

        (second_dimension, first_dimension).

    For a cost function ``cost(x, u)``, call this function with
    ``first_point=x`` and ``second_point=u`` to obtain ``C_ux``.
    """

    first_point = _validate_point(first_point, "first_point")
    second_point = _validate_point(second_point, "second_point")
    relative_step = _validate_step(relative_step)

    first_steps = _perturbation_sizes(
        first_point,
        relative_step,
    )

    second_steps = _perturbation_sizes(
        second_point,
        relative_step,
    )

    result = np.zeros(
        (second_point.size, first_point.size),
        dtype=float,
    )

    for second_index in range(second_point.size):
        second_perturbation = np.zeros(
            second_point.size,
            dtype=float,
        )
        second_perturbation[second_index] = second_steps[
            second_index
        ]

        for first_index in range(first_point.size):
            first_perturbation = np.zeros(
                first_point.size,
                dtype=float,
            )
            first_perturbation[first_index] = first_steps[
                first_index
            ]

            value_plus_plus = float(
                function(
                    first_point + first_perturbation,
                    second_point + second_perturbation,
                )
            )

            value_plus_minus = float(
                function(
                    first_point + first_perturbation,
                    second_point - second_perturbation,
                )
            )

            value_minus_plus = float(
                function(
                    first_point - first_perturbation,
                    second_point + second_perturbation,
                )
            )

            value_minus_minus = float(
                function(
                    first_point - first_perturbation,
                    second_point - second_perturbation,
                )
            )

            result[second_index, first_index] = (
                value_plus_plus
                - value_plus_minus
                - value_minus_plus
                + value_minus_minus
            ) / (
                4.0
                * second_steps[second_index]
                * first_steps[first_index]
            )

    if not np.all(np.isfinite(result)):
        raise ValueError(
            "The calculated cross Hessian contains non-finite values."
        )

    return result


def differentiate_dynamics(
    problem: BoxILQRProblem,
    state: Array,
    control: Array,
    k: int,
    *,
    relative_step: float = 1.0e-6,
) -> DynamicsDerivatives:
    """Calculate the discrete dynamics Jacobians F_x and F_u."""

    state = problem.validate_state(state)
    control = problem.validate_control(control)

    fx = finite_difference_jacobian(
        lambda perturbed_state: problem.step(
            perturbed_state,
            control,
            k,
        ),
        state,
        relative_step=relative_step,
    )

    fu = finite_difference_jacobian(
        lambda perturbed_control: problem.step(
            state,
            perturbed_control,
            k,
        ),
        control,
        relative_step=relative_step,
    )

    return DynamicsDerivatives(
        fx=fx,
        fu=fu,
    )


def differentiate_running_cost(
    problem: BoxILQRProblem,
    state: Array,
    control: Array,
    k: int,
    *,
    gradient_step: float = 1.0e-6,
    hessian_step: float = 1.0e-4,
) -> RunningCostDerivatives:
    """Calculate running-cost gradients and Hessians."""

    state = problem.validate_state(state)
    control = problem.validate_control(control)

    state_cost = lambda perturbed_state: (
        problem.evaluate_running_cost(
            perturbed_state,
            control,
            k,
        )
    )

    control_cost = lambda perturbed_control: (
        problem.evaluate_running_cost(
            state,
            perturbed_control,
            k,
        )
    )

    joint_cost = lambda perturbed_state, perturbed_control: (
        problem.evaluate_running_cost(
            perturbed_state,
            perturbed_control,
            k,
        )
    )

    lx = finite_difference_gradient(
        state_cost,
        state,
        relative_step=gradient_step,
    )

    lu = finite_difference_gradient(
        control_cost,
        control,
        relative_step=gradient_step,
    )

    lxx = finite_difference_hessian(
        state_cost,
        state,
        relative_step=hessian_step,
    )

    luu = finite_difference_hessian(
        control_cost,
        control,
        relative_step=hessian_step,
    )

    lux = finite_difference_cross_hessian(
        joint_cost,
        state,
        control,
        relative_step=hessian_step,
    )

    return RunningCostDerivatives(
        lx=lx,
        lu=lu,
        lxx=lxx,
        luu=luu,
        lux=lux,
    )


def differentiate_terminal_cost(
    problem: BoxILQRProblem,
    state: Array,
    *,
    gradient_step: float = 1.0e-6,
    hessian_step: float = 1.0e-4,
) -> TerminalCostDerivatives:
    """Calculate terminal-cost gradient and Hessian."""

    state = problem.validate_state(state)

    terminal_function = lambda perturbed_state: (
        problem.evaluate_terminal_cost(perturbed_state)
    )

    phix = finite_difference_gradient(
        terminal_function,
        state,
        relative_step=gradient_step,
    )

    phixx = finite_difference_hessian(
        terminal_function,
        state,
        relative_step=hessian_step,
    )

    return TerminalCostDerivatives(
        phix=phix,
        phixx=phixx,
    )
