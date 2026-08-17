"""Tests for logarithmic box barriers."""

import numpy as np
import pytest

from boxilqr import (
    Bounds,
    InfeasiblePointError,
    evaluate_box_barrier,
    expand_barrier_parameters,
)


def test_expand_scalar_barrier_parameter() -> None:
    parameters = expand_barrier_parameters(
        parameters=0.5,
        dimension=3,
    )

    np.testing.assert_allclose(
        parameters,
        np.array([0.5, 0.5, 0.5]),
    )


def test_two_sided_barrier_derivatives() -> None:
    bounds = Bounds(
        lower=np.array([0.0]),
        upper=np.array([2.0]),
    )

    evaluation = evaluate_box_barrier(
        value=np.array([0.5]),
        bounds=bounds,
        parameters=np.array([0.3]),
    )

    expected_value = -0.3 * (
        np.log(0.5) + np.log(1.5)
    )

    expected_gradient = np.array([
        -0.3 / 0.5 + 0.3 / 1.5
    ])

    expected_hessian = np.array([
        0.3 / 0.5**2 + 0.3 / 1.5**2
    ])

    np.testing.assert_allclose(
        evaluation.value,
        expected_value,
    )

    np.testing.assert_allclose(
        evaluation.gradient,
        expected_gradient,
    )

    np.testing.assert_allclose(
        evaluation.hessian_diagonal,
        expected_hessian,
    )


def test_one_sided_and_unbounded_components() -> None:
    bounds = Bounds(
        lower=np.array([0.0, -np.inf]),
        upper=np.array([np.inf, np.inf]),
    )

    evaluation = evaluate_box_barrier(
        value=np.array([2.0, 100.0]),
        bounds=bounds,
        parameters=np.array([0.5, 10.0]),
    )

    expected_value = -0.5 * np.log(2.0)

    np.testing.assert_allclose(
        evaluation.value,
        expected_value,
    )

    np.testing.assert_allclose(
        evaluation.gradient,
        np.array([-0.25, 0.0]),
    )

    np.testing.assert_allclose(
        evaluation.hessian_diagonal,
        np.array([0.125, 0.0]),
    )


def test_infeasible_point_is_rejected() -> None:
    bounds = Bounds(
        lower=np.array([0.0]),
        upper=np.array([1.0]),
    )

    with pytest.raises(InfeasiblePointError) as error:
        evaluate_box_barrier(
            value=np.array([0.0]),
            bounds=bounds,
            parameters=1.0,
        )

    assert error.value.index == 0
    assert error.value.side == "lower"


def test_barrier_gradient_using_finite_difference() -> None:
    bounds = Bounds(
        lower=np.array([-1.0, -2.0]),
        upper=np.array([2.0, 3.0]),
    )

    parameters = np.array([0.7, 1.2])
    point = np.array([0.4, -0.5])
    step = 1.0e-6

    analytical = evaluate_box_barrier(
        point,
        bounds,
        parameters,
    )

    numerical_gradient = np.zeros(2)

    for index in range(2):
        perturbation = np.zeros(2)
        perturbation[index] = step

        value_plus = evaluate_box_barrier(
            point + perturbation,
            bounds,
            parameters,
        ).value

        value_minus = evaluate_box_barrier(
            point - perturbation,
            bounds,
            parameters,
        ).value

        numerical_gradient[index] = (
            value_plus - value_minus
        ) / (2.0 * step)

    np.testing.assert_allclose(
        analytical.gradient,
        numerical_gradient,
        rtol=1.0e-6,
        atol=1.0e-8,
    )
