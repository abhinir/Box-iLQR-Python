"""Tests for solver options and solution structures."""

import numpy as np
import pytest

from boxilqr import (
    BarrierOptions,
    Bounds,
    BoxILQROptions,
    BoxILQRSolution,
    ILQROptions,
    SolverStatus,
)


def test_default_options() -> None:
    options = BoxILQROptions()

    assert options.ilqr.max_iterations == 200
    assert options.barrier.tolerance == 1.0e-6


def test_invalid_options_are_rejected() -> None:
    with pytest.raises(ValueError):
        ILQROptions(line_search_decay=1.0)

    with pytest.raises(ValueError):
        BarrierOptions(control_reduction_factor=0.0)


def test_solution_feedback_control() -> None:
    states = np.zeros((4, 2))
    controls = np.zeros((3, 1))

    feedback_gains = np.zeros((3, 1, 2))
    feedback_gains[:, 0, 0] = -1.0

    feedforward_terms = np.zeros((3, 1))

    solution = BoxILQRSolution(
        states=states,
        controls=controls,
        feedback_gains=feedback_gains,
        feedforward_terms=feedforward_terms,
        cost=1.0,
        status=SolverStatus.CONVERGED,
        message="Test solution.",
    )

    bounds = Bounds(
        lower=np.array([-0.1]),
        upper=np.array([0.1]),
    )

    control = solution.feedback_control(
        state=np.array([0.2, 0.0]),
        k=0,
        control_bounds=bounds,
    )

    np.testing.assert_allclose(
        control,
        np.array([-0.1]),
    )

    assert solution.success
    assert solution.horizon == 3
    assert solution.state_dim == 2
    assert solution.control_dim == 1
