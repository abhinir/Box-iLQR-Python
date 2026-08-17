"""Tests for convergence diagnostics."""

import matplotlib
import numpy as np

matplotlib.use("Agg")

from boxilqr import (
    IterationRecord,
    extract_convergence_history,
    plot_convergence,
)


def create_history() -> tuple[IterationRecord, ...]:
    return (
        IterationRecord(
            outer_iteration=0,
            inner_iteration=0,
            cost=10.0,
            regularization=1.0e-4,
            accepted=True,
            cost_change=-5.0,
            expected_reduction=4.5,
            step_size=1.0,
            feedforward_norm=0.5,
            state_barrier=np.array([0.1, 0.1]),
            control_barrier=np.array([0.1]),
        ),
        IterationRecord(
            outer_iteration=1,
            inner_iteration=0,
            cost=4.0,
            regularization=1.0e-5,
            accepted=True,
            cost_change=-1.0,
            expected_reduction=0.9,
            step_size=0.5,
            feedforward_norm=0.05,
            state_barrier=np.array([0.02, 0.02]),
            control_barrier=np.array([0.02]),
        ),
    )


def test_extract_convergence_history() -> None:
    data = extract_convergence_history(
        create_history()
    )

    assert data.size == 2

    np.testing.assert_allclose(
        data.cost,
        np.array([10.0, 4.0]),
    )

    np.testing.assert_allclose(
        data.step_size,
        np.array([1.0, 0.5]),
    )

    np.testing.assert_allclose(
        data.control_barrier_norm,
        np.array([0.1, 0.02]),
    )


def test_plot_convergence(tmp_path) -> None:
    import matplotlib.pyplot as plt

    output_path = (
        tmp_path
        / "convergence.png"
    )

    figure = plot_convergence(
        create_history(),
        save_path=output_path,
        show=False,
    )

    assert output_path.exists()
    assert len(figure.axes) == 6

    plt.close(figure)
