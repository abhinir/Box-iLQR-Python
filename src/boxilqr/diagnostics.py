"""Convergence diagnostics for Box-iLQR."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from boxilqr.problem import Array
from boxilqr.solution import IterationRecord


@dataclass(frozen=True)
class ConvergenceHistory:
    """Array representation of Box-iLQR iteration history."""

    outer_iteration: Array
    inner_iteration: Array

    cost: Array
    accepted: Array

    cost_change: Array
    expected_reduction: Array
    step_size: Array

    regularization: Array
    feedforward_norm: Array

    state_barrier_norm: Array
    control_barrier_norm: Array

    @property
    def size(self) -> int:
        """Number of stored iterations."""

        return self.cost.size


def _optional_scalar(
    value: float | None,
) -> float:
    """Convert an optional scalar to a float or NaN."""

    if value is None:
        return np.nan

    return float(value)


def _parameter_norm(
    value: Array | None,
) -> float:
    """Calculate a barrier-parameter norm."""

    if value is None:
        return np.nan

    array = np.asarray(value, dtype=float)

    if array.size == 0:
        return np.nan

    return float(np.linalg.norm(array))


def extract_convergence_history(
    history: Sequence[IterationRecord],
) -> ConvergenceHistory:
    """Convert iteration records into numerical arrays."""

    records = tuple(history)

    return ConvergenceHistory(
        outer_iteration=np.array(
            [
                record.outer_iteration
                for record in records
            ],
            dtype=int,
        ),
        inner_iteration=np.array(
            [
                record.inner_iteration
                for record in records
            ],
            dtype=int,
        ),
        cost=np.array(
            [record.cost for record in records],
            dtype=float,
        ),
        accepted=np.array(
            [record.accepted for record in records],
            dtype=bool,
        ),
        cost_change=np.array(
            [
                _optional_scalar(record.cost_change)
                for record in records
            ],
            dtype=float,
        ),
        expected_reduction=np.array(
            [
                _optional_scalar(
                    record.expected_reduction
                )
                for record in records
            ],
            dtype=float,
        ),
        step_size=np.array(
            [
                _optional_scalar(record.step_size)
                for record in records
            ],
            dtype=float,
        ),
        regularization=np.array(
            [
                record.regularization
                for record in records
            ],
            dtype=float,
        ),
        feedforward_norm=np.array(
            [
                _optional_scalar(
                    record.feedforward_norm
                )
                for record in records
            ],
            dtype=float,
        ),
        state_barrier_norm=np.array(
            [
                _parameter_norm(
                    record.state_barrier
                )
                for record in records
            ],
            dtype=float,
        ),
        control_barrier_norm=np.array(
            [
                _parameter_norm(
                    record.control_barrier
                )
                for record in records
            ],
            dtype=float,
        ),
    )


def _positive_values(values: Array) -> Array:
    """Replace nonpositive values with NaN for logarithmic plots."""

    values = np.asarray(values, dtype=float).copy()

    invalid = (
        ~np.isfinite(values)
        | (values <= 0.0)
    )

    values[invalid] = np.nan

    return values


def plot_convergence(
    history: Sequence[IterationRecord],
    *,
    save_path: str | Path | None = None,
    show: bool = True,
) -> object:
    """Plot the Box-iLQR convergence history."""

    # Imported here so plotting remains an optional dependency.
    import matplotlib.pyplot as plt

    data = extract_convergence_history(history)

    if data.size == 0:
        raise ValueError(
            "The iteration history is empty. Set "
            "store_iteration_history=True before solving."
        )

    iteration = np.arange(1, data.size + 1)

    figure, axes_array = plt.subplots(
        3,
        2,
        figsize=(12, 11),
        sharex=True,
    )

    axes = axes_array.ravel()

    # ---------------------------------------------------------
    # Augmented cost
    # ---------------------------------------------------------
    axes[0].plot(
        iteration,
        data.cost,
        color="tab:blue",
        linewidth=1.8,
        marker="o",
        markersize=3,
        label="Augmented cost",
    )

    rejected = ~data.accepted

    if np.any(rejected):
        axes[0].scatter(
            iteration[rejected],
            data.cost[rejected],
            color="tab:red",
            marker="x",
            s=35,
            label="Rejected/terminal iteration",
        )

    axes[0].set_ylabel("Cost")
    axes[0].set_title("Barrier-Augmented Cost")
    axes[0].legend()

    # ---------------------------------------------------------
    # Actual and predicted reductions
    # ---------------------------------------------------------
    axes[1].semilogy(
        iteration,
        _positive_values(
            np.abs(data.cost_change)
        ),
        color="tab:green",
        linewidth=1.8,
        marker="o",
        markersize=3,
        label=r"Actual $|\Delta J|$",
    )

    axes[1].semilogy(
        iteration,
        _positive_values(
            data.expected_reduction
        ),
        color="tab:purple",
        linewidth=1.5,
        linestyle="--",
        label="Predicted reduction",
    )

    axes[1].set_ylabel("Cost reduction")
    axes[1].set_title("Actual and Predicted Reduction")
    axes[1].legend()

    # ---------------------------------------------------------
    # Line-search step
    # ---------------------------------------------------------
    axes[2].plot(
        iteration,
        data.step_size,
        color="tab:orange",
        linewidth=1.8,
        marker="o",
        markersize=3,
    )

    axes[2].set_ylabel(r"Step size $\alpha$")
    axes[2].set_ylim(-0.05, 1.05)
    axes[2].set_title("Line-Search Step Size")

    # ---------------------------------------------------------
    # Regularization
    # ---------------------------------------------------------
    axes[3].semilogy(
        iteration,
        _positive_values(
            data.regularization
        ),
        color="tab:red",
        linewidth=1.8,
        marker="o",
        markersize=3,
    )

    axes[3].set_ylabel("Regularization")
    axes[3].set_title(r"$Q_{uu}$ Regularization")

    # ---------------------------------------------------------
    # Feedforward norm
    # ---------------------------------------------------------
    axes[4].semilogy(
        iteration,
        _positive_values(
            data.feedforward_norm
        ),
        color="tab:brown",
        linewidth=1.8,
        marker="o",
        markersize=3,
    )

    axes[4].set_xlabel("Stored iteration")
    axes[4].set_ylabel(
        r"$\max_k\|k_k\|_\infty$"
    )
    axes[4].set_title("Feedforward Correction")

    # ---------------------------------------------------------
    # Barrier parameters
    # ---------------------------------------------------------
    state_barrier_values = _positive_values(
        data.state_barrier_norm
    )

    control_barrier_values = _positive_values(
        data.control_barrier_norm
    )

    if np.any(np.isfinite(state_barrier_values)):
        axes[5].semilogy(
            iteration,
            state_barrier_values,
            color="tab:blue",
            linewidth=1.8,
            marker="o",
            markersize=3,
            label=r"State $\|\mu\|$",
        )

    if np.any(np.isfinite(control_barrier_values)):
        axes[5].semilogy(
            iteration,
            control_barrier_values,
            color="tab:red",
            linewidth=1.8,
            marker="s",
            markersize=3,
            label=r"Control $\|\sigma\|$",
        )

    axes[5].set_xlabel("Stored iteration")
    axes[5].set_ylabel("Barrier norm")
    axes[5].set_title("Barrier Continuation")
    axes[5].legend()

    # Mark transitions between outer iterations.
    transitions = np.flatnonzero(
        np.diff(data.outer_iteration) != 0
    )

    for transition in transitions:
        boundary = transition + 1.5

        for axis in axes:
            axis.axvline(
                boundary,
                color="0.7",
                linestyle=":",
                linewidth=0.8,
            )

    for axis in axes:
        axis.grid(True, alpha=0.3)

    figure.suptitle(
        "Box-iLQR Convergence History",
        fontsize=14,
    )

    figure.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        figure.savefig(
            save_path,
            dpi=200,
            bbox_inches="tight",
        )

    if show:
        plt.show()

    return figure
