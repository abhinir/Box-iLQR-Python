"""Numerical integration utilities for continuous dynamics."""

from __future__ import annotations

from typing import Callable, Literal, TypeAlias

import numpy as np

from boxilqr.problem import Array, Dynamics


ContinuousDynamics: TypeAlias = Callable[[float, Array, Array], Array]
IntegrationMethod: TypeAlias = Literal["euler", "rk4"]


def _validate_vector(value: Array, name: str) -> Array:
    """Convert and validate a one-dimensional numerical vector."""

    vector = np.asarray(value, dtype=float)

    if vector.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional array.")

    if vector.size == 0:
        raise ValueError(f"{name} cannot be empty.")

    if not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must contain only finite values.")

    return vector


def _validate_time_step(dt: float) -> float:
    """Validate a positive integration time step."""

    dt = float(dt)

    if not np.isfinite(dt):
        raise ValueError("dt must be finite.")

    if dt <= 0.0:
        raise ValueError("dt must be strictly positive.")

    return dt


def _evaluate_ode(
    ode: ContinuousDynamics,
    time: float,
    state: Array,
    control: Array,
) -> Array:
    """Evaluate and validate a continuous dynamics function."""

    derivative = np.asarray(
        ode(float(time), state, control),
        dtype=float,
    )

    if derivative.shape != state.shape:
        raise ValueError(
            "The dynamics derivative must have the same shape as "
            f"the state. Expected {state.shape}, but received "
            f"{derivative.shape}."
        )

    if not np.all(np.isfinite(derivative)):
        raise ValueError(
            "The continuous dynamics returned non-finite values."
        )

    return derivative


def euler_step(
    ode: ContinuousDynamics,
    time: float,
    state: Array,
    control: Array,
    dt: float,
) -> Array:
    """Propagate one step using the forward Euler method."""

    state = _validate_vector(state, "state")
    control = _validate_vector(control, "control")
    dt = _validate_time_step(dt)

    derivative = _evaluate_ode(
        ode,
        time,
        state,
        control,
    )

    next_state = state + dt * derivative

    return _validate_vector(next_state, "next_state")


def rk4_step(
    ode: ContinuousDynamics,
    time: float,
    state: Array,
    control: Array,
    dt: float,
) -> Array:
    """Propagate one step using fourth-order Runge–Kutta integration.

    The control is held constant over the entire integration interval.
    """

    state = _validate_vector(state, "state")
    control = _validate_vector(control, "control")
    dt = _validate_time_step(dt)

    k1 = _evaluate_ode(
        ode,
        time,
        state,
        control,
    )

    k2 = _evaluate_ode(
        ode,
        time + 0.5 * dt,
        state + 0.5 * dt * k1,
        control,
    )

    k3 = _evaluate_ode(
        ode,
        time + 0.5 * dt,
        state + 0.5 * dt * k2,
        control,
    )

    k4 = _evaluate_ode(
        ode,
        time + dt,
        state + dt * k3,
        control,
    )

    next_state = state + (dt / 6.0) * (
        k1 + 2.0 * k2 + 2.0 * k3 + k4
    )

    return _validate_vector(next_state, "next_state")


def discretize_dynamics(
    ode: ContinuousDynamics,
    dt: float,
    *,
    method: IntegrationMethod = "rk4",
    initial_time: float = 0.0,
    substeps: int = 1,
) -> Dynamics:
    """Convert continuous dynamics into discrete dynamics.

    Parameters
    ----------
    ode
        Continuous dynamics with signature ``ode(t, x, u)``.
    dt
        Time between consecutive control points.
    method
        Either ``"euler"`` or ``"rk4"``.
    initial_time
        Time corresponding to index ``k = 0``.
    substeps
        Number of integration substeps within one control interval.

    Returns
    -------
    Dynamics
        Discrete dynamics with signature ``dynamics(x, u, k)``.
    """

    if not callable(ode):
        raise TypeError("ode must be callable.")

    dt = _validate_time_step(dt)
    initial_time = float(initial_time)

    if not np.isfinite(initial_time):
        raise ValueError("initial_time must be finite.")

    if method not in ("euler", "rk4"):
        raise ValueError("method must be either 'euler' or 'rk4'.")

    if not isinstance(substeps, int) or substeps <= 0:
        raise ValueError("substeps must be a positive integer.")

    integration_step = dt / substeps

    if method == "euler":
        step_function = euler_step
    else:
        step_function = rk4_step

    def discrete_dynamics(
        state: Array,
        control: Array,
        k: int,
    ) -> Array:
        """Propagate one complete control interval."""

        if not isinstance(k, (int, np.integer)):
            raise TypeError("The time index k must be an integer.")

        current_state = _validate_vector(state, "state").copy()
        control_vector = _validate_vector(control, "control")

        interval_start = initial_time + int(k) * dt

        for substep_index in range(substeps):
            substep_time = (
                interval_start
                + substep_index * integration_step
            )

            current_state = step_function(
                ode,
                substep_time,
                current_state,
                control_vector,
                integration_step,
            )

        return current_state

    return discrete_dynamics
