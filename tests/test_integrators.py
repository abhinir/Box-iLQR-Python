"""Tests for numerical integration."""

import numpy as np
import pytest

from boxilqr import (
    discretize_dynamics,
    euler_step,
    rk4_step,
)


def test_euler_step() -> None:
    def decay_ode(
        time: float,
        state: np.ndarray,
        control: np.ndarray,
    ) -> np.ndarray:
        del time, control
        return -state

    next_state = euler_step(
        decay_ode,
        time=0.0,
        state=np.array([1.0]),
        control=np.array([0.0]),
        dt=0.1,
    )

    np.testing.assert_allclose(
        next_state,
        np.array([0.9]),
    )


def test_rk4_exponential_decay() -> None:
    def decay_ode(
        time: float,
        state: np.ndarray,
        control: np.ndarray,
    ) -> np.ndarray:
        del time, control
        return -state

    next_state = rk4_step(
        decay_ode,
        time=0.0,
        state=np.array([1.0]),
        control=np.array([0.0]),
        dt=0.1,
    )

    np.testing.assert_allclose(
        next_state,
        np.array([np.exp(-0.1)]),
        rtol=1.0e-6,
    )


def test_rk4_double_integrator() -> None:
    def double_integrator_ode(
        time: float,
        state: np.ndarray,
        control: np.ndarray,
    ) -> np.ndarray:
        del time

        position, velocity = state
        acceleration = control[0]

        return np.array([
            velocity,
            acceleration,
        ])

    next_state = rk4_step(
        double_integrator_ode,
        time=0.0,
        state=np.array([0.0, 0.0]),
        control=np.array([2.0]),
        dt=0.1,
    )

    np.testing.assert_allclose(
        next_state,
        np.array([0.01, 0.2]),
        atol=1.0e-14,
    )


def test_discretized_dynamics() -> None:
    def double_integrator_ode(
        time: float,
        state: np.ndarray,
        control: np.ndarray,
    ) -> np.ndarray:
        del time

        return np.array([
            state[1],
            control[0],
        ])

    dynamics = discretize_dynamics(
        double_integrator_ode,
        dt=0.1,
        method="rk4",
        substeps=2,
    )

    next_state = dynamics(
        np.array([0.0, 0.0]),
        np.array([2.0]),
        0,
    )

    np.testing.assert_allclose(
        next_state,
        np.array([0.01, 0.2]),
        atol=1.0e-14,
    )


def test_invalid_time_step_is_rejected() -> None:
    def ode(
        time: float,
        state: np.ndarray,
        control: np.ndarray,
    ) -> np.ndarray:
        del time, control
        return state

    with pytest.raises(ValueError):
        discretize_dynamics(ode, dt=0.0)
