"""Open-loop and feedback simulation utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, TypeAlias

import numpy as np

from boxilqr.problem import Array, BoxILQRProblem
from boxilqr.solution import BoxILQRSolution


MeasurementNoise: TypeAlias = Callable[[Array, int], Array]
ControlNoise: TypeAlias = Callable[[Array, int], Array]
ProcessNoise: TypeAlias = Callable[[Array, Array, int], Array]

StateDifference: TypeAlias = Callable[
    [Array, Array],
    Array,
]


@dataclass(frozen=True)
class SimulationResult:
    """Result of an open-loop or feedback simulation."""

    states: Array
    measured_states: Array
    commanded_controls: Array
    applied_controls: Array
    feedback_enabled: bool

    def __post_init__(self) -> None:
        states = np.asarray(self.states, dtype=float)
        measured_states = np.asarray(
            self.measured_states,
            dtype=float,
        )
        commanded_controls = np.asarray(
            self.commanded_controls,
            dtype=float,
        )
        applied_controls = np.asarray(
            self.applied_controls,
            dtype=float,
        )

        if states.ndim != 2:
            raise ValueError("states must be two-dimensional.")

        if measured_states.ndim != 2:
            raise ValueError(
                "measured_states must be two-dimensional."
            )

        if commanded_controls.ndim != 2:
            raise ValueError(
                "commanded_controls must be two-dimensional."
            )

        if applied_controls.ndim != 2:
            raise ValueError(
                "applied_controls must be two-dimensional."
            )

        horizon = applied_controls.shape[0]
        state_dim = states.shape[1]

        if states.shape[0] != horizon + 1:
            raise ValueError(
                "states must contain horizon + 1 samples."
            )

        if measured_states.shape != (
            horizon,
            state_dim,
        ):
            raise ValueError(
                "measured_states has an incorrect shape."
            )

        if commanded_controls.shape != (
            applied_controls.shape
        ):
            raise ValueError(
                "commanded_controls and applied_controls "
                "must have the same shape."
            )

        for name, array in (
            ("states", states),
            ("measured_states", measured_states),
            ("commanded_controls", commanded_controls),
            ("applied_controls", applied_controls),
        ):
            if not np.all(np.isfinite(array)):
                raise ValueError(
                    f"{name} contains non-finite values."
                )

        object.__setattr__(
            self,
            "states",
            states.copy(),
        )
        object.__setattr__(
            self,
            "measured_states",
            measured_states.copy(),
        )
        object.__setattr__(
            self,
            "commanded_controls",
            commanded_controls.copy(),
        )
        object.__setattr__(
            self,
            "applied_controls",
            applied_controls.copy(),
        )

    @property
    def horizon(self) -> int:
        """Number of simulated control intervals."""

        return self.applied_controls.shape[0]

    @property
    def state_dim(self) -> int:
        """Dimension of the simulated state."""

        return self.states.shape[1]

    @property
    def control_dim(self) -> int:
        """Dimension of the simulated control."""

        return self.applied_controls.shape[1]

    @property
    def controls(self) -> Array:
        """Alias for the controls applied to the dynamics."""

        return self.applied_controls

    def tracking_errors(
        self,
        solution: BoxILQRSolution,
    ) -> Array:
        """Calculate errors relative to the nominal trajectory."""

        if self.states.shape != solution.states.shape:
            raise ValueError(
                "The simulation and solution state trajectories "
                "have different shapes."
            )

        return self.states - solution.states


def _validate_noise(
    noise: Array,
    expected_shape: tuple[int, ...],
    name: str,
) -> Array:
    """Validate a supplied disturbance or noise vector."""

    noise = np.asarray(noise, dtype=float)

    if noise.shape != expected_shape:
        raise ValueError(
            f"{name} must have shape {expected_shape}, "
            f"but received {noise.shape}."
        )

    if not np.all(np.isfinite(noise)):
        raise ValueError(
            f"{name} contains non-finite values."
        )

    return noise


def simulate_closed_loop(
    problem: BoxILQRProblem,
    solution: BoxILQRSolution,
    *,
    initial_state: Array | None = None,
    feedback: bool = True,
    measurement_noise: MeasurementNoise | None = None,
    control_noise: ControlNoise | None = None,
    process_noise: ProcessNoise | None = None,
    state_difference: StateDifference | None = None,
    project_controls: bool = True,
) -> SimulationResult:
    """Simulate a nominal or feedback-controlled trajectory.

    Noise functions return additive perturbations. Their signatures are

        measurement_noise(true_state, k)
        control_noise(commanded_control, k)
        process_noise(true_state, applied_control, k)

    Process noise is added after the discrete dynamics step.
    """

    if solution.horizon != problem.horizon:
        raise ValueError(
            "The solution and problem horizons do not match."
        )

    if solution.state_dim != problem.state_dim:
        raise ValueError(
            "The solution and problem state dimensions "
            "do not match."
        )

    if solution.control_dim != problem.control_dim:
        raise ValueError(
            "The solution and problem control dimensions "
            "do not match."
        )

    if initial_state is None:
        initial_state = solution.states[0]

    initial_state = problem.validate_state(
        initial_state
    )

    if state_difference is None:
        def state_difference(
            state: Array,
            nominal_state: Array,
        ) -> Array:
            return state - nominal_state

    states = np.zeros(
        (problem.horizon + 1, problem.state_dim),
        dtype=float,
    )

    measured_states = np.zeros(
        (problem.horizon, problem.state_dim),
        dtype=float,
    )

    commanded_controls = np.zeros(
        (problem.horizon, problem.control_dim),
        dtype=float,
    )

    applied_controls = np.zeros(
        (problem.horizon, problem.control_dim),
        dtype=float,
    )

    states[0] = initial_state

    for k in range(problem.horizon):
        true_state = states[k]

        measured_state = true_state.copy()

        if measurement_noise is not None:
            measurement_perturbation = _validate_noise(
                measurement_noise(
                    true_state.copy(),
                    k,
                ),
                (problem.state_dim,),
                "measurement_noise",
            )

            measured_state += measurement_perturbation

        measured_state = problem.validate_state(
            measured_state
        )

        measured_states[k] = measured_state

        if feedback:
            state_error = np.asarray(
                state_difference(
                    measured_state,
                    solution.states[k],
                ),
                dtype=float,
            )

            if state_error.shape != (
                problem.state_dim,
            ):
                raise ValueError(
                    "state_difference returned an "
                    "incorrect shape."
                )

            commanded_control = (
                solution.controls[k]
                + solution.feedback_gains[k]
                @ state_error
            )
        else:
            commanded_control = (
                solution.controls[k].copy()
            )

        commanded_control = problem.validate_control(
            commanded_control
        )

        commanded_controls[k] = commanded_control

        applied_control = commanded_control.copy()

        if control_noise is not None:
            control_perturbation = _validate_noise(
                control_noise(
                    commanded_control.copy(),
                    k,
                ),
                (problem.control_dim,),
                "control_noise",
            )

            applied_control += control_perturbation

        if (
            project_controls
            and problem.control_bounds is not None
        ):
            applied_control = (
                problem.control_bounds.project(
                    applied_control
                )
            )

        applied_control = problem.validate_control(
            applied_control
        )

        applied_controls[k] = applied_control

        next_state = problem.step(
            true_state,
            applied_control,
            k,
        )

        if process_noise is not None:
            process_perturbation = _validate_noise(
                process_noise(
                    true_state.copy(),
                    applied_control.copy(),
                    k,
                ),
                (problem.state_dim,),
                "process_noise",
            )

            next_state += process_perturbation

        states[k + 1] = problem.validate_state(
            next_state
        )

    return SimulationResult(
        states=states,
        measured_states=measured_states,
        commanded_controls=commanded_controls,
        applied_controls=applied_controls,
        feedback_enabled=feedback,
    )
