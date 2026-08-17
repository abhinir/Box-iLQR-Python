"""Inner and outer solvers for Box-iLQR."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from boxilqr.backward_pass import (
    BackwardPassError,
    BackwardPassResult,
    backward_pass,
)
from boxilqr.barriers import (
    InfeasiblePointError,
    expand_barrier_parameters,
)
from boxilqr.forward_pass import LineSearchResult, line_search
from boxilqr.options import (
    BoxILQROptions,
    ILQROptions,
    ScalarOrArray,
)
from boxilqr.problem import (
    Array,
    Bounds,
    BoxILQRProblem,
)
from boxilqr.rollout import RolloutResult, rollout
from boxilqr.solution import (
    BoxILQRSolution,
    IterationRecord,
    SolverStatus,
)


@dataclass(frozen=True)
class ILQRSubproblemResult:
    """Result of solving one fixed-barrier iLQR subproblem."""

    rollout: RolloutResult | None
    backward_result: BackwardPassResult | None

    status: SolverStatus
    message: str

    iterations: int
    regularization: float

    history: tuple[IterationRecord, ...] = ()

    failed_constraint_type: str | None = None
    failed_constraint_index: int | None = None
    failed_constraint_side: str | None = None

    @property
    def success(self) -> bool:
        """Whether the fixed-barrier problem converged."""

        return self.status is SolverStatus.CONVERGED


def _copy_parameter(
    parameter: ScalarOrArray | None,
) -> Array | None:
    """Create an array copy for the iteration history."""

    if parameter is None:
        return None

    array = np.asarray(parameter, dtype=float)

    if array.ndim == 0:
        return array.reshape(1).copy()

    return array.copy()


def _failure_information(
    line_search_result: LineSearchResult,
) -> tuple[str | None, int | None, str | None]:
    """Extract the most recent constraint violation."""

    for attempt in reversed(line_search_result.attempts):
        if attempt.constraint_type is not None:
            return (
                attempt.constraint_type,
                attempt.constraint_index,
                attempt.constraint_side,
            )

    return None, None, None


def _error_information(
    error: InfeasiblePointError,
) -> tuple[str | None, int | None, str | None]:
    """Extract constraint information from an infeasibility error."""

    variable_name = error.variable_name.lower()

    if "control" in variable_name:
        constraint_type = "control"
    elif "state" in variable_name:
        constraint_type = "state"
    else:
        constraint_type = None

    return (
        constraint_type,
        error.index,
        error.side,
    )


def solve_ilqr_subproblem(
    problem: BoxILQRProblem,
    initial_state: Array,
    initial_controls: Array,
    *,
    state_barrier: ScalarOrArray | None = None,
    control_barrier: ScalarOrArray | None = None,
    minimum_margin: float = 1.0e-12,
    options: ILQROptions | None = None,
    outer_iteration: int = 0,
    store_history: bool = True,
    verbose: bool = False,
) -> ILQRSubproblemResult:
    """Solve an iLQR problem for fixed barrier parameters."""

    if options is None:
        options = ILQROptions()

    initial_controls = np.asarray(
        initial_controls,
        dtype=float,
    )

    expected_control_shape = (
        problem.horizon,
        problem.control_dim,
    )

    if initial_controls.shape != expected_control_shape:
        raise ValueError(
            f"initial_controls must have shape "
            f"{expected_control_shape}, but received "
            f"{initial_controls.shape}."
        )

    try:
        current_rollout = rollout(
            problem=problem,
            initial_state=initial_state,
            controls=initial_controls,
            state_barrier=state_barrier,
            control_barrier=control_barrier,
            minimum_margin=minimum_margin,
        )
    except InfeasiblePointError as error:
        (
            constraint_type,
            constraint_index,
            constraint_side,
        ) = _error_information(error)

        return ILQRSubproblemResult(
            rollout=None,
            backward_result=None,
            status=SolverStatus.INFEASIBLE_INITIAL_TRAJECTORY,
            message=str(error),
            iterations=0,
            regularization=options.regularization_initial,
            failed_constraint_type=constraint_type,
            failed_constraint_index=constraint_index,
            failed_constraint_side=constraint_side,
        )

    history: list[IterationRecord] = []

    regularization = options.regularization_initial
    last_backward_result: BackwardPassResult | None = None

    last_constraint_type: str | None = None
    last_constraint_index: int | None = None
    last_constraint_side: str | None = None

    state_barrier_history = _copy_parameter(
        state_barrier
    )

    control_barrier_history = _copy_parameter(
        control_barrier
    )

    def compute_backward_pass(
        trajectory: RolloutResult,
        starting_regularization: float,
    ) -> tuple[
        BackwardPassResult | None,
        float,
        BackwardPassError | None,
    ]:
        """Retry the backward pass with increasing regularization."""

        trial_regularization = starting_regularization

        while True:
            try:
                result = backward_pass(
                    problem=problem,
                    states=trajectory.states,
                    controls=trajectory.controls,
                    state_barrier=state_barrier,
                    control_barrier=control_barrier,
                    regularization=trial_regularization,
                    minimum_margin=minimum_margin,
                )

                return result, trial_regularization, None

            except BackwardPassError as error:
                if (
                    trial_regularization
                    >= options.regularization_maximum
                ):
                    return (
                        None,
                        trial_regularization,
                        error,
                    )

                new_regularization = min(
                    options.regularization_maximum,
                    trial_regularization
                    * options.regularization_increase,
                )

                if new_regularization <= trial_regularization:
                    return (
                        None,
                        trial_regularization,
                        error,
                    )

                trial_regularization = new_regularization

    for inner_iteration in range(options.max_iterations):
        (
            backward_result,
            regularization,
            backward_error,
        ) = compute_backward_pass(
            current_rollout,
            regularization,
        )

        if backward_result is None:
            message = (
                "The backward pass failed at the maximum "
                "regularization."
            )

            if backward_error is not None:
                message = f"{message} {backward_error}"

            return ILQRSubproblemResult(
                rollout=current_rollout,
                backward_result=None,
                status=SolverStatus.REGULARIZATION_LIMIT,
                message=message,
                iterations=inner_iteration + 1,
                regularization=regularization,
                history=tuple(history),
                failed_constraint_type=last_constraint_type,
                failed_constraint_index=last_constraint_index,
                failed_constraint_side=last_constraint_side,
            )

        last_backward_result = backward_result

        if (
            backward_result.max_feedforward_norm
            <= options.feedforward_tolerance
        ):
            if store_history:
                history.append(
                    IterationRecord(
                        outer_iteration=outer_iteration,
                        inner_iteration=inner_iteration,
                        cost=current_rollout.total_cost,
                        regularization=regularization,
                        accepted=False,
                        cost_change=0.0,
                        expected_reduction=0.0,
                        step_size=0.0,
                        feedforward_norm=(
                            backward_result.max_feedforward_norm
                        ),
                        state_barrier=state_barrier_history,
                        control_barrier=control_barrier_history,
                    )
                )

            return ILQRSubproblemResult(
                rollout=current_rollout,
                backward_result=backward_result,
                status=SolverStatus.CONVERGED,
                message=(
                    "The feedforward correction satisfied "
                    "the convergence tolerance."
                ),
                iterations=inner_iteration + 1,
                regularization=regularization,
                history=tuple(history),
            )

        line_search_result = line_search(
            problem=problem,
            nominal_rollout=current_rollout,
            backward_result=backward_result,
            state_barrier=state_barrier,
            control_barrier=control_barrier,
            minimum_margin=minimum_margin,
            options=options,
        )

        if not line_search_result.accepted:
            (
                constraint_type,
                constraint_index,
                constraint_side,
            ) = _failure_information(
                line_search_result
            )

            if constraint_type is not None:
                last_constraint_type = constraint_type
                last_constraint_index = constraint_index
                last_constraint_side = constraint_side

            if store_history:
                history.append(
                    IterationRecord(
                        outer_iteration=outer_iteration,
                        inner_iteration=inner_iteration,
                        cost=current_rollout.total_cost,
                        regularization=regularization,
                        accepted=False,
                        feedforward_norm=(
                            backward_result.max_feedforward_norm
                        ),
                        state_barrier=state_barrier_history,
                        control_barrier=control_barrier_history,
                    )
                )

            if verbose:
                print(
                    f"Outer {outer_iteration:3d} | "
                    f"Inner {inner_iteration:3d} | "
                    f"cost {current_rollout.total_cost:.8e} | "
                    f"line search failed | "
                    f"regularization {regularization:.3e}"
                )

            if (
                regularization
                >= options.regularization_maximum
            ):
                return ILQRSubproblemResult(
                    rollout=current_rollout,
                    backward_result=backward_result,
                    status=SolverStatus.LINE_SEARCH_FAILED,
                    message=(
                        "The line search failed at the maximum "
                        "regularization."
                    ),
                    iterations=inner_iteration + 1,
                    regularization=regularization,
                    history=tuple(history),
                    failed_constraint_type=last_constraint_type,
                    failed_constraint_index=last_constraint_index,
                    failed_constraint_side=last_constraint_side,
                )

            regularization = min(
                options.regularization_maximum,
                regularization
                * options.regularization_increase,
            )

            continue

        previous_cost = current_rollout.total_cost
        current_rollout = line_search_result.rollout

        if current_rollout is None:
            raise RuntimeError(
                "An accepted line search did not return a rollout."
            )

        actual_cost_change = (
            line_search_result.actual_cost_change
        )

        if actual_cost_change is None:
            raise RuntimeError(
                "An accepted line search did not return "
                "an actual cost change."
            )

        relative_cost_change = (
            abs(actual_cost_change)
            / max(1.0, abs(previous_cost))
        )

        if store_history:
            expected_reduction = None

            if (
                line_search_result.predicted_cost_change
                is not None
            ):
                expected_reduction = -float(
                    line_search_result.predicted_cost_change
                )

            history.append(
                IterationRecord(
                    outer_iteration=outer_iteration,
                    inner_iteration=inner_iteration,
                    cost=current_rollout.total_cost,
                    regularization=regularization,
                    accepted=True,
                    cost_change=float(actual_cost_change),
                    expected_reduction=expected_reduction,
                    step_size=line_search_result.step_size,
                    feedforward_norm=(
                        backward_result.max_feedforward_norm
                    ),
                    state_barrier=state_barrier_history,
                    control_barrier=control_barrier_history,
                )
            )

        if verbose:
            print(
                f"Outer {outer_iteration:3d} | "
                f"Inner {inner_iteration:3d} | "
                f"cost {current_rollout.total_cost:.8e} | "
                f"alpha {line_search_result.step_size:.3e} | "
                f"regularization {regularization:.3e}"
            )

        regularization = max(
            options.regularization_minimum,
            regularization
            * options.regularization_decrease,
        )

        if (
            relative_cost_change
            <= options.relative_cost_tolerance
        ):
            (
                final_backward_result,
                regularization,
                final_backward_error,
            ) = compute_backward_pass(
                current_rollout,
                regularization,
            )

            if final_backward_result is None:
                message = (
                    "The trajectory satisfied the cost tolerance, "
                    "but the final backward pass failed."
                )

                if final_backward_error is not None:
                    message = (
                        f"{message} {final_backward_error}"
                    )

                return ILQRSubproblemResult(
                    rollout=current_rollout,
                    backward_result=None,
                    status=SolverStatus.REGULARIZATION_LIMIT,
                    message=message,
                    iterations=inner_iteration + 1,
                    regularization=regularization,
                    history=tuple(history),
                )

            return ILQRSubproblemResult(
                rollout=current_rollout,
                backward_result=final_backward_result,
                status=SolverStatus.CONVERGED,
                message=(
                    "The relative cost change satisfied "
                    "the convergence tolerance."
                ),
                iterations=inner_iteration + 1,
                regularization=regularization,
                history=tuple(history),
            )

    return ILQRSubproblemResult(
        rollout=current_rollout,
        backward_result=last_backward_result,
        status=SolverStatus.MAX_ITERATIONS,
        message=(
            "The inner iLQR solver reached its maximum "
            "number of iterations."
        ),
        iterations=options.max_iterations,
        regularization=regularization,
        history=tuple(history),
        failed_constraint_type=last_constraint_type,
        failed_constraint_index=last_constraint_index,
        failed_constraint_side=last_constraint_side,
    )

def _initialize_continuation_setting(
    bounds: Bounds | None,
    dimension: int,
    initial_parameter: ScalarOrArray,
    reduction_factor: ScalarOrArray,
    parameter_name: str,
    reduction_name: str,
) -> tuple[Array | None, Array | None, Array | None]:
    """Initialize one state or control continuation setting."""

    if bounds is None or not bounds.has_constraints:
        return None, None, None

    parameters = expand_barrier_parameters(
        initial_parameter,
        dimension,
        name=parameter_name,
    )

    reduction_array = np.asarray(
        reduction_factor,
        dtype=float,
    )

    if reduction_array.ndim == 0:
        reduction_array = np.full(
            dimension,
            float(reduction_array),
            dtype=float,
        )
    elif reduction_array.shape == (dimension,):
        reduction_array = reduction_array.copy()
    else:
        raise ValueError(
            f"{reduction_name} must be a scalar or have "
            f"shape ({dimension},), but received "
            f"{reduction_array.shape}."
        )

    if not np.all(np.isfinite(reduction_array)):
        raise ValueError(
            f"{reduction_name} must contain finite values."
        )

    if (
        np.any(reduction_array <= 0.0)
        or np.any(reduction_array >= 1.0)
    ):
        raise ValueError(
            f"{reduction_name} must lie strictly "
            "between zero and one."
        )

    active_mask = (
        bounds.lower_is_finite
        | bounds.upper_is_finite
    )

    return parameters, reduction_array, active_mask


def _active_norm(
    parameters: Array | None,
    active_mask: Array | None,
) -> float:
    """Calculate the norm of active barrier parameters."""

    if parameters is None or active_mask is None:
        return 0.0

    if not np.any(active_mask):
        return 0.0

    return float(
        np.linalg.norm(parameters[active_mask])
    )


def _continuation_complete(
    state_parameters: Array | None,
    control_parameters: Array | None,
    state_active: Array | None,
    control_active: Array | None,
    tolerance: float,
) -> bool:
    """Check the barrier-continuation termination condition."""

    return bool(
        _active_norm(
            state_parameters,
            state_active,
        )
        <= tolerance
        and _active_norm(
            control_parameters,
            control_active,
        )
        <= tolerance
    )


def _reduce_parameters(
    parameters: Array | None,
    reduction_factors: Array | None,
    active_mask: Array | None,
) -> Array | None:
    """Apply a componentwise barrier reduction."""

    if parameters is None:
        return None

    reduced = parameters.copy()

    if reduction_factors is not None and active_mask is not None:
        reduced[active_mask] *= reduction_factors[active_mask]

    return reduced


def _has_continuation_progress(
    previous: Array | None,
    candidate: Array | None,
    active_mask: Array | None,
) -> bool:
    """Check whether at least one active parameter decreased."""

    if previous is None:
        return False

    if candidate is None or active_mask is None:
        return False

    if not np.any(active_mask):
        return False

    return bool(
        np.any(
            candidate[active_mask]
            < previous[active_mask]
        )
    )


def _adapt_reduction_factors(
    state_reduction: Array | None,
    control_reduction: Array | None,
    state_active: Array | None,
    control_active: Array | None,
    failed_type: str | None,
    failed_index: int | None,
    update_rate: float,
) -> bool:
    """Make failed barrier reductions more conservative."""

    changed = False

    if (
        failed_type == "state"
        and failed_index is not None
        and state_reduction is not None
        and state_active is not None
        and 0 <= failed_index < state_reduction.size
        and state_active[failed_index]
    ):
        old_value = state_reduction[failed_index]

        state_reduction[failed_index] = min(
            1.0,
            old_value * update_rate,
        )

        return bool(
            state_reduction[failed_index] > old_value
        )

    if (
        failed_type == "control"
        and failed_index is not None
        and control_reduction is not None
        and control_active is not None
        and 0 <= failed_index < control_reduction.size
        and control_active[failed_index]
    ):
        old_value = control_reduction[failed_index]

        control_reduction[failed_index] = min(
            1.0,
            old_value * update_rate,
        )

        return bool(
            control_reduction[failed_index] > old_value
        )

    # If a specific failed constraint was not identified,
    # make all active reductions more conservative.
    if state_reduction is not None and state_active is not None:
        old_values = state_reduction[state_active].copy()

        state_reduction[state_active] = np.minimum(
            1.0,
            state_reduction[state_active] * update_rate,
        )

        changed = changed or bool(
            np.any(
                state_reduction[state_active]
                > old_values
            )
        )

    if (
        control_reduction is not None
        and control_active is not None
    ):
        old_values = control_reduction[
            control_active
        ].copy()

        control_reduction[control_active] = np.minimum(
            1.0,
            control_reduction[control_active] * update_rate,
        )

        changed = changed or bool(
            np.any(
                control_reduction[control_active]
                > old_values
            )
        )

    return changed


def _build_solution(
    result: ILQRSubproblemResult,
    status: SolverStatus,
    message: str,
    history: list[IterationRecord],
    state_barrier: Array | None,
    control_barrier: Array | None,
) -> BoxILQRSolution:
    """Convert a fixed-barrier result into a public solution."""

    if result.rollout is None:
        raise RuntimeError(
            "Cannot build a solution without a trajectory."
        )

    if result.backward_result is None:
        raise RuntimeError(
            "Cannot build a solution without a final backward pass."
        )

    return BoxILQRSolution(
        states=result.rollout.states,
        controls=result.rollout.controls,
        feedback_gains=(
            result.backward_result.feedback_gains
        ),
        feedforward_terms=(
            result.backward_result.feedforward_terms
        ),
        cost=result.rollout.original_cost,
        augmented_cost=result.rollout.total_cost,
        barrier_cost=result.rollout.barrier_cost,
        status=status,
        message=message,
        history=tuple(history),
        state_barrier=state_barrier,
        control_barrier=control_barrier,
    )


def solve(
    problem: BoxILQRProblem,
    x0: Array,
    u_initial: Array,
    options: BoxILQROptions | None = None,
) -> BoxILQRSolution:
    """Solve a complete Box-iLQR problem.

    Parameters
    ----------
    problem
        Optimal-control problem definition.
    x0
        Initial state with shape ``(state_dim,)``.
    u_initial
        Strictly feasible initial control trajectory with shape
        ``(horizon, control_dim)``.
    options
        Inner-iLQR and outer-continuation options.
    """

    if options is None:
        options = BoxILQROptions()

    x0 = problem.validate_state(x0)

    u_initial = np.asarray(
        u_initial,
        dtype=float,
    )

    expected_control_shape = (
        problem.horizon,
        problem.control_dim,
    )

    if u_initial.shape != expected_control_shape:
        raise ValueError(
            f"u_initial must have shape "
            f"{expected_control_shape}, but received "
            f"{u_initial.shape}."
        )

    (
        state_parameters,
        state_reduction,
        state_active,
    ) = _initialize_continuation_setting(
        bounds=problem.state_bounds,
        dimension=problem.state_dim,
        initial_parameter=(
            options.barrier.initial_state_barrier
        ),
        reduction_factor=(
            options.barrier.state_reduction_factor
        ),
        parameter_name="initial_state_barrier",
        reduction_name="state_reduction_factor",
    )

    (
        control_parameters,
        control_reduction,
        control_active,
    ) = _initialize_continuation_setting(
        bounds=problem.control_bounds,
        dimension=problem.control_dim,
        initial_parameter=(
            options.barrier.initial_control_barrier
        ),
        reduction_factor=(
            options.barrier.control_reduction_factor
        ),
        parameter_name="initial_control_barrier",
        reduction_name="control_reduction_factor",
    )

    # Validate strict feasibility before beginning continuation.
    rollout(
        problem=problem,
        initial_state=x0,
        controls=u_initial,
        state_barrier=state_parameters,
        control_barrier=control_parameters,
        minimum_margin=options.barrier.minimum_margin,
    )

    current_controls = u_initial.copy()

    complete_history: list[IterationRecord] = []

    last_successful_result: ILQRSubproblemResult | None = None
    last_successful_state_parameters: Array | None = None
    last_successful_control_parameters: Array | None = None

    consecutive_failed_reductions = 0

    for outer_iteration in range(
        options.barrier.max_outer_iterations
    ):
        if options.verbose:
            state_norm = _active_norm(
                state_parameters,
                state_active,
            )

            control_norm = _active_norm(
                control_parameters,
                control_active,
            )

            print(
                f"Outer {outer_iteration:3d} | "
                f"||mu|| {state_norm:.3e} | "
                f"||sigma|| {control_norm:.3e}"
            )

        subproblem_result = solve_ilqr_subproblem(
            problem=problem,
            initial_state=x0,
            initial_controls=current_controls,
            state_barrier=state_parameters,
            control_barrier=control_parameters,
            minimum_margin=(
                options.barrier.minimum_margin
            ),
            options=options.ilqr,
            outer_iteration=outer_iteration,
            store_history=(
                options.store_iteration_history
            ),
            verbose=options.verbose,
        )

        complete_history.extend(
            subproblem_result.history
        )

        if subproblem_result.success:
            if subproblem_result.rollout is None:
                raise RuntimeError(
                    "A successful subproblem returned no rollout."
                )

            last_successful_result = subproblem_result

            last_successful_state_parameters = (
                None
                if state_parameters is None
                else state_parameters.copy()
            )

            last_successful_control_parameters = (
                None
                if control_parameters is None
                else control_parameters.copy()
            )

            current_controls = (
                subproblem_result.rollout.controls.copy()
            )

            consecutive_failed_reductions = 0

            if _continuation_complete(
                state_parameters=state_parameters,
                control_parameters=control_parameters,
                state_active=state_active,
                control_active=control_active,
                tolerance=options.barrier.tolerance,
            ):
                return _build_solution(
                    result=subproblem_result,
                    status=SolverStatus.CONVERGED,
                    message=(
                        "Box-iLQR converged and the barrier "
                        "parameters satisfied the termination "
                        "tolerance."
                    ),
                    history=complete_history,
                    state_barrier=state_parameters,
                    control_barrier=control_parameters,
                )

            next_state_parameters = _reduce_parameters(
                state_parameters,
                state_reduction,
                state_active,
            )

            next_control_parameters = _reduce_parameters(
                control_parameters,
                control_reduction,
                control_active,
            )

            state_progress = _has_continuation_progress(
                state_parameters,
                next_state_parameters,
                state_active,
            )

            control_progress = _has_continuation_progress(
                control_parameters,
                next_control_parameters,
                control_active,
            )

            if not state_progress and not control_progress:
                return _build_solution(
                    result=subproblem_result,
                    status=(
                        SolverStatus.BARRIER_CONTINUATION_FAILED
                    ),
                    message=(
                        "The barrier parameters remain above "
                        "tolerance, but the reduction factors "
                        "cannot produce further progress."
                    ),
                    history=complete_history,
                    state_barrier=state_parameters,
                    control_barrier=control_parameters,
                )

            state_parameters = next_state_parameters
            control_parameters = next_control_parameters

            continue

        consecutive_failed_reductions += 1

        if last_successful_result is None:
            raise RuntimeError(
                "The initial barrier subproblem failed: "
                f"{subproblem_result.message}"
            )

        if (
            consecutive_failed_reductions
            > options.barrier.max_failed_reductions
        ):
            return _build_solution(
                result=last_successful_result,
                status=(
                    SolverStatus.BARRIER_CONTINUATION_FAILED
                ),
                message=(
                    "Box-iLQR exceeded the maximum number "
                    "of consecutive failed barrier reductions."
                ),
                history=complete_history,
                state_barrier=(
                    last_successful_state_parameters
                ),
                control_barrier=(
                    last_successful_control_parameters
                ),
            )

        reduction_changed = _adapt_reduction_factors(
            state_reduction=state_reduction,
            control_reduction=control_reduction,
            state_active=state_active,
            control_active=control_active,
            failed_type=(
                subproblem_result.failed_constraint_type
            ),
            failed_index=(
                subproblem_result.failed_constraint_index
            ),
            update_rate=(
                options.barrier.reduction_update_rate
            ),
        )

        if not reduction_changed:
            return _build_solution(
                result=last_successful_result,
                status=(
                    SolverStatus.BARRIER_CONTINUATION_FAILED
                ),
                message=(
                    "The failed barrier reduction could not "
                    "be made more conservative."
                ),
                history=complete_history,
                state_barrier=(
                    last_successful_state_parameters
                ),
                control_barrier=(
                    last_successful_control_parameters
                ),
            )

        state_parameters = _reduce_parameters(
            last_successful_state_parameters,
            state_reduction,
            state_active,
        )

        control_parameters = _reduce_parameters(
            last_successful_control_parameters,
            control_reduction,
            control_active,
        )

        state_progress = _has_continuation_progress(
            last_successful_state_parameters,
            state_parameters,
            state_active,
        )

        control_progress = _has_continuation_progress(
            last_successful_control_parameters,
            control_parameters,
            control_active,
        )

        if not state_progress and not control_progress:
            return _build_solution(
                result=last_successful_result,
                status=(
                    SolverStatus.BARRIER_CONTINUATION_FAILED
                ),
                message=(
                    "The adapted barrier reduction factors "
                    "cannot produce a smaller barrier value."
                ),
                history=complete_history,
                state_barrier=(
                    last_successful_state_parameters
                ),
                control_barrier=(
                    last_successful_control_parameters
                ),
            )

        current_controls = (
            last_successful_result.rollout.controls.copy()
        )

    if last_successful_result is None:
        raise RuntimeError(
            "Box-iLQR reached the outer-iteration limit "
            "without a successful subproblem."
        )

    return _build_solution(
        result=last_successful_result,
        status=SolverStatus.MAX_ITERATIONS,
        message=(
            "Box-iLQR reached the maximum number of "
            "outer continuation iterations."
        ),
        history=complete_history,
        state_barrier=last_successful_state_parameters,
        control_barrier=last_successful_control_parameters,
    )
