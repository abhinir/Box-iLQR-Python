"""Feasibility-preserving forward pass and line search."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from boxilqr.backward_pass import BackwardPassResult
from boxilqr.barriers import InfeasiblePointError
from boxilqr.options import ILQROptions, ScalarOrArray
from boxilqr.problem import Array, BoxILQRProblem
from boxilqr.rollout import RolloutResult, rollout_policy


@dataclass(frozen=True)
class LineSearchAttempt:
    """Information about one candidate line-search step."""

    step_size: float
    predicted_cost_change: float
    feasible: bool | None
    accepted: bool

    candidate_cost: float | None = None
    actual_cost_change: float | None = None
    reduction_ratio: float | None = None
    message: str = ""
    
    constraint_type: str | None = None
    constraint_index: int | None = None
    constraint_side: str | None = None

@dataclass(frozen=True)
class LineSearchResult:
    """Result of the feasibility-preserving line search."""

    accepted: bool
    rollout: RolloutResult | None
    attempts: tuple[LineSearchAttempt, ...]

    step_size: float | None = None
    actual_cost_change: float | None = None
    predicted_cost_change: float | None = None
    reduction_ratio: float | None = None

    @property
    def number_of_attempts(self) -> int:
        """Number of step sizes examined."""

        return len(self.attempts)


def line_search(
    problem: BoxILQRProblem,
    nominal_rollout: RolloutResult,
    backward_result: BackwardPassResult,
    *,
    state_barrier: ScalarOrArray | None = None,
    control_barrier: ScalarOrArray | None = None,
    minimum_margin: float = 1.0e-12,
    options: ILQROptions | None = None,
) -> LineSearchResult:
    """Perform a feasibility-preserving backtracking line search."""

    if options is None:
        options = ILQROptions()

    expected_state_shape = (
        problem.horizon + 1,
        problem.state_dim,
    )

    expected_control_shape = (
        problem.horizon,
        problem.control_dim,
    )

    expected_feedback_shape = (
        problem.horizon,
        problem.control_dim,
        problem.state_dim,
    )

    if nominal_rollout.states.shape != expected_state_shape:
        raise ValueError(
            "The nominal state trajectory has an incorrect shape."
        )

    if nominal_rollout.controls.shape != expected_control_shape:
        raise ValueError(
            "The nominal control trajectory has an incorrect shape."
        )

    if (
        backward_result.feedforward_terms.shape
        != expected_control_shape
    ):
        raise ValueError(
            "The feedforward trajectory has an incorrect shape."
        )

    if (
        backward_result.feedback_gains.shape
        != expected_feedback_shape
    ):
        raise ValueError(
            "The feedback-gain trajectory has an incorrect shape."
        )

    nominal_cost = nominal_rollout.total_cost
    initial_state = nominal_rollout.states[0]

    attempts: list[LineSearchAttempt] = []

    for attempt_index in range(
        options.max_line_search_steps
    ):
        step_size = (
            options.line_search_decay**attempt_index
        )

        predicted_cost_change = (
            backward_result.predicted_cost_change(
                step_size
            )
        )

        if (
            not np.isfinite(predicted_cost_change)
            or predicted_cost_change >= 0.0
        ):
            attempts.append(
                LineSearchAttempt(
                    step_size=step_size,
                    predicted_cost_change=float(
                        predicted_cost_change
                    ),
                    feasible=None,
                    accepted=False,
                    message=(
                        "The local model did not predict "
                        "a cost decrease."
                    ),
                )
            )

            continue

        def candidate_policy(
            current_state: Array,
            k: int,
        ) -> Array:
            state_error = (
                current_state
                - nominal_rollout.states[k]
            )

            return (
                nominal_rollout.controls[k]
                + step_size
                * backward_result.feedforward_terms[k]
                + backward_result.feedback_gains[k]
                @ state_error
            )

        try:
            candidate_rollout = rollout_policy(
                problem=problem,
                initial_state=initial_state,
                policy=candidate_policy,
                state_barrier=state_barrier,
                control_barrier=control_barrier,
                minimum_margin=minimum_margin,
            )
        except InfeasiblePointError as error:
            variable_name = error.variable_name.lower()

            if "control" in variable_name:
                constraint_type = "control"
            elif "state" in variable_name:
                constraint_type = "state"
            else:
                constraint_type = None

            attempts.append(
                LineSearchAttempt(
                    step_size=step_size,
                    predicted_cost_change=float(
                        predicted_cost_change
                    ),
                    feasible=False,
                    accepted=False,
                    message=str(error),
                    constraint_type=constraint_type,
                    constraint_index=error.index,
                    constraint_side=error.side,
                )
            )

            continue

        candidate_cost = candidate_rollout.total_cost

        actual_cost_change = (
            candidate_cost - nominal_cost
        )

        reduction_ratio = (
            actual_cost_change
            / predicted_cost_change
        )

        accepted = bool(
            actual_cost_change < 0.0
            and np.isfinite(reduction_ratio)
            and reduction_ratio
            >= options.acceptance_ratio
        )

        attempt = LineSearchAttempt(
            step_size=float(step_size),
            predicted_cost_change=float(
                predicted_cost_change
            ),
            feasible=True,
            accepted=accepted,
            candidate_cost=float(candidate_cost),
            actual_cost_change=float(
                actual_cost_change
            ),
            reduction_ratio=float(reduction_ratio),
            message=(
                "Candidate accepted."
                if accepted
                else "Candidate did not provide sufficient decrease."
            ),
        )

        attempts.append(attempt)

        if accepted:
            return LineSearchResult(
                accepted=True,
                rollout=candidate_rollout,
                attempts=tuple(attempts),
                step_size=float(step_size),
                actual_cost_change=float(
                    actual_cost_change
                ),
                predicted_cost_change=float(
                    predicted_cost_change
                ),
                reduction_ratio=float(
                    reduction_ratio
                ),
            )

    return LineSearchResult(
        accepted=False,
        rollout=None,
        attempts=tuple(attempts),
    )
