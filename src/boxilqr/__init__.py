"""Box-iLQR: constrained nonlinear trajectory optimization."""

from boxilqr.options import (
    BarrierOptions,
    BoxILQROptions,
    ILQROptions,
)
from boxilqr.problem import (
    Array,
    Bounds,
    BoxILQRProblem,
    Dynamics,
    RunningCost,
    TerminalCost,
)
from boxilqr.solution import (
    BoxILQRSolution,
    IterationRecord,
    SolverStatus,
)
from boxilqr.integrators import (
    ContinuousDynamics,
    discretize_dynamics,
    euler_step,
    rk4_step,
)
from boxilqr.barriers import (
    BarrierEvaluation,
    InfeasiblePointError,
    evaluate_box_barrier,
    expand_barrier_parameters,
)
from boxilqr.derivatives import (
    DynamicsDerivatives,
    RunningCostDerivatives,
    TerminalCostDerivatives,
    differentiate_dynamics,
    differentiate_running_cost,
    differentiate_terminal_cost,
    finite_difference_cross_hessian,
    finite_difference_gradient,
    finite_difference_hessian,
    finite_difference_jacobian,
)
from boxilqr.rollout import (
    ControlPolicy,
    RolloutResult,
    TrajectoryCost,
    rollout,
    rollout_policy,
)
from boxilqr.backward_pass import (
    BackwardPassError,
    BackwardPassResult,
    backward_pass,
)
from boxilqr.forward_pass import (
    LineSearchAttempt,
    LineSearchResult,
    line_search,
)
from boxilqr.solver import (
    ILQRSubproblemResult,
    solve,
    solve_ilqr_subproblem,
)
from boxilqr.diagnostics import (
    ConvergenceHistory,
    extract_convergence_history,
    plot_convergence,
)
from boxilqr.simulation import (
    ControlNoise,
    MeasurementNoise,
    ProcessNoise,
    SimulationResult,
    StateDifference,
    simulate_closed_loop,
)

__version__ = "0.1.0"

__all__ = [
    "Array",
    "BarrierOptions",
    "Bounds",
    "BoxILQROptions",
    "BoxILQRProblem",
    "BoxILQRSolution",
    "Dynamics",
    "ILQROptions",
    "IterationRecord",
    "RunningCost",
    "SolverStatus",
    "TerminalCost",
    "ContinuousDynamics",
    "discretize_dynamics",
    "euler_step",
    "rk4_step",
    "BarrierEvaluation",
    "InfeasiblePointError",
    "evaluate_box_barrier",
    "expand_barrier_parameters",
    "DynamicsDerivatives",
    "RunningCostDerivatives",
    "TerminalCostDerivatives",
    "differentiate_dynamics",
    "differentiate_running_cost",
    "differentiate_terminal_cost",
    "finite_difference_cross_hessian",
    "finite_difference_gradient",
    "finite_difference_hessian",
    "finite_difference_jacobian",
    "ControlPolicy",
    "RolloutResult",
    "TrajectoryCost",
    "rollout",
    "rollout_policy",
    "BackwardPassError",
    "BackwardPassResult",
    "backward_pass",
    "LineSearchAttempt",
    "LineSearchResult",
    "line_search",
    "ILQRSubproblemResult",
    "solve_ilqr_subproblem",
    "solve",
    "ConvergenceHistory",
    "extract_convergence_history",
    "plot_convergence",
    "ControlNoise",
    "MeasurementNoise",
    "ProcessNoise",
    "SimulationResult",
    "StateDifference",
    "simulate_closed_loop",
]
