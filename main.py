from boxilqr import BoxILQRProblem, BoxILQROptions, solve

problem = BoxILQRProblem(
    dynamics=dynamics,              # x_next = dynamics(x, u, k)
    running_cost=running_cost,
    terminal_cost=terminal_cost,
    horizon=N,
    state_bounds=(x_lower, x_upper),
    control_bounds=(u_lower, u_upper),
)

solution = solve(
    problem,
    x0=x0,
    u_initial=u_guess,
    options=BoxILQROptions(),
)

x_opt = solution.states
u_opt = solution.controls
K_opt = solution.feedback_gains
