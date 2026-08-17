"""Bounded double-integrator example for Box-iLQR."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from boxilqr import (
    BarrierOptions,
    BoxILQROptions,
    BoxILQRProblem,
    ILQROptions,
    discretize_dynamics,
    solve,
    plot_convergence,
    simulate_closed_loop,
)


def main() -> None:
    """Solve and plot a bounded double-integrator problem."""

    # ---------------------------------------------------------
    # Simulation settings
    # ---------------------------------------------------------
    dt = 0.1
    horizon = 60

    initial_state = np.array([-1.0, 0.0])
    target_state = np.array([1.0, 0.0])

    state_lower = np.array([-1.5, -1.0])
    state_upper = np.array([1.5, 1.0])

    control_lower = np.array([-0.5])
    control_upper = np.array([0.5])

    # ---------------------------------------------------------
    # Continuous dynamics: x_dot = f(t, x, u)
    # ---------------------------------------------------------
    def continuous_dynamics(
        time: float,
        state: np.ndarray,
        control: np.ndarray,
    ) -> np.ndarray:
        del time

        position, velocity = state
        acceleration = control[0]

        del position

        return np.array([
            velocity,
            acceleration,
        ])

    discrete_dynamics = discretize_dynamics(
        continuous_dynamics,
        dt=dt,
        method="rk4",
    )

    # ---------------------------------------------------------
    # Cost function
    # ---------------------------------------------------------
    state_cost_matrix = np.diag([0.1, 0.01])
    control_cost_matrix = np.array([[0.01]])
    terminal_cost_matrix = np.diag([200.0, 50.0])

    def running_cost(
        state: np.ndarray,
        control: np.ndarray,
        k: int,
    ) -> float:
        del k

        state_error = state - target_state

        return dt * (
            0.5
            * state_error
            @ state_cost_matrix
            @ state_error
            + 0.5
            * control
            @ control_cost_matrix
            @ control
        )

    def terminal_cost(state: np.ndarray) -> float:
        state_error = state - target_state

        return (
            0.5
            * state_error
            @ terminal_cost_matrix
            @ state_error
        )

    # ---------------------------------------------------------
    # Define the Box-iLQR problem
    # ---------------------------------------------------------
    problem = BoxILQRProblem(
        dynamics=discrete_dynamics,
        running_cost=running_cost,
        terminal_cost=terminal_cost,
        horizon=horizon,
        state_dim=2,
        control_dim=1,
        state_bounds=(
            state_lower,
            state_upper,
        ),
        control_bounds=(
            control_lower,
            control_upper,
        ),
        name="bounded_double_integrator",
    )

    # Zero control is a strictly feasible initial guess.
    initial_controls = np.zeros(
        (horizon, 1),
        dtype=float,
    )

    # ---------------------------------------------------------
    # Solver settings
    # ---------------------------------------------------------
    options = BoxILQROptions(
        ilqr=ILQROptions(
            max_iterations=100,
            relative_cost_tolerance=1.0e-8,
            feedforward_tolerance=1.0e-5,
            max_line_search_steps=20,
            line_search_decay=0.5,
            acceptance_ratio=1.0e-4,
            regularization_initial=1.0e-6,
        ),
        barrier=BarrierOptions(
            initial_state_barrier=0.1,
            initial_control_barrier=0.1,
            state_reduction_factor=0.2,
            control_reduction_factor=0.2,
            reduction_update_rate=1.5,
            tolerance=1.0e-4,
            max_outer_iterations=30,
            max_failed_reductions=20,
            minimum_margin=1.0e-12,
        ),
        verbose=True,
        store_iteration_history=True,
    )

    # ---------------------------------------------------------
    # Solve
    # ---------------------------------------------------------
    solution = solve(
        problem=problem,
        x0=initial_state,
        u_initial=initial_controls,
        options=options,
    )

    # ---------------------------------------------------------
    # Display numerical results
    # ---------------------------------------------------------
    np.set_printoptions(
        precision=8,
        suppress=True,
    )

    print()
    print("Box-iLQR result")
    print("----------------")
    print(f"Status:          {solution.status.value}")
    print(f"Success:         {solution.success}")
    print(f"Message:         {solution.message}")
    print(f"Original cost:   {solution.original_cost:.8f}")

    if solution.augmented_cost is not None:
        print(
            f"Augmented cost:  "
            f"{solution.augmented_cost:.8f}"
        )

    if solution.barrier_cost is not None:
        print(
            f"Barrier cost:    "
            f"{solution.barrier_cost:.8f}"
        )

    print(f"Initial state:   {solution.states[0]}")
    print(f"Terminal state:  {solution.states[-1]}")
    print(f"Target state:    {target_state}")

    minimum_state_lower_margin = np.min(
        solution.states - state_lower
    )

    minimum_state_upper_margin = np.min(
        state_upper - solution.states
    )

    minimum_control_lower_margin = np.min(
        solution.controls - control_lower
    )

    minimum_control_upper_margin = np.min(
        control_upper - solution.controls
    )

    print()
    print("Minimum constraint margins")
    print("--------------------------")
    print(
        "State lower margin:   "
        f"{minimum_state_lower_margin:.6e}"
    )
    print(
        "State upper margin:   "
        f"{minimum_state_upper_margin:.6e}"
    )
    print(
        "Control lower margin: "
        f"{minimum_control_lower_margin:.6e}"
    )
    print(
        "Control upper margin: "
        f"{minimum_control_upper_margin:.6e}"
    )

    if not solution.success:
        raise RuntimeError(
            f"Box-iLQR did not converge: {solution.message}"
        )

    if minimum_state_lower_margin <= 0.0:
        raise RuntimeError("A lower state bound was violated.")

    if minimum_state_upper_margin <= 0.0:
        raise RuntimeError("An upper state bound was violated.")

    if minimum_control_lower_margin <= 0.0:
        raise RuntimeError("A lower control bound was violated.")

    if minimum_control_upper_margin <= 0.0:
        raise RuntimeError("An upper control bound was violated.")

    # ---------------------------------------------------------
    # Plot results
    # ---------------------------------------------------------
    state_time = np.arange(horizon + 1) * dt
    control_time = np.arange(horizon) * dt

    figure, axes = plt.subplots(
        3,
        1,
        figsize=(9, 9),
        sharex=True,
    )

    # Position
    axes[0].plot(
        state_time,
        solution.states[:, 0],
        color="tab:blue",
        linewidth=2.0,
        label="Position",
    )

    axes[0].axhline(
        target_state[0],
        color="tab:green",
        linestyle="--",
        linewidth=1.5,
        label="Target",
    )

    axes[0].axhline(
        state_lower[0],
        color="black",
        linestyle=":",
        label="Bounds",
    )

    axes[0].axhline(
        state_upper[0],
        color="black",
        linestyle=":",
    )

    axes[0].set_ylabel("Position")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    # Velocity
    axes[1].plot(
        state_time,
        solution.states[:, 1],
        color="tab:orange",
        linewidth=2.0,
        label="Velocity",
    )

    axes[1].axhline(
        state_lower[1],
        color="black",
        linestyle=":",
        label="Bounds",
    )

    axes[1].axhline(
        state_upper[1],
        color="black",
        linestyle=":",
    )

    axes[1].axhline(
        target_state[1],
        color="tab:green",
        linestyle="--",
        linewidth=1.5,
        label="Target",
    )

    axes[1].set_ylabel("Velocity")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    # Control
    axes[2].step(
        control_time,
        solution.controls[:, 0],
        where="post",
        color="tab:red",
        linewidth=2.0,
        label="Acceleration",
    )

    axes[2].axhline(
        control_lower[0],
        color="black",
        linestyle=":",
        label="Bounds",
    )

    axes[2].axhline(
        control_upper[0],
        color="black",
        linestyle=":",
    )

    axes[2].set_xlabel("Time [s]")
    axes[2].set_ylabel("Acceleration")
    axes[2].grid(True, alpha=0.3)
    axes[2].legend()

    figure.suptitle(
        "Box-iLQR: Bounded Double Integrator",
        fontsize=14,
    )

    figure.tight_layout()

    output_directory = Path("results")
    output_directory.mkdir(exist_ok=True)

    output_path = (
        output_directory
        / "double_integrator_solution.png"
    )

    figure.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight",
    )

    print()
    print(f"Figure saved to: {output_path}")
    convergence_path = (
        output_directory
        / "double_integrator_convergence.png"
    )

    plot_convergence(
        solution.history,
        save_path=convergence_path,
        show=False,
    )

    print(
        f"Convergence figure saved to: "
        f"{convergence_path}"
    )
    
        # ---------------------------------------------------------
    # Disturbed open-loop and feedback simulations
    # ---------------------------------------------------------
    random_generator = np.random.default_rng(7)

    measurement_samples = random_generator.normal(
        loc=0.0,
        scale=np.array([0.005, 0.005]),
        size=(horizon, 2),
    )

    control_samples = random_generator.normal(
        loc=0.0,
        scale=0.01,
        size=(horizon, 1),
    )

    process_samples = random_generator.normal(
        loc=0.0,
        scale=np.array([0.0005, 0.003]),
        size=(horizon, 2),
    )

    perturbed_initial_state = (
        initial_state
        + np.array([0.05, -0.05])
    )

    def measurement_noise(
        state: np.ndarray,
        k: int,
    ) -> np.ndarray:
        del state
        return measurement_samples[k]

    def control_noise(
        control: np.ndarray,
        k: int,
    ) -> np.ndarray:
        del control
        return control_samples[k]

    def process_noise(
        state: np.ndarray,
        control: np.ndarray,
        k: int,
    ) -> np.ndarray:
        del state, control
        return process_samples[k]

    open_loop_simulation = simulate_closed_loop(
        problem=problem,
        solution=solution,
        initial_state=perturbed_initial_state,
        feedback=False,
        measurement_noise=measurement_noise,
        control_noise=control_noise,
        process_noise=process_noise,
    )

    feedback_simulation = simulate_closed_loop(
        problem=problem,
        solution=solution,
        initial_state=perturbed_initial_state,
        feedback=True,
        measurement_noise=measurement_noise,
        control_noise=control_noise,
        process_noise=process_noise,
    )

    open_loop_error = (
        open_loop_simulation.states
        - solution.states
    )

    feedback_error = (
        feedback_simulation.states
        - solution.states
    )

    print()
    print("Terminal tracking errors")
    print("------------------------")
    print(
        "Open-loop error: "
        f"{open_loop_error[-1]}"
    )
    print(
        "Feedback error:  "
        f"{feedback_error[-1]}"
    )

    feedback_figure, feedback_axes = plt.subplots(
        3,
        1,
        figsize=(9, 9),
        sharex=True,
    )

    feedback_axes[0].plot(
        state_time,
        open_loop_error[:, 0],
        color="tab:red",
        linewidth=1.8,
        label="Open loop",
    )

    feedback_axes[0].plot(
        state_time,
        feedback_error[:, 0],
        color="tab:blue",
        linewidth=1.8,
        label="Feedback",
    )

    feedback_axes[0].set_ylabel("Position error")
    feedback_axes[0].grid(True, alpha=0.3)
    feedback_axes[0].legend()

    feedback_axes[1].plot(
        state_time,
        open_loop_error[:, 1],
        color="tab:red",
        linewidth=1.8,
        label="Open loop",
    )

    feedback_axes[1].plot(
        state_time,
        feedback_error[:, 1],
        color="tab:blue",
        linewidth=1.8,
        label="Feedback",
    )

    feedback_axes[1].set_ylabel("Velocity error")
    feedback_axes[1].grid(True, alpha=0.3)
    feedback_axes[1].legend()

    feedback_axes[2].step(
        control_time,
        solution.controls[:, 0],
        where="post",
        color="black",
        linewidth=1.5,
        label="Nominal",
    )

    feedback_axes[2].step(
        control_time,
        open_loop_simulation.controls[:, 0],
        where="post",
        color="tab:red",
        alpha=0.8,
        label="Open loop",
    )

    feedback_axes[2].step(
        control_time,
        feedback_simulation.controls[:, 0],
        where="post",
        color="tab:blue",
        alpha=0.8,
        label="Feedback",
    )

    feedback_axes[2].axhline(
        control_lower[0],
        color="black",
        linestyle=":",
    )

    feedback_axes[2].axhline(
        control_upper[0],
        color="black",
        linestyle=":",
    )

    feedback_axes[2].set_xlabel("Time [s]")
    feedback_axes[2].set_ylabel("Applied control")
    feedback_axes[2].grid(True, alpha=0.3)
    feedback_axes[2].legend()

    feedback_figure.suptitle(
        "Open-Loop and Box-iLQR Feedback Comparison",
        fontsize=14,
    )

    feedback_figure.tight_layout()

    feedback_path = (
        output_directory
        / "double_integrator_feedback.png"
    )

    feedback_figure.savefig(
        feedback_path,
        dpi=200,
        bbox_inches="tight",
    )

    print(
        f"Feedback figure saved to: "
        f"{feedback_path}"
    )
    
    plt.show()
    

if __name__ == "__main__":
    main()
