# Box-iLQR

Box-iLQR is a Python implementation of a logarithmic-barrier extension of the iterative Linear Quadratic Regulator (iLQR) for nonlinear optimal-control problems with componentwise state and control bounds.

The solver produces both a nominal state-control trajectory and time-varying feedback gains. It is intended as an easy-to-use research implementation for constrained trajectory optimization and closed-loop execution.

> **Development status:** This package is under active development. The core solver and bounded double-integrator example are working, but the public API may change before the first stable release.

## Main features

- Nonlinear discrete-time dynamics
- RK4 and forward-Euler discretization for continuous-time models
- Componentwise state and control bounds
- Analytical logarithmic-barrier values, gradients, and Hessians
- Inner iLQR iterations with regularized backward passes
- Feasibility-preserving backtracking line search
- Adaptive, componentwise barrier continuation
- Warm starts between barrier subproblems
- Time-varying feedforward terms and feedback gains
- Finite-difference dynamics and cost derivatives
- Solver histories and convergence diagnostics
- Automated tests with `pytest`

## Mathematical problem

Box-iLQR considers finite-horizon problems of the form

```text
minimize    Phi(x_N) + sum(C(x_k, u_k, k))

subject to  x_(k+1) = F(x_k, u_k, k)
            x_lower < x_k < x_upper
            u_lower < u_k < u_upper
```

The box constraints are incorporated through logarithmic barriers. For a two-sided scalar constraint,

```text
lower_i < z_i < upper_i,
```

the barrier contribution is

```text
-mu_i [log(z_i - lower_i) + log(upper_i - z_i)].
```

An outer continuation loop progressively reduces the state and control barrier parameters. For each fixed set of barrier parameters, an inner iLQR solver performs regularized backward passes and feasibility-preserving forward line searches.

The resulting local policy is

```text
u_k = u_nominal_k + K_k (x_k - x_nominal_k).
```

## Requirements

- Python 3.10 or newer
- NumPy
- SciPy
- Matplotlib for examples and diagnostic plots
- pytest for testing

## Installation from the local source folder

Open Terminal and enter the Box-iLQR directory:

```bash
cd ~/Documents/Box-iLQR
```

Create and activate a virtual environment:

```bash
python3 -m venv ilqrenv
source ilqrenv/bin/activate
```

Upgrade `pip` and install the package in editable mode:

```bash
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

Editable installation makes changes inside `src/boxilqr` immediately available without reinstalling the package.

## Quick start

The core solver uses discrete dynamics with the signature

```python
next_state = dynamics(state, control, k)
```

Continuous-time dynamics can be converted to discrete dynamics using `discretize_dynamics`.

```python
import numpy as np

from boxilqr import (
    BarrierOptions,
    BoxILQROptions,
    BoxILQRProblem,
    ILQROptions,
    discretize_dynamics,
    solve,
)


dt = 0.1
horizon = 60

x0 = np.array([-1.0, 0.0])
target = np.array([1.0, 0.0])


def continuous_dynamics(time, state, control):
    del time

    position, velocity = state
    acceleration = control[0]

    del position

    return np.array([
        velocity,
        acceleration,
    ])


dynamics = discretize_dynamics(
    continuous_dynamics,
    dt=dt,
    method="rk4",
)

Q = np.diag([0.1, 0.01])
R = np.array([[0.01]])
Qf = np.diag([200.0, 50.0])


def running_cost(state, control, k):
    del k
    error = state - target

    return dt * (
        0.5 * error @ Q @ error
        + 0.5 * control @ R @ control
    )


def terminal_cost(state):
    error = state - target
    return 0.5 * error @ Qf @ error


problem = BoxILQRProblem(
    dynamics=dynamics,
    running_cost=running_cost,
    terminal_cost=terminal_cost,
    horizon=horizon,
    state_dim=2,
    control_dim=1,
    state_bounds=(
        np.array([-1.5, -1.0]),
        np.array([1.5, 1.0]),
    ),
    control_bounds=(
        np.array([-0.5]),
        np.array([0.5]),
    ),
    name="bounded_double_integrator",
)

options = BoxILQROptions(
    ilqr=ILQROptions(
        max_iterations=100,
        relative_cost_tolerance=1.0e-8,
        feedforward_tolerance=1.0e-5,
    ),
    barrier=BarrierOptions(
        initial_state_barrier=0.1,
        initial_control_barrier=0.1,
        state_reduction_factor=0.2,
        control_reduction_factor=0.2,
        tolerance=1.0e-4,
    ),
    verbose=True,
)

u_initial = np.zeros((horizon, 1))

solution = solve(
    problem=problem,
    x0=x0,
    u_initial=u_initial,
    options=options,
)

print(solution.status)
print(solution.states[-1])

x_opt = solution.states
u_opt = solution.controls
K_opt = solution.feedback_gains
```

## Solver output

The `BoxILQRSolution` object contains:

| Attribute | Description |
|---|---|
| `states` | Nominal state trajectory with shape `(N + 1, n)` |
| `controls` | Nominal control trajectory with shape `(N, m)` |
| `feedback_gains` | Time-varying gains with shape `(N, m, n)` |
| `feedforward_terms` | Feedforward corrections with shape `(N, m)` |
| `original_cost` | Objective value without barrier terms |
| `augmented_cost` | Objective value including barrier terms |
| `barrier_cost` | Total state and control barrier contribution |
| `status` | Solver termination status |
| `success` | Whether the full continuation procedure converged |
| `history` | Stored inner and outer iteration diagnostics |

## Run the example

From the main project directory, run:

```bash
python examples/double_integrator.py
```

The bounded double-integrator example demonstrates:

- Continuous dynamics discretized using RK4
- State and control box constraints
- Barrier continuation
- Near-active control bounds
- Nominal trajectory and control plots
- Convergence-history plots

Generated figures are placed in the local `results/` directory.

## Run the tests

Activate the virtual environment and run:

```bash
source ilqrenv/bin/activate
python -m pytest -q
```

To run a specific test file:

```bash
python -m pytest -q tests/test_barriers.py
```

## Current package structure

```text
Box-iLQR/
├── pyproject.toml
├── README.md
├── LICENSE
├── src/
│   └── boxilqr/
│       ├── __init__.py
│       ├── problem.py
│       ├── options.py
│       ├── solution.py
│       ├── integrators.py
│       ├── barriers.py
│       ├── derivatives.py
│       ├── rollout.py
│       ├── backward_pass.py
│       ├── forward_pass.py
│       ├── solver.py
│       └── diagnostics.py
├── examples/
│   └── double_integrator.py
└── tests/
```

## Current limitations

- The current derivative backend uses central finite differences.
- A strictly feasible initial state-control trajectory is required.
- Only componentwise box constraints are currently supported.
- Terminal equality constraints are not yet handled directly.
- Final time is fixed and is not an optimization variable.
- The API has not yet reached a stable release.
- The package has not yet been published on PyPI.

## Planned additions

- User-supplied analytical derivatives
- Optional automatic differentiation
- Faster derivative evaluation for long-horizon aerospace problems
- Closed-loop simulation and Monte Carlo utilities
- Spacecraft attitude-control example
- Planar minimum-fuel orbit-transfer example
- Earth-Moon L2-to-L2 Halo-orbit transfer example
- Direct terminal equality constraints
- Free-final-time optimization
- Expanded documentation and API reference
- Continuous integration and public release packaging

## Research applications

The intended aerospace applications include:

- Spacecraft attitude maneuvers with actuator saturation
- Fuel-optimal orbit transfers
- Cislunar trajectory optimization
- Rendezvous and proximity operations
- Constrained trajectory tracking
- Closed-loop execution using the feedback gains obtained during optimization

## Citation

This software implements the Box-iLQR method developed in:

> Abhijeet, Tarun Hejmadi, and Suman Chakravorty, “An Efficient iLQR Algorithm for Optimal Control with Bounded States and Inputs: Application to Spacecraft Maneuvers.”

Complete publication and BibTeX information will be added when available.

## License

The software license will be finalized before the first public release.

## Contact

For research questions, implementation feedback, or collaboration inquiries, please open an issue after the public repository is released.
