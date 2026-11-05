//! # Reference example: transient heat transfer in a 1D bar ([#175](https://github.com/biface/oxiflow/issues/175))
//!
//! Heat conduction in a steel bar, in physical units, checked against two
//! exact solutions:
//!
//! - **(a) cooling of a homogeneous bar.** The bar starts at 100 °C and both
//!   ends are held at 20 °C from t = 0. Validated against the Fourier-series
//!   solution, with Crank–Nicolson and BDF2.
//! - **(b) graded bar.** The conductivity increases linearly along the bar.
//!   The bar starts at 20 °C, the end x = 0 is held at 100 °C and the end
//!   x = L at 20 °C. Validated against the exact steady profile, reached by
//!   Backward Euler with large steps.
//!
//! Run with: `cargo run --example heat_transfer_1d`. The example exits with
//! an error if a check exceeds its tolerance.
//!
//! The model, the material, the two runs and their exact solutions are
//! defined once in `rsc/models/heat_transfer_1d.rs`, included below as the
//! `heat_transfer_1d` module and shared with `heat_transfer_1d_plot`. The
//! graded conductivity of case (b) comes from
//! `rsc/models/graded_conductivity.rs`, shared with the coefficient-field
//! dispatch benchmark.
//!
//! ## Physics
//!
//! ```text
//! rho*c_p * dT/dt = d/dx( lambda(x) * dT/dx ) = lambda * T'' + lambda' * T',   0 < x < L
//! T(0, t) = T_lower,   T(L, t) = T_upper   (Dirichlet, both ends)
//! T(x, 0) = T_initial
//! ```
//!
//! Material: carbon steel, rho = 7850 kg/m³, c_p = 460 J/(kg·K),
//! lambda0 = 50 W/(m·K), so the diffusivity is
//! alpha = lambda0 / (rho*c_p) ≈ 1.385e-5 m²/s. Bar length L = 0.1 m, 101
//! nodes (dx = 1 mm). The slowest mode of case (a) decays with the time
//! constant L² / (π² alpha) ≈ 73 s.
//!
//! The conduction term is written in non-conservative form, lambda*T'' +
//! lambda'*T'. Each factor is a context value: the FD Laplacian
//! (`FDLaplacianCalculator`), the centred FD gradient
//! (`FDGradientCalculator`, `FDScheme::Central`) and two coefficient fields
//! (`CoefficientField::of_x`), lambda and its derivative lambda'. Both
//! stencils are second order in space. lambda' is given in closed form
//! rather than differentiated numerically. The same model serves both
//! cases: in case (a), lambda is constant and lambda' is zero.
//!
//! The coefficient fields are calculators, attached to the
//! `SolverConfiguration` like every context calculator, although lambda(x)
//! is a material property. This follows from where coefficient fields were
//! placed ([DD-066](https://github.com/biface/oxiflow/issues/181)), next to `ExternalTabulated`.
//!
//! Both ends are held by `boundary::Dirichlet` conditions on the two faces
//! of the grid. The solvers hold the constrained nodes exactly, and the
//! saved states carry the prescribed values, the initial state included.
//!
//! ## (a) Cooling of a homogeneous bar: Fourier series
//!
//! With theta = T - T_ends and the step theta(x, 0) = T_initial - T_ends
//! (Carslaw & Jaeger, *Conduction of Heat in Solids*, 1959):
//!
//! ```text
//! T(x, t) = T_ends + sum_{k odd} 4 (T_initial - T_ends) / (k pi)
//!                     * sin(k pi x / L) * exp(-(k pi / L)² alpha t)
//! ```
//!
//! The series is summed until the bound of the next term,
//! 4 |T_initial - T_ends| / (k pi) * exp(-(k pi / L)² alpha t), falls below
//! 1e-12 °C. From one odd k to the next, that bound shrinks by at least
//! exp(-8 (pi / L)² alpha t), so the neglected tail is at most the bound
//! of the first neglected term divided by 1 - exp(-8 (pi / L)² alpha t):
//! about 1.5 times that bound at t = 10 s, the earliest time compared.
//!
//! The initial data are discontinuous at the ends. Crank–Nicolson is
//! A-stable but not L-stable: for the stiffest modes its amplification
//! factor tends to -1 as alpha*dt/dx² grows, so the discontinuity leaves
//! oscillations that are damped slowly. BDF2 is L-stable and damps them
//! within a few steps. With dt = 0.5 s (alpha*dt/dx² ≈ 6.9), both solvers
//! agree with the series to within the tolerance from t = 30 s, but at
//! t = 10 s the Crank–Nicolson error is still more than ten times the BDF2
//! one. The example prints both, along with the minimum temperature after
//! the first step: Crank–Nicolson undershoots the end temperature, which
//! the exact solution never does. These early values are reported, not
//! checked.
//!
//! ## (b) Graded bar: exact steady profile
//!
//! lambda(x) = lambda0 (1 + beta x / L), beta = 1, so the conductivity
//! doubles from one end to the other. At steady state the heat flux
//! q = -lambda T' is uniform, which gives
//!
//! ```text
//! T(x) = T_lower - (T_lower - T_upper) * ln(1 + beta x / L) / ln(1 + beta)
//! ```
//!
//! Backward Euler with dt = 50 s reaches it: the slowest mode decays by a
//! factor of 1.7 per step or more (1.7 is the value for a uniform
//! lambda0, the lowest conductivity of the bar), and the transient is below
//! the spatial error after about 1500 s. The run goes to 3000 s. Omitting
//! the lambda'T' term would give an error of several degrees.
//!
//! ## Newton iterations
//!
//! The problem is linear, so one Newton correction would be exact with an
//! exact Jacobian. The implicit solvers build theirs by finite differences,
//! and the rounding of those differences grows with alpha*dt/dx²: with
//! dt = 50 s (alpha*dt/dx² ≈ 690), the residual after one correction is
//! about 1.5e-2, above the default tolerance, and the solvers' default
//! budget of one iteration reports `NewtonNotConverged`. A second
//! correction absorbs it. Every implicit solver here is given a budget of
//! `NEWTON_ITERATIONS`; convergence is checked after each correction, so a
//! step that converges at once costs nothing more.
//!
//! ## Validation
//!
//! | Case | Solvers | Checked at | Tolerance (max over the nodes) |
//! |---|---|---|---|
//! | (a) | Crank–Nicolson, BDF2, dt = 0.5 s | t = 30, 60, 120 s | 1e-2 °C |
//! | (b) | Backward Euler, dt = 50 s | t = 3000 s | 5e-4 °C |
//!
//! Errors measured when the example was written: (a) at most 4.1e-3 °C
//! (Crank–Nicolson at 30 s); (b) 6.1e-5 °C. Both cases converge at second
//! order in space (measured on 51, 101 and 201 nodes).

use std::error::Error;

use nalgebra::DVector;
use oxiflow::solver::{
    config::IntegratorKind,
    methods::{BDF2Solver, BackwardEulerSolver, CrankNicolsonSolver},
    Solver,
};

#[path = "../rsc/models/graded_conductivity.rs"]
mod graded_conductivity;
#[path = "../rsc/models/heat_transfer_1d.rs"]
mod heat_transfer_1d;

use heat_transfer_1d::{
    bar, cooling_run, fourier_cooling, graded_run, graded_steady, positions, solve, state_at, Run,
    COOLING_DT, COOLING_ENDS, COOLING_INITIAL, GRADED_COLD, GRADED_DT, GRADED_END_TIME, GRADED_HOT,
    LENGTH, NEWTON_ITERATIONS,
};

// ── Checks ────────────────────────────────────────────────────────────────

/// Case (a): times checked against the Fourier series.
const COOLING_CHECK_TIMES: [f64; 3] = [30.0, 60.0, 120.0];
/// Reported, not checked (see the module documentation).
const COOLING_EARLY_TIME: f64 = 10.0;
const COOLING_TOLERANCE: f64 = 1e-2;
/// Case (b): tolerance at the end of the run, against the steady profile.
const GRADED_TOLERANCE: f64 = 5e-4;

/// max |numerical - exact| over the nodes.
fn max_error(numerical: &DVector<f64>, xs: &[f64], exact: impl Fn(f64) -> f64) -> f64 {
    numerical
        .iter()
        .zip(xs)
        .map(|(value, &x)| (value - exact(x)).abs())
        .fold(0.0, f64::max)
}

/// Fails with the run, the time, the error and the tolerance.
fn check(label: &str, time: f64, error: f64, tolerance: f64) -> Result<(), Box<dyn Error>> {
    if error <= tolerance {
        Ok(())
    } else {
        Err(format!(
            "{label}: max |error| = {error:.3e} °C at t = {time} s exceeds the tolerance {tolerance:.1e} °C"
        )
        .into())
    }
}

fn cooling(xs: &[f64]) -> Result<(), Box<dyn Error>> {
    println!(
        "(a) cooling: steel bar, {COOLING_INITIAL} °C, ends held at {COOLING_ENDS} °C, dt = {COOLING_DT} s"
    );
    let solvers: [(&str, Box<dyn Solver>, IntegratorKind); 2] = [
        (
            "Crank-Nicolson",
            Box::new(CrankNicolsonSolver::new().with_max_newton_iterations(NEWTON_ITERATIONS)),
            IntegratorKind::CrankNicolson,
        ),
        (
            "BDF2",
            Box::new(BDF2Solver::new().with_max_newton_iterations(NEWTON_ITERATIONS)),
            IntegratorKind::BDF2,
        ),
    ];
    for (name, solver, integrator) in solvers {
        let run = cooling_run(integrator);
        let label = format!("(a) {name}");
        let result = solve(&run, solver.as_ref())?;

        // Early behaviour, reported only: one step from the discontinuous
        // initial data, then t = 10 s.
        let first = solve(
            &Run {
                end_time: COOLING_DT,
                save_every: 1,
                ..run
            },
            solver.as_ref(),
        )?;
        let after_one_step = state_at(&first, COOLING_DT, &label)?;
        let early = state_at(&result, COOLING_EARLY_TIME, &label)?;
        let early_error = max_error(early, xs, |x| fourier_cooling(x, COOLING_EARLY_TIME));
        println!(
            "  {name:<14} min T after one step = {:7.3} °C   t = {COOLING_EARLY_TIME:>5} s: max |error| = {early_error:.3e} °C (reported)",
            after_one_step.min()
        );

        for &time in &COOLING_CHECK_TIMES {
            let state = state_at(&result, time, &label)?;
            let error = max_error(state, xs, |x| fourier_cooling(x, time));
            println!("  {name:<14} t = {time:>5} s: max |error| = {error:.3e} °C");
            check(&label, time, error, COOLING_TOLERANCE)?;
        }
    }
    Ok(())
}

fn graded(xs: &[f64]) -> Result<(), Box<dyn Error>> {
    let run = graded_run();
    println!(
        "(b) graded bar: lambda from {} to {} W/(m·K), ends held at {GRADED_HOT} and {GRADED_COLD} °C, dt = {GRADED_DT} s",
        run.conductivity.value(0.0),
        run.conductivity.value(LENGTH)
    );
    let solver = BackwardEulerSolver::new().with_max_newton_iterations(NEWTON_ITERATIONS);
    let result = solve(&run, &solver)?;
    for (time, state) in result.times.iter().zip(&result.states).skip(1) {
        let error = max_error(state.as_scalar_field()?, xs, graded_steady);
        println!(
            "  Backward Euler t = {time:>6} s: max |error to the steady profile| = {error:.3e} °C"
        );
    }
    let last = state_at(&result, GRADED_END_TIME, "(b) Backward Euler")?;
    let error = max_error(last, xs, graded_steady);
    check(
        "(b) Backward Euler",
        GRADED_END_TIME,
        error,
        GRADED_TOLERANCE,
    )
}

fn main() -> Result<(), Box<dyn Error>> {
    let xs = positions(&bar()?);
    cooling(&xs)?;
    graded(&xs)?;
    println!("heat_transfer_1d: every check within its tolerance");
    Ok(())
}
