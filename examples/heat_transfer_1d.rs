//! # Reference example: transient heat transfer in a 1D bar (#175)
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
//! The graded conductivity of case (b) is defined once in
//! `rsc/models/graded_conductivity.rs`, included below as the
//! `graded_conductivity` module and shared with the coefficient-field
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
//! placed (DD-066), next to `ExternalTabulated`.
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
use std::f64::consts::PI;
use std::sync::Arc;

use nalgebra::DVector;
use oxiflow::{
    boundary::{BoundaryCondition, Dirichlet},
    context::{
        calculators::{CoefficientField, FDGradientCalculator, FDLaplacianCalculator, FDScheme},
        compute::ComputeContext,
        error::OxiflowError,
        value::ContextValue,
        variable::ContextVariable,
    },
    mesh::{BoundaryRegion, FaceSide, Mesh, UniformGrid1D},
    model::traits::{PhysicalModel, RequiresContext},
    solver::{
        config::{IntegratorKind, StepControl, TimeConfiguration},
        methods::{BDF2Solver, BackwardEulerSolver, CrankNicolsonSolver},
        scenario::{Domain, Scenario},
        SimulationResult, Solver, SolverConfiguration,
    },
};

#[path = "../rsc/models/graded_conductivity.rs"]
mod graded_conductivity;

use graded_conductivity::GradedConductivity;

// ── Model ─────────────────────────────────────────────────────────────────

/// Context slot of the FD Laplacian of the temperature.
fn laplacian_variable() -> ContextVariable {
    ContextVariable::External {
        name: "laplacian".into(),
    }
}

/// Context slot of the centred FD gradient of the temperature.
fn gradient_variable() -> ContextVariable {
    ContextVariable::SpatialGradient {
        dimension: 0,
        component: None,
    }
}

/// Name of the conductivity coefficient field lambda(x).
const CONDUCTIVITY: &str = "conductivity";

/// Name of the coefficient field lambda'(x), the derivative of the
/// conductivity along the bar.
const CONDUCTIVITY_SLOPE: &str = "conductivity_slope";

fn coefficient_variable(name: &'static str) -> ContextVariable {
    ContextVariable::Coefficient { name: name.into() }
}

/// `rho*c_p*dT/dt = lambda*T'' + lambda'*T'`: the conduction term in
/// non-conservative form, from the Laplacian and the centred gradient of
/// the temperature and the two coefficient fields lambda and lambda'.
struct HeatConduction1D {
    /// Volumetric heat capacity rho*c_p, in J/(m^3*K).
    rho_cp: f64,
    /// Uniform initial temperature, in degrees Celsius.
    initial_temperature: f64,
}

impl RequiresContext for HeatConduction1D {
    fn required_variables(&self) -> Vec<ContextVariable> {
        vec![
            laplacian_variable(),
            gradient_variable(),
            coefficient_variable(CONDUCTIVITY),
            coefficient_variable(CONDUCTIVITY_SLOPE),
        ]
    }
}

impl PhysicalModel for HeatConduction1D {
    fn compute_physics(
        &self,
        _state: &ContextValue,
        ctx: &ComputeContext,
    ) -> Result<ContextValue, OxiflowError> {
        let laplacian = ctx.external(laplacian_variable())?.as_scalar_field()?;
        let gradient = ctx.gradient(0)?;
        let lambda = ctx
            .external(coefficient_variable(CONDUCTIVITY))?
            .as_scalar_field()?;
        let slope = ctx
            .external(coefficient_variable(CONDUCTIVITY_SLOPE))?
            .as_scalar_field()?;
        let rate = DVector::from_iterator(
            laplacian.len(),
            (0..laplacian.len())
                .map(|i| (lambda[i] * laplacian[i] + slope[i] * gradient[i]) / self.rho_cp),
        );
        Ok(ContextValue::ScalarField(rate))
    }

    fn initial_state(&self, mesh: &dyn Mesh) -> ContextValue {
        ContextValue::ScalarField(DVector::from_element(
            mesh.n_dof(),
            self.initial_temperature,
        ))
    }

    fn name(&self) -> &str {
        "heat_transfer_1d"
    }
}

// ── Material and bar ──────────────────────────────────────────────────────

/// Density of carbon steel, kg/m³.
const DENSITY: f64 = 7850.0;
/// Specific heat capacity of carbon steel, J/(kg·K).
const SPECIFIC_HEAT: f64 = 460.0;
/// Conductivity of carbon steel, W/(m·K); lambda(0) in case (b).
const LAMBDA0: f64 = 50.0;
/// Bar length, m.
const LENGTH: f64 = 0.1;
/// Grid nodes, ends included (dx = 1 mm).
const N_NODES: usize = 101;
/// Newton budget of every implicit solver (see the module documentation).
const NEWTON_ITERATIONS: usize = 5;

// ── Case (a) ──────────────────────────────────────────────────────────────

const COOLING_INITIAL: f64 = 100.0;
const COOLING_ENDS: f64 = 20.0;
const COOLING_DT: f64 = 0.5;
const COOLING_END_TIME: f64 = 120.0;
/// One saved state every 10 s.
const COOLING_SAVE_EVERY: usize = 20;
const COOLING_CHECK_TIMES: [f64; 3] = [30.0, 60.0, 120.0];
/// Reported, not checked (see the module documentation).
const COOLING_EARLY_TIME: f64 = 10.0;
const COOLING_TOLERANCE: f64 = 1e-2;

// ── Case (b) ──────────────────────────────────────────────────────────────

const GRADED_BETA: f64 = 1.0;
const GRADED_HOT: f64 = 100.0;
const GRADED_COLD: f64 = 20.0;
const GRADED_DT: f64 = 50.0;
const GRADED_END_TIME: f64 = 3000.0;
/// One saved state every 500 s.
const GRADED_SAVE_EVERY: usize = 10;
const GRADED_TOLERANCE: f64 = 5e-4;

fn rho_cp() -> f64 {
    DENSITY * SPECIFIC_HEAT
}

fn bar() -> Result<UniformGrid1D, OxiflowError> {
    UniformGrid1D::new(N_NODES, 0.0, LENGTH)
}

/// x of every node.
fn positions(mesh: &dyn Mesh) -> Vec<f64> {
    (0..mesh.n_dof()).map(|i| mesh.coordinates(i)[0]).collect()
}

fn face(side: FaceSide) -> BoundaryRegion {
    BoundaryRegion::Face { axis: 0, side }
}

/// lambda(x) = lambda0 (1 + beta x / L); case (a) is beta = 0.
fn conductivity(beta: f64) -> GradedConductivity {
    GradedConductivity {
        lambda0: LAMBDA0,
        beta,
        length: LENGTH,
    }
}

/// What differs between the runs of the example.
struct Run {
    initial: f64,
    lower: f64,
    upper: f64,
    conductivity: GradedConductivity,
    integrator: IntegratorKind,
    dt: f64,
    end_time: f64,
    save_every: usize,
}

/// Solves one run on the bar.
fn solve(run: &Run, solver: &dyn Solver) -> Result<SimulationResult, OxiflowError> {
    let model = HeatConduction1D {
        rho_cp: rho_cp(),
        initial_temperature: run.initial,
    };
    let conditions: Vec<Box<dyn BoundaryCondition>> = vec![
        Box::new(Dirichlet::constant(face(FaceSide::Lower), run.lower)),
        Box::new(Dirichlet::constant(face(FaceSide::Upper), run.upper)),
    ];
    let domain =
        Domain::new("bar", Box::new(model), Box::new(bar()?)).with_boundary_conditions(conditions);
    let scenario = Scenario::multi(vec![domain])?;

    let mesh: Arc<dyn Mesh> = Arc::new(bar()?);
    let lambda = run.conductivity;
    let config = SolverConfiguration::new(
        TimeConfiguration::new(run.end_time, StepControl::Fixed { dt: run.dt })
            .saving_every(run.save_every),
        run.integrator.clone(),
    )
    .with_calculator(Box::new(FDLaplacianCalculator::new(
        Arc::clone(&mesh),
        laplacian_variable(),
    )))
    .with_calculator(Box::new(FDGradientCalculator::new(
        Arc::clone(&mesh),
        0,
        None,
        FDScheme::Central,
    )))
    .with_calculator(Box::new(CoefficientField::of_x(
        Arc::clone(&mesh),
        CONDUCTIVITY,
        move |x| lambda.value(x[0]),
    )?))
    .with_calculator(Box::new(CoefficientField::of_x(
        mesh,
        CONDUCTIVITY_SLOPE,
        move |_| lambda.slope(),
    )?));
    solver.solve(&scenario, &config)
}

/// The saved state at `time`, or an error naming the run.
fn state_at<'a>(
    result: &'a SimulationResult,
    time: f64,
    label: &str,
) -> Result<&'a DVector<f64>, Box<dyn Error>> {
    let index = result
        .times
        .iter()
        .position(|&t| (t - time).abs() <= 1e-9 * time.max(1.0))
        .ok_or_else(|| format!("{label}: no state saved at t = {time} s"))?;
    Ok(result.states[index].as_scalar_field()?)
}

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

/// Fourier-series temperature of case (a).
fn fourier_cooling(x: f64, t: f64) -> f64 {
    let alpha = LAMBDA0 / rho_cp();
    let step = COOLING_INITIAL - COOLING_ENDS;
    let mut sum = 0.0;
    let mut k = 1.0_f64;
    loop {
        let amplitude = 4.0 * step / (k * PI);
        let decay = (-(k * PI / LENGTH).powi(2) * alpha * t).exp();
        if (amplitude * decay).abs() < 1e-12 {
            break;
        }
        sum += amplitude * (k * PI * x / LENGTH).sin() * decay;
        k += 2.0;
    }
    COOLING_ENDS + sum
}

/// Exact steady temperature of case (b).
fn graded_steady(x: f64) -> f64 {
    let profile = (1.0 + GRADED_BETA * x / LENGTH).ln() / (1.0 + GRADED_BETA).ln();
    GRADED_HOT - (GRADED_HOT - GRADED_COLD) * profile
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
        let run = Run {
            initial: COOLING_INITIAL,
            lower: COOLING_ENDS,
            upper: COOLING_ENDS,
            conductivity: conductivity(0.0),
            integrator: integrator.clone(),
            dt: COOLING_DT,
            end_time: COOLING_END_TIME,
            save_every: COOLING_SAVE_EVERY,
        };
        let label = format!("(a) {name}");
        let result = solve(&run, solver.as_ref())?;

        // Early behaviour, reported only: one step from the discontinuous
        // initial data, then t = 10 s.
        let first = solve(
            &Run {
                end_time: COOLING_DT,
                save_every: 1,
                integrator,
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
    let profile = conductivity(GRADED_BETA);
    println!(
        "(b) graded bar: lambda from {} to {} W/(m·K), ends held at {GRADED_HOT} and {GRADED_COLD} °C, dt = {GRADED_DT} s",
        profile.value(0.0),
        profile.value(LENGTH)
    );
    let run = Run {
        initial: GRADED_COLD,
        lower: GRADED_HOT,
        upper: GRADED_COLD,
        conductivity: profile,
        integrator: IntegratorKind::BackwardEuler,
        dt: GRADED_DT,
        end_time: GRADED_END_TIME,
        save_every: GRADED_SAVE_EVERY,
    };
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
