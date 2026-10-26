//! # Reference example: Terzaghi 1D consolidation of a clay layer (#178)
//!
//! A saturated clay layer is loaded at t = 0. The load is first carried by
//! the pore water, as an excess pore pressure, which then dissipates as the
//! water drains out of the layer. The example follows that dissipation in
//! physical units and checks it against Terzaghi's series solution, both
//! the pressure profiles (isochrones) and the average degree of
//! consolidation U(Tv).
//!
//! Run with: `cargo run --example consolidation_1d`. The example exits with
//! an error if a check exceeds its tolerance.
//!
//! ## Physics
//!
//! With u(z, t) the excess pore pressure and z the depth:
//!
//! ```text
//! du/dt = cv * d2u/dz2,   0 < z < 2 H_dr
//! u(0, t) = 0,   u(2 H_dr, t) = 0   (Dirichlet, drained faces)
//! u(z, 0) = u0
//! ```
//!
//! Clay with a coefficient of consolidation cv = 1 m²/year
//! (3.169e-8 m²/s, with the Julian year of 365.25 days). The load of
//! 100 kPa is carried at first by the water alone, so u0 = 100 kPa. The
//! layer is 4 m thick, on 101 nodes (dz = 4 cm).
//!
//! The model is a single context value: the FD Laplacian of the pressure
//! (`FDLaplacianCalculator`, second order in space), scaled by cv. Both
//! faces are held by `boundary::Dirichlet` conditions on the two faces of
//! the grid; the saved states carry the prescribed values, the initial
//! state included.
//!
//! ## Drained and undrained base
//!
//! The run reads in two ways, with the same drainage path H_dr = 2 m:
//!
//! - a 4 m layer between two sand layers, drained at the top and at the
//!   base;
//! - a 2 m layer on an impermeable base, drained at the top only: the upper
//!   half of the run.
//!
//! The second reading rests on symmetry. The run is symmetric about
//! z = H_dr, so no water crosses the mid-plane, which therefore acts as an
//! impermeable base. On the grid, the mid-plane node has symmetric
//! neighbours, which is the second-order mirror form of a zero-flux
//! condition. This avoids a zero-flux boundary condition, which the crate
//! does not provide on boundary regions yet. The example prints the
//! largest departure from symmetry, reported, not checked: at most
//! 4.3e-7 kPa when the example was written.
//!
//! ## Terzaghi's series solution
//!
//! With the time factor Tv = cv t / H_dr², M_m = pi (2m + 1) / 2 and z
//! measured from the top face:
//!
//! ```text
//! u(z, t) = sum_{m >= 0} (2 u0 / M_m) * sin(M_m z / H_dr) * exp(-M_m² Tv)
//! U(Tv)   = 1 - sum_{m >= 0} (2 / M_m²) * exp(-M_m² Tv)
//! ```
//!
//! Over the whole layer, 0 < z < 2 H_dr, the sum for u is symmetric about
//! the mid-plane: since 2 M_m = pi (2m + 1), sin(M_m (2 H_dr - z) / H_dr)
//! equals sin(M_m z / H_dr). It covers both readings without a separate
//! formula for the lower half.
//!
//! U is the average degree of consolidation: one minus the mean excess
//! pressure over u0. On the computed field, the mean is taken by the
//! trapezoidal rule over the whole layer.
//!
//! Both series are truncated with an explicit bound on what is left out.
//! For u, from one term to the next the bound
//! 2 u0 / M_m * exp(-M_m² Tv) shrinks by at least exp(-2 pi² Tv), since
//! M_{m+1}² - M_m² = 2 pi² (m + 1). The sum stops when the bound of the
//! next term falls below 1e-12 kPa, and the neglected tail is at most that
//! bound divided by 1 - exp(-2 pi² Tv): 1.6 times the bound at Tv = 0.05,
//! the earliest time compared. For U, the coefficients 2 / M_m² add up to
//! 1, so the tail after the last term kept is at most exp(-M² Tv) for the
//! first neglected M; the sum stops when that falls below 1e-14.
//!
//! ## Integrator
//!
//! The initial data are discontinuous at the drained faces. BDF2 is
//! L-stable and damps the stiffest modes that discontinuity excites. The
//! time step is 0.005 in time factor, about 7.3 days, so that
//! cv*dt/dz² = 12.5. One Newton correction converges on this linear problem
//! at that value, and the solver keeps its default budget of one
//! iteration. The run goes to Tv = 1, four years.
//!
//! At early times the profile error is dominated by the discontinuity at
//! the drained faces: at Tv = 0.05 the example prints it without checking
//! it. The degree of consolidation, an average, is already accurate then
//! and is checked.
//!
//! ## Validation
//!
//! | Check | Checked at | Tolerance |
//! |---|---|---|
//! | Isochrones, max over the nodes of abs(u - series) | Tv = 0.1, 0.2, 0.5, 1 | 3e-2 kPa |
//! | Degree of consolidation, abs(U - series) | Tv = 0.05, 0.1, 0.2, 0.5, 1 | 1e-4 |
//!
//! Errors measured when the example was written: isochrones at most
//! 1.31e-2 kPa (Tv = 0.1); degree of consolidation at most 3.4e-5
//! (Tv = 0.1). The isochrone error converges at second order in space
//! (measured on 51, 101 and 201 nodes at Tv = 0.1 and 0.5).

use std::error::Error;
use std::f64::consts::PI;
use std::sync::Arc;

use nalgebra::DVector;
use oxiflow::{
    boundary::{BoundaryCondition, Dirichlet},
    context::{
        calculators::FDLaplacianCalculator, compute::ComputeContext, error::OxiflowError,
        value::ContextValue, variable::ContextVariable,
    },
    mesh::{BoundaryRegion, FaceSide, Mesh, UniformGrid1D},
    model::traits::{PhysicalModel, RequiresContext},
    solver::{
        config::{IntegratorKind, StepControl, TimeConfiguration},
        methods::BDF2Solver,
        scenario::{Domain, Scenario},
        SimulationResult, Solver, SolverConfiguration,
    },
};

// ── Model ─────────────────────────────────────────────────────────────────

/// Context slot of the FD Laplacian of the pressure.
fn laplacian_variable() -> ContextVariable {
    ContextVariable::External {
        name: "laplacian".into(),
    }
}

/// `du/dt = cv * d2u/dz2`, from the FD Laplacian of the excess pore
/// pressure.
struct Consolidation1D {
    /// Coefficient of consolidation, in m²/s.
    cv: f64,
    /// Uniform initial excess pore pressure, in kPa.
    initial_pressure: f64,
}

impl RequiresContext for Consolidation1D {
    fn required_variables(&self) -> Vec<ContextVariable> {
        vec![laplacian_variable()]
    }
}

impl PhysicalModel for Consolidation1D {
    fn compute_physics(
        &self,
        _state: &ContextValue,
        ctx: &ComputeContext,
    ) -> Result<ContextValue, OxiflowError> {
        let laplacian = ctx.external(laplacian_variable())?.as_scalar_field()?;
        Ok(ContextValue::ScalarField(laplacian.map(|v| self.cv * v)))
    }

    /// The whole load carried by the pore water at t = 0.
    fn initial_state(&self, mesh: &dyn Mesh) -> ContextValue {
        ContextValue::ScalarField(DVector::from_element(mesh.n_dof(), self.initial_pressure))
    }

    fn name(&self) -> &str {
        "consolidation_1d"
    }
}

// ── Clay layer ────────────────────────────────────────────────────────────

/// Julian year, s.
const YEAR: f64 = 365.25 * 86_400.0;
/// Coefficient of consolidation, m²/s (1 m²/year).
const CV: f64 = 1.0 / YEAR;
/// Load, carried at first by the pore water: initial excess pressure, kPa.
const LOAD: f64 = 100.0;
/// Drainage path H_dr, m: half the layer thickness.
const DRAINAGE_PATH: f64 = 2.0;
/// Layer thickness, m, drained at both faces.
const THICKNESS: f64 = 2.0 * DRAINAGE_PATH;
/// Grid nodes, faces included (dz = 4 cm).
const N_NODES: usize = 101;

/// Time of one unit of time factor, H_dr² / cv, s (four years).
const TIME_SCALE: f64 = DRAINAGE_PATH * DRAINAGE_PATH / CV;
/// Time step, s: 0.005 in time factor, about 7.3 days.
const DT: f64 = 0.005 * TIME_SCALE;
const END_TIME: f64 = TIME_SCALE;
/// One saved state every 0.05 in time factor.
const SAVE_EVERY: usize = 10;
const ISOCHRONE_TIME_FACTORS: [f64; 4] = [0.1, 0.2, 0.5, 1.0];
const CONSOLIDATION_TIME_FACTORS: [f64; 5] = [0.05, 0.1, 0.2, 0.5, 1.0];
/// Reported, not checked (see the module documentation).
const EARLY_TIME_FACTOR: f64 = 0.05;
/// Tolerance on the isochrones, kPa.
const ISOCHRONE_TOLERANCE: f64 = 3e-2;
/// Tolerance on the degree of consolidation.
const CONSOLIDATION_TOLERANCE: f64 = 1e-4;

fn layer() -> Result<UniformGrid1D, OxiflowError> {
    UniformGrid1D::new(N_NODES, 0.0, THICKNESS)
}

/// z of every node.
fn positions(mesh: &dyn Mesh) -> Vec<f64> {
    (0..mesh.n_dof()).map(|i| mesh.coordinates(i)[0]).collect()
}

fn face(side: FaceSide) -> BoundaryRegion {
    BoundaryRegion::Face { axis: 0, side }
}

/// Solves the dissipation of the excess pore pressure.
fn solve() -> Result<SimulationResult, OxiflowError> {
    let conditions: Vec<Box<dyn BoundaryCondition>> = vec![
        Box::new(Dirichlet::constant(face(FaceSide::Lower), 0.0)),
        Box::new(Dirichlet::constant(face(FaceSide::Upper), 0.0)),
    ];
    let model = Consolidation1D {
        cv: CV,
        initial_pressure: LOAD,
    };
    let domain = Domain::new("clay", Box::new(model), Box::new(layer()?))
        .with_boundary_conditions(conditions);
    let scenario = Scenario::multi(vec![domain])?;

    let mesh: Arc<dyn Mesh> = Arc::new(layer()?);
    let config = SolverConfiguration::new(
        TimeConfiguration::new(END_TIME, StepControl::Fixed { dt: DT }).saving_every(SAVE_EVERY),
        IntegratorKind::BDF2,
    )
    .with_calculator(Box::new(FDLaplacianCalculator::new(
        mesh,
        laplacian_variable(),
    )));
    BDF2Solver::new().solve(&scenario, &config)
}

// ── Series solution ───────────────────────────────────────────────────────

/// M_m = pi (2m + 1) / 2.
fn eigenvalue(m: u32) -> f64 {
    PI * f64::from(2 * m + 1) / 2.0
}

/// Terzaghi's excess pore pressure at depth `z` (from the top face) and
/// time factor `tv`, over the whole layer.
fn series_pressure(z: f64, tv: f64) -> f64 {
    let mut sum = 0.0;
    let mut m = 0;
    loop {
        let eigen = eigenvalue(m);
        let bound = 2.0 * LOAD / eigen * (-eigen * eigen * tv).exp();
        if bound < 1e-12 {
            break;
        }
        sum += bound * (eigen * z / DRAINAGE_PATH).sin();
        m += 1;
    }
    sum
}

/// Terzaghi's average degree of consolidation at time factor `tv`.
fn series_consolidation(tv: f64) -> f64 {
    let mut sum = 0.0;
    let mut m = 0;
    loop {
        let eigen = eigenvalue(m);
        let decay = (-eigen * eigen * tv).exp();
        if decay < 1e-14 {
            break;
        }
        sum += 2.0 / (eigen * eigen) * decay;
        m += 1;
    }
    1.0 - sum
}

// ── Checks ────────────────────────────────────────────────────────────────

/// The saved state at time factor `tv`, or an error naming it.
fn state_at(result: &SimulationResult, tv: f64) -> Result<&DVector<f64>, Box<dyn Error>> {
    let time = tv * TIME_SCALE;
    let index = result
        .times
        .iter()
        .position(|&t| (t - time).abs() <= 1e-9 * time)
        .ok_or_else(|| format!("no state saved at Tv = {tv}"))?;
    Ok(result.states[index].as_scalar_field()?)
}

/// max |numerical - series| over the nodes, kPa.
fn isochrone_error(pressure: &DVector<f64>, zs: &[f64], tv: f64) -> f64 {
    pressure
        .iter()
        .zip(zs)
        .map(|(value, &z)| (value - series_pressure(z, tv)).abs())
        .fold(0.0, f64::max)
}

/// One minus the mean excess pressure over u0, trapezoidal rule.
fn degree_of_consolidation(pressure: &DVector<f64>, zs: &[f64]) -> f64 {
    let integral: f64 = (0..pressure.len() - 1)
        .map(|i| 0.5 * (pressure[i] + pressure[i + 1]) * (zs[i + 1] - zs[i]))
        .sum();
    1.0 - integral / (LOAD * THICKNESS)
}

/// Largest |u(z) - u(2 H_dr - z)|, kPa.
fn symmetry_departure(pressure: &DVector<f64>) -> f64 {
    let n = pressure.len();
    (0..n / 2)
        .map(|i| (pressure[i] - pressure[n - 1 - i]).abs())
        .fold(0.0, f64::max)
}

/// Fails with the quantity, the time factor, the error and the tolerance.
fn check(quantity: &str, tv: f64, error: f64, tolerance: f64) -> Result<(), Box<dyn Error>> {
    if error <= tolerance {
        Ok(())
    } else {
        Err(format!(
            "{quantity}: error {error:.3e} at Tv = {tv} exceeds the tolerance {tolerance:.1e}"
        )
        .into())
    }
}

fn main() -> Result<(), Box<dyn Error>> {
    let zs = positions(&layer()?);
    println!(
        "Clay layer of {THICKNESS} m drained at both faces (or {DRAINAGE_PATH} m on an impermeable base), cv = 1 m²/year, u0 = {LOAD} kPa"
    );
    println!(
        "BDF2, dt = {:.2} days, one unit of Tv = {:.1} years",
        DT / 86_400.0,
        TIME_SCALE / YEAR
    );
    let result = solve()?;

    let early = state_at(&result, EARLY_TIME_FACTOR)?;
    println!(
        "  Tv = {EARLY_TIME_FACTOR:<4}: max |u - series| = {:.3e} kPa (reported)",
        isochrone_error(early, &zs, EARLY_TIME_FACTOR)
    );
    for &tv in &ISOCHRONE_TIME_FACTORS {
        let pressure = state_at(&result, tv)?;
        let error = isochrone_error(pressure, &zs, tv);
        println!(
            "  Tv = {tv:<4} (t = {:.2} years): max |u - series| = {error:.3e} kPa, symmetry departure {:.1e} kPa (reported)",
            tv * TIME_SCALE / YEAR,
            symmetry_departure(pressure)
        );
        check("isochrone (kPa)", tv, error, ISOCHRONE_TOLERANCE)?;
    }
    for &tv in &CONSOLIDATION_TIME_FACTORS {
        let numerical = degree_of_consolidation(state_at(&result, tv)?, &zs);
        let expected = series_consolidation(tv);
        let error = (numerical - expected).abs();
        println!(
            "  Tv = {tv:<4}: U = {numerical:.6} (series {expected:.6}), |error| = {error:.2e}"
        );
        check(
            "degree of consolidation",
            tv,
            error,
            CONSOLIDATION_TOLERANCE,
        )?;
    }
    println!("consolidation_1d: every check within its tolerance");
    Ok(())
}
