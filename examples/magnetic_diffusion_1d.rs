//! # Reference example: magnetic diffusion into a conductor (#179)
//!
//! A uniform magnetic field B0 is switched on at t = 0, parallel to the
//! flat surface of a thick conductor that holds no field beforehand. The
//! field diffuses into the metal; the example follows it in copper and in
//! aluminium, in physical units, and checks it against the exact
//! half-space solution and against the diffusive scaling of the
//! penetration depth with sqrt(eta*t).
//!
//! This is a prototype of the diffusive electromagnetic regime that the
//! future `oxiflow-em` framework targets, not a Maxwell solver: the
//! displacement current is neglected, which is the usual assumption in a
//! good conductor.
//!
//! Run with: `cargo run --example magnetic_diffusion_1d`. The example
//! exits with an error if a check exceeds its tolerance.
//!
//! ## Physics
//!
//! For the field component parallel to the surface, B(x, t), with x the
//! depth below the surface:
//!
//! ```text
//! dB/dt = eta * d2B/dx2,   eta = 1 / (mu0 * sigma),   0 < x < L
//! B(0, t) = B0,   B(L, t) = 0   (Dirichlet, both faces)
//! B(x, 0) = 0
//! ```
//!
//! mu0 is taken as 4*pi*1e-7 H/m. Since the 2019 SI revision it is a
//! measured constant; the CODATA 2018 value differs from 4*pi*1e-7 by
//! 5.4e-10 in relative terms, far below anything this example resolves.
//! Conductivities are typical values at 20 °C: copper 5.96e7 S/m
//! (eta ≈ 1.335e-2 m²/s), aluminium 3.77e7 S/m (eta ≈ 2.111e-2 m²/s).
//! B0 = 0.1 T. The plate is L = 3 cm thick, on 121 nodes (dx = 0.25 mm).
//!
//! The model is a single context value: the FD Laplacian of the field
//! (`FDLaplacianCalculator`, second order in space), scaled by eta. The
//! surface and the far face are held by `boundary::Dirichlet` conditions
//! on the two faces of the grid; the saved states carry the prescribed
//! values, the initial state included.
//!
//! ## Exact solution: half-space
//!
//! For a half-space (L infinite), the field is
//!
//! ```text
//! B(x, t) = B0 * erfc( x / (2 sqrt(eta t)) )
//! ```
//!
//! The plate is finite, with its far face held field-free. By images, the
//! plate solution differs from the half-space one by at most
//! erfc(L / (2 sqrt(eta t))) * B0 over the plate. At the last time
//! compared, t = 1 ms, that bound is 6.4e-9 * B0 in copper and
//! 3.9e-6 * B0 in aluminium, more than ten times below the tolerance.
//!
//! `erfc` is computed by the classical Chebyshev approximation `erfcc`,
//! written below. Its error, measured against a reference implementation
//! on [0, 10], is at most 8.3e-8 in absolute value.
//!
//! ## Penetration depth
//!
//! The depth at which the field has reached half its surface value,
//! delta_half(t), follows from erfc(z) = 1/2:
//!
//! ```text
//! delta_half(t) = 2 erfc^-1(1/2) * sqrt(eta t) ≈ 0.953873 * sqrt(eta t)
//! ```
//!
//! The factor 2 erfc^-1(1/2) is computed by bisection on the same `erfc`;
//! it comes out within 8.0e-8 of the exact value.
//! The example measures delta_half on the computed field, by linear
//! interpolation between the two nodes that bracket B0/2, and checks the
//! ratio delta_half / sqrt(eta t): the same value is expected at every time
//! and in both metals. It also prints the aluminium-to-copper ratio of the
//! depths, expected at sqrt(sigma_Cu / sigma_Al) ≈ 1.257, without checking
//! it.
//!
//! ## Integrator
//!
//! The initial data are discontinuous at the surface. BDF2 is L-stable and
//! damps the stiffest modes that discontinuity excites. The time step,
//! dt = 1e-5 s, gives eta*dt/dx² ≈ 2.1 in copper and 3.4 in aluminium. At
//! those values one Newton correction converges on this linear problem, so
//! the solver keeps its default budget of one iteration (compare
//! `heat_transfer_1d`, where eta*dt/dx² reaches 690 and the budget is
//! raised).
//!
//! ## Validation
//!
//! | Check | Checked at | Tolerance |
//! |---|---|---|
//! | Profile, max over the nodes of abs(B - B0 erfc) | t = 0.25, 0.5, 1 ms | 5e-5 T (5e-4 B0) |
//! | delta_half / sqrt(eta t) against 2 erfc^-1(1/2) | t = 0.25, 0.5, 1 ms | 1e-3, relative |
//!
//! Errors measured when the example was written: profile at most
//! 2.27e-5 T (copper, t = 0.25 ms); depth ratio at most 3.5e-4 (aluminium,
//! t = 0.25 ms). The profile error converges at second order in space
//! (measured on 61, 121 and 241 nodes, aluminium, t = 0.25 ms).

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

/// Context slot of the FD Laplacian of the field.
fn laplacian_variable() -> ContextVariable {
    ContextVariable::External {
        name: "laplacian".into(),
    }
}

/// `dB/dt = eta * d2B/dx2`, from the FD Laplacian of the field.
struct MagneticDiffusion1D {
    /// Magnetic diffusivity eta = 1 / (mu0 * sigma), in m²/s.
    eta: f64,
}

impl RequiresContext for MagneticDiffusion1D {
    fn required_variables(&self) -> Vec<ContextVariable> {
        vec![laplacian_variable()]
    }
}

impl PhysicalModel for MagneticDiffusion1D {
    fn compute_physics(
        &self,
        _state: &ContextValue,
        ctx: &ComputeContext,
    ) -> Result<ContextValue, OxiflowError> {
        let laplacian = ctx.external(laplacian_variable())?.as_scalar_field()?;
        Ok(ContextValue::ScalarField(laplacian.map(|v| self.eta * v)))
    }

    /// No field in the conductor before t = 0.
    fn initial_state(&self, mesh: &dyn Mesh) -> ContextValue {
        ContextValue::ScalarField(DVector::zeros(mesh.n_dof()))
    }

    fn name(&self) -> &str {
        "magnetic_diffusion_1d"
    }
}

// ── Conductors and plate ──────────────────────────────────────────────────

/// Vacuum permeability, H/m (see the module documentation).
const MU0: f64 = 4.0e-7 * PI;
/// Conductors and their electrical conductivity at 20 °C, S/m.
const CONDUCTORS: [(&str, f64); 2] = [("copper", 5.96e7), ("aluminium", 3.77e7)];
/// Field applied at the surface from t = 0, T.
const SURFACE_FIELD: f64 = 0.1;
/// Plate thickness, m.
const THICKNESS: f64 = 0.03;
/// Grid nodes, faces included (dx = 0.25 mm).
const N_NODES: usize = 121;

const DT: f64 = 1e-5;
const END_TIME: f64 = 1e-3;
/// One saved state every 0.25 ms.
const SAVE_EVERY: usize = 25;
const CHECK_TIMES: [f64; 3] = [2.5e-4, 5e-4, 1e-3];
/// Tolerance on the profile, T.
const PROFILE_TOLERANCE: f64 = 5e-5;
/// Relative tolerance on delta_half / sqrt(eta t).
const DEPTH_TOLERANCE: f64 = 1e-3;

fn diffusivity(sigma: f64) -> f64 {
    1.0 / (MU0 * sigma)
}

fn plate() -> Result<UniformGrid1D, OxiflowError> {
    UniformGrid1D::new(N_NODES, 0.0, THICKNESS)
}

/// x of every node.
fn positions(mesh: &dyn Mesh) -> Vec<f64> {
    (0..mesh.n_dof()).map(|i| mesh.coordinates(i)[0]).collect()
}

fn face(side: FaceSide) -> BoundaryRegion {
    BoundaryRegion::Face { axis: 0, side }
}

/// Solves the field step in a conductor of diffusivity `eta`.
fn solve(eta: f64) -> Result<SimulationResult, OxiflowError> {
    let conditions: Vec<Box<dyn BoundaryCondition>> = vec![
        Box::new(Dirichlet::constant(face(FaceSide::Lower), SURFACE_FIELD)),
        Box::new(Dirichlet::constant(face(FaceSide::Upper), 0.0)),
    ];
    let domain = Domain::new(
        "conductor",
        Box::new(MagneticDiffusion1D { eta }),
        Box::new(plate()?),
    )
    .with_boundary_conditions(conditions);
    let scenario = Scenario::multi(vec![domain])?;

    let mesh: Arc<dyn Mesh> = Arc::new(plate()?);
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

// ── Exact solution ────────────────────────────────────────────────────────

/// Complementary error function, Chebyshev approximation `erfcc` (absolute
/// error at most 8.3e-8, see the module documentation).
fn erfc(x: f64) -> f64 {
    let z = x.abs();
    let t = 1.0 / (1.0 + 0.5 * z);
    let polynomial = -z * z - 1.265_512_23
        + t * (1.000_023_68
            + t * (0.374_091_96
                + t * (0.096_784_18
                    + t * (-0.186_288_06
                        + t * (0.278_868_07
                            + t * (-1.135_203_98
                                + t * (1.488_515_87 + t * (-0.822_152_23 + t * 0.170_872_77))))))));
    let value = t * polynomial.exp();
    if x >= 0.0 {
        value
    } else {
        2.0 - value
    }
}

/// 2 erfc^-1(1/2), by bisection on `erfc`, which decreases on [0, 2].
fn half_field_factor() -> f64 {
    let (mut low, mut high) = (0.0_f64, 2.0_f64);
    for _ in 0..100 {
        let middle = 0.5 * (low + high);
        if erfc(middle) > 0.5 {
            low = middle;
        } else {
            high = middle;
        }
    }
    low + high
}

/// Half-space field at depth `x` and time `t`.
fn half_space_field(eta: f64, x: f64, t: f64) -> f64 {
    SURFACE_FIELD * erfc(x / (2.0 * (eta * t).sqrt()))
}

// ── Checks ────────────────────────────────────────────────────────────────

/// The saved state at `time`, or an error naming the conductor.
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

/// Depth where the field crosses B0/2, interpolated linearly between the
/// two nodes that bracket it.
fn half_field_depth(field: &DVector<f64>, xs: &[f64], label: &str) -> Result<f64, Box<dyn Error>> {
    let half = 0.5 * SURFACE_FIELD;
    (0..field.len() - 1)
        .find(|&i| field[i] >= half && field[i + 1] < half)
        .map(|i| xs[i] + (field[i] - half) / (field[i] - field[i + 1]) * (xs[i + 1] - xs[i]))
        .ok_or_else(|| format!("{label}: the field never crosses B0/2").into())
}

/// Fails with the conductor, the time, the error and the tolerance.
fn check(
    label: &str,
    quantity: &str,
    time: f64,
    error: f64,
    tolerance: f64,
) -> Result<(), Box<dyn Error>> {
    if error <= tolerance {
        Ok(())
    } else {
        Err(format!(
            "{label}: {quantity} error {error:.3e} at t = {time} s exceeds the tolerance {tolerance:.1e}"
        )
        .into())
    }
}

fn main() -> Result<(), Box<dyn Error>> {
    let xs = positions(&plate()?);
    let factor = half_field_factor();
    println!(
        "Field step of {SURFACE_FIELD} T at the surface of a {} mm plate, BDF2, dt = {DT} s",
        THICKNESS * 1e3
    );
    println!("Expected delta_half / sqrt(eta t) = 2 erfc^-1(1/2) = {factor:.7}");

    let mut depths: Vec<Vec<f64>> = Vec::new();
    for (name, sigma) in CONDUCTORS {
        let eta = diffusivity(sigma);
        println!("{name}: sigma = {sigma:.3e} S/m, eta = {eta:.4e} m²/s");
        let result = solve(eta)?;
        let mut conductor_depths = Vec::new();
        for &time in &CHECK_TIMES {
            let field = state_at(&result, time, name)?;
            let profile_error = field
                .iter()
                .zip(&xs)
                .map(|(value, &x)| (value - half_space_field(eta, x, time)).abs())
                .fold(0.0, f64::max);
            let depth = half_field_depth(field, &xs, name)?;
            let ratio = depth / (eta * time).sqrt();
            let depth_error = (ratio - factor).abs() / factor;
            println!(
                "  t = {:>4} ms: max |B - B0 erfc| = {profile_error:.3e} T   delta_half = {:.4} mm   delta_half / sqrt(eta t) = {ratio:.6} (rel. error {depth_error:.2e})",
                time * 1e3,
                depth * 1e3
            );
            check(name, "profile (T)", time, profile_error, PROFILE_TOLERANCE)?;
            check(name, "relative depth", time, depth_error, DEPTH_TOLERANCE)?;
            conductor_depths.push(depth);
        }
        depths.push(conductor_depths);
    }

    let expected = (CONDUCTORS[0].1 / CONDUCTORS[1].1).sqrt();
    for (i, &time) in CHECK_TIMES.iter().enumerate() {
        println!(
            "  t = {:>4} ms: delta_half aluminium / copper = {:.4} (sqrt(sigma_Cu / sigma_Al) = {expected:.4}, reported)",
            time * 1e3,
            depths[1][i] / depths[0][i]
        );
    }
    println!("magnetic_diffusion_1d: every check within its tolerance");
    Ok(())
}
