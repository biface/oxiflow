//! Magnetic diffusion into a conducting plate, the scenario of the
//! magnetic-diffusion reference example: the model, the conductors, the
//! plate, the runs and the exact half-space solution.
//!
//! Single definition used by `examples/magnetic_diffusion_1d.rs`, which
//! documents the physics and validates the runs, and by the examples that
//! plot and animate them, each of which includes this file as a module with
//! `#[path]`. The run in one conductor is [`solve`], with [`SAVE_EVERY`] for
//! the reference runs; the exact field is [`half_space_field`].

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
pub(crate) const MU0: f64 = 4.0e-7 * PI;
/// Conductors and their electrical conductivity at 20 °C, S/m.
pub(crate) const CONDUCTORS: [(&str, f64); 2] = [("copper", 5.96e7), ("aluminium", 3.77e7)];
/// Field applied at the surface from t = 0, T.
pub(crate) const SURFACE_FIELD: f64 = 0.1;
/// Plate thickness, m.
pub(crate) const THICKNESS: f64 = 0.03;
/// Grid nodes, faces included (dx = 0.25 mm).
const N_NODES: usize = 121;

pub(crate) const DT: f64 = 1e-5;
pub(crate) const END_TIME: f64 = 1e-3;
/// One saved state every 0.25 ms: the reference runs.
pub(crate) const SAVE_EVERY: usize = 25;

pub(crate) fn diffusivity(sigma: f64) -> f64 {
    1.0 / (MU0 * sigma)
}

pub(crate) fn plate() -> Result<UniformGrid1D, OxiflowError> {
    UniformGrid1D::new(N_NODES, 0.0, THICKNESS)
}

/// x of every node.
pub(crate) fn positions(mesh: &dyn Mesh) -> Vec<f64> {
    (0..mesh.n_dof()).map(|i| mesh.coordinates(i)[0]).collect()
}

fn face(side: FaceSide) -> BoundaryRegion {
    BoundaryRegion::Face { axis: 0, side }
}

/// Solves the field step in a conductor of diffusivity `eta`, saving one
/// state every `save_every` steps ([`SAVE_EVERY`] for the reference runs).
pub(crate) fn solve(eta: f64, save_every: usize) -> Result<SimulationResult, OxiflowError> {
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
        TimeConfiguration::new(END_TIME, StepControl::Fixed { dt: DT }).saving_every(save_every),
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
pub(crate) fn half_field_factor() -> f64 {
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
pub(crate) fn half_space_field(eta: f64, x: f64, t: f64) -> f64 {
    SURFACE_FIELD * erfc(x / (2.0 * (eta * t).sqrt()))
}

// ── Saved states ─────────────────────────────────────────────────────────

/// The saved state at `time`, or an error naming the conductor.
pub(crate) fn state_at<'a>(
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
pub(crate) fn half_field_depth(
    field: &DVector<f64>,
    xs: &[f64],
    label: &str,
) -> Result<f64, Box<dyn Error>> {
    let half = 0.5 * SURFACE_FIELD;
    (0..field.len() - 1)
        .find(|&i| field[i] >= half && field[i + 1] < half)
        .map(|i| xs[i] + (field[i] - half) / (field[i] - field[i + 1]) * (xs[i + 1] - xs[i]))
        .ok_or_else(|| format!("{label}: the field never crosses B0/2").into())
}
