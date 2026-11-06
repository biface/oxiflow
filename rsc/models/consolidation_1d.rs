//! Terzaghi consolidation of a clay layer, the scenario of the
//! consolidation reference example: the model, the layer, the reference
//! run and Terzaghi's series solutions.
//!
//! Single definition used by `examples/consolidation_1d.rs`, which
//! documents the physics and validates the run, and by the example that
//! plots it, each of which includes this file as a module with `#[path]`.
//! The run is [`solve`], with [`SAVE_EVERY`] for the reference run; the
//! series are [`series_pressure`] and [`series_consolidation`].

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
pub(crate) const YEAR: f64 = 365.25 * 86_400.0;
/// Coefficient of consolidation, m²/s (1 m²/year).
const CV: f64 = 1.0 / YEAR;
/// Load, carried at first by the pore water: initial excess pressure, kPa.
pub(crate) const LOAD: f64 = 100.0;
/// Drainage path H_dr, m: half the layer thickness.
pub(crate) const DRAINAGE_PATH: f64 = 2.0;
/// Layer thickness, m, drained at both faces.
pub(crate) const THICKNESS: f64 = 2.0 * DRAINAGE_PATH;
/// Grid nodes, faces included (dz = 4 cm).
const N_NODES: usize = 101;

/// Time of one unit of time factor, H_dr² / cv, s (four years).
pub(crate) const TIME_SCALE: f64 = DRAINAGE_PATH * DRAINAGE_PATH / CV;
/// Time step, s: 0.005 in time factor, about 7.3 days.
pub(crate) const DT: f64 = 0.005 * TIME_SCALE;
const END_TIME: f64 = TIME_SCALE;
/// One saved state every 0.05 in time factor: the reference run.
pub(crate) const SAVE_EVERY: usize = 10;

pub(crate) fn layer() -> Result<UniformGrid1D, OxiflowError> {
    UniformGrid1D::new(N_NODES, 0.0, THICKNESS)
}

/// z of every node.
pub(crate) fn positions(mesh: &dyn Mesh) -> Vec<f64> {
    (0..mesh.n_dof()).map(|i| mesh.coordinates(i)[0]).collect()
}

fn face(side: FaceSide) -> BoundaryRegion {
    BoundaryRegion::Face { axis: 0, side }
}

/// Solves the dissipation of the excess pore pressure, saving one state
/// every `save_every` steps ([`SAVE_EVERY`] for the reference run).
pub(crate) fn solve(save_every: usize) -> Result<SimulationResult, OxiflowError> {
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
        TimeConfiguration::new(END_TIME, StepControl::Fixed { dt: DT }).saving_every(save_every),
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
pub(crate) fn series_pressure(z: f64, tv: f64) -> f64 {
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
pub(crate) fn series_consolidation(tv: f64) -> f64 {
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

// ── Saved states ─────────────────────────────────────────────────────────

/// The saved state at time factor `tv`, or an error naming it.
pub(crate) fn state_at(
    result: &SimulationResult,
    tv: f64,
) -> Result<&DVector<f64>, Box<dyn Error>> {
    let time = tv * TIME_SCALE;
    let index = result
        .times
        .iter()
        .position(|&t| (t - time).abs() <= 1e-9 * time)
        .ok_or_else(|| format!("no state saved at Tv = {tv}"))?;
    Ok(result.states[index].as_scalar_field()?)
}

/// One minus the mean excess pressure over u0, trapezoidal rule.
pub(crate) fn degree_of_consolidation(pressure: &DVector<f64>, zs: &[f64]) -> f64 {
    let integral: f64 = (0..pressure.len() - 1)
        .map(|i| 0.5 * (pressure[i] + pressure[i + 1]) * (zs[i + 1] - zs[i]))
        .sum();
    1.0 - integral / (LOAD * THICKNESS)
}
