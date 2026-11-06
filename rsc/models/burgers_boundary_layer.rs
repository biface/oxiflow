//! Viscous Burgers shock running into a wall, the scenario of the boundary
//! layer reference example: the scenario, the runs at the three Reynolds
//! numbers and the exact steady layer.
//!
//! Single definition used by `examples/burgers_boundary_layer.rs`, which
//! documents the physics and validates the runs, and by the examples that
//! plot and animate them, each of which includes this file as a module with
//! `#[path]`. The run at one viscosity is [`solve`], with [`SAVES`] for the
//! reference runs; the steady layer is [`steady_velocity`].
//!
//! Requires `rsc/models/viscous_burgers.rs` as a sibling module named
//! `viscous_burgers_model`.

use std::error::Error;
use std::sync::Arc;

use super::viscous_burgers_model::{laplacian_variable, ViscousBurgers};
use nalgebra::DVector;
use oxiflow::{
    boundary::{BoundaryCondition, Dirichlet},
    context::{
        calculators::{FDGradientCalculator, FDLaplacianCalculator, FDScheme},
        error::OxiflowError,
    },
    mesh::{BoundaryRegion, FaceSide, Mesh, UniformGrid1D},
    solver::{
        config::{IntegratorKind, StepControl, TimeConfiguration},
        methods::RK4Solver,
        scenario::{Domain, Scenario},
        SimulationResult, Solver, SolverConfiguration,
    },
};

// ── Scenario ──────────────────────────────────────────────────────────────

/// Inflow velocity, in units of U.
pub(crate) const INFLOW: f64 = 1.0;
/// Initial centre of the viscous shock.
pub(crate) const SHOCK_CENTRE: f64 = 0.6;
/// Grid nodes, both ends included (dx = 1/400).
pub(crate) const N_NODES: usize = 401;
pub(crate) const END_TIME: f64 = 2.4;
/// Bound on nu * dt / dx² for RK4 (see the module documentation).
const DIFFUSION_NUMBER: f64 = 0.6;
/// Saved states per reference run, after the initial one; the number of
/// steps is a multiple of it.
pub(crate) const SAVES: usize = 8;
/// Level of the 99 % thickness, as a fraction of A.
pub(crate) const THICKNESS_LEVEL: f64 = 0.99;
/// The three Reynolds numbers, Re = 1 / nu.
pub(crate) const REYNOLDS: [f64; 3] = [50.0, 100.0, 200.0];

pub(crate) fn grid() -> Result<UniformGrid1D, OxiflowError> {
    UniformGrid1D::new(N_NODES, 0.0, 1.0)
}

/// x of every node.
pub(crate) fn positions(mesh: &dyn Mesh) -> Vec<f64> {
    (0..mesh.n_dof()).map(|i| mesh.coordinates(i)[0]).collect()
}

fn face(side: FaceSide) -> BoundaryRegion {
    BoundaryRegion::Face { axis: 0, side }
}

/// Number of steps: the smallest multiple of `SAVES` that keeps
/// nu * dt / dx² at or below `DIFFUSION_NUMBER`.
pub(crate) fn steps(nu: f64) -> usize {
    let dx = 1.0 / (N_NODES - 1) as f64;
    let least = (END_TIME * nu / (DIFFUSION_NUMBER * dx * dx)).ceil() as usize;
    least.div_ceil(SAVES) * SAVES
}

/// Runs the shock into the wall at viscosity `nu`, saving `saves` states
/// after the initial one ([`SAVES`] for the reference runs). The time step
/// is that of the reference run; `saves` must divide its number of steps.
pub(crate) fn solve(nu: f64, saves: usize) -> Result<SimulationResult, Box<dyn Error>> {
    let n_steps = steps(nu);
    if n_steps % saves != 0 {
        return Err(format!("{saves} saves do not divide the {n_steps} steps at nu = {nu}").into());
    }
    let conditions: Vec<Box<dyn BoundaryCondition>> = vec![
        Box::new(Dirichlet::constant(face(FaceSide::Lower), INFLOW)),
        Box::new(Dirichlet::constant(face(FaceSide::Upper), 0.0)),
    ];
    let model = ViscousBurgers {
        nu,
        u_l: INFLOW,
        u_r: 0.0,
        x0: SHOCK_CENTRE,
    };
    let domain = Domain::new("channel", Box::new(model), Box::new(grid()?))
        .with_boundary_conditions(conditions);
    let scenario = Scenario::multi(vec![domain])?;

    let mesh: Arc<dyn Mesh> = Arc::new(grid()?);
    let config = SolverConfiguration::new(
        TimeConfiguration::new(
            END_TIME,
            StepControl::Fixed {
                dt: END_TIME / n_steps as f64,
            },
        )
        .saving_every(n_steps / saves),
        IntegratorKind::RK4,
    )
    .with_calculator(Box::new(FDGradientCalculator::new(
        Arc::clone(&mesh),
        0,
        None,
        FDScheme::Central,
    )))
    .with_calculator(Box::new(FDLaplacianCalculator::new(
        mesh,
        laplacian_variable(),
    )));
    Ok(RK4Solver.solve(&scenario, &config)?)
}

// ── Steady solution ───────────────────────────────────────────────────────

/// Outer velocity A of the steady layer: A tanh(A / (2 nu)) = 1, by Newton
/// from A = 1.
pub(crate) fn outer_velocity(nu: f64) -> f64 {
    let mut a = INFLOW;
    for _ in 0..50 {
        let z = a / (2.0 * nu);
        let residual = a * z.tanh() - INFLOW;
        let slope = z.tanh() + z / z.cosh().powi(2);
        a -= residual / slope;
    }
    a
}

/// Steady velocity at `x`.
pub(crate) fn steady_velocity(x: f64, a: f64, nu: f64) -> f64 {
    a * (a * (1.0 - x) / (2.0 * nu)).tanh()
}

/// delta99 of the steady layer.
pub(crate) fn steady_thickness(a: f64, nu: f64) -> f64 {
    2.0 * nu / a * THICKNESS_LEVEL.atanh()
}

// ── Saved states ─────────────────────────────────────────────────────────

/// The state saved at `time`, or an error naming it.
pub(crate) fn state_at(
    result: &SimulationResult,
    time: f64,
) -> Result<&DVector<f64>, Box<dyn Error>> {
    let index = result
        .times
        .iter()
        .position(|&t| (t - time).abs() <= 1e-9 * time)
        .ok_or_else(|| format!("no state saved at t = {time}"))?;
    Ok(result.states[index].as_scalar_field()?)
}
