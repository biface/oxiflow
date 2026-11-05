//! Heat conduction in a steel bar, the scenario of the heat-transfer
//! reference example: the model, the material, the bar, the two reference
//! runs and their exact solutions.
//!
//! Single definition used by `examples/heat_transfer_1d.rs`, which
//! documents the physics and validates the runs, and by the examples that
//! plot them, each of which includes this file as a module with `#[path]`.
//! The runs are [`cooling_run`] (case (a), cooling of a homogeneous bar)
//! and [`graded_run`] (case (b), graded bar); their exact solutions are
//! [`fourier_cooling`] and [`graded_steady`].
//!
//! Requires `rsc/models/graded_conductivity.rs` as a sibling module named
//! `graded_conductivity`.

use std::f64::consts::PI;
use std::sync::Arc;

use super::graded_conductivity::GradedConductivity;
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
        scenario::{Domain, Scenario},
        SimulationResult, Solver, SolverConfiguration,
    },
};

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
pub(crate) const LENGTH: f64 = 0.1;
/// Grid nodes, ends included (dx = 1 mm).
const N_NODES: usize = 101;
/// Newton budget of every implicit solver (see the documentation of
/// `examples/heat_transfer_1d.rs`).
pub(crate) const NEWTON_ITERATIONS: usize = 5;

// ── Case (a): cooling of a homogeneous bar ────────────────────────────────

pub(crate) const COOLING_INITIAL: f64 = 100.0;
pub(crate) const COOLING_ENDS: f64 = 20.0;
pub(crate) const COOLING_DT: f64 = 0.5;
const COOLING_END_TIME: f64 = 120.0;
/// One saved state every 10 s.
const COOLING_SAVE_EVERY: usize = 20;

// ── Case (b): graded bar ──────────────────────────────────────────────────

const GRADED_BETA: f64 = 1.0;
pub(crate) const GRADED_HOT: f64 = 100.0;
pub(crate) const GRADED_COLD: f64 = 20.0;
pub(crate) const GRADED_DT: f64 = 50.0;
pub(crate) const GRADED_END_TIME: f64 = 3000.0;
/// One saved state every 500 s.
const GRADED_SAVE_EVERY: usize = 10;

fn rho_cp() -> f64 {
    DENSITY * SPECIFIC_HEAT
}

pub(crate) fn bar() -> Result<UniformGrid1D, OxiflowError> {
    UniformGrid1D::new(N_NODES, 0.0, LENGTH)
}

/// x of every node.
pub(crate) fn positions(mesh: &dyn Mesh) -> Vec<f64> {
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

// ── Runs ──────────────────────────────────────────────────────────────────

/// What differs between the runs on the bar.
pub(crate) struct Run {
    pub(crate) initial: f64,
    pub(crate) lower: f64,
    pub(crate) upper: f64,
    pub(crate) conductivity: GradedConductivity,
    pub(crate) integrator: IntegratorKind,
    pub(crate) dt: f64,
    pub(crate) end_time: f64,
    pub(crate) save_every: usize,
}

/// Reference run of case (a) with `integrator`: the bar starts at
/// [`COOLING_INITIAL`], both ends are held at [`COOLING_ENDS`].
pub(crate) fn cooling_run(integrator: IntegratorKind) -> Run {
    Run {
        initial: COOLING_INITIAL,
        lower: COOLING_ENDS,
        upper: COOLING_ENDS,
        conductivity: conductivity(0.0),
        integrator,
        dt: COOLING_DT,
        end_time: COOLING_END_TIME,
        save_every: COOLING_SAVE_EVERY,
    }
}

/// Reference run of case (b), Backward Euler: the bar starts at
/// [`GRADED_COLD`], the end x = 0 is held at [`GRADED_HOT`] and the end
/// x = L at [`GRADED_COLD`].
pub(crate) fn graded_run() -> Run {
    Run {
        initial: GRADED_COLD,
        lower: GRADED_HOT,
        upper: GRADED_COLD,
        conductivity: conductivity(GRADED_BETA),
        integrator: IntegratorKind::BackwardEuler,
        dt: GRADED_DT,
        end_time: GRADED_END_TIME,
        save_every: GRADED_SAVE_EVERY,
    }
}

/// Solves one run on the bar with `solver`, which must match
/// `run.integrator`.
pub(crate) fn solve(run: &Run, solver: &dyn Solver) -> Result<SimulationResult, OxiflowError> {
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

// ── Exact solutions ───────────────────────────────────────────────────────

/// Fourier-series temperature of case (a), summed until the bound of the
/// next term falls below 1e-12 °C.
pub(crate) fn fourier_cooling(x: f64, t: f64) -> f64 {
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
pub(crate) fn graded_steady(x: f64) -> f64 {
    let profile = (1.0 + GRADED_BETA * x / LENGTH).ln() / (1.0 + GRADED_BETA).ln();
    GRADED_HOT - (GRADED_HOT - GRADED_COLD) * profile
}
