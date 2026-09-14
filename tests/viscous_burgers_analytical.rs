//! # Integration test: viscous Burgers model vs. exact solution (#137)
//!
//! Companion to `examples/viscous_burgers.rs` — same physics, duplicated
//! here (same convention as `tests/diffusion_1d_analytical.rs`) so this
//! file asserts the acceptance criteria from #137 as a regression gate:
//! matches the exact traveling-shock solution within a documented
//! tolerance, and the measured convergence order approaches 2 as the mesh
//! is refined.

use std::sync::Arc;

use nalgebra::DVector;
use oxiflow::{
    context::{
        calculators::{FDGradientCalculator, FDLaplacianCalculator, FDScheme},
        compute::ComputeContext,
        error::OxiflowError,
        value::ContextValue,
        variable::ContextVariable,
    },
    mesh::{Mesh, UniformGrid1D},
    model::traits::{PhysicalModel, RequiresContext},
    solver::{
        config::{IntegratorKind, StepControl, TimeConfiguration},
        methods::ForwardEulerSolver,
        scenario::Scenario,
        Solver, SolverConfiguration,
    },
};

const NU: f64 = 0.05;
const U_L: f64 = 1.0;
const U_R: f64 = 0.0;
const X0: f64 = -2.0;
const T_END: f64 = 2.0;
const DOMAIN_HALF_WIDTH: f64 = 3.0;

fn laplacian_variable() -> ContextVariable {
    ContextVariable::External {
        name: "laplacian".into(),
    }
}

struct ViscousBurgers {
    nu: f64,
    u_l: f64,
    u_r: f64,
    x0: f64,
}

impl ViscousBurgers {
    fn exact(&self, x: f64, t: f64) -> f64 {
        let speed = (self.u_l + self.u_r) / 2.0;
        let half_jump = (self.u_l - self.u_r) / 2.0;
        let arg = (self.u_l - self.u_r) / (4.0 * self.nu) * (x - self.x0 - speed * t);
        speed - half_jump * arg.tanh()
    }
}

impl RequiresContext for ViscousBurgers {
    fn required_variables(&self) -> Vec<ContextVariable> {
        vec![
            ContextVariable::SpatialGradient {
                dimension: 0,
                component: None,
            },
            laplacian_variable(),
        ]
    }
}

impl PhysicalModel for ViscousBurgers {
    fn compute_physics(
        &self,
        state: &ContextValue,
        ctx: &ComputeContext,
    ) -> Result<ContextValue, OxiflowError> {
        let u = state.as_scalar_field()?;
        let du_dx = ctx.gradient(0)?;
        let lap = ctx.external(laplacian_variable())?.as_scalar_field()?;
        let du_dt = -u.component_mul(du_dx) + lap.map(|v| self.nu * v);
        Ok(ContextValue::ScalarField(du_dt))
    }

    fn initial_state(&self, mesh: &dyn Mesh) -> ContextValue {
        let u0 = DVector::from_iterator(
            mesh.n_dof(),
            (0..mesh.n_dof()).map(|i| {
                let x = mesh.coordinates(i)[0];
                self.exact(x, 0.0)
            }),
        );
        ContextValue::ScalarField(u0)
    }

    fn name(&self) -> &str {
        "viscous_burgers"
    }
}

fn make_model() -> ViscousBurgers {
    ViscousBurgers {
        nu: NU,
        u_l: U_L,
        u_r: U_R,
        x0: X0,
    }
}

/// Runs the model at `n_nodes` and returns the max absolute error against
/// the exact traveling-shock solution at `T_END`. `dt` scales with
/// `dx^2/nu` (diffusive stability) -- verified during design to give clean
/// ~2nd-order convergence, same reasoning as `diffusion_1d`'s own dt
/// choice.
fn run_and_measure_error(n_nodes: usize) -> f64 {
    let dx = 2.0 * DOMAIN_HALF_WIDTH / (n_nodes as f64 - 1.0);
    let dt = 0.4 * dx * dx / NU;

    let mesh = UniformGrid1D::new(n_nodes, -DOMAIN_HALF_WIDTH, DOMAIN_HALF_WIDTH).unwrap();
    let gradient_mesh: Arc<dyn Mesh> =
        Arc::new(UniformGrid1D::new(n_nodes, -DOMAIN_HALF_WIDTH, DOMAIN_HALF_WIDTH).unwrap());
    let laplacian_mesh: Arc<dyn Mesh> =
        Arc::new(UniformGrid1D::new(n_nodes, -DOMAIN_HALF_WIDTH, DOMAIN_HALF_WIDTH).unwrap());
    let mesh_for_error =
        UniformGrid1D::new(n_nodes, -DOMAIN_HALF_WIDTH, DOMAIN_HALF_WIDTH).unwrap();

    let model = make_model();
    let scenario = Scenario::single(Box::new(model), Box::new(mesh));
    let config = SolverConfiguration::new(
        TimeConfiguration::new(T_END, StepControl::Fixed { dt }),
        IntegratorKind::Euler,
    )
    .with_calculator(Box::new(FDGradientCalculator::new(
        gradient_mesh,
        0,
        None,
        FDScheme::Central,
    )))
    .with_calculator(Box::new(FDLaplacianCalculator::new(
        laplacian_mesh,
        laplacian_variable(),
    )));

    let solver = ForwardEulerSolver;
    let result = solver.solve(&scenario, &config).unwrap();
    let final_state = result.states.last().unwrap().as_scalar_field().unwrap();

    let exact_model = make_model();
    (0..n_nodes)
        .map(|i| {
            let x = mesh_for_error.coordinates(i)[0];
            (final_state[i] - exact_model.exact(x, T_END)).abs()
        })
        .fold(0.0, f64::max)
}

/// Documented tolerance -- a standalone FD replica (outside oxiflow, during
/// design) measured ~1.6e-3 to ~6.8e-3 max error across the dt/N
/// combinations tried at similar resolutions; this bound keeps comfortable
/// margin above that range without hiding a real regression.
const ABSOLUTE_TOLERANCE: f64 = 1e-2;

#[test]
fn matches_exact_solution_within_documented_tolerance() {
    let max_error = run_and_measure_error(401);
    assert!(
        max_error < ABSOLUTE_TOLERANCE,
        "max |error| = {max_error:e}, expected < {ABSOLUTE_TOLERANCE:e}"
    );
}

/// Doubling the node count (dt scaling with dx^2 at each resolution)
/// should roughly quarter the error -- verified during design (standalone
/// replica) to give a clean, consistent order (~2.0-2.05) for this
/// resolution pair.
#[test]
fn convergence_order_approaches_two() {
    let error_coarse = run_and_measure_error(201);
    let error_fine = run_and_measure_error(401);
    let observed_order = (error_coarse / error_fine).log2();
    assert!(
        observed_order > 1.7,
        "observed convergence order {observed_order:.2} (coarse err={error_coarse:e}, \
         fine err={error_fine:e}), expected > 1.7 approaching the theoretical order 2"
    );
}
