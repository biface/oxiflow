//! # Integration test: 1D diffusion model vs. analytical solution (#136)
//!
//! Companion to `examples/diffusion_1d.rs` — same physics, duplicated here
//! (same convention as `tests/lahar_lake_proto.rs` vs. its example) so this
//! file asserts the acceptance criteria from #136 as a regression gate,
//! which a `fn main()` example alone cannot enforce:
//!
//! - The model matches the analytical Gaussian solution within a
//!   documented tolerance.
//! - The measured convergence order approaches 2 (the theoretical order of
//!   `FDLaplacianCalculator`'s centered stencil) as the mesh is refined --
//!   matching the analytical solution at one resolution alone cannot
//!   distinguish a correct discretization from a lucky cancellation of
//!   errors at that specific `n`.

use std::sync::Arc;

use nalgebra::DVector;
use oxiflow::{
    context::{
        calculators::FDLaplacianCalculator, compute::ComputeContext, error::OxiflowError,
        value::ContextValue, variable::ContextVariable,
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

const ALPHA: f64 = 0.01;
const AMPLITUDE: f64 = 1.0;
const CENTER: f64 = 0.0;
const SIGMA0: f64 = 0.05;
const T_END: f64 = 0.2;
// See examples/diffusion_1d.rs's module docs for the dt sweep this value
// comes from -- small enough that Euler's O(dt) temporal error stays well
// below the stencil's O(dx^2) spatial error, which is what
// `convergence_order_approaches_two` below actually needs to measure.
const DT: f64 = 1e-5;
const DOMAIN_HALF_WIDTH: f64 = 1.0;

fn laplacian_variable() -> ContextVariable {
    ContextVariable::External {
        name: "laplacian".into(),
    }
}

struct Diffusion1D {
    alpha: f64,
    amplitude: f64,
    center: f64,
    sigma0: f64,
}

impl Diffusion1D {
    fn exact(&self, x: f64, t: f64) -> f64 {
        let sigma_t = (self.sigma0.powi(2) + 2.0 * self.alpha * t).sqrt();
        let envelope = self.amplitude * self.sigma0 / sigma_t;
        let exponent = -(x - self.center).powi(2) / (2.0 * sigma_t.powi(2));
        envelope * exponent.exp()
    }
}

impl RequiresContext for Diffusion1D {
    fn required_variables(&self) -> Vec<ContextVariable> {
        vec![laplacian_variable()]
    }
}

impl PhysicalModel for Diffusion1D {
    fn compute_physics(
        &self,
        _state: &ContextValue,
        ctx: &ComputeContext,
    ) -> Result<ContextValue, OxiflowError> {
        let lap = ctx.external(laplacian_variable())?.as_scalar_field()?;
        Ok(ContextValue::ScalarField(lap.map(|v| self.alpha * v)))
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
        "diffusion_1d"
    }
}

fn make_model() -> Diffusion1D {
    Diffusion1D {
        alpha: ALPHA,
        amplitude: AMPLITUDE,
        center: CENTER,
        sigma0: SIGMA0,
    }
}

/// Runs the model at `n_nodes` and returns the max absolute error against
/// the analytical solution at `T_END`.
fn run_and_measure_error(n_nodes: usize) -> f64 {
    let mesh = UniformGrid1D::new(n_nodes, -DOMAIN_HALF_WIDTH, DOMAIN_HALF_WIDTH).unwrap();
    let mesh_arc: Arc<dyn Mesh> =
        Arc::new(UniformGrid1D::new(n_nodes, -DOMAIN_HALF_WIDTH, DOMAIN_HALF_WIDTH).unwrap());
    let mesh_for_error =
        UniformGrid1D::new(n_nodes, -DOMAIN_HALF_WIDTH, DOMAIN_HALF_WIDTH).unwrap();

    let model = make_model();
    let scenario = Scenario::single(Box::new(model), Box::new(mesh));
    let config = SolverConfiguration::new(
        TimeConfiguration::new(T_END, StepControl::Fixed { dt: DT }),
        IntegratorKind::Euler,
    )
    .with_calculator(Box::new(FDLaplacianCalculator::new(
        mesh_arc,
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

/// Documented tolerance for the acceptance criterion "matches the
/// analytical solution" -- at N=401 (dx=0.005, dt=1e-5), measured error is
/// ~1.8e-4 (see the dt-sweep note on `DT`); this bound keeps comfortable
/// margin above that without being loose enough to hide a real regression.
const ABSOLUTE_TOLERANCE: f64 = 1e-3;

#[test]
fn matches_analytical_solution_within_documented_tolerance() {
    let max_error = run_and_measure_error(401);
    assert!(
        max_error < ABSOLUTE_TOLERANCE,
        "max |error| = {max_error:e}, expected < {ABSOLUTE_TOLERANCE:e}"
    );
}

/// Doubling the node count should roughly quarter the error (2nd-order
/// centered stencil). N=401/801 rather than 201/401 -- verified by a dt
/// sweep during design that this pair gives a clean, consistent order
/// (~2.0-2.1) at `DT`, unlike coarser pairs which still show transitional
/// noise from the temporal error not yet being fully subdominant.
#[test]
fn convergence_order_approaches_two() {
    let error_coarse = run_and_measure_error(401);
    let error_fine = run_and_measure_error(801);
    let observed_order = (error_coarse / error_fine).log2();
    assert!(
        observed_order > 1.8,
        "observed convergence order {observed_order:.2} (coarse err={error_coarse:e}, \
         fine err={error_fine:e}), expected > 1.8 approaching the theoretical order 2"
    );
}
