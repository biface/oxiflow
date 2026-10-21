//! # Integration test: 1D diffusion model vs. analytical solution (#136)
//!
//! Companion to `examples/diffusion_1d.rs`, on the same model
//! (`rsc/models/diffusion_1d.rs`): this file asserts the acceptance
//! criteria from #136 as a regression gate, which a `fn main()` example
//! alone cannot enforce:
//!
//! - The model matches the analytical Gaussian solution within a
//!   documented tolerance.
//! - The measured convergence order approaches 2 (the theoretical order of
//!   `FDLaplacianCalculator`'s centered stencil) as the mesh is refined --
//!   matching the analytical solution at one resolution alone cannot
//!   distinguish a correct discretization from a lucky cancellation of
//!   errors at that specific `n`.

use std::sync::Arc;

use oxiflow::{
    context::calculators::FDLaplacianCalculator,
    mesh::{Mesh, UniformGrid1D},
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

#[path = "../rsc/models/diffusion_1d.rs"]
mod diffusion_1d_model;

use diffusion_1d_model::{laplacian_variable, Diffusion1D};

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
