//! Criterion benchmarks for the Langmuir multi-species model (#138, feeds
//! #53). The model comes from `rsc/models/langmuir_multi.rs`, shared with
//! the examples and the reference test.
//!
//! Same two-measurement split as the other three models, but the second
//! one targets a different site than usual:
//!
//! - [`full_run`] — fixed-size performance on the `ascorbic_erythorbic`
//!   reference scenario (100 points, 4000 steps).
//! - [`row_dispatch_diagnostic`] — times the model's `compute_physics`
//!   with its per-node loop (jacobian assembly + an `n_species x
//!   n_species` LU solve, plus a fresh heap allocation per row) forced
//!   sequential or parallel, parameterized by node count. This is the model's actual cost site -- unlike
//!   diffusion_1d/viscous_burgers, no single calculator call dominates
//!   here, so calibrating a calculator's own dispatch (as those two did)
//!   would not measure anything representative of this model.
//! - [`implicit_jacobian_dispatch`] — dispatch of the implicit solvers'
//!   finite-difference Jacobian assembly (#199) on a `VectorField` state,
//!   sequential vs. parallel, through one Backward Euler step: dense path,
//!   and banded path with the `sparse` feature.

use std::hint::black_box;

#[cfg(feature = "parallel")]
use criterion::BenchmarkId;
use criterion::Criterion;
#[cfg(feature = "parallel")]
use nalgebra::{DMatrix, DVector};
use oxiflow::{
    context::calculators::{FDGradientCalculator, FDScheme},
    mesh::{Mesh, UniformGrid1D},
    solver::{
        config::{IntegratorKind, StepControl, TimeConfiguration},
        methods::ForwardEulerSolver,
        scenario::Scenario,
        Solver, SolverConfiguration,
    },
};
#[cfg(feature = "parallel")]
use oxiflow::{
    context::{compute::ComputeContext, value::ContextValue, variable::ContextVariable},
    model::traits::PhysicalModel,
};
use std::sync::Arc;

const N_POINTS: usize = 100;
const COLUMN_LENGTH: f64 = 0.25;
const TOTAL_TIME: f64 = 800.0;
const N_STEPS: usize = 4000;

#[path = "../rsc/models/langmuir_multi.rs"]
mod langmuir_multi_model;

use langmuir_multi_model::make_model;

/// #53's fixed-size target for this model: the `ascorbic_erythorbic`
/// reference scenario (100 points, 4000 steps).
pub fn full_run(c: &mut Criterion) {
    c.bench_function("langmuir_multi_full_run", |b| {
        b.iter(|| {
            let dz = COLUMN_LENGTH / (N_POINTS as f64 - 1.0);
            let model = make_model(dz);
            let n_species = model.n_species();
            let mesh = UniformGrid1D::new(N_POINTS, 0.0, COLUMN_LENGTH).unwrap();

            let mut config = SolverConfiguration::new(
                TimeConfiguration::new(
                    TOTAL_TIME,
                    StepControl::Fixed {
                        dt: TOTAL_TIME / N_STEPS as f64,
                    },
                ),
                IntegratorKind::Euler,
            );
            for i in 0..n_species {
                let mesh_arc: Arc<dyn Mesh> =
                    Arc::new(UniformGrid1D::new(N_POINTS, 0.0, COLUMN_LENGTH).unwrap());
                config = config.with_calculator(Box::new(FDGradientCalculator::new(
                    mesh_arc,
                    0,
                    Some(i),
                    FDScheme::Backward,
                )));
            }

            let scenario = Scenario::single(Box::new(model), Box::new(mesh));
            let result = ForwardEulerSolver.solve(&scenario, &config).unwrap();
            black_box(result)
        })
    });
}

/// Dispatch of the per-row loop inside the model's own `compute_physics`
/// (jacobian assembly + an `n_species x n_species` LU solve per row),
/// sequential vs. parallel, parameterized by node count. The threshold is
/// forced on the shared model (`usize::MAX`, then `1`), the way
/// `diffusion_1d`'s `laplacian_dispatch` forces it on its calculator, so
/// the timed code is the model itself and follows any change made to it.
/// The gradients the model reads are inserted into the `ComputeContext`
/// once, outside the timed loop; what is timed is one full
/// `compute_physics` call, gradient copies and column-major assembly
/// included. This is the actual calibration target for this model (see
/// the module docs for why the calculator-dispatch pattern used by
/// diffusion_1d/viscous_burgers doesn't apply here). Only exists when the
/// crate is built with the `parallel` feature -- there is nothing to
/// compare against without it.
#[cfg(feature = "parallel")]
pub fn row_dispatch_diagnostic(c: &mut Criterion) {
    let mut group = c.benchmark_group("langmuir_multi_row_dispatch");
    for &n_points in &[100usize, 1_000, 10_000, 100_000] {
        let mut model = make_model(COLUMN_LENGTH / (N_POINTS as f64 - 1.0));
        let n_species = model.n_species();

        // Representative concentrations -- comparable order of magnitude
        // to the reference scenario's own dilute regime (~1e-6 mol/L),
        // not the exact simulated field: only the dispatch mechanism is
        // being measured here, not the physics. t = 0 lies inside the
        // injection window, so row 0 takes its injection branch as in a
        // real run.
        let state = ContextValue::VectorField(DMatrix::from_element(n_points, n_species, 1e-6));
        let mut ctx = ComputeContext::new(0.0, TOTAL_TIME / N_STEPS as f64);
        for i in 0..n_species {
            ctx.insert(
                ContextVariable::SpatialGradient {
                    dimension: 0,
                    component: Some(i),
                },
                ContextValue::ScalarField(DVector::from_element(n_points, 1e-8)),
            );
        }

        model.parallel_threshold = usize::MAX;
        group.bench_with_input(
            BenchmarkId::new("sequential", n_points),
            &n_points,
            |b, _| b.iter(|| black_box(model.compute_physics(&state, &ctx).unwrap())),
        );

        model.parallel_threshold = 1;
        group.bench_with_input(BenchmarkId::new("parallel", n_points), &n_points, |b, _| {
            b.iter(|| black_box(model.compute_physics(&state, &ctx).unwrap()))
        });
    }
    group.finish();
}

/// Dispatch of the finite-difference Jacobian assembly of the implicit
/// solvers (#199, DT-069) on a `VectorField` state, sequential vs.
/// parallel, through the public `BackwardEulerSolver::with_parallel_threshold`.
/// Same principle as `diffusion_1d::implicit_jacobian_dispatch`: one full
/// Backward Euler step is timed, and the model's row loop and the gradient
/// calculators are kept sequential in both, so that only the Jacobian
/// dispatch differs. Sizes are node counts; the system has
/// `n_species` times as many unknowns.
///
/// Banded path (`sparse` feature): half-bandwidth `d·(s + 1) − 1` for `d`
/// species and the backward stencil (`s = 1`), so `2·(2d − 1) + 1` colours,
/// that is as many tasks. Only exists when the crate is built with the
/// `parallel` feature.
#[cfg(feature = "parallel")]
pub fn implicit_jacobian_dispatch(c: &mut Criterion) {
    use crate::implicit_dispatch::sequential_vs_parallel;
    use oxiflow::solver::methods::backward_euler::BackwardEulerSolver;

    let mut group = c.benchmark_group("langmuir_multi_implicit_jacobian_dispatch");
    group.sample_size(20);

    for &n_points in &[32usize, 64, 128, 256] {
        let (scenario, config, _) = one_backward_euler_step(n_points);
        sequential_vs_parallel(
            &mut group,
            "dense",
            n_points,
            &scenario,
            &config,
            |threshold| BackwardEulerSolver::new().with_parallel_threshold(threshold),
        );
    }

    #[cfg(feature = "sparse")]
    for &n_points in &[1_000usize, 10_000, 50_000] {
        use oxiflow::solver::sparse::FaerSparseSolver;

        let (scenario, config, bandwidth) = one_backward_euler_step(n_points);
        sequential_vs_parallel(
            &mut group,
            "banded",
            n_points,
            &scenario,
            &config,
            |threshold| {
                BackwardEulerSolver::new()
                    .with_sparse_solver(Box::new(FaerSparseSolver))
                    .with_jacobian_bandwidth(bandwidth)
                    .with_sparse_threshold(0)
                    .with_parallel_threshold(threshold)
            },
        );
    }

    group.finish();
}

/// Scenario and configuration of a single Backward Euler step on
/// `n_points` nodes, with the model's row loop and the gradient
/// calculators kept sequential, and the Jacobian half-bandwidth.
#[cfg(feature = "parallel")]
fn one_backward_euler_step(n_points: usize) -> (Scenario, SolverConfiguration, usize) {
    let dt = TOTAL_TIME / N_STEPS as f64;
    let mut model = make_model(COLUMN_LENGTH / (n_points as f64 - 1.0));
    model.parallel_threshold = usize::MAX;
    let n_species = model.n_species();

    let mut config = SolverConfiguration::new(
        TimeConfiguration::new(dt, StepControl::Fixed { dt }),
        IntegratorKind::BackwardEuler,
    );
    for i in 0..n_species {
        let mesh_arc: Arc<dyn Mesh> =
            Arc::new(UniformGrid1D::new(n_points, 0.0, COLUMN_LENGTH).unwrap());
        config = config.with_calculator(Box::new(
            FDGradientCalculator::new(mesh_arc, 0, Some(i), FDScheme::Backward)
                .with_parallel_threshold(usize::MAX),
        ));
    }

    let mesh = UniformGrid1D::new(n_points, 0.0, COLUMN_LENGTH).unwrap();
    let scenario = Scenario::single(Box::new(model), Box::new(mesh));
    (scenario, config, 2 * n_species - 1)
}
