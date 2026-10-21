//! Criterion benchmarks for the viscous Burgers model (#137, feeds #53).
//! The model comes from `rsc/models/viscous_burgers.rs`, shared with the
//! example and the analytical test.
//!
//! Same two-measurement split as `diffusion_1d.rs`:
//!
//! - [`full_run`] — fixed-size performance, no external target documented
//!   for this model in #53 (unlike diffusion_1d's explicit "< 100 ms")\;
//!   measured for its own sake (v0.9.0, #53).
//! - [`combined_dispatch_diagnostic`] — Burgers needs *two* calculators per
//!   step (`FDGradientCalculator` for the nonlinear term,
//!   `FDLaplacianCalculator` for the diffusive one), both already sharing
//!   `crate::operators::fd::default_parallel_threshold()`. This measures
//!   their *combined* per-node cost, sequential vs. parallel -- the
//!   relevant profile for this model's place in the #53 abaque, heavier
//!   than diffusion_1d's Laplacian-only cost.

use std::hint::black_box;
use std::sync::Arc;

#[cfg(feature = "parallel")]
use criterion::BenchmarkId;
use criterion::Criterion;
#[cfg(feature = "parallel")]
use nalgebra::DVector;
#[cfg(feature = "parallel")]
use oxiflow::context::{compute::ComputeContext, value::ContextValue, ContextCalculator};
use oxiflow::{
    context::calculators::{FDGradientCalculator, FDLaplacianCalculator, FDScheme},
    mesh::{Mesh, UniformGrid1D},
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
const DOMAIN_HALF_WIDTH: f64 = 3.0;

#[path = "../rsc/models/viscous_burgers.rs"]
mod viscous_burgers_model;

use viscous_burgers_model::{laplacian_variable, ViscousBurgers};

/// Fixed-size performance run -- no external ms target documented for this
/// model in #53; measured on its own terms (v0.9.0). `dt` scaled
/// with `dx^2/nu` for stability, same choice as the accuracy test.
pub fn full_run(c: &mut Criterion) {
    const N_NODES: usize = 1000;
    const N_STEPS: u64 = 10_000;

    let dx = 2.0 * DOMAIN_HALF_WIDTH / (N_NODES as f64 - 1.0);
    let dt = 0.4 * dx * dx / NU;

    c.bench_function("viscous_burgers_full_run", |b| {
        b.iter(|| {
            let model = ViscousBurgers {
                nu: NU,
                u_l: U_L,
                u_r: U_R,
                x0: X0,
            };
            let mesh = UniformGrid1D::new(N_NODES, -DOMAIN_HALF_WIDTH, DOMAIN_HALF_WIDTH).unwrap();
            let gradient_mesh: Arc<dyn Mesh> = Arc::new(
                UniformGrid1D::new(N_NODES, -DOMAIN_HALF_WIDTH, DOMAIN_HALF_WIDTH).unwrap(),
            );
            let laplacian_mesh: Arc<dyn Mesh> = Arc::new(
                UniformGrid1D::new(N_NODES, -DOMAIN_HALF_WIDTH, DOMAIN_HALF_WIDTH).unwrap(),
            );

            let scenario = Scenario::single(Box::new(model), Box::new(mesh));
            let config = SolverConfiguration::new(
                TimeConfiguration::new(N_STEPS as f64 * dt, StepControl::Fixed { dt }),
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

            let result = ForwardEulerSolver.solve(&scenario, &config).unwrap();
            black_box(result)
        })
    });
}

/// Combined gradient+laplacian dispatch cost, sequential vs. parallel,
/// forced via the public `with_parallel_threshold` API on both
/// calculators -- not the full solver loop, to isolate the calculators'
/// own per-call cost the way `diffusion_1d`'s `laplacian_dispatch` does.
/// Only exists when the crate is built with the `parallel` feature.
#[cfg(feature = "parallel")]
pub fn combined_dispatch_diagnostic(c: &mut Criterion) {
    let ctx = ComputeContext::new(0.0, 1e-4);
    let mut group = c.benchmark_group("viscous_burgers_combined_dispatch");

    for &n in &[1_000usize, 10_000, 100_000, 1_000_000] {
        let mesh: Arc<dyn Mesh> = Arc::new(UniformGrid1D::new(n, -1.0, 1.0).unwrap());
        let field = ContextValue::ScalarField(DVector::from_iterator(
            n,
            (0..n).map(|i| (i as f64 * 0.01).sin()),
        ));

        let seq_grad = FDGradientCalculator::new(Arc::clone(&mesh), 0, None, FDScheme::Central)
            .with_parallel_threshold(usize::MAX);
        let seq_lap = FDLaplacianCalculator::new(Arc::clone(&mesh), laplacian_variable())
            .with_parallel_threshold(usize::MAX);
        group.bench_with_input(BenchmarkId::new("sequential", n), &n, |b, _| {
            b.iter(|| {
                let g = seq_grad.compute(&field, &ctx).unwrap();
                let l = seq_lap.compute(&field, &ctx).unwrap();
                black_box((g, l))
            })
        });

        let par_grad = FDGradientCalculator::new(Arc::clone(&mesh), 0, None, FDScheme::Central)
            .with_parallel_threshold(1);
        let par_lap = FDLaplacianCalculator::new(Arc::clone(&mesh), laplacian_variable())
            .with_parallel_threshold(1);
        group.bench_with_input(BenchmarkId::new("parallel", n), &n, |b, _| {
            b.iter(|| {
                let g = par_grad.compute(&field, &ctx).unwrap();
                let l = par_lap.compute(&field, &ctx).unwrap();
                black_box((g, l))
            })
        });
    }

    group.finish();
}
