//! Criterion benchmarks for the 1D diffusion model (#136, feeds #53).
//! The model comes from `rsc/models/diffusion_1d.rs`, shared with the
//! example and the analytical test.
//!
//! Two distinct measurements, not one:
//!
//! - [`full_run`] — the fixed-size performance target from #53's original
//!   acceptance table ("1D diffusion, 1000 points, 10 000 steps, < 100
//!   ms"). Criterion itself does not assert against that target -- compare
//!   the reported mean against it by hand once results are in.
//! - [`laplacian_dispatch`] — parameterized across problem size via
//!   `BenchmarkId`, sequential vs. parallel forced through the public
//!   `with_parallel_threshold` API (`usize::MAX` / `1`) on
//!   `FDLaplacianCalculator` directly. This is the actual crossover
//!   measurement for this model's per-node cost profile -- see the
//!   session's benchmark plan for why Criterion, not
//!   `calibrate_parallel_threshold`, is the instrument for calibrating
//!   shipped defaults (statistical rigor: warm-up, repeated samples,
//!   outlier detection -- a single `Instant::now()` pair does not have
//!   that).
//! - [`implicit_jacobian_dispatch`] — dispatch of the implicit solvers'
//!   finite-difference Jacobian assembly (#199), sequential vs. parallel,
//!   through one Backward Euler step: dense path, and banded path with the
//!   `sparse` feature.

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
    context::calculators::FDLaplacianCalculator,
    mesh::{Mesh, UniformGrid1D},
    solver::{
        config::{IntegratorKind, StepControl, TimeConfiguration},
        methods::ForwardEulerSolver,
        scenario::Scenario,
        Solver, SolverConfiguration,
    },
};

#[path = "../rsc/models/diffusion_1d.rs"]
mod diffusion_1d_model;

use diffusion_1d_model::{laplacian_variable, Diffusion1D};

/// #53's original fixed-size target: 1000 points, 10 000 steps, < 100 ms.
///
/// `dt` chosen only for numerical stability at this `N` (alpha*dt/dx^2 <=
/// 0.5) -- this is a performance benchmark, not an accuracy one, so it
/// does not need the much smaller `dt` `tests/diffusion_1d_analytical.rs`
/// uses to isolate spatial convergence order.
pub fn full_run(c: &mut Criterion) {
    const N_NODES: usize = 1000;
    const N_STEPS: u64 = 10_000;
    const DT: f64 = 1e-4;
    const ALPHA: f64 = 0.01;

    c.bench_function("diffusion_1d_full_run", |b| {
        b.iter(|| {
            let model = Diffusion1D {
                alpha: ALPHA,
                amplitude: 1.0,
                center: 0.0,
                sigma0: 0.05,
            };
            let mesh = UniformGrid1D::new(N_NODES, -1.0, 1.0).unwrap();
            let mesh_arc: Arc<dyn Mesh> = Arc::new(UniformGrid1D::new(N_NODES, -1.0, 1.0).unwrap());

            let scenario = Scenario::single(Box::new(model), Box::new(mesh));
            let config = SolverConfiguration::new(
                TimeConfiguration::new(N_STEPS as f64 * DT, StepControl::Fixed { dt: DT }),
                IntegratorKind::Euler,
            )
            .with_calculator(Box::new(FDLaplacianCalculator::new(
                mesh_arc,
                laplacian_variable(),
            )));

            let result = ForwardEulerSolver.solve(&scenario, &config).unwrap();
            black_box(result)
        })
    });
}

/// Parallel-threshold crossover for `FDLaplacianCalculator`'s dispatch, as
/// exercised by this model's specific field sizes -- not the full
/// solver loop (which would dilute the calculator's own per-call cost
/// with unrelated solver/context overhead), just the calculator call
/// itself, forced sequential vs. parallel via the public
/// `with_parallel_threshold` API. Only exists when the crate is built
/// with the `parallel` feature -- `with_parallel_threshold` itself only
/// exists then.
#[cfg(feature = "parallel")]
pub fn laplacian_dispatch(c: &mut Criterion) {
    let ctx = ComputeContext::new(0.0, 1e-4);
    let mut group = c.benchmark_group("diffusion_1d_laplacian_dispatch");

    // Extended past 1e6, then past 1e7 (measured at v0.9.0, #53): crossover
    // confirmed between 1e7 (still 1.02x slower) and 3e7 (1.03x faster),
    // plateauing around 3-4% by 1e8 -- the signature of a
    // memory-bandwidth-bound kernel, not one still climbing toward a
    // larger win. Sizes beyond 1e8 were not pursued further on that
    // basis.
    for &n in &[
        1_000usize,
        10_000,
        100_000,
        1_000_000,
        3_000_000,
        10_000_000,
        30_000_000,
        100_000_000,
    ] {
        let mesh: Arc<dyn Mesh> = Arc::new(UniformGrid1D::new(n, -1.0, 1.0).unwrap());
        let field = ContextValue::ScalarField(DVector::from_iterator(
            n,
            (0..n).map(|i| (i as f64 * 0.01).sin()),
        ));

        let sequential = FDLaplacianCalculator::new(Arc::clone(&mesh), laplacian_variable())
            .with_parallel_threshold(usize::MAX);
        group.bench_with_input(BenchmarkId::new("sequential", n), &n, |b, _| {
            b.iter(|| black_box(sequential.compute(&field, &ctx).unwrap()))
        });

        let parallel = FDLaplacianCalculator::new(Arc::clone(&mesh), laplacian_variable())
            .with_parallel_threshold(1);
        group.bench_with_input(BenchmarkId::new("parallel", n), &n, |b, _| {
            b.iter(|| black_box(parallel.compute(&field, &ctx).unwrap()))
        });
    }

    group.finish();
}

/// Diagnostic, not a calibration result in itself: isolates the *raw*
/// Rayon dispatch cost for a stencil-sized per-element closure -- no
/// `FDLaplacianCalculator`, no `compute_from_dx`'s double pass (parallel
/// `collect()` into a temporary `Vec`, then a second sequential copy into
/// the result buffer). If this shows a real crossover on a given machine
/// but `laplacian_dispatch` above does not, the double pass is the
/// dominant cost, not the per-node work itself or the core count -- worth
/// checking before concluding anything about that machine's parallel
/// threshold. Only exists when the crate is built with the `parallel`
/// feature -- there is nothing to compare against without it.
#[cfg(feature = "parallel")]
pub fn raw_rayon_dispatch_diagnostic(c: &mut Criterion) {
    use rayon::prelude::*;

    // Same per-element cost class as the Laplacian stencil (a handful of
    // reads and flops) but entirely self-contained -- no field, no
    // boundary handling, no allocation beyond the one output buffer both
    // paths need.
    let stencil = |i: usize| -> f64 {
        let x = i as f64 * 0.01;
        0.3 * x.sin() + 0.5 * x.cos() - 0.2 * x
    };

    let mut group = c.benchmark_group("raw_rayon_dispatch_diagnostic");
    for &n in &[1_000usize, 10_000, 100_000, 1_000_000] {
        group.bench_with_input(BenchmarkId::new("sequential", n), &n, |b, &n| {
            b.iter(|| {
                let mut out = vec![0.0; n];
                for (i, slot) in out.iter_mut().enumerate() {
                    *slot = stencil(i);
                }
                black_box(out)
            })
        });

        group.bench_with_input(BenchmarkId::new("parallel", n), &n, |b, &n| {
            b.iter(|| {
                let out: Vec<f64> = (0..n).into_par_iter().map(stencil).collect();
                black_box(out)
            })
        });
    }
    group.finish();
}

/// Dispatch of the finite-difference Jacobian assembly of the implicit
/// solvers (#199, DT-069), sequential vs. parallel, through the public
/// `BackwardEulerSolver::with_parallel_threshold`. What is timed is one
/// full Backward Euler step: the threshold changes only the Jacobian
/// assembly, so the difference between the two timings is the assembly's,
/// while the rest of the step (the linear solve above all, `O(n³)` on the
/// dense path) is the same in both. The Laplacian calculator is kept
/// sequential in both, so that only the Jacobian dispatch differs.
///
/// Dense path: one task per column. Banded path (`sparse` feature,
/// bandwidth 1): one task per colour, so at most 3 tasks whatever the
/// size — measures how much that limited parallelism still gains. Only
/// exists when the crate is built with the `parallel` feature.
#[cfg(feature = "parallel")]
pub fn implicit_jacobian_dispatch(c: &mut Criterion) {
    use crate::implicit_dispatch::sequential_vs_parallel;
    use oxiflow::solver::methods::backward_euler::BackwardEulerSolver;

    let mut group = c.benchmark_group("diffusion_1d_implicit_jacobian_dispatch");
    group.sample_size(20);

    for &n in &[64usize, 128, 256, 512] {
        let (scenario, config) = one_backward_euler_step(n);
        sequential_vs_parallel(&mut group, "dense", n, &scenario, &config, |threshold| {
            BackwardEulerSolver::new().with_parallel_threshold(threshold)
        });
    }

    #[cfg(feature = "sparse")]
    for &n in &[1_000usize, 10_000, 100_000] {
        use oxiflow::solver::sparse::FaerSparseSolver;

        let (scenario, config) = one_backward_euler_step(n);
        sequential_vs_parallel(&mut group, "banded", n, &scenario, &config, |threshold| {
            BackwardEulerSolver::new()
                .with_sparse_solver(Box::new(FaerSparseSolver))
                .with_jacobian_bandwidth(1)
                .with_sparse_threshold(0)
                .with_parallel_threshold(threshold)
        });
    }

    group.finish();
}

/// Scenario and configuration of a single Backward Euler step on `n`
/// nodes, with a sequential Laplacian calculator.
#[cfg(feature = "parallel")]
fn one_backward_euler_step(n: usize) -> (Scenario, SolverConfiguration) {
    const DT: f64 = 1e-3;

    let model = Diffusion1D {
        alpha: 0.01,
        amplitude: 1.0,
        center: 0.0,
        sigma0: 0.05,
    };
    let mesh = UniformGrid1D::new(n, -1.0, 1.0).unwrap();
    let mesh_arc: Arc<dyn Mesh> = Arc::new(UniformGrid1D::new(n, -1.0, 1.0).unwrap());

    let scenario = Scenario::single(Box::new(model), Box::new(mesh));
    let config = SolverConfiguration::new(
        TimeConfiguration::new(DT, StepControl::Fixed { dt: DT }),
        IntegratorKind::BackwardEuler,
    )
    .with_calculator(Box::new(
        FDLaplacianCalculator::new(mesh_arc, laplacian_variable())
            .with_parallel_threshold(usize::MAX),
    ));
    (scenario, config)
}
