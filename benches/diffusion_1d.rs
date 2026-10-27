//! Criterion benchmarks for the 1D diffusion model (#136, feeds #53).
//! The model comes from `rsc/models/diffusion_1d.rs`, shared with the
//! example and the analytical test.
//!
//! Distinct measurements, not one:
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
//! - [`boundary_condition_overhead`] — what boundary conditions cost a
//!   solve (#202): the same model without conditions, with the two
//!   Danckwerts conditions and with two `Dirichlet` conditions, under RK4
//!   and Backward Euler. Independent of the `parallel` feature.

use std::hint::black_box;
use std::sync::Arc;

use criterion::BenchmarkId;
use criterion::Criterion;
#[cfg(feature = "parallel")]
use nalgebra::DVector;
#[cfg(feature = "parallel")]
use oxiflow::context::{compute::ComputeContext, value::ContextValue, ContextCalculator};
use oxiflow::{
    boundary::{BoundaryCondition, DanckwertsInlet, DanckwertsOutlet, Dirichlet},
    context::{
        calculators::{ExternalTabulated, FDLaplacianCalculator, Interpolation},
        variable::ContextVariable,
    },
    mesh::{BoundaryRegion, FaceSide, Mesh, UniformGrid1D},
    solver::{
        config::{IntegratorKind, StepControl, TimeConfiguration},
        methods::{BackwardEulerSolver, ForwardEulerSolver, RK4Solver},
        scenario::{Domain, Scenario},
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

/// Boundary conditions of one configuration of
/// [`boundary_condition_overhead`].
#[derive(Clone, Copy)]
enum Boundaries {
    /// No condition: the solver skips every boundary step.
    Without,
    /// `DanckwertsInlet` on the lower face, fed by a tabulated constant,
    /// and `DanckwertsOutlet` on the upper one.
    Danckwerts,
    /// A constant `Dirichlet` condition on each face.
    Dirichlet,
}

impl Boundaries {
    const ALL: [Self; 3] = [Self::Without, Self::Danckwerts, Self::Dirichlet];

    fn label(self) -> &'static str {
        match self {
            Self::Without => "none",
            Self::Danckwerts => "danckwerts",
            Self::Dirichlet => "dirichlet",
        }
    }
}

/// Feed concentration of the Danckwerts inlet.
fn feed_variable() -> ContextVariable {
    ContextVariable::External {
        name: "feed_concentration".into(),
    }
}

/// Scenario and configuration of `n_steps` fixed steps of `dt` on `n`
/// nodes, under `integrator`, with the given boundary conditions.
///
/// The Gaussian stays small at both ends of `[-1, 1]` over these runs
/// (below `1e-6`), so holding them at zero, or feeding zero at the inlet,
/// barely changes the solution: the configurations differ in what the
/// solver does, not in the problem solved. The Danckwerts configuration
/// includes the tabulated feed calculator, part of what that condition
/// costs a user; its dispersion and velocity only set the inlet weight
/// `kappa = D / (v dx)`.
fn bounded_run(
    n: usize,
    integrator: IntegratorKind,
    dt: f64,
    n_steps: u32,
    boundaries: Boundaries,
) -> (Scenario, SolverConfiguration) {
    const ALPHA: f64 = 0.01;
    let lower = BoundaryRegion::Face {
        axis: 0,
        side: FaceSide::Lower,
    };
    let upper = BoundaryRegion::Face {
        axis: 0,
        side: FaceSide::Upper,
    };
    let t_end = f64::from(n_steps) * dt;

    let model = Diffusion1D {
        alpha: ALPHA,
        amplitude: 1.0,
        center: 0.0,
        sigma0: 0.05,
    };
    let mesh_arc: Arc<dyn Mesh> = Arc::new(UniformGrid1D::new(n, -1.0, 1.0).unwrap());
    let mut config = SolverConfiguration::new(
        TimeConfiguration::new(t_end, StepControl::Fixed { dt }),
        integrator,
    )
    .with_calculator(Box::new(FDLaplacianCalculator::new(
        mesh_arc,
        laplacian_variable(),
    )));

    let conditions: Vec<Box<dyn BoundaryCondition>> = match boundaries {
        Boundaries::Without => vec![],
        Boundaries::Danckwerts => {
            config = config.with_calculator(Box::new(
                ExternalTabulated::new(
                    feed_variable(),
                    vec![(0.0, 0.0), (2.0 * t_end, 0.0)],
                    Interpolation::Linear,
                )
                .unwrap(),
            ));
            vec![
                Box::new(DanckwertsInlet::new(ALPHA, 1.0, feed_variable())),
                Box::new(DanckwertsOutlet::new()),
            ]
        }
        Boundaries::Dirichlet => vec![
            Box::new(Dirichlet::constant(lower, 0.0)),
            Box::new(Dirichlet::constant(upper, 0.0)),
        ],
    };
    let domain = Domain::new(
        "bar",
        Box::new(model),
        Box::new(UniformGrid1D::new(n, -1.0, 1.0).unwrap()),
    )
    .with_boundary_conditions(conditions);
    (Scenario::multi(vec![domain]).unwrap(), config)
}

/// What boundary conditions cost a solve (#202): the model without
/// conditions, with the two Danckwerts conditions and with two `Dirichlet`
/// conditions, reported as `<integrator>/<conditions>` for each size.
///
/// - RK4, 200 steps at a fifth of the diffusion limit `dx^2 / alpha`, on
///   101 and 1001 nodes: four derivative evaluations per step, each
///   applying the conditions and holding their nodes (DD-070).
/// - Backward Euler, 5 steps of `1e-3`, on 101 and 401 nodes: every
///   column of the finite-difference Jacobian is one more derivative
///   evaluation, so the per-evaluation cost is multiplied by `n`.
///
/// The difference between a configuration and `none` is the cost of the
/// conditions, calculators they need included. It grows with the number
/// of derivative evaluations, not with the number of nodes: the conditions
/// set two nodes, and the solver's bookkeeping (which calculators the
/// conditions need, which nodes they declare) runs at each evaluation
/// whatever `n` (#201). Not tied to the `parallel` feature: the Laplacian
/// stays below its dispatch threshold at these sizes.
pub fn boundary_condition_overhead(c: &mut Criterion) {
    let mut group = c.benchmark_group("diffusion_1d_boundary_condition_overhead");
    group.sample_size(20);

    for &n in &[101usize, 1001] {
        let dx = 2.0 / (n - 1) as f64;
        let dt = 0.2 * dx * dx / 0.01;
        for boundaries in Boundaries::ALL {
            let (scenario, config) = bounded_run(n, IntegratorKind::RK4, dt, 200, boundaries);
            group.bench_with_input(
                BenchmarkId::new(format!("rk4/{}", boundaries.label()), n),
                &n,
                |b, _| b.iter(|| black_box(RK4Solver.solve(&scenario, &config).unwrap())),
            );
        }
    }

    for &n in &[101usize, 401] {
        for boundaries in Boundaries::ALL {
            let (scenario, config) =
                bounded_run(n, IntegratorKind::BackwardEuler, 1e-3, 5, boundaries);
            let solver = BackwardEulerSolver::new();
            group.bench_with_input(
                BenchmarkId::new(format!("backward_euler/{}", boundaries.label()), n),
                &n,
                |b, _| b.iter(|| black_box(solver.solve(&scenario, &config).unwrap())),
            );
        }
    }

    group.finish();
}
