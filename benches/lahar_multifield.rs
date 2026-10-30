//! Criterion benchmarks for the lahar multi-field model ([#139](https://github.com/biface/oxiflow/issues/139), feeds [#53](https://github.com/biface/oxiflow/issues/53)).
//!
//! The model comes from `rsc/models/lahar_physics.rs`, `lahar_flux.rs`
//! and `lahar_source.rs`, and the validation scenario [`full_run`] times
//! from `rsc/models/lahar_validation.rs`, all shared with the examples and
//! the analytical test.
//!
//! Two dispatch sites were added for this model (session-confirmed:
//! unlike diffusion_1d/viscous_burgers/langmuir_multi, this model needed
//! two independent calibrations, not one -- [DA-048](https://github.com/biface/oxiflow/issues/140) P1/P2/P3, granularity
//! is a per-site decision):
//!
//! - [`full_run`] -- fixed-size performance on the Chezy-drag-dominated
//!   validation scenario (1001 points, 10s / 20,000 steps) already
//!   confirmed against the analytical scaling law in
//!   `tests/lahar_multifield_analytical.rs`. Doesn't engage erosion/
//!   deposition/bed evolution (that scenario is deliberately
//!   non-erodible, per paper Sec. 4.1) -- a bulking-representative
//!   `full_run` variant can be added once one exists as a validated
//!   reference scenario, same as this one was.
//! - [`source_dispatch_diagnostic`] -- times `LaharSource::compute_physics`
//!   with its per-node loop (three-branch Pouliquen-Forterre friction plus
//!   the Shields/settling-velocity closures) forced sequential or
//!   parallel, parameterized by node count.
//! - [`flux_dispatch_diagnostic`] -- times `ShallowLayerFlux::apply` with
//!   its divergence loop (two Rusanov face evaluations per node) forced
//!   the same way, same sweep. Expected cheaper per-node than the source loop (no
//!   `powf`/`tanh`-heavy closures, no allocation), so likely a higher
//!   crossover -- to be confirmed by the measurement itself, not assumed.

#[cfg(feature = "parallel")]
use nalgebra::{DMatrix, DVector};
use oxiflow::solver::{
    config::{IntegratorKind, StepControl, TimeConfiguration},
    methods::ForwardEulerSolver,
    Solver, SolverConfiguration,
};
#[cfg(feature = "parallel")]
use oxiflow::{
    context::{compute::ComputeContext, value::ContextValue, variable::ContextVariable},
    mesh::UniformGrid1D,
    model::PhysicalModel,
    operators::{limiters::Limiter, FluxDivergenceOperator},
};

use crate::lahar_validation;

#[cfg(feature = "parallel")]
use crate::lahar_flux::ShallowLayerFlux;
#[cfg(feature = "parallel")]
use crate::lahar_physics::{
    ErosionDepositionParams, FrictionParams, LaharPhysics, SourceInjection, COL_B, COL_M, COL_P,
    COL_S,
};
#[cfg(feature = "parallel")]
use crate::lahar_source::LaharSource;
use crate::lahar_source::ValidationRegime;

use std::hint::black_box;

#[cfg(feature = "parallel")]
use criterion::BenchmarkId;
use criterion::Criterion;

/// [#53](https://github.com/biface/oxiflow/issues/53)'s fixed-size target for this model: the Chezy-drag-dominated
/// scenario already confirmed against the analytical scaling law
/// (`tests/lahar_multifield_analytical.rs`, 7.7% at t=10s) -- the same
/// scenario as that test, built by `rsc/models/lahar_validation.rs`, minus
/// the trajectory recording. The scenario is built inside the timed loop,
/// as before.
pub fn full_run(c: &mut Criterion) {
    const N_POINTS: usize = 1001;
    const X_HALF_WIDTH: f64 = 50.0;
    const T_END: f64 = 10.0;
    const DT: f64 = 5.0e-4;

    c.bench_function("lahar_multifield_full_run", |b| {
        b.iter(|| {
            let config = SolverConfiguration::new(
                TimeConfiguration::new(T_END, StepControl::Fixed { dt: DT }),
                IntegratorKind::Euler,
            )
            .with_calculator(Box::new(
                lahar_validation::bed_gradient(
                    N_POINTS,
                    X_HALF_WIDTH,
                    #[cfg(feature = "parallel")]
                    None,
                )
                .unwrap(),
            ));
            let scenario = lahar_validation::scenario(
                ValidationRegime::ChezyDragDominated,
                N_POINTS,
                X_HALF_WIDTH,
                #[cfg(feature = "parallel")]
                None,
            )
            .unwrap();
            let result = ForwardEulerSolver.solve(&scenario, &config).unwrap();
            black_box(result)
        })
    });
}

/// Dispatch of `LaharSource`'s per-node loop (three-branch
/// Pouliquen-Forterre friction plus the Shields/settling-velocity
/// closures), sequential vs. parallel, parameterized by node count. The
/// timed code is the shared model's own `compute_physics`: two instances
/// are built, threshold `usize::MAX` and `1`, the way `diffusion_1d`'s
/// `laplacian_dispatch` forces its calculator, so a change to the model
/// reaches this measurement. The state is representative but synthetic
/// (h~0.5m, u~2m/s, moderate solids fraction, flat bed with a uniform
/// slope; same order of magnitude as a real flow, not a simulated field),
/// with an erodible bed and granular friction so that every closure runs.
/// The bed-slope gradient the model reads is inserted into the
/// `ComputeContext` once, outside the timed loop.
#[cfg(feature = "parallel")]
pub fn source_dispatch_diagnostic(c: &mut Criterion) {
    let physics = LaharPhysics {
        rho_f: 1000.0,
        rho_s: 2000.0,
    };
    let friction = FrictionParams {
        cd: 0.04,
        tan_d1: 0.1,
        tan_d2: 0.4,
        tan_d3: 0.1175,   // tan(6.7 deg), paper's own d1 ~= d3 - 1 deg reduction
        length_l: 1.5e-3, // 3*d/2, d=1mm (paper's own choice)
        beta: 0.136,
        gamma: 1.0e-3,
        alpha: 5.0,
        psi0: 0.4,
    };
    let erosion = ErosionDepositionParams {
        epsilon: 0.001,
        nu: 1.2e-6,
        grain_d: 1.0e-3,
        psi_b: 0.6,
    };
    let injection = SourceInjection {
        rate: 1.0,
        x_lo: -0.5,
        x_hi: 0.5,
        psi: 0.0,
    };

    let h = 0.5_f64;
    let u = 2.0_f64;
    let psi = 0.1_f64;
    let m = physics.rho_bar(psi) * h;
    let p = m * u;
    let s = h * psi;

    let mut group = c.benchmark_group("lahar_multifield_source_dispatch");
    for &n in &[100usize, 1_000, 10_000, 100_000] {
        let mesh = UniformGrid1D::new(n, -50.0, 50.0).unwrap();
        let mut state = DMatrix::zeros(n, 4);
        state.column_mut(COL_M).fill(m);
        state.column_mut(COL_P).fill(p);
        state.column_mut(COL_S).fill(s);
        let state = ContextValue::VectorField(state);
        let mut ctx = ComputeContext::new(0.0, 5.0e-4);
        ctx.insert(
            ContextVariable::SpatialGradient {
                dimension: 0,
                component: Some(COL_B),
            },
            ContextValue::ScalarField(DVector::from_element(n, 0.01)),
        );

        let sequential = LaharSource::new(
            &mesh,
            physics,
            friction,
            erosion,
            injection,
            0.0,
            usize::MAX,
        );
        group.bench_with_input(BenchmarkId::new("sequential", n), &n, |b, _| {
            b.iter(|| black_box(sequential.compute_physics(&state, &ctx).unwrap()))
        });
        let parallel = LaharSource::new(&mesh, physics, friction, erosion, injection, 0.0, 1);
        group.bench_with_input(BenchmarkId::new("parallel", n), &n, |b, _| {
            b.iter(|| black_box(parallel.compute_physics(&state, &ctx).unwrap()))
        });
    }
    group.finish();
}

/// Dispatch of `ShallowLayerFlux`'s divergence loop (two Rusanov face
/// evaluations per node, after MinMod reconstruction), sequential vs.
/// parallel, parameterized by node count. The timed code is the shared
/// operator's own `apply`, with `parallel_threshold` set to `usize::MAX`
/// and to `1`. The state is representative but synthetic (same reasoning
/// as `source_dispatch_diagnostic`): a uniform flow, so the reconstruction
/// and the domain-edge fallback run as in a real call.
#[cfg(feature = "parallel")]
pub fn flux_dispatch_diagnostic(c: &mut Criterion) {
    let physics = LaharPhysics {
        rho_f: 1000.0,
        rho_s: 2000.0,
    };
    let dx = 0.1_f64;

    let h = 0.5_f64;
    let u = 2.0_f64;
    let psi = 0.1_f64;
    let m = physics.rho_bar(psi) * h;
    let p = m * u;
    let s = h * psi;

    let mut group = c.benchmark_group("lahar_multifield_flux_dispatch");
    for &n in &[100usize, 1_000, 10_000, 100_000, 1_000_000] {
        let mesh = UniformGrid1D::new(n, 0.0, dx * (n as f64 - 1.0)).unwrap();
        let mut state = DMatrix::zeros(n, 4);
        state.column_mut(COL_M).fill(m);
        state.column_mut(COL_P).fill(p);
        state.column_mut(COL_S).fill(s);
        let state = ContextValue::VectorField(state);
        let ctx = ComputeContext::new(0.0, 5.0e-4);

        let sequential = ShallowLayerFlux {
            physics,
            limiter: Limiter::MinMod,
            parallel_threshold: usize::MAX,
        };
        group.bench_with_input(BenchmarkId::new("sequential", n), &n, |b, _| {
            b.iter(|| black_box(sequential.apply(&state, &mesh, &ctx).unwrap()))
        });
        let parallel = ShallowLayerFlux {
            physics,
            limiter: Limiter::MinMod,
            parallel_threshold: 1,
        };
        group.bench_with_input(BenchmarkId::new("parallel", n), &n, |b, _| {
            b.iter(|| black_box(parallel.apply(&state, &mesh, &ctx).unwrap()))
        });
    }
    group.finish();
}
