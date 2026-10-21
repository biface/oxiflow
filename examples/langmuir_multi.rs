//! # Example: multi-compound Langmuir benchmark model (#138, feeds #53)
//!
//! Competitive adsorption in liquid chromatography, ported from chrom-rs's
//! `LangmuirMulti` (session reference: `ascorbic_erythorbic` config case).
//! The heaviest of the four #53 benchmark models: each spatial node needs
//! its own `n_species x n_species` linear solve, not just a stencil
//! evaluation.
//!
//! Run with: `cargo run --example langmuir_multi`
//!
//! The model (`LangmuirMulti`, its parameters and the
//! `ascorbic_erythorbic` reference case built by `make_model`) is defined
//! once in `rsc/models/langmuir_multi.rs`, included below as the
//! `langmuir_multi_model` module and shared with the plot example, the
//! reference test and the benchmark.
//!
//! ## Model equations (chrom-rs's own documentation, reproduced exactly)
//!
//! Competitive Langmuir isotherm for species `i`:
//!
//! ```text
//! C_bar_i = lambda_i * C_i + N_bar_i * K_i * C_i / (1 + sum_j K_j * C_j)
//! ```
//!
//! Jacobian `M_ij = d(C_bar_i)/d(C_j)`:
//!
//! ```text
//! M_ii = lambda_i + N_bar_i * K_i * (denom - K_i * C_i) / denom^2
//! M_ij = -N_bar_i * K_i * K_j * C_i / denom^2               (i != j)
//! ```
//!
//! where `denom = 1 + sum_k K_k * C_k` and `N_bar_i = (1-porosity) * N_i`.
//!
//! Transport: `dC/dt = -(I + Fe*M)^-1 * ue * dC/dz`, `Fe = (1-porosity)/porosity`,
//! `ue = velocity/porosity`. Solved per spatial node (an
//! `n_species x n_species` system each), not once globally — this is the
//! per-node cost that makes this model heavier than diffusion_1d or
//! viscous_burgers.
//!
//! ## Why this crate doesn't use `FDGradientCalculator` for the whole
//! gradient
//!
//! Interior nodes use it directly (`component: Some(i)`, one calculator
//! per species -- the #138 prerequisite that generalized
//! `ContextValue`/`FDGradientCalculator` for `VectorField` state). The
//! inlet node (row 0) is overridden manually: its upstream neighbor isn't
//! a real domain value, it's `injection.evaluate(t)` -- a genuinely
//! time-varying ghost value the generic calculator has no way to supply
//! (see `boundary::injection`'s own module docs for why `ghost_value()`
//! needed a `ctx` parameter for exactly this case).
//!
//! ## Validation
//!
//! Two levels, not one:
//!
//! - **Analytical** (dilute/linear regime): the exact retention time
//!   `t_retention = t0 * (1 + Fe*Ka0)`, `Ka0 = lambda + N_bar*K`,
//!   `t0 = column_length / ue` -- independent of chrom-rs's own numerics.
//! - **Numerical**: peak times and values compared against a real chrom-rs
//!   binary run of the same `ascorbic_erythorbic` reference case.
//!
//! See `tests/langmuir_multi_reference.rs` for both as regression gates.

use std::sync::Arc;

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

#[path = "../rsc/models/langmuir_multi.rs"]
mod langmuir_multi_model;

use langmuir_multi_model::make_model;

fn main() {
    const N_POINTS: usize = 100;
    const COLUMN_LENGTH: f64 = 0.25;
    const TOTAL_TIME: f64 = 800.0;
    const N_STEPS: usize = 4000;

    let dz = COLUMN_LENGTH / (N_POINTS as f64 - 1.0);
    let model = make_model(dz);
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
    for i in 0..model.n_species() {
        let mesh_arc: Arc<dyn Mesh> =
            Arc::new(UniformGrid1D::new(N_POINTS, 0.0, COLUMN_LENGTH).unwrap());
        config = config.with_calculator(Box::new(FDGradientCalculator::new(
            mesh_arc,
            0,
            Some(i),
            FDScheme::Backward,
        )));
    }

    let solver = ForwardEulerSolver;
    let result = solver
        .solve(&Scenario::single(Box::new(model), Box::new(mesh)), &config)
        .unwrap();

    // Re-derive retention times for reporting (the model was moved into
    // the scenario above).
    let report_model = make_model(dz);
    for i in 0..report_model.n_species() {
        let t_retention = report_model.retention_time(i, COLUMN_LENGTH);
        let (peak_t, peak_c) = result
            .states
            .iter()
            .zip(&result.times)
            .map(|(state, &t)| (t, state.as_vector_field().unwrap()[(N_POINTS - 1, i)]))
            .fold(
                (0.0, f64::MIN),
                |acc, (t, c)| if c > acc.1 { (t, c) } else { acc },
            );
        println!(
            "species {i}: analytical t_retention={t_retention:.2}s, simulated peak at outlet: \
             t={peak_t:.2}s, C={peak_c:e}"
        );
    }
}
