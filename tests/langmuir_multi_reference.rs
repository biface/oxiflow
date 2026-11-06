//! # Integration test: Langmuir multi-species model vs. references ([#138](https://github.com/biface/oxiflow/issues/138))
//!
//! Companion to `examples/langmuir_multi.rs`, on the same model
//! (`rsc/models/langmuir_multi.rs`). Two independent validation levels,
//! neither alone sufficient:
//!
//! - **Analytical** (dilute/linear regime): exact retention time formula,
//!   independent of chrom-rs's own numerics entirely.
//! - **Numerical**: peak time/value compared against a real run of
//!   chrom-rs's own binary on the `ascorbic_erythorbic` reference case
//!   (session artifact — not chrom-rs source code, just its numeric
//!   output, used here as a reference dataset).
//!
//! The two disagree by design at the ~1% level (verified during design via
//! a standalone Rust replica, outside oxiflow, before writing this test):
//! this port's own peak time (448.2s) tracks the *analytical* retention
//! time (448.0s) slightly more closely than chrom-rs's actual binary output
//! (443.8s) does — a small, expected discretization difference between two
//! independent implementations of the same continuous equations, not a
//! defect in either one. The tolerances below reflect that: tight against
//! the analytical formula, looser against chrom-rs's own numbers.

use oxiflow::solver::{
    config::IntegratorKind, methods::ForwardEulerSolver, scenario::Scenario, Solver,
    SolverConfiguration,
};

#[path = "../rsc/models/langmuir_multi.rs"]
mod langmuir_multi_model;

use langmuir_multi_model::{
    column_grid, column_spacing, gradient_calculators, make_model, reference_time, COLUMN_LENGTH,
    N_POINTS,
};

/// Runs the reference scenario and returns, per species, the outlet's
/// (peak_time, peak_concentration).
fn run_and_find_outlet_peaks() -> Vec<(f64, f64)> {
    let model = make_model(column_spacing(N_POINTS));
    let n_species = model.n_species();
    let mesh = column_grid(N_POINTS).unwrap();

    let mut config = SolverConfiguration::new(reference_time(), IntegratorKind::Euler);
    for calculator in gradient_calculators(
        N_POINTS,
        n_species,
        #[cfg(feature = "parallel")]
        None,
    )
    .unwrap()
    {
        config = config.with_calculator(calculator);
    }

    let solver = ForwardEulerSolver;
    let scenario = Scenario::single(Box::new(model), Box::new(mesh));
    let result = solver.solve(&scenario, &config).unwrap();

    (0..n_species)
        .map(|i| {
            result
                .states
                .iter()
                .zip(&result.times)
                .map(|(state, &t)| (t, state.as_vector_field().unwrap()[(N_POINTS - 1, i)]))
                .fold(
                    (0.0, f64::MIN),
                    |acc, (t, c)| if c > acc.1 { (t, c) } else { acc },
                )
        })
        .collect()
}

/// Relative-error helper — every quantity checked here (retention times in
/// seconds, concentrations in mol/L) spans orders of magnitude, so a
/// relative tolerance is the meaningful one, not a fixed absolute bound.
fn relative_error(observed: f64, reference: f64) -> f64 {
    (observed - reference).abs() / reference.abs()
}

#[test]
fn outlet_peak_time_matches_analytical_retention_time_within_one_percent() {
    let model = make_model(column_spacing(N_POINTS));
    let peaks = run_and_find_outlet_peaks();

    for (i, &(observed_t, _)) in peaks.iter().enumerate() {
        let expected = model.retention_time(i, COLUMN_LENGTH);
        let err = relative_error(observed_t, expected);
        assert!(
            err < 0.01,
            "species {i}: peak time {observed_t:.2}s vs analytical retention time \
             {expected:.2}s, relative error {err:.4} >= 0.01"
        );
    }
}

/// Reference values from a real chrom-rs binary run of the
/// `ascorbic_erythorbic` config case (session artifact, not chrom-rs
/// source): (peak_time_s, peak_concentration_mol_per_l) per species.
const CHROM_RS_REFERENCE_PEAKS: [(f64, f64); 2] = [
    (443.8, 3.659378e-6), // Ascorbic
    (550.6, 2.935205e-6), // Erythorbic
];

#[test]
fn outlet_peaks_match_chrom_rs_reference_within_documented_tolerance() {
    let peaks = run_and_find_outlet_peaks();

    for (i, &(ref_t, ref_c)) in CHROM_RS_REFERENCE_PEAKS.iter().enumerate() {
        let (observed_t, observed_c) = peaks[i];
        let t_err = relative_error(observed_t, ref_t);
        let c_err = relative_error(observed_c, ref_c);
        // 3% -- comfortable margin above the ~1% gap measured during design
        // (standalone replica vs. the same chrom-rs reference), itself
        // consistent with two independent discretizations of the same
        // continuous equations, not a porting defect.
        assert!(
            t_err < 0.03,
            "species {i}: peak time {observed_t:.2}s vs chrom-rs reference {ref_t:.2}s, \
             relative error {t_err:.4} >= 0.03"
        );
        assert!(
            c_err < 0.03,
            "species {i}: peak concentration {observed_c:e} vs chrom-rs reference {ref_c:e}, \
             relative error {c_err:.4} >= 0.03"
        );
    }
}
