//! # Integration test: Langmuir multi-species model vs. references (#138)
//!
//! Companion to `examples/langmuir_multi.rs` — same physics, duplicated
//! here (same convention as the other three benchmark models this
//! session). Two independent validation levels, neither alone sufficient:
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

use std::sync::Arc;

use nalgebra::{DMatrix, DVector};
use oxiflow::{
    boundary::TemporalInjection,
    context::{
        calculators::{FDGradientCalculator, FDScheme},
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

const N_POINTS: usize = 100;
const COLUMN_LENGTH: f64 = 0.25;
const TOTAL_TIME: f64 = 800.0;
const N_STEPS: usize = 4000;

struct SpeciesParams {
    lambda: f64,
    langmuir_k: f64,
    port_number: f64,
    injection: TemporalInjection,
}

struct LangmuirMulti {
    species: Vec<SpeciesParams>,
    porosity: f64,
    velocity: f64,
    dz: f64,
    /// See `examples/langmuir_multi.rs`'s own field docs for the full
    /// reasoning -- the real per-node cost site for this model, not yet
    /// measured.
    #[cfg(feature = "parallel")]
    parallel_threshold: usize,
}

/// See `examples/langmuir_multi.rs`'s own docs.
/// Measured: `benches/langmuir_multi.rs`'s `row_dispatch_diagnostic`
/// shows the real crossover between n=100 (1.36x slower) and n=1,000
/// (1.53x faster) on the design machine (12 logical cores) -- see
/// BENCHMARKS.md. `1_000` sits right at that crossover rather than
/// waiting for a larger safety margin, since the win grows quickly past
/// it (2.6x faster by n=10,000). Originally a back-of-envelope `2_000`
/// (picked from between fd.rs's 49_999 and the orchestrator's 5_000, see
/// git history) before this measurement replaced the guess.
#[cfg(feature = "parallel")]
fn default_parallel_threshold() -> usize {
    1_000
}

impl LangmuirMulti {
    fn n_species(&self) -> usize {
        self.species.len()
    }

    fn fe(&self) -> f64 {
        (1.0 - self.porosity) / self.porosity
    }

    fn ue(&self) -> f64 {
        self.velocity / self.porosity
    }

    fn n_bar(&self, i: usize) -> f64 {
        (1.0 - self.porosity) * self.species[i].port_number
    }

    fn jacobian(&self, c: &[f64]) -> DMatrix<f64> {
        let n = self.n_species();
        let sum_kc: f64 = self
            .species
            .iter()
            .zip(c)
            .map(|(sp, &ci)| sp.langmuir_k * ci)
            .sum();
        let denom = 1.0 + sum_kc;
        let denom2 = denom * denom;

        let mut m = DMatrix::zeros(n, n);
        for i in 0..n {
            let ki = self.species[i].langmuir_k;
            let ni = self.n_bar(i);
            for j in 0..n {
                m[(i, j)] = if i == j {
                    self.species[i].lambda + ni * ki * (denom - ki * c[i]) / denom2
                } else {
                    let kj = self.species[j].langmuir_k;
                    -ni * ki * kj * c[i] / denom2
                };
            }
        }
        m
    }

    fn retention_time(&self, i: usize, column_length: f64) -> f64 {
        let t0 = column_length / self.ue();
        let ka0 = self.species[i].lambda + self.n_bar(i) * self.species[i].langmuir_k;
        t0 * (1.0 + self.fe() * ka0)
    }
}

impl RequiresContext for LangmuirMulti {
    fn required_variables(&self) -> Vec<ContextVariable> {
        (0..self.n_species())
            .map(|i| ContextVariable::SpatialGradient {
                dimension: 0,
                component: Some(i),
            })
            .collect()
    }
}

impl PhysicalModel for LangmuirMulti {
    fn compute_physics(
        &self,
        state: &ContextValue,
        ctx: &ComputeContext,
    ) -> Result<ContextValue, OxiflowError> {
        let c = state.as_vector_field()?;
        let n_points = c.nrows();
        let n_species = self.n_species();
        let t = ctx.time();

        let mut gradients: Vec<DVector<f64>> = Vec::with_capacity(n_species);
        for i in 0..n_species {
            let var = ContextVariable::SpatialGradient {
                dimension: 0,
                component: Some(i),
            };
            gradients.push(ctx.external(var)?.as_scalar_field()?.clone());
        }

        let fe = self.fe();
        let ue = self.ue();
        let identity = DMatrix::<f64>::identity(n_species, n_species);

        let compute_row = |row: usize| -> Vec<f64> {
            let c_row: Vec<f64> = (0..n_species).map(|j| c[(row, j)]).collect();
            let m = self.jacobian(&c_row);
            let lhs: DMatrix<f64> = &identity + m.scale(fe);

            let mut rhs = DVector::zeros(n_species);
            for (j, species) in self.species.iter().enumerate() {
                let grad = if row == 0 {
                    let injected = species.injection.evaluate(t);
                    (c[(0, j)] - injected) / self.dz
                } else {
                    gradients[j][row]
                };
                rhs[j] = ue * grad;
            }

            let x = lhs.clone().lu().solve(&rhs).unwrap_or_else(|| rhs.clone());
            x.iter().map(|&xj| -xj).collect()
        };

        #[cfg(feature = "parallel")]
        let rows: Vec<Vec<f64>> = if n_points >= self.parallel_threshold {
            use rayon::prelude::*;
            (0..n_points).into_par_iter().map(compute_row).collect()
        } else {
            (0..n_points).map(compute_row).collect()
        };
        #[cfg(not(feature = "parallel"))]
        let rows: Vec<Vec<f64>> = (0..n_points).map(compute_row).collect();

        let mut dc_dt = DMatrix::zeros(n_points, n_species);
        for (row, row_result) in rows.into_iter().enumerate() {
            for (j, val) in row_result.into_iter().enumerate() {
                dc_dt[(row, j)] = val;
            }
        }

        Ok(ContextValue::VectorField(dc_dt))
    }

    fn initial_state(&self, mesh: &dyn Mesh) -> ContextValue {
        ContextValue::VectorField(DMatrix::zeros(mesh.n_dof(), self.n_species()))
    }

    fn name(&self) -> &str {
        "langmuir_multi"
    }
}

fn make_model(dz: f64) -> LangmuirMulti {
    let injection = TemporalInjection::Rectangle {
        start: 0.0,
        end: 0.4,
        concentration: 0.001,
    };
    LangmuirMulti {
        species: vec![
            SpeciesParams {
                lambda: 1.0,
                langmuir_k: 1.1,
                port_number: 2.0,
                injection: injection.clone(),
            },
            SpeciesParams {
                lambda: 1.0,
                langmuir_k: 1.7,
                port_number: 2.0,
                injection,
            },
        ],
        porosity: 0.4,
        velocity: 0.001,
        dz,
        #[cfg(feature = "parallel")]
        parallel_threshold: default_parallel_threshold(),
    }
}

/// Runs the reference scenario and returns, per species, the outlet's
/// (peak_time, peak_concentration).
fn run_and_find_outlet_peaks() -> Vec<(f64, f64)> {
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
    let dz = COLUMN_LENGTH / (N_POINTS as f64 - 1.0);
    let model = make_model(dz);
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
