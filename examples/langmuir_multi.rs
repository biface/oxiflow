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

/// One adsorbing species' physical parameters.
struct SpeciesParams {
    lambda: f64,
    langmuir_k: f64,
    port_number: f64,
    injection: TemporalInjection,
}

/// `dC/dt = -(I + Fe*M)^-1 * ue * dC/dz`, per spatial node.
struct LangmuirMulti {
    species: Vec<SpeciesParams>,
    porosity: f64,
    velocity: f64,
    /// Spatial step -- `column_length / (n_points - 1)`, matching
    /// chrom-rs's own documented formula for `LangmuirMulti::dz` exactly
    /// (its config YAML's literal `dz` field rounds this to 4 decimal
    /// places; the formula itself, not that rounded literal, is what this
    /// crate reproduces).
    dz: f64,
    /// Rayon dispatch threshold for the per-node loop (jacobian assembly
    /// plus an n_species x n_species LU solve, plus a fresh heap
    /// allocation for the jacobian on every row) -- the real cost site
    /// for this model, unlike #136/#137 where a single calculator call
    /// dominated. Measured: `benches/langmuir_multi.rs`'s
    /// `row_dispatch_diagnostic` puts the real crossover between n=100
    /// and n=1,000 -- see [`default_parallel_threshold`]'s own docs and
    /// BENCHMARKS.md for the numbers.
    #[cfg(feature = "parallel")]
    parallel_threshold: usize,
}

/// Back-of-envelope only -- not measured, see `LangmuirMulti::parallel_threshold`'s
/// own docs for the reasoning and `benches/langmuir_multi.rs` for the
/// calibration this default is meant to be replaced by.
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

    /// Jacobian `M = dC_bar/dC` at local concentration row `c` -- exact
    /// port of chrom-rs's `LangmuirMulti::jacobian`.
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

    /// Analytical dilute-limit retention time -- see the module docs.
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

        // Interior gradients, one per species, via the generalized
        // FDGradientCalculator (component: Some(i)) -- row 0 is
        // recomputed below with the injection-aware ghost value instead;
        // the calculator's own (non-injection-aware) value there is
        // simply not read.
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

        // Per-row computation, independent across rows -- reads only
        // c/gradients/species (all shared, immutable), writes nothing
        // shared. Real cost site for this model (jacobian assembly + an
        // n_species x n_species LU solve per row), unlike diffusion_1d/
        // viscous_burgers where a single calculator call dominated.
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

            // Falls back to the identity propagation (dC/dt = -ue*dC/dz,
            // pure advection) if (I + Fe*M) is singular -- same posture
            // chrom-rs documents for its own inverse_propagation: "the
            // caller falls back to the identity matrix" on a singular
            // system, a physically pathological case this reference
            // scenario never actually reaches.
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

        // Sequential assembly into the column-major DMatrix VectorField
        // needs: unlike #144's sites, this pass is proportionally
        // negligible here -- per-row cost (jacobian assembly + an LU
        // solve) is far heavier than a plain copy, so this is not the
        // same "double pass erases the parallel benefit" defect fixed
        // there, just an unavoidable consequence of DMatrix's column-major
        // storage not lining up with row-wise parallel results.
        let mut dc_dt = DMatrix::zeros(n_points, n_species);
        for (row, row_result) in rows.into_iter().enumerate() {
            for (j, val) in row_result.into_iter().enumerate() {
                dc_dt[(row, j)] = val;
            }
        }

        Ok(ContextValue::VectorField(dc_dt))
    }

    fn initial_state(&self, mesh: &dyn Mesh) -> ContextValue {
        // The column starts empty at t=0 -- concentration enters only
        // through the inlet injection profile as time advances.
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
