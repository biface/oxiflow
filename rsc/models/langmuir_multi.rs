//! Multi-compound Langmuir model (competitive adsorption in liquid
//! chromatography), ported from chrom-rs's `LangmuirMulti`, with its
//! `ascorbic_erythorbic` reference parameters.
//!
//! Single definition shared by `examples/langmuir_multi.rs`,
//! `examples/langmuir_multi_plot.rs`, `tests/langmuir_multi_reference.rs`
//! and `benches/langmuir_multi.rs`, each of which includes this file as a
//! module with `#[path]`. The model equations, the inlet treatment and the
//! validation levels are documented in `examples/langmuir_multi.rs`.
//! Scenario constants (point count, column length, time span) stay in each
//! consumer.

use nalgebra::{DMatrix, DVector};
use oxiflow::{
    boundary::TemporalInjection,
    context::{
        compute::ComputeContext, error::OxiflowError, value::ContextValue,
        variable::ContextVariable,
    },
    mesh::Mesh,
    model::traits::{PhysicalModel, RequiresContext},
};

/// One adsorbing species' physical parameters.
pub(crate) struct SpeciesParams {
    pub(crate) lambda: f64,
    pub(crate) langmuir_k: f64,
    pub(crate) port_number: f64,
    pub(crate) injection: TemporalInjection,
}

/// `dC/dt = -(I + Fe*M)^-1 * ue * dC/dz`, per spatial node.
pub(crate) struct LangmuirMulti {
    pub(crate) species: Vec<SpeciesParams>,
    pub(crate) porosity: f64,
    pub(crate) velocity: f64,
    /// Spatial step -- `column_length / (n_points - 1)`, matching
    /// chrom-rs's own documented formula for `LangmuirMulti::dz` exactly
    /// (its config YAML's literal `dz` field rounds this to 4 decimal
    /// places; the formula itself, not that rounded literal, is what this
    /// crate reproduces).
    pub(crate) dz: f64,
    /// Rayon dispatch threshold for the per-node loop (jacobian assembly
    /// plus an n_species x n_species LU solve, plus a fresh heap
    /// allocation for the jacobian on every row) -- the real cost site
    /// for this model, unlike #136/#137 where a single calculator call
    /// dominated. Measured: `benches/langmuir_multi.rs`'s
    /// `row_dispatch_diagnostic` puts the real crossover between n=100
    /// and n=1,000 -- see [`default_parallel_threshold`]'s own docs for
    /// the numbers (measured at v0.9.0, #53).
    #[cfg(feature = "parallel")]
    pub(crate) parallel_threshold: usize,
}

/// Measured: `benches/langmuir_multi.rs`'s `row_dispatch_diagnostic`
/// shows the real crossover between n=100 (1.36x slower) and n=1,000
/// (1.53x faster) on the design machine (12 logical cores), measured at
/// v0.9.0 (#53). `1_000` sits right at that crossover rather than
/// waiting for a larger safety margin, since the win grows quickly past
/// it (2.6x faster by n=10,000). Originally a back-of-envelope `2_000`
/// (picked from between fd.rs's 49_999 and the orchestrator's 5_000, see
/// git history) before this measurement replaced the guess. At v0.9.0
/// the diagnostic timed a copy of the row loop; it now times this
/// model's `compute_physics`, so the numbers above hold until the next
/// measurement.
#[cfg(feature = "parallel")]
pub(crate) fn default_parallel_threshold() -> usize {
    1_000
}

impl LangmuirMulti {
    pub(crate) fn n_species(&self) -> usize {
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

    /// Analytical dilute-limit retention time -- see the module docs of
    /// `examples/langmuir_multi.rs`.
    // Used by the examples and the reference test, not by the benchmark,
    // which only times the solver: unused in that one target.
    #[allow(dead_code)]
    pub(crate) fn retention_time(&self, i: usize, column_length: f64) -> f64 {
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

/// The `ascorbic_erythorbic` reference case: two species, rectangular
/// injection of 0.001 mol/L over `[0, 0.4]` s, shared by every consumer so
/// that the example, the test and the benchmark run the same column.
pub(crate) fn make_model(dz: f64) -> LangmuirMulti {
    ascorbic_erythorbic(
        dz,
        TemporalInjection::Rectangle {
            start: 0.0,
            end: 0.4,
            concentration: 0.001,
        },
    )
}

/// The `ascorbic_erythorbic` column and isotherm, with both species fed
/// through the same inlet profile `injection`.
pub(crate) fn ascorbic_erythorbic(dz: f64, injection: TemporalInjection) -> LangmuirMulti {
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
