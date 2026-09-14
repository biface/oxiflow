//! Criterion benchmarks for the Langmuir multi-species model (#138, feeds
//! #53).
//!
//! Same two-measurement split as the other three models, but the second
//! one targets a different site than usual:
//!
//! - [`full_run`] — fixed-size performance on the `ascorbic_erythorbic`
//!   reference scenario (100 points, 4000 steps).
//! - [`row_dispatch_diagnostic`] — isolates the per-node loop (jacobian
//!   assembly + an `n_species x n_species` LU solve, plus a fresh heap
//!   allocation per row), sequential vs. parallel, parameterized by node
//!   count. This is the model's actual cost site -- unlike
//!   diffusion_1d/viscous_burgers, no single calculator call dominates
//!   here, so calibrating a calculator's own dispatch (as those two did)
//!   would not measure anything representative of this model.

use std::hint::black_box;

#[cfg(feature = "parallel")]
use criterion::BenchmarkId;
use criterion::Criterion;
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
#[cfg(feature = "parallel")]
use rayon::prelude::*;
use std::sync::Arc;

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
    #[cfg(feature = "parallel")]
    parallel_threshold: usize,
}

/// Measured: `row_dispatch_diagnostic` below puts the real crossover
/// between n=100 and n=1,000 -- see BENCHMARKS.md.
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

/// Isolates the per-node loop alone -- no calculators, no solver, just the
/// jacobian-assembly-plus-solve cost repeated `n` times, sequential vs.
/// parallel. This is the actual calibration target for this model (see
/// the module docs for why the calculator-dispatch pattern used by
/// diffusion_1d/viscous_burgers doesn't apply here). Only exists when the
/// crate is built with the `parallel` feature -- there is nothing to
/// compare against without it.
#[cfg(feature = "parallel")]
pub fn row_dispatch_diagnostic(c: &mut Criterion) {
    let model = make_model(COLUMN_LENGTH / (N_POINTS as f64 - 1.0));
    let fe = model.fe();
    let ue = model.ue();
    let n_species = model.n_species();
    let identity = DMatrix::<f64>::identity(n_species, n_species);

    let mut group = c.benchmark_group("langmuir_multi_row_dispatch");
    for &n_points in &[100usize, 1_000, 10_000, 100_000] {
        // Representative concentrations -- comparable order of magnitude
        // to the reference scenario's own dilute regime (~1e-6 mol/L),
        // not the exact simulated field: only the dispatch mechanism is
        // being measured here, not the physics.
        let c_field = DMatrix::from_element(n_points, n_species, 1e-6);
        let gradients: Vec<DVector<f64>> = (0..n_species)
            .map(|_| DVector::from_element(n_points, 1e-8))
            .collect();

        let compute_row = |row: usize| -> Vec<f64> {
            let c_row: Vec<f64> = (0..n_species).map(|j| c_field[(row, j)]).collect();
            let m = model.jacobian(&c_row);
            let lhs: DMatrix<f64> = &identity + m.scale(fe);
            let mut rhs = DVector::zeros(n_species);
            for j in 0..n_species {
                rhs[j] = ue * gradients[j][row];
            }
            let x = lhs.clone().lu().solve(&rhs).unwrap_or_else(|| rhs.clone());
            x.iter().map(|&xj| -xj).collect()
        };

        group.bench_with_input(
            BenchmarkId::new("sequential", n_points),
            &n_points,
            |b, &n| {
                b.iter(|| {
                    let rows: Vec<Vec<f64>> = (0..n).map(compute_row).collect();
                    black_box(rows)
                })
            },
        );

        group.bench_with_input(
            BenchmarkId::new("parallel", n_points),
            &n_points,
            |b, &n| {
                b.iter(|| {
                    let rows: Vec<Vec<f64>> = (0..n).into_par_iter().map(compute_row).collect();
                    black_box(rows)
                })
            },
        );
    }
    group.finish();
}
