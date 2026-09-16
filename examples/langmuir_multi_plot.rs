//! # Example: langmuir_multi reference plot (#145)
//!
//! Chromatogram: outlet concentration `C_i(t)` vs. time, one curve per
//! species -- the standard chrom-rs output this model was ported from
//! (session reference: `ascorbic_erythorbic` config case, same as
//! `examples/langmuir_multi.rs`). Each species' analytical dilute-limit
//! retention time is marked with a vertical line rather than the
//! circle-marker convention used for `diffusion_1d_plot.rs`/
//! `viscous_burgers_plot.rs`: a single scalar time doesn't sit on the
//! curve as a point to mark, but a vertical line at that time is both a
//! standard chromatography annotation and easy in `plotters` (a two-point
//! line series), same reasoning as those two files' choice to avoid
//! `plotters`' unreliable dashed-line support.
//!
//! Model code duplicated from `examples/langmuir_multi.rs` (same
//! convention already established for `tests/`/`benches/` -- see that
//! file's own module doc for the physics).
//!
//! Run with: `cargo run --example langmuir_multi_plot` -- writes
//! `langmuir_multi.svg` to the working directory.

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
use plotters::prelude::*;

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
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    const N_POINTS: usize = 100;
    const COLUMN_LENGTH: f64 = 0.25;
    const TOTAL_TIME: f64 = 800.0;
    const N_STEPS: usize = 4000;
    const SAVE_EVERY: usize = 4; // 1000 recorded points -- smooth enough
                                 // for a chromatogram curve without
                                 // carrying all 4000 raw steps.

    let dz = COLUMN_LENGTH / (N_POINTS as f64 - 1.0);
    let model = make_model(dz);
    let mesh = UniformGrid1D::new(N_POINTS, 0.0, COLUMN_LENGTH)?;

    let mut config = SolverConfiguration::new(
        TimeConfiguration::new(
            TOTAL_TIME,
            StepControl::Fixed {
                dt: TOTAL_TIME / N_STEPS as f64,
            },
        )
        .saving_every(SAVE_EVERY),
        IntegratorKind::Euler,
    );
    for i in 0..model.n_species() {
        let mesh_arc: Arc<dyn Mesh> = Arc::new(UniformGrid1D::new(N_POINTS, 0.0, COLUMN_LENGTH)?);
        config = config.with_calculator(Box::new(FDGradientCalculator::new(
            mesh_arc,
            0,
            Some(i),
            FDScheme::Backward,
        )));
    }

    let solver = ForwardEulerSolver;
    let result = solver.solve(&Scenario::single(Box::new(model), Box::new(mesh)), &config)?;

    let report_model = make_model(dz);
    let n_species = report_model.n_species();

    // Outlet (last node) concentration series per species, plus the peak
    // value across all species -- used only to size the y-axis. The
    // envelope (sum across species at each recorded time) is the total
    // detector signal a real chromatogram shows -- standard convention
    // when individual species peaks overlap.
    let mut series: Vec<Vec<(f64, f64)>> = vec![Vec::new(); n_species];
    let mut envelope: Vec<(f64, f64)> = Vec::new();
    let mut max_c = 0.0_f64;
    for (state, &t) in result.states.iter().zip(&result.times) {
        let c = state.as_vector_field()?;
        let mut total = 0.0_f64;
        for i in 0..n_species {
            let outlet_c = c[(N_POINTS - 1, i)];
            max_c = max_c.max(outlet_c);
            series[i].push((t, outlet_c));
            total += outlet_c;
        }
        max_c = max_c.max(total);
        envelope.push((t, total));
    }

    let root = SVGBackend::new("langmuir_multi.svg", (900, 600)).into_drawing_area();
    root.fill(&WHITE)?;
    let mut chart = ChartBuilder::on(&root)
        .caption(
            "langmuir_multi: chromatogram (outlet concentration vs. time)",
            ("sans-serif", 24),
        )
        .margin(15)
        .x_label_area_size(35)
        .y_label_area_size(55)
        .build_cartesian_2d(0.0f64..TOTAL_TIME, 0.0f64..max_c * 1.1)?;
    chart
        .configure_mesh()
        .x_desc("time (s)")
        .y_desc("outlet concentration")
        .draw()?;

    // Envelope (total signal) -- solid gray line: the continuous total
    // detector trace a real chromatogram shows.
    let gray = RGBColor(128, 128, 128);
    chart
        .draw_series(LineSeries::new(envelope.iter().copied(), &gray))?
        .label("envelope (total signal)")
        .legend(move |(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], gray));

    // Individual species -- widely-spaced dots rather than solid lines,
    // so the continuous gray envelope reads as the primary curve and
    // each species' own contribution as an overlay on it. Same
    // circle-marker technique as diffusion_1d_plot.rs/
    // viscous_burgers_plot.rs's analytical overlays, for the same
    // reason (`plotters` has no reliable dashed-line style).
    let palette = [&RED, &BLUE, &GREEN, &MAGENTA];
    for i in 0..n_species {
        let color = palette[i % palette.len()];
        chart
            .draw_series(
                series[i]
                    .iter()
                    .step_by(15)
                    .map(|&(t, c)| Circle::new((t, c), 2, color.filled())),
            )?
            .label(format!("species {i}"))
            .legend(move |(x, y)| Circle::new((x + 10, y), 2, color.filled()));

        // Analytical dilute-limit retention time -- a solid vertical
        // line rather than a circle marker (see module doc).
        let t_retention = report_model.retention_time(i, COLUMN_LENGTH);
        chart.draw_series(LineSeries::new(
            vec![(t_retention, 0.0), (t_retention, max_c * 1.1)],
            color.stroke_width(1),
        ))?;
    }

    chart
        .configure_series_labels()
        .background_style(WHITE.mix(0.8))
        .border_style(BLACK)
        .draw()?;
    root.present()?;

    println!(
        "langmuir_multi_plot: wrote langmuir_multi.svg ({} species, {} recorded points)",
        n_species,
        result.states.len()
    );
    Ok(())
}
