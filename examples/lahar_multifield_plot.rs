//! # Example: lahar_multifield reference plots (#145)
//!
//! Two views, matching the paper's own Fig. 4 (Woodhouse, Hogg &
//! Phillips, LaharFlowPaper.pdf):
//!
//! - `lahar_multifield_front.svg` -- flow-front position `x_f(t)` on a
//!   log-log scale, exactly the paper's own presentation for this
//!   quantity.
//! - `lahar_multifield_profiles.svg` -- `h(x)` depth profiles at several
//!   times, showing the flow spreading outward from the source.
//!
//! Both views reuse the single Chezy-drag-dominated run already
//! validated in `tests/lahar_multifield_analytical.rs` (7.7% at t=10s)
//! and used as `benches/lahar_multifield.rs`'s `full_run` fixed-size
//! target -- not a new, unvalidated scenario.
//!
//! The model comes from `rsc/models/lahar_physics.rs`, `lahar_flux.rs`
//! and `lahar_source.rs`, shared with `examples/lahar_multifield.rs` (see
//! that file's module doc for the physics).
//!
//! Run with: `cargo run --example lahar_multifield_plot` -- writes both
//! SVGs to the working directory.

use std::sync::Arc;

use oxiflow::{
    context::calculators::{FDGradientCalculator, FDScheme},
    mesh::{Mesh, UniformGrid1D},
    model::{CompositeModel, DiscretizedModel},
    operators::limiters::Limiter,
    solver::{
        config::{IntegratorKind, StepControl, TimeConfiguration},
        methods::ForwardEulerSolver,
        scenario::Scenario,
        Solver, SolverConfiguration,
    },
};

#[path = "../rsc/models/lahar_flux.rs"]
mod lahar_flux;
#[path = "../rsc/models/lahar_physics.rs"]
mod lahar_physics;
#[path = "../rsc/models/lahar_source.rs"]
mod lahar_source;

#[cfg(feature = "parallel")]
use lahar_flux::default_flux_parallel_threshold;
use lahar_flux::ShallowLayerFlux;
use lahar_physics::{front_position, LaharPhysics, COL_B, COL_M};
use lahar_source::{build_source, ValidationRegime};

use plotters::prelude::*;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    const N_POINTS: usize = 1001;
    const X_HALF_WIDTH: f64 = 50.0;
    const T_END: f64 = 10.0;
    const DT: f64 = 5.0e-4;
    const Q: f64 = 1.0;
    const SAVE_EVERY: usize = 100; // 200 recorded points over 20,000 steps

    let source_mesh = UniformGrid1D::new(N_POINTS, -X_HALF_WIDTH, X_HALF_WIDTH)?;
    let source = build_source(&source_mesh, ValidationRegime::ChezyDragDominated, Q);

    let flux_mesh: Arc<UniformGrid1D> =
        Arc::new(UniformGrid1D::new(N_POINTS, -X_HALF_WIDTH, X_HALF_WIDTH)?);
    let flux = ShallowLayerFlux {
        physics: LaharPhysics {
            rho_f: 1000.0,
            rho_s: 2000.0,
        },
        limiter: Limiter::MinMod,
        #[cfg(feature = "parallel")]
        parallel_threshold: default_flux_parallel_threshold(),
    };
    let discretized = DiscretizedModel::new(Arc::new(flux), flux_mesh, "lahar_flux");

    let composite = CompositeModel::new(
        vec![Box::new(source), Box::new(discretized)],
        "lahar_multifield",
    )?;

    let grad_mesh: Arc<dyn Mesh> =
        Arc::new(UniformGrid1D::new(N_POINTS, -X_HALF_WIDTH, X_HALF_WIDTH)?);
    let config = SolverConfiguration::new(
        TimeConfiguration::new(T_END, StepControl::Fixed { dt: DT }).saving_every(SAVE_EVERY),
        IntegratorKind::Euler,
    )
    .with_calculator(Box::new(FDGradientCalculator::new(
        grad_mesh,
        0,
        Some(COL_B),
        FDScheme::Central,
    )));

    let scenario_mesh = UniformGrid1D::new(N_POINTS, -X_HALF_WIDTH, X_HALF_WIDTH)?;
    let plot_mesh = UniformGrid1D::new(N_POINTS, -X_HALF_WIDTH, X_HALF_WIDTH)?;
    let x_coords: Vec<f64> = (0..N_POINTS).map(|i| plot_mesh.coordinates(i)[0]).collect();

    let solver = ForwardEulerSolver;
    let result = solver.solve(
        &Scenario::single(Box::new(composite), Box::new(scenario_mesh)),
        &config,
    )?;

    // ── View 1: x_f(t), log-log ──────────────────────────────────────
    // t=0 excluded -- undefined on a log scale, and x_f=0 there anyway.
    let front_trajectory: Vec<(f64, f64)> = result
        .times
        .iter()
        .zip(&result.states)
        .filter(|(&t, _)| t > 0.0)
        .map(|(&t, state)| {
            let m_col = state.as_vector_field().unwrap().column(COL_M).into_owned();
            (t, front_position(&m_col, &x_coords, 1000.0))
        })
        .filter(|&(_, xf)| xf > 0.0)
        .collect();

    let t_min = front_trajectory.first().map(|&(t, _)| t).unwrap_or(DT);
    let xf_max = front_trajectory
        .iter()
        .map(|&(_, xf)| xf)
        .fold(0.0, f64::max);

    let root = SVGBackend::new("lahar_multifield_front.svg", (900, 600)).into_drawing_area();
    root.fill(&WHITE)?;
    let mut chart = ChartBuilder::on(&root)
        .caption(
            "lahar_multifield: flow front position x_f(t), log-log",
            ("sans-serif", 24),
        )
        .margin(15)
        .x_label_area_size(35)
        .y_label_area_size(55)
        .build_cartesian_2d(
            (t_min..T_END).log_scale(),
            (front_trajectory
                .iter()
                .map(|&(_, xf)| xf)
                .fold(f64::MAX, f64::min)
                .max(1e-3)..xf_max * 1.1)
                .log_scale(),
        )?;
    chart
        .configure_mesh()
        .x_desc("t (s)")
        .y_desc("x_f (m)")
        .draw()?;
    chart
        .draw_series(LineSeries::new(front_trajectory.iter().copied(), &RED))?
        .label("simulated (Chezy-drag-dominated)")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], RED));
    chart
        .configure_series_labels()
        .background_style(WHITE.mix(0.8))
        .border_style(BLACK)
        .draw()?;
    root.present()?;

    // ── View 2: h(x) profiles at several times ───────────────────────
    const N_SNAPSHOTS: usize = 5;
    let stride = (result.states.len() / N_SNAPSHOTS).max(1);
    let mut h_max = 0.0_f64;
    let mut snapshots: Vec<(f64, Vec<f64>)> = Vec::new();
    for (idx, (t, state)) in result.times.iter().zip(&result.states).enumerate() {
        if idx % stride == 0 {
            let m_col = state.as_vector_field()?.column(COL_M).into_owned();
            let h: Vec<f64> = m_col.iter().map(|&m| m / 1000.0).collect();
            h_max = h_max.max(h.iter().copied().fold(0.0, f64::max));
            snapshots.push((*t, h));
        }
    }

    let root2 = SVGBackend::new("lahar_multifield_profiles.svg", (900, 600)).into_drawing_area();
    root2.fill(&WHITE)?;
    let mut chart2 = ChartBuilder::on(&root2)
        .caption(
            "lahar_multifield: h(x) profiles over time",
            ("sans-serif", 24),
        )
        .margin(15)
        .x_label_area_size(35)
        .y_label_area_size(45)
        .build_cartesian_2d(-X_HALF_WIDTH..X_HALF_WIDTH, 0.0f64..h_max * 1.1)?;
    chart2
        .configure_mesh()
        .x_desc("x (m)")
        .y_desc("h (m)")
        .draw()?;

    let palette = [&RED, &BLUE, &GREEN, &MAGENTA, &CYAN, &BLACK];
    for (idx, (t, h)) in snapshots.iter().enumerate() {
        let color = palette[idx % palette.len()];
        chart2
            .draw_series(LineSeries::new(
                x_coords.iter().zip(h).map(|(&x, &hv)| (x, hv)),
                color,
            ))?
            .label(format!("t={t:.2}"))
            .legend(move |(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], *color));
    }
    chart2
        .configure_series_labels()
        .background_style(WHITE.mix(0.8))
        .border_style(BLACK)
        .draw()?;
    root2.present()?;

    println!(
        "lahar_multifield_plot: wrote lahar_multifield_front.svg and lahar_multifield_profiles.svg \
         ({} recorded points)",
        result.states.len()
    );
    Ok(())
}
