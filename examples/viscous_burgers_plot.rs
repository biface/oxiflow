//! # Example: viscous_burgers reference plot (#145)
//!
//! `u(x)` profiles at several times showing the traveling front, overlaid
//! with the analytical Cole-Hopf traveling-shock solution -- same
//! convention as `diffusion_1d_plot.rs`.
//!
//! The model comes from `rsc/models/viscous_burgers.rs`, shared with
//! `examples/viscous_burgers.rs` (see that file's module doc for the
//! physics and the exact-solution derivation).
//!
//! Run with: `cargo run --example viscous_burgers_plot` -- writes
//! `viscous_burgers.svg` to the working directory.

use std::sync::Arc;

use oxiflow::{
    context::calculators::{FDGradientCalculator, FDLaplacianCalculator, FDScheme},
    mesh::{Mesh, UniformGrid1D},
    solver::{
        config::{IntegratorKind, StepControl, TimeConfiguration},
        methods::ForwardEulerSolver,
        scenario::Scenario,
        Solver, SolverConfiguration,
    },
};
use plotters::prelude::*;

#[path = "../rsc/models/viscous_burgers.rs"]
mod viscous_burgers_model;

use viscous_burgers_model::{laplacian_variable, ViscousBurgers};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    const N_NODES: usize = 401;
    const NU: f64 = 0.05;
    const U_L: f64 = 1.0;
    const U_R: f64 = 0.0;
    const X0: f64 = -2.0;
    const T_END: f64 = 2.0;
    const DOMAIN_HALF_WIDTH: f64 = 3.0;
    const N_SNAPSHOTS: usize = 5;

    let model = ViscousBurgers {
        nu: NU,
        u_l: U_L,
        u_r: U_R,
        x0: X0,
    };
    let mesh = UniformGrid1D::new(N_NODES, -DOMAIN_HALF_WIDTH, DOMAIN_HALF_WIDTH)?;
    let dx = 2.0 * DOMAIN_HALF_WIDTH / (N_NODES as f64 - 1.0);
    let gradient_mesh: Arc<dyn Mesh> = Arc::new(UniformGrid1D::new(
        N_NODES,
        -DOMAIN_HALF_WIDTH,
        DOMAIN_HALF_WIDTH,
    )?);
    let laplacian_mesh: Arc<dyn Mesh> = Arc::new(UniformGrid1D::new(
        N_NODES,
        -DOMAIN_HALF_WIDTH,
        DOMAIN_HALF_WIDTH,
    )?);
    let plot_mesh = UniformGrid1D::new(N_NODES, -DOMAIN_HALF_WIDTH, DOMAIN_HALF_WIDTH)?;
    let x_coords: Vec<f64> = (0..N_NODES).map(|i| plot_mesh.coordinates(i)[0]).collect();

    let dt = 0.4 * dx * dx / NU;
    let total_steps = (T_END / dt).round() as usize;
    let save_every = (total_steps / N_SNAPSHOTS).max(1);

    let scenario = Scenario::single(Box::new(model), Box::new(mesh));
    let config = SolverConfiguration::new(
        TimeConfiguration::new(T_END, StepControl::Fixed { dt }).saving_every(save_every),
        IntegratorKind::Euler,
    )
    .with_calculator(Box::new(FDGradientCalculator::new(
        gradient_mesh,
        0,
        None,
        FDScheme::Central,
    )))
    .with_calculator(Box::new(FDLaplacianCalculator::new(
        laplacian_mesh,
        laplacian_variable(),
    )));

    let solver = ForwardEulerSolver;
    let result = solver.solve(&scenario, &config)?;

    let exact_model = ViscousBurgers {
        nu: NU,
        u_l: U_L,
        u_r: U_R,
        x0: X0,
    };

    let root = SVGBackend::new("viscous_burgers.svg", (900, 600)).into_drawing_area();
    root.fill(&WHITE)?;
    let mut chart = ChartBuilder::on(&root)
        .caption(
            "viscous_burgers: u(x,t) vs. analytical traveling-shock solution",
            ("sans-serif", 24),
        )
        .margin(15)
        .x_label_area_size(35)
        .y_label_area_size(45)
        .build_cartesian_2d(-DOMAIN_HALF_WIDTH..DOMAIN_HALF_WIDTH, U_R - 0.1..U_L + 0.1)?;
    chart.configure_mesh().x_desc("x").y_desc("u").draw()?;

    let palette = [&RED, &BLUE, &GREEN, &MAGENTA, &CYAN, &BLACK];
    for (idx, (t, state)) in result.times.iter().zip(&result.states).enumerate() {
        let color = palette[idx % palette.len()];
        let field = state.as_scalar_field()?;

        chart
            .draw_series(LineSeries::new(
                x_coords.iter().zip(field.iter()).map(|(&x, &u)| (x, u)),
                color,
            ))?
            .label(format!("t={t:.3} (simulated)"))
            .legend(move |(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], *color));

        chart.draw_series(
            x_coords
                .iter()
                .step_by(20)
                .map(|&x| Circle::new((x, exact_model.exact(x, *t)), 3, color.filled())),
        )?;
    }

    chart
        .configure_series_labels()
        .background_style(WHITE.mix(0.8))
        .border_style(BLACK)
        .draw()?;
    root.present()?;

    println!(
        "viscous_burgers_plot: wrote viscous_burgers.svg ({} snapshots)",
        result.states.len()
    );
    Ok(())
}
