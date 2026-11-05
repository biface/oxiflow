//! # Example: burgers_boundary_layer result plots ([#149](https://github.com/biface/oxiflow/issues/149))
//!
//! Replays the runs of `examples/burgers_boundary_layer.rs` at Re = 50, 100
//! and 200 and draws the steady boundary layers they reach at t = 2.4
//! against the exact steady profile. The scenario comes from
//! `rsc/models/burgers_boundary_layer.rs`, shared with the example, whose
//! module documentation describes the physics and the validation; this file
//! draws and does not check.
//!
//! - `burgers_boundary_layer_profiles.svg` -- u against the distance from
//!   the wall 1 - x, the exact profiles A tanh(A (1 - x) / (2 nu)) as lines
//!   and RK4 as dots (every other node), with the exact 99 % thickness of
//!   each layer dashed.
//! - `burgers_boundary_layer_collapse.svg` -- the same states against
//!   (1 - x) / nu: the three layers fall on one curve, tanh of half that
//!   variable (A = 1 to double precision), and every node is drawn. The
//!   departure grows with the cell Reynolds number dx / nu, which is the
//!   only parameter of the discrete error.
//!
//! Run with: `cargo run --release --example burgers_boundary_layer_plot` --
//! writes both files to the working directory.

use std::error::Error;

use plotters::prelude::*;

#[path = "../rsc/models/burgers_boundary_layer.rs"]
mod burgers_boundary_layer;
#[path = "../rsc/models/viscous_burgers.rs"]
mod viscous_burgers_model;

use burgers_boundary_layer::{
    grid, outer_velocity, positions, solve, state_at, steady_thickness, steady_velocity, END_TIME,
    REYNOLDS, SAVES,
};

/// Distance from the wall drawn in the profile figure.
const WALL_RANGE: f64 = 0.15;
/// Range of (1 - x) / nu drawn in the collapse figure.
const SCALED_RANGE: f64 = 10.0;
/// One dot every `MARKER_STRIDE` nodes in the profile figure.
const MARKER_STRIDE: usize = 2;
/// Points of each exact curve.
const CURVE_POINTS: usize = 301;
/// Size of both figures, in pixels.
const SIZE: (u32, u32) = (900, 600);

const COLORS: [RGBColor; 3] = [
    RGBColor(0xc0, 0x39, 0x2b),
    RGBColor(0x1f, 0x6f, 0x8b),
    RGBColor(0x2e, 0x7d, 0x32),
];

/// One steady layer: the viscosity, its outer velocity A, and the computed
/// velocity at t = 2.4 as (distance from the wall, u) from the wall out.
struct Layer {
    reynolds: f64,
    nu: f64,
    a: f64,
    points: Vec<(f64, f64)>,
}

fn layers() -> Result<Vec<Layer>, Box<dyn Error>> {
    let xs = positions(&grid()?);
    let mut layers = Vec::with_capacity(REYNOLDS.len());
    for &reynolds in &REYNOLDS {
        let nu = 1.0 / reynolds;
        let result = solve(nu, SAVES)?;
        let velocity = state_at(&result, END_TIME)?;
        let points = xs
            .iter()
            .zip(velocity.iter())
            .rev()
            .map(|(&x, &u)| (1.0 - x, u))
            .collect();
        layers.push(Layer {
            reynolds,
            nu,
            a: outer_velocity(nu),
            points,
        });
    }
    Ok(layers)
}

fn profiles(layers: &[Layer]) -> Result<(), Box<dyn Error>> {
    let file = "burgers_boundary_layer_profiles.svg";
    let root = SVGBackend::new(file, SIZE).into_drawing_area();
    root.fill(&WHITE)?;
    let mut chart = ChartBuilder::on(&root)
        .caption(
            "Steady boundary layers at t = 2.4 (lines: exact, dots: RK4, dashed: exact δ99)",
            ("sans-serif", 20),
        )
        .margin(15)
        .x_label_area_size(40)
        .y_label_area_size(55)
        .build_cartesian_2d(0.0..WALL_RANGE, 0.0..1.05)?;
    chart
        .configure_mesh()
        .x_desc("distance from the wall, 1 − x")
        .y_desc("u")
        .draw()?;

    for (layer, color) in layers.iter().zip(COLORS) {
        let (a, nu) = (layer.a, layer.nu);
        let line = (0..CURVE_POINTS).map(|i| {
            let d = WALL_RANGE * i as f64 / (CURVE_POINTS - 1) as f64;
            (d, steady_velocity(1.0 - d, a, nu))
        });
        chart
            .draw_series(LineSeries::new(line, color.stroke_width(2)))?
            .label(format!(
                "Re = {}, ν = {}, δ99 = {:.4}",
                layer.reynolds,
                nu,
                steady_thickness(a, nu)
            ))
            .legend(move |(x, y)| {
                EmptyElement::at((x, y))
                    + PathElement::new(vec![(0, 0), (20, 0)], color)
                    + Circle::new((10, 0), 3, color.filled())
            });
        chart.draw_series(
            layer
                .points
                .iter()
                .filter(|(d, _)| *d <= WALL_RANGE)
                .step_by(MARKER_STRIDE)
                .map(|&point| Circle::new(point, 3, color.filled())),
        )?;
        let thickness = steady_thickness(a, nu);
        chart.draw_series(DashedLineSeries::new(
            [(thickness, 0.0), (thickness, 1.05)],
            6,
            4,
            color.stroke_width(1),
        ))?;
    }
    chart
        .configure_series_labels()
        .position(SeriesLabelPosition::LowerRight)
        .background_style(WHITE.mix(0.85))
        .border_style(BLACK)
        .draw()?;
    root.present()?;
    println!("burgers_boundary_layer_plot: wrote {file}");
    Ok(())
}

fn collapse(layers: &[Layer]) -> Result<(), Box<dyn Error>> {
    let file = "burgers_boundary_layer_collapse.svg";
    let root = SVGBackend::new(file, SIZE).into_drawing_area();
    root.fill(&WHITE)?;
    let mut chart = ChartBuilder::on(&root)
        .caption(
            "The three layers against (1 − x) / ν (line: tanh of half of it, dots: RK4)",
            ("sans-serif", 20),
        )
        .margin(15)
        .x_label_area_size(40)
        .y_label_area_size(55)
        .build_cartesian_2d(0.0..SCALED_RANGE, 0.0..1.05)?;
    chart
        .configure_mesh()
        .x_desc("(1 − x) / ν")
        .y_desc("u")
        .draw()?;
    let line = (0..CURVE_POINTS).map(|i| {
        let s = SCALED_RANGE * i as f64 / (CURVE_POINTS - 1) as f64;
        (s, (0.5 * s).tanh())
    });
    chart
        .draw_series(LineSeries::new(line, BLACK.stroke_width(2)))?
        .label("tanh((1 − x) / (2ν))")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], BLACK));
    for (layer, color) in layers.iter().zip(COLORS) {
        let nu = layer.nu;
        chart
            .draw_series(
                layer
                    .points
                    .iter()
                    .map(|&(d, u)| (d / nu, u))
                    .filter(|(s, _)| *s <= SCALED_RANGE)
                    .map(|point| Circle::new(point, 3, color.filled())),
            )?
            .label(format!(
                "Re = {}, Δx / ν = {}",
                layer.reynolds,
                ((layer.points[1].0 - layer.points[0].0) / nu * 1e9).round() / 1e9
            ))
            .legend(move |(x, y)| Circle::new((x + 10, y), 3, color.filled()));
    }
    chart
        .configure_series_labels()
        .position(SeriesLabelPosition::LowerRight)
        .background_style(WHITE.mix(0.85))
        .border_style(BLACK)
        .draw()?;
    root.present()?;
    println!("burgers_boundary_layer_plot: wrote {file}");
    Ok(())
}

fn main() -> Result<(), Box<dyn Error>> {
    let layers = layers()?;
    profiles(&layers)?;
    collapse(&layers)?;
    Ok(())
}
