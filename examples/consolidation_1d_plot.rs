//! # Example: consolidation_1d result plots ([#149](https://github.com/biface/oxiflow/issues/149))
//!
//! Replays the run of `examples/consolidation_1d.rs` and draws it against
//! Terzaghi's series solution. The scenario comes from
//! `rsc/models/consolidation_1d.rs`, shared with the example, whose module
//! documentation describes the physics and the validation; this file draws
//! and does not check.
//!
//! - `consolidation_1d_isochrones.svg` -- the isochrones: excess pore
//!   pressure u(z) across the whole 4 m layer at Tv = 0.05, 0.1, 0.2, 0.5
//!   and 1, the series as lines and BDF2 as dots (every fifth node), depth
//!   increasing downwards as in soil mechanics. The dashed mid-plane is
//!   crossed by no water: it is the impermeable base of the 2 m reading.
//! - `consolidation_1d_degree.svg` -- the average degree of consolidation
//!   U(Tv), the series as a line and the computed values as dots, Tv on a
//!   logarithmic axis and U increasing downwards, the usual Terzaghi curve
//!   of soil-mechanics textbooks.
//!
//! The degree of consolidation is taken from a run of the same settings
//! saved at every step (Tv every 0.005); the reference run saves one state
//! every 0.05.
//!
//! Run with: `cargo run --release --example consolidation_1d_plot` --
//! writes both files to the working directory.

use std::error::Error;

use plotters::prelude::*;
use plotters::style::text_anchor::{HPos, Pos, VPos};

#[path = "../rsc/models/consolidation_1d.rs"]
mod consolidation_1d;

use consolidation_1d::{
    degree_of_consolidation, layer, positions, series_consolidation, series_pressure, solve,
    state_at, DRAINAGE_PATH, LOAD, SAVE_EVERY, THICKNESS,
};

/// Time factors of the isochrones drawn.
const ISOCHRONES: [f64; 5] = [0.05, 0.1, 0.2, 0.5, 1.0];
/// Depth at which each isochrone is labelled, m.
const LABEL_DEPTH: f64 = 1.0;
/// Time step in time factor, from the reference settings (0.005).
const TV_STEP: f64 = 0.005;
/// Multiples of `TV_STEP` drawn as dots on the U(Tv) curve, roughly evenly
/// spaced on the logarithmic axis.
const DEGREE_STEPS: [usize; 21] = [
    1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20, 25, 30, 40, 50, 60, 80, 100, 120, 150, 200,
];
/// One marker every `MARKER_STRIDE` nodes.
const MARKER_STRIDE: usize = 5;
/// Points of each series curve.
const CURVE_POINTS: usize = 201;
/// Size of both figures, in pixels.
const SIZE: (u32, u32) = (900, 600);

const COLORS: [RGBColor; 5] = [
    RGBColor(0xc0, 0x39, 0x2b),
    RGBColor(0xd6, 0x8a, 0x10),
    RGBColor(0x1f, 0x6f, 0x8b),
    RGBColor(0x2e, 0x7d, 0x32),
    RGBColor(0x6a, 0x3d, 0x9a),
];
const GREY: RGBColor = RGBColor(0x90, 0x90, 0x90);

/// Text left-aligned at `(x, y)` in data coordinates.
fn label(text: String, x: f64, y: f64, color: RGBColor) -> Text<'static, (f64, f64), String> {
    Text::new(
        text,
        (x, y),
        ("sans-serif", 15)
            .into_font()
            .color(&color)
            .pos(Pos::new(HPos::Left, VPos::Center)),
    )
}

/// Isochrones. The vertical axis carries -z, so that depth increases
/// downwards; its labels show z.
fn isochrones(zs: &[f64]) -> Result<(), Box<dyn Error>> {
    let result = solve(SAVE_EVERY)?;

    let file = "consolidation_1d_isochrones.svg";
    let root = SVGBackend::new(file, SIZE).into_drawing_area();
    root.fill(&WHITE)?;
    let mut chart = ChartBuilder::on(&root)
        .caption(
            "Excess pore pressure (lines: Terzaghi series, dots: BDF2, dashed: no-flow mid-plane)",
            ("sans-serif", 20),
        )
        .margin(15)
        .x_label_area_size(40)
        .y_label_area_size(55)
        .build_cartesian_2d(0.0..LOAD * 1.15, -THICKNESS..0.0)?;
    chart
        .configure_mesh()
        .x_desc("u (kPa)")
        .y_desc("z (m)")
        .y_label_formatter(&|v| format!("{:.1}", 0.0 - v))
        .draw()?;

    let initial = state_at(&result, 0.0)?;
    chart.draw_series(LineSeries::new(
        zs.iter().zip(initial.iter()).map(|(&z, &u)| (u, -z)),
        GREY.stroke_width(1),
    ))?;
    chart.draw_series(std::iter::once(label(
        "Tv = 0".into(),
        LOAD + 1.5,
        -LABEL_DEPTH,
        GREY,
    )))?;

    chart.draw_series(DashedLineSeries::new(
        [(0.0, -DRAINAGE_PATH), (LOAD * 1.15, -DRAINAGE_PATH)],
        6,
        4,
        GREY.stroke_width(1),
    ))?;

    for (tv, color) in ISOCHRONES.iter().zip(COLORS) {
        let series = (0..CURVE_POINTS).map(|i| {
            let z = THICKNESS * i as f64 / (CURVE_POINTS - 1) as f64;
            (series_pressure(z, *tv), -z)
        });
        chart.draw_series(LineSeries::new(series, color.stroke_width(2)))?;
        let state = state_at(&result, *tv)?;
        chart.draw_series(
            zs.iter()
                .zip(state.iter())
                .step_by(MARKER_STRIDE)
                .map(|(&z, &u)| Circle::new((u, -z), 3, color.filled())),
        )?;
        chart.draw_series(std::iter::once(label(
            format!("Tv = {tv}"),
            series_pressure(LABEL_DEPTH, *tv) + 2.0,
            -LABEL_DEPTH,
            color,
        )))?;
    }
    root.present()?;
    println!("consolidation_1d_plot: wrote {file}");
    Ok(())
}

/// Degree of consolidation. The vertical axis carries -U, so that U
/// increases downwards; its labels show U.
fn degree(zs: &[f64]) -> Result<(), Box<dyn Error>> {
    let result = solve(1)?;
    let first = TV_STEP;
    let last = TV_STEP * DEGREE_STEPS[DEGREE_STEPS.len() - 1] as f64;

    let file = "consolidation_1d_degree.svg";
    let root = SVGBackend::new(file, SIZE).into_drawing_area();
    root.fill(&WHITE)?;
    let mut chart = ChartBuilder::on(&root)
        .caption(
            "Average degree of consolidation (line: Terzaghi series, dots: BDF2)",
            ("sans-serif", 20),
        )
        .margin(15)
        .x_label_area_size(40)
        .y_label_area_size(55)
        .build_cartesian_2d((first..last).log_scale(), -1.0..0.0)?;
    chart
        .configure_mesh()
        .x_desc("Tv")
        .y_desc("U")
        .y_label_formatter(&|v| format!("{:.1}", 0.0 - v))
        .draw()?;

    let ratio = last / first;
    let series = (0..CURVE_POINTS).map(|i| {
        let tv = first * ratio.powf(i as f64 / (CURVE_POINTS - 1) as f64);
        (tv, -series_consolidation(tv))
    });
    chart.draw_series(LineSeries::new(series, BLACK.stroke_width(2)))?;

    let mut dots = Vec::with_capacity(DEGREE_STEPS.len());
    for &step in &DEGREE_STEPS {
        let tv = TV_STEP * step as f64;
        dots.push((tv, -degree_of_consolidation(state_at(&result, tv)?, zs)));
    }
    chart.draw_series(
        dots.into_iter()
            .map(|point| Circle::new(point, 4, COLORS[2].filled())),
    )?;
    root.present()?;
    println!("consolidation_1d_plot: wrote {file}");
    Ok(())
}

fn main() -> Result<(), Box<dyn Error>> {
    let zs = positions(&layer()?);
    isochrones(&zs)?;
    degree(&zs)?;
    Ok(())
}
