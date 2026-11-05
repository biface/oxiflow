//! # Example: magnetic_diffusion_1d result plots ([#149](https://github.com/biface/oxiflow/issues/149))
//!
//! Replays the runs of `examples/magnetic_diffusion_1d.rs`, in copper and in
//! aluminium, and draws them against the exact half-space solution. The
//! scenario comes from `rsc/models/magnetic_diffusion_1d.rs`, shared with the
//! example, whose module documentation describes the physics and the
//! validation; this file draws and does not check.
//!
//! - `magnetic_diffusion_1d_profiles.svg` -- the field B(x) at t = 0.25, 0.5
//!   and 1 ms, one panel per metal, B0 erfc as lines and BDF2 as dots.
//! - `magnetic_diffusion_1d_depth.svg` -- the depth delta_half at which the
//!   field reaches B0/2, measured on the computed field, against
//!   sqrt(eta t): both metals fall on the line of slope 2 erfc^-1(1/2).
//! - `magnetic_diffusion_1d_current.svg` -- the induced current density
//!   J = -(1/mu0) dB/dx at the same times, one panel per metal: the exact
//!   Gaussian B0 / (mu0 sqrt(pi eta t)) exp(-x² / (4 eta t)) as lines, and
//!   as dots the centred difference of the computed field at the interior
//!   nodes.
//!
//! The depths are taken from runs of the same settings saved at every step;
//! the reference runs save one state every 0.25 ms.
//!
//! Run with: `cargo run --release --example magnetic_diffusion_1d_plot` --
//! writes the three files to the working directory.

use std::error::Error;
use std::f64::consts::PI;

use plotters::coord::Shift;
use plotters::prelude::*;

#[path = "../rsc/models/magnetic_diffusion_1d.rs"]
mod magnetic_diffusion_1d;

use magnetic_diffusion_1d::{
    diffusivity, half_field_depth, half_field_factor, half_space_field, plate, positions, solve,
    state_at, CONDUCTORS, MU0, SAVE_EVERY, SURFACE_FIELD,
};

/// Times drawn in the profile panels, s.
const TIMES: [f64; 3] = [2.5e-4, 5e-4, 1e-3];
/// Depth range of the profile panels, m.
const DEPTH_RANGE: f64 = 0.015;
/// One dot every `MARKER_STRIDE` nodes in the profile panels.
const MARKER_STRIDE: usize = 2;
/// One depth measured every `DEPTH_STRIDE` steps.
const DEPTH_STRIDE: usize = 5;
/// Points of each exact curve.
const CURVE_POINTS: usize = 241;
/// Size of every figure, in pixels.
const SIZE: (u32, u32) = (900, 600);

const COLORS: [RGBColor; 3] = [
    RGBColor(0xc0, 0x39, 0x2b),
    RGBColor(0x1f, 0x6f, 0x8b),
    RGBColor(0x2e, 0x7d, 0x32),
];

/// x in mm.
fn mm(x: f64) -> f64 {
    1e3 * x
}

/// Exact current density at depth `x` and time `t`, A/m².
fn half_space_current(eta: f64, x: f64, t: f64) -> f64 {
    SURFACE_FIELD / (MU0 * (PI * eta * t).sqrt()) * (-x * x / (4.0 * eta * t)).exp()
}

/// One panel per metal: for each time, `exact(eta, x, t)` as a line and
/// the points of `computed(field, xs)` as dots, both scaled by `scale`.
#[allow(clippy::too_many_arguments)]
fn panels(
    file: &str,
    caption: &str,
    y_desc: &str,
    y_max: f64,
    scale: f64,
    xs: &[f64],
    exact: impl Fn(f64, f64, f64) -> f64,
    computed: impl Fn(&[f64], &[f64]) -> Vec<(f64, f64)>,
) -> Result<(), Box<dyn Error>> {
    let root = SVGBackend::new(file, SIZE).into_drawing_area();
    root.fill(&WHITE)?;
    let root = root.titled(caption, ("sans-serif", 20))?;
    let areas: Vec<DrawingArea<SVGBackend, Shift>> = root.split_evenly((1, CONDUCTORS.len()));
    for ((name, sigma), area) in CONDUCTORS.iter().zip(&areas) {
        let eta = diffusivity(*sigma);
        let result = solve(eta, SAVE_EVERY)?;
        let mut chart = ChartBuilder::on(area)
            .caption(*name, ("sans-serif", 17))
            .margin(12)
            .x_label_area_size(40)
            .y_label_area_size(55)
            .build_cartesian_2d(0.0..mm(DEPTH_RANGE), 0.0..y_max)?;
        chart
            .configure_mesh()
            .x_desc("x (mm)")
            .y_desc(y_desc)
            .draw()?;
        for (time, color) in TIMES.iter().zip(COLORS) {
            let line = (0..CURVE_POINTS).map(|i| {
                let x = DEPTH_RANGE * i as f64 / (CURVE_POINTS - 1) as f64;
                (mm(x), scale * exact(eta, x, *time))
            });
            chart
                .draw_series(LineSeries::new(line, color.stroke_width(2)))?
                .label(format!("t = {} ms", time * 1e3))
                .legend(move |(x, y)| {
                    EmptyElement::at((x, y))
                        + PathElement::new(vec![(0, 0), (20, 0)], color)
                        + Circle::new((10, 0), 3, color.filled())
                });
            let field: Vec<f64> = state_at(&result, *time, name)?.iter().copied().collect();
            chart.draw_series(
                computed(&field, xs)
                    .into_iter()
                    .filter(|(x, _)| *x <= DEPTH_RANGE)
                    .step_by(MARKER_STRIDE)
                    .map(|(x, y)| Circle::new((mm(x), scale * y), 3, color.filled())),
            )?;
        }
        chart
            .configure_series_labels()
            .position(SeriesLabelPosition::UpperRight)
            .background_style(WHITE.mix(0.85))
            .border_style(BLACK)
            .draw()?;
    }
    root.present()?;
    println!("magnetic_diffusion_1d_plot: wrote {file}");
    Ok(())
}

/// delta_half against sqrt(eta t), both metals, with the line of slope
/// 2 erfc^-1(1/2).
fn depth(xs: &[f64]) -> Result<(), Box<dyn Error>> {
    let factor = half_field_factor();
    let mut points = Vec::new();
    let mut reach: f64 = 0.0;
    for (name, sigma) in CONDUCTORS {
        let eta = diffusivity(sigma);
        let result = solve(eta, 1)?;
        let mut conductor = Vec::new();
        for (t, state) in result
            .times
            .iter()
            .zip(&result.states)
            .skip(DEPTH_STRIDE)
            .step_by(DEPTH_STRIDE)
        {
            let depth = half_field_depth(state.as_scalar_field()?, xs, name)?;
            let scale = (eta * t).sqrt();
            reach = reach.max(scale);
            conductor.push((mm(scale), mm(depth)));
        }
        points.push((name, conductor));
    }

    let file = "magnetic_diffusion_1d_depth.svg";
    let root = SVGBackend::new(file, SIZE).into_drawing_area();
    root.fill(&WHITE)?;
    let x_max = mm(reach) * 1.05;
    let mut chart = ChartBuilder::on(&root)
        .caption(
            "Depth δ½ where B = B₀/2 against √(η·t), measured every 5 steps",
            ("sans-serif", 20),
        )
        .margin(15)
        .x_label_area_size(40)
        .y_label_area_size(55)
        .build_cartesian_2d(0.0..x_max, 0.0..x_max * factor * 1.05)?;
    chart
        .configure_mesh()
        .x_desc("√(η·t) (mm)")
        .y_desc("δ½ (mm)")
        .draw()?;
    chart
        .draw_series(LineSeries::new(
            [(0.0, 0.0), (x_max, factor * x_max)],
            BLACK.stroke_width(2),
        ))?
        .label(format!("slope 2·erfc⁻¹(1/2) = {factor:.4}"))
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], BLACK));
    for ((name, conductor), color) in points.into_iter().zip(COLORS) {
        chart
            .draw_series(
                conductor
                    .into_iter()
                    .map(|point| Circle::new(point, 4, color.filled())),
            )?
            .label(name)
            .legend(move |(x, y)| Circle::new((x + 10, y), 4, color.filled()));
    }
    chart
        .configure_series_labels()
        .position(SeriesLabelPosition::UpperLeft)
        .background_style(WHITE.mix(0.85))
        .border_style(BLACK)
        .draw()?;
    root.present()?;
    println!("magnetic_diffusion_1d_plot: wrote {file}");
    Ok(())
}

fn main() -> Result<(), Box<dyn Error>> {
    let xs = positions(&plate()?);
    let dx = xs[1] - xs[0];

    panels(
        "magnetic_diffusion_1d_profiles.svg",
        "Magnetic field in the plate (lines: B₀·erfc, dots: BDF2)",
        "B (mT)",
        mm(SURFACE_FIELD) * 1.05,
        1e3,
        &xs,
        half_space_field,
        |field, xs| xs.iter().copied().zip(field.iter().copied()).collect(),
    )?;

    depth(&xs)?;

    let peak = half_space_current(diffusivity(CONDUCTORS[0].1), 0.0, TIMES[0]);
    panels(
        "magnetic_diffusion_1d_current.svg",
        "Induced current density (lines: exact, dots: centred difference of BDF2)",
        "J (A/mm²)",
        1e-6 * peak * 1.05,
        1e-6,
        &xs,
        half_space_current,
        |field, xs| {
            (1..field.len() - 1)
                .map(|i| (xs[i], -(field[i + 1] - field[i - 1]) / (2.0 * dx * MU0)))
                .collect()
        },
    )?;
    Ok(())
}
