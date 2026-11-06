//! # Example: heat_transfer_1d result plots ([#149](https://github.com/biface/oxiflow/issues/149))
//!
//! Replays the two runs of `examples/heat_transfer_1d.rs` and draws them
//! against their exact solutions. The scenario comes from
//! `rsc/models/heat_transfer_1d.rs`, shared with the example, whose module
//! documentation describes the physics and the validation; this file draws
//! and does not check.
//!
//! - `heat_transfer_1d_cooling.svg` -- case (a), cooling of a homogeneous
//!   steel bar from 100 °C with both ends held at 20 °C: T(x) at
//!   t = 10, 30, 60 and 120 s, the Fourier series as lines and BDF2 as
//!   dots (every fifth node), each curve labelled with its time.
//! - `heat_transfer_1d_graded.svg` -- case (b), graded bar whose
//!   conductivity doubles along its length: the approach to steady state
//!   by Backward Euler at t = 50, 100, 150 and 200 s, the state at
//!   t = 3000 s as markers on the exact steady profile, and, dashed, the
//!   linear steady profile of a homogeneous bar between the same end
//!   temperatures.
//!
//! The transient of case (b) is taken from a run of the same settings
//! stopped at 200 s and saved at every step: the reference run saves one
//! state every 500 s, and by then the bar is already within 5e-2 °C of the
//! steady profile.
//!
//! Run with: `cargo run --release --example heat_transfer_1d_plot` --
//! writes both files to the working directory.

use std::error::Error;

use oxiflow::solver::{
    config::IntegratorKind,
    methods::{BDF2Solver, BackwardEulerSolver},
};
use plotters::prelude::*;
use plotters::style::text_anchor::{HPos, Pos, VPos};

#[path = "../rsc/models/graded_conductivity.rs"]
mod graded_conductivity;
#[path = "../rsc/models/heat_transfer_1d.rs"]
mod heat_transfer_1d;

use heat_transfer_1d::{
    bar, cooling_run, fourier_cooling, graded_run, graded_steady, positions, solve, state_at, Run,
    COOLING_INITIAL, GRADED_COLD, GRADED_END_TIME, GRADED_HOT, LENGTH, NEWTON_ITERATIONS,
};

/// Times of case (a) drawn, in s.
const COOLING_TIMES: [f64; 4] = [10.0, 30.0, 60.0, 120.0];
/// Times of the approach to steady state in case (b), in s.
const GRADED_TIMES: [f64; 4] = [50.0, 100.0, 150.0, 200.0];
/// One marker every `MARKER_STRIDE` nodes.
const MARKER_STRIDE: usize = 5;
/// Points of each exact curve.
const CURVE_POINTS: usize = 201;
/// Size of both figures, in pixels.
const SIZE: (u32, u32) = (900, 600);

const COLORS: [RGBColor; 4] = [
    RGBColor(0xc0, 0x39, 0x2b),
    RGBColor(0xd6, 0x8a, 0x10),
    RGBColor(0x1f, 0x6f, 0x8b),
    RGBColor(0x2e, 0x7d, 0x32),
];
const GREY: RGBColor = RGBColor(0x90, 0x90, 0x90);

/// x in mm.
fn mm(x: f64) -> f64 {
    1e3 * x
}

/// Label `t = time s` centred on the middle of the bar at height `y`.
fn time_label(time: f64, y: f64, color: RGBColor) -> Text<'static, (f64, f64), String> {
    Text::new(
        format!("t = {time} s"),
        (mm(0.5 * LENGTH), y),
        ("sans-serif", 15)
            .into_font()
            .color(&color)
            .pos(Pos::new(HPos::Center, VPos::Center)),
    )
}

/// `CURVE_POINTS` points of `f` along the bar, x in mm.
fn curve(f: impl Fn(f64) -> f64) -> Vec<(f64, f64)> {
    (0..CURVE_POINTS)
        .map(|i| {
            let x = LENGTH * i as f64 / (CURVE_POINTS - 1) as f64;
            (mm(x), f(x))
        })
        .collect()
}

/// Case (a): Fourier series as lines, BDF2 as dots, one label per time.
fn cooling(xs: &[f64]) -> Result<(), Box<dyn Error>> {
    let bdf2 = BDF2Solver::new().with_max_newton_iterations(NEWTON_ITERATIONS);
    let bdf2_result = solve(&cooling_run(IntegratorKind::BDF2), &bdf2)?;

    let file = "heat_transfer_1d_cooling.svg";
    let root = SVGBackend::new(file, SIZE).into_drawing_area();
    root.fill(&WHITE)?;
    let mut chart = ChartBuilder::on(&root)
        .caption(
            "Steel bar cooling from 100 °C, ends at 20 °C (lines: Fourier series, dots: BDF2)",
            ("sans-serif", 22),
        )
        .margin(15)
        .x_label_area_size(40)
        .y_label_area_size(55)
        .build_cartesian_2d(0.0..mm(LENGTH), 0.0..COOLING_INITIAL * 1.08)?;
    chart
        .configure_mesh()
        .x_desc("x (mm)")
        .y_desc("T (°C)")
        .draw()?;

    let initial = state_at(&bdf2_result, 0.0, "(a) BDF2")?;
    chart.draw_series(LineSeries::new(
        xs.iter().zip(initial.iter()).map(|(&x, &t)| (mm(x), t)),
        GREY.stroke_width(1),
    ))?;
    chart.draw_series(std::iter::once(time_label(
        0.0,
        COOLING_INITIAL + 2.5,
        GREY,
    )))?;

    for (time, color) in COOLING_TIMES.iter().zip(COLORS) {
        chart.draw_series(LineSeries::new(
            curve(|x| fourier_cooling(x, *time)),
            color.stroke_width(2),
        ))?;
        let state = state_at(&bdf2_result, *time, "(a) BDF2")?;
        chart.draw_series(
            xs.iter()
                .zip(state.iter())
                .step_by(MARKER_STRIDE)
                .map(|(&x, &t)| Circle::new((mm(x), t), 3, color.filled())),
        )?;
        let peak = fourier_cooling(0.5 * LENGTH, *time);
        chart.draw_series(std::iter::once(time_label(*time, peak - 4.5, color)))?;
    }
    root.present()?;
    println!("heat_transfer_1d_plot: wrote {file}");
    Ok(())
}

/// Case (b): approach to steady state, final state on the exact profile,
/// linear profile of a homogeneous bar dashed.
fn graded(xs: &[f64]) -> Result<(), Box<dyn Error>> {
    let solver = BackwardEulerSolver::new().with_max_newton_iterations(NEWTON_ITERATIONS);
    let reference = solve(&graded_run(), &solver)?;
    let last_time = GRADED_TIMES[GRADED_TIMES.len() - 1];
    let transient = solve(
        &Run {
            end_time: last_time,
            save_every: 1,
            ..graded_run()
        },
        &solver,
    )?;

    let file = "heat_transfer_1d_graded.svg";
    let root = SVGBackend::new(file, SIZE).into_drawing_area();
    root.fill(&WHITE)?;
    let mut chart = ChartBuilder::on(&root)
        .caption(
            "Graded steel bar, ends at 100 °C and 20 °C",
            ("sans-serif", 22),
        )
        .margin(15)
        .x_label_area_size(40)
        .y_label_area_size(55)
        .build_cartesian_2d(0.0..mm(LENGTH), 0.0..GRADED_HOT * 1.08)?;
    chart
        .configure_mesh()
        .x_desc("x (mm)")
        .y_desc("T (°C)")
        .draw()?;

    for (time, color) in GRADED_TIMES.iter().zip(COLORS) {
        let state = state_at(&transient, *time, "(b) Backward Euler")?;
        chart
            .draw_series(LineSeries::new(
                xs.iter().zip(state.iter()).map(|(&x, &t)| (mm(x), t)),
                color.stroke_width(1),
            ))?
            .label(format!("t = {time} s, Backward Euler"))
            .legend(move |(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], color));
    }

    chart
        .draw_series(LineSeries::new(curve(graded_steady), BLACK.stroke_width(2)))?
        .label("exact steady profile")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], BLACK));
    let last = state_at(&reference, GRADED_END_TIME, "(b) Backward Euler")?;
    chart
        .draw_series(
            xs.iter()
                .zip(last.iter())
                .step_by(MARKER_STRIDE)
                .map(|(&x, &t)| Circle::new((mm(x), t), 3, BLACK.filled())),
        )?
        .label(format!("t = {GRADED_END_TIME} s, Backward Euler"))
        .legend(|(x, y)| Circle::new((x + 10, y), 3, BLACK.filled()));
    chart
        .draw_series(DashedLineSeries::new(
            curve(|x| GRADED_HOT - (GRADED_HOT - GRADED_COLD) * x / LENGTH),
            6,
            4,
            GREY.stroke_width(2),
        ))?
        .label("steady profile of a homogeneous bar")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], GREY));

    chart
        .configure_series_labels()
        .position(SeriesLabelPosition::UpperRight)
        .background_style(WHITE.mix(0.85))
        .border_style(BLACK)
        .draw()?;
    root.present()?;
    println!("heat_transfer_1d_plot: wrote {file}");
    Ok(())
}

fn main() -> Result<(), Box<dyn Error>> {
    let xs = positions(&bar()?);
    cooling(&xs)?;
    graded(&xs)?;
    Ok(())
}
