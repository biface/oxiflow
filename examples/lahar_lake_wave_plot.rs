//! # Example: lahar_lake_wave result plots ([#149](https://github.com/biface/oxiflow/issues/149))
//!
//! Replays the run of `examples/lahar_lake_wave.rs` (`REFERENCE_RUN`: dx =
//! 5 m, explicit Euler, dt = 0.1 s, a state every 5 s up to t = 600 s) and
//! draws the bore it raises in the lake against the Rankine–Hugoniot jump
//! conditions. The scenario and the bore measurement come from
//! `rsc/models/lahar_outfall.rs`, shared with the example, whose module
//! documentation describes the physics and the validation; this file draws
//! and does not check.
//!
//! With h0 the depth at rest, a bore with depth h1 behind it runs at
//! c = sqrt(g h1 (h1 + h0) / (2 h0)). Each figure uses this relation one way:
//!
//! - `lahar_lake_wave_state.svg` -- the state at t = 600 s, in two panels on
//!   one horizontal scale. Top, the depth of the lahar along the channel;
//!   bottom, the free surface of the lake above its level at rest,
//!   b + h - h0. Computed values as dots, one node in eight in the channel and
//!   one in four in the lake. The line is the step that Rankine–Hugoniot
//!   predicts from the measured celerity: the depth h1 it gives behind the
//!   measured front, h0 ahead of it.
//! - `lahar_lake_wave_front.svg` -- the position x_f of the front against
//!   time, from the first state where it can be measured (the averaging
//!   window behind the toe holds a lake node). The line runs at the Rankine–Hugoniot celerity of
//!   the measured h1, the dashed line at the long-wave speed sqrt(g h0), both
//!   through the front at t = 500 s.
//!
//! The measured celerity is the least-squares slope of x_f over the saved
//! states from t = 500 s, the window of the example's check, and h1 is
//! averaged over the same states.
//!
//! As in the animation, the last channel node and the first lake node are
//! not drawn: their saved state follows the exchange across the lip
//! (operator splitting), see `examples/lahar_lake_wave_simul.rs`.
//!
//! Run with: `cargo run --release --example lahar_lake_wave_plot` -- writes
//! both files to the working directory.

use std::error::Error;

use nalgebra::DMatrix;
use oxiflow::{mesh::Mesh, solver::orchestrator::MultiDomainSimulationResult};
use plotters::prelude::*;

#[path = "../rsc/models/lahar_flux.rs"]
mod lahar_flux;
#[path = "../rsc/models/lahar_outfall.rs"]
mod lahar_outfall;
#[path = "../rsc/models/lahar_physics.rs"]
mod lahar_physics;
#[path = "../rsc/models/lahar_source.rs"]
mod lahar_source;

use lahar_outfall::{
    bore, channel_grid, channel_id, field, initial_states, lake_grid, lake_id, primitives, run,
    CHANNEL_LENGTH, LAKE_DEPTH, LAKE_LENGTH, REFERENCE_RUN,
};
use lahar_physics::{COL_B, G};

const DX: f64 = REFERENCE_RUN.dx;
const END_TIME: f64 = REFERENCE_RUN.end_time;

/// Start of the window over which the celerity and h1 are measured, as in
/// the example's check.
const BORE_START: f64 = 500.0;
/// One dot every `CHANNEL_STRIDE` nodes in the channel (40 m), every
/// `LAKE_STRIDE` in the lake (20 m).
const CHANNEL_STRIDE: usize = 8;
const LAKE_STRIDE: usize = 4;
/// Length of the Rankine–Hugoniot plateau drawn behind the front.
const STEP_LENGTH: f64 = 500.0;
/// Size of both figures, in pixels.
const SIZE: (u32, u32) = (900, 600);
/// Margins of the plotting area: outer margin and y label area, so that the
/// channel plot is 800 px wide (4 km) and the lake plot 600 px (3 km).
const MARGIN: u32 = 15;
const Y_LABEL_AREA: u32 = 55;

const RED: RGBColor = RGBColor(0xc0, 0x39, 0x2b);
const ORANGE: RGBColor = RGBColor(0xd6, 0x8a, 0x10);
const BLUE: RGBColor = RGBColor(0x1f, 0x6f, 0x8b);

/// Rankine–Hugoniot celerity of a bore with depth `h1` into still water of
/// depth h0.
fn celerity(h1: f64) -> f64 {
    (G * h1 * (h1 + LAKE_DEPTH) / (2.0 * LAKE_DEPTH)).sqrt()
}

/// Depth behind a bore running at `c` into still water of depth h0: the
/// positive root of h1² + h0 h1 - 2 h0 c² / g = 0.
fn depth_behind(c: f64) -> f64 {
    0.5 * (-LAKE_DEPTH + (LAKE_DEPTH * LAKE_DEPTH + 8.0 * LAKE_DEPTH * c * c / G).sqrt())
}

/// Least-squares slope of `y` against `x`.
fn slope(points: &[(f64, f64)]) -> f64 {
    let n = points.len() as f64;
    let (mx, my) = points
        .iter()
        .fold((0.0, 0.0), |(a, b), &(x, y)| (a + x / n, b + y / n));
    let (sxy, sxx) = points.iter().fold((0.0, 0.0), |(a, b), &(x, y)| {
        (a + (x - mx) * (y - my), b + (x - mx) * (x - mx))
    });
    sxy / sxx
}

/// Position and value of the nodes `nodes` of one domain state.
fn sample(
    state: &DMatrix<f64>,
    xs: &[f64],
    nodes: impl Iterator<Item = usize>,
    value: impl Fn(&DMatrix<f64>, usize) -> Result<f64, Box<dyn Error>>,
) -> Result<Vec<(f64, f64)>, Box<dyn Error>> {
    nodes.map(|i| Ok((xs[i], value(state, i)?))).collect()
}

/// The bore in one saved state: time, front position and depth behind it.
struct FrontPoint {
    t: f64,
    x_f: f64,
    h1: f64,
}

/// Front of the bore at each saved state from the first where it can be
/// measured.
fn front_track(
    result: &MultiDomainSimulationResult,
    lake_xs: &[f64],
) -> Result<Vec<FrontPoint>, Box<dyn Error>> {
    let mut track = Vec::new();
    for (t, state) in result.times.iter().zip(&result.states) {
        match bore(field(state, &lake_id())?, lake_xs) {
            Ok((x_f, h1, _)) => track.push(FrontPoint { t: *t, x_f, h1 }),
            // No bore yet, or too close to the inlet to be measured.
            Err(_) if track.is_empty() => {}
            Err(e) => return Err(e.into()),
        }
    }
    Ok(track)
}

fn state_figure(
    channel: &[(f64, f64)],
    lake: &[(f64, f64)],
    x_f: f64,
    h1_rh: f64,
    c: f64,
) -> Result<(), Box<dyn Error>> {
    let file = "lahar_lake_wave_state.svg";
    let root = SVGBackend::new(file, SIZE).into_drawing_area();
    root.fill(&WHITE)?;
    let (top, bottom) = root.split_vertically(SIZE.1 / 2);

    let depth_max = (channel.iter().fold(0.0_f64, |a, p| a.max(p.1)) * 1.15).ceil();
    let mut chart = ChartBuilder::on(&top)
        .caption(
            format!("Channel at t = {END_TIME} s: depth of the lahar"),
            ("sans-serif", 18),
        )
        .margin(MARGIN)
        .x_label_area_size(35)
        .y_label_area_size(Y_LABEL_AREA)
        .build_cartesian_2d(0.0..CHANNEL_LENGTH / 1000.0, 0.0..depth_max)?;
    chart
        .configure_mesh()
        .x_desc("x (km)")
        .y_desc("h (m)")
        .draw()?;
    chart.draw_series(
        channel
            .iter()
            .map(|&(x, h)| Circle::new((x / 1000.0, h), 2, BLUE.filled())),
    )?;

    // Same horizontal scale as the channel: the lake plot is 3/4 as wide.
    let lake_px = (SIZE.0 - 2 * MARGIN - Y_LABEL_AREA) as f64 * LAKE_LENGTH / CHANNEL_LENGTH;
    let right_margin = SIZE.0 - MARGIN - Y_LABEL_AREA - lake_px.round() as u32;
    let x0 = CHANNEL_LENGTH / 1000.0;
    let x1 = (CHANNEL_LENGTH + LAKE_LENGTH) / 1000.0;
    let mut chart = ChartBuilder::on(&bottom)
        .caption(
            format!("Lake at t = {END_TIME} s: free surface above the level at rest"),
            ("sans-serif", 18),
        )
        .margin(MARGIN)
        .margin_right(right_margin)
        .x_label_area_size(35)
        .y_label_area_size(Y_LABEL_AREA)
        .build_cartesian_2d(x0..x1, -0.5..2.0)?;
    chart
        .configure_mesh()
        .x_desc("x (km)")
        .y_desc("b + h − h0 (m)")
        .draw()?;
    let jump = h1_rh - LAKE_DEPTH;
    let x_front = x_f / 1000.0;
    chart
        .draw_series(LineSeries::new(
            [
                (x_front - STEP_LENGTH / 1000.0, jump),
                (x_front, jump),
                (x_front, 0.0),
                (x1, 0.0),
            ],
            RED.stroke_width(2),
        ))?
        .label(format!("Rankine–Hugoniot, c = {c:.2} m/s"))
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], RED.stroke_width(2)));
    chart
        .draw_series(
            lake.iter()
                .map(|&(x, s)| Circle::new((x / 1000.0, s), 2, BLUE.filled())),
        )?
        .label("explicit Euler")
        .legend(|(x, y)| Circle::new((x + 10, y), 2, BLUE.filled()));
    chart
        .configure_series_labels()
        .position(SeriesLabelPosition::MiddleMiddle)
        .background_style(WHITE.mix(0.85))
        .border_style(BLACK)
        .draw()?;
    root.present()?;
    println!("lahar_lake_wave_plot: wrote {file}");
    Ok(())
}

fn front_figure(
    track: &[FrontPoint],
    anchor: &FrontPoint,
    c_rh: f64,
) -> Result<(), Box<dyn Error>> {
    let file = "lahar_lake_wave_front.svg";
    let root = SVGBackend::new(file, SIZE).into_drawing_area();
    root.fill(&WHITE)?;
    let t0 = (track[0].t / 50.0).floor() * 50.0;
    let x0 = CHANNEL_LENGTH / 1000.0;
    let x1 = (CHANNEL_LENGTH + LAKE_LENGTH) / 1000.0;
    let mut chart = ChartBuilder::on(&root)
        .caption(
            "Position of the front in the lake (lines: through x_f at 500 s, dots: computed)",
            ("sans-serif", 20),
        )
        .margin(MARGIN)
        .x_label_area_size(40)
        .y_label_area_size(Y_LABEL_AREA)
        .build_cartesian_2d(t0..END_TIME, x0..x1)?;
    chart
        .configure_mesh()
        .x_desc("t (s)")
        .y_desc("x_f (km)")
        .x_label_formatter(&|t| format!("{t:.0}"))
        .draw()?;
    let line =
        |speed: f64| [t0, END_TIME].map(|t| (t, (anchor.x_f + speed * (t - anchor.t)) / 1000.0));
    chart
        .draw_series(LineSeries::new(line(c_rh), RED.stroke_width(2)))?
        .label(format!(
            "Rankine–Hugoniot, c = √(g·h1·(h1 + h0) / (2·h0)) = {c_rh:.2} m/s"
        ))
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], RED.stroke_width(2)));
    let long_wave = (G * LAKE_DEPTH).sqrt();
    chart
        .draw_series(DashedLineSeries::new(
            line(long_wave),
            6,
            4,
            ORANGE.stroke_width(2),
        ))?
        .label(format!("long wave, √(g·h0) = {long_wave:.2} m/s"))
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], ORANGE.stroke_width(2)));
    chart
        .draw_series(
            track
                .iter()
                .map(|p| Circle::new((p.t, p.x_f / 1000.0), 3, BLUE.filled())),
        )?
        .label("explicit Euler")
        .legend(|(x, y)| Circle::new((x + 10, y), 3, BLUE.filled()));
    chart
        .configure_series_labels()
        .position(SeriesLabelPosition::UpperLeft)
        .background_style(WHITE.mix(0.85))
        .border_style(BLACK)
        .draw()?;
    root.present()?;
    println!("lahar_lake_wave_plot: wrote {file}");
    Ok(())
}

fn main() -> Result<(), Box<dyn Error>> {
    let (channel0, lake0) = initial_states(DX)?;
    let result = run(&REFERENCE_RUN, &channel0, &lake0)?;

    let positions = |grid: &dyn Mesh| -> Vec<f64> {
        (0..grid.n_dof()).map(|i| grid.coordinates(i)[0]).collect()
    };
    let channel_xs = positions(&channel_grid(DX)?);
    let lake_xs = positions(&lake_grid(DX)?);

    // Front track, celerity and h1 over the window of the example's check.
    let track = front_track(&result, &lake_xs)?;
    let window: Vec<&FrontPoint> = track
        .iter()
        .filter(|p| p.t >= BORE_START - 0.5 * REFERENCE_RUN.dt)
        .collect();
    let anchor = *window.first().ok_or("no bore at 500 s")?;
    let c = slope(&window.iter().map(|p| (p.t, p.x_f)).collect::<Vec<_>>());
    let h1 = window.iter().map(|p| p.h1).sum::<f64>() / window.len() as f64;
    let (h1_rh, c_rh) = (depth_behind(c), celerity(h1));
    println!(
        "  front measured from t = {} s; from t = {BORE_START} s: celerity {c:.3} m/s, \
         h1 {h1:.3} m",
        track[0].t
    );
    println!(
        "  Rankine–Hugoniot: h1 from the celerity {h1_rh:.3} m, celerity from h1 {c_rh:.3} m/s"
    );

    // State at the end of the run, without the two nodes on either side of
    // the lip.
    let last = result.states.last().ok_or("no saved state")?;
    let channel_state = field(last, &channel_id())?;
    let lake_state = field(last, &lake_id())?;
    let channel = sample(
        channel_state,
        &channel_xs,
        (0..channel_xs.len() - 1).step_by(CHANNEL_STRIDE),
        |s, i| Ok(primitives(s, i)?.0),
    )?;
    let lake = sample(
        lake_state,
        &lake_xs,
        (LAKE_STRIDE..lake_xs.len()).step_by(LAKE_STRIDE),
        |s, i| Ok(s[(i, COL_B)] + primitives(s, i)?.0 - LAKE_DEPTH),
    )?;
    let x_f = track.last().ok_or("no bore measured")?.x_f;

    state_figure(&channel, &lake, x_f, h1_rh, c)?;
    front_figure(&track, anchor, c_rh)?;
    Ok(())
}
