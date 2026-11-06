//! # Animation: a viscous shock running into a wall ([#149](https://github.com/biface/oxiflow/issues/149))
//!
//! Replays the runs of `examples/burgers_boundary_layer.rs` as an animated
//! SVG, `burgers_boundary_layer_simul.svg`, written to the working
//! directory: the viscous shock moving towards the wall at x = 1 at speed
//! 1/2, reaching it at t = 0.8, then the boundary layer relaxing to its
//! steady state, at Re = 50, 100 and 200 on the same axes. The scenario and
//! the runs are those of `rsc/models/burgers_boundary_layer.rs`, with the
//! time steps of the validated example, saved here 100 times up to t = 2.4;
//! see that example for the physics and the checks.
//!
//! The animation uses SVG's own animation elements (`<animate>`), with no
//! script, so it plays where the SVG is shown as an image, in a browser or
//! on the wiki. Each saved state is one frame, and every other node is
//! drawn. A viewer that does not play SVG animations shows the first frame,
//! the three shocks at x0 = 0.6. The drawing pieces come from
//! `rsc/render/svg_animation.rs`, shared by the `_simul` examples.
//!
//! Run with: `cargo run --release --example burgers_boundary_layer_simul`.

use std::error::Error;

// The animation uses the runs, not the steady solution.
#[allow(dead_code)]
#[path = "../rsc/models/burgers_boundary_layer.rs"]
mod burgers_boundary_layer;
#[path = "../rsc/render/svg_animation.rs"]
mod svg_animation;
#[path = "../rsc/models/viscous_burgers.rs"]
mod viscous_burgers_model;

use burgers_boundary_layer::{grid, positions, solve, END_TIME, REYNOLDS};
use svg_animation::{animated_curve, begin, end, legend, ticks, time_bar, Panel};

// Drawing.
const WIDTH: f64 = 900.0;
const HEIGHT: f64 = 520.0;
const LEFT: f64 = 70.0;
const RIGHT: f64 = 870.0;
/// Seconds of animation per frame.
const FRAME_SECONDS: f64 = 0.1;
/// Saved states per run after the initial one; divides the number of steps
/// at every Reynolds number.
const SAVES: usize = 100;
/// Every `STRIDE`-th node is drawn.
const STRIDE: usize = 2;
/// Colour of each run, in the order of `REYNOLDS`.
const COLORS: [&str; 3] = ["#c0392b", "#1f6f8b", "#2e7d32"];

fn main() -> Result<(), Box<dyn Error>> {
    let xs: Vec<f64> = positions(&grid()?).into_iter().step_by(STRIDE).collect();

    let panel = Panel {
        x0: 0.0,
        x1: 1.0,
        y0: 0.0,
        y1: 1.05,
        left: LEFT,
        right: RIGHT,
        top: 50.0,
        bottom: 400.0,
        x_unit: 1.0,
        x_label: "x (wall at x = 1)",
    };

    let mut svg = begin(WIDTH, HEIGHT);
    panel.axes(
        &mut svg,
        "Velocity u(x): a viscous shock running into a wall where u = 0",
        &ticks(0.0, 1.0, 0.1),
        &ticks(0.0, 1.0, 0.2),
        "u",
    );

    let mut times = Vec::new();
    let mut frames_len = 0;
    for (&reynolds, color) in REYNOLDS.iter().zip(COLORS) {
        let result = solve(1.0 / reynolds, SAVES)?;
        let mut frames = Vec::with_capacity(result.states.len());
        for state in &result.states {
            let u: Vec<f64> = state
                .as_scalar_field()?
                .iter()
                .copied()
                .step_by(STRIDE)
                .collect();
            frames.push(panel.open_points(&xs, &u));
        }
        frames_len = frames.len();
        animated_curve(
            &mut svg,
            &frames,
            frames_len as f64 * FRAME_SECONDS,
            color,
            false,
        );
        times = result.times;
    }
    let duration = frames_len as f64 * FRAME_SECONDS;

    let labels: Vec<String> = REYNOLDS
        .iter()
        .map(|re| format!("Re = {re}, ν = {}", 1.0 / re))
        .collect();
    let entries: Vec<(&str, &str)> = COLORS
        .iter()
        .zip(&labels)
        .map(|(color, label)| (*color, label.as_str()))
        .collect();
    legend(&mut svg, LEFT + 20.0, 250.0, &entries);

    time_bar(
        &mut svg,
        LEFT,
        RIGHT,
        460.0,
        &times,
        END_TIME,
        duration,
        &format!(
            "time, 0 to {END_TIME} (one frame every {}); the shock reaches the wall at t = 0.8",
            END_TIME / SAVES as f64
        ),
        0.4,
        1.0,
    );
    end(&mut svg);

    std::fs::write("burgers_boundary_layer_simul.svg", &svg)?;
    println!(
        "burgers_boundary_layer_simul.svg: {frames_len} frames, {duration:.0} s of animation, {} kB",
        svg.len() / 1024
    );
    Ok(())
}
