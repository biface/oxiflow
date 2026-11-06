//! # Animation: a magnetic field diffusing into copper and aluminium ([#149](https://github.com/biface/oxiflow/issues/149))
//!
//! Replays the runs of `examples/magnetic_diffusion_1d.rs` as an animated
//! SVG, `magnetic_diffusion_1d_simul.svg`, written to the working directory:
//! the field B(x) advancing into the plate from the surface, in copper and
//! in aluminium on the same axes. Aluminium conducts less, so the field
//! enters it faster: its depths are larger by sqrt(sigma_Cu / sigma_Al),
//! about 1.26. The scenario and the runs are those of
//! `rsc/models/magnetic_diffusion_1d.rs`, with the settings of the validated
//! example (BDF2, dt = 1e-5 s, up to t = 1 ms), saved here at every step;
//! see that example for the physics and the checks.
//!
//! The animation uses SVG's own animation elements (`<animate>`), with no
//! script, so it plays where the SVG is shown as an image, in a browser or
//! on the wiki. Each step is one frame. A viewer that does not play SVG
//! animations shows the first frame, the plate without field. The drawing
//! pieces come from `rsc/render/svg_animation.rs`, shared by the `_simul`
//! examples.
//!
//! Run with: `cargo run --release --example magnetic_diffusion_1d_simul`.

use std::error::Error;

// The animation uses the runs, not the exact solution nor the checks.
#[allow(dead_code)]
#[path = "../rsc/models/magnetic_diffusion_1d.rs"]
mod magnetic_diffusion_1d;
#[path = "../rsc/render/svg_animation.rs"]
mod svg_animation;

use magnetic_diffusion_1d::{
    diffusivity, plate, positions, solve, CONDUCTORS, DT, END_TIME, SURFACE_FIELD,
};
use svg_animation::{animated_curve, begin, end, legend, ticks, time_bar, Panel};

// Drawing.
const WIDTH: f64 = 900.0;
const HEIGHT: f64 = 520.0;
const LEFT: f64 = 70.0;
const RIGHT: f64 = 870.0;
/// Seconds of animation per frame.
const FRAME_SECONDS: f64 = 0.1;
/// Depth drawn, m.
const DEPTH_RANGE: f64 = 0.015;
/// Colour of each conductor, in the order of `CONDUCTORS`.
const COLORS: [&str; 2] = ["#c0392b", "#1f6f8b"];

fn main() -> Result<(), Box<dyn Error>> {
    let xs = positions(&plate()?);
    let drawn = xs.iter().take_while(|&&x| x <= DEPTH_RANGE + 1e-12).count();
    let xs = &xs[..drawn];

    let panel = Panel {
        x0: 0.0,
        x1: DEPTH_RANGE,
        y0: 0.0,
        y1: 1e3 * SURFACE_FIELD * 1.05,
        left: LEFT,
        right: RIGHT,
        top: 50.0,
        bottom: 400.0,
        x_unit: 1e-3,
        x_label: "x (mm)",
    };

    let mut svg = begin(WIDTH, HEIGHT);
    panel.axes(
        &mut svg,
        "Magnetic field B(x) in the plate, field applied at the surface from t = 0",
        &ticks(0.0, DEPTH_RANGE, 1e-3),
        &ticks(0.0, 1e3 * SURFACE_FIELD, 10.0),
        "B (mT)",
    );

    let mut times = Vec::new();
    let mut frames_len = 0;
    for ((_, sigma), color) in CONDUCTORS.iter().zip(COLORS) {
        let result = solve(diffusivity(*sigma), 1)?;
        let mut frames = Vec::with_capacity(result.states.len());
        for state in &result.states {
            let millitesla: Vec<f64> = state
                .as_scalar_field()?
                .iter()
                .take(drawn)
                .map(|b| 1e3 * b)
                .collect();
            frames.push(panel.open_points(xs, &millitesla));
        }
        frames_len = frames.len();
        let duration = frames_len as f64 * FRAME_SECONDS;
        animated_curve(&mut svg, &frames, duration, color, false);
        times = result.times;
    }
    let duration = frames_len as f64 * FRAME_SECONDS;

    let labels: Vec<String> = CONDUCTORS
        .iter()
        .map(|(name, sigma)| format!("{name}, η = {:.4} m²/s", diffusivity(*sigma)))
        .collect();
    let entries: Vec<(&str, &str)> = COLORS
        .iter()
        .zip(&labels)
        .map(|(color, label)| (*color, label.as_str()))
        .collect();
    legend(&mut svg, RIGHT - 230.0, 72.0, &entries);

    time_bar(
        &mut svg,
        LEFT,
        RIGHT,
        460.0,
        &times,
        END_TIME,
        duration,
        &format!(
            "time, 0 to {} ms (one frame every {} ms)",
            END_TIME * 1e3,
            DT * 1e3
        ),
        1e-4,
        1e-3,
    );
    end(&mut svg);

    std::fs::write("magnetic_diffusion_1d_simul.svg", &svg)?;
    println!(
        "magnetic_diffusion_1d_simul.svg: {frames_len} frames, {duration:.0} s of animation, {} kB",
        svg.len() / 1024
    );
    Ok(())
}
