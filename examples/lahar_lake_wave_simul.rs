//! # Animation: a lahar front running into a lake ([#180](https://github.com/biface/oxiflow/issues/180))
//!
//! Replays the run of `examples/lahar_lake_wave.rs` as an animated SVG,
//! `lahar_lake_wave_simul.svg`, written to the working directory: the lahar
//! running down the channel, then the front it pushes across the lake. The
//! scenario and the run are those of `rsc/models/lahar_outfall.rs`, with the
//! settings of the validated example (`REFERENCE_RUN`: dx = 5 m, explicit
//! Euler, dt = 0.1 s, a state every 5 s up to t = 600 s); see that example
//! for the physics and the checks.
//!
//! The animation uses SVG's own animation elements (`<animate>`), with no
//! script, so it plays where the SVG is shown as an image, in a browser or
//! on the wiki. Each saved state is one frame. A viewer that does not play
//! SVG animations (an IDE or system preview) shows the first frame, the lake
//! at rest, and a note under the time bar says where to open the file.
//!
//! - Top: depth of the flow along the channel, `h(x)`.
//! - Bottom: free surface of the lake above its level at rest, `b + h - h0`,
//!   with `b` the deposit and `h0` the depth at rest. Over the first 250 m
//!   or so, the lake holds the lahar itself (solids fraction about 0.16 at
//!   t = 600 s): denser than the water ahead, this layer stands a little
//!   below the level at rest while pressing harder on the bed, and the front
//!   ahead of it stands about 1.7 m above that level.
//!
//! The drawing pieces (panels, animated curves, time bar) come from
//! `rsc/render/svg_animation.rs`, shared by the `_simul` examples.
//!
//! Run with: `cargo run --release --example lahar_lake_wave_simul`.

use std::error::Error;

use nalgebra::DMatrix;
use oxiflow::mesh::Mesh;

#[path = "../rsc/models/lahar_flux.rs"]
mod lahar_flux;
#[path = "../rsc/models/lahar_outfall.rs"]
mod lahar_outfall;
#[path = "../rsc/models/lahar_physics.rs"]
mod lahar_physics;
#[path = "../rsc/models/lahar_source.rs"]
mod lahar_source;
#[path = "../rsc/render/svg_animation.rs"]
mod svg_animation;

use lahar_outfall::{
    channel_grid, channel_id, field, initial_states, lake_grid, lake_id, physics, run, LAKE_DEPTH,
    REFERENCE_RUN,
};
use lahar_physics::{COL_B, COL_M, COL_P, COL_S};
use svg_animation::{animated_curve, begin, end, ticks, time_bar, Panel, CURVE};

const DX: f64 = REFERENCE_RUN.dx;
const END_TIME: f64 = REFERENCE_RUN.end_time;

// Drawing.
const WIDTH: f64 = 900.0;
const HEIGHT: f64 = 590.0;
const LEFT: f64 = 70.0;
const RIGHT: f64 = 870.0;
/// Seconds of animation per frame.
const FRAME_SECONDS: f64 = 0.1;
/// Every `STRIDE`-th node is drawn (10 m).
const STRIDE: usize = 2;
/// Depth `h` and bed `b` of every drawn node of one domain state.
fn depth_and_bed(state: &DMatrix<f64>) -> Result<(Vec<f64>, Vec<f64>), Box<dyn Error>> {
    let (mut h, mut b) = (Vec::new(), Vec::new());
    for i in (0..state.nrows()).step_by(STRIDE) {
        let (depth, _, _) =
            physics().primitives(state[(i, COL_M)], state[(i, COL_P)], state[(i, COL_S)])?;
        h.push(depth);
        b.push(state[(i, COL_B)]);
    }
    Ok((h, b))
}

fn main() -> Result<(), Box<dyn Error>> {
    let (channel0, lake0) = initial_states(DX)?;
    let result = run(&REFERENCE_RUN, &channel0, &lake0)?;

    let coordinates = |grid: &dyn Mesh| -> Vec<f64> {
        (0..grid.n_dof())
            .step_by(STRIDE)
            .map(|i| grid.coordinates(i)[0])
            .collect()
    };
    let channel_xs = coordinates(&channel_grid(DX)?);
    let lake_xs = coordinates(&lake_grid(DX)?);

    let mut channel_depths = Vec::new();
    let mut lake_surfaces = Vec::new();
    for state in &result.states {
        let (h, _) = depth_and_bed(field(state, &channel_id())?)?;
        channel_depths.push(h);
        let (h, b) = depth_and_bed(field(state, &lake_id())?)?;
        lake_surfaces.push(
            h.iter()
                .zip(&b)
                .map(|(h, b)| b + h - LAKE_DEPTH)
                .collect::<Vec<_>>(),
        );
    }
    let max_of = |frames: &[Vec<f64>]| frames.iter().flatten().fold(0.0_f64, |a, &v| a.max(v));
    let depth_max = (max_of(&channel_depths) * 1.15).ceil();
    let surface_max = (max_of(&lake_surfaces) * 1.15 * 2.0).ceil() / 2.0;

    let channel = Panel {
        x0: channel_xs[0],
        x1: channel_xs[channel_xs.len() - 1],
        y0: 0.0,
        y1: depth_max,
        left: LEFT,
        right: RIGHT,
        top: 50.0,
        bottom: 220.0,
        x_unit: 1000.0,
        x_label: "x (km)",
    };
    let lake = Panel {
        x0: lake_xs[0],
        x1: lake_xs[lake_xs.len() - 1],
        y0: -0.5,
        y1: surface_max,
        left: LEFT,
        right: RIGHT,
        top: 290.0,
        bottom: 460.0,
        x_unit: 1000.0,
        x_label: "x (km)",
    };
    let n = result.states.len();
    let duration = n as f64 * FRAME_SECONDS;

    let mut svg = begin(WIDTH, HEIGHT);
    channel.axes(
        &mut svg,
        "Channel: depth of the lahar",
        &ticks(channel.x0, channel.x1, 500.0),
        &ticks(0.0, depth_max, 1.0),
        "h (m)",
    );
    lake.axes(
        &mut svg,
        "Lake: free surface above the level at rest",
        &ticks(lake.x0, lake.x1, 500.0),
        &ticks(-0.5, surface_max, 0.5),
        "elevation (m)",
    );
    let channel_frames: Vec<String> = channel_depths
        .iter()
        .map(|h| channel.filled_points(&channel_xs, h))
        .collect();
    let lake_frames: Vec<String> = lake_surfaces
        .iter()
        .map(|s| lake.filled_points(&lake_xs, s))
        .collect();
    animated_curve(&mut svg, &channel_frames, duration, CURVE, true);
    animated_curve(&mut svg, &lake_frames, duration, CURVE, true);

    time_bar(
        &mut svg,
        LEFT,
        RIGHT,
        540.0,
        &result.times,
        END_TIME,
        duration,
        &format!(
            "time, 0 to {END_TIME} s (one frame every {} s)",
            REFERENCE_RUN.dt * REFERENCE_RUN.save_every as f64
        ),
        100.0,
        1.0,
    );
    end(&mut svg);
    std::fs::write("lahar_lake_wave_simul.svg", &svg)?;
    println!(
        "lahar_lake_wave_simul.svg: {n} frames, {:.0} s of animation, {} kB",
        duration,
        svg.len() / 1024
    );
    Ok(())
}
