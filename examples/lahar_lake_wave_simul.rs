//! # Animation: lahar–lake wave generation ([#180](https://github.com/biface/oxiflow/issues/180))
//!
//! Replays the run of `examples/lahar_lake_wave.rs` as an animated SVG,
//! `lahar_lake_wave_simul.svg`, written to the working directory: the lahar
//! running down the channel, then the bore it raises across the lake. The
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
//!   below the level at rest while pressing harder on the bed, and the bore
//!   it pushes stands about 1.7 m above that level.
//!
//! Run with: `cargo run --release --example lahar_lake_wave_simul`.

use std::error::Error;
use std::fmt::Write as _;

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

use lahar_outfall::{
    channel_grid, channel_id, field, initial_states, lake_grid, lake_id, physics, run, LAKE_DEPTH,
    REFERENCE_RUN,
};
use lahar_physics::{COL_B, COL_M, COL_P, COL_S};

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
const CURVE: &str = "#1f6f8b";
const INK: &str = "#333333";
const GRID: &str = "#d0d0d0";

/// A plotting panel: data ranges mapped onto a pixel box.
struct Panel {
    x0: f64,
    x1: f64,
    y0: f64,
    y1: f64,
    top: f64,
    bottom: f64,
}

impl Panel {
    fn px(&self, x: f64) -> f64 {
        LEFT + (x - self.x0) / (self.x1 - self.x0) * (RIGHT - LEFT)
    }
    fn py(&self, y: f64) -> f64 {
        self.bottom - (y - self.y0) / (self.y1 - self.y0) * (self.bottom - self.top)
    }
    /// One frame as the `points` of a polyline, closed down to `y0` so that
    /// it can be filled.
    fn points(&self, xs: &[f64], ys: &[f64]) -> String {
        let mut s = format!("{:.1},{:.1}", self.px(xs[0]), self.py(self.y0));
        for (x, y) in xs.iter().zip(ys) {
            let _ = write!(
                s,
                " {:.1},{:.1}",
                self.px(*x),
                self.py(y.clamp(self.y0, self.y1))
            );
        }
        let _ = write!(
            s,
            " {:.1},{:.1}",
            self.px(xs[xs.len() - 1]),
            self.py(self.y0)
        );
        s
    }
    /// Axes, grid lines and labels.
    fn axes(&self, svg: &mut String, title: &str, x_ticks: &[f64], y_ticks: &[f64], y_label: &str) {
        for &y in y_ticks {
            let py = self.py(y);
            let _ = writeln!(
                svg,
                r#"  <line x1="{LEFT}" y1="{py:.1}" x2="{RIGHT}" y2="{py:.1}" stroke="{GRID}" stroke-width="0.6"/>
  <text x="{:.1}" y="{:.1}" font-size="12" fill="{INK}" text-anchor="end">{y}</text>"#,
                LEFT - 6.0,
                py + 4.0
            );
        }
        for &x in x_ticks {
            let px = self.px(x);
            let _ = writeln!(
                svg,
                r#"  <line x1="{px:.1}" y1="{b:.1}" x2="{px:.1}" y2="{:.1}" stroke="{INK}"/>
  <text x="{px:.1}" y="{:.1}" font-size="12" fill="{INK}" text-anchor="middle">{}</text>"#,
                self.bottom + 5.0,
                self.bottom + 19.0,
                x / 1000.0,
                b = self.bottom
            );
        }
        let _ = writeln!(
            svg,
            r#"  <rect x="{LEFT}" y="{:.1}" width="{}" height="{:.1}" fill="none" stroke="{INK}" stroke-width="1.2"/>
  <text x="{LEFT}" y="{:.1}" font-size="14" font-weight="bold" fill="{INK}">{title}</text>
  <text x="{:.1}" y="{:.1}" font-size="12" fill="{INK}" transform="rotate(-90 {:.1} {:.1})" text-anchor="middle">{y_label}</text>
  <text x="{RIGHT}" y="{:.1}" font-size="12" fill="{INK}" text-anchor="end">x (km)</text>"#,
            self.top,
            RIGHT - LEFT,
            self.bottom - self.top,
            self.top - 8.0,
            LEFT - 42.0,
            0.5 * (self.top + self.bottom),
            LEFT - 42.0,
            0.5 * (self.top + self.bottom),
            self.bottom + 34.0
        );
    }
}

/// An animated, filled curve: one frame per entry of `frames`.
fn animated_curve(svg: &mut String, frames: &[String], duration: f64) {
    let first = &frames[0];
    let _ = writeln!(
        svg,
        r#"  <polygon points="{first}" fill="{CURVE}" fill-opacity="0.25" stroke="{CURVE}" stroke-width="1.6">
    <animate attributeName="points" dur="{duration:.1}s" repeatCount="indefinite" calcMode="discrete" values="{}"/>
  </polygon>"#,
        frames.join(";")
    );
}

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
        top: 50.0,
        bottom: 220.0,
    };
    let lake = Panel {
        x0: lake_xs[0],
        x1: lake_xs[lake_xs.len() - 1],
        y0: -0.5,
        y1: surface_max,
        top: 290.0,
        bottom: 460.0,
    };
    let n = result.states.len();
    let duration = n as f64 * FRAME_SECONDS;

    let mut svg = String::new();
    let _ = writeln!(
        svg,
        r##"<svg viewBox="0 0 {WIDTH} {HEIGHT}" xmlns="http://www.w3.org/2000/svg" font-family="Georgia, serif">
  <rect x="0" y="0" width="{WIDTH}" height="{HEIGHT}" fill="#ffffff"/>"##
    );
    let ticks = |a: f64, b: f64, step: f64| -> Vec<f64> {
        let mut v = Vec::new();
        let mut x = (a / step).ceil() * step;
        while x <= b + 1e-9 {
            v.push(x);
            x += step;
        }
        v
    };
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
        .map(|h| channel.points(&channel_xs, h))
        .collect();
    let lake_frames: Vec<String> = lake_surfaces
        .iter()
        .map(|s| lake.points(&lake_xs, s))
        .collect();
    animated_curve(&mut svg, &channel_frames, duration);
    animated_curve(&mut svg, &lake_frames, duration);

    // Time bar.
    let (bar_top, bar_height) = (540.0, 10.0);
    let widths: Vec<String> = result
        .times
        .iter()
        .map(|t| format!("{:.1}", t / END_TIME * (RIGHT - LEFT)))
        .collect();
    let _ = writeln!(
        svg,
        r#"  <rect x="{LEFT}" y="{bar_top}" width="{}" height="{bar_height}" fill="none" stroke="{INK}"/>
  <rect x="{LEFT}" y="{bar_top}" width="{}" height="{bar_height}" fill="{CURVE}">
    <animate attributeName="width" dur="{duration:.1}s" repeatCount="indefinite" calcMode="discrete" values="{}"/>
  </rect>
  <text x="{LEFT}" y="{:.1}" font-size="12" fill="{INK}">time, 0 to {END_TIME} s (one frame every {} s)</text>
  <text x="{RIGHT}" y="{:.1}" font-size="12" fill="{INK}" text-anchor="end" font-style="italic">animated SVG: open it in a web browser</text>"#,
        RIGHT - LEFT,
        widths[0],
        widths.join(";"),
        bar_top + bar_height + 18.0,
        REFERENCE_RUN.dt * REFERENCE_RUN.save_every as f64,
        bar_top + bar_height + 18.0,
    );
    for t in ticks(0.0, END_TIME, 100.0) {
        let px = LEFT + t / END_TIME * (RIGHT - LEFT);
        let _ = writeln!(
            svg,
            r#"  <text x="{px:.1}" y="{:.1}" font-size="11" fill="{INK}" text-anchor="middle">{t}</text>"#,
            bar_top - 4.0
        );
    }
    svg.push_str("</svg>\n");
    std::fs::write("lahar_lake_wave_simul.svg", &svg)?;
    println!(
        "lahar_lake_wave_simul.svg: {n} frames, {:.0} s of animation, {} kB",
        duration,
        svg.len() / 1024
    );
    Ok(())
}
