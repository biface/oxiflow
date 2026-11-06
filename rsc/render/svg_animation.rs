//! Animated SVG built from SVG's own animation elements (`<animate>`), with
//! no script: the file plays wherever an SVG is shown as an image, in a web
//! browser or on the wiki, and a viewer that does not play SVG animations
//! shows the first frame.
//!
//! Generic drawing pieces, with no physics, for the `_simul` examples, each
//! of which includes this file as a module with `#[path]`:
//!
//! - [`begin`] and [`end`] open and close the document;
//! - a [`Panel`] maps a data range onto a pixel box, draws its axes and
//!   turns a curve into the `points` of an SVG polygon or polyline;
//! - [`animated_curve`] plays one curve per frame, filled down to the
//!   bottom of its panel or open;
//! - [`legend`] names the curves and [`time_bar`] shows the time of the
//!   frame on display.
//!
//! The caller collects the frames (one entry per saved state) and decides
//! how many it keeps: the size of the file grows with the number of frames
//! times the number of points per curve.

// A toolbox: each `_simul` example uses only the pieces it needs.
#![allow(dead_code)]

use std::fmt::Write as _;

/// Default colour of a curve.
pub(crate) const CURVE: &str = "#1f6f8b";
/// Colour of the text, the frames and the ticks.
pub(crate) const INK: &str = "#333333";
/// Colour of the grid lines.
pub(crate) const GRID: &str = "#d0d0d0";

/// Opens a document of `width` × `height` pixels on a white background.
pub(crate) fn begin(width: f64, height: f64) -> String {
    let mut svg = String::new();
    let _ = writeln!(
        svg,
        r##"<svg viewBox="0 0 {width} {height}" xmlns="http://www.w3.org/2000/svg" font-family="Georgia, serif">
  <rect x="0" y="0" width="{width}" height="{height}" fill="#ffffff"/>"##
    );
    svg
}

/// Closes the document.
pub(crate) fn end(svg: &mut String) {
    svg.push_str("</svg>\n");
}

/// `v` rounded to nine decimals, so that a tick computed as a sum of
/// steps is labelled `0.3` and not `0.30000000000000004`.
fn tidy(v: f64) -> f64 {
    (v * 1e9).round() / 1e9
}

/// Multiples of `step` from `a` to `b`, both included when they fall on a
/// multiple.
pub(crate) fn ticks(a: f64, b: f64, step: f64) -> Vec<f64> {
    let mut v = Vec::new();
    let mut x = (a / step).ceil() * step;
    while x <= b + 1e-9 {
        v.push(x);
        x += step;
    }
    v
}

/// A plotting panel: the data range `x0..x1`, `y0..y1` mapped onto the
/// pixel box `left..right`, `top..bottom`.
pub(crate) struct Panel {
    pub(crate) x0: f64,
    pub(crate) x1: f64,
    pub(crate) y0: f64,
    pub(crate) y1: f64,
    pub(crate) left: f64,
    pub(crate) right: f64,
    pub(crate) top: f64,
    pub(crate) bottom: f64,
    /// The x tick labels show `x / x_unit` (1000 for metres shown in km).
    pub(crate) x_unit: f64,
    /// Label of the x axis, with its unit.
    pub(crate) x_label: &'static str,
}

impl Panel {
    pub(crate) fn px(&self, x: f64) -> f64 {
        self.left + (x - self.x0) / (self.x1 - self.x0) * (self.right - self.left)
    }

    pub(crate) fn py(&self, y: f64) -> f64 {
        self.bottom - (y - self.y0) / (self.y1 - self.y0) * (self.bottom - self.top)
    }

    /// One frame as the `points` of a polygon, closed down to `y0` so that
    /// it can be filled. Values are clamped to the panel.
    pub(crate) fn filled_points(&self, xs: &[f64], ys: &[f64]) -> String {
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

    /// One frame as the `points` of an open polyline. Values are clamped to
    /// the panel.
    pub(crate) fn open_points(&self, xs: &[f64], ys: &[f64]) -> String {
        let mut s = String::new();
        for (i, (x, y)) in xs.iter().zip(ys).enumerate() {
            let separator = if i == 0 { "" } else { " " };
            let _ = write!(
                s,
                "{separator}{:.1},{:.1}",
                self.px(*x),
                self.py(y.clamp(self.y0, self.y1))
            );
        }
        s
    }

    /// Frame, grid lines, tick labels, title and axis labels.
    pub(crate) fn axes(
        &self,
        svg: &mut String,
        title: &str,
        x_ticks: &[f64],
        y_ticks: &[f64],
        y_label: &str,
    ) {
        let (left, right) = (self.left, self.right);
        for &y in y_ticks {
            let py = self.py(y);
            let _ = writeln!(
                svg,
                r#"  <line x1="{left}" y1="{py:.1}" x2="{right}" y2="{py:.1}" stroke="{GRID}" stroke-width="0.6"/>
  <text x="{:.1}" y="{:.1}" font-size="12" fill="{INK}" text-anchor="end">{}</text>"#,
                left - 6.0,
                py + 4.0,
                tidy(y)
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
                tidy(x / self.x_unit),
                b = self.bottom
            );
        }
        let _ = writeln!(
            svg,
            r#"  <rect x="{left}" y="{:.1}" width="{}" height="{:.1}" fill="none" stroke="{INK}" stroke-width="1.2"/>
  <text x="{left}" y="{:.1}" font-size="14" font-weight="bold" fill="{INK}">{title}</text>
  <text x="{:.1}" y="{:.1}" font-size="12" fill="{INK}" transform="rotate(-90 {:.1} {:.1})" text-anchor="middle">{y_label}</text>
  <text x="{right}" y="{:.1}" font-size="12" fill="{INK}" text-anchor="end">{}</text>"#,
            self.top,
            right - left,
            self.bottom - self.top,
            self.top - 8.0,
            left - 42.0,
            0.5 * (self.top + self.bottom),
            left - 42.0,
            0.5 * (self.top + self.bottom),
            self.bottom + 34.0,
            self.x_label
        );
    }
}

/// A curve that changes with each frame, one entry of `frames` per frame
/// (from [`Panel::filled_points`] when `filled`, from
/// [`Panel::open_points`] otherwise), played in a loop of `duration`
/// seconds.
pub(crate) fn animated_curve(
    svg: &mut String,
    frames: &[String],
    duration: f64,
    color: &str,
    filled: bool,
) {
    let first = &frames[0];
    let (element, paint) = if filled {
        (
            "polygon",
            format!(r#"fill="{color}" fill-opacity="0.25" stroke="{color}""#),
        )
    } else {
        ("polyline", format!(r#"fill="none" stroke="{color}""#))
    };
    let _ = writeln!(
        svg,
        r#"  <{element} points="{first}" {paint} stroke-width="1.6">
    <animate attributeName="points" dur="{duration:.1}s" repeatCount="indefinite" calcMode="discrete" values="{}"/>
  </{element}>"#,
        frames.join(";")
    );
}

/// One line per entry, a short stroke of its colour then its label, from
/// the top-left corner `(x, y)` downwards.
pub(crate) fn legend(svg: &mut String, x: f64, y: f64, entries: &[(&str, &str)]) {
    for (i, (color, label)) in entries.iter().enumerate() {
        let row = y + 18.0 * i as f64;
        let _ = writeln!(
            svg,
            r#"  <line x1="{x:.1}" y1="{row:.1}" x2="{:.1}" y2="{row:.1}" stroke="{color}" stroke-width="2.4"/>
  <text x="{:.1}" y="{:.1}" font-size="12" fill="{INK}">{label}</text>"#,
            x + 22.0,
            x + 28.0,
            row + 4.0
        );
    }
}

/// A bar from `left` to `right` that fills as the frames play, the time of
/// frame `i` being `times[i]` out of `end_time`. Labels every `tick_step`
/// above the bar, showing `t / tick_unit`; `caption` under it, and on the
/// right a note on where to open the file.
#[allow(clippy::too_many_arguments)]
pub(crate) fn time_bar(
    svg: &mut String,
    left: f64,
    right: f64,
    top: f64,
    times: &[f64],
    end_time: f64,
    duration: f64,
    caption: &str,
    tick_step: f64,
    tick_unit: f64,
) {
    let height = 10.0;
    let widths: Vec<String> = times
        .iter()
        .map(|t| format!("{:.1}", t / end_time * (right - left)))
        .collect();
    let _ = writeln!(
        svg,
        r#"  <rect x="{left}" y="{top}" width="{}" height="{height}" fill="none" stroke="{INK}"/>
  <rect x="{left}" y="{top}" width="{}" height="{height}" fill="{CURVE}">
    <animate attributeName="width" dur="{duration:.1}s" repeatCount="indefinite" calcMode="discrete" values="{}"/>
  </rect>
  <text x="{left}" y="{:.1}" font-size="12" fill="{INK}">{caption}</text>
  <text x="{right}" y="{:.1}" font-size="12" fill="{INK}" text-anchor="end" font-style="italic">animated SVG: open it in a web browser</text>"#,
        right - left,
        widths[0],
        widths.join(";"),
        top + height + 18.0,
        top + height + 18.0,
    );
    for t in ticks(0.0, end_time, tick_step) {
        let px = left + t / end_time * (right - left);
        let _ = writeln!(
            svg,
            r#"  <text x="{px:.1}" y="{:.1}" font-size="11" fill="{INK}" text-anchor="middle">{}</text>"#,
            top - 4.0,
            tidy(t / tick_unit)
        );
    }
}
