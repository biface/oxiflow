//! # Reference example: viscous Burgers boundary layer at a wall ([#177](https://github.com/biface/oxiflow/issues/177))
//!
//! A viscous shock runs into a wall where the velocity is held at zero. It
//! stops there and relaxes to a steady boundary layer, whose thickness
//! scales with the viscosity. The example computes the steady layer for
//! three Reynolds numbers and checks it against the exact steady profile,
//! both the whole profile and the 99 % thickness.
//!
//! Run with: `cargo run --example burgers_boundary_layer`. The example exits
//! with an error if a check exceeds its tolerance.
//!
//! The scenario, the runs and the steady solution are defined once in
//! `rsc/models/burgers_boundary_layer.rs`, included below as the
//! `burgers_boundary_layer` module and shared with the examples that plot
//! and animate the runs.
//!
//! ## Physics
//!
//! Viscous Burgers, a model equation of fluid mechanics with no material
//! behind it, written in dimensionless form: x in units of the domain
//! length L, u in units of the inflow velocity U, t in units of L / U, and
//! a viscosity nu = 1 / Re.
//!
//! ```text
//! du/dt + u * du/dx = nu * d2u/dx2,   0 < x < 1
//! u(0, t) = 1,   u(1, t) = 0   (Dirichlet: inflow, wall)
//! u(x, 0) = 1/2 - 1/2 * tanh((x - x0) / (4 nu)),   x0 = 0.6
//! ```
//!
//! The initial state is the viscous shock from 1 to 0 centred at x0, the
//! exact travelling solution on an unbounded line. It moves towards the wall
//! at speed 1/2 and reaches it at t = 0.8. The wall then stops it, and the
//! layer relaxes to its steady state. The run goes to t = 2.4.
//!
//! The model is `rsc/models/viscous_burgers.rs`, shared with the
//! `viscous_burgers` benchmark and used here unchanged. It reads two
//! context values: the centred FD gradient (`FDGradientCalculator`) and the
//! FD Laplacian (`FDLaplacianCalculator`), both second order in space. Both
//! ends are held by `boundary::Dirichlet` conditions on the two faces of the
//! grid; the saved states carry the prescribed values.
//!
//! ## Steady boundary layer
//!
//! A steady state satisfies u u' = nu u''. Integrated once, with the
//! velocity gradient vanishing where u reaches its outer value A:
//!
//! ```text
//! nu * du/dx = (u² - A²) / 2,   so   u(x) = A * tanh(A (1 - x) / (2 nu))
//! ```
//!
//! which vanishes at the wall. A is set by the inflow condition,
//! A tanh(A / (2 nu)) = 1, and the example computes it by Newton's method.
//! For the three viscosities, A / (2 nu) is at least 25 and A differs from
//! 1 by less than 1e-21: A equals 1 to double precision (the example prints
//! the difference, 0).
//!
//! The 99 % thickness is the distance from the wall at which u reaches
//! 0.99 A:
//!
//! ```text
//! delta99 = (2 nu / A) * atanh(0.99)   (about 5.29 nu)
//! ```
//!
//! On the grid, it is located by linear interpolation between the two nodes
//! that bracket 0.99 A.
//!
//! ## Grid and integrator
//!
//! 401 nodes, dx = 1/400. The integrator is RK4. The time step is the
//! largest dt = 2.4 / n, with n a multiple of 8, that keeps
//! nu * dt / dx² at or below 0.6: dt = 1.875e-4, 3.75e-4 and 7.5e-4 for
//! Re = 50, 100 and 200. The run therefore ends exactly at t = 2.4, and the
//! state is saved every n / 8 steps, at every 0.3 time units.
//!
//! The layer is steady at t = 2.4: the example prints the largest change
//! between the states saved at t = 2.1 and 2.4, at most 2.1e-8 (Re = 50)
//! when the example was written, four orders below the profile error.
//!
//! ## Validation
//!
//! Tolerances:
//!
//! | Re | nu | delta99 / dx | Profile, max over the nodes of abs(u - steady) | delta99, relative error |
//! |---|---|---|---|---|
//! | 50 | 0.02 | 42.3 | 1.2e-3 | 2e-3 |
//! | 100 | 0.01 | 21.2 | 5e-3 | 1e-2 |
//! | 200 | 0.005 | 10.6 | 2e-2 | 3e-2 |
//!
//! Errors measured when the example was written: profile 5.84e-4, 2.35e-3
//! and 9.59e-3; delta99 relative errors 1.0e-3, 4.9e-3 and 1.56e-2; every
//! tolerance is about twice the measured error. Both errors converge at
//! second order in space (measured on 201, 401 and 801 nodes). They depend
//! only on the cell Reynolds number dx / nu: Re = 50 on 201 nodes gives the
//! same errors as Re = 100 on 401 nodes, as the steady profile depends on
//! x through (1 - x) / nu alone.

use std::error::Error;

use nalgebra::DVector;

#[path = "../rsc/models/burgers_boundary_layer.rs"]
mod burgers_boundary_layer;
#[path = "../rsc/models/viscous_burgers.rs"]
mod viscous_burgers_model;

use burgers_boundary_layer::{
    grid, outer_velocity, positions, solve, state_at, steady_thickness, steady_velocity, steps,
    END_TIME, INFLOW, N_NODES, REYNOLDS, SAVES, SHOCK_CENTRE, THICKNESS_LEVEL,
};

// ── Checks ────────────────────────────────────────────────────────────────

/// Tolerances at one Reynolds number, in the order of `REYNOLDS`.
struct Case {
    /// Tolerance on max abs(u - steady) over the nodes.
    profile_tolerance: f64,
    /// Tolerance on the relative error of delta99.
    thickness_tolerance: f64,
}

const CASES: [Case; 3] = [
    Case {
        profile_tolerance: 1.2e-3,
        thickness_tolerance: 2e-3,
    },
    Case {
        profile_tolerance: 5e-3,
        thickness_tolerance: 1e-2,
    },
    Case {
        profile_tolerance: 2e-2,
        thickness_tolerance: 3e-2,
    },
];

/// max abs(u - steady) over the nodes.
fn profile_error(velocity: &DVector<f64>, xs: &[f64], a: f64, nu: f64) -> f64 {
    velocity
        .iter()
        .zip(xs)
        .map(|(value, &x)| (value - steady_velocity(x, a, nu)).abs())
        .fold(0.0, f64::max)
}

/// Distance from the wall at which the computed velocity reaches
/// `THICKNESS_LEVEL * a`, by linear interpolation between the two nodes
/// that bracket it, scanning from the wall.
fn computed_thickness(velocity: &DVector<f64>, xs: &[f64], a: f64) -> Result<f64, Box<dyn Error>> {
    let level = THICKNESS_LEVEL * a;
    let i = (0..velocity.len() - 1)
        .rev()
        .find(|&i| velocity[i] >= level && velocity[i + 1] < level)
        .ok_or("the velocity never reaches 0.99 A")?;
    let crossing =
        xs[i] + (velocity[i] - level) / (velocity[i] - velocity[i + 1]) * (xs[i + 1] - xs[i]);
    Ok(1.0 - crossing)
}

/// Fails with the quantity, the Reynolds number, the error and the
/// tolerance.
fn check(quantity: &str, reynolds: f64, error: f64, tolerance: f64) -> Result<(), Box<dyn Error>> {
    if error <= tolerance {
        Ok(())
    } else {
        Err(format!(
            "{quantity}: error {error:.3e} at Re = {reynolds} exceeds the tolerance {tolerance:.1e}"
        )
        .into())
    }
}

fn main() -> Result<(), Box<dyn Error>> {
    let xs = positions(&grid()?);
    println!(
        "Viscous shock from x0 = {SHOCK_CENTRE} into a wall at x = 1, {N_NODES} nodes, RK4, steady layer at t = {END_TIME}"
    );
    for (&reynolds, case) in REYNOLDS.iter().zip(&CASES) {
        let nu = 1.0 / reynolds;
        let result = solve(nu, SAVES)?;
        let velocity = state_at(&result, END_TIME)?;
        let previous = state_at(&result, END_TIME * (SAVES - 1) as f64 / SAVES as f64)?;
        let a = outer_velocity(nu);

        let profile = profile_error(velocity, &xs, a, nu);
        let expected = steady_thickness(a, nu);
        let thickness = computed_thickness(velocity, &xs, a)?;
        let thickness_error = (thickness - expected).abs() / expected;
        println!(
            "  Re = {:<3} (nu = {nu}, dt = {:.3e}): |A - 1| = {:.1e}, max |u - steady| = {profile:.3e}, delta99 = {thickness:.5} (exact {expected:.5}, relative error {thickness_error:.2e}), change since t = {:.1} {:.1e} (reported)",
            reynolds,
            END_TIME / steps(nu) as f64,
            (a - INFLOW).abs(),
            END_TIME * (SAVES - 1) as f64 / SAVES as f64,
            (velocity - previous).amax()
        );
        check("steady profile", reynolds, profile, case.profile_tolerance)?;
        check(
            "99 % thickness (relative)",
            reynolds,
            thickness_error,
            case.thickness_tolerance,
        )?;
    }
    println!("burgers_boundary_layer: every check within its tolerance");
    Ok(())
}
