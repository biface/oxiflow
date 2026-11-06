//! # Reference example: magnetic diffusion into a conductor ([#179](https://github.com/biface/oxiflow/issues/179))
//!
//! A uniform magnetic field B0 is switched on at t = 0, parallel to the
//! flat surface of a thick conductor that holds no field beforehand. The
//! field diffuses into the metal; the example follows it in copper and in
//! aluminium, in physical units, and checks it against the exact
//! half-space solution and against the diffusive scaling of the
//! penetration depth with sqrt(eta*t).
//!
//! This is a prototype of the diffusive electromagnetic regime that the
//! future `oxiflow-em` framework targets, not a Maxwell solver: the
//! displacement current is neglected, which is the usual assumption in a
//! good conductor.
//!
//! Run with: `cargo run --example magnetic_diffusion_1d`. The example
//! exits with an error if a check exceeds its tolerance.
//!
//! The model, the conductors, the plate, the runs and the exact solution
//! are defined once in `rsc/models/magnetic_diffusion_1d.rs`, included below
//! as the `magnetic_diffusion_1d` module and shared with the examples that
//! plot and animate the runs.
//!
//! ## Physics
//!
//! For the field component parallel to the surface, B(x, t), with x the
//! depth below the surface:
//!
//! ```text
//! dB/dt = eta * d2B/dx2,   eta = 1 / (mu0 * sigma),   0 < x < L
//! B(0, t) = B0,   B(L, t) = 0   (Dirichlet, both faces)
//! B(x, 0) = 0
//! ```
//!
//! mu0 is taken as 4*pi*1e-7 H/m. Since the 2019 SI revision it is a
//! measured constant; the CODATA 2018 value differs from 4*pi*1e-7 by
//! 5.4e-10 in relative terms, far below anything this example resolves.
//! Conductivities are typical values at 20 °C: copper 5.96e7 S/m
//! (eta ≈ 1.335e-2 m²/s), aluminium 3.77e7 S/m (eta ≈ 2.111e-2 m²/s).
//! B0 = 0.1 T. The plate is L = 3 cm thick, on 121 nodes (dx = 0.25 mm).
//!
//! The model is a single context value: the FD Laplacian of the field
//! (`FDLaplacianCalculator`, second order in space), scaled by eta. The
//! surface and the far face are held by `boundary::Dirichlet` conditions
//! on the two faces of the grid; the saved states carry the prescribed
//! values, the initial state included.
//!
//! ## Exact solution: half-space
//!
//! For a half-space (L infinite), the field is
//!
//! ```text
//! B(x, t) = B0 * erfc( x / (2 sqrt(eta t)) )
//! ```
//!
//! The plate is finite, with its far face held field-free. By images, the
//! plate solution differs from the half-space one by at most
//! erfc(L / (2 sqrt(eta t))) * B0 over the plate. At the last time
//! compared, t = 1 ms, that bound is 6.4e-9 * B0 in copper and
//! 3.9e-6 * B0 in aluminium, more than ten times below the tolerance.
//!
//! `erfc` is computed by the classical Chebyshev approximation `erfcc`,
//! written below. Its error, measured against a reference implementation
//! on [0, 10], is at most 8.3e-8 in absolute value.
//!
//! ## Penetration depth
//!
//! The depth at which the field has reached half its surface value,
//! delta_half(t), follows from erfc(z) = 1/2:
//!
//! ```text
//! delta_half(t) = 2 erfc^-1(1/2) * sqrt(eta t) ≈ 0.953873 * sqrt(eta t)
//! ```
//!
//! The factor 2 erfc^-1(1/2) is computed by bisection on the same `erfc`;
//! it comes out within 8.0e-8 of the exact value.
//! The example measures delta_half on the computed field, by linear
//! interpolation between the two nodes that bracket B0/2, and checks the
//! ratio delta_half / sqrt(eta t): the same value is expected at every time
//! and in both metals. It also prints the aluminium-to-copper ratio of the
//! depths, expected at sqrt(sigma_Cu / sigma_Al) ≈ 1.257, without checking
//! it.
//!
//! ## Integrator
//!
//! The initial data are discontinuous at the surface. BDF2 is L-stable and
//! damps the stiffest modes that discontinuity excites. The time step,
//! dt = 1e-5 s, gives eta*dt/dx² ≈ 2.1 in copper and 3.4 in aluminium. At
//! those values one Newton correction converges on this linear problem, so
//! the solver keeps its default budget of one iteration (compare
//! `heat_transfer_1d`, where eta*dt/dx² reaches 690 and the budget is
//! raised).
//!
//! ## Validation
//!
//! | Check | Checked at | Tolerance |
//! |---|---|---|
//! | Profile, max over the nodes of abs(B - B0 erfc) | t = 0.25, 0.5, 1 ms | 5e-5 T (5e-4 B0) |
//! | delta_half / sqrt(eta t) against 2 erfc^-1(1/2) | t = 0.25, 0.5, 1 ms | 1e-3, relative |
//!
//! Errors measured when the example was written: profile at most
//! 2.27e-5 T (copper, t = 0.25 ms); depth ratio at most 3.5e-4 (aluminium,
//! t = 0.25 ms). The profile error converges at second order in space
//! (measured on 61, 121 and 241 nodes, aluminium, t = 0.25 ms).

use std::error::Error;

#[path = "../rsc/models/magnetic_diffusion_1d.rs"]
mod magnetic_diffusion_1d;

use magnetic_diffusion_1d::{
    diffusivity, half_field_depth, half_field_factor, half_space_field, plate, positions, solve,
    state_at, CONDUCTORS, DT, SAVE_EVERY, SURFACE_FIELD, THICKNESS,
};

// ── Checks ────────────────────────────────────────────────────────────────

const CHECK_TIMES: [f64; 3] = [2.5e-4, 5e-4, 1e-3];
/// Tolerance on the profile, T.
const PROFILE_TOLERANCE: f64 = 5e-5;
/// Relative tolerance on delta_half / sqrt(eta t).
const DEPTH_TOLERANCE: f64 = 1e-3;

/// Fails with the conductor, the time, the error and the tolerance.
fn check(
    label: &str,
    quantity: &str,
    time: f64,
    error: f64,
    tolerance: f64,
) -> Result<(), Box<dyn Error>> {
    if error <= tolerance {
        Ok(())
    } else {
        Err(format!(
            "{label}: {quantity} error {error:.3e} at t = {time} s exceeds the tolerance {tolerance:.1e}"
        )
        .into())
    }
}

fn main() -> Result<(), Box<dyn Error>> {
    let xs = positions(&plate()?);
    let factor = half_field_factor();
    println!(
        "Field step of {SURFACE_FIELD} T at the surface of a {} mm plate, BDF2, dt = {DT} s",
        THICKNESS * 1e3
    );
    println!("Expected delta_half / sqrt(eta t) = 2 erfc^-1(1/2) = {factor:.7}");

    let mut depths: Vec<Vec<f64>> = Vec::new();
    for (name, sigma) in CONDUCTORS {
        let eta = diffusivity(sigma);
        println!("{name}: sigma = {sigma:.3e} S/m, eta = {eta:.4e} m²/s");
        let result = solve(eta, SAVE_EVERY)?;
        let mut conductor_depths = Vec::new();
        for &time in &CHECK_TIMES {
            let field = state_at(&result, time, name)?;
            let profile_error = field
                .iter()
                .zip(&xs)
                .map(|(value, &x)| (value - half_space_field(eta, x, time)).abs())
                .fold(0.0, f64::max);
            let depth = half_field_depth(field, &xs, name)?;
            let ratio = depth / (eta * time).sqrt();
            let depth_error = (ratio - factor).abs() / factor;
            println!(
                "  t = {:>4} ms: max |B - B0 erfc| = {profile_error:.3e} T   delta_half = {:.4} mm   delta_half / sqrt(eta t) = {ratio:.6} (rel. error {depth_error:.2e})",
                time * 1e3,
                depth * 1e3
            );
            check(name, "profile (T)", time, profile_error, PROFILE_TOLERANCE)?;
            check(name, "relative depth", time, depth_error, DEPTH_TOLERANCE)?;
            conductor_depths.push(depth);
        }
        depths.push(conductor_depths);
    }

    let expected = (CONDUCTORS[0].1 / CONDUCTORS[1].1).sqrt();
    for (i, &time) in CHECK_TIMES.iter().enumerate() {
        println!(
            "  t = {:>4} ms: delta_half aluminium / copper = {:.4} (sqrt(sigma_Cu / sigma_Al) = {expected:.4}, reported)",
            time * 1e3,
            depths[1][i] / depths[0][i]
        );
    }
    println!("magnetic_diffusion_1d: every check within its tolerance");
    Ok(())
}
