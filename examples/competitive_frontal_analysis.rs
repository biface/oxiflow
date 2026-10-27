//! # Reference example: binary frontal analysis with competitive adsorption (#174)
//!
//! A clean chromatographic column is fed, from t = 0 on, with a constant
//! mixture of two solutes that compete for the same adsorption sites. Each
//! solute leaves the column behind its own concentration front. Between the
//! two fronts, the less retained solute comes out above its feed
//! concentration: the more retained one, arriving later, displaces it from
//! the sites (roll-up). The example checks the arrival times of both fronts,
//! the height of the roll-up plateau and the mass balance against the exact
//! solution of the equilibrium theory.
//!
//! Run with: `cargo run --example competitive_frontal_analysis`. The example
//! exits with an error if a check exceeds its tolerance.
//!
//! ## Physics
//!
//! Ideal model of chromatography: plug flow, no axial dispersion,
//! instantaneous equilibrium between the mobile phase and the adsorbent
//! (the "PF-Equil" model of Nicoud, 2015). For each solute i, with c the
//! mobile-phase concentrations (mol/L) and q_i(c) the adsorbed amount per
//! unit volume of adsorbent:
//!
//! ```text
//! dc_i/dt + Fe * dq_i(c)/dt + ue * dc_i/dz = 0,   0 < z < L
//! q_i(c) = lambda * c_i + N_bar_i * K_i * c_i / (1 + K_1 * c_1 + K_2 * c_2)
//! c(z, 0) = 0,   c(0, t) = c_F   (t >= 0)
//! ```
//!
//! with Fe = (1 - eps) / eps and ue = v / eps, the interstitial velocity.
//! The isotherm is the competitive Langmuir isotherm with a linear term.
//!
//! The model is `rsc/models/langmuir_multi.rs`, shared with the
//! `langmuir_multi` benchmark and used here unchanged: the column and the
//! isotherm of its `ascorbic_erythorbic` case, fed through a
//! `TemporalInjection::SustainedStep`. It writes the transport equation
//! as dc/dt = -(I + Fe * M)^-1 * ue * dc/dz, with M = dq/dc, and takes the
//! first-order upwind gradient (`FDGradientCalculator`, `FDScheme::Backward`),
//! the feed concentration standing in as the ghost value upstream of the
//! inlet node.
//!
//! | Parameter | Value |
//! |---|---|
//! | Column length L | 0.25 m |
//! | Porosity eps | 0.4 |
//! | Superficial velocity v | 1 mm/s (ue = 2.5 mm/s) |
//! | Linear term lambda | 1 (both solutes) |
//! | Saturation capacity N | 2 mol/L (N_bar = (1 - eps) N = 1.2 mol/L) |
//! | Langmuir constants K_1, K_2 | 1.1 and 1.7 L/mol |
//! | Feed c_F | 0.1 mol/L of each solute |
//!
//! At 0.1 mol/L, K * c_F is 0.11 and 0.17: the adsorption is far enough
//! from the linear regime for the competition to show (a 29 % roll-up),
//! while the explicit integration stays well inside its stability limit
//! (see "Grid and integrator").
//!
//! ## Exact solution (equilibrium theory)
//!
//! Reference: R.-M. Nicoud, *Chromatographic Processes: Modeling,
//! Simulation, and Design*, Cambridge University Press, 2015, chapter 3
//! (non-linear chromatography, equilibrium theory) and appendix D
//! (binary chromatograms).
//!
//! For a step feed into a clean column, with a favourable (Langmuir)
//! isotherm, the solution is self-similar and made of two shocks:
//!
//! ```text
//! (0, 0)  --shock 1-->  (c1*, 0)  --shock 2-->  (c1F, c2F)
//! ```
//!
//! A shock between the states a and b travels at
//! ue / (1 + Fe * (q_i(b) - q_i(a)) / (c_i(b) - c_i(a))), the same for
//! every solute that jumps across it (Rankine–Hugoniot). It reaches the
//! outlet at t = (L / ue) * (1 + Fe * dq/dc), dq/dc being that ratio.
//!
//! * Shock 2 carries solute 2 from 0 to c2F: its ratio is
//!   r = q_2(c_F) / c2F. Solute 1 must jump across it with the same ratio,
//!   (q_1(c_F) - q_1(c1*, 0)) / (c1F - c1*) = r. With a_1 = N_bar_1 * K_1,
//!   D_F = 1 + K_1 c1F + K_2 c2F and s = r - lambda, this is the quadratic
//!
//!   ```text
//!   s K_1 x² + (K_1 a_1 c1F / D_F - a_1 - s K_1 c1F + s) x + (a_1 c1F / D_F - s c1F) = 0
//!   ```
//!
//!   in x = c1*. With the same lambda for both solutes, the product of its
//!   roots has the sign of a_1 - N_bar_2 K_2, negative here (K_1 < K_2, same
//!   N): c1* is its only positive root.
//! * Shock 1 carries solute 1 alone, from 0 to c1*: its ratio is
//!   lambda + a_1 / (1 + K_1 c1*).
//!
//! Values: c1* = 0.12861 mol/L (1.29 c1F), shock 1 at the outlet at
//! t = 423.46 s and shock 2 at t = 489.06 s. In the dilute limit, the two
//! fronts would arrive at 447.8 s and 555.1 s, with no plateau above c1F.
//!
//! ## Grid and integrator
//!
//! 401 nodes, dz = L / 400 = 0.625 mm. RK4 with dt = 0.5 s, up to
//! t = 600 s; every step is saved. The fastest characteristic speed in the
//! solution is 0.65 mm/s (feed state), so the Courant number is 0.52. The
//! time step does not show in the measured errors: dt = 0.25 s gives the
//! same front times to 1e-3 s.
//!
//! The upwind scheme is conservative despite its non-conservative look:
//! the flux ue * c is linear, so (I + Fe * M) dc/dt is the exact time
//! derivative of c + Fe * q at each node, and the upwind differences
//! telescope when summed over the nodes. Each node then carries a cell of
//! width dz upstream of it (node 0 included), and the semi-discrete
//! system conserves eps * c + (1 - eps) * q over these cells exactly. This
//! is why the scheme converges to the Rankine–Hugoniot fronts, and why the
//! mass balance below only measures the time integration.
//!
//! ## Validation
//!
//! | Check | Exact | Tolerance | Error measured when written |
//! |---|---|---|---|
//! | Time at which c_1 reaches c1* / 2 at the outlet | 423.46 s | 2 s | +0.96 s |
//! | Time at which c_2 reaches c2F / 2 at the outlet | 489.06 s | 2 s | +1.03 s |
//! | Highest c_1 at the outlet (roll-up plateau), relative | 0.12861 mol/L | 2.5e-2 | -1.22e-2 |
//! | Mass balance per solute, relative (fed = held + eluted) | 0 | 1e-6 | -4.1e-7, -3.9e-7 |
//!
//! The front times are found by linear interpolation between the saved
//! states. The held amount sums dz * (eps * c_i + (1 - eps) * q_i) over
//! the nodes at t = 600 s; the eluted amount integrates v * c_i at the
//! outlet by the trapezoidal rule over the saved states.
//!
//! The front-time errors and the plateau deficit come from the numerical
//! diffusion of the first-order upwind scheme, which smears the fronts: on
//! 801 nodes, the front errors halve (+0.46 s and +0.54 s) and the plateau
//! deficit drops to 5e-4. Each tolerance is about twice the measured error.

use std::error::Error;
use std::sync::Arc;

use oxiflow::{
    boundary::TemporalInjection,
    context::{
        calculators::{FDGradientCalculator, FDScheme},
        error::OxiflowError,
    },
    mesh::{Mesh, UniformGrid1D},
    solver::{
        config::{IntegratorKind, StepControl, TimeConfiguration},
        methods::RK4Solver,
        scenario::Scenario,
        SimulationResult, Solver, SolverConfiguration,
    },
};

#[path = "../rsc/models/langmuir_multi.rs"]
mod langmuir_multi_model;

use langmuir_multi_model::{ascorbic_erythorbic, LangmuirMulti};

// ── Scenario ──────────────────────────────────────────────────────────────

/// Column length (m).
const COLUMN_LENGTH: f64 = 0.25;
/// Grid nodes, both ends included (dz = L / 400).
const N_NODES: usize = 401;
/// Feed concentration of each solute (mol/L).
const FEED: f64 = 0.1;
const END_TIME: f64 = 600.0;
const TIME_STEP: f64 = 0.5;

/// Tolerance on the outlet front times (s).
const FRONT_TIME_TOLERANCE: f64 = 2.0;
/// Tolerance on the relative error of the roll-up plateau.
const PLATEAU_TOLERANCE: f64 = 2.5e-2;
/// Tolerance on the relative mass-balance error of each solute.
const MASS_BALANCE_TOLERANCE: f64 = 1e-6;

fn grid() -> Result<UniformGrid1D, OxiflowError> {
    UniformGrid1D::new(N_NODES, 0.0, COLUMN_LENGTH)
}

fn spacing() -> f64 {
    COLUMN_LENGTH / (N_NODES - 1) as f64
}

fn column() -> LangmuirMulti {
    ascorbic_erythorbic(
        spacing(),
        TemporalInjection::SustainedStep {
            start: 0.0,
            concentration: FEED,
        },
    )
}

/// Feeds the clean column up to `END_TIME`.
fn solve() -> Result<SimulationResult, OxiflowError> {
    let mut config = SolverConfiguration::new(
        TimeConfiguration::new(END_TIME, StepControl::Fixed { dt: TIME_STEP }),
        IntegratorKind::RK4,
    );
    for species in 0..column().n_species() {
        let mesh: Arc<dyn Mesh> = Arc::new(grid()?);
        config = config.with_calculator(Box::new(FDGradientCalculator::new(
            mesh,
            0,
            Some(species),
            FDScheme::Backward,
        )));
    }
    let scenario = Scenario::single(Box::new(column()), Box::new(grid()?));
    RK4Solver.solve(&scenario, &config)
}

// ── Exact solution ────────────────────────────────────────────────────────

/// Adsorbed amount q_i(c) of the model's isotherm.
fn adsorbed(model: &LangmuirMulti, c: [f64; 2], i: usize) -> f64 {
    let denominator = 1.0
        + model
            .species
            .iter()
            .zip(c)
            .map(|(species, ci)| species.langmuir_k * ci)
            .sum::<f64>();
    let species = &model.species[i];
    let n_bar = (1.0 - model.porosity) * species.port_number;
    species.lambda * c[i] + n_bar * species.langmuir_k * c[i] / denominator
}

/// The two-shock solution of the equilibrium theory.
struct Fronts {
    /// Roll-up plateau of solute 1, c1* (mol/L).
    plateau: f64,
    /// Outlet arrival times of shocks 1 and 2 (s).
    times: [f64; 2],
}

fn exact_fronts(model: &LangmuirMulti) -> Fronts {
    let feed = [FEED, FEED];
    let solute_1 = &model.species[0];
    let k1 = solute_1.langmuir_k;
    let a1 = (1.0 - model.porosity) * solute_1.port_number * k1;
    let d_feed = 1.0
        + model
            .species
            .iter()
            .zip(feed)
            .map(|(species, ci)| species.langmuir_k * ci)
            .sum::<f64>();

    // Shock 2: solute 2 from 0 to c2F, solute 1 from c1* to c1F, one ratio.
    let ratio_2 = adsorbed(model, feed, 1) / feed[1];
    let s = ratio_2 - solute_1.lambda;
    let a = s * k1;
    let b = k1 * a1 * feed[0] / d_feed - a1 - s * k1 * feed[0] + s;
    let c = a1 * feed[0] / d_feed - s * feed[0];
    let plateau = (-b + (b * b - 4.0 * a * c).sqrt()) / (2.0 * a);

    // Shock 1: solute 1 alone, from 0 to c1*.
    let ratio_1 = adsorbed(model, [plateau, 0.0], 0) / plateau;

    let fe = (1.0 - model.porosity) / model.porosity;
    let transit = COLUMN_LENGTH * model.porosity / model.velocity;
    Fronts {
        plateau,
        times: [
            transit * (1.0 + fe * ratio_1),
            transit * (1.0 + fe * ratio_2),
        ],
    }
}

// ── Checks ────────────────────────────────────────────────────────────────

/// Outlet concentration of `species` at every saved time.
fn outlet(result: &SimulationResult, species: usize) -> Result<Vec<f64>, OxiflowError> {
    result
        .states
        .iter()
        .map(|state| Ok(state.as_vector_field()?[(N_NODES - 1, species)]))
        .collect()
}

/// First time at which `values` reaches `level`, by linear interpolation
/// between the two saved states that bracket it.
fn crossing_time(times: &[f64], values: &[f64], level: f64) -> Result<f64, Box<dyn Error>> {
    let k = (1..values.len())
        .find(|&k| values[k - 1] < level && values[k] >= level)
        .ok_or_else(|| format!("the outlet concentration never reaches {level:.5} mol/L"))?;
    let fraction = (level - values[k - 1]) / (values[k] - values[k - 1]);
    Ok(times[k - 1] + fraction * (times[k] - times[k - 1]))
}

/// (held + eluted - fed) / fed for `species`, per unit column cross-section.
fn mass_balance_error(
    model: &LangmuirMulti,
    result: &SimulationResult,
    species: usize,
) -> Result<f64, Box<dyn Error>> {
    let last = result
        .states
        .last()
        .ok_or("no saved state")?
        .as_vector_field()?;
    let eps = model.porosity;
    let held: f64 = (0..N_NODES)
        .map(|node| {
            let c = [last[(node, 0)], last[(node, 1)]];
            spacing() * (eps * c[species] + (1.0 - eps) * adsorbed(model, c, species))
        })
        .sum();
    let values = outlet(result, species)?;
    let eluted: f64 = model.velocity
        * (1..values.len())
            .map(|k| 0.5 * (values[k] + values[k - 1]) * (result.times[k] - result.times[k - 1]))
            .sum::<f64>();
    let end = *result.times.last().ok_or("no saved time")?;
    let fed = model.velocity * FEED * end;
    Ok((held + eluted - fed) / fed)
}

/// Fails with the quantity, the error and the tolerance.
fn check(quantity: &str, error: f64, tolerance: f64) -> Result<(), Box<dyn Error>> {
    if error.abs() <= tolerance {
        Ok(())
    } else {
        Err(format!("{quantity}: error {error:+.3e} exceeds the tolerance {tolerance:.1e}").into())
    }
}

fn main() -> Result<(), Box<dyn Error>> {
    let model = column();
    let exact = exact_fronts(&model);
    println!(
        "Binary frontal analysis, c_F = {FEED} mol/L of each solute, {N_NODES} nodes, RK4, dt = {TIME_STEP} s"
    );
    println!(
        "  exact: plateau c1* = {:.5} mol/L ({:.3} c1F), fronts at the outlet at t = {:.2} s and {:.2} s",
        exact.plateau,
        exact.plateau / FEED,
        exact.times[0],
        exact.times[1]
    );

    let result = solve()?;
    let solute_1 = outlet(&result, 0)?;
    let solute_2 = outlet(&result, 1)?;
    let front_1 = crossing_time(&result.times, &solute_1, 0.5 * exact.plateau)?;
    let front_2 = crossing_time(&result.times, &solute_2, 0.5 * FEED)?;
    let highest = solute_1.iter().copied().fold(f64::MIN, f64::max);
    let plateau_error = (highest - exact.plateau) / exact.plateau;
    let balance = [
        mass_balance_error(&model, &result, 0)?,
        mass_balance_error(&model, &result, 1)?,
    ];

    println!(
        "  front 1: t = {front_1:.2} s (error {:+.3} s), front 2: t = {front_2:.2} s (error {:+.3} s)",
        front_1 - exact.times[0],
        front_2 - exact.times[1]
    );
    println!(
        "  roll-up plateau: highest c_1 = {highest:.5} mol/L (relative error {plateau_error:+.2e})"
    );
    println!(
        "  mass balance, relative: solute 1 {:+.2e}, solute 2 {:+.2e}",
        balance[0], balance[1]
    );

    check(
        "front 1 time",
        front_1 - exact.times[0],
        FRONT_TIME_TOLERANCE,
    )?;
    check(
        "front 2 time",
        front_2 - exact.times[1],
        FRONT_TIME_TOLERANCE,
    )?;
    check(
        "roll-up plateau (relative)",
        plateau_error,
        PLATEAU_TOLERANCE,
    )?;
    check(
        "mass balance of solute 1",
        balance[0],
        MASS_BALANCE_TOLERANCE,
    )?;
    check(
        "mass balance of solute 2",
        balance[1],
        MASS_BALANCE_TOLERANCE,
    )?;
    println!("competitive_frontal_analysis: every check within its tolerance");
    Ok(())
}
