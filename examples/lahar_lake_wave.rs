//! # Reference example: a lahar front running into a lake, on coupled grids ([#180](https://github.com/biface/oxiflow/issues/180))
//!
//! A lahar runs down a 4 km channel and pours over a lip into a 15 m deep
//! lake. The inflow raises a bore, a front that runs across the lake: as
//! long as the lahar feeds it, the lake stays raised behind the front, so
//! what crosses the lake is a front, not a wave that would leave the inlet
//! behind. Channel and lake are two domains on their own grids, advanced
//! together by `MultiDomainOrchestrator` and coupled only through
//! `CouplingOperator`s (INV-3). The example checks that the lake stays
//! exactly at rest until the lahar reaches it, that the coupling moves mass
//! and solids from one grid to the other without loss, and that the bore
//! obeys the Rankine–Hugoniot jump conditions.
//!
//! Run with: `cargo run --example lahar_lake_wave`. The example exits with an
//! error if a check exceeds its tolerance.
//!
//! ## Physics
//!
//! Both domains use the 1D multi-field lahar model of [DQ-032](https://github.com/biface/oxiflow/issues/85) (Woodhouse,
//! Hogg & Phillips), shared with `lahar_multifield` in `rsc/models/`: a
//! depth-averaged layer of water and solids with conserved state
//! `(m, p, s, b)`, mass depth, momentum depth, solid volume depth and bed
//! elevation (see `examples/lahar_multifield.rs` for the equations). The lake
//! is the same model with other parameters: a layer of pure water at rest,
//! on a flat bed, into which the solids brought by the lahar are carried.
//!
//! | Parameter | Value |
//! |---|---|
//! | Water density rho_f, grain density rho_s | 1000, 2650 kg/m³ (quartz) |
//! | Inflow, per unit width | q = 20 m²/s, solids fraction psi = 0.2 (hyperconcentrated flow) |
//! | Grain diameter, water viscosity | 0.05 mm (coarse silt), 1e-6 m²/s |
//! | Channel | 4000 m at 3°, non-erodible, source over x in (50, 150) m |
//! | Lake | 3000 m, 15 m deep, flat bed, closed at both ends |
//! | Basal drag | Chezy, Cd = 0.01 (Manning n of about 0.04 for a depth of 2 m) |
//! | Deposit packing psi_b | 0.6 |
//!
//! The granular friction branch is off (`tan_d1 = tan_d2 = tan_d3 = 0`, and
//! `psi0 = 1` so that the Chezy term keeps its full weight): a dilute,
//! turbulent flow. The channel bed does not erode (`epsilon = 0`): with the
//! erosion efficiency of the benchmark (1e-3), the closure would take about
//! 1.8 m/s of solids from the bed at 10 m/s, some 3 m of bed per second, far
//! beyond what a lahar erodes. Solids settle everywhere; with silt, about
//! three quarters of them reach the lake (psi = 0.165 and 19.0 m²/s at the
//! outlet, against 0.2 and 21 m²/s at the source).
//!
//! The source injects the bulk mass rho_bar(psi) * q of the inflow. Over the
//! 21 nodes of the source strip, the discrete rate is 21 m²/s rather than 20.
//!
//! ## Coupling
//!
//! The channel ends at a lip above the lake surface, the front of a delta:
//! the lake cannot flow back into the channel, and the exchange goes one
//! way. On the channel side, the outer face of the last cell carries no flux
//! of its own (`EdgeTreatment::Exchanged`); on the lake side, the inlet is a
//! wall (`EdgeTreatment::Wall`), which holds the lake's own hydrostatic
//! pressure. The upper end of the channel and the far shore of the lake are
//! walls as well.
//!
//! Two coupling operators carry the outflow of the channel, the physical
//! flux F(U) of its last cell (free outfall), across the lip:
//!
//! - `channel → lake` adds dt * F / dx to the first cell of the lake;
//! - `lake → channel` takes the same amount from the last cell of the
//!   channel.
//!
//! An operator writes only its target domain, so the exchange needs two of
//! them. Both read the channel state, which the first one leaves untouched:
//! they compute the same flux, and what the channel loses, the lake gains,
//! to rounding. Their order matters (gain first). Mass and solids are
//! exchanged exactly; the momentum the channel loses enters the lake, but
//! the reaction of the lip is not followed, so momentum is not conserved
//! across the interface.
//!
//! The exchange follows the step of each domain (operator splitting). The
//! saved state of the last channel cell is the state after the exchange,
//! lower than the flow through the lip by dt * F / dx (0.4 m of depth at
//! dt = 0.1 s): the example reads the outflow at the node before it.
//!
//! The orchestrator builds a single calculator chain for every domain, and
//! the bed-slope gradient is computed with one spacing: both grids therefore
//! share dx = 5 m.
//!
//! ## Grid and integrator
//!
//! The scenario (constants, grids, initial states, domain models, coupling)
//! and the settings of this run (`REFERENCE_RUN`) are defined in
//! `rsc/models/lahar_outfall.rs`, shared with the animation
//! `examples/lahar_lake_wave_simul.rs`; this file holds the diagnostics and
//! the checks.
//!
//! 801 channel nodes and 601 lake nodes, dx = 5 m. Explicit Euler, the
//! integrator the overshoot guards of the source term were written for,
//! dt = 0.1 s (largest wave speed times dt / dx about 0.3); the state is
//! saved every 5 s, up to t = 600 s, before the bore reaches the far shore.
//!
//! ## Validation
//!
//! 1. The lake stays bitwise at rest until the channel outlet is wet: every
//!    saved lake state before that equals the initial one.
//! 2. Exchange: the mass the channel has lost through the lip (injected mass
//!    minus the change of its inventory) equals the mass the lake has
//!    gained. Inventories count the layer and the deposit, m + rho_b b per
//!    unit length (s + psi_b b for the solids), each node as a cell of width
//!    dx. The difference, relative to the injected mass, is checked at every
//!    saved state, for the mass and for the solids.
//! 3. Bore: in still water of depth h0, a bore with depth h1 behind it runs
//!    at
//!
//!    ```text
//!    c = sqrt(g h1 (h1 + h0) / (2 h0))         (momentum)
//!    u1 = c (1 - h0 / h1)                      (mass)
//!    ```
//!
//!    The bore runs into pure water: the solids of the lahar follow it at
//!    u1 < c, so both relations hold whatever the density of the inflow. At
//!    each saved state from t = 500 s, the bore position x_f is where the
//!    depth crosses (h0 + h1) / 2, with h1 and u1 averaged over the nodes
//!    40 to 100 m behind the toe (last node 1 cm above h0). The celerity
//!    comes from centred differences of x_f between saved states.
//!
//! Tolerances, about twice the bore errors measured when the example was
//! written:
//!
//! | Check | Measured | Tolerance |
//! |---|---|---|
//! | Exchange, mass (relative to the injected mass) | 1.6e-14 | 1e-13 |
//! | Exchange, solids (relative to the injected solids) | 3.3e-14 | 1e-13 |
//! | Celerity, abs(c - c_RH) / c_RH | 1.15e-3 | 2.5e-3 |
//! | Mass relation, abs(u1 - c (1 - h0 / h1)) / u1 | 2.11e-3 | 4e-3 |
//!
//! The exchange errors are rounding: their tolerances keep a margin of three
//! to six.
//!
//! The bore errors depend neither on dx (2.5 and 1.25 m), nor on dt
//! (0.05 s), nor on the integrator (RK4): they come from the averaging of
//! h1 behind a front that friction slowly reshapes. Averaged 100 to 300 m
//! behind the toe, h1 gives errors up to 6e-3 on the mass relation; without
//! friction in the lake, under 3e-3 in both windows.

use std::error::Error;

use nalgebra::DMatrix;
use oxiflow::{context::error::OxiflowError, mesh::Mesh};

#[path = "../rsc/models/lahar_flux.rs"]
mod lahar_flux;
#[path = "../rsc/models/lahar_outfall.rs"]
mod lahar_outfall;
#[path = "../rsc/models/lahar_physics.rs"]
mod lahar_physics;
#[path = "../rsc/models/lahar_source.rs"]
mod lahar_source;

use lahar_outfall::{
    channel_grid, channel_id, deposit_density, field, initial_states, injection, lake_grid,
    lake_id, physics, run, CHANNEL_LENGTH, CHANNEL_SLOPE_DEG, DEPOSIT_PACKING, INFLOW,
    INFLOW_SOLIDS, LAKE_DEPTH, LAKE_LENGTH, REFERENCE_RUN,
};
use lahar_physics::{COL_B, COL_M, COL_P, COL_S, G, H_MIN};

// ── Run settings ──────────────────────────────────────────────────────────

/// Spacing shared by both grids, and time step, of the reference run.
const DX: f64 = REFERENCE_RUN.dx;
const DT: f64 = REFERENCE_RUN.dt;

/// Bore measurement: start time, toe threshold above h0, and the window
/// behind the toe over which h1 and u1 are averaged.
const BORE_START: f64 = 500.0;
const TOE_THRESHOLD: f64 = 0.01;
const PLATEAU_WINDOW: (f64, f64) = (100.0, 40.0);

const TOL_MASS_EXCHANGE: f64 = 1e-13;
const TOL_SOLIDS_EXCHANGE: f64 = 1e-13;
const TOL_CELERITY: f64 = 2.5e-3;
const TOL_MASS_RELATION: f64 = 4e-3;

// ── Diagnostics ───────────────────────────────────────────────────────────

/// Change of a domain's inventory since the initial state, per unit width:
/// the column `col` plus `weight` times the bed, each node a cell of width dx.
fn inventory_change(state: &DMatrix<f64>, initial: &DMatrix<f64>, col: usize, weight: f64) -> f64 {
    (0..state.nrows())
        .map(|i| {
            (state[(i, col)] - initial[(i, col)])
                + weight * (state[(i, COL_B)] - initial[(i, COL_B)])
        })
        .sum::<f64>()
        * DX
}

/// Depth, velocity and solids fraction of one node.
fn primitives(state: &DMatrix<f64>, i: usize) -> Result<(f64, f64, f64), OxiflowError> {
    physics().primitives(state[(i, COL_M)], state[(i, COL_P)], state[(i, COL_S)])
}

/// Bore position, and depth and velocity behind it, in one lake state.
fn bore(lake: &DMatrix<f64>, xs: &[f64]) -> Result<(f64, f64, f64), Box<dyn Error>> {
    let mut depth = Vec::with_capacity(xs.len());
    let mut velocity = Vec::with_capacity(xs.len());
    for i in 0..xs.len() {
        let (h, u, _) = primitives(lake, i)?;
        depth.push(h);
        velocity.push(u);
    }
    let toe = (0..xs.len())
        .rev()
        .find(|&i| depth[i] - LAKE_DEPTH > TOE_THRESHOLD)
        .ok_or("no bore in the lake")?;
    let behind: Vec<usize> = (0..toe)
        .filter(|&i| xs[i] > xs[toe] - PLATEAU_WINDOW.0 && xs[i] < xs[toe] - PLATEAU_WINDOW.1)
        .collect();
    let n = behind.len() as f64;
    let h1 = behind.iter().map(|&i| depth[i]).sum::<f64>() / n;
    let u1 = behind.iter().map(|&i| velocity[i]).sum::<f64>() / n;
    let level = 0.5 * (LAKE_DEPTH + h1);
    let i = (0..xs.len() - 1)
        .rev()
        .find(|&i| depth[i] > level)
        .ok_or("the depth never crosses mid-bore")?;
    let front = xs[i] + (depth[i] - level) / (depth[i] - depth[i + 1]) * (xs[i + 1] - xs[i]);
    Ok((front, h1, u1))
}

fn check(quantity: &str, error: f64, tolerance: f64) -> Result<(), Box<dyn Error>> {
    if error <= tolerance {
        Ok(())
    } else {
        Err(format!("{quantity}: error {error:.3e} exceeds the tolerance {tolerance:.1e}").into())
    }
}

fn main() -> Result<(), Box<dyn Error>> {
    let (channel0, lake0) = initial_states(DX)?;
    println!(
        "Lahar of {INFLOW} m²/s (psi = {INFLOW_SOLIDS}) down {CHANNEL_LENGTH} m at {CHANNEL_SLOPE_DEG}°, \
         into a lake {LAKE_DEPTH} m deep and {LAKE_LENGTH} m long, dx = {DX} m, dt = {DT} s, explicit Euler"
    );
    let result = run(&REFERENCE_RUN, &channel0, &lake0)?;

    let channel_xs = channel_grid(DX)?;
    let source_rate: f64 = (0..channel_xs.n_dof())
        .map(|i| injection().density(channel_xs.coordinates(i)[0]))
        .sum::<f64>()
        * DX;
    let lake = lake_grid(DX)?;
    let lake_xs: Vec<f64> = (0..lake.n_dof()).map(|i| lake.coordinates(i)[0]).collect();

    // 1. Lake at rest until the channel outlet is wet.
    let mut wet_at = None;
    for (t, state) in result.times.iter().zip(&result.states) {
        let channel = field(state, &channel_id())?;
        if primitives(channel, channel.nrows() - 1)?.0 > H_MIN {
            wet_at = Some(*t);
            break;
        }
        if field(state, &lake_id())? != &lake0 {
            return Err(format!("the lake moved at t = {t} s, before the lahar reached it").into());
        }
    }
    let wet_at = wet_at.ok_or("the lahar never reached the lake")?;
    println!("  lake bitwise at rest until the outlet is wet, at t = {wet_at} s");

    // 2. Exchange, and outflow at the node before the outlet.
    let (mut mass_error, mut solids_error) = (0.0_f64, 0.0_f64);
    let rho_b = deposit_density();
    for (t, state) in result.times.iter().zip(&result.states).skip(1) {
        let channel = field(state, &channel_id())?;
        let lake_state = field(state, &lake_id())?;
        let injected_mass = physics().rho_bar(INFLOW_SOLIDS) * source_rate * t;
        let injected_solids = INFLOW_SOLIDS * source_rate * t;
        let lost_mass = injected_mass - inventory_change(channel, &channel0, COL_M, rho_b);
        let lost_solids =
            injected_solids - inventory_change(channel, &channel0, COL_S, DEPOSIT_PACKING);
        let gained_mass = inventory_change(lake_state, &lake0, COL_M, rho_b);
        let gained_solids = inventory_change(lake_state, &lake0, COL_S, DEPOSIT_PACKING);
        mass_error = mass_error.max((lost_mass - gained_mass).abs() / injected_mass);
        solids_error = solids_error.max((lost_solids - gained_solids).abs() / injected_solids);
        if (t % 100.0).abs() < 0.5 * DT {
            let (h, u, psi) = primitives(channel, channel.nrows() - 2)?;
            println!(
                "  t = {t:>3.0} s: outflow h = {h:.3} m, u = {u:.3} m/s, psi = {psi:.3}, Froude {:.2}; \
                 mass through the lip {gained_mass:.4e} kg/m, solids {gained_solids:.4e} m²",
                if h > H_MIN { u / (G * h).sqrt() } else { 0.0 }
            );
        }
    }
    println!(
        "  exchange: channel loss vs lake gain, relative to the injected amount: mass {mass_error:.2e}, solids {solids_error:.2e}"
    );

    // 3. Bore.
    let mut track = Vec::new();
    for (t, state) in result.times.iter().zip(&result.states) {
        if *t >= BORE_START - 0.5 * DT {
            track.push((*t, bore(field(state, &lake_id())?, &lake_xs)?));
        }
    }
    let (mut celerity_error, mut relation_error) = (0.0_f64, 0.0_f64);
    for w in track.windows(3) {
        let (t0, (x0, _, _)) = w[0];
        let (_, (_, h1, u1)) = w[1];
        let (t2, (x2, _, _)) = w[2];
        let c = (x2 - x0) / (t2 - t0);
        let c_rh = (G * h1 * (h1 + LAKE_DEPTH) / (2.0 * LAKE_DEPTH)).sqrt();
        celerity_error = celerity_error.max((c - c_rh).abs() / c_rh);
        relation_error = relation_error.max((u1 - c * (1.0 - LAKE_DEPTH / h1)).abs() / u1);
    }
    let (t_last, (x_last, h1_last, u1_last)) = *track.last().ok_or("no bore measured")?;
    println!(
        "  bore at t = {t_last} s: x_f = {x_last:.1} m, h1 = {h1_last:.3} m, u1 = {u1_last:.3} m/s \
         (long-wave speed sqrt(g h0) = {:.3} m/s)",
        (G * LAKE_DEPTH).sqrt()
    );
    println!(
        "  Rankine–Hugoniot over {} states from t = {BORE_START} s: celerity {celerity_error:.2e}, mass relation {relation_error:.2e}",
        track.len() - 2
    );

    check("exchange, mass", mass_error, TOL_MASS_EXCHANGE)?;
    check("exchange, solids", solids_error, TOL_SOLIDS_EXCHANGE)?;
    check("bore celerity", celerity_error, TOL_CELERITY)?;
    check("bore mass relation", relation_error, TOL_MASS_RELATION)?;
    println!("lahar_lake_wave: every check within its tolerance");
    Ok(())
}
