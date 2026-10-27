//! # Reference example: lahar–lake wave generation on coupled grids (#180)
//!
//! A lahar runs down a 4 km channel and pours over a lip into a 15 m deep
//! lake. The inflow raises a bore that runs across the lake. Channel and
//! lake are two domains on their own grids, advanced together by
//! `MultiDomainOrchestrator` and coupled only through `CouplingOperator`s
//! (INV-3). The example checks that the lake stays exactly at rest until the
//! lahar reaches it, that the coupling moves mass and solids from one grid to
//! the other without loss, and that the bore obeys the Rankine–Hugoniot jump
//! conditions.
//!
//! Run with: `cargo run --example lahar_lake_wave`. The example exits with an
//! error if a check exceeds its tolerance.
//!
//! ## Physics
//!
//! Both domains use the 1D multi-field lahar model of DD-032 (Woodhouse,
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
use std::sync::Arc;

use nalgebra::DMatrix;
use oxiflow::{
    context::{
        calculators::{FDGradientCalculator, FDScheme},
        compute::ComputeContext,
        error::OxiflowError,
        quantity::PhysicalQuantity,
        state::MultiDomainState,
        value::ContextValue,
        variable::ContextVariable,
    },
    coupling::{CouplingOperator, Interface},
    mesh::{Mesh, UniformGrid1D},
    model::{CompositeModel, DiscretizedModel, PhysicalModel, RequiresContext},
    operators::limiters::Limiter,
    solver::{
        config::{StepControl, TimeConfiguration},
        methods::ForwardEulerSolver,
        orchestrator::{MultiDomainOrchestrator, MultiDomainSimulationResult, OrchestratorConfig},
        scenario::{Domain, DomainId, Scenario},
    },
};

#[path = "../rsc/models/lahar_flux.rs"]
mod lahar_flux;
#[path = "../rsc/models/lahar_physics.rs"]
mod lahar_physics;
#[path = "../rsc/models/lahar_source.rs"]
mod lahar_source;

#[cfg(feature = "parallel")]
use lahar_flux::default_flux_parallel_threshold;
use lahar_flux::{EdgeTreatment, Edges, ShallowLayerFlux, ShallowLayerFluxWithEdges};
use lahar_physics::{
    ErosionDepositionParams, FrictionParams, LaharPhysics, SourceInjection, COL_B, COL_M, COL_P,
    COL_S, G, H_MIN,
};
#[cfg(feature = "parallel")]
use lahar_source::default_source_parallel_threshold;
use lahar_source::LaharSource;

// ── Scenario ──────────────────────────────────────────────────────────────

const RHO_F: f64 = 1000.0;
/// Quartz.
const RHO_S: f64 = 2650.0;
const VISCOSITY: f64 = 1.0e-6;
const GRAIN_DIAMETER: f64 = 5.0e-5;
const DEPOSIT_PACKING: f64 = 0.6;
const CHEZY_DRAG: f64 = 0.01;

/// Inflow per unit width [m²/s] and its solids fraction.
const INFLOW: f64 = 20.0;
const INFLOW_SOLIDS: f64 = 0.2;
const SOURCE_STRIP: (f64, f64) = (50.0, 150.0);

const CHANNEL_LENGTH: f64 = 4000.0;
const CHANNEL_SLOPE_DEG: f64 = 3.0;
const LAKE_LENGTH: f64 = 3000.0;
const LAKE_DEPTH: f64 = 15.0;

/// Spacing shared by both grids (see the module documentation).
const DX: f64 = 5.0;
const DT: f64 = 0.1;
const END_TIME: f64 = 600.0;
/// Steps between saved states (5 s).
const SAVE_EVERY: usize = 50;

/// Bore measurement: start time, toe threshold above h0, and the window
/// behind the toe over which h1 and u1 are averaged.
const BORE_START: f64 = 500.0;
const TOE_THRESHOLD: f64 = 0.01;
const PLATEAU_WINDOW: (f64, f64) = (100.0, 40.0);

const TOL_MASS_EXCHANGE: f64 = 1e-13;
const TOL_SOLIDS_EXCHANGE: f64 = 1e-13;
const TOL_CELERITY: f64 = 2.5e-3;
const TOL_MASS_RELATION: f64 = 4e-3;

fn physics() -> LaharPhysics {
    LaharPhysics {
        rho_f: RHO_F,
        rho_s: RHO_S,
    }
}

/// Bulk density of the deposit.
fn deposit_density() -> f64 {
    RHO_F * (1.0 - DEPOSIT_PACKING) + RHO_S * DEPOSIT_PACKING
}

fn layer() -> PhysicalQuantity {
    PhysicalQuantity::custom("shallow_layer")
}

fn channel_id() -> DomainId {
    DomainId::new("channel")
}

fn lake_id() -> DomainId {
    DomainId::new("lake")
}

fn channel_grid() -> Result<UniformGrid1D, OxiflowError> {
    UniformGrid1D::new(nodes(CHANNEL_LENGTH), 0.0, CHANNEL_LENGTH)
}

fn lake_grid() -> Result<UniformGrid1D, OxiflowError> {
    UniformGrid1D::new(
        nodes(LAKE_LENGTH),
        CHANNEL_LENGTH,
        CHANNEL_LENGTH + LAKE_LENGTH,
    )
}

fn nodes(length: f64) -> usize {
    (length / DX).round() as usize + 1
}

fn injection() -> SourceInjection {
    SourceInjection {
        rate: INFLOW,
        x_lo: SOURCE_STRIP.0,
        x_hi: SOURCE_STRIP.1,
        psi: INFLOW_SOLIDS,
    }
}

// ── Models ────────────────────────────────────────────────────────────────

/// The source term of the lahar model with an explicit initial state: a
/// sloping dry channel or a lake at rest. `CompositeModel` takes its initial
/// state from its first operator.
struct SourceWithInitialState {
    source: LaharSource,
    initial: DMatrix<f64>,
}

impl RequiresContext for SourceWithInitialState {
    fn required_variables(&self) -> Vec<ContextVariable> {
        self.source.required_variables()
    }
}

impl PhysicalModel for SourceWithInitialState {
    fn compute_physics(
        &self,
        state: &ContextValue,
        ctx: &ComputeContext,
    ) -> Result<ContextValue, OxiflowError> {
        self.source.compute_physics(state, ctx)
    }

    fn initial_state(&self, _mesh: &dyn Mesh) -> ContextValue {
        ContextValue::VectorField(self.initial.clone())
    }

    fn name(&self) -> &str {
        self.source.name()
    }
}

fn layer_model(
    grid: &UniformGrid1D,
    injection: SourceInjection,
    initial: DMatrix<f64>,
    edges: Edges,
    name: &str,
) -> Result<CompositeModel, OxiflowError> {
    let friction = FrictionParams {
        cd: CHEZY_DRAG,
        tan_d1: 0.0,
        tan_d2: 0.0,
        tan_d3: 0.0,
        length_l: 1.0, // unused: the granular branch is zero.
        beta: 0.136,
        gamma: 1.0e-3,
        alpha: 5.0,
        psi0: 1.0,
    };
    let erosion = ErosionDepositionParams {
        epsilon: 0.0,
        nu: VISCOSITY,
        grain_d: GRAIN_DIAMETER,
        psi_b: DEPOSIT_PACKING,
    };
    let source = LaharSource::new(
        grid,
        physics(),
        friction,
        erosion,
        injection,
        0.0, // unused: the initial state is given.
        #[cfg(feature = "parallel")]
        default_source_parallel_threshold(),
    );
    let flux = ShallowLayerFluxWithEdges {
        flux: ShallowLayerFlux {
            physics: physics(),
            limiter: Limiter::MinMod,
            #[cfg(feature = "parallel")]
            parallel_threshold: default_flux_parallel_threshold(),
        },
        edges,
    };
    let flux_grid = Arc::new(UniformGrid1D::new(
        grid.n_dof(),
        grid.coordinates(0)[0],
        grid.coordinates(grid.n_dof() - 1)[0],
    )?);
    let discretized = DiscretizedModel::new(Arc::new(flux), flux_grid, name);
    CompositeModel::new(
        vec![
            Box::new(SourceWithInitialState { source, initial }),
            Box::new(discretized),
        ],
        name,
    )
}

// ── Coupling ──────────────────────────────────────────────────────────────

/// Which side of the lip an [`OutfallExchange`] writes.
#[derive(Debug, Clone, Copy)]
enum LipSide {
    /// First cell of the lake: gains the outflow.
    LakeInlet,
    /// Last cell of the channel: loses it.
    ChannelOutlet,
}

/// Carries the outflow of the channel over the lip. Both instances compute
/// the flux from the channel state alone, which neither of them writes
/// before the other has read it (gain first, see the module documentation).
struct OutfallExchange {
    interface: Interface,
    side: LipSide,
}

impl RequiresContext for OutfallExchange {
    fn required_variables(&self) -> Vec<ContextVariable> {
        vec![]
    }
}

fn field<'a>(
    states: &'a MultiDomainState,
    domain: &DomainId,
) -> Result<&'a DMatrix<f64>, OxiflowError> {
    states
        .get(domain, &layer())
        .ok_or_else(|| OxiflowError::PreconditionFailed {
            context: "lahar_lake_wave",
            message: format!("domain '{domain}' has no shallow-layer state"),
        })?
        .as_vector_field()
}

/// Physical flux of the last channel cell, zero while it is dry.
fn outfall_flux(states: &MultiDomainState) -> Result<[f64; 3], OxiflowError> {
    let channel = field(states, &channel_id())?;
    let last = channel.nrows() - 1;
    let (m, p, s) = (
        channel[(last, COL_M)],
        channel[(last, COL_P)],
        channel[(last, COL_S)],
    );
    let (h, u, psi) = physics().primitives(m, p, s)?;
    if h <= H_MIN {
        return Ok([0.0; 3]);
    }
    Ok(physics().physical_flux(m, p, s, h, u, psi))
}

impl CouplingOperator for OutfallExchange {
    fn apply(
        &self,
        states: &MultiDomainState,
        ctx: &ComputeContext,
        interface: &Interface,
    ) -> Result<MultiDomainState, OxiflowError> {
        let flux = outfall_flux(states)?;
        let target = interface.target();
        let mut updated = field(states, target)?.clone();
        let (row, sign) = match self.side {
            LipSide::LakeInlet => (0, 1.0),
            LipSide::ChannelOutlet => (updated.nrows() - 1, -1.0),
        };
        let scale = ctx.time_step() / DX;
        for (k, col) in [COL_M, COL_P, COL_S].into_iter().enumerate() {
            updated[(row, col)] += sign * (scale * flux[k]);
        }
        let mut result = states.clone();
        result.set(target.clone(), layer(), ContextValue::VectorField(updated));
        Ok(result)
    }

    fn interface(&self) -> &Interface {
        &self.interface
    }
}

// ── Run ───────────────────────────────────────────────────────────────────

/// Initial states: a dry channel on its slope, a lake at rest.
fn initial_states() -> Result<(DMatrix<f64>, DMatrix<f64>), OxiflowError> {
    let channel = channel_grid()?;
    let tan = CHANNEL_SLOPE_DEG.to_radians().tan();
    let mut dry = DMatrix::zeros(channel.n_dof(), 4);
    for i in 0..channel.n_dof() {
        dry[(i, COL_B)] = (CHANNEL_LENGTH - channel.coordinates(i)[0]) * tan;
    }
    let mut lake = DMatrix::zeros(nodes(LAKE_LENGTH), 4);
    lake.column_mut(COL_M).fill(RHO_F * LAKE_DEPTH);
    Ok((dry, lake))
}

fn run(
    channel0: &DMatrix<f64>,
    lake0: &DMatrix<f64>,
) -> Result<MultiDomainSimulationResult, OxiflowError> {
    let channel_model = layer_model(
        &channel_grid()?,
        injection(),
        channel0.clone(),
        Edges {
            left: EdgeTreatment::Wall,
            right: EdgeTreatment::Exchanged,
        },
        "channel",
    )?;
    let still_water = SourceInjection {
        rate: 0.0,
        ..injection()
    };
    let lake_model = layer_model(
        &lake_grid()?,
        still_water,
        lake0.clone(),
        Edges {
            left: EdgeTreatment::Wall,
            right: EdgeTreatment::Wall,
        },
        "lake",
    )?;

    let scenario = Scenario::multi(vec![
        Domain::new(
            channel_id(),
            Box::new(channel_model),
            Box::new(channel_grid()?),
        ),
        Domain::new(lake_id(), Box::new(lake_model), Box::new(lake_grid()?)),
    ])?
    .with_coupling(Box::new(OutfallExchange {
        interface: Interface::new(channel_id(), lake_id()).with_label("outfall, lake gain"),
        side: LipSide::LakeInlet,
    }))
    .with_coupling(Box::new(OutfallExchange {
        interface: Interface::new(lake_id(), channel_id()).with_label("outfall, channel loss"),
        side: LipSide::ChannelOutlet,
    }));

    let orchestrator = MultiDomainOrchestrator::new()
        .with_domain(channel_id(), Box::new(ForwardEulerSolver), layer())
        .with_domain(lake_id(), Box::new(ForwardEulerSolver), layer());
    let config = OrchestratorConfig::new(
        TimeConfiguration::new(END_TIME, StepControl::Fixed { dt: DT }).saving_every(SAVE_EVERY),
    )
    .with_calculator(Box::new(FDGradientCalculator::new(
        Arc::new(channel_grid()?),
        0,
        Some(COL_B),
        FDScheme::Central,
    )));
    orchestrator.run(&scenario, &config)
}

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
    let (channel0, lake0) = initial_states()?;
    println!(
        "Lahar of {INFLOW} m²/s (psi = {INFLOW_SOLIDS}) down {CHANNEL_LENGTH} m at {CHANNEL_SLOPE_DEG}°, \
         into a lake {LAKE_DEPTH} m deep and {LAKE_LENGTH} m long, dx = {DX} m, dt = {DT} s, explicit Euler"
    );
    let result = run(&channel0, &lake0)?;

    let channel_xs = channel_grid()?;
    let source_rate: f64 = (0..channel_xs.n_dof())
        .map(|i| injection().density(channel_xs.coordinates(i)[0]))
        .sum::<f64>()
        * DX;
    let lake = lake_grid()?;
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
