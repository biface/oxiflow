//! Lahar–lake scenario on coupled grids: a lahar runs down a sloping channel
//! and over a lip into a lake at rest, each a shallow-layer domain, coupled
//! by the outflow of the channel's last cell (`OutfallExchange`).
//!
//! Single definition used by `examples/lahar_lake_wave.rs`, its plots
//! `examples/lahar_lake_wave_plot.rs`, its animation
//! `examples/lahar_lake_wave_simul.rs` and `benches/orchestrator_phase1.rs`,
//! each of which includes this file as a module with `#[path]`; the example
//! documents the physics, the coupling and the validation, and checks the
//! bore measured by [`bore`] against Rankine–Hugoniot, which the plots
//! draw. The grid spacing
//! `dx`, shared by both grids, is a parameter of every function that builds
//! a grid, a state or the coupling. The settings of the reference run, which
//! the example validates and its animation replays, are [`REFERENCE_RUN`];
//! the benchmark sets its own.
//!
//! Requires `rsc/models/lahar_physics.rs`, `rsc/models/lahar_source.rs` and
//! `rsc/models/lahar_flux.rs` as sibling modules named `lahar_physics`,
//! `lahar_source` and `lahar_flux`.
//!
//! With the `parallel` feature, the builders take an
//! `internal_threshold: Option<usize>` for the dispatch sites inside a
//! domain (source term, flux, bed-gradient calculator): `None` keeps each
//! site's default, `Some(t)` sets all of them to `t`.

use std::sync::Arc;

#[cfg(feature = "parallel")]
use super::lahar_flux::default_flux_parallel_threshold;
use super::lahar_flux::{EdgeTreatment, Edges, ShallowLayerFlux, ShallowLayerFluxWithEdges};
use super::lahar_physics::{
    ErosionDepositionParams, FrictionParams, LaharPhysics, SourceInjection, COL_B, COL_M, COL_P,
    COL_S, H_MIN,
};
#[cfg(feature = "parallel")]
use super::lahar_source::default_source_parallel_threshold;
use super::lahar_source::LaharSource;
use nalgebra::DMatrix;
use oxiflow::{
    context::{
        calculators::{FDGradientCalculator, FDScheme},
        error::OxiflowError,
        quantity::PhysicalQuantity,
        state::MultiDomainState,
        value::ContextValue,
        variable::ContextVariable,
    },
    coupling::{CouplingContext, CouplingOperator, Interface},
    mesh::{Mesh, UniformGrid1D},
    model::{CompositeModel, DiscretizedModel, PrescribedInitialState, RequiresContext},
    operators::limiters::Limiter,
    solver::{
        config::{StepControl, TimeConfiguration},
        methods::ForwardEulerSolver,
        orchestrator::{MultiDomainOrchestrator, MultiDomainSimulationResult, OrchestratorConfig},
        scenario::{Domain, DomainId, Scenario},
    },
};

// ── Physical and geometric constants ──────────────────────────────────────

const RHO_F: f64 = 1000.0;
/// Quartz.
const RHO_S: f64 = 2650.0;
const VISCOSITY: f64 = 1.0e-6;
const GRAIN_DIAMETER: f64 = 5.0e-5;
pub(crate) const DEPOSIT_PACKING: f64 = 0.6;
const CHEZY_DRAG: f64 = 0.01;

/// Inflow per unit width [m²/s] and its solids fraction.
pub(crate) const INFLOW: f64 = 20.0;
pub(crate) const INFLOW_SOLIDS: f64 = 0.2;
const SOURCE_STRIP: (f64, f64) = (50.0, 150.0);

pub(crate) const CHANNEL_LENGTH: f64 = 4000.0;
pub(crate) const CHANNEL_SLOPE_DEG: f64 = 3.0;
pub(crate) const LAKE_LENGTH: f64 = 3000.0;
pub(crate) const LAKE_DEPTH: f64 = 15.0;

pub(crate) fn physics() -> LaharPhysics {
    LaharPhysics {
        rho_f: RHO_F,
        rho_s: RHO_S,
    }
}

/// Bulk density of the deposit.
// The benchmark times the run alone and does not read the deposit.
#[allow(dead_code)]
pub(crate) fn deposit_density() -> f64 {
    RHO_F * (1.0 - DEPOSIT_PACKING) + RHO_S * DEPOSIT_PACKING
}

/// The quantity both domains carry.
fn layer() -> PhysicalQuantity {
    PhysicalQuantity::custom("shallow_layer")
}

pub(crate) fn channel_id() -> DomainId {
    DomainId::new("channel")
}

pub(crate) fn lake_id() -> DomainId {
    DomainId::new("lake")
}

/// Injection of the lahar over the source strip of the channel.
pub(crate) fn injection() -> SourceInjection {
    SourceInjection {
        rate: INFLOW,
        x_lo: SOURCE_STRIP.0,
        x_hi: SOURCE_STRIP.1,
        psi: INFLOW_SOLIDS,
    }
}

// ── Grids and initial states ──────────────────────────────────────────────

/// Number of nodes of a domain of `length` at spacing `dx`.
fn nodes(length: f64, dx: f64) -> usize {
    (length / dx).round() as usize + 1
}

pub(crate) fn channel_grid(dx: f64) -> Result<UniformGrid1D, OxiflowError> {
    UniformGrid1D::new(nodes(CHANNEL_LENGTH, dx), 0.0, CHANNEL_LENGTH)
}

pub(crate) fn lake_grid(dx: f64) -> Result<UniformGrid1D, OxiflowError> {
    UniformGrid1D::new(
        nodes(LAKE_LENGTH, dx),
        CHANNEL_LENGTH,
        CHANNEL_LENGTH + LAKE_LENGTH,
    )
}

/// Initial states: a dry channel on its slope, a lake at rest.
pub(crate) fn initial_states(dx: f64) -> Result<(DMatrix<f64>, DMatrix<f64>), OxiflowError> {
    let channel = channel_grid(dx)?;
    let tan = CHANNEL_SLOPE_DEG.to_radians().tan();
    let mut dry = DMatrix::zeros(channel.n_dof(), 4);
    for i in 0..channel.n_dof() {
        dry[(i, COL_B)] = (CHANNEL_LENGTH - channel.coordinates(i)[0]) * tan;
    }
    let mut lake = DMatrix::zeros(nodes(LAKE_LENGTH, dx), 4);
    lake.column_mut(COL_M).fill(RHO_F * LAKE_DEPTH);
    Ok((dry, lake))
}

// ── Domain models ─────────────────────────────────────────────────────────

/// The shallow-layer model of one domain, source term then flux, started
/// from `initial`.
fn layer_model(
    grid: &UniformGrid1D,
    injection: SourceInjection,
    initial: DMatrix<f64>,
    edges: Edges,
    name: &str,
    #[cfg(feature = "parallel")] internal_threshold: Option<usize>,
) -> Result<PrescribedInitialState, OxiflowError> {
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
        0.0, // unused: the initial state is prescribed.
        #[cfg(feature = "parallel")]
        internal_threshold.unwrap_or_else(default_source_parallel_threshold),
    );
    let flux = ShallowLayerFluxWithEdges {
        flux: ShallowLayerFlux {
            physics: physics(),
            limiter: Limiter::MinMod,
            #[cfg(feature = "parallel")]
            parallel_threshold: internal_threshold.unwrap_or_else(default_flux_parallel_threshold),
        },
        edges,
    };
    let flux_grid = Arc::new(UniformGrid1D::new(
        grid.n_dof(),
        grid.coordinates(0)[0],
        grid.coordinates(grid.n_dof() - 1)[0],
    )?);
    let discretized = DiscretizedModel::new(Arc::new(flux), flux_grid, name);
    let model = CompositeModel::new(vec![Box::new(source), Box::new(discretized)], name)?;
    Ok(PrescribedInitialState::new(
        Box::new(model),
        ContextValue::VectorField(initial),
    ))
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
/// before the other has read it (gain first, see the documentation of
/// `examples/lahar_lake_wave.rs`).
struct OutfallExchange {
    interface: Interface,
    side: LipSide,
    /// Cell width of both grids.
    dx: f64,
}

impl RequiresContext for OutfallExchange {
    fn required_variables(&self) -> Vec<ContextVariable> {
        vec![]
    }
}

/// The shallow-layer state of `domain`.
pub(crate) fn field<'a>(
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
        ctx: &CouplingContext,
        interface: &Interface,
    ) -> Result<MultiDomainState, OxiflowError> {
        let flux = outfall_flux(states)?;
        let target = interface.target();
        let mut updated = field(states, target)?.clone();
        let (row, sign) = match self.side {
            LipSide::LakeInlet => (0, 1.0),
            LipSide::ChannelOutlet => (updated.nrows() - 1, -1.0),
        };
        let scale = ctx.time_step() / self.dx;
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

// ── Scenario, orchestrator, calculator ────────────────────────────────────

/// The two domains at spacing `dx`, started from `channel0` and `lake0`,
/// and the two halves of the outfall coupling, gain first.
pub(crate) fn scenario(
    dx: f64,
    channel0: &DMatrix<f64>,
    lake0: &DMatrix<f64>,
    #[cfg(feature = "parallel")] internal_threshold: Option<usize>,
) -> Result<Scenario, OxiflowError> {
    let channel_model = layer_model(
        &channel_grid(dx)?,
        injection(),
        channel0.clone(),
        Edges {
            left: EdgeTreatment::Wall,
            right: EdgeTreatment::Exchanged,
        },
        "channel",
        #[cfg(feature = "parallel")]
        internal_threshold,
    )?;
    let still_water = SourceInjection {
        rate: 0.0,
        ..injection()
    };
    let lake_model = layer_model(
        &lake_grid(dx)?,
        still_water,
        lake0.clone(),
        Edges {
            left: EdgeTreatment::Wall,
            right: EdgeTreatment::Wall,
        },
        "lake",
        #[cfg(feature = "parallel")]
        internal_threshold,
    )?;

    Ok(Scenario::multi(vec![
        Domain::new(
            channel_id(),
            Box::new(channel_model),
            Box::new(channel_grid(dx)?),
        ),
        Domain::new(lake_id(), Box::new(lake_model), Box::new(lake_grid(dx)?)),
    ])?
    .with_coupling(Box::new(OutfallExchange {
        interface: Interface::new(channel_id(), lake_id()).with_label("outfall, lake gain"),
        side: LipSide::LakeInlet,
        dx,
    }))
    .with_coupling(Box::new(OutfallExchange {
        interface: Interface::new(lake_id(), channel_id()).with_label("outfall, channel loss"),
        side: LipSide::ChannelOutlet,
        dx,
    })))
}

/// Explicit Euler on both domains.
pub(crate) fn orchestrator() -> MultiDomainOrchestrator {
    MultiDomainOrchestrator::new()
        .with_domain(channel_id(), Box::new(ForwardEulerSolver), layer())
        .with_domain(lake_id(), Box::new(ForwardEulerSolver), layer())
}

/// Gradient of the bed, built on the channel grid and shared by both
/// domains: the orchestrator holds one calculator chain, and the two grids
/// have the same spacing.
pub(crate) fn bed_gradient(
    dx: f64,
    #[cfg(feature = "parallel")] internal_threshold: Option<usize>,
) -> Result<FDGradientCalculator, OxiflowError> {
    let calculator = FDGradientCalculator::new(
        Arc::new(channel_grid(dx)?),
        0,
        Some(COL_B),
        FDScheme::Central,
    );
    #[cfg(feature = "parallel")]
    let calculator = match internal_threshold {
        Some(threshold) => calculator.with_parallel_threshold(threshold),
        None => calculator,
    };
    Ok(calculator)
}

// ── Reference run ─────────────────────────────────────────────────────────

/// Settings of a run: grid spacing, time step, end time and saving.
// The benchmark sets its own runs and reads none of this section.
#[allow(dead_code)]
pub(crate) struct RunSettings {
    /// Spacing shared by both grids.
    pub(crate) dx: f64,
    pub(crate) dt: f64,
    pub(crate) end_time: f64,
    /// Steps between saved states.
    pub(crate) save_every: usize,
}

/// The run `examples/lahar_lake_wave.rs` validates: dx = 5 m, dt = 0.1 s
/// (largest wave speed times dt / dx about 0.3), a state every 5 s up to
/// t = 600 s, before the bore reaches the far shore.
#[allow(dead_code)]
pub(crate) const REFERENCE_RUN: RunSettings = RunSettings {
    dx: 5.0,
    dt: 0.1,
    end_time: 600.0,
    save_every: 50,
};

/// Runs the scenario with `settings`, from `channel0` and `lake0`, with
/// explicit Euler on both domains and the shared bed gradient.
#[allow(dead_code)]
pub(crate) fn run(
    settings: &RunSettings,
    channel0: &DMatrix<f64>,
    lake0: &DMatrix<f64>,
) -> Result<MultiDomainSimulationResult, OxiflowError> {
    let scenario = scenario(
        settings.dx,
        channel0,
        lake0,
        #[cfg(feature = "parallel")]
        None,
    )?;
    let config = OrchestratorConfig::new(
        TimeConfiguration::new(settings.end_time, StepControl::Fixed { dt: settings.dt })
            .saving_every(settings.save_every),
    )
    .with_calculator(Box::new(bed_gradient(
        settings.dx,
        #[cfg(feature = "parallel")]
        None,
    )?));
    orchestrator().run(&scenario, &config)
}

// ── Bore measurement ──────────────────────────────────────────────────────

/// Toe of the bore: the last lake node more than this above the depth at
/// rest, in metres.
// The benchmark measures no bore and reads none of this section.
#[allow(dead_code)]
pub(crate) const TOE_THRESHOLD: f64 = 0.01;

/// Window behind the toe over which the depth h1 and the velocity u1 of the
/// bore are averaged, as distances from the toe: from 100 m to 40 m.
#[allow(dead_code)]
pub(crate) const PLATEAU_WINDOW: (f64, f64) = (100.0, 40.0);

/// Depth, velocity and solids fraction of node `i` of a domain state.
#[allow(dead_code)]
pub(crate) fn primitives(state: &DMatrix<f64>, i: usize) -> Result<(f64, f64, f64), OxiflowError> {
    physics().primitives(state[(i, COL_M)], state[(i, COL_P)], state[(i, COL_S)])
}

/// Bore in one lake state, at node positions `xs`: its position x_f, where
/// the depth crosses (h0 + h1) / 2, and the depth h1 and velocity u1 behind
/// it, averaged over [`PLATEAU_WINDOW`] behind the toe.
///
/// Fails when the lake holds no bore yet: no node above the toe threshold,
/// or a toe too close to the inlet for the window to hold a node.
#[allow(dead_code)]
pub(crate) fn bore(lake: &DMatrix<f64>, xs: &[f64]) -> Result<(f64, f64, f64), OxiflowError> {
    let no_bore = |message: &str| OxiflowError::PreconditionFailed {
        context: "lahar_lake_wave",
        message: message.to_string(),
    };
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
        .ok_or_else(|| no_bore("no bore in the lake"))?;
    let behind: Vec<usize> = (0..toe)
        .filter(|&i| xs[i] > xs[toe] - PLATEAU_WINDOW.0 && xs[i] < xs[toe] - PLATEAU_WINDOW.1)
        .collect();
    if behind.is_empty() {
        return Err(no_bore("the bore is too close to the inlet to be measured"));
    }
    let n = behind.len() as f64;
    let h1 = behind.iter().map(|&i| depth[i]).sum::<f64>() / n;
    let u1 = behind.iter().map(|&i| velocity[i]).sum::<f64>() / n;
    let level = 0.5 * (LAKE_DEPTH + h1);
    let i = (0..xs.len() - 1)
        .rev()
        .find(|&i| depth[i] > level)
        .ok_or_else(|| no_bore("the depth never crosses mid-bore"))?;
    let front = xs[i] + (depth[i] - level) / (depth[i] - depth[i + 1]) * (xs[i + 1] - xs[i]);
    Ok((front, h1, u1))
}
