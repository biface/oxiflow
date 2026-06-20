//! # Example: lahar–lake coupled prototype (#40, J3 exit criterion)
//!
//! Demonstrates `MultiDomainOrchestrator` (DD-031) driving two coupled
//! domains — a lahar (volcanic mudflow) losing mass and a lake receiving
//! it — through `CouplingOperator` (INV-3, DD-011).
//!
//! Run with: `cargo run --example lahar_lake_proto`
//!
//! ## Simplified physics
//!
//! Both domains are intentionally 0D-per-node ODEs, not real Bingham
//! viscoplastic rheology or shallow-water flux divergence — those need
//! `DiscreteOperator` (INV-2), which lands at J4b/v0.5.0. This prototype
//! validates the multi-domain *architecture* (INV-1, INV-3), not spatial
//! accuracy:
//!
//! - **Lahar** (granular flow stand-in): `dC/dt = -lambda * C` — mass
//!   leaving the flow at rate `lambda`.
//! - **Lake** (shallow-water stand-in): `dC/dt = 0` — passive receiver.
//! - **Coupling**: adds `lambda * C_lahar * dt` to the lake at every step
//!   — exactly the lahar's own loss rate, so total mass is conserved up to
//!   the splitting error inherent to applying domain physics and coupling
//!   as separate sequential steps (see `MultiDomainOrchestrator`'s module
//!   docs). Per `CouplingOperator`'s contract, only the lake (target)
//!   entry is written here — the lahar's loss happens through its own
//!   `compute_physics`, using the *same* `lambda`.

use nalgebra::DVector;
use oxiflow::{
    context::{
        compute::ComputeContext, error::OxiflowError, quantity::PhysicalQuantity,
        state::MultiDomainState, value::ContextValue, variable::ContextVariable,
    },
    coupling::{CouplingOperator, Interface},
    mesh::{Mesh, UniformGrid1D},
    model::traits::{PhysicalModel, RequiresContext},
    solver::{
        config::{StepControl, TimeConfiguration},
        methods::ForwardEulerSolver,
        orchestrator::{MultiDomainOrchestrator, OrchestratorConfig},
        scenario::{Domain, DomainId, Scenario},
    },
};

const N_NODES: usize = 10;
/// Shared rate [1/s] — both the lahar's intrinsic decay and the coupling's
/// transfer rate use this same value, which is what makes total mass
/// conservation hold (see module docs above).
const LAMBDA: f64 = 0.2;
const T_END: f64 = 20.0;
const DT: f64 = 0.01;

// ── Models ──────────────────────────────────────────────────────────────────

/// Granular flow stand-in — exponential mass loss, `dC/dt = -lambda * C`.
struct GranularFlow {
    lambda: f64,
}

impl RequiresContext for GranularFlow {
    fn required_variables(&self) -> Vec<ContextVariable> {
        vec![]
    }
}

impl PhysicalModel for GranularFlow {
    fn compute_physics(
        &self,
        state: &ContextValue,
        _ctx: &ComputeContext,
    ) -> Result<ContextValue, OxiflowError> {
        let c = state.as_scalar_field()?;
        Ok(ContextValue::ScalarField(c.map(|ci| -self.lambda * ci)))
    }

    fn initial_state(&self, mesh: &dyn Mesh) -> ContextValue {
        ContextValue::ScalarField(DVector::from_element(mesh.n_dof(), 1.0))
    }

    fn name(&self) -> &str {
        "granular_flow"
    }

    fn description(&self) -> Option<&str> {
        Some("Bingham-like granular flow stand-in — exponential decay, #40 multi-domain proto")
    }
}

/// Shallow-water stand-in — passive receiver, `dC/dt = 0`.
struct ShallowWaterReceiver;

impl RequiresContext for ShallowWaterReceiver {
    fn required_variables(&self) -> Vec<ContextVariable> {
        vec![]
    }
}

impl PhysicalModel for ShallowWaterReceiver {
    fn compute_physics(
        &self,
        state: &ContextValue,
        _ctx: &ComputeContext,
    ) -> Result<ContextValue, OxiflowError> {
        let c = state.as_scalar_field()?;
        Ok(ContextValue::ScalarField(DVector::zeros(c.len())))
    }

    fn initial_state(&self, mesh: &dyn Mesh) -> ContextValue {
        ContextValue::ScalarField(DVector::zeros(mesh.n_dof()))
    }

    fn name(&self) -> &str {
        "shallow_water_receiver"
    }

    fn description(&self) -> Option<&str> {
        Some("Shallow-water stand-in — passive receiver, #40 multi-domain proto")
    }
}

// ── Coupling ──────────────────────────────────────────────────────────────────

/// Transfers mass from the lahar to the lake at exactly the lahar's own
/// decay rate. Only writes the target entry — never modifies the source,
/// per `CouplingOperator`'s contract.
struct MassTransfer {
    lambda: f64,
    interface: Interface,
}

impl RequiresContext for MassTransfer {
    fn required_variables(&self) -> Vec<ContextVariable> {
        vec![]
    }
}

impl CouplingOperator for MassTransfer {
    fn apply(
        &self,
        states: &MultiDomainState,
        ctx: &ComputeContext,
        interface: &Interface,
    ) -> Result<MultiDomainState, OxiflowError> {
        let dt = ctx.time_step();
        let quantity = PhysicalQuantity::concentration();

        let source = states
            .get(interface.source(), &quantity)
            .ok_or_else(|| OxiflowError::PreconditionFailed {
                context: "MassTransfer".into(),
                message: format!(
                    "source domain '{}' has no Concentration field",
                    interface.source()
                ),
            })?
            .as_scalar_field()?;
        let target = states
            .get(interface.target(), &quantity)
            .ok_or_else(|| OxiflowError::PreconditionFailed {
                context: "MassTransfer".into(),
                message: format!(
                    "target domain '{}' has no Concentration field",
                    interface.target()
                ),
            })?
            .as_scalar_field()?;

        let flux = source.map(|c| self.lambda * c);
        let new_target: DVector<f64> = target.clone() + flux * dt;

        let mut result = states.clone();
        result.set(
            interface.target().clone(),
            quantity,
            ContextValue::ScalarField(new_target),
        );
        Ok(result)
    }

    fn interface(&self) -> &Interface {
        &self.interface
    }
}

// ── Scenario ──────────────────────────────────────────────────────────────────

fn lahar_id() -> DomainId {
    DomainId::new("lahar")
}
fn lake_id() -> DomainId {
    DomainId::new("lake")
}

fn build_scenario() -> Scenario {
    let lahar_mesh = Box::new(UniformGrid1D::new(N_NODES, 0.0, 1.0).unwrap());
    let lake_mesh = Box::new(UniformGrid1D::new(N_NODES, 0.0, 1.0).unwrap());

    let lahar = Domain::new(
        lahar_id(),
        Box::new(GranularFlow { lambda: LAMBDA }),
        lahar_mesh,
    );
    let lake = Domain::new(lake_id(), Box::new(ShallowWaterReceiver), lake_mesh);

    let interface = Interface::new(lahar_id(), lake_id()).with_label("lahar-lake");
    let coupling = Box::new(MassTransfer {
        lambda: LAMBDA,
        interface,
    });

    Scenario::multi(vec![lahar, lake])
        .expect("two domains is a valid scenario")
        .with_coupling(coupling)
}

fn total_mass(state: &MultiDomainState, quantity: &PhysicalQuantity) -> f64 {
    let lahar_field = state
        .get(&lahar_id(), quantity)
        .expect("lahar entry present")
        .as_scalar_field()
        .expect("lahar state is a ScalarField");
    let lake_field = state
        .get(&lake_id(), quantity)
        .expect("lake entry present")
        .as_scalar_field()
        .expect("lake state is a ScalarField");
    lahar_field.sum() + lake_field.sum()
}

// ── Entry point ───────────────────────────────────────────────────────────────

fn main() {
    let scenario = build_scenario();
    let quantity = PhysicalQuantity::concentration();

    let orchestrator = MultiDomainOrchestrator::new()
        .with_domain(lahar_id(), Box::new(ForwardEulerSolver), quantity.clone())
        .with_domain(lake_id(), Box::new(ForwardEulerSolver), quantity.clone());

    let config =
        OrchestratorConfig::new(TimeConfiguration::new(T_END, StepControl::Fixed { dt: DT }));

    let result = orchestrator
        .run(&scenario, &config)
        .expect("orchestrated run should succeed");

    let initial_mass = total_mass(&result.states[0], &quantity);
    let final_mass = total_mass(result.states.last().expect("at least one state"), &quantity);
    let relative_drift = (final_mass - initial_mass).abs() / initial_mass;

    println!("oxiflow — lahar-lake prototype (#40, DD-031)");
    println!("  steps run:      {}", result.n_steps);
    println!("  initial mass:   {initial_mass:.6}");
    println!("  final mass:     {final_mass:.6}");
    println!("  relative drift: {relative_drift:.3e}  (expected: splitting error, dt = {DT})");

    // NOTE: this 1% bound has not been empirically tuned against a real
    // run yet -- I have no compiler/runtime in the environment that
    // produced this file. Run it and tighten or loosen as the actual
    // observed drift dictates.
    assert!(
        relative_drift < 0.01,
        "mass conservation drift exceeds 1%: {relative_drift:.3e}"
    );
    println!("  mass conservation OK (within splitting-error tolerance)");
}
