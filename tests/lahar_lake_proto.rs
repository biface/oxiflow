//! # Integration test: lahar–lake mass conservation (#40, DD-031)
//!
//! Companion to `examples/lahar_lake_proto.rs` — same physics, run through
//! `MultiDomainOrchestrator`, asserting the acceptance criteria from #40
//! that an example alone cannot enforce as a regression gate:
//! - Mass conservation within the expected splitting-error tolerance
//! - `CouplingOperator` invoked at every synchronised step (not zero,
//!   not double-counted)
//!
//! Structural-only validation (`Scenario::multi`, `with_coupling`, INV-3
//! composite keying) is already covered by `tests/coupling_proto.rs`
//! (v0.3.0) — this file specifically exercises the *time-stepping* path
//! that DD-031 added on top of it. Kept as a separate file rather than
//! extending `coupling_proto.rs`: its existing `MassTransferCoupling`
//! stub uses overwrite semantics tailored to its own structural-only
//! assertions, not mass conservation — changing it would risk breaking
//! an already-passing v0.3.0 regression baseline for an unrelated need.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

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

const N_NODES: usize = 5;
const LAMBDA: f64 = 0.2;

// ── Models (duplicated from examples/lahar_lake_proto.rs) ───────────────────
//
// Examples and integration tests are separate Cargo compilation units —
// they cannot share code without a dedicated (currently nonexistent)
// test-utils crate or module. Kept intentionally minimal; if a third
// consumer needs the same fixtures, that's the trigger to extract one.

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
}

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
}

/// Same as the example's `MassTransfer`, with an added invocation counter
/// for the "invoked at every step" assertion.
struct MassTransfer {
    lambda: f64,
    interface: Interface,
    calls: Arc<AtomicUsize>,
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
        self.calls.fetch_add(1, Ordering::SeqCst);

        let dt = ctx.time_step();
        let quantity = PhysicalQuantity::concentration();

        let source = states
            .get(interface.source(), &quantity)
            .ok_or_else(|| OxiflowError::PreconditionFailed {
                context: "MassTransfer".into(),
                message: "missing source field".into(),
            })?
            .as_scalar_field()?;
        let target = states
            .get(interface.target(), &quantity)
            .ok_or_else(|| OxiflowError::PreconditionFailed {
                context: "MassTransfer".into(),
                message: "missing target field".into(),
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

fn lahar_id() -> DomainId {
    DomainId::new("lahar")
}
fn lake_id() -> DomainId {
    DomainId::new("lake")
}

fn make_mesh() -> Box<dyn Mesh> {
    Box::new(UniformGrid1D::new(N_NODES, 0.0, 1.0).unwrap())
}

/// Builds the scenario and returns a shared counter for the coupling's
/// invocation count, observable after the orchestrator run (the operator
/// itself is moved into the `Scenario`, trait-object-erased — no downcast
/// available, hence the `Arc` handle kept on the side).
fn build_scenario() -> (Scenario, Arc<AtomicUsize>) {
    let lahar = Domain::new(
        lahar_id(),
        Box::new(GranularFlow { lambda: LAMBDA }),
        make_mesh(),
    );
    let lake = Domain::new(lake_id(), Box::new(ShallowWaterReceiver), make_mesh());
    let interface = Interface::new(lahar_id(), lake_id()).with_label("lahar-lake");
    let calls = Arc::new(AtomicUsize::new(0));
    let coupling = Box::new(MassTransfer {
        lambda: LAMBDA,
        interface,
        calls: calls.clone(),
    });
    let scenario = Scenario::multi(vec![lahar, lake])
        .unwrap()
        .with_coupling(coupling);
    (scenario, calls)
}

fn make_orchestrator() -> MultiDomainOrchestrator {
    let quantity = PhysicalQuantity::concentration();
    MultiDomainOrchestrator::new()
        .with_domain(lahar_id(), Box::new(ForwardEulerSolver), quantity.clone())
        .with_domain(lake_id(), Box::new(ForwardEulerSolver), quantity)
}

fn total_mass(state: &MultiDomainState, quantity: &PhysicalQuantity) -> f64 {
    let lahar_field = state
        .get(&lahar_id(), quantity)
        .unwrap()
        .as_scalar_field()
        .unwrap();
    let lake_field = state
        .get(&lake_id(), quantity)
        .unwrap()
        .as_scalar_field()
        .unwrap();
    lahar_field.sum() + lake_field.sum()
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[test]
fn j4a_orchestrated_run_completes() {
    let (scenario, _calls) = build_scenario();
    let orchestrator = make_orchestrator();
    let config = OrchestratorConfig::new(TimeConfiguration::new(
        20.0,
        StepControl::Fixed { dt: 0.01 },
    ));

    let result = orchestrator.run(&scenario, &config).unwrap();
    assert_eq!(result.n_steps, 2000);
}

#[test]
fn j4a_coupling_invoked_at_every_step() {
    let (scenario, calls) = build_scenario();
    let orchestrator = make_orchestrator();
    let config =
        OrchestratorConfig::new(TimeConfiguration::new(5.0, StepControl::Fixed { dt: 0.1 }));

    let result = orchestrator.run(&scenario, &config).unwrap();
    assert_eq!(result.n_steps, 50);
    assert_eq!(
        calls.load(Ordering::SeqCst),
        50,
        "expected exactly one coupling invocation per synchronised step"
    );
}

#[test]
fn j4a_mass_conservation_within_splitting_tolerance() {
    let (scenario, _calls) = build_scenario();
    let orchestrator = make_orchestrator();
    let quantity = PhysicalQuantity::concentration();
    let config = OrchestratorConfig::new(TimeConfiguration::new(
        20.0,
        StepControl::Fixed { dt: 0.01 },
    ));

    let result = orchestrator.run(&scenario, &config).unwrap();

    let initial_mass = total_mass(&result.states[0], &quantity);
    let final_mass = total_mass(result.states.last().unwrap(), &quantity);
    let drift = (final_mass - initial_mass).abs() / initial_mass;

    // Splitting error from applying domain physics and coupling as
    // separate sequential sub-steps -- not exact, see
    // MultiDomainOrchestrator's module docs. Tolerance not empirically
    // tuned against a real run (no compiler in the environment that wrote
    // this) -- tighten or loosen once it has actually been run.
    assert!(
        drift < 0.01,
        "mass conservation drift {drift:.3e} exceeds 1% tolerance"
    );
}

#[test]
fn j4a_lake_receives_nonzero_mass() {
    let (scenario, _calls) = build_scenario();
    let orchestrator = make_orchestrator();
    let quantity = PhysicalQuantity::concentration();
    let config =
        OrchestratorConfig::new(TimeConfiguration::new(5.0, StepControl::Fixed { dt: 0.1 }));

    let result = orchestrator.run(&scenario, &config).unwrap();
    let final_state = result.states.last().unwrap();
    let lake_field = final_state
        .get(&lake_id(), &quantity)
        .unwrap()
        .as_scalar_field()
        .unwrap();
    assert!(lake_field.iter().all(|v| *v > 0.0));
}

#[test]
fn j4a_lahar_loses_mass_over_time() {
    let (scenario, _calls) = build_scenario();
    let orchestrator = make_orchestrator();
    let quantity = PhysicalQuantity::concentration();
    let config =
        OrchestratorConfig::new(TimeConfiguration::new(5.0, StepControl::Fixed { dt: 0.1 }));

    let result = orchestrator.run(&scenario, &config).unwrap();

    let initial_lahar = result.states[0]
        .get(&lahar_id(), &quantity)
        .unwrap()
        .as_scalar_field()
        .unwrap()
        .sum();
    let final_lahar = result
        .states
        .last()
        .unwrap()
        .get(&lahar_id(), &quantity)
        .unwrap()
        .as_scalar_field()
        .unwrap()
        .sum();

    assert!(
        final_lahar < initial_lahar,
        "lahar mass should decrease: initial={initial_lahar}, final={final_lahar}"
    );
}
