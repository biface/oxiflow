//! # Coupling pipeline — lahar–lake proto (J3 exit criterion)
//!
//! This integration test validates the multi-domain coupling architecture:
//!
//! ```text
//! Scenario::multi() + with_coupling()
//!   → context_requirements() aggregates coupling variables
//!   → CouplingOperator::apply() exchanges state across Interface
//!   → MultiDomainState correctly keyed by (DomainId, PhysicalQuantity)
//! ```
//!
//! ## Physical model (stub)
//!
//! Two domains — `lahar` and `lake` — connected by a mass-transfer interface.
//! Physics are intentionally simplified (0D per-node ODE) to isolate the
//! structural validation from spatial accuracy concerns.
//!
//! **Lahar domain** — exponential decay:
//! $$\frac{\partial c}{\partial t} = -\lambda \cdot c$$
//!
//! **Lake domain** — passive receiver (zero source term):
//! $$\frac{\partial c}{\partial t} = 0$$
//!
//! **MassTransferCoupling** — transfers a fraction `alpha` of the lahar
//! concentration field to the lake at each coupling call.
//!
//! ## Acceptance criteria (DD-011, INV-3)
//!
//! - `Scenario::multi()` accepts two domains without error
//! - `with_coupling()` registers the operator and its interface
//! - `context_requirements()` aggregates variables from both domains and coupling
//! - `CouplingOperator::apply()` returns a valid `MultiDomainState`
//! - Composite key `(DomainId, PhysicalQuantity)` correctly distinguishes domains
//! - Source domain entries are not mutated by the coupling operator

use nalgebra::DVector;
use oxiflow::{
    context::{
        compute::ComputeContext, error::OxiflowError, quantity::PhysicalQuantity,
        state::MultiDomainState, value::ContextValue, variable::ContextVariable,
    },
    coupling::{CouplingOperator, Interface},
    mesh::{Mesh, UniformGrid1D},
    model::traits::{PhysicalModel, RequiresContext},
    solver::scenario::{Domain, DomainId, Scenario},
};

// ── Models ────────────────────────────────────────────────────────────────────

/// Lahar domain — exponential concentration decay.
struct LaharModel {
    /// First-order decay rate [1/s].
    lambda: f64,
}

impl RequiresContext for LaharModel {
    fn required_variables(&self) -> Vec<ContextVariable> {
        vec![ContextVariable::Time]
    }
}

impl PhysicalModel for LaharModel {
    fn compute_physics(
        &self,
        state: &ContextValue,
        _ctx: &ComputeContext,
    ) -> Result<ContextValue, OxiflowError> {
        let c = state.as_scalar_field()?;
        let dc_dt = c.map(|ci| -self.lambda * ci);
        Ok(ContextValue::ScalarField(dc_dt))
    }

    fn initial_state(&self, mesh: &dyn Mesh) -> ContextValue {
        // Start at uniform concentration of 1.0
        ContextValue::ScalarField(DVector::from_element(mesh.n_dof(), 1.0))
    }

    fn name(&self) -> &str {
        "lahar_decay"
    }

    fn description(&self) -> Option<&str> {
        Some("Stub lahar model — exponential decay, J3 coupling proto")
    }
}

/// Lake domain — passive receiver, zero source term.
struct LakeModel;

impl RequiresContext for LakeModel {
    fn required_variables(&self) -> Vec<ContextVariable> {
        vec![]
    }
}

impl PhysicalModel for LakeModel {
    fn compute_physics(
        &self,
        state: &ContextValue,
        _ctx: &ComputeContext,
    ) -> Result<ContextValue, OxiflowError> {
        let c = state.as_scalar_field()?;
        // Passive: dc/dt = 0
        Ok(ContextValue::ScalarField(DVector::zeros(c.len())))
    }

    fn initial_state(&self, mesh: &dyn Mesh) -> ContextValue {
        ContextValue::ScalarField(DVector::zeros(mesh.n_dof()))
    }

    fn name(&self) -> &str {
        "lake_passive"
    }

    fn description(&self) -> Option<&str> {
        Some("Stub lake model — passive receiver, J3 coupling proto")
    }
}

// ── Coupling operator ─────────────────────────────────────────────────────────

/// Mass-transfer coupling — transfers fraction `alpha` of the lahar
/// concentration to the lake across the shared interface.
///
/// Reads:  `(lahar, Concentration { component: 0 })`
/// Writes: `(lake,  Concentration { component: 0 })` ← `alpha * lahar_field`
struct MassTransferCoupling {
    /// Transfer fraction (0.0–1.0).
    alpha: f64,
    interface: Interface,
}

impl RequiresContext for MassTransferCoupling {
    fn required_variables(&self) -> Vec<ContextVariable> {
        vec![]
    }
}

impl CouplingOperator for MassTransferCoupling {
    fn apply(
        &self,
        states: &MultiDomainState,
        _ctx: &ComputeContext,
        interface: &Interface,
    ) -> Result<MultiDomainState, OxiflowError> {
        let source_id = interface.source();
        let target_id = interface.target();
        let quantity = PhysicalQuantity::concentration();

        // Read source field
        let source_field = states
            .get(source_id, &quantity)
            .ok_or_else(|| OxiflowError::PreconditionFailed {
                context: "MassTransferCoupling".into(),
                message: format!("source domain '{}' has no Concentration field", source_id),
            })?
            .as_scalar_field()?;

        // Compute transferred field: alpha * source
        let transferred = source_field.map(|ci| self.alpha * ci);

        // Build updated state: start from existing state, set target entry
        let mut result = states.clone();
        result.set(
            target_id.clone(),
            quantity,
            ContextValue::ScalarField(transferred),
        );
        Ok(result)
    }

    fn interface(&self) -> &Interface {
        &self.interface
    }
}

// ── Parameters ────────────────────────────────────────────────────────────────

const N_NODES: usize = 10;
const LAMBDA: f64 = 0.1; // [1/s]
const ALPHA: f64 = 0.3; // transfer fraction

fn lahar_id() -> DomainId {
    DomainId::new("lahar")
}
fn lake_id() -> DomainId {
    DomainId::new("lake")
}

fn make_scenario() -> Scenario {
    let lahar_mesh = Box::new(UniformGrid1D::new(N_NODES, 0.0, 1.0).unwrap());
    let lake_mesh = Box::new(UniformGrid1D::new(N_NODES, 0.0, 1.0).unwrap());

    let lahar_domain = Domain::new(
        lahar_id(),
        Box::new(LaharModel { lambda: LAMBDA }),
        lahar_mesh,
    );
    let lake_domain = Domain::new(lake_id(), Box::new(LakeModel), lake_mesh);

    let interface = Interface::new(lahar_id(), lake_id()).with_label("lahar-lake");
    let coupling = Box::new(MassTransferCoupling {
        alpha: ALPHA,
        interface,
    });

    Scenario::multi(vec![lahar_domain, lake_domain])
        .unwrap()
        .with_coupling(coupling)
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[test]
fn j3_multi_domain_scenario_builds_without_error() {
    let _ = make_scenario();
}

#[test]
fn j3_scenario_has_two_domains() {
    let scenario = make_scenario();
    assert_eq!(scenario.n_domains(), 2);
}

#[test]
fn j3_scenario_has_one_coupling() {
    let scenario = make_scenario();
    assert_eq!(scenario.n_couplings(), 1);
}

#[test]
fn j3_scenario_interface_is_registered() {
    let scenario = make_scenario();
    assert_eq!(scenario.interfaces().len(), 1);
    let iface = &scenario.interfaces()[0];
    assert_eq!(iface.source(), &lahar_id());
    assert_eq!(iface.target(), &lake_id());
    assert_eq!(iface.label(), Some("lahar-lake"));
}

#[test]
fn j3_context_requirements_aggregates_both_domains() {
    let scenario = make_scenario();
    let reqs = scenario.context_requirements();
    // LaharModel requires Time; LakeModel and MassTransferCoupling require nothing extra
    assert!(
        reqs.contains(&ContextVariable::Time),
        "Time not in aggregated requirements: {:?}",
        reqs
    );
}

#[test]
fn j3_coupling_apply_writes_to_target_domain() {
    let scenario = make_scenario();
    let coupling = &scenario.couplings()[0];
    let interface = coupling.interface();

    // Build a source state: lahar has concentration 1.0 at all nodes
    let mut states = MultiDomainState::new();
    states.set(
        lahar_id(),
        PhysicalQuantity::concentration(),
        ContextValue::ScalarField(DVector::from_element(N_NODES, 1.0)),
    );
    states.set(
        lake_id(),
        PhysicalQuantity::concentration(),
        ContextValue::ScalarField(DVector::zeros(N_NODES)),
    );

    let ctx = ComputeContext::new(0.0, 0.1);
    let result = coupling.apply(&states, &ctx, interface).unwrap();

    // Target domain receives alpha * source
    let lake_field = result
        .get(&lake_id(), &PhysicalQuantity::concentration())
        .unwrap()
        .as_scalar_field()
        .unwrap();

    for v in lake_field.iter() {
        assert!((v - ALPHA).abs() < 1e-12, "expected {ALPHA}, got {v}");
    }
}

#[test]
fn j3_coupling_does_not_mutate_source_domain() {
    let scenario = make_scenario();
    let coupling = &scenario.couplings()[0];
    let interface = coupling.interface();

    let source_field = DVector::from_element(N_NODES, 2.5);

    let mut states = MultiDomainState::new();
    states.set(
        lahar_id(),
        PhysicalQuantity::concentration(),
        ContextValue::ScalarField(source_field.clone()),
    );
    states.set(
        lake_id(),
        PhysicalQuantity::concentration(),
        ContextValue::ScalarField(DVector::zeros(N_NODES)),
    );

    let ctx = ComputeContext::new(0.0, 0.1);
    let result = coupling.apply(&states, &ctx, interface).unwrap();

    // Source domain entry must be unchanged
    let lahar_field = result
        .get(&lahar_id(), &PhysicalQuantity::concentration())
        .unwrap()
        .as_scalar_field()
        .unwrap();

    assert_eq!(lahar_field, &source_field);
}

#[test]
fn j3_composite_key_distinguishes_domains() {
    let mut states = MultiDomainState::new();
    let quantity = PhysicalQuantity::concentration();

    states.set(
        lahar_id(),
        quantity.clone(),
        ContextValue::ScalarField(DVector::from_element(N_NODES, 1.0)),
    );
    states.set(
        lake_id(),
        quantity.clone(),
        ContextValue::ScalarField(DVector::from_element(N_NODES, 0.0)),
    );

    assert_eq!(states.len(), 2);

    let lahar_val = states
        .get(&lahar_id(), &quantity)
        .unwrap()
        .as_scalar_field()
        .unwrap()[0];
    let lake_val = states
        .get(&lake_id(), &quantity)
        .unwrap()
        .as_scalar_field()
        .unwrap()[0];

    assert!((lahar_val - 1.0).abs() < 1e-12);
    assert!((lake_val - 0.0).abs() < 1e-12);
}

#[test]
fn j3_multi_domain_state_is_empty_initially() {
    let states = MultiDomainState::new();
    assert!(states.is_empty());
}
