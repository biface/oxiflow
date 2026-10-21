//! Lahar–lake coupled prototype: the two domain models (`GranularFlow`,
//! `ShallowWaterReceiver`) and the coupling operator (`MassTransfer`).
//!
//! Single definition shared by `examples/lahar_lake_proto.rs` and
//! `tests/lahar_lake_proto.rs`, each of which includes this file as a
//! module with `#[path]`. The simplified physics and the mass-conservation
//! argument are documented in `examples/lahar_lake_proto.rs`. Scenario
//! constants (node count, rate, time span) and domain identifiers stay in
//! each consumer.

use nalgebra::DVector;
use oxiflow::{
    context::{
        compute::ComputeContext, error::OxiflowError, quantity::PhysicalQuantity,
        state::MultiDomainState, value::ContextValue, variable::ContextVariable,
    },
    coupling::{CouplingOperator, Interface},
    mesh::Mesh,
    model::traits::{PhysicalModel, RequiresContext},
};

// ── Models ──────────────────────────────────────────────────────────────────

/// Granular flow stand-in — exponential mass loss, `dC/dt = -lambda * C`.
pub(crate) struct GranularFlow {
    pub(crate) lambda: f64,
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
pub(crate) struct ShallowWaterReceiver;

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
pub(crate) struct MassTransfer {
    pub(crate) lambda: f64,
    pub(crate) interface: Interface,
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
                context: "MassTransfer",
                message: format!(
                    "source domain '{}' has no Concentration field",
                    interface.source()
                ),
            })?
            .as_scalar_field()?;
        let target = states
            .get(interface.target(), &quantity)
            .ok_or_else(|| OxiflowError::PreconditionFailed {
                context: "MassTransfer",
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
