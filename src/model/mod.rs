//! # Module `model`
//!
//! Physical models — `PhysicalModel` trait and component traits.
//!
//! A physical model *declares* its context variable needs via `RequiresContext`
//! and *computes* field `u` derivatives at each time step. It does not configure
//! the solving nor orchestrate the time loop — those are the responsibilities of
//! `SolverConfiguration` and `Solver`.

pub mod composite;
pub mod discretized;
pub mod traits;

pub use composite::CompositeModel;
pub use discretized::DiscretizedModel;
pub use traits::PhysicalModel;
pub use traits::RequiresContext;

// ── Test models ───────────────────────────────────────────────────────────────
//
// Scalar models shared by the unit tests of the solvers, each with no
// context requirement. Defined once here rather than in every test module.

#[cfg(test)]
use crate::context::{
    compute::ComputeContext, error::OxiflowError, value::ContextValue, variable::ContextVariable,
};
#[cfg(test)]
use crate::mesh::Mesh;
#[cfg(test)]
use nalgebra::DVector;

/// `du/dt = -lambda * u`, from `u = 1` at every node. Exact solution
/// `u(t) = exp(-lambda * t)`.
#[cfg(test)]
#[derive(Debug)]
pub(crate) struct ExponentialDecay {
    pub(crate) lambda: f64,
}

#[cfg(test)]
impl RequiresContext for ExponentialDecay {
    fn required_variables(&self) -> Vec<ContextVariable> {
        vec![]
    }
}

#[cfg(test)]
impl PhysicalModel for ExponentialDecay {
    fn compute_physics(
        &self,
        state: &ContextValue,
        _ctx: &ComputeContext,
    ) -> Result<ContextValue, OxiflowError> {
        let u = state.as_scalar_field()?;
        Ok(ContextValue::ScalarField(u.map(|v| -self.lambda * v)))
    }

    fn initial_state(&self, mesh: &dyn Mesh) -> ContextValue {
        ContextValue::ScalarField(DVector::from_element(mesh.n_dof(), 1.0))
    }

    fn name(&self) -> &str {
        "exponential_decay"
    }
}

/// `du/dt = 0`: the field keeps its initial value, `initial` at every node.
#[cfg(test)]
#[derive(Debug)]
pub(crate) struct ZeroDerivative {
    pub(crate) initial: f64,
}

#[cfg(test)]
impl RequiresContext for ZeroDerivative {
    fn required_variables(&self) -> Vec<ContextVariable> {
        vec![]
    }
}

#[cfg(test)]
impl PhysicalModel for ZeroDerivative {
    fn compute_physics(
        &self,
        state: &ContextValue,
        _ctx: &ComputeContext,
    ) -> Result<ContextValue, OxiflowError> {
        let u = state.as_scalar_field()?;
        Ok(ContextValue::ScalarField(DVector::zeros(u.len())))
    }

    fn initial_state(&self, mesh: &dyn Mesh) -> ContextValue {
        ContextValue::ScalarField(DVector::from_element(mesh.n_dof(), self.initial))
    }

    fn name(&self) -> &str {
        "zero_derivative"
    }
}

/// `du/dt = -u^3`, from `u = 1` at every node: nonlinear, for the
/// iterated Newton tests of the implicit solvers.
#[cfg(test)]
#[derive(Debug)]
pub(crate) struct CubicDecay;

#[cfg(test)]
impl RequiresContext for CubicDecay {
    fn required_variables(&self) -> Vec<ContextVariable> {
        vec![]
    }
}

#[cfg(test)]
impl PhysicalModel for CubicDecay {
    fn compute_physics(
        &self,
        state: &ContextValue,
        _ctx: &ComputeContext,
    ) -> Result<ContextValue, OxiflowError> {
        let u = state.as_scalar_field()?;
        Ok(ContextValue::ScalarField(u.map(|v| -v * v * v)))
    }

    fn initial_state(&self, mesh: &dyn Mesh) -> ContextValue {
        ContextValue::ScalarField(DVector::from_element(mesh.n_dof(), 1.0))
    }

    fn name(&self) -> &str {
        "cubic_decay"
    }
}
