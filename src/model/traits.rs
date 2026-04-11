//! # Module `model::traits`
//!
//! Core traits for physical model components (DD-005, issue #29, #32).
//!
//! ## `RequiresContext`
//!
//! Standalone trait — `required_variables()` is mandatory (DD-005).
//!
//! ## `PhysicalModel`
//!
//! Supertrait of `RequiresContext`. Declares the physical equations and computes
//! field derivatives. The method is named `compute_physics` — the `_v2` suffix
//! from chrom-rs is dropped because oxiflow never had a v1 (DD-006).

use crate::context::compute::ComputeContext;
use crate::context::error::OxiflowError;
use crate::context::value::ContextValue;
use crate::context::variable::ContextVariable;
use crate::mesh::Mesh;

/// Declares context variable requirements for any engine component.
///
/// Implemented by `PhysicalModel`, `BoundaryCondition` (J2), `SourceTerm`,
/// and `CouplingOperator` (J3). The solver aggregates declarations from all
/// components before solving and guarantees every required variable is available.
///
/// `required_variables()` has no default — every implementor must declare
/// its requirements explicitly, even if that declaration is `vec![]`.
pub trait RequiresContext {
    /// Variables this component **must** have. Solver raises
    /// `OxiflowError::MissingCalculator` if any is absent.
    ///
    /// No default — explicit declaration is mandatory.
    fn required_variables(&self) -> Vec<ContextVariable>;

    /// Variables used when available but not strictly required.
    fn optional_variables(&self) -> Vec<ContextVariable> {
        vec![]
    }

    /// Variables that must be computed before this component runs.
    ///
    /// Used by `chain.rs` for topological ordering (DD-009, J2).
    fn depends_on(&self) -> Vec<ContextVariable> {
        vec![]
    }

    /// Execution priority within the calculator chain (lower = earlier).
    ///
    /// Reserved ranges: 0 = system (Time, TimeStep), 50 = external data,
    /// 100 = default for derived quantities.
    fn priority(&self) -> u32 {
        100
    }
}

/// Physical model — declares needs and computes field derivatives.
///
/// Supertrait of `RequiresContext`. A model declares *what* context variables
/// it needs and *how* to compute the time derivative of the primary field:
///
/// $$\frac{\partial u}{\partial t} = -\nabla \cdot F(u, \nabla u) + S(u, \mathbf{x}, t)$$
/// It does not configure solving nor orchestrate the time loop — those are
/// the responsibilities of `SolverConfiguration` and `Solver`.
///
/// # Naming
///
/// The method is named `compute_physics` (not `compute_physics_v2`).
/// oxiflow never had a v1 API — the `_v2` suffix was a chrom-rs migration
/// artifact that does not apply here (DD-006).
///
/// # Examples
///
/// ```rust
/// use oxiflow::model::traits::{PhysicalModel, RequiresContext};
/// use oxiflow::context::variable::ContextVariable;
/// use oxiflow::context::value::ContextValue;
/// use oxiflow::context::compute::ComputeContext;
/// use oxiflow::context::error::OxiflowError;
/// use oxiflow::mesh::Mesh;
/// use nalgebra::DVector;
///
/// struct PureDecay { rate: f64 }
///
/// impl RequiresContext for PureDecay {
///     fn required_variables(&self) -> Vec<ContextVariable> {
///         vec![ContextVariable::Time]
///     }
/// }
///
/// impl PhysicalModel for PureDecay {
///     fn compute_physics(
///         &self,
///         state: &ContextValue,
///         _ctx: &ComputeContext,
///     ) -> Result<ContextValue, OxiflowError> {
///         let u = state.as_scalar_field()?;
///         let du_dt = u.map(|v| -self.rate * v);
///         Ok(ContextValue::ScalarField(du_dt))
///     }
///
///     fn initial_state(&self, mesh: &dyn Mesh) -> ContextValue {
///         ContextValue::ScalarField(DVector::from_element(mesh.n_dof(), 1.0))
///     }
///
///     fn name(&self) -> &str { "pure_decay" }
/// }
/// ```
pub trait PhysicalModel: RequiresContext + Send + Sync {
    /// Computes the time derivative `du/dt` of the primary field.
    ///
    /// # Arguments
    ///
    /// - `state` — current field `u`, typically `ContextValue::ScalarField`
    ///   for 1D problems or `ContextValue::VectorField` for multi-component.
    /// - `ctx`   — fully populated context for this time step; all variables
    ///   declared in `required_variables()` are guaranteed present.
    ///
    /// # Returns
    ///
    /// The derivative field $\partial u / \partial t$, same shape as `state`.
    fn compute_physics(
        &self,
        state: &ContextValue,
        ctx: &ComputeContext,
    ) -> Result<ContextValue, OxiflowError>;

    /// Returns the initial condition `u(x, t_start)` on `mesh`.
    fn initial_state(&self, mesh: &dyn Mesh) -> ContextValue;

    /// Human-readable identifier for logging and `SimulationResult` metadata.
    fn name(&self) -> &str;

    /// Optional longer description.
    fn description(&self) -> Option<&str> {
        None
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mesh::structured::UniformGrid1D;
    use nalgebra::DVector;

    // ── Fixtures ──────────────────────────────────────────────────────────────

    struct FullModel;
    impl RequiresContext for FullModel {
        fn required_variables(&self) -> Vec<ContextVariable> {
            vec![
                ContextVariable::Time,
                ContextVariable::SpatialGradient {
                    dimension: 0,
                    component: None,
                },
            ]
        }
        fn optional_variables(&self) -> Vec<ContextVariable> {
            vec![ContextVariable::External { name: "T_amb" }]
        }
        fn depends_on(&self) -> Vec<ContextVariable> {
            vec![ContextVariable::Time]
        }
        fn priority(&self) -> u32 {
            200
        }
    }

    impl PhysicalModel for FullModel {
        fn compute_physics(
            &self,
            state: &ContextValue,
            _ctx: &ComputeContext,
        ) -> Result<ContextValue, OxiflowError> {
            let u = state.as_scalar_field()?;
            Ok(ContextValue::ScalarField(u.clone()))
        }
        fn initial_state(&self, mesh: &dyn Mesh) -> ContextValue {
            ContextValue::ScalarField(DVector::from_element(mesh.n_dof(), 0.0))
        }
        fn name(&self) -> &str {
            "full_model"
        }
        fn description(&self) -> Option<&str> {
            Some("test model")
        }
    }

    struct MinimalModel;
    impl RequiresContext for MinimalModel {
        fn required_variables(&self) -> Vec<ContextVariable> {
            vec![]
        }
    }
    impl PhysicalModel for MinimalModel {
        fn compute_physics(
            &self,
            state: &ContextValue,
            _ctx: &ComputeContext,
        ) -> Result<ContextValue, OxiflowError> {
            Ok(state.clone())
        }
        fn initial_state(&self, mesh: &dyn Mesh) -> ContextValue {
            ContextValue::ScalarField(DVector::from_element(mesh.n_dof(), 1.0))
        }
        fn name(&self) -> &str {
            "minimal"
        }
    }

    // ── RequiresContext ───────────────────────────────────────────────────────

    #[test]
    fn required_variables_returns_declared() {
        let vars = FullModel.required_variables();
        assert_eq!(vars.len(), 2);
        assert!(vars.contains(&ContextVariable::Time));
    }

    #[test]
    fn empty_required_variables_is_valid() {
        assert!(MinimalModel.required_variables().is_empty());
    }

    #[test]
    fn optional_and_depends_on_defaults_are_empty() {
        assert!(MinimalModel.optional_variables().is_empty());
        assert!(MinimalModel.depends_on().is_empty());
    }

    #[test]
    fn priority_default_is_100() {
        assert_eq!(MinimalModel.priority(), 100);
    }

    #[test]
    fn priority_custom_value() {
        assert_eq!(FullModel.priority(), 200);
    }

    // ── PhysicalModel ─────────────────────────────────────────────────────────

    #[test]
    fn compute_physics_returns_derivative() {
        let ctx = ComputeContext::new(0.0, 0.01);
        let state = ContextValue::ScalarField(DVector::from_vec(vec![1.0, 2.0, 3.0]));
        let result = FullModel.compute_physics(&state, &ctx).unwrap();
        assert!(result.is_scalar_field());
        assert_eq!(result.as_scalar_field().unwrap().len(), 3);
    }

    #[test]
    fn compute_physics_wrong_state_type_returns_error() {
        let ctx = ComputeContext::new(0.0, 0.01);
        let state = ContextValue::Scalar(1.0);
        let err = FullModel.compute_physics(&state, &ctx).unwrap_err();
        assert!(matches!(err, OxiflowError::TypeMismatch { .. }));
    }

    #[test]
    fn initial_state_matches_mesh_n_dof() {
        let mesh = UniformGrid1D::new(10, 0.0, 1.0).unwrap();
        let state = FullModel.initial_state(&mesh);
        assert_eq!(state.as_scalar_field().unwrap().len(), 10);
    }

    #[test]
    fn name_returns_identifier() {
        assert_eq!(FullModel.name(), "full_model");
    }

    #[test]
    fn description_returns_some_or_none() {
        assert_eq!(FullModel.description(), Some("test model"));
        assert_eq!(MinimalModel.description(), None);
    }

    // ── Object safety ─────────────────────────────────────────────────────────

    #[test]
    fn physical_model_is_object_safe() {
        let models: Vec<Box<dyn PhysicalModel>> = vec![Box::new(FullModel), Box::new(MinimalModel)];
        assert_eq!(models[0].name(), "full_model");
        assert_eq!(models[1].name(), "minimal");
    }
}
