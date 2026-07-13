//! # Module `model::discretized`
//!
//! `DiscretizedModel<Op>` — wraps a [`FluxDivergenceOperator`] scheme (FV,
//! WENO) as an ordinary [`PhysicalModel`], returning only the `-∇·F(u, ∇u)`
//! contribution of the canonical form (DD-038, #93).
//!
//! ## Why
//!
//! FD (gradient/Laplacian, DD-012) is consumed transparently through
//! [`crate::context::ContextCalculator`] — a model declares a required
//! context variable and never knows which scheme produced it. FV (#48) and
//! WENO/limiters (#49) are different: they compute the complete `∇·F` term
//! directly — exactly what `PhysicalModel::compute_physics` must return.
//! `DiscretizedModel<Op>` is the insertion point for that case: a spatial
//! composite, cousin of
//! [`crate::solver::methods::imex::SplitOperator`] (DD-037, temporal
//! composition) but for wiring a single spatial scheme into the engine.
//!
//! A trait `SourceTerm` mirroring `DiscreteOperator`'s `S(u, x, t)` was
//! considered and rejected (DD-038, amendment 1) — nothing in the v0.5.0
//! scope (#48, #49) needs a source term richer than an ordinary
//! `PhysicalModel` already expresses. When a source term exists, it composes
//! with `DiscretizedModel<Op>` through
//! [`crate::model::CompositeModel`] (DD-037) — already delivered, already
//! tested, no new trait:
//!
//! ```rust,ignore
//! let full_model = CompositeModel::new(
//!     vec![
//!         Box::new(DiscretizedModel::new(scheme, mesh, "advection")),
//!         Box::new(my_source_model), // ordinary PhysicalModel, S(u, x, t)
//!     ],
//!     "advection_reaction",
//! )?;
//! ```
//!
//! `Op` is bound to [`FluxDivergenceOperator`] (DD-039), not
//! `DiscreteOperator` (DD-038, amendment 2) — an FD scheme cannot be plugged
//! in here; that is a compile error, not a silent runtime bug.

use std::borrow::Cow;
use std::sync::Arc;

use nalgebra::DVector;

use crate::context::compute::ComputeContext;
use crate::context::error::OxiflowError;
use crate::context::value::ContextValue;
use crate::context::variable::ContextVariable;
use crate::mesh::Mesh;
use crate::model::traits::{PhysicalModel, RequiresContext};
use crate::operators::FluxDivergenceOperator;

/// Negates a [`ContextValue`] element-wise.
///
/// `Boolean` does not support negation as a numeric operation.
fn negate_context_value(value: ContextValue) -> Result<ContextValue, OxiflowError> {
    use ContextValue::*;
    match value {
        Scalar(x) => Ok(Scalar(-x)),
        Vector(x) => Ok(Vector(-x)),
        Matrix(x) => Ok(Matrix(-x)),
        ScalarField(x) => Ok(ScalarField(-x)),
        VectorField(x) => Ok(VectorField(-x)),
        other => Err(OxiflowError::TypeMismatch {
            expected: "negatable ContextValue (Scalar/Vector/Matrix/ScalarField/VectorField)",
            actual: other.variant_name(),
        }),
    }
}

/// Computes the flux via `scheme` — not yet registered in any
/// `ContextCalculator` chain (DD-038: private posture). `instance_id` is
/// reserved, unused today, to allow multi-instance naming (DD-031, DD-037)
/// the day DD-027 requires publication into `ComputeContext` (VTK/HDF5
/// export, v0.6.0). No concrete need justifies publishing it sooner.
#[allow(dead_code)]
struct FluxDivergenceCalculator<Op: FluxDivergenceOperator> {
    scheme: Arc<Op>,
    mesh: Arc<Op::MeshType>,
    instance_id: Cow<'static, str>,
}

/// Wraps a spatial discretization scheme as an ordinary [`PhysicalModel`] —
/// the `-∇·F(u, ∇u)` contribution of the canonical form, nothing else.
/// Cousin of [`crate::solver::methods::imex::SplitOperator`] (DD-037), but
/// spatial rather than temporal.
///
/// See the module documentation for how to compose a source term via
/// [`crate::model::CompositeModel`].
pub struct DiscretizedModel<Op: FluxDivergenceOperator> {
    scheme: Arc<Op>,
    mesh: Arc<Op::MeshType>,
    // Not read yet — reserved for DD-027 (VTK/HDF5 export, v0.6.0). See
    // FluxDivergenceCalculator's doc comment for why publication is deferred.
    #[allow(dead_code)]
    flux_calculator: FluxDivergenceCalculator<Op>,
    name: String,
}

impl<Op: FluxDivergenceOperator> DiscretizedModel<Op> {
    /// Wraps `scheme` evaluated on `mesh` as a `PhysicalModel` named `name`.
    pub fn new(scheme: Arc<Op>, mesh: Arc<Op::MeshType>, name: impl Into<String>) -> Self {
        let name = name.into();
        let flux_calculator = FluxDivergenceCalculator {
            scheme: Arc::clone(&scheme),
            mesh: Arc::clone(&mesh),
            instance_id: Cow::Owned(name.clone()),
        };
        Self {
            scheme,
            mesh,
            flux_calculator,
            name,
        }
    }
}

impl<Op: FluxDivergenceOperator> RequiresContext for DiscretizedModel<Op> {
    fn required_variables(&self) -> Vec<ContextVariable> {
        self.scheme.required_variables()
    }

    fn optional_variables(&self) -> Vec<ContextVariable> {
        self.scheme.optional_variables()
    }

    fn depends_on(&self) -> Vec<ContextVariable> {
        self.scheme.depends_on()
    }

    fn priority(&self) -> u32 {
        self.scheme.priority()
    }
}

impl<Op: FluxDivergenceOperator> PhysicalModel for DiscretizedModel<Op> {
    fn compute_physics(
        &self,
        state: &ContextValue,
        ctx: &ComputeContext,
    ) -> Result<ContextValue, OxiflowError> {
        negate_context_value(self.scheme.apply(state, &self.mesh, ctx)?)
    }

    fn initial_state(&self, mesh: &dyn Mesh) -> ContextValue {
        // DiscretizedModel represents the flux term only, not the physical
        // initial condition — delegates to the mesh for sizing, zero
        // elsewhere. The real initial condition comes from whichever
        // PhysicalModel composed alongside it (CompositeModel::initial_state
        // uses its first operand).
        ContextValue::ScalarField(DVector::from_element(mesh.n_dof(), 0.0))
    }

    fn name(&self) -> &str {
        &self.name
    }

    fn description(&self) -> Option<&str> {
        Some("discretized model — wraps a FluxDivergenceOperator as -∇·F(u, ∇u) (DD-038)")
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mesh::structured::UniformGrid1D;

    /// Minimal `FluxDivergenceOperator` fixture: returns a constant field,
    /// declares one required variable to exercise delegation.
    struct ConstantFlux(f64);

    impl RequiresContext for ConstantFlux {
        fn required_variables(&self) -> Vec<ContextVariable> {
            vec![ContextVariable::Time]
        }
    }

    impl FluxDivergenceOperator for ConstantFlux {
        type MeshType = UniformGrid1D;

        fn apply(
            &self,
            field: &ContextValue,
            _mesh: &Self::MeshType,
            _ctx: &ComputeContext,
        ) -> Result<ContextValue, OxiflowError> {
            let n = field.as_scalar_field()?.len();
            Ok(ContextValue::ScalarField(DVector::from_element(n, self.0)))
        }
    }

    fn fixture(value: f64) -> DiscretizedModel<ConstantFlux> {
        let mesh = Arc::new(UniformGrid1D::new(4, 0.0, 1.0).unwrap());
        DiscretizedModel::new(Arc::new(ConstantFlux(value)), mesh, "constant_flux_test")
    }

    #[test]
    fn compute_physics_negates_scheme_output() {
        let model = fixture(3.0);
        let state = ContextValue::ScalarField(DVector::from_element(4, 0.0));
        let ctx = ComputeContext::new(0.0, 0.01);
        let result = model.compute_physics(&state, &ctx).unwrap();
        assert!(result
            .as_scalar_field()
            .unwrap()
            .iter()
            .all(|&v| (v + 3.0).abs() < 1e-12));
    }

    #[test]
    fn required_variables_delegates_to_scheme() {
        let model = fixture(1.0);
        assert_eq!(model.required_variables(), vec![ContextVariable::Time]);
    }

    #[test]
    fn initial_state_is_zero_sized_from_mesh() {
        let model = fixture(1.0);
        let mesh = UniformGrid1D::new(6, 0.0, 1.0).unwrap();
        let state = model.initial_state(&mesh);
        let field = state.as_scalar_field().unwrap();
        assert_eq!(field.len(), 6);
        assert!(field.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn name_returns_identifier() {
        let model = fixture(1.0);
        assert_eq!(model.name(), "constant_flux_test");
    }

    #[test]
    fn negate_context_value_rejects_boolean() {
        let err = negate_context_value(ContextValue::Boolean(true)).unwrap_err();
        assert!(matches!(err, OxiflowError::TypeMismatch { .. }));
    }
}
