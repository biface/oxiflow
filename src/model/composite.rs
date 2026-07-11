//! # Module `model::composite`
//!
//! `CompositeModel` — sum of multiple [`PhysicalModel`] on the same state
//! (DD-037, #45).
//!
//! ## Why
//!
//! oxiflow's canonical form already names two separate terms:
//!
//! $$\frac{\partial u}{\partial t} + \nabla \cdot F(u, \nabla u) = S(u, \mathbf{x}, t)$$
//!
//! `CompositeModel` represents the **complete** physics — the sum of all
//! contributions — as an ordinary `PhysicalModel`. It serves two purposes:
//!
//! 1. **Bookkeeping** for [`crate::solver::methods::imex::OperatorSplittingSolver`]:
//!    the outer `Domain` required by [`crate::solver::Solver::solve`]
//!    (DD-037) carries a `CompositeModel` whose `required_variables()`/
//!    `initial_state()` correctly cover the union of the sub-models —
//!    otherwise `build_calculator_chain` would miss a variable required by
//!    a single operator.
//! 2. **Monolithic reference**: run through any ordinary solver
//!    (`ForwardEulerSolver`, `RK4Solver`, …), it computes the *same*
//!    complete equation as the split version — exactly what acceptance
//!    criterion 1 of #45 needs to compare the split solution against a
//!    combined reference solution.
//!
//! `CompositeModel` knows nothing about Strang splitting, half-steps, or
//! any temporal decomposition — it just computes a sum.
//! `OperatorSplittingSolver` (`solver::methods::imex`) is the one that knows
//! how to integrate each contribution separately.

use crate::context::compute::ComputeContext;
use crate::context::error::OxiflowError;
use crate::context::value::ContextValue;
use crate::context::variable::ContextVariable;
use crate::mesh::Mesh;
use crate::model::traits::{PhysicalModel, RequiresContext};

/// Sum of multiple [`PhysicalModel`] evaluated on the same state.
///
/// See the module documentation for context (DD-037).
pub struct CompositeModel {
    operators: Vec<Box<dyn PhysicalModel>>,
    name: String,
}

impl CompositeModel {
    /// Builds a `CompositeModel` from at least one operator.
    ///
    /// # Errors
    ///
    /// `OxiflowError::PreconditionFailed` if `operators` is empty — a
    /// `CompositeModel` with no contribution makes no sense.
    pub fn new(
        operators: Vec<Box<dyn PhysicalModel>>,
        name: impl Into<String>,
    ) -> Result<Self, OxiflowError> {
        if operators.is_empty() {
            return Err(OxiflowError::PreconditionFailed {
                context: "CompositeModel::new",
                message: "at least one operator is required".into(),
            });
        }
        Ok(Self {
            operators,
            name: name.into(),
        })
    }

    /// Number of summed contributions.
    pub fn len(&self) -> usize {
        self.operators.len()
    }

    /// Always `false` in practice — `new` rejects an empty list — but
    /// required by convention (`clippy::len_without_is_empty`) whenever a
    /// public type exposes `len()`.
    pub fn is_empty(&self) -> bool {
        self.operators.is_empty()
    }
}

/// Adds two [`ContextValue`] element-wise.
///
/// Both values must be the same variant (`ScalarField` + `ScalarField`,
/// etc.) — `Boolean` does not support addition.
fn add_context_values(a: ContextValue, b: ContextValue) -> Result<ContextValue, OxiflowError> {
    use ContextValue::*;
    match (a, b) {
        (Scalar(x), Scalar(y)) => Ok(Scalar(x + y)),
        (Vector(x), Vector(y)) => Ok(Vector(x + y)),
        (Matrix(x), Matrix(y)) => Ok(Matrix(x + y)),
        (ScalarField(x), ScalarField(y)) => Ok(ScalarField(x + y)),
        (VectorField(x), VectorField(y)) => Ok(VectorField(x + y)),
        (a, b) => Err(OxiflowError::TypeMismatch {
            expected: a.variant_name(),
            actual: b.variant_name(),
        }),
    }
}

/// Extends `dest` with the elements of `extra` not already present —
/// same deduplication approach as
/// [`crate::solver::scenario::Scenario::context_requirements`]
/// (`ContextVariable` does not implement `Ord`).
fn merge_variables(dest: &mut Vec<ContextVariable>, extra: Vec<ContextVariable>) {
    dest.extend(extra);
    dest.sort_by(|a, b| format!("{a:?}").cmp(&format!("{b:?}")));
    dest.dedup();
}

impl std::fmt::Debug for CompositeModel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CompositeModel")
            .field("name", &self.name)
            .field(
                "operator_names",
                &self
                    .operators
                    .iter()
                    .map(|op| op.name())
                    .collect::<Vec<_>>(),
            )
            .finish()
    }
}

impl RequiresContext for CompositeModel {
    fn required_variables(&self) -> Vec<ContextVariable> {
        let mut vars = Vec::new();
        for op in &self.operators {
            merge_variables(&mut vars, op.required_variables());
        }
        vars
    }

    fn optional_variables(&self) -> Vec<ContextVariable> {
        let mut vars = Vec::new();
        for op in &self.operators {
            merge_variables(&mut vars, op.optional_variables());
        }
        vars
    }

    fn depends_on(&self) -> Vec<ContextVariable> {
        let mut vars = Vec::new();
        for op in &self.operators {
            merge_variables(&mut vars, op.depends_on());
        }
        vars
    }
}

impl PhysicalModel for CompositeModel {
    fn compute_physics(
        &self,
        state: &ContextValue,
        ctx: &ComputeContext,
    ) -> Result<ContextValue, OxiflowError> {
        let mut iter = self.operators.iter();
        let first = iter
            .next()
            .expect("CompositeModel::new guarantees at least one operator")
            .compute_physics(state, ctx)?;

        iter.try_fold(first, |total, op| {
            let contribution = op.compute_physics(state, ctx)?;
            add_context_values(total, contribution)
        })
    }

    fn initial_state(&self, mesh: &dyn Mesh) -> ContextValue {
        // All operators share the same starting physical state; the first
        // one suffices. See the module docs for this assumption.
        self.operators[0].initial_state(mesh)
    }

    fn name(&self) -> &str {
        &self.name
    }

    fn description(&self) -> Option<&str> {
        Some("composite model — sums multiple PhysicalModel contributions (DD-037)")
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mesh::structured::UniformGrid1D;
    use nalgebra::DVector;

    struct ConstantSource(f64);
    impl RequiresContext for ConstantSource {
        fn required_variables(&self) -> Vec<ContextVariable> {
            vec![]
        }
    }
    impl PhysicalModel for ConstantSource {
        fn compute_physics(
            &self,
            state: &ContextValue,
            _ctx: &ComputeContext,
        ) -> Result<ContextValue, OxiflowError> {
            let n = state.as_scalar_field()?.len();
            Ok(ContextValue::ScalarField(DVector::from_element(n, self.0)))
        }
        fn initial_state(&self, mesh: &dyn Mesh) -> ContextValue {
            ContextValue::ScalarField(DVector::from_element(mesh.n_dof(), 0.0))
        }
        fn name(&self) -> &str {
            "constant_source"
        }
    }

    struct NeedsTime;
    impl RequiresContext for NeedsTime {
        fn required_variables(&self) -> Vec<ContextVariable> {
            vec![ContextVariable::Time]
        }
    }
    impl PhysicalModel for NeedsTime {
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
            "needs_time"
        }
    }

    #[test]
    fn rejects_empty_operator_list() {
        let err = CompositeModel::new(vec![], "empty").unwrap_err();
        assert!(matches!(err, OxiflowError::PreconditionFailed { .. }));
    }

    #[test]
    fn compute_physics_sums_contributions() {
        let model = CompositeModel::new(
            vec![Box::new(ConstantSource(2.0)), Box::new(ConstantSource(3.0))],
            "sum_test",
        )
        .unwrap();
        let state = ContextValue::ScalarField(DVector::from_element(4, 0.0));
        let ctx = ComputeContext::new(0.0, 0.01);
        let result = model.compute_physics(&state, &ctx).unwrap();
        let field = result.as_scalar_field().unwrap();
        assert!(field.iter().all(|&v| (v - 5.0).abs() < 1e-12));
    }

    #[test]
    fn required_variables_is_union_deduplicated() {
        let model =
            CompositeModel::new(vec![Box::new(NeedsTime), Box::new(NeedsTime)], "dedup_test")
                .unwrap();
        let vars = model.required_variables();
        assert_eq!(vars, vec![ContextVariable::Time]);
    }

    #[test]
    fn initial_state_delegates_to_first_operator() {
        let model = CompositeModel::new(
            vec![Box::new(NeedsTime), Box::new(ConstantSource(0.0))],
            "init_test",
        )
        .unwrap();
        let mesh = UniformGrid1D::new(10, 0.0, 1.0).unwrap();
        let state = model.initial_state(&mesh);
        assert!(state.as_scalar_field().unwrap().iter().all(|&v| v == 1.0));
    }

    #[test]
    fn len_reports_operator_count() {
        let model = CompositeModel::new(
            vec![Box::new(ConstantSource(1.0)), Box::new(ConstantSource(2.0))],
            "len_test",
        )
        .unwrap();
        assert_eq!(model.len(), 2);
        assert!(!model.is_empty());
    }
}
