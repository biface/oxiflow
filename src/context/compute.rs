//! # Module `context::compute`
//!
//! Type-safe computation context (DD-006, issue #30).
//!
//! ## `ComputeContext`
//!
//! Provides typed access to all context variables resolved by the solver before
//! each time step. Physical models receive a `&ComputeContext` in
//! `compute_physics_v2()` and access variables through typed accessors —
//! no string keys, no `HashMap<String, f64>`, no runtime type guessing.
//!
//! ## Revised signatures (DD-006)
//!
//! - `gradient(dim)` returns `&DVector<f64>` — the full nodal field, consistent
//!   with `ContextValue::ScalarField`. The model indexes the node it needs.
//! - `external(var)` takes a typed `ContextVariable` — no `&str` shortcut that
//!   would bypass the type system.

use std::collections::HashMap;

use nalgebra::{DMatrix, DVector};

use crate::context::error::OxiflowError;
use crate::context::value::ContextValue;
use crate::context::variable::ContextVariable;

/// Type-safe computation context provided to physical models during time integration.
///
/// Built by the solver at each time step from registered calculators. Models access
/// variables through typed accessors — every access is either correct at compile time
/// or fails explicitly via `OxiflowError` at solve startup.
///
/// # Examples
///
/// ```rust
/// use oxiflow::context::compute::ComputeContext;
/// use oxiflow::context::variable::ContextVariable;
/// use oxiflow::context::value::ContextValue;
///
/// let mut ctx = ComputeContext::new(1.5, 0.01);
/// ctx.insert(ContextVariable::SpatialGradient { dimension: 0, component: None },
///            ContextValue::ScalarField(nalgebra::DVector::from_vec(vec![0.1, 0.2, 0.3])));
///
/// assert_eq!(ctx.time(), 1.5);
/// assert_eq!(ctx.time_step(), 0.01);
/// assert_eq!(ctx.gradient(0).unwrap().len(), 3);
/// ```
pub struct ComputeContext {
    /// Current simulation time `t`.
    time: f64,
    /// Current time step `dt`.
    time_step: f64,
    /// Resolved context variables, keyed by typed `ContextVariable`.
    variables: HashMap<ContextVariable, ContextValue>,
}

impl ComputeContext {
    /// Creates a new context for a given time and time step.
    ///
    /// Variables are inserted after construction via [`insert`](Self::insert).
    pub fn new(time: f64, time_step: f64) -> Self {
        Self {
            time,
            time_step,
            variables: HashMap::new(),
        }
    }

    /// Inserts a resolved variable into the context.
    ///
    /// Called by the solver after each calculator runs. Overwrites any previous
    /// value for the same key.
    pub fn insert(&mut self, var: ContextVariable, value: ContextValue) {
        self.variables.insert(var, value);
    }

    // ── System accessors (infallible) ─────────────────────────────────────────

    /// Returns the current simulation time `t`.
    pub fn time(&self) -> f64 {
        self.time
    }

    /// Returns the current time step `dt`.
    pub fn time_step(&self) -> f64 {
        self.time_step
    }

    // ── Typed accessors ───────────────────────────────────────────────────────

    /// Returns the scalar value of a `Scalar` context variable.
    ///
    /// # Errors
    ///
    /// - `OxiflowError::MissingCalculator` if the variable is not in the context.
    /// - `OxiflowError::TypeMismatch` if the variable is present but not a `Scalar`.
    pub fn scalar(&self, var: ContextVariable) -> Result<f64, OxiflowError> {
        self.get_value(&var)?.as_scalar()
    }

    /// Borrows the `DVector` of a `Vector` context variable.
    ///
    /// # Errors
    ///
    /// - `OxiflowError::MissingCalculator` if the variable is not in the context.
    /// - `OxiflowError::TypeMismatch` if the variable is present but not a `Vector`.
    pub fn vector(&self, var: ContextVariable) -> Result<&DVector<f64>, OxiflowError> {
        self.get_value(&var)?.as_vector()
    }

    /// Borrows the `DMatrix` of a `Matrix` context variable.
    ///
    /// # Errors
    ///
    /// - `OxiflowError::MissingCalculator` if the variable is not in the context.
    /// - `OxiflowError::TypeMismatch` if the variable is present but not a `Matrix`.
    pub fn matrix(&self, var: ContextVariable) -> Result<&DMatrix<f64>, OxiflowError> {
        self.get_value(&var)?.as_matrix()
    }

    /// Borrows the full nodal gradient field for spatial dimension `dim`.
    ///
    /// Returns the `DVector` of `ContextValue::ScalarField` stored under
    /// `ContextVariable::SpatialGradient { dimension: dim, component: None }`.
    /// The model indexes the node it needs from the returned slice.
    ///
    /// # Errors
    ///
    /// - `OxiflowError::MissingCalculator` if no gradient calculator is registered
    ///   for `dim`.
    /// - `OxiflowError::TypeMismatch` if the value is not a `ScalarField`.
    pub fn gradient(&self, dim: usize) -> Result<&DVector<f64>, OxiflowError> {
        let var = ContextVariable::SpatialGradient {
            dimension: dim,
            component: None,
        };
        self.get_value(&var)?.as_scalar_field()
    }

    /// Borrows any context variable by typed key, returning the raw `ContextValue`.
    ///
    /// Use this for `External` variables or when the caller needs to handle multiple
    /// variant types. For common cases prefer the typed accessors.
    ///
    /// # Errors
    ///
    /// - `OxiflowError::MissingCalculator` if the variable is not in the context.
    pub fn external(&self, var: ContextVariable) -> Result<&ContextValue, OxiflowError> {
        self.get_value(&var)
    }

    /// Returns a reference to a context variable if present, or `None` if absent.
    ///
    /// Non-failing alternative to the typed accessors — use for optional variables.
    pub fn try_get(&self, var: ContextVariable) -> Option<&ContextValue> {
        self.variables.get(&var)
    }

    // ── Internal helper ───────────────────────────────────────────────────────

    fn get_value(&self, var: &ContextVariable) -> Result<&ContextValue, OxiflowError> {
        self.variables
            .get(var)
            .ok_or_else(|| OxiflowError::MissingCalculator(var.clone()))
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::{DMatrix, DVector};

    // ── Helpers ───────────────────────────────────────────────────────────────

    fn ctx_with_all_variants() -> ComputeContext {
        let mut ctx = ComputeContext::new(2.0, 0.01);
        ctx.insert(
            ContextVariable::External { name: "coeff" },
            ContextValue::Scalar(42.0),
        );
        ctx.insert(
            ContextVariable::External { name: "flag" },
            ContextValue::Boolean(true),
        );
        ctx.insert(
            ContextVariable::External { name: "vel" },
            ContextValue::Vector(DVector::from_vec(vec![1.0, 2.0, 3.0])),
        );
        ctx.insert(
            ContextVariable::External { name: "tensor" },
            ContextValue::Matrix(DMatrix::from_element(2, 2, 5.0)),
        );
        ctx.insert(
            ContextVariable::SpatialGradient {
                dimension: 0,
                component: None,
            },
            ContextValue::ScalarField(DVector::from_vec(vec![0.1, 0.2, 0.3])),
        );
        ctx.insert(
            ContextVariable::External { name: "vfield" },
            ContextValue::VectorField(DMatrix::from_element(3, 2, 1.5)),
        );
        ctx
    }

    // ── System accessors ──────────────────────────────────────────────────────

    #[test]
    fn time_returns_correct_value() {
        let ctx = ComputeContext::new(3.14, 0.001);
        assert_eq!(ctx.time(), 3.14);
    }

    #[test]
    fn time_step_returns_correct_value() {
        let ctx = ComputeContext::new(0.0, 0.05);
        assert_eq!(ctx.time_step(), 0.05);
    }

    // ── scalar ────────────────────────────────────────────────────────────────

    #[test]
    fn scalar_returns_value_for_scalar_variable() {
        let ctx = ctx_with_all_variants();
        let v = ctx
            .scalar(ContextVariable::External { name: "coeff" })
            .unwrap();
        assert_eq!(v, 42.0);
    }

    #[test]
    fn scalar_returns_missing_calculator_when_absent() {
        let ctx = ComputeContext::new(0.0, 0.01);
        let err = ctx.scalar(ContextVariable::Time).unwrap_err();
        assert!(matches!(err, OxiflowError::MissingCalculator(_)));
    }

    #[test]
    fn scalar_returns_type_mismatch_for_non_scalar() {
        let ctx = ctx_with_all_variants();
        let err = ctx
            .scalar(ContextVariable::External { name: "vel" })
            .unwrap_err();
        assert!(matches!(
            err,
            OxiflowError::TypeMismatch {
                expected: "Scalar",
                ..
            }
        ));
    }

    // ── vector ────────────────────────────────────────────────────────────────

    #[test]
    fn vector_returns_reference_for_vector_variable() {
        let ctx = ctx_with_all_variants();
        let v = ctx
            .vector(ContextVariable::External { name: "vel" })
            .unwrap();
        assert_eq!(v.len(), 3);
        assert_eq!(v[1], 2.0);
    }

    #[test]
    fn vector_returns_missing_calculator_when_absent() {
        let ctx = ComputeContext::new(0.0, 0.01);
        let err = ctx
            .vector(ContextVariable::External { name: "missing" })
            .unwrap_err();
        assert!(matches!(err, OxiflowError::MissingCalculator(_)));
    }

    #[test]
    fn vector_returns_type_mismatch_for_non_vector() {
        let ctx = ctx_with_all_variants();
        let err = ctx
            .vector(ContextVariable::External { name: "coeff" })
            .unwrap_err();
        assert!(matches!(
            err,
            OxiflowError::TypeMismatch {
                expected: "Vector",
                ..
            }
        ));
    }

    // ── matrix ────────────────────────────────────────────────────────────────

    #[test]
    fn matrix_returns_reference_for_matrix_variable() {
        let ctx = ctx_with_all_variants();
        let m = ctx
            .matrix(ContextVariable::External { name: "tensor" })
            .unwrap();
        assert_eq!(m.shape(), (2, 2));
        assert_eq!(m[(0, 0)], 5.0);
    }

    #[test]
    fn matrix_returns_type_mismatch_for_non_matrix() {
        let ctx = ctx_with_all_variants();
        let err = ctx
            .matrix(ContextVariable::External { name: "coeff" })
            .unwrap_err();
        assert!(matches!(
            err,
            OxiflowError::TypeMismatch {
                expected: "Matrix",
                ..
            }
        ));
    }

    // ── gradient ──────────────────────────────────────────────────────────────

    #[test]
    fn gradient_returns_full_nodal_field() {
        let ctx = ctx_with_all_variants();
        let g = ctx.gradient(0).unwrap();
        assert_eq!(g.len(), 3);
        assert!((g[0] - 0.1).abs() < 1e-12);
        assert!((g[2] - 0.3).abs() < 1e-12);
    }

    #[test]
    fn gradient_missing_dimension_returns_missing_calculator() {
        let ctx = ctx_with_all_variants();
        let err = ctx.gradient(1).unwrap_err();
        assert!(matches!(err, OxiflowError::MissingCalculator(_)));
    }

    #[test]
    fn gradient_with_wrong_type_returns_type_mismatch() {
        let mut ctx = ComputeContext::new(0.0, 0.01);
        // Stored as Scalar instead of ScalarField — misconfigured calculator
        ctx.insert(
            ContextVariable::SpatialGradient {
                dimension: 0,
                component: None,
            },
            ContextValue::Scalar(1.0),
        );
        let err = ctx.gradient(0).unwrap_err();
        assert!(matches!(
            err,
            OxiflowError::TypeMismatch {
                expected: "ScalarField",
                ..
            }
        ));
    }

    // ── external ─────────────────────────────────────────────────────────────

    #[test]
    fn external_returns_raw_context_value() {
        let ctx = ctx_with_all_variants();
        let val = ctx
            .external(ContextVariable::External { name: "flag" })
            .unwrap();
        assert!(matches!(val, ContextValue::Boolean(true)));
    }

    #[test]
    fn external_returns_missing_calculator_when_absent() {
        let ctx = ComputeContext::new(0.0, 0.01);
        let err = ctx
            .external(ContextVariable::External { name: "absent" })
            .unwrap_err();
        assert!(matches!(err, OxiflowError::MissingCalculator(_)));
    }

    #[test]
    fn external_works_for_any_variant_type() {
        let ctx = ctx_with_all_variants();
        // VectorField via external()
        let val = ctx
            .external(ContextVariable::External { name: "vfield" })
            .unwrap();
        assert!(val.is_vector_field());
    }

    // ── try_get ───────────────────────────────────────────────────────────────

    #[test]
    fn try_get_returns_some_when_present() {
        let ctx = ctx_with_all_variants();
        let val = ctx.try_get(ContextVariable::SpatialGradient {
            dimension: 0,
            component: None,
        });
        assert!(val.is_some());
        assert!(val.unwrap().is_scalar_field());
    }

    #[test]
    fn try_get_returns_none_when_absent() {
        let ctx = ComputeContext::new(0.0, 0.01);
        assert!(ctx.try_get(ContextVariable::Time).is_none());
    }

    // ── insert overwrites ─────────────────────────────────────────────────────

    #[test]
    fn insert_overwrites_previous_value() {
        let mut ctx = ComputeContext::new(0.0, 0.01);
        let var = ContextVariable::External { name: "x" };
        ctx.insert(var.clone(), ContextValue::Scalar(1.0));
        ctx.insert(var.clone(), ContextValue::Scalar(2.0));
        assert_eq!(ctx.scalar(var).unwrap(), 2.0);
    }
}
