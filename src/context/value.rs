//! # Module `context::value`
//!
//! Typed values carried by the compute context (DD-003, issue #27).
//!
//! ## Two orthogonal axes
//!
//! `ContextValue` is structured along two independent axes:
//!
//! - **Rank** (algebraic): scalar (rank 0), vector (rank 1), matrix (rank 2 tensor).
//! - **Distribution**: pointwise algebraic object vs. nodal field (one value per mesh node).
//!
//! ## Reserved variants (J7)
//!
//! `Tensor4` (rank-4 tensor C_ijkl) and `TensorField` (rank-2 tensor per node) are
//! deliberately absent. They require `DiscreteOperator` (INV-2, J5) for covariant
//! transformation semantics and will be added at J7. Tensors of rank > 4 are out of
//! scope for all target physics; extension for third-party frameworks follows INV-4.
//!
//! ## Distinction from `PhysicalData`
//!
//! `PhysicalData` is the primary field `u` discretised on the mesh — a dimensionality-
//! based array container with no tensor semantics. `ContextValue` holds coefficients,
//! parameters and derived quantities consumed by physical models during time integration.

use nalgebra::{DMatrix, DVector};

use crate::context::error::OxiflowError;

/// Typed value of a context variable.
///
/// # Pointwise algebraic objects
///
/// | Variant | Rank | Example |
/// |---------|------|---------|
/// | `Scalar` | 0 | time, axial dispersion coefficient D_ax |
/// | `Boolean` | — | convergence flag, saturation condition |
/// | `Vector` | 1 | velocity at a point, pointwise gradient |
/// | `Matrix` | 2 | diffusion tensor D_ij, stress tensor σ_ij |
///
/// # Nodal fields
///
/// | Variant | Content | Example |
/// |---------|---------|---------|
/// | `ScalarField` | `DVector` — one scalar per node | porosity(x), T(x) |
/// | `VectorField` | `DMatrix` — n_nodes × dim | velocity field u(x,y) |
///
/// # Reserved (J7 — requires INV-2 / DiscreteOperator)
///
/// ```text
/// // Tensor4(Array4<f64>)  — rank-4 tensor C_ijkl (elastic stiffness)
/// // TensorField(...)      — rank-2 tensor per node D_ij(x)
/// // Tensors of rank > 4 are out of scope; extension via INV-4 for third-party frameworks.
/// ```
///
/// # Examples
///
/// ```rust
/// use oxiflow::context::value::ContextValue;
/// use nalgebra::DVector;
///
/// let t   = ContextValue::Scalar(1.5);
/// let flag = ContextValue::Boolean(false);
/// let field = ContextValue::ScalarField(DVector::from_vec(vec![0.1, 0.2, 0.3]));
///
/// assert_eq!(t.as_scalar().unwrap(), 1.5);
/// assert!(!field.as_scalar_field().unwrap().is_empty());
/// ```
#[derive(Debug, Clone, PartialEq)]
pub enum ContextValue {
    // ── Pointwise algebraic objects ───────────────────────────────────────────
    /// Rank-0 scalar: time, step size, uniform coefficient.
    Scalar(f64),

    /// Logical flag: convergence condition, saturation state.
    Boolean(bool),

    /// Rank-1 vector: pointwise velocity, pointwise gradient.
    Vector(DVector<f64>),

    /// Rank-2 tensor: diffusion tensor D_ij, stress σ_ij, permeability K_ij.
    ///
    /// Covariant transformation (`D' = J·D·Jᵀ`) is the caller's responsibility
    /// via `DiscreteOperator` (INV-2, J7). This variant stores components in the
    /// current reference frame only.
    Matrix(DMatrix<f64>),

    // ── Nodal fields ──────────────────────────────────────────────────────────
    /// One scalar value per mesh node: porosity(x), temperature field T(x).
    ///
    /// Length equals the number of degrees of freedom (`Mesh::n_dof()`).
    ScalarField(DVector<f64>),

    /// One vector per mesh node, stored as `n_nodes × spatial_dim` matrix.
    ///
    /// Row `i` holds the vector at node `i`.
    VectorField(DMatrix<f64>),
    // Reserved for J7 — requires DiscreteOperator (INV-2):
    // Tensor4(ndarray::Array4<f64>)  — rank-4 tensor C_ijkl
    // TensorField(...)               — rank-2 tensor per node D_ij(x)
    // Rank > 4: out of scope; extension via INV-4 for third-party frameworks.
}

impl ContextValue {
    // ── Variant name (used in TypeMismatch errors) ────────────────────────────

    /// Returns the variant name as a static string.
    ///
    /// Used to build `OxiflowError::TypeMismatch` messages.
    pub fn variant_name(&self) -> &'static str {
        match self {
            Self::Scalar(_) => "Scalar",
            Self::Boolean(_) => "Boolean",
            Self::Vector(_) => "Vector",
            Self::Matrix(_) => "Matrix",
            Self::ScalarField(_) => "ScalarField",
            Self::VectorField(_) => "VectorField",
        }
    }

    // ── Pointwise accessors ───────────────────────────────────────────────────

    /// Unwraps the inner `f64` of a `Scalar` variant.
    ///
    /// # Errors
    ///
    /// Returns `OxiflowError::TypeMismatch` if the variant is not `Scalar`.
    pub fn as_scalar(&self) -> Result<f64, OxiflowError> {
        match self {
            Self::Scalar(v) => Ok(*v),
            other => Err(OxiflowError::TypeMismatch {
                expected: "Scalar",
                actual: other.variant_name(),
            }),
        }
    }

    /// Unwraps the inner `bool` of a `Boolean` variant.
    ///
    /// # Errors
    ///
    /// Returns `OxiflowError::TypeMismatch` if the variant is not `Boolean`.
    pub fn as_bool(&self) -> Result<bool, OxiflowError> {
        match self {
            Self::Boolean(v) => Ok(*v),
            other => Err(OxiflowError::TypeMismatch {
                expected: "Boolean",
                actual: other.variant_name(),
            }),
        }
    }

    /// Borrows the inner `DVector` of a `Vector` variant.
    ///
    /// # Errors
    ///
    /// Returns `OxiflowError::TypeMismatch` if the variant is not `Vector`.
    pub fn as_vector(&self) -> Result<&DVector<f64>, OxiflowError> {
        match self {
            Self::Vector(v) => Ok(v),
            other => Err(OxiflowError::TypeMismatch {
                expected: "Vector",
                actual: other.variant_name(),
            }),
        }
    }

    /// Borrows the inner `DMatrix` of a `Matrix` variant.
    ///
    /// # Errors
    ///
    /// Returns `OxiflowError::TypeMismatch` if the variant is not `Matrix`.
    pub fn as_matrix(&self) -> Result<&DMatrix<f64>, OxiflowError> {
        match self {
            Self::Matrix(m) => Ok(m),
            other => Err(OxiflowError::TypeMismatch {
                expected: "Matrix",
                actual: other.variant_name(),
            }),
        }
    }

    // ── Nodal field accessors ─────────────────────────────────────────────────

    /// Borrows the inner `DVector` of a `ScalarField` variant.
    ///
    /// # Errors
    ///
    /// Returns `OxiflowError::TypeMismatch` if the variant is not `ScalarField`.
    pub fn as_scalar_field(&self) -> Result<&DVector<f64>, OxiflowError> {
        match self {
            Self::ScalarField(v) => Ok(v),
            other => Err(OxiflowError::TypeMismatch {
                expected: "ScalarField",
                actual: other.variant_name(),
            }),
        }
    }

    /// Borrows the inner `DMatrix` of a `VectorField` variant.
    ///
    /// Rows correspond to nodes, columns to spatial dimensions.
    ///
    /// # Errors
    ///
    /// Returns `OxiflowError::TypeMismatch` if the variant is not `VectorField`.
    pub fn as_vector_field(&self) -> Result<&DMatrix<f64>, OxiflowError> {
        match self {
            Self::VectorField(m) => Ok(m),
            other => Err(OxiflowError::TypeMismatch {
                expected: "VectorField",
                actual: other.variant_name(),
            }),
        }
    }

    // ── Type predicates ───────────────────────────────────────────────────────

    /// Returns `true` if this is a `Scalar`.
    pub fn is_scalar(&self) -> bool {
        matches!(self, Self::Scalar(_))
    }
    /// Returns `true` if this is a `Boolean`.
    pub fn is_bool(&self) -> bool {
        matches!(self, Self::Boolean(_))
    }
    /// Returns `true` if this is a `Vector`.
    pub fn is_vector(&self) -> bool {
        matches!(self, Self::Vector(_))
    }
    /// Returns `true` if this is a `Matrix`.
    pub fn is_matrix(&self) -> bool {
        matches!(self, Self::Matrix(_))
    }
    /// Returns `true` if this is a `ScalarField`.
    pub fn is_scalar_field(&self) -> bool {
        matches!(self, Self::ScalarField(_))
    }
    /// Returns `true` if this is a `VectorField`.
    pub fn is_vector_field(&self) -> bool {
        matches!(self, Self::VectorField(_))
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::{DMatrix, DVector};

    // ── Helpers ───────────────────────────────────────────────────────────────

    fn scalar() -> ContextValue {
        ContextValue::Scalar(3.14)
    }
    fn boolean() -> ContextValue {
        ContextValue::Boolean(true)
    }
    fn vector() -> ContextValue {
        ContextValue::Vector(DVector::from_vec(vec![1.0, 2.0, 3.0]))
    }
    fn matrix() -> ContextValue {
        ContextValue::Matrix(DMatrix::from_element(2, 2, 1.0))
    }
    fn scalar_field() -> ContextValue {
        ContextValue::ScalarField(DVector::from_vec(vec![0.1, 0.2]))
    }
    fn vector_field() -> ContextValue {
        ContextValue::VectorField(DMatrix::from_element(3, 2, 0.5))
    }

    fn all_variants() -> Vec<ContextValue> {
        vec![
            scalar(),
            boolean(),
            vector(),
            matrix(),
            scalar_field(),
            vector_field(),
        ]
    }

    // ── variant_name ─────────────────────────────────────────────────────────

    #[test]
    fn variant_names_are_correct() {
        assert_eq!(scalar().variant_name(), "Scalar");
        assert_eq!(boolean().variant_name(), "Boolean");
        assert_eq!(vector().variant_name(), "Vector");
        assert_eq!(matrix().variant_name(), "Matrix");
        assert_eq!(scalar_field().variant_name(), "ScalarField");
        assert_eq!(vector_field().variant_name(), "VectorField");
    }

    // ── as_scalar ─────────────────────────────────────────────────────────────

    #[test]
    fn as_scalar_on_scalar_returns_value() {
        assert_eq!(scalar().as_scalar().unwrap(), 3.14);
    }

    #[test]
    fn as_scalar_on_wrong_variant_returns_type_mismatch() {
        for v in [
            boolean(),
            vector(),
            matrix(),
            scalar_field(),
            vector_field(),
        ] {
            let err = v.as_scalar().unwrap_err();
            assert!(
                matches!(
                    err,
                    OxiflowError::TypeMismatch {
                        expected: "Scalar",
                        ..
                    }
                ),
                "expected TypeMismatch for {:?}",
                v
            );
        }
    }

    // ── as_bool ───────────────────────────────────────────────────────────────

    #[test]
    fn as_bool_on_boolean_returns_value() {
        assert!(boolean().as_bool().unwrap());
        assert!(!ContextValue::Boolean(false).as_bool().unwrap());
    }

    #[test]
    fn as_bool_on_wrong_variant_returns_type_mismatch() {
        for v in [scalar(), vector(), matrix(), scalar_field(), vector_field()] {
            let err = v.as_bool().unwrap_err();
            assert!(matches!(
                err,
                OxiflowError::TypeMismatch {
                    expected: "Boolean",
                    ..
                }
            ));
        }
    }

    // ── as_vector ─────────────────────────────────────────────────────────────

    #[test]
    fn as_vector_on_vector_returns_reference() {
        let v = vector();
        let inner = v.as_vector().unwrap();
        assert_eq!(inner.len(), 3);
        assert_eq!(inner[0], 1.0);
    }

    #[test]
    fn as_vector_on_wrong_variant_returns_type_mismatch() {
        for v in [
            scalar(),
            boolean(),
            matrix(),
            scalar_field(),
            vector_field(),
        ] {
            let err = v.as_vector().unwrap_err();
            assert!(matches!(
                err,
                OxiflowError::TypeMismatch {
                    expected: "Vector",
                    ..
                }
            ));
        }
    }

    // ── as_matrix ─────────────────────────────────────────────────────────────

    #[test]
    fn as_matrix_on_matrix_returns_reference() {
        let m = matrix();
        let inner = m.as_matrix().unwrap();
        assert_eq!(inner.shape(), (2, 2));
    }

    #[test]
    fn as_matrix_on_wrong_variant_returns_type_mismatch() {
        for v in [
            scalar(),
            boolean(),
            vector(),
            scalar_field(),
            vector_field(),
        ] {
            let err = v.as_matrix().unwrap_err();
            assert!(matches!(
                err,
                OxiflowError::TypeMismatch {
                    expected: "Matrix",
                    ..
                }
            ));
        }
    }

    // ── as_scalar_field ───────────────────────────────────────────────────────

    #[test]
    fn as_scalar_field_on_scalar_field_returns_reference() {
        let sf = scalar_field();
        let inner = sf.as_scalar_field().unwrap();
        assert_eq!(inner.len(), 2);
        assert!((inner[0] - 0.1).abs() < 1e-12);
    }

    #[test]
    fn as_scalar_field_on_wrong_variant_returns_type_mismatch() {
        for v in [scalar(), boolean(), vector(), matrix(), vector_field()] {
            let err = v.as_scalar_field().unwrap_err();
            assert!(matches!(
                err,
                OxiflowError::TypeMismatch {
                    expected: "ScalarField",
                    ..
                }
            ));
        }
    }

    // ── as_vector_field ───────────────────────────────────────────────────────

    #[test]
    fn as_vector_field_on_vector_field_returns_reference() {
        let vf = vector_field();
        let inner = vf.as_vector_field().unwrap();
        assert_eq!(inner.shape(), (3, 2));
    }

    #[test]
    fn as_vector_field_on_wrong_variant_returns_type_mismatch() {
        for v in [scalar(), boolean(), vector(), matrix(), scalar_field()] {
            let err = v.as_vector_field().unwrap_err();
            assert!(matches!(
                err,
                OxiflowError::TypeMismatch {
                    expected: "VectorField",
                    ..
                }
            ));
        }
    }

    // ── Type predicates ───────────────────────────────────────────────────────

    #[test]
    fn is_predicates_return_true_for_correct_variant() {
        assert!(scalar().is_scalar());
        assert!(boolean().is_bool());
        assert!(vector().is_vector());
        assert!(matrix().is_matrix());
        assert!(scalar_field().is_scalar_field());
        assert!(vector_field().is_vector_field());
    }

    #[test]
    fn is_scalar_returns_false_for_other_variants() {
        for v in [
            boolean(),
            vector(),
            matrix(),
            scalar_field(),
            vector_field(),
        ] {
            assert!(!v.is_scalar(), "expected false for {:?}", v);
        }
    }

    #[test]
    fn is_scalar_field_returns_false_for_scalar() {
        assert!(!scalar().is_scalar_field());
    }

    // ── Clone & PartialEq ─────────────────────────────────────────────────────

    #[test]
    fn clone_preserves_equality() {
        for v in all_variants() {
            assert_eq!(v.clone(), v);
        }
    }

    #[test]
    fn distinct_variants_are_not_equal() {
        assert_ne!(scalar(), boolean());
        assert_ne!(vector(), scalar_field()); // same inner type, different semantic
        assert_ne!(matrix(), vector_field());
    }

    #[test]
    fn scalar_values_compared_by_content() {
        assert_eq!(ContextValue::Scalar(1.0), ContextValue::Scalar(1.0));
        assert_ne!(ContextValue::Scalar(1.0), ContextValue::Scalar(2.0));
    }

    // ── Debug ─────────────────────────────────────────────────────────────────

    #[test]
    fn debug_is_non_empty_for_all_variants() {
        for v in all_variants() {
            assert!(!format!("{:?}", v).is_empty());
        }
    }
}
