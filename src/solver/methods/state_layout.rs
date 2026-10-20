//! # Module `solver::methods::state_layout`
//!
//! Ordering of the unknowns of a state for the implicit integrators (#192).
//!
//! Newton's method and the finite-difference Jacobians work on one flat
//! vector of unknowns. A `ScalarField` already is one. A `VectorField`
//! (n nodes × d components, one row per node) is flattened **interleaved by
//! node**:
//!
//! $$\left(u_0^{(0)}, \dots, u_0^{(d-1)},\ u_1^{(0)}, \dots, u_1^{(d-1)},\ \dots\right)$$
//!
//! so the unknowns of one node are contiguous. A stencil of half-width `s`
//! couples node `i` to nodes `i − s … i + s`, and every component of a node
//! to every other: the Jacobian stays banded, with half-width
//! `k = d·(s + 1) − 1` in flat unknowns (`k = s` for a `ScalarField`, d = 1).
//! That `k` is the value to give `with_jacobian_bandwidth`.
//!
//! The ordering is internal to the solver (HOW, INV-8): models and
//! calculators only ever see `ContextValue` states. The general numbering
//! of degrees of freedom for the finite-element work stays with DA-049
//! (#150), DT-055 (#157) and #133.

use std::borrow::Cow;

use nalgebra::{DMatrix, DVector};

use crate::context::error::OxiflowError;
use crate::context::value::ContextValue;

/// Shape of a state, captured once per step, with the conversions between
/// the state and its flat vector of unknowns. See the [module docs](self)
/// for the ordering.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum StateLayout {
    /// `ScalarField` of `n` nodes: the flat vector is the field itself.
    Scalar { nodes: usize },
    /// `VectorField` of `nodes` × `components`, flattened by node.
    Vector { nodes: usize, components: usize },
}

impl StateLayout {
    /// Layout of `state`.
    ///
    /// # Errors
    ///
    /// `OxiflowError::TypeMismatch` if `state` is neither `ScalarField`
    /// nor `VectorField`.
    pub(crate) fn of(state: &ContextValue) -> Result<Self, OxiflowError> {
        match state {
            ContextValue::ScalarField(v) => Ok(Self::Scalar { nodes: v.len() }),
            ContextValue::VectorField(m) => Ok(Self::Vector {
                nodes: m.nrows(),
                components: m.ncols(),
            }),
            other => Err(OxiflowError::TypeMismatch {
                expected: "ScalarField or VectorField",
                actual: other.variant_name(),
            }),
        }
    }

    /// Number of unknowns in the flat vector.
    pub(crate) fn len(&self) -> usize {
        match *self {
            Self::Scalar { nodes } => nodes,
            Self::Vector { nodes, components } => nodes * components,
        }
    }

    /// Flat vector of unknowns of `value`, which must have this layout.
    /// Borrows a `ScalarField` as is; a `VectorField` is copied in node
    /// order.
    ///
    /// # Errors
    ///
    /// - `OxiflowError::TypeMismatch` if `value` is not the variant of this
    ///   layout.
    /// - `OxiflowError::InvalidDomain` if its shape differs.
    pub(crate) fn flatten<'a>(
        &self,
        value: &'a ContextValue,
    ) -> Result<Cow<'a, DVector<f64>>, OxiflowError> {
        match (*self, value) {
            (Self::Scalar { nodes }, ContextValue::ScalarField(v)) => {
                if v.len() != nodes {
                    return Err(OxiflowError::InvalidDomain(format!(
                        "state length {} != expected length {nodes}",
                        v.len()
                    )));
                }
                Ok(Cow::Borrowed(v))
            }
            (Self::Vector { nodes, components }, ContextValue::VectorField(m)) => {
                if m.shape() != (nodes, components) {
                    return Err(OxiflowError::InvalidDomain(format!(
                        "state shape {:?} != expected shape {:?}",
                        m.shape(),
                        (nodes, components)
                    )));
                }
                // Column-major storage of the transpose (d × n) is exactly
                // the node-interleaved order.
                Ok(Cow::Owned(DVector::from_column_slice(
                    m.transpose().as_slice(),
                )))
            }
            (layout, other) => Err(OxiflowError::TypeMismatch {
                expected: layout.variant_name(),
                actual: other.variant_name(),
            }),
        }
    }

    /// State of this layout whose flat vector of unknowns is `flat`.
    ///
    /// # Errors
    ///
    /// `OxiflowError::InvalidDomain` if `flat.len() != self.len()`.
    pub(crate) fn rebuild(&self, flat: DVector<f64>) -> Result<ContextValue, OxiflowError> {
        if flat.len() != self.len() {
            return Err(OxiflowError::InvalidDomain(format!(
                "flat vector length {} != {} unknowns",
                flat.len(),
                self.len()
            )));
        }
        Ok(match *self {
            Self::Scalar { .. } => ContextValue::ScalarField(flat),
            Self::Vector { nodes, components } => ContextValue::VectorField(
                DMatrix::from_row_slice(nodes, components, flat.as_slice()),
            ),
        })
    }

    fn variant_name(&self) -> &'static str {
        match self {
            Self::Scalar { .. } => "ScalarField",
            Self::Vector { .. } => "VectorField",
        }
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn vector_field() -> ContextValue {
        // 3 nodes × 2 components: node i holds (10·i, 10·i + 1).
        ContextValue::VectorField(DMatrix::from_row_slice(
            3,
            2,
            &[0.0, 1.0, 10.0, 11.0, 20.0, 21.0],
        ))
    }

    #[test]
    fn vector_field_is_flattened_by_node() {
        let state = vector_field();
        let layout = StateLayout::of(&state).unwrap();
        assert_eq!(
            layout,
            StateLayout::Vector {
                nodes: 3,
                components: 2
            }
        );
        assert_eq!(layout.len(), 6);
        let flat = layout.flatten(&state).unwrap();
        assert_eq!(flat.as_slice(), &[0.0, 1.0, 10.0, 11.0, 20.0, 21.0]);
    }

    #[test]
    fn rebuild_inverts_flatten() {
        let state = vector_field();
        let layout = StateLayout::of(&state).unwrap();
        let flat = layout.flatten(&state).unwrap().into_owned();
        assert_eq!(layout.rebuild(flat).unwrap(), state);
    }

    #[test]
    fn scalar_field_is_borrowed_as_is() {
        let state = ContextValue::ScalarField(DVector::from_vec(vec![1.0, 2.0, 3.0]));
        let layout = StateLayout::of(&state).unwrap();
        let flat = layout.flatten(&state).unwrap();
        assert!(matches!(flat, Cow::Borrowed(_)));
        assert_eq!(layout.rebuild(flat.into_owned()).unwrap(), state);
    }

    #[test]
    fn flatten_rejects_other_shape_or_variant() {
        let layout = StateLayout::of(&vector_field()).unwrap();
        let other_shape = ContextValue::VectorField(DMatrix::zeros(2, 3));
        let scalar = ContextValue::ScalarField(DVector::zeros(6));
        assert!(matches!(
            layout.flatten(&other_shape),
            Err(OxiflowError::InvalidDomain(_))
        ));
        assert!(matches!(
            layout.flatten(&scalar),
            Err(OxiflowError::TypeMismatch { .. })
        ));
    }

    #[test]
    fn rebuild_rejects_wrong_length_and_of_rejects_non_fields() {
        let layout = StateLayout::of(&vector_field()).unwrap();
        assert!(matches!(
            layout.rebuild(DVector::zeros(5)),
            Err(OxiflowError::InvalidDomain(_))
        ));
        assert!(matches!(
            StateLayout::of(&ContextValue::Scalar(1.0)),
            Err(OxiflowError::TypeMismatch { .. })
        ));
    }
}
