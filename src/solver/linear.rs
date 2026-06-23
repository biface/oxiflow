//! # Module `solver::linear`
//!
//! Linear system solver abstraction — DD-013.
//!
//! Lives at `solver::linear`, not nested under `solver::methods`: solving
//! `A * x = b` knows nothing about `t`, `dt`, or any `Domain` — it is a
//! generic service, not a temporal-integration concern. The implicit
//! integrators ([`crate::solver::methods::implicit`]) are its first
//! consumer, but DD-013 already anticipates a second one unrelated to time
//! integration: sparse FEM systems at v2.0/J7, via a `faer` backend
//! implementing this same trait.
//!
//! [`NalgebraDenseSolver`] (dense LU) is the default for the small systems
//! implicit integrators produce at J4a. The sparse `faer` backend arrives
//! at v0.5.0 (DD-013, second phase) as an additional implementation of
//! this same trait — no change needed here when it does.

use nalgebra::{DMatrix, DVector};

use crate::context::error::OxiflowError;

/// Solves a dense linear system `A * x = b`.
///
/// Implementors may assume `a` is square and `b.len() == a.nrows()` —
/// callers are responsible for that invariant (see
/// [`crate::solver::methods::implicit`]).
pub trait LinearSolver: Send + Sync {
    /// Solves `a * x = b` for `x`.
    ///
    /// # Errors
    ///
    /// Returns `OxiflowError::PreconditionFailed` if the system is singular
    /// or near-singular (decomposition failure).
    fn solve(&self, a: &DMatrix<f64>, b: &DVector<f64>) -> Result<DVector<f64>, OxiflowError>;
}

/// Default backend — dense LU decomposition via `nalgebra`.
///
/// Appropriate for the small, dense systems implicit integrators (Backward
/// Euler, Crank-Nicolson) produce before `DiscreteOperator` (INV-2,
/// v0.5.0+) introduces large sparse operators.
#[derive(Debug, Default, Clone, Copy)]
pub struct NalgebraDenseSolver;

impl LinearSolver for NalgebraDenseSolver {
    fn solve(&self, a: &DMatrix<f64>, b: &DVector<f64>) -> Result<DVector<f64>, OxiflowError> {
        a.clone()
            .lu()
            .solve(b)
            .ok_or_else(|| OxiflowError::PreconditionFailed {
                context: "NalgebraDenseSolver",
                message: "linear system is singular or near-singular (LU decomposition failed)"
                    .into(),
            })
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn solves_identity_system() {
        let a = DMatrix::<f64>::identity(3, 3);
        let b = DVector::from_vec(vec![1.0, 2.0, 3.0]);
        let x = NalgebraDenseSolver.solve(&a, &b).unwrap();
        assert!((x[0] - 1.0).abs() < 1e-12);
        assert!((x[1] - 2.0).abs() < 1e-12);
        assert!((x[2] - 3.0).abs() < 1e-12);
    }

    #[test]
    fn solves_diagonal_system() {
        // 2x = 4, 5y = 10 -> x=2, y=2
        let a = DMatrix::from_diagonal(&DVector::from_vec(vec![2.0, 5.0]));
        let b = DVector::from_vec(vec![4.0, 10.0]);
        let x = NalgebraDenseSolver.solve(&a, &b).unwrap();
        assert!((x[0] - 2.0).abs() < 1e-12);
        assert!((x[1] - 2.0).abs() < 1e-12);
    }

    #[test]
    fn singular_system_returns_error() {
        // Rank-deficient: second row is a multiple of the first.
        let a = DMatrix::from_row_slice(2, 2, &[1.0, 1.0, 2.0, 2.0]);
        let b = DVector::from_vec(vec![1.0, 2.0]);
        let err = NalgebraDenseSolver.solve(&a, &b).unwrap_err();
        assert!(matches!(err, OxiflowError::PreconditionFailed { .. }));
    }
}
