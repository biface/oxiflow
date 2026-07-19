//! # Module `solver::sparse`
//!
//! Sparse linear system solver abstraction — DD-013 (second phase), DD-043.
//!
//! **`#[cfg(feature = "sparse")]` only** — this module does not exist in a
//! build without the `sparse` feature, and links no `faer` code in that
//! case (zero overhead for the cas simple, same posture as `parallel`/
//! `serde`/`hdf5`).
//!
//! ## Why a sibling trait, not an extension of `LinearSolver`
//!
//! [`crate::solver::linear::LinearSolver::solve`] takes `&DMatrix<f64>` — a
//! dense `nalgebra` type. `faer`'s sparse column matrix
//! (`faer::sparse::SparseColMat`) is structurally incompatible with
//! `DMatrix` without a conversion that would erase the memory benefit
//! sparsity exists to provide in the first place. `SparseLinearSolver`
//! therefore has its own `solve()` signature over the sparse type — the
//! same posture as [`crate::operators::FluxDivergenceOperator`] relative to
//! [`crate::operators::DiscreteOperator`] (DD-039): a sibling trait when
//! the two signatures genuinely diverge, not a sub-trait forcing one onto
//! the other (DD-043, "Décision — forme du trait", option C).
//!
//! `b` and the return value stay `nalgebra::DVector<f64>` — consistent with
//! the rest of the codebase; converting to/from `faer`'s own vector
//! representation is the implementation's job ([`FaerSparseSolver`]), a
//! negligible cost next to the sparse factorization itself.
//!
//! ## Verification status
//!
//! **Written without a working Rust toolchain for the `sparse` feature** —
//! this environment's `rustc`/`cargo` cannot resolve `faer`'s own
//! dependency graph (`edition2024` requirement upstream, unavailable
//! here). The exact `faer` 0.24 API surface used below (method names,
//! error types, `Triplet` construction) is transcribed from the
//! architecture notes carried over from the interrupted session
//! (`prompt_next_session.md`) and general `faer` familiarity, **not
//! confirmed against a real compile**. Before relying on this module:
//! `cargo doc --open -p faer --features sparse` (or the docs.rs page for
//! `faer` 0.24) to check `SparseColMat::try_new_from_triplets`, `Triplet`,
//! `sp_lu()`'s error type in particular — this file treats that error as
//! `impl Debug` via `{e:?}` deliberately, never matched on a specific
//! variant, so a signature mismatch there is the most likely first
//! compile error.

use nalgebra::DVector;

use crate::context::error::OxiflowError;

/// Solves a sparse linear system `A * x = b`.
///
/// Sibling to [`crate::solver::linear::LinearSolver`] — see the module docs
/// for why this is a separate trait rather than an extension.
pub trait SparseLinearSolver: Send + Sync {
    /// Solves `a * x = b` for `x`, where `a` is stored in a sparse format.
    ///
    /// # Errors
    ///
    /// Returns `OxiflowError::PreconditionFailed` if the factorization
    /// fails (singular or near-singular system).
    fn solve(
        &self,
        a: &faer::sparse::SparseColMat<usize, f64>,
        b: &DVector<f64>,
    ) -> Result<DVector<f64>, OxiflowError>;
}

/// Default sparse backend — LU factorization via `faer`.
///
/// Tested standalone against systems built directly in sparse form (e.g. a
/// tridiagonal system assembled as triplets) — not through
/// `solver::methods::implicit`'s finite-difference jacobian, which stays
/// dense (DD-043: banded assembly is a separate, later addition).
#[derive(Debug, Default, Clone, Copy)]
pub struct FaerSparseSolver;

impl SparseLinearSolver for FaerSparseSolver {
    fn solve(
        &self,
        a: &faer::sparse::SparseColMat<usize, f64>,
        b: &DVector<f64>,
    ) -> Result<DVector<f64>, OxiflowError> {
        let n = b.len();
        if a.nrows() != n || a.ncols() != n {
            return Err(OxiflowError::PreconditionFailed {
                context: "FaerSparseSolver",
                message: format!(
                    "dimension mismatch: a is {}x{}, b has length {n}",
                    a.nrows(),
                    a.ncols()
                ),
            });
        }

        // faer's own column vector type — b converted in, x converted back
        // out to nalgebra::DVector at the end. See module docs: this
        // conversion is unverified against a real faer 0.24 build.
        let b_faer = faer::Col::from_fn(n, |i| b[i]);

        let lu = a.sp_lu().map_err(|e| OxiflowError::PreconditionFailed {
            context: "FaerSparseSolver",
            message: format!(
                "sparse LU factorization failed (singular or near-singular system): {e:?}"
            ),
        })?;

        let x_faer = lu.solve(&b_faer);

        Ok(DVector::from_fn(n, |i, _| x_faer[i]))
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────
//
// Unverified by compilation (see module docs) — a first draft to check
// against a real `faer` 0.24 build before trusting these pass.

#[cfg(test)]
mod tests {
    use super::*;
    use faer::sparse::{SparseColMat, Triplet};

    /// Builds an `n x n` tridiagonal system `[-1, 2, -1]` (a standard
    /// discrete-Laplacian-like test matrix) directly in sparse form.
    fn tridiagonal(n: usize) -> SparseColMat<usize, f64> {
        let mut triplets = Vec::with_capacity(3 * n);
        for i in 0..n {
            triplets.push(Triplet::new(i, i, 2.0));
            if i > 0 {
                triplets.push(Triplet::new(i, i - 1, -1.0));
            }
            if i + 1 < n {
                triplets.push(Triplet::new(i, i + 1, -1.0));
            }
        }
        SparseColMat::try_new_from_triplets(n, n, &triplets)
            .expect("valid triplets for a tridiagonal system")
    }

    #[test]
    fn solves_small_tridiagonal_system() {
        let n = 5;
        let a = tridiagonal(n);
        let b = DVector::from_element(n, 1.0);
        let x = FaerSparseSolver.solve(&a, &b).unwrap();

        // Residual check rather than a hand-derived closed form — robust to
        // the exact API details above being slightly off in a way that
        // still produces a self-consistent solve.
        for i in 0..n {
            let mut residual = 2.0 * x[i];
            if i > 0 {
                residual -= x[i - 1];
            }
            if i + 1 < n {
                residual -= x[i + 1];
            }
            assert!(
                (residual - b[i]).abs() < 1e-9,
                "residual too large at row {i}: {residual} vs {}",
                b[i]
            );
        }
    }

    #[test]
    fn solves_500x500_tridiagonal_system() {
        // Acceptance criterion from #50: faer-sparse solves Ax=b for a
        // 500x500 tridiagonal system.
        let n = 500;
        let a = tridiagonal(n);
        let b = DVector::from_element(n, 1.0);
        let x = FaerSparseSolver.solve(&a, &b).unwrap();

        for i in 0..n {
            let mut residual = 2.0 * x[i];
            if i > 0 {
                residual -= x[i - 1];
            }
            if i + 1 < n {
                residual -= x[i + 1];
            }
            assert!((residual - b[i]).abs() < 1e-6);
        }
    }

    #[test]
    fn dimension_mismatch_returns_error() {
        let a = tridiagonal(5);
        let b = DVector::from_element(3, 1.0);
        let err = FaerSparseSolver.solve(&a, &b).unwrap_err();
        assert!(matches!(err, OxiflowError::PreconditionFailed { .. }));
    }
}
