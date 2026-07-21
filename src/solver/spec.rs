//! # Module `solver::spec`
//!
//! External, serialisable DTO describing *which* integration method to
//! build and with what parameters — the HOW-axis config-loading increment
//! of DD-027 amendment 1 (issue #104).
//!
//! ## Not `IntegratorKind`
//!
//! [`IntegratorSpec`] is deliberately a separate type from
//! [`super::IntegratorKind`]. The latter is a pure descriptive label,
//! already carried by [`super::SolverConfiguration`] and
//! [`super::snapshot::SimulationSnapshot`] — it is never consumed to
//! *build* a solver. `IntegratorSpec` exists specifically to be consumed:
//! [`TryFrom<IntegratorSpec> for Box<dyn Solver>`](TryFrom) bridges it to
//! the existing `BackwardEulerSolver` builder, with no duplicated
//! construction logic. Naming history: this type was called `SolverConfig`
//! in issue #104's original body — renamed before implementation to avoid
//! colliding with [`super::SolverConfiguration`], which already covers the
//! full HOW pole (`TimeConfiguration` + calculators); see #104's closing
//! comment for the rationale.
//!
//! ## Scope of this first increment
//!
//! Only [`IntegratorSpec::BackwardEuler`], covering the DD-043 (#50) sparse
//! dispatch path — this whole module is gated behind the `sparse` feature:
//! without it, a config DTO for `BackwardEuler` adds nothing over the
//! existing [`super::IntegratorKind::BackwardEuler`] label, since the dense
//! path takes no parameters. Remaining integrators (`RK4`, `CrankNicolson`,
//! `BDF2`, `DoPri45`) and the WHAT axis (`ModelConfig`/`MeshConfig`/
//! `BoundaryConditionConfig`) are tracked as follow-up, not blocking this
//! issue's closure (per #104's own stated scope).
//!
//! The sparse backend itself is always [`FaerSparseSolver`] — a fixed,
//! non-configurable choice: `Box<dyn SparseLinearSolver>` is a trait
//! object and cannot be deserialised, so there is no way for a config file
//! to name an arbitrary backend (see #104's discussion of this
//! constraint).

use super::methods::backward_euler::BackwardEulerSolver;
use super::sparse::FaerSparseSolver;
use super::Solver;
use crate::context::error::OxiflowError;

/// External DTO describing which integrator to build and with what
/// parameters. See the [module docs](self) for why this is not
/// `IntegratorKind`, and for this increment's deliberately narrow scope.
#[non_exhaustive]
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Deserialize))]
pub enum IntegratorSpec {
    /// Backward Euler (DD-... implicit method), optionally dispatching to
    /// the sparse linear solver path (DD-043, #50) above `sparse_threshold`
    /// once `jacobian_bandwidth` is also supplied. The sparse backend
    /// itself is always [`FaerSparseSolver`] (see [module docs](self)).
    BackwardEuler {
        /// System size above which the sparse path is used — see
        /// [`BackwardEulerSolver::with_sparse_threshold`].
        #[cfg_attr(feature = "serde", serde(default = "default_sparse_threshold"))]
        sparse_threshold: usize,
        /// Half-bandwidth of the Jacobian (e.g. `scheme.stencil_radius()`,
        /// DD-039) — see [`BackwardEulerSolver::with_jacobian_bandwidth`].
        /// `None` keeps the dense path regardless of `sparse_threshold`.
        #[cfg_attr(feature = "serde", serde(default))]
        jacobian_bandwidth: Option<usize>,
    },
}

/// Default for `sparse_threshold` when omitted from a deserialised config.
///
/// Duplicates the literal `100` from `BackwardEulerSolver`'s private
/// `Default` impl (`backward_euler.rs`) — that field isn't publicly
/// readable, so this can't delegate to it directly. If that default ever
/// changes, update this constant too.
#[cfg(feature = "serde")]
fn default_sparse_threshold() -> usize {
    100
}

impl TryFrom<IntegratorSpec> for Box<dyn Solver> {
    type Error = OxiflowError;

    fn try_from(spec: IntegratorSpec) -> Result<Self, Self::Error> {
        match spec {
            IntegratorSpec::BackwardEuler {
                sparse_threshold,
                jacobian_bandwidth,
            } => {
                let mut solver = BackwardEulerSolver::new().with_sparse_threshold(sparse_threshold);
                if let Some(bandwidth) = jacobian_bandwidth {
                    solver = solver
                        .with_jacobian_bandwidth(bandwidth)
                        .with_sparse_solver(Box::new(FaerSparseSolver));
                }
                Ok(Box::new(solver))
            }
        }
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dense_path_when_no_jacobian_bandwidth() {
        let spec = IntegratorSpec::BackwardEuler {
            sparse_threshold: 100,
            jacobian_bandwidth: None,
        };
        let solver: Box<dyn Solver> = spec.try_into().unwrap();
        // Object-safety + successful construction is the assertion here —
        // BackwardEulerSolver's internal sparse_solver field is private,
        // not observable from outside the crate.
        let _ = solver;
    }

    #[test]
    fn sparse_path_when_jacobian_bandwidth_given() {
        let spec = IntegratorSpec::BackwardEuler {
            sparse_threshold: 50,
            jacobian_bandwidth: Some(3),
        };
        let solver: Box<dyn Solver> = spec.try_into().unwrap();
        let _ = solver;
    }

    #[cfg(feature = "serde")]
    #[test]
    fn deserialises_with_defaults() {
        let json = r#"{ "BackwardEuler": {} }"#;
        let spec: IntegratorSpec = serde_json::from_str(json).unwrap();
        match spec {
            IntegratorSpec::BackwardEuler {
                sparse_threshold,
                jacobian_bandwidth,
            } => {
                assert_eq!(sparse_threshold, default_sparse_threshold());
                assert_eq!(jacobian_bandwidth, None);
            }
        }
    }

    #[cfg(feature = "serde")]
    #[test]
    fn deserialises_explicit_fields() {
        let json = r#"{ "BackwardEuler": { "sparse_threshold": 42, "jacobian_bandwidth": 2 } }"#;
        let spec: IntegratorSpec = serde_json::from_str(json).unwrap();
        match spec {
            IntegratorSpec::BackwardEuler {
                sparse_threshold,
                jacobian_bandwidth,
            } => {
                assert_eq!(sparse_threshold, 42);
                assert_eq!(jacobian_bandwidth, Some(2));
            }
        }
    }
}
