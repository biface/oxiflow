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
//! issue's closure (per #104's own stated scope). `newton_convergence`/
//! `jacobian_strategy`/`max_iterations` (DD-044, #114) extend this same
//! variant rather than waiting for `CrankNicolson`/`BDF2` variants to land
//! (#115/#116) — the Newton configuration is orthogonal to which
//! integrator carries it.
//!
//! The sparse backend itself is always [`FaerSparseSolver`] — a fixed,
//! non-configurable choice: `Box<dyn SparseLinearSolver>` is a trait
//! object and cannot be deserialised, so there is no way for a config file
//! to name an arbitrary backend (see #104's discussion of this
//! constraint).

use super::methods::backward_euler::BackwardEulerSolver;
use super::methods::implicit::stiff_jacobian_convergence;
use super::methods::newton::{JacobianStrategy, NewtonConvergence};
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
        /// Newton convergence criterion (DD-044, #114) — see
        /// [`BackwardEulerSolver::with_newton_convergence`]. Defaults to
        /// [`stiff_jacobian_convergence`], not [`NewtonConvergence::default`]
        /// — matching `BackwardEulerSolver`'s own zero-config value (see
        /// that function's docs for why they differ).
        #[cfg_attr(feature = "serde", serde(default = "default_newton_convergence"))]
        newton_convergence: NewtonConvergence,
        /// Jacobian refresh strategy (DD-044, #114) — see
        /// [`BackwardEulerSolver::with_jacobian_strategy`].
        #[cfg_attr(feature = "serde", serde(default))]
        jacobian_strategy: JacobianStrategy,
        /// Newton iteration budget (DD-044, #114) — see
        /// [`BackwardEulerSolver::with_max_newton_iterations`]. Defaults
        /// to `1`, matching `BackwardEulerSolver`'s own zero-config value
        /// (the pre-DD-044 single-correction contract).
        #[cfg_attr(feature = "serde", serde(default = "default_max_newton_iterations"))]
        max_iterations: usize,
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

/// Default for `newton_convergence` when omitted — see the field's own
/// docs for why this is [`stiff_jacobian_convergence`], not
/// [`NewtonConvergence::default`].
#[cfg(feature = "serde")]
fn default_newton_convergence() -> NewtonConvergence {
    stiff_jacobian_convergence()
}

/// Default for `max_iterations` when omitted — see the field's own docs.
#[cfg(feature = "serde")]
fn default_max_newton_iterations() -> usize {
    1
}

impl TryFrom<IntegratorSpec> for Box<dyn Solver> {
    type Error = OxiflowError;

    fn try_from(spec: IntegratorSpec) -> Result<Self, Self::Error> {
        match spec {
            IntegratorSpec::BackwardEuler {
                sparse_threshold,
                jacobian_bandwidth,
                newton_convergence,
                jacobian_strategy,
                max_iterations,
            } => {
                let mut solver = BackwardEulerSolver::new()
                    .with_sparse_threshold(sparse_threshold)
                    .with_newton_convergence(newton_convergence)
                    .with_jacobian_strategy(jacobian_strategy)
                    .with_max_newton_iterations(max_iterations);
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
            newton_convergence: NewtonConvergence::default(),
            jacobian_strategy: JacobianStrategy::default(),
            max_iterations: 1,
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
            newton_convergence: NewtonConvergence::default(),
            jacobian_strategy: JacobianStrategy::default(),
            max_iterations: 1,
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
                newton_convergence,
                jacobian_strategy,
                max_iterations,
            } => {
                assert_eq!(sparse_threshold, default_sparse_threshold());
                assert_eq!(jacobian_bandwidth, None);
                assert_eq!(newton_convergence, default_newton_convergence());
                assert_eq!(jacobian_strategy, JacobianStrategy::default());
                assert_eq!(max_iterations, default_max_newton_iterations());
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
                ..
            } => {
                assert_eq!(sparse_threshold, 42);
                assert_eq!(jacobian_bandwidth, Some(2));
            }
        }
    }

    #[cfg(feature = "serde")]
    #[test]
    fn deserialises_newton_fields_with_defaults() {
        let json = r#"{ "BackwardEuler": {} }"#;
        let spec: IntegratorSpec = serde_json::from_str(json).unwrap();
        match spec {
            IntegratorSpec::BackwardEuler {
                newton_convergence,
                jacobian_strategy,
                max_iterations,
                ..
            } => {
                // Matches `BackwardEulerSolver::default()`'s own values —
                // not `NewtonConvergence::default()` (see the field's docs).
                assert_eq!(
                    newton_convergence,
                    NewtonConvergence::ResidualOnly {
                        tol_abs: 1e-5,
                        tol_rel: 1e-6,
                    }
                );
                assert_eq!(jacobian_strategy, JacobianStrategy::ModifiedFrozen);
                assert_eq!(max_iterations, 1);
            }
        }
    }

    #[cfg(feature = "serde")]
    #[test]
    fn deserialises_explicit_newton_fields() {
        let json = r#"{
            "BackwardEuler": {
                "newton_convergence": { "CombinedOr": { "tol_abs": 1e-9, "tol_rel": 1e-7 } },
                "jacobian_strategy": "FullNewton",
                "max_iterations": 30
            }
        }"#;
        let spec: IntegratorSpec = serde_json::from_str(json).unwrap();
        match spec {
            IntegratorSpec::BackwardEuler {
                newton_convergence,
                jacobian_strategy,
                max_iterations,
                ..
            } => {
                assert_eq!(
                    newton_convergence,
                    NewtonConvergence::CombinedOr {
                        tol_abs: 1e-9,
                        tol_rel: 1e-7,
                    }
                );
                assert_eq!(jacobian_strategy, JacobianStrategy::FullNewton);
                assert_eq!(max_iterations, 30);
            }
        }
    }

    #[test]
    fn newton_fields_take_effect_through_try_from() {
        // Object-safety + successful construction, mirroring
        // `dense_path_when_no_jacobian_bandwidth` above -- `BackwardEulerSolver`'s
        // Newton fields are private, not observable from outside the
        // crate, so this only proves the builders are actually called
        // (a typo'd builder name would fail to compile, not silently
        // no-op).
        let spec = IntegratorSpec::BackwardEuler {
            sparse_threshold: 100,
            jacobian_bandwidth: None,
            newton_convergence: NewtonConvergence::default(),
            jacobian_strategy: JacobianStrategy::FullNewton,
            max_iterations: 25,
        };
        let solver: Box<dyn Solver> = spec.try_into().unwrap();
        let _ = solver;
    }
}
