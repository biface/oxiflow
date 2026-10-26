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
//! [`IntegratorSpec::BackwardEuler`], [`IntegratorSpec::CrankNicolson`]
//! (#115), and [`IntegratorSpec::BDF2`] (#116, sparse fields since #133),
//! covering the DD-043 (#50) sparse dispatch path — this whole module is
//! gated behind the `sparse` feature:
//! without it, a config DTO for these integrators adds nothing over the
//! existing [`super::IntegratorKind`] labels, since the dense path takes
//! no parameters. Remaining integrators (`RK4`, `DoPri45`) and the WHAT
//! axis (`ModelConfig`/`MeshConfig`/`BoundaryConditionConfig`) are tracked
//! as follow-up, not blocking this issue's closure (per #104's own stated
//! scope). `newton_convergence`/`jacobian_strategy`/`max_iterations`
//! (DD-044, [#114](https://github.com/biface/oxiflow/issues/114)/[#115](https://github.com/biface/oxiflow/issues/115)/[#116](https://github.com/biface/oxiflow/issues/116)) extend all three variants — the Newton
//! configuration is orthogonal to which integrator carries it.
//!
//! The sparse backend itself is always [`FaerSparseSolver`] — a fixed,
//! non-configurable choice: `Box<dyn SparseLinearSolver>` is a trait
//! object and cannot be deserialised, so there is no way for a config file
//! to name an arbitrary backend (see #104's discussion of this
//! constraint).

use super::methods::backward_euler::BackwardEulerSolver;
use super::methods::bdf2::BDF2Solver;
use super::methods::crank_nicolson::CrankNicolsonSolver;
#[cfg(feature = "serde")]
use super::methods::implicit::{stiff_jacobian_convergence, DEFAULT_SPARSE_THRESHOLD};
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
        /// Newton convergence criterion (DD-044, [#114](https://github.com/biface/oxiflow/issues/114)) — see
        /// [`BackwardEulerSolver::with_newton_convergence`]. Defaults to
        /// `stiff_jacobian_convergence`, not [`NewtonConvergence::default`]
        /// — matching `BackwardEulerSolver`'s own zero-config value (see
        /// that function's docs for why they differ).
        #[cfg_attr(feature = "serde", serde(default = "default_newton_convergence"))]
        newton_convergence: NewtonConvergence,
        /// Jacobian refresh strategy (DD-044, [#114](https://github.com/biface/oxiflow/issues/114)) — see
        /// [`BackwardEulerSolver::with_jacobian_strategy`].
        #[cfg_attr(feature = "serde", serde(default))]
        jacobian_strategy: JacobianStrategy,
        /// Newton iteration budget (DD-044, [#114](https://github.com/biface/oxiflow/issues/114)) — see
        /// [`BackwardEulerSolver::with_max_newton_iterations`]. Defaults
        /// to `1`, matching `BackwardEulerSolver`'s own zero-config value
        /// (the pre-DD-044 single-correction contract).
        #[cfg_attr(feature = "serde", serde(default = "default_max_newton_iterations"))]
        max_iterations: usize,
    },
    /// Crank-Nicolson (semi-implicit, 2nd order), symmetric to
    /// [`BackwardEuler`](Self::BackwardEuler) in every field — sparse
    /// dispatch (DD-043) and Newton configuration (DD-044, [#115](https://github.com/biface/oxiflow/issues/115)) both
    /// apply identically. `theta = 0.5` is not exposed as a field —
    /// implicit to the variant, consistent with `BackwardEuler`'s
    /// `theta = 1.0` never being exposed either.
    CrankNicolson {
        /// See [`BackwardEuler::sparse_threshold`](Self::BackwardEuler).
        #[cfg_attr(feature = "serde", serde(default = "default_sparse_threshold"))]
        sparse_threshold: usize,
        /// See [`BackwardEuler::jacobian_bandwidth`](Self::BackwardEuler).
        #[cfg_attr(feature = "serde", serde(default))]
        jacobian_bandwidth: Option<usize>,
        /// See [`BackwardEuler::newton_convergence`](Self::BackwardEuler).
        #[cfg_attr(feature = "serde", serde(default = "default_newton_convergence"))]
        newton_convergence: NewtonConvergence,
        /// See [`BackwardEuler::jacobian_strategy`](Self::BackwardEuler).
        #[cfg_attr(feature = "serde", serde(default))]
        jacobian_strategy: JacobianStrategy,
        /// See [`BackwardEuler::max_iterations`](Self::BackwardEuler).
        #[cfg_attr(feature = "serde", serde(default = "default_max_newton_iterations"))]
        max_iterations: usize,
    },
    /// BDF2 (implicit multi-step, 2nd order), symmetric to
    /// [`BackwardEuler`](Self::BackwardEuler) in every field: sparse
    /// dispatch (DD-043, [#133](https://github.com/biface/oxiflow/issues/133)), for the Backward Euler bootstrap
    /// step and the BDF2 update alike, and Newton configuration (DD-044,
    /// [#116](https://github.com/biface/oxiflow/issues/116)).
    BDF2 {
        /// See [`BackwardEuler::sparse_threshold`](Self::BackwardEuler).
        #[cfg_attr(feature = "serde", serde(default = "default_sparse_threshold"))]
        sparse_threshold: usize,
        /// See [`BackwardEuler::jacobian_bandwidth`](Self::BackwardEuler).
        #[cfg_attr(feature = "serde", serde(default))]
        jacobian_bandwidth: Option<usize>,
        /// See [`BackwardEuler::newton_convergence`](Self::BackwardEuler).
        #[cfg_attr(feature = "serde", serde(default = "default_newton_convergence"))]
        newton_convergence: NewtonConvergence,
        /// See [`BackwardEuler::jacobian_strategy`](Self::BackwardEuler).
        #[cfg_attr(feature = "serde", serde(default))]
        jacobian_strategy: JacobianStrategy,
        /// See [`BackwardEuler::max_iterations`](Self::BackwardEuler).
        #[cfg_attr(feature = "serde", serde(default = "default_max_newton_iterations"))]
        max_iterations: usize,
    },
}

/// Default for `sparse_threshold` when omitted from a deserialised config:
/// the implicit solvers' own default, [`DEFAULT_SPARSE_THRESHOLD`].
#[cfg(feature = "serde")]
fn default_sparse_threshold() -> usize {
    DEFAULT_SPARSE_THRESHOLD
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
            IntegratorSpec::CrankNicolson {
                sparse_threshold,
                jacobian_bandwidth,
                newton_convergence,
                jacobian_strategy,
                max_iterations,
            } => {
                let mut solver = CrankNicolsonSolver::new()
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
            IntegratorSpec::BDF2 {
                sparse_threshold,
                jacobian_bandwidth,
                newton_convergence,
                jacobian_strategy,
                max_iterations,
            } => {
                let mut solver = BDF2Solver::new()
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
    use crate::mesh::UniformGrid1D;
    use crate::model::{CubicDecay, PhysicalModel, TridiagonalDiffusion};
    use crate::solver::methods::implicit::stiff_jacobian_convergence;
    use crate::solver::{IntegratorKind, Scenario, SimulationResult, SolverConfiguration};
    use crate::solver::{StepControl, TimeConfiguration};

    // ── Behaviour of the built solvers, every implicit variant ────────────────

    /// The integrators an `IntegratorSpec` can build.
    fn implicit_kinds() -> [IntegratorKind; 3] {
        [
            IntegratorKind::BackwardEuler,
            IntegratorKind::CrankNicolson,
            IntegratorKind::BDF2,
        ]
    }

    /// The `kind` variant with the given fields.
    fn spec(
        kind: &IntegratorKind,
        sparse_threshold: usize,
        jacobian_bandwidth: Option<usize>,
        newton_convergence: NewtonConvergence,
        jacobian_strategy: JacobianStrategy,
        max_iterations: usize,
    ) -> IntegratorSpec {
        match kind {
            IntegratorKind::BackwardEuler => IntegratorSpec::BackwardEuler {
                sparse_threshold,
                jacobian_bandwidth,
                newton_convergence,
                jacobian_strategy,
                max_iterations,
            },
            IntegratorKind::CrankNicolson => IntegratorSpec::CrankNicolson {
                sparse_threshold,
                jacobian_bandwidth,
                newton_convergence,
                jacobian_strategy,
                max_iterations,
            },
            IntegratorKind::BDF2 => IntegratorSpec::BDF2 {
                sparse_threshold,
                jacobian_bandwidth,
                newton_convergence,
                jacobian_strategy,
                max_iterations,
            },
            other => panic!("not built from an IntegratorSpec: {other:?}"),
        }
    }

    /// The `kind` variant with the solvers' default Newton configuration
    /// and the given sparse fields.
    fn sparse_spec(
        kind: &IntegratorKind,
        sparse_threshold: usize,
        jacobian_bandwidth: Option<usize>,
    ) -> IntegratorSpec {
        spec(
            kind,
            sparse_threshold,
            jacobian_bandwidth,
            stiff_jacobian_convergence(),
            JacobianStrategy::default(),
            1,
        )
    }

    /// Runs the solver built from `spec` on `model`, over `nodes` nodes,
    /// for `steps` steps of `dt`.
    fn run(
        spec: IntegratorSpec,
        kind: &IntegratorKind,
        model: Box<dyn PhysicalModel>,
        nodes: usize,
        dt: f64,
        steps: usize,
    ) -> Result<SimulationResult, OxiflowError> {
        let solver: Box<dyn Solver> = spec.try_into()?;
        let scenario = Scenario::single(model, Box::new(UniformGrid1D::new(nodes, 0.0, 1.0)?));
        let config = SolverConfiguration::new(
            TimeConfiguration::new(steps as f64 * dt, StepControl::Fixed { dt }),
            kind.clone(),
        );
        solver.solve(&scenario, &config)
    }

    /// Two steps of `TridiagonalDiffusion` on 10 unknowns -- the bootstrap
    /// step and the BDF2 update for `BDF2`.
    fn run_diffusion(
        spec: IntegratorSpec,
        kind: &IntegratorKind,
    ) -> Result<SimulationResult, OxiflowError> {
        run(
            spec,
            kind,
            Box::new(TridiagonalDiffusion { diffusion: 1.0 }),
            10,
            0.1,
            2,
        )
    }

    #[test]
    fn declared_bandwidth_selects_the_banded_path() {
        // The diffusion is tridiagonal: a declared half-bandwidth of 0
        // drops its off-diagonal entries, so the single default correction
        // is inexact on the banded path and the step reports it. The dense
        // path, taken without a bandwidth, is exact on this affine model.
        for kind in implicit_kinds() {
            let err = run_diffusion(sparse_spec(&kind, 0, Some(0)), &kind).unwrap_err();
            assert!(
                matches!(err, OxiflowError::NewtonNotConverged { iterations: 1, .. }),
                "{kind:?}, band 0: got {err:?}"
            );
            run_diffusion(sparse_spec(&kind, 0, None), &kind)
                .unwrap_or_else(|e| panic!("{kind:?}, no band: {e:?}"));
        }
    }

    #[test]
    fn systems_up_to_the_sparse_threshold_stay_dense() {
        // 10 unknowns, threshold 10: the banded path needs strictly more
        // unknowns than the threshold, so the too-narrow band is never used.
        for kind in implicit_kinds() {
            run_diffusion(sparse_spec(&kind, 10, Some(0)), &kind)
                .unwrap_or_else(|e| panic!("{kind:?}: {e:?}"));
        }
    }

    /// Root of one theta-method step of f(u) = -u^3 from u = 1:
    /// u + theta*dt*u^3 = 1 - (1 - theta)*dt, solved by scalar Newton.
    fn cubic_decay_root(theta: f64, dt: f64) -> f64 {
        let rhs = 1.0 - (1.0 - theta) * dt;
        let mut u = 1.0_f64;
        for _ in 0..100 {
            let g = u + theta * dt * u.powi(3) - rhs;
            let dg = 1.0 + 3.0 * theta * dt * u.powi(2);
            u -= g / dg;
        }
        u
    }

    #[test]
    fn newton_fields_take_effect() {
        // One step of f(u) = -u^3 at dt = 0.5 (the Backward Euler bootstrap
        // step for BDF2). Two frozen-Jacobian corrections do not reach the
        // default tolerance: the step reports the configured budget. A
        // full Newton loop under a tolerance far below the default one
        // reaches the root.
        let dt = 0.5;
        for kind in implicit_kinds() {
            let short = spec(
                &kind,
                100,
                None,
                stiff_jacobian_convergence(),
                JacobianStrategy::ModifiedFrozen,
                2,
            );
            let err = run(short, &kind, Box::new(CubicDecay), 2, dt, 1).unwrap_err();
            assert!(
                matches!(err, OxiflowError::NewtonNotConverged { iterations: 2, .. }),
                "{kind:?}, budget 2: got {err:?}"
            );

            let tight = spec(
                &kind,
                100,
                None,
                NewtonConvergence::ResidualOnly {
                    tol_abs: 1e-13,
                    tol_rel: 0.0,
                },
                JacobianStrategy::FullNewton,
                10,
            );
            let result = run(tight, &kind, Box::new(CubicDecay), 2, dt, 1).unwrap();
            let theta = if kind == IntegratorKind::CrankNicolson {
                0.5
            } else {
                1.0
            };
            let expected = cubic_decay_root(theta, dt);
            for v in result.states[1].as_scalar_field().unwrap().iter() {
                assert!(
                    (v - expected).abs() < 1e-12,
                    "{kind:?}: got {v}, expected {expected}"
                );
            }
        }
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
            IntegratorSpec::CrankNicolson { .. } => panic!("expected BackwardEuler variant"),
            IntegratorSpec::BDF2 { .. } => panic!("expected BackwardEuler variant"),
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
            IntegratorSpec::CrankNicolson { .. } => panic!("expected BackwardEuler variant"),
            IntegratorSpec::BDF2 { .. } => panic!("expected BackwardEuler variant"),
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
            IntegratorSpec::CrankNicolson { .. } => panic!("expected BackwardEuler variant"),
            IntegratorSpec::BDF2 { .. } => panic!("expected BackwardEuler variant"),
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
            IntegratorSpec::CrankNicolson { .. } => panic!("expected BackwardEuler variant"),
            IntegratorSpec::BDF2 { .. } => panic!("expected BackwardEuler variant"),
        }
    }

    // ── CrankNicolson (#115) — mirrors the BackwardEuler coverage above ────────

    #[cfg(feature = "serde")]
    #[test]
    fn crank_nicolson_deserialises_with_defaults() {
        let json = r#"{ "CrankNicolson": {} }"#;
        let spec: IntegratorSpec = serde_json::from_str(json).unwrap();
        match spec {
            IntegratorSpec::CrankNicolson {
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
            IntegratorSpec::BackwardEuler { .. } => {
                panic!("expected CrankNicolson variant")
            }
            IntegratorSpec::BDF2 { .. } => panic!("expected CrankNicolson variant"),
        }
    }

    #[cfg(feature = "serde")]
    #[test]
    fn crank_nicolson_deserialises_explicit_fields() {
        let json = r#"{
            "CrankNicolson": {
                "sparse_threshold": 42,
                "jacobian_bandwidth": 2,
                "newton_convergence": { "CombinedAnd": { "tol_abs": 1e-9, "tol_rel": 1e-7 } },
                "jacobian_strategy": "FullNewton",
                "max_iterations": 30
            }
        }"#;
        let spec: IntegratorSpec = serde_json::from_str(json).unwrap();
        match spec {
            IntegratorSpec::CrankNicolson {
                sparse_threshold,
                jacobian_bandwidth,
                newton_convergence,
                jacobian_strategy,
                max_iterations,
            } => {
                assert_eq!(sparse_threshold, 42);
                assert_eq!(jacobian_bandwidth, Some(2));
                assert_eq!(
                    newton_convergence,
                    NewtonConvergence::CombinedAnd {
                        tol_abs: 1e-9,
                        tol_rel: 1e-7,
                    }
                );
                assert_eq!(jacobian_strategy, JacobianStrategy::FullNewton);
                assert_eq!(max_iterations, 30);
            }
            IntegratorSpec::BackwardEuler { .. } => {
                panic!("expected CrankNicolson variant")
            }
            IntegratorSpec::BDF2 { .. } => panic!("expected CrankNicolson variant"),
        }
    }

    // ── BDF2 (#116, sparse fields since #133) ──────────────────────────────────

    #[cfg(feature = "serde")]
    #[test]
    fn bdf2_deserialises_with_defaults() {
        let json = r#"{ "BDF2": {} }"#;
        let spec: IntegratorSpec = serde_json::from_str(json).unwrap();
        match spec {
            IntegratorSpec::BDF2 {
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
            _ => panic!("expected BDF2 variant"),
        }
    }

    #[cfg(feature = "serde")]
    #[test]
    fn bdf2_deserialises_explicit_fields() {
        let json = r#"{
            "BDF2": {
                "sparse_threshold": 42,
                "jacobian_bandwidth": 2,
                "newton_convergence": { "ResidualOnly": { "tol_abs": 1e-9, "tol_rel": 1e-7 } },
                "jacobian_strategy": "FullNewton",
                "max_iterations": 30
            }
        }"#;
        let spec: IntegratorSpec = serde_json::from_str(json).unwrap();
        match spec {
            IntegratorSpec::BDF2 {
                sparse_threshold,
                jacobian_bandwidth,
                newton_convergence,
                jacobian_strategy,
                max_iterations,
            } => {
                assert_eq!(sparse_threshold, 42);
                assert_eq!(jacobian_bandwidth, Some(2));
                assert_eq!(
                    newton_convergence,
                    NewtonConvergence::ResidualOnly {
                        tol_abs: 1e-9,
                        tol_rel: 1e-7,
                    }
                );
                assert_eq!(jacobian_strategy, JacobianStrategy::FullNewton);
                assert_eq!(max_iterations, 30);
            }
            _ => panic!("expected BDF2 variant"),
        }
    }
}
