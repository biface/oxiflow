//! # Module `solver::methods::newton`
//!
//! Iterated Newton solver — shared, integrator-agnostic core (DD-044,
//! issue #109, activating DD-033's Option A).
//!
//! ## Why this exists
//!
//! [`super::implicit::theta_method_step`] and [`super::bdf2::bdf2_step`]
//! (through v0.6.0) each performed exactly **one** Newton-style
//! correction with a frozen Jacobian — exact when the underlying
//! `compute_physics` is affine in `u`, a first-order approximation
//! otherwise. This module provides the genuinely iterated alternative
//! DD-033 always anticipated, shared by both call sites rather than
//! duplicated: theta-method and BDF2 (this sprint's two concrete
//! consumers) differ only in how they express their residual as an
//! affine transform of `u` plus an implicit term — see
//! [the residual convention](#residual-convention) below.
//!
//! ## Residual convention
//!
//! Both consumers' implicit equations reduce to the same shape:
//!
//! $$g(u) = \alpha (u - u^{*}) - c \cdot f(u)$$
//!
//! - **theta-method** (Backward Euler `θ=1`, Crank-Nicolson `θ=0.5`):
//!   $\alpha = 1$, $c = \theta \Delta t$, $u^{*} = u^n + \Delta t(1-\theta)
//!   f(u^n)$.
//! - **BDF2**: $\alpha = 3/2$, $c = \Delta t$, $u^{*} = \frac{4u^n -
//!   u^{n-1}}{3}$.
//!
//! `u^{*}` folds in every term that doesn't depend on the unknown `u` —
//! callers compute it once, outside the loop, from already-known past
//! states. [`residual`] and [`linear_correction`] are the two building
//! blocks operating on this shape; [`solve`] is the generic loop wiring
//! them together with a [`NewtonConvergence`] criterion and a
//! [`JacobianStrategy`].
//!
//! ## Dense/sparse agnosticism
//!
//! This module knows nothing about `Domain`, `ContextCalculator`, or
//! `faer` — [`solve`] is parameterised over caller-supplied closures for
//! evaluating $f(u)$ and its Jacobian, and over `&dyn LinearSolver` for
//! the correction step. The existing sparse/banded dispatch
//! (DD-043, `theta_method_step_adaptive`) is therefore untouched by this
//! module: composing the two is [`super::implicit`]'s concern (chantier
//! 2), not this one's.
//!
//! ## Wiring status
//!
//! [`solve`] is called from
//! [`super::implicit::theta_method_step_newton`] (DD-044, #111) and
//! [`super::bdf2::bdf2_step_newton`] (DD-044, #112) — used respectively by
//! [`BackwardEulerSolver`](super::backward_euler::BackwardEulerSolver)/
//! [`CrankNicolsonSolver`](super::crank_nicolson::CrankNicolsonSolver) and
//! [`BDF2Solver`](super::bdf2::BDF2Solver). [`super::bdf2`]'s
//! startup/bootstrap step is the one exception, unconditionally using
//! [`super::implicit::theta_method_step`]'s non-configurable path — see
//! that module's docs.

use nalgebra::{DMatrix, DVector};

use crate::context::error::OxiflowError;
use crate::solver::linear::LinearSolver;

// ── Configuration ────────────────────────────────────────────────────────────

/// Convergence criterion for the iterated Newton loop ([`solve`]).
///
/// Every variant carries the same `tol_abs`/`tol_rel` pair: convergence
/// is tested against `tol_abs + tol_rel * ||u||`, evaluated on the
/// iterate *after* applying the current correction (so that, e.g.,
/// `max_iterations = 1` with an exactly affine `f` succeeds immediately —
/// the single correction is already exact up to floating-point rounding).
///
/// Default: [`ResidualOnly`](Self::ResidualOnly) with `tol_abs = 1e-8`,
/// `tol_rel = 1e-6` — starting points, not fixed constants; tune via the
/// solver builders once this lands (#111/#112).
///
/// Note: this default is *not* what
/// [`theta_method_step`](super::implicit::theta_method_step) or the
/// implicit solvers' own zero-config constructors use internally — those
/// need a looser, explicitly named tolerance to guarantee the pre-DD-044
/// single-correction contract on very stiff affine problems (see
/// [`super::implicit`]'s module docs for why). This `Default` impl is a
/// general starting point for callers configuring the loop themselves,
/// not a guarantee that it tolerates every finite-difference precision
/// floor.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Deserialize))]
pub enum NewtonConvergence {
    /// Converged once `||g(u)|| < tol_abs + tol_rel * ||u||`.
    ///
    /// The residual is available at zero extra cost — `f(u)` is already
    /// evaluated to build the next correction — so this is the cheapest
    /// criterion, not merely the default for convenience.
    ResidualOnly { tol_abs: f64, tol_rel: f64 },
    /// Converged once **either** the residual **or** the correction norm
    /// (`||δu||`) satisfies the tolerance.
    CombinedOr { tol_abs: f64, tol_rel: f64 },
    /// Converged only once **both** the residual **and** the correction
    /// norm satisfy the tolerance — the strictest of the three.
    CombinedAnd { tol_abs: f64, tol_rel: f64 },
}

impl Default for NewtonConvergence {
    fn default() -> Self {
        NewtonConvergence::ResidualOnly {
            tol_abs: 1e-8,
            tol_rel: 1e-6,
        }
    }
}

impl NewtonConvergence {
    /// Extracts the common `(tol_abs, tol_rel)` pair, regardless of
    /// variant.
    fn tolerances(&self) -> (f64, f64) {
        match *self {
            NewtonConvergence::ResidualOnly { tol_abs, tol_rel }
            | NewtonConvergence::CombinedOr { tol_abs, tol_rel }
            | NewtonConvergence::CombinedAnd { tol_abs, tol_rel } => (tol_abs, tol_rel),
        }
    }
}

/// Jacobian refresh strategy for the iterated Newton loop ([`solve`]).
///
/// Default: [`ModifiedFrozen`](Self::ModifiedFrozen) — matches the
/// pre-DD-044 behaviour (Jacobian evaluated once, at the initial guess,
/// never refreshed) as a special case of the generic loop, rather than a
/// separate code path.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Default)]
#[cfg_attr(feature = "serde", derive(serde::Deserialize))]
pub enum JacobianStrategy {
    /// Evaluate the Jacobian once, at the initial guess, and reuse it for
    /// every correction (modified Newton).
    #[default]
    ModifiedFrozen,
    /// Re-evaluate the Jacobian at every iterate (full Newton) — most
    /// robust on strongly nonlinear problems, at the cost of one
    /// [`finite_difference_jacobian`](super::implicit::finite_difference_jacobian)
    /// (or its banded equivalent) per iteration.
    FullNewton,
    /// Reuse the frozen Jacobian while convergence is proceeding
    /// acceptably; re-evaluate it once the residual stops shrinking fast
    /// enough. Refreshed whenever
    /// `residual_norm_k / residual_norm_{k-1} > stagnation_ratio` — an
    /// explicit extension point for J8 (v0.8.0) cost/robustness tuning,
    /// not yet auto-tuned.
    ModifiedWithStagnationCheck { stagnation_ratio: f64 },
}

// ── Building blocks ───────────────────────────────────────────────────────────

/// Evaluates the residual $g(u) = \alpha (u - u^{*}) - c \cdot f(u)$ — see
/// the [module docs](self#residual-convention) for what `alpha`/`coeff`/
/// `u_star` mean for each of this loop's two consumers.
///
/// `f_u` is `f(u)`, already evaluated by the caller (it is also needed
/// to decide, e.g., whether to refresh the Jacobian — evaluating it here
/// again would be redundant).
pub(crate) fn residual(
    u: &DVector<f64>,
    u_star: &DVector<f64>,
    alpha: f64,
    coeff: f64,
    f_u: &DVector<f64>,
) -> DVector<f64> {
    let delta: DVector<f64> = u - u_star;
    delta * alpha - f_u.clone() * coeff
}

/// Solves the linearised correction $[\alpha I - c J_f] \delta u = -g(u)$
/// for $\delta u$, given an already-assembled Jacobian $J_f =
/// \partial f/\partial u$.
///
/// Resolution only — `jacobian` is supplied by the caller (typically
/// [`finite_difference_jacobian`](super::implicit::finite_difference_jacobian),
/// unchanged) rather than computed here, so this function stays
/// dense/sparse-agnostic in principle; only the dense path
/// (`&dyn LinearSolver`) is wired at this increment (#110/#111) —
/// banded/sparse composition remains a known gap, see
/// [`super::implicit`]'s module docs.
pub(crate) fn linear_correction(
    jacobian: &DMatrix<f64>,
    alpha: f64,
    coeff: f64,
    residual: &DVector<f64>,
    linear_solver: &dyn LinearSolver,
) -> Result<DVector<f64>, OxiflowError> {
    let n = jacobian.nrows();
    let identity = DMatrix::<f64>::identity(n, n);
    let system_matrix = identity * alpha - jacobian.clone() * coeff;
    let rhs: DVector<f64> = -residual.clone();
    linear_solver.solve(&system_matrix, &rhs)
}

// ── Generic loop ──────────────────────────────────────────────────────────────

/// Parameters governing the iterated Newton loop ([`solve`]), gathered
/// into one struct so the loop's own argument list stays manageable
/// alongside the residual-shape parameters (`u_star`/`alpha`/`coeff`)
/// and the two evaluation closures.
#[derive(Debug, Clone, Copy)]
pub(crate) struct NewtonParams {
    pub convergence: NewtonConvergence,
    pub jacobian_strategy: JacobianStrategy,
    pub max_iterations: usize,
}

/// Runs the iterated Newton loop for $g(u) = \alpha(u - u^{*}) - c f(u) =
/// 0$, starting from `u0`.
///
/// `f_eval` evaluates $f(u)$ for a candidate iterate (in production,
/// this wraps [`evaluate_derivative`](super::evaluate_derivative) —
/// including its boundary-condition re-application, see
/// [`super::implicit`]'s known-limitation note); `jac_eval` evaluates
/// $\partial f/\partial u$ at a candidate iterate (wraps
/// [`finite_difference_jacobian`](super::implicit::finite_difference_jacobian)
/// or its banded equivalent). Both are called at most `max_iterations + 1`
/// times each.
///
/// Convergence is checked **after** applying each correction, uniformly
/// regardless of `max_iterations` — there is no special case for
/// `max_iterations == 1`. This is deliberate: whether `f` is affine in
/// `u` is not tracked or asserted anywhere (doing so would require the
/// HOW pole to know something about the WHAT pole's `PhysicalModel`,
/// which DD-044 does not introduce) — it is instead *inferred for free*
/// from the residual itself. For an affine `f`, a single correction is
/// exact, so the post-application residual already sits at
/// floating-point precision and passes any reasonable tolerance; for a
/// genuinely nonlinear `f`, it generally does not, and
/// `max_iterations == 1` then correctly reports
/// [`OxiflowError::NewtonNotConverged`] instead of silently returning a
/// first-order-accurate value.
///
/// Consequence for callers migrating from the pre-DD-044
/// `theta_method_step`/`bdf2_step` (which never checked convergence, and
/// so could never fail this way): calling this loop with the historical
/// defaults (`max_iterations = 1`, `ModifiedFrozen`) reproduces the exact
/// same numeric result on every existing (affine) regression case, but
/// — unlike before — now surfaces an explicit error rather than a silent
/// first-order approximation on a genuinely nonlinear model. Widen
/// `max_iterations` (or the tolerance) for such models rather than
/// relying on the single-correction default.
///
/// # Errors
///
/// Returns [`OxiflowError::NewtonNotConverged`] if `max_iterations` is
/// exhausted without satisfying `params.convergence`. Propagates any
/// error from `f_eval`, `jac_eval`, or the linear solve unchanged.
#[allow(clippy::too_many_arguments)]
pub(crate) fn solve(
    u0: DVector<f64>,
    u_star: &DVector<f64>,
    alpha: f64,
    coeff: f64,
    mut f_eval: impl FnMut(&DVector<f64>) -> Result<DVector<f64>, OxiflowError>,
    mut jac_eval: impl FnMut(&DVector<f64>) -> Result<DMatrix<f64>, OxiflowError>,
    linear_solver: &dyn LinearSolver,
    params: &NewtonParams,
) -> Result<DVector<f64>, OxiflowError> {
    let (tol_abs, tol_rel) = params.convergence.tolerances();

    let mut u = u0;
    let mut jacobian = jac_eval(&u)?;
    let mut f_u = f_eval(&u)?;
    let mut prev_residual_norm: Option<f64> = None;
    let mut last_residual_norm = f64::INFINITY;

    for _ in 0..params.max_iterations {
        let res = residual(&u, u_star, alpha, coeff, &f_u);
        let correction = linear_correction(&jacobian, alpha, coeff, &res, linear_solver)?;
        let correction_norm = correction.norm();
        u += correction;

        f_u = f_eval(&u)?;
        let res_new = residual(&u, u_star, alpha, coeff, &f_u);
        let res_new_norm = res_new.norm();
        last_residual_norm = res_new_norm;

        let tol = tol_abs + tol_rel * u.norm();
        let residual_ok = res_new_norm < tol;
        let correction_ok = correction_norm < tol;
        let converged = match params.convergence {
            NewtonConvergence::ResidualOnly { .. } => residual_ok,
            NewtonConvergence::CombinedOr { .. } => residual_ok || correction_ok,
            NewtonConvergence::CombinedAnd { .. } => residual_ok && correction_ok,
        };
        if converged {
            return Ok(u);
        }

        match params.jacobian_strategy {
            JacobianStrategy::ModifiedFrozen => {}
            JacobianStrategy::FullNewton => {
                jacobian = jac_eval(&u)?;
            }
            JacobianStrategy::ModifiedWithStagnationCheck { stagnation_ratio } => {
                let stagnating = prev_residual_norm
                    .map(|prev| prev > 0.0 && res_new_norm / prev > stagnation_ratio)
                    .unwrap_or(false);
                if stagnating {
                    jacobian = jac_eval(&u)?;
                }
            }
        }
        prev_residual_norm = Some(res_new_norm);
    }

    Err(OxiflowError::NewtonNotConverged {
        iterations: params.max_iterations,
        residual: last_residual_norm,
    })
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::solver::linear::NalgebraDenseSolver;

    // Two synthetic, domain-free test problems — this module is
    // integrator-agnostic, so its own tests exercise `solve` directly
    // against hand-written f/jacobian closures rather than a full
    // Domain/Scenario, unlike `implicit.rs`/`bdf2.rs`'s tests.

    /// f(u) = -lambda*u — affine, so a single correction is exact
    /// regardless of `max_iterations`/`jacobian_strategy`.
    fn affine_decay(
        lambda: f64,
    ) -> (
        impl FnMut(&DVector<f64>) -> Result<DVector<f64>, OxiflowError>,
        impl FnMut(&DVector<f64>) -> Result<DMatrix<f64>, OxiflowError>,
    ) {
        let f = move |u: &DVector<f64>| Ok(u.map(|v| -lambda * v));
        let jac =
            move |u: &DVector<f64>| Ok(DMatrix::<f64>::identity(u.len(), u.len()) * (-lambda));
        (f, jac)
    }

    /// f(u) = -u^3 — genuinely nonlinear; one frozen-Jacobian correction
    /// is only a first-order approximation, iterated Newton is needed
    /// for tight convergence.
    fn cubic_decay() -> (
        impl FnMut(&DVector<f64>) -> Result<DVector<f64>, OxiflowError>,
        impl FnMut(&DVector<f64>) -> Result<DMatrix<f64>, OxiflowError>,
    ) {
        let f = move |u: &DVector<f64>| Ok(u.map(|v| -v * v * v));
        let jac =
            move |u: &DVector<f64>| Ok(DMatrix::<f64>::from_diagonal(&u.map(|v| -3.0 * v * v)));
        (f, jac)
    }

    #[test]
    fn residual_matches_hand_derivation() {
        let u = DVector::from_vec(vec![2.0, 3.0]);
        let u_star = DVector::from_vec(vec![1.0, 1.0]);
        let f_u = DVector::from_vec(vec![0.5, 0.5]);
        // g(u) = 1.5*(u - u_star) - 0.1*f(u)
        let res = residual(&u, &u_star, 1.5, 0.1, &f_u);
        assert!((res[0] - (1.5 * 1.0 - 0.1 * 0.5)).abs() < 1e-12);
        assert!((res[1] - (1.5 * 2.0 - 0.1 * 0.5)).abs() < 1e-12);
    }

    #[test]
    fn linear_correction_solves_the_linearised_system() {
        // alpha*I - coeff*J with J = -2*I, alpha=1, coeff=0.1 -> 1.2*I.
        let jacobian = DMatrix::<f64>::identity(2, 2) * -2.0;
        let res = DVector::from_vec(vec![1.2, 2.4]);
        let delta = linear_correction(&jacobian, 1.0, 0.1, &res, &NalgebraDenseSolver).unwrap();
        assert!((delta[0] - (-1.0)).abs() < 1e-9);
        assert!((delta[1] - (-2.0)).abs() < 1e-9);
    }

    #[test]
    fn single_iteration_is_exact_for_affine_problem() {
        // Backward-Euler-style step: alpha=1, coeff=theta*dt=0.1,
        // u_star = u_n (theta=1 so the (1-theta) term vanishes).
        let lambda = 3.0;
        let (f, jac) = affine_decay(lambda);
        let u0 = DVector::from_element(3, 1.0);
        let u_star = u0.clone();
        let params = NewtonParams {
            convergence: NewtonConvergence::default(),
            jacobian_strategy: JacobianStrategy::ModifiedFrozen,
            max_iterations: 1,
        };

        let u_next = solve(u0, &u_star, 1.0, 0.1, f, jac, &NalgebraDenseSolver, &params).unwrap();

        let expected = 1.0 / (1.0 + lambda * 0.1);
        for v in u_next.iter() {
            assert!((v - expected).abs() < 1e-9, "got {v}, expected {expected}");
        }
    }

    #[test]
    fn full_newton_converges_on_nonaffine_problem() {
        let (f, jac) = cubic_decay();
        let u0 = DVector::from_element(2, 1.0);
        let u_star = u0.clone();
        let params = NewtonParams {
            convergence: NewtonConvergence::default(),
            jacobian_strategy: JacobianStrategy::FullNewton,
            max_iterations: 20,
        };

        let u_next = solve(u0, &u_star, 1.0, 0.1, f, jac, &NalgebraDenseSolver, &params).unwrap();

        // Cross-check against an independently solved scalar equation:
        // u + 0.1*u^3 = 1.0 (Backward-Euler-style implicit cubic decay).
        let mut u_scalar = 1.0_f64;
        for _ in 0..50 {
            let g = u_scalar + 0.1 * u_scalar.powi(3) - 1.0;
            let dg = 1.0 + 0.3 * u_scalar.powi(2);
            u_scalar -= g / dg;
        }
        for v in u_next.iter() {
            assert!((v - u_scalar).abs() < 1e-5, "got {v}, expected {u_scalar}");
        }
    }

    #[test]
    fn modified_frozen_with_single_correction_does_not_converge_on_nonaffine_problem() {
        // A single frozen-Jacobian correction is only a first-order
        // approximation for this nonaffine f (contrast with
        // `full_newton_converges_on_nonaffine_problem`, which reaches the
        // true root): under the loop's default tolerance, that shows up
        // as an honest convergence failure -- not a silently inaccurate
        // "success" -- since nothing about affine-ness is special-cased;
        // see `solve`'s docs.
        let (f, jac) = cubic_decay();
        let u0 = DVector::from_element(1, 1.0);
        let u_star = u0.clone();
        let params = NewtonParams {
            convergence: NewtonConvergence::default(),
            jacobian_strategy: JacobianStrategy::ModifiedFrozen,
            max_iterations: 1,
        };

        let err = solve(u0, &u_star, 1.0, 0.1, f, jac, &NalgebraDenseSolver, &params).unwrap_err();
        assert!(matches!(
            err,
            OxiflowError::NewtonNotConverged { iterations: 1, .. }
        ));
    }

    #[test]
    fn not_converged_when_iteration_budget_too_small() {
        let (f, jac) = cubic_decay();
        let u0 = DVector::from_element(1, 1.0);
        let u_star = u0.clone();
        let params = NewtonParams {
            convergence: NewtonConvergence::ResidualOnly {
                tol_abs: 1e-14,
                tol_rel: 0.0,
            },
            jacobian_strategy: JacobianStrategy::ModifiedFrozen,
            max_iterations: 1,
        };

        let err = solve(u0, &u_star, 1.0, 0.1, f, jac, &NalgebraDenseSolver, &params).unwrap_err();
        assert!(matches!(
            err,
            OxiflowError::NewtonNotConverged { iterations: 1, .. }
        ));
    }

    #[test]
    fn combined_or_converges_when_correction_is_small_even_if_residual_is_not() {
        let f = |u: &DVector<f64>| Ok(u.map(|v| v + 1.0)); // never zero for u >= 0
        let jac = |u: &DVector<f64>| Ok(DMatrix::<f64>::identity(u.len(), u.len()) * -1.0e6);
        let u0 = DVector::from_element(1, 0.0);
        let u_star = u0.clone();
        // A deliberately stiff Jacobian (-1e6) makes the correction
        // collapse to a tiny step even though the residual itself does
        // not shrink below tolerance -- see the closures above.
        let params_or = NewtonParams {
            convergence: NewtonConvergence::CombinedOr {
                tol_abs: 1e-3,
                tol_rel: 0.0,
            },
            jacobian_strategy: JacobianStrategy::ModifiedFrozen,
            max_iterations: 1,
        };
        let params_residual_only = NewtonParams {
            convergence: NewtonConvergence::ResidualOnly {
                tol_abs: 1e-3,
                tol_rel: 0.0,
            },
            jacobian_strategy: JacobianStrategy::ModifiedFrozen,
            max_iterations: 1,
        };

        let out_or = solve(
            u0.clone(),
            &u_star,
            1.0,
            1.0,
            f,
            jac,
            &NalgebraDenseSolver,
            &params_or,
        );
        let out_residual_only = solve(
            u0,
            &u_star,
            1.0,
            1.0,
            f,
            jac,
            &NalgebraDenseSolver,
            &params_residual_only,
        );

        assert!(out_or.is_ok(), "CombinedOr should accept a tiny correction");
        assert!(
            out_residual_only.is_err(),
            "ResidualOnly should reject a large residual"
        );
    }
}
