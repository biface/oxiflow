//! # Module `solver::methods::implicit`
//!
//! Shared machinery for implicit (theta-method) integrators — DD-033.
//!
//! [`BackwardEulerSolver`](super::backward_euler::BackwardEulerSolver) (θ=1)
//! and [`CrankNicolsonSolver`](super::crank_nicolson::CrankNicolsonSolver)
//! (θ=0.5) are thin wrappers around [`theta_method_step_newton`] (DD-044,
//! #111), which itself relies on [`finite_difference_jacobian`] and the
//! shared [`super::newton`] loop. [`theta_method_step`] keeps the
//! original, non-configurable entry point for direct callers — see below.
//!
//! ## Why a single shared function for two methods
//!
//! The generalised theta method is:
//!
//! $$u^{n+1} = u^n + \Delta t\left[(1-\theta) f(u^n) + \theta f(u^{n+1})\right]$$
//!
//! Linearising the residual $g(u) = u - u^n - \Delta t[(1-\theta)f(u^n) +
//! \theta f(u)]$ at $u = u^n$ gives $g(u^n) = -\Delta t \cdot f(u^n)$ for
//! **any** $\theta$ — the right-hand side of the linear correction doesn't
//! depend on θ at all. Only the system matrix does
//! ($I - \theta \Delta t J_f$). One function, one parameter.
//!
//! ## v1 scope — frozen Jacobian, single correction (DD-033) by default
//!
//! [`theta_method_step`] performs exactly **one** Newton-style correction,
//! with the Jacobian frozen at $u^n$ — this is exact when `compute_physics`
//! is affine in `u` and a first-order approximation otherwise, and stays
//! the behaviour of [`theta_method_step`] itself (used directly by
//! callers that don't need more) and of
//! [`BackwardEulerSolver`](super::backward_euler::BackwardEulerSolver)/
//! [`CrankNicolsonSolver`](super::crank_nicolson::CrankNicolsonSolver)
//! when left unconfigured.
//!
//! [`theta_method_step_newton`] (DD-044, #111) is the genuinely iterated
//! alternative DD-033 anticipated: it routes through the shared
//! [`super::newton`] loop with caller-supplied convergence/Jacobian-
//! strategy/iteration-budget configuration.
//! `BackwardEulerSolver`/`CrankNicolsonSolver` call it directly (with
//! `with_newton_convergence`/`with_jacobian_strategy`/
//! `with_max_newton_iterations` builders); [`theta_method_step`] itself
//! stays at its original signature by delegating to it with
//! `k_max = 1`/[`JacobianStrategy::ModifiedFrozen`](super::newton::JacobianStrategy::ModifiedFrozen)
//! baked in, reproducing its historical output exactly (see
//! [`super::newton`]'s own docs for why no special-casing is needed for
//! this to hold on affine problems).
//!
//! ## Known gap — sparse path not yet Newton-aware
//!
//! [`theta_method_step_adaptive`] (the sparse/banded dispatch, DD-043)
//! still performs a single frozen-Jacobian correction unconditionally —
//! it is not wired through [`super::newton`] at this increment. Composing
//! the two was verified, not implemented, per #111's stated scope:
//! configuring both a sparse backend *and* a non-default Newton budget on
//! the same solver silently keeps the sparse path single-shot. Tracked as
//! a follow-up, mirroring the existing BDF2/sparse gap.
//!
//! ## Known untested limitation — boundary conditions
//!
//! [`finite_difference_jacobian`] perturbs each component of the state and
//! re-evaluates [`evaluate_derivative`], which re-applies boundary
//! conditions on every call. A Dirichlet-constrained node will have its
//! perturbation overwritten before `compute_physics` sees it — physically
//! correct (a BC-constrained node isn't a free unknown), but **no test
//! case exercises this yet**. Validate explicitly before using an implicit
//! solver on a domain with boundary conditions.

use nalgebra::{DMatrix, DVector};

use super::evaluate_derivative;
use super::newton::{self, JacobianStrategy, NewtonConvergence};
use crate::context::error::OxiflowError;
use crate::context::value::ContextValue;
use crate::context::ContextCalculator;
use crate::solver::linear::LinearSolver;
use crate::solver::scenario::Domain;
#[cfg(feature = "sparse")]
use crate::solver::sparse::SparseLinearSolver;

/// Finite-difference step size for [`finite_difference_jacobian`].
///
/// Not scaled by state magnitude (v1 simplification) — fine for the O(1)
/// stiff linear test problems this targets; revisit if used on
/// significantly different magnitude scales.
const FD_EPSILON: f64 = 1e-7;

/// Estimates $\partial f/\partial u$ at `state`, `t`, by forward
/// differences.
///
/// For `f` affine in `u` (the case these implicit methods are validated
/// against), forward differences are exact regardless of step size — no
/// truncation error from the linear approximation itself.
///
/// See the [known limitation on boundary conditions](self#known-untested-limitation--boundary-conditions).
pub(crate) fn finite_difference_jacobian(
    domain: &Domain,
    chain: &[&dyn ContextCalculator],
    state: &ContextValue,
    t: f64,
    dt: f64,
) -> Result<DMatrix<f64>, OxiflowError> {
    let base_field = state.as_scalar_field()?.clone();
    let n = base_field.len();

    let mut base_state = state.clone();
    let f0 = evaluate_derivative(domain, chain, &mut base_state, t, dt)?;
    let f0_field = f0.as_scalar_field()?.clone();

    let mut jacobian = DMatrix::<f64>::zeros(n, n);

    for j in 0..n {
        let mut perturbed_field = base_field.clone();
        perturbed_field[j] += FD_EPSILON;
        let mut perturbed_state = ContextValue::ScalarField(perturbed_field);

        let f_j = evaluate_derivative(domain, chain, &mut perturbed_state, t, dt)?;
        let f_j_field = f_j.as_scalar_field()?;

        for i in 0..n {
            jacobian[(i, j)] = (f_j_field[i] - f0_field[i]) / FD_EPSILON;
        }
    }

    Ok(jacobian)
}

/// Convergence criterion guaranteeing [`theta_method_step`]'s
/// single-correction contract succeeds on the stiffest affine problems
/// this crate currently tests (`λ = 1e4`-scale exponential decay).
///
/// Deliberately distinct from [`NewtonConvergence::default`] (`tol_abs =
/// 1e-8`): on such problems, [`finite_difference_jacobian`]'s fixed step
/// size suffers catastrophic-cancellation error whose absolute magnitude
/// scales roughly as `λ · ε_machine / FD_EPSILON` — empirically up to
/// ~1e-6 for the stiffest cases tested. The single correction itself is
/// still mathematically exact for affine `f`; this is a
/// finite-difference precision floor, not a nonlinearity signal, and
/// [`NewtonConvergence::default`] remains the right general-purpose
/// starting point for callers configuring the loop themselves — it just
/// isn't loose enough to guarantee success on this specific,
/// already-tested edge of the stiffness range. Used here and by
/// [`BackwardEulerSolver::default`](super::backward_euler::BackwardEulerSolver)/
/// [`CrankNicolsonSolver::default`](super::crank_nicolson::CrankNicolsonSolver)
/// — not by `NewtonConvergence`'s own `Default` impl — precisely so that
/// zero-config solver construction keeps reproducing history exactly,
/// without changing what a fresh, explicitly-configured
/// `NewtonConvergence::default()` means elsewhere.
pub(crate) fn stiff_jacobian_convergence() -> NewtonConvergence {
    NewtonConvergence::ResidualOnly {
        tol_abs: 1e-5,
        tol_rel: 1e-6,
    }
}

/// Performs one step of the generalised theta method, with the Jacobian
/// frozen at `state` (one Newton-style correction, not iterated — see
/// [module docs](self)).
///
/// `theta = 1.0` is Backward Euler; `theta = 0.5` is Crank-Nicolson.
///
/// `state` is mutated in-place by boundary condition application — same
/// contract as the explicit solvers (see
/// [`crate::solver::methods::evaluate_derivative`]): callers should not
/// assume `state` is left unchanged after this call.
///
/// Delegates to [`theta_method_step_newton`] with the pre-DD-044 defaults
/// (`k_max = 1`, [`JacobianStrategy::ModifiedFrozen`],
/// [`stiff_jacobian_convergence`]) baked in — kept at its original
/// signature (DD-044, #110) for direct callers that don't need the
/// configurable path.
pub(crate) fn theta_method_step(
    domain: &Domain,
    chain: &[&dyn ContextCalculator],
    state: &mut ContextValue,
    t: f64,
    dt: f64,
    theta: f64,
    linear_solver: &dyn LinearSolver,
) -> Result<ContextValue, OxiflowError> {
    theta_method_step_newton(
        domain,
        chain,
        state,
        t,
        dt,
        theta,
        linear_solver,
        stiff_jacobian_convergence(),
        JacobianStrategy::default(),
        1,
    )
}

/// Same contract as [`theta_method_step`], routed through the generic
/// iterated Newton loop ([`newton::solve`]) with caller-supplied
/// convergence/Jacobian-strategy/iteration-budget configuration (DD-044,
/// #111) — [`BackwardEulerSolver`](super::backward_euler::BackwardEulerSolver)
/// and [`CrankNicolsonSolver`](super::crank_nicolson::CrankNicolsonSolver)
/// call this directly, passing their own configured fields; their
/// defaults reproduce [`theta_method_step`]'s call above exactly.
///
/// Follows [`newton`]'s residual convention with $\alpha = 1$, $c = \theta
/// \Delta t$, $u^{*} = u^n + \Delta t(1-\theta)f(u^n)$ — see that
/// module's docs. `f(u^n)` is evaluated once, at time `t` (the known
/// term); every subsequent candidate `f`/Jacobian evaluation inside the
/// Newton loop is at time `t + dt` (the implicit/target time), matching
/// [`theta_method_step`]'s original convention.
#[allow(clippy::too_many_arguments)]
pub(crate) fn theta_method_step_newton(
    domain: &Domain,
    chain: &[&dyn ContextCalculator],
    state: &mut ContextValue,
    t: f64,
    dt: f64,
    theta: f64,
    linear_solver: &dyn LinearSolver,
    newton_convergence: NewtonConvergence,
    jacobian_strategy: JacobianStrategy,
    max_iterations: usize,
) -> Result<ContextValue, OxiflowError> {
    // f(u^n, t) -- BCs applied in-place to `state` itself, consistent with
    // the explicit solvers' contract (same as `theta_method_step`).
    let f_n = evaluate_derivative(domain, chain, state, t, dt)?;
    let u_n_field = state.as_scalar_field()?.clone();
    let f_n_field = f_n.as_scalar_field()?.clone();

    let u0 = u_n_field.clone();
    let u_star: DVector<f64> = u_n_field + f_n_field * (dt * (1.0 - theta));
    let coeff = theta * dt;
    let t_target = t + dt;

    let u_next = newton::solve(
        u0,
        &u_star,
        1.0,
        coeff,
        move |candidate: &DVector<f64>| {
            let mut candidate_state = ContextValue::ScalarField(candidate.clone());
            let f = evaluate_derivative(domain, chain, &mut candidate_state, t_target, dt)?;
            Ok(f.as_scalar_field()?.clone())
        },
        move |candidate: &DVector<f64>| {
            let candidate_state = ContextValue::ScalarField(candidate.clone());
            finite_difference_jacobian(domain, chain, &candidate_state, t_target, dt)
        },
        linear_solver,
        &newton::NewtonParams {
            convergence: newton_convergence,
            jacobian_strategy,
            max_iterations,
        },
    )?;

    Ok(ContextValue::ScalarField(u_next))
}

// ── Sparse / banded path (DD-013 second phase, DD-043) ─────────────────────────
//
// `#[cfg(feature = "sparse")]` only. Bandwidth is always an explicit
// parameter here, supplied by the caller (the solver's own HOW-side
// configuration, e.g. `BackwardEulerSolver::with_jacobian_bandwidth`) — it
// is never read from `Domain`/`PhysicalModel`. See DD-043, amendment 1, for
// why `PhysicalModel::jacobian_bandwidth` was rejected: bandwidth is a
// property of the numerical scheme (HOW), not of the physical model (WHAT).

/// A single Jacobian entry: `(row, col, value)` — one nonzero of `∂f/∂u`
/// within the band, as produced by [`banded_jacobian_entries`]. Not yet a
/// `faer` triplet: callers convert only when they actually need the
/// sparse matrix type (see [`banded_finite_difference_jacobian`] and
/// [`theta_method_step_adaptive`]), rather than paying that conversion
/// cost inside the assembly loop itself.
#[cfg(feature = "sparse")]
type SparseEntry = (usize, usize, f64);

/// Return type of [`banded_jacobian_entries`]: the system size `n`, and
/// the raw list of [`SparseEntry`] within the declared band. Each
/// `(row, col)` pair appears at most once — every column belongs to
/// exactly one CPR color, so no two colors can ever produce the same
/// entry.
#[cfg(feature = "sparse")]
type BandedJacobianResult = Result<(usize, Vec<SparseEntry>), OxiflowError>;

/// Assembles the nonzero entries of $\partial f/\partial u$ within a band of
/// half-width `bandwidth` around the diagonal, via Curtis-Powell-Reid (CPR)
/// graph coloring: for a genuinely banded Jacobian, columns more than
/// `2*bandwidth + 1` apart never share a nonzero row, so all columns of the
/// same color can be perturbed **simultaneously** in a single
/// [`evaluate_derivative`] call — `2*bandwidth + 1` evaluations total,
/// instead of [`finite_difference_jacobian`]'s `n`.
///
/// Returns `(n, triplets)` — raw `(row, col, value)` entries, not yet a
/// `faer` sparse matrix, so callers computing the *system* matrix
/// (`I - theta*dt*J`, see [`theta_method_step_adaptive`]) can fold the
/// identity in before building the sparse type once, rather than building
/// the raw Jacobian in sparse form only to rebuild it again.
#[cfg(feature = "sparse")]
pub(crate) fn banded_jacobian_entries(
    domain: &Domain,
    chain: &[&dyn ContextCalculator],
    state: &ContextValue,
    t: f64,
    dt: f64,
    bandwidth: usize,
) -> BandedJacobianResult {
    let base_field = state.as_scalar_field()?.clone();
    let n = base_field.len();

    let mut base_state = state.clone();
    let f0 = evaluate_derivative(domain, chain, &mut base_state, t, dt)?;
    let f0_field = f0.as_scalar_field()?.clone();

    let n_colors = 2 * bandwidth + 1;
    let mut triplets = Vec::new();

    for color in 0..n_colors {
        let columns: Vec<usize> = (color..n).step_by(n_colors).collect();
        if columns.is_empty() {
            continue;
        }

        let mut perturbed_field = base_field.clone();
        for &j in &columns {
            perturbed_field[j] += FD_EPSILON;
        }
        let mut perturbed_state = ContextValue::ScalarField(perturbed_field);

        let f_c = evaluate_derivative(domain, chain, &mut perturbed_state, t, dt)?;
        let f_c_field = f_c.as_scalar_field()?;

        // Columns of the same color are spaced `n_colors > 2*bandwidth`
        // apart, so their bands ([j-bandwidth, j+bandwidth]) never
        // overlap — each perturbed row's response is attributable to
        // exactly one column, no disambiguation needed.
        for &j in &columns {
            let lo = j.saturating_sub(bandwidth);
            let hi = (j + bandwidth + 1).min(n);
            for i in lo..hi {
                let deriv = (f_c_field[i] - f0_field[i]) / FD_EPSILON;
                triplets.push((i, j, deriv));
            }
        }
    }

    Ok((n, triplets))
}

/// [`banded_jacobian_entries`], assembled into a `faer` sparse matrix.
///
/// Kept separate from [`banded_jacobian_entries`] for dense/sparse parity
/// testing against [`finite_difference_jacobian`] — see the test module.
// TODO(v0.6.0): wire in once BackwardEulerSolver/CrankNicolsonSolver
// select the banded path in practice (DD-043) — currently only exercised
// by the dense/sparse parity test.
#[allow(dead_code)]
#[cfg(feature = "sparse")]
pub(crate) fn banded_finite_difference_jacobian(
    domain: &Domain,
    chain: &[&dyn ContextCalculator],
    state: &ContextValue,
    t: f64,
    dt: f64,
    bandwidth: usize,
) -> Result<faer::sparse::SparseColMat<usize, f64>, OxiflowError> {
    let (n, triplets) = banded_jacobian_entries(domain, chain, state, t, dt, bandwidth)?;

    let faer_triplets: Vec<faer::sparse::Triplet<usize, usize, f64>> = triplets
        .into_iter()
        .map(|(i, j, v)| faer::sparse::Triplet::new(i, j, v))
        .collect();

    faer::sparse::SparseColMat::try_new_from_triplets(n, n, &faer_triplets).map_err(|e| {
        OxiflowError::PreconditionFailed {
            context: "banded_finite_difference_jacobian",
            message: format!("failed to build sparse matrix from triplets: {e:?}"),
        }
    })
}

/// Same contract as [`theta_method_step`], with an additional sparse path:
/// when `jacobian_bandwidth` is `Some(k)` **and** `n > sparse_threshold`,
/// assembles `I - theta*dt*J` directly in sparse (banded) form via
/// [`banded_jacobian_entries`] and solves it with `sparse_solver`. Falls
/// back to [`theta_method_step`] unchanged in every other case — `None`
/// bandwidth, or a small enough system that the dense path is cheaper
/// regardless of bandwidth.
///
/// `jacobian_bandwidth`/`sparse_threshold`/`sparse_solver` are supplied by
/// the caller's own HOW-side configuration (DD-043, amendment 1) — never
/// read from `Domain` or `PhysicalModel`.
#[cfg(feature = "sparse")]
#[allow(clippy::too_many_arguments)]
pub(crate) fn theta_method_step_adaptive(
    domain: &Domain,
    chain: &[&dyn ContextCalculator],
    state: &mut ContextValue,
    t: f64,
    dt: f64,
    theta: f64,
    dense_solver: &dyn LinearSolver,
    sparse_solver: &dyn SparseLinearSolver,
    sparse_threshold: usize,
    jacobian_bandwidth: Option<usize>,
) -> Result<ContextValue, OxiflowError> {
    let n = state.as_scalar_field()?.len();

    match jacobian_bandwidth {
        Some(k) if n > sparse_threshold => {
            let f_n = evaluate_derivative(domain, chain, state, t, dt)?;
            let u_n_field = state.as_scalar_field()?.clone();
            let f_n_field = f_n.as_scalar_field()?.clone();

            // Jacobian frozen at the (now BC-corrected) u^n, evaluated at
            // t + dt — same convention as theta_method_step.
            let (n_check, raw_triplets) =
                banded_jacobian_entries(domain, chain, state, t + dt, dt, k)?;
            debug_assert_eq!(n_check, n);

            // I - theta*dt*J, folded into the triplets directly rather than
            // building J's sparse matrix first and combining afterward —
            // banded_jacobian_entries always includes the diagonal (j is
            // always within its own band), so every row gets its "+1.0".
            let coeff = -theta * dt;
            let system_triplets: Vec<faer::sparse::Triplet<usize, usize, f64>> = raw_triplets
                .into_iter()
                .map(|(i, j, deriv)| {
                    let mut value = coeff * deriv;
                    if i == j {
                        value += 1.0;
                    }
                    faer::sparse::Triplet::new(i, j, value)
                })
                .collect();

            let system_matrix =
                faer::sparse::SparseColMat::try_new_from_triplets(n, n, &system_triplets).map_err(
                    |e| OxiflowError::PreconditionFailed {
                        context: "theta_method_step_adaptive",
                        message: format!("failed to build sparse system matrix: {e:?}"),
                    },
                )?;

            let rhs = f_n_field * dt;
            let delta_u = sparse_solver.solve(&system_matrix, &rhs)?;

            let u_next: DVector<f64> = u_n_field + delta_u;
            Ok(ContextValue::ScalarField(u_next))
        }
        _ => theta_method_step(domain, chain, state, t, dt, theta, dense_solver),
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::context::compute::ComputeContext;
    use crate::context::variable::ContextVariable;
    use crate::mesh::{Mesh, UniformGrid1D};
    use crate::model::traits::{PhysicalModel, RequiresContext};
    use crate::solver::chain::build_calculator_chain;
    use crate::solver::linear::NalgebraDenseSolver;
    use crate::solver::scenario::Scenario;

    #[derive(Debug)]
    struct ExponentialDecay {
        lambda: f64,
    }

    impl RequiresContext for ExponentialDecay {
        fn required_variables(&self) -> Vec<ContextVariable> {
            vec![]
        }
    }

    impl PhysicalModel for ExponentialDecay {
        fn compute_physics(
            &self,
            state: &ContextValue,
            _ctx: &ComputeContext,
        ) -> Result<ContextValue, OxiflowError> {
            let u = state.as_scalar_field()?;
            Ok(ContextValue::ScalarField(u.map(|v| -self.lambda * v)))
        }

        fn initial_state(&self, mesh: &dyn Mesh) -> ContextValue {
            ContextValue::ScalarField(DVector::from_element(mesh.n_dof(), 1.0))
        }

        fn name(&self) -> &str {
            "exponential_decay"
        }
    }

    fn make_mesh(n: usize) -> Box<dyn Mesh> {
        Box::new(UniformGrid1D::new(n, 0.0, 1.0).unwrap())
    }

    #[test]
    fn jacobian_of_linear_decay_is_minus_lambda_identity() {
        let lambda = 2.5;
        let scenario = Scenario::single(Box::new(ExponentialDecay { lambda }), make_mesh(3));
        let domain = scenario.single_domain().unwrap();
        let requirements = scenario.context_requirements();
        let chain = build_calculator_chain(&requirements, &[]).unwrap();

        let state = domain.model.initial_state(domain.mesh.as_ref());
        let jac = finite_difference_jacobian(domain, &chain, &state, 0.0, 0.1).unwrap();

        for i in 0..3 {
            for j in 0..3 {
                let expected = if i == j { -lambda } else { 0.0 };
                assert!(
                    (jac[(i, j)] - expected).abs() < 1e-4,
                    "jac[{i},{j}] = {} (expected {expected})",
                    jac[(i, j)]
                );
            }
        }
    }

    #[test]
    fn backward_euler_theta_one_matches_analytical_for_linear_decay() {
        // For du/dt = -lambda*u, backward Euler gives:
        // u^{n+1} = u^n / (1 + lambda*dt)
        let lambda = 3.0;
        let dt = 0.1;
        let scenario = Scenario::single(Box::new(ExponentialDecay { lambda }), make_mesh(2));
        let domain = scenario.single_domain().unwrap();
        let requirements = scenario.context_requirements();
        let chain = build_calculator_chain(&requirements, &[]).unwrap();

        let mut state = domain.model.initial_state(domain.mesh.as_ref());
        let next = theta_method_step(
            domain,
            &chain,
            &mut state,
            0.0,
            dt,
            1.0,
            &NalgebraDenseSolver,
        )
        .unwrap();

        let expected = 1.0 / (1.0 + lambda * dt);
        let field = next.as_scalar_field().unwrap();
        for v in field.iter() {
            assert!((v - expected).abs() < 1e-9, "got {v}, expected {expected}");
        }
    }

    #[test]
    fn crank_nicolson_theta_half_matches_analytical_for_linear_decay() {
        // For du/dt = -lambda*u, Crank-Nicolson gives:
        // u^{n+1} = u^n * (1 - lambda*dt/2) / (1 + lambda*dt/2)
        let lambda = 3.0;
        let dt = 0.1;
        let scenario = Scenario::single(Box::new(ExponentialDecay { lambda }), make_mesh(2));
        let domain = scenario.single_domain().unwrap();
        let requirements = scenario.context_requirements();
        let chain = build_calculator_chain(&requirements, &[]).unwrap();

        let mut state = domain.model.initial_state(domain.mesh.as_ref());
        let next = theta_method_step(
            domain,
            &chain,
            &mut state,
            0.0,
            dt,
            0.5,
            &NalgebraDenseSolver,
        )
        .unwrap();

        let expected = (1.0 - lambda * dt / 2.0) / (1.0 + lambda * dt / 2.0);
        let field = next.as_scalar_field().unwrap();
        for v in field.iter() {
            assert!((v - expected).abs() < 1e-9, "got {v}, expected {expected}");
        }
    }

    #[test]
    fn backward_euler_stable_for_very_stiff_problem() {
        // lambda*dt = 1000 -- far beyond any explicit method's stability
        // limit. Backward Euler must remain bounded and well-behaved.
        let lambda = 1.0e4;
        let dt = 0.1; // lambda*dt = 1000
        let scenario = Scenario::single(Box::new(ExponentialDecay { lambda }), make_mesh(2));
        let domain = scenario.single_domain().unwrap();
        let requirements = scenario.context_requirements();
        let chain = build_calculator_chain(&requirements, &[]).unwrap();

        let mut state = domain.model.initial_state(domain.mesh.as_ref());
        let next = theta_method_step(
            domain,
            &chain,
            &mut state,
            0.0,
            dt,
            1.0,
            &NalgebraDenseSolver,
        )
        .unwrap();

        let field = next.as_scalar_field().unwrap();
        for v in field.iter() {
            assert!(v.is_finite(), "value diverged: {v}");
            assert!(v.abs() < 1.0, "expected strong damping, got {v}");
        }
    }
}

#[cfg(all(test, feature = "sparse"))]
mod sparse_tests {
    use super::*;
    use crate::context::compute::ComputeContext;
    use crate::context::variable::ContextVariable;
    use crate::mesh::{Mesh, UniformGrid1D};
    use crate::model::traits::{PhysicalModel, RequiresContext};
    use crate::solver::chain::build_calculator_chain;
    use crate::solver::linear::NalgebraDenseSolver;
    use crate::solver::scenario::Scenario;
    use crate::solver::sparse::FaerSparseSolver;

    /// Discrete tridiagonal diffusion — `du_i/dt = D*(u[i-1] - 2u[i] +
    /// u[i+1])`, zero outside the domain. Not a real `BoundaryCondition` —
    /// this fixture exists only to exercise the banded path with a
    /// genuinely local, bandwidth-1 coupling (real Dirichlet/Neumann
    /// boundary handling is out of scope here, see the module's own
    /// known-untested-limitation note).
    #[derive(Debug)]
    struct TridiagonalDiffusion {
        diffusion: f64,
    }

    impl RequiresContext for TridiagonalDiffusion {
        fn required_variables(&self) -> Vec<ContextVariable> {
            vec![]
        }
    }

    impl PhysicalModel for TridiagonalDiffusion {
        fn compute_physics(
            &self,
            state: &ContextValue,
            _ctx: &ComputeContext,
        ) -> Result<ContextValue, OxiflowError> {
            let u = state.as_scalar_field()?;
            let n = u.len();
            let d = self.diffusion;
            let out = DVector::from_fn(n, |i, _| {
                let left = if i > 0 { u[i - 1] } else { 0.0 };
                let right = if i + 1 < n { u[i + 1] } else { 0.0 };
                d * (left - 2.0 * u[i] + right)
            });
            Ok(ContextValue::ScalarField(out))
        }

        fn initial_state(&self, mesh: &dyn Mesh) -> ContextValue {
            let n = mesh.n_dof();
            ContextValue::ScalarField(DVector::from_fn(
                n,
                |i, _| if i == n / 2 { 1.0 } else { 0.0 },
            ))
        }

        fn name(&self) -> &str {
            "tridiagonal_diffusion_test"
        }
    }

    /// A [`SparseLinearSolver`] that panics if ever called — used to prove
    /// the dense path was actually taken (threshold/bandwidth conditions
    /// not met), not just that the result happens to look dense-like.
    struct PanicIfCalled;

    impl SparseLinearSolver for PanicIfCalled {
        fn solve(
            &self,
            _a: &faer::sparse::SparseColMat<usize, f64>,
            _b: &DVector<f64>,
        ) -> Result<DVector<f64>, OxiflowError> {
            panic!("sparse solver called when the dense path should have been taken");
        }
    }

    fn make_mesh(n: usize) -> Box<dyn Mesh> {
        Box::new(UniformGrid1D::new(n, 0.0, 1.0).unwrap())
    }

    #[test]
    fn banded_jacobian_entries_match_dense_jacobian() {
        let n = 12;
        let scenario = Scenario::single(
            Box::new(TridiagonalDiffusion { diffusion: 0.7 }),
            make_mesh(n),
        );
        let domain = scenario.single_domain().unwrap();
        let requirements = scenario.context_requirements();
        let chain = build_calculator_chain(&requirements, &[]).unwrap();
        let state = domain.model.initial_state(domain.mesh.as_ref());

        let dense = finite_difference_jacobian(domain, &chain, &state, 0.0, 0.01).unwrap();
        let (n_check, banded_triplets) =
            banded_jacobian_entries(domain, &chain, &state, 0.0, 0.01, 1).unwrap();
        assert_eq!(n_check, n);

        // Every banded entry must match the corresponding dense entry.
        for (i, j, v) in &banded_triplets {
            assert!(
                (dense[(*i, *j)] - v).abs() < 1e-6,
                "mismatch at ({i},{j}): dense={}, banded={v}",
                dense[(*i, *j)]
            );
        }

        // Every dense entry outside the band must be (near) zero — a
        // genuinely tridiagonal problem has no coupling beyond bandwidth 1,
        // so nothing should be missing from the banded set that matters.
        for i in 0..n {
            for j in 0..n {
                if i.abs_diff(j) > 1 {
                    assert!(
                        dense[(i, j)].abs() < 1e-6,
                        "unexpected nonzero outside the band at ({i},{j}): {}",
                        dense[(i, j)]
                    );
                }
            }
        }
    }

    #[test]
    fn adaptive_step_with_no_bandwidth_matches_theta_method_step_exactly() {
        let n = 6;
        let scenario = Scenario::single(
            Box::new(TridiagonalDiffusion { diffusion: 0.5 }),
            make_mesh(n),
        );
        let domain = scenario.single_domain().unwrap();
        let requirements = scenario.context_requirements();
        let chain = build_calculator_chain(&requirements, &[]).unwrap();

        let mut state_a = domain.model.initial_state(domain.mesh.as_ref());
        let mut state_b = state_a.clone();

        let via_adaptive = theta_method_step_adaptive(
            domain,
            &chain,
            &mut state_a,
            0.0,
            0.05,
            1.0,
            &NalgebraDenseSolver,
            &PanicIfCalled,
            100,
            None, // no bandwidth declared -> dense path, unconditionally
        )
        .unwrap();

        let via_plain = theta_method_step(
            domain,
            &chain,
            &mut state_b,
            0.0,
            0.05,
            1.0,
            &NalgebraDenseSolver,
        )
        .unwrap();

        let a = via_adaptive.as_scalar_field().unwrap();
        let b = via_plain.as_scalar_field().unwrap();
        for i in 0..n {
            assert!(
                (a[i] - b[i]).abs() < 1e-12,
                "mismatch at {i}: {} vs {}",
                a[i],
                b[i]
            );
        }
    }

    #[test]
    fn small_system_stays_dense_even_with_bandwidth_declared() {
        let n = 6; // below sparse_threshold below
        let scenario = Scenario::single(
            Box::new(TridiagonalDiffusion { diffusion: 0.5 }),
            make_mesh(n),
        );
        let domain = scenario.single_domain().unwrap();
        let requirements = scenario.context_requirements();
        let chain = build_calculator_chain(&requirements, &[]).unwrap();
        let mut state = domain.model.initial_state(domain.mesh.as_ref());

        // PanicIfCalled proves the sparse path was never entered: bandwidth
        // is declared (Some(1)) but n (6) does not exceed sparse_threshold
        // (100), so the dense path must be taken regardless.
        let result = theta_method_step_adaptive(
            domain,
            &chain,
            &mut state,
            0.0,
            0.05,
            1.0,
            &NalgebraDenseSolver,
            &PanicIfCalled,
            100,
            Some(1),
        )
        .unwrap();

        assert!(result
            .as_scalar_field()
            .unwrap()
            .iter()
            .all(|v| v.is_finite()));
    }

    #[test]
    fn adaptive_step_with_bandwidth_matches_dense_result_above_threshold() {
        let n = 40; // above the small sparse_threshold used below
        let diffusion = 0.3;
        let dt = 0.02;

        let scenario_a =
            Scenario::single(Box::new(TridiagonalDiffusion { diffusion }), make_mesh(n));
        let domain_a = scenario_a.single_domain().unwrap();
        let requirements_a = scenario_a.context_requirements();
        let chain_a = build_calculator_chain(&requirements_a, &[]).unwrap();
        let mut state_a = domain_a.model.initial_state(domain_a.mesh.as_ref());

        let scenario_b =
            Scenario::single(Box::new(TridiagonalDiffusion { diffusion }), make_mesh(n));
        let domain_b = scenario_b.single_domain().unwrap();
        let requirements_b = scenario_b.context_requirements();
        let chain_b = build_calculator_chain(&requirements_b, &[]).unwrap();
        let mut state_b = domain_b.model.initial_state(domain_b.mesh.as_ref());

        let via_sparse = theta_method_step_adaptive(
            domain_a,
            &chain_a,
            &mut state_a,
            0.0,
            dt,
            1.0,
            &NalgebraDenseSolver,
            &FaerSparseSolver,
            10, // sparse_threshold well below n=40
            Some(1),
        )
        .unwrap();

        let via_dense = theta_method_step(
            domain_b,
            &chain_b,
            &mut state_b,
            0.0,
            dt,
            1.0,
            &NalgebraDenseSolver,
        )
        .unwrap();

        let a = via_sparse.as_scalar_field().unwrap();
        let b = via_dense.as_scalar_field().unwrap();
        for i in 0..n {
            assert!(
                (a[i] - b[i]).abs() < 1e-6,
                "sparse/dense mismatch at {i}: {} vs {}",
                a[i],
                b[i]
            );
        }
    }
}
