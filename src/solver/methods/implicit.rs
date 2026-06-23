//! # Module `solver::methods::implicit`
//!
//! Shared machinery for implicit (theta-method) integrators — DD-033.
//!
//! [`BackwardEulerSolver`](super::backward_euler::BackwardEulerSolver) (θ=1)
//! and [`CrankNicolsonSolver`](super::crank_nicolson::CrankNicolsonSolver)
//! (θ=0.5) are thin wrappers around [`theta_method_step`], which itself
//! relies on [`finite_difference_jacobian`].
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
//! ## v1 scope — frozen Jacobian, single correction (DD-033)
//!
//! [`theta_method_step`] performs exactly **one** Newton-style correction,
//! with the Jacobian frozen at $u^n$. This is exact when `compute_physics`
//! is affine in `u` — the stiff *linear* test problems these methods
//! target (`λΔt ≫ 1`) — and a first-order approximation otherwise.
//!
//! A future nonlinear solver (Newton iterated to convergence, v0.6.0–v1.0.0,
//! DD-033) would call this same function repeatedly, re-evaluating
//! [`finite_difference_jacobian`] at each updated guess if the frozen
//! approximation is dropped — neither function needs rewriting, only the
//! calling loop changes.
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
use crate::context::error::OxiflowError;
use crate::context::value::ContextValue;
use crate::context::ContextCalculator;
use crate::solver::linear::LinearSolver;
use crate::solver::scenario::Domain;

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
pub(crate) fn theta_method_step(
    domain: &Domain,
    chain: &[&dyn ContextCalculator],
    state: &mut ContextValue,
    t: f64,
    dt: f64,
    theta: f64,
    linear_solver: &dyn LinearSolver,
) -> Result<ContextValue, OxiflowError> {
    // f(u^n, t) -- BCs applied in-place to `state` itself, consistent with
    // the explicit solvers' contract. Everything below reads the
    // now-BC-corrected `state`, not a separate untouched clone.
    let f_n = evaluate_derivative(domain, chain, state, t, dt)?;
    let u_n_field = state.as_scalar_field()?.clone();
    let f_n_field = f_n.as_scalar_field()?.clone();

    let n = u_n_field.len();

    // Jacobian frozen at the (now BC-corrected) u^n, evaluated at the
    // *target* time t + dt (the point the implicit equation is actually
    // stated at).
    let jacobian = finite_difference_jacobian(domain, chain, state, t + dt, dt)?;

    // System matrix: I - theta * dt * J_f. RHS: dt * f(u^n) -- identical
    // for any theta, see module docs.
    let identity = DMatrix::<f64>::identity(n, n);
    let system_matrix = identity - jacobian * (theta * dt);
    let rhs = f_n_field * dt;

    let delta_u = linear_solver.solve(&system_matrix, &rhs)?;

    let u_next: DVector<f64> = u_n_field + delta_u;
    Ok(ContextValue::ScalarField(u_next))
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
