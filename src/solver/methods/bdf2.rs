//! # Module `solver::methods::bdf2`
//!
//! BDF2 integrator — implicit multi-step, 2nd order (issue #44).
//!
//! ## Algorithm
//!
//! $$\frac{3}{2}u^{n+1} - 2u^n + \frac{1}{2}u^{n-1} = \Delta t \cdot f(u^{n+1}, t^{n+1})$$
//!
//! Unlike Backward Euler / Crank-Nicolson ([`super::implicit`]), this needs
//! **two** past states, not one — `SteppableSolver::history_depth()`
//! returns `1` here (DD-034), and [`BDF2Solver::step`] receives `u^{n-1}`
//! via the `history` parameter.
//!
//! `bdf2_step` (below) is BDF2-specific and stays in this file rather than
//! [`super::implicit`]: unlike `theta_method_step`, which serves two
//! solvers (Backward Euler, Crank-Nicolson), this formula has exactly one
//! consumer. It still reuses [`super::implicit::finite_difference_jacobian`]
//! and [`super::evaluate_derivative`] — the genuinely shared pieces.
//!
//! ## Startup (acceptance criterion, #44)
//!
//! BDF2 needs `u^{n-1}`, which doesn't exist at the very first step. When
//! `history` is empty, [`BDF2Solver::step`] falls back to a single
//! Backward Euler step (θ=1, via [`super::implicit::theta_method_step`])
//! to produce `u^1` from `u^0` — the standard bootstrap for 2-step methods.
//! From the second step onward, the real BDF2 update runs.
//!
//! ## Scope at J4a — fixed `dt` only
//!
//! *"Step-size change logic for adaptive dt"* (as #44 originally asked)
//! is **not implemented**: `StepControl::Adaptive` doesn't exist yet
//! anywhere in `oxiflow` (rejected by every solver, including this one),
//! and #42 (DoPri45) — the issue that would introduce it — hasn't landed.
//! Writing a variable-step BDF2 formula now would mean testing it against
//! nothing real. Revisit once #42 lands; the classical fixed-step
//! coefficients used here (3/2, -2, 1/2) would need to become
//! ratio-dependent for a varying step size between `u^{n-1}` and `u^n`.
//!
//! ## Stability
//!
//! A-stable and, unlike Crank-Nicolson, L-stable — strong damping on
//! stiff modes, same qualitative behaviour as Backward Euler but 2nd
//! order accurate.

use nalgebra::{DMatrix, DVector};

use crate::context::error::OxiflowError;
use crate::context::value::ContextValue;
use crate::context::ContextCalculator;
use crate::solver::linear::{LinearSolver, NalgebraDenseSolver};
use crate::solver::methods::implicit::{finite_difference_jacobian, theta_method_step};
use crate::solver::methods::{evaluate_derivative, SteppableSolver};
use crate::solver::scenario::{Domain, Scenario};
use crate::solver::{SimulationResult, Solver, SolverConfiguration};

/// BDF2 solver — implicit multi-step, 2nd order.
///
/// See [module docs](self) for the startup phase and the fixed-`dt`-only
/// scope at J4a.
pub struct BDF2Solver {
    linear_solver: Box<dyn LinearSolver>,
}

impl Default for BDF2Solver {
    fn default() -> Self {
        Self {
            linear_solver: Box::new(NalgebraDenseSolver),
        }
    }
}

impl BDF2Solver {
    /// Creates a solver using the default `nalgebra` dense backend.
    pub fn new() -> Self {
        Self::default()
    }

    /// Substitutes the linear solver backend (DD-013).
    pub fn with_linear_solver(mut self, linear_solver: Box<dyn LinearSolver>) -> Self {
        self.linear_solver = linear_solver;
        self
    }
}

impl Solver for BDF2Solver {
    fn solve(
        &self,
        scenario: &Scenario,
        config: &SolverConfiguration,
    ) -> Result<SimulationResult, OxiflowError> {
        // `solve_fixed_step` (DD-034 follow-up) manages the history buffer
        // generically via `self.history_depth()` -- exactly equivalent to
        // the hand-written `history = vec![u]` this used to do, since
        // `insert(0, prev); truncate(1)` is the same operation for a
        // depth-1 buffer. No BDF2-specific loop needed any more.
        self.solve_fixed_step(scenario, config)
    }
}

impl SteppableSolver for BDF2Solver {
    fn history_depth(&self) -> usize {
        1
    }

    fn step(
        &self,
        domain: &Domain,
        chain: &[&dyn ContextCalculator],
        state: &mut ContextValue,
        history: &[ContextValue],
        t: f64,
        dt: f64,
    ) -> Result<ContextValue, OxiflowError> {
        match history.first() {
            // Startup (#44 acceptance criterion): no u^{n-1} yet -- bootstrap
            // with one Backward Euler step to produce u^1 from u^0.
            None => theta_method_step(
                domain,
                chain,
                state,
                t,
                dt,
                1.0,
                self.linear_solver.as_ref(),
            ),
            Some(u_prev) => bdf2_step(
                domain,
                chain,
                state,
                u_prev,
                t,
                dt,
                self.linear_solver.as_ref(),
            ),
        }
    }
}

/// Performs one BDF2 step, given `u^{n-1}` (`u_prev`) and `u^n` (`state`).
///
/// Same frozen-Jacobian, single-Newton-correction approach as
/// [`super::implicit::theta_method_step`] (DD-033): exact when `f` is
/// affine in `u`, a first-order approximation otherwise.
///
/// Derivation: linearising `g(u) = (3/2)u - 2u^n + (1/2)u^{n-1} - dt*f(u)`
/// at `u = u^n` gives `g(u^n) = -(1/2)u^n + (1/2)u^{n-1} - dt*f(u^n)` and
/// Jacobian `(3/2)I - dt*J_f`. The correction solves
/// `[(3/2)I - dt*J_f] * delta_u = -g(u^n)`.
fn bdf2_step(
    domain: &Domain,
    chain: &[&dyn ContextCalculator],
    state: &mut ContextValue,
    u_prev: &ContextValue,
    t: f64,
    dt: f64,
    linear_solver: &dyn LinearSolver,
) -> Result<ContextValue, OxiflowError> {
    // f(u^n, t) -- BCs applied in-place to `state` itself.
    let f_n = evaluate_derivative(domain, chain, state, t, dt)?;
    let u_n_field = state.as_scalar_field()?.clone();
    let f_n_field = f_n.as_scalar_field()?.clone();
    let u_prev_field = u_prev.as_scalar_field()?.clone();

    let n = u_n_field.len();
    if u_prev_field.len() != n {
        return Err(OxiflowError::InvalidDomain(format!(
            "history state length {} != current state length {n}",
            u_prev_field.len()
        )));
    }

    // Jacobian frozen at u^n, evaluated at the target time t + dt.
    let jacobian = finite_difference_jacobian(domain, chain, state, t + dt, dt)?;

    // System matrix: (3/2)*I - dt*J_f.
    let identity = DMatrix::<f64>::identity(n, n);
    let system_matrix = identity * 1.5 - jacobian * dt;

    // RHS: -g(u^n) = (1/2)*u^n - (1/2)*u^{n-1} + dt*f(u^n). Built from
    // fully-owned operands throughout -- see DD-033 rationale on avoiding
    // ambiguous reference/owned nalgebra operator resolution.
    let half_u_n = u_n_field.clone() * 0.5;
    let half_u_prev = u_prev_field * 0.5;
    let dt_f_n = f_n_field * dt;
    let rhs: DVector<f64> = half_u_n + dt_f_n - half_u_prev;

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
    use crate::solver::config::{
        IntegratorKind, SolverConfiguration, StepControl, TimeConfiguration,
    };
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;

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

    #[derive(Debug)]
    struct ZeroDerivative;

    impl RequiresContext for ZeroDerivative {
        fn required_variables(&self) -> Vec<ContextVariable> {
            vec![]
        }
    }

    impl PhysicalModel for ZeroDerivative {
        fn compute_physics(
            &self,
            state: &ContextValue,
            _ctx: &ComputeContext,
        ) -> Result<ContextValue, OxiflowError> {
            let u = state.as_scalar_field()?;
            Ok(ContextValue::ScalarField(DVector::zeros(u.len())))
        }

        fn initial_state(&self, mesh: &dyn Mesh) -> ContextValue {
            ContextValue::ScalarField(DVector::from_element(mesh.n_dof(), 2.5))
        }

        fn name(&self) -> &str {
            "zero_derivative"
        }
    }

    #[derive(Debug)]
    struct CountingLinearSolver {
        calls: Arc<AtomicUsize>,
    }

    impl LinearSolver for CountingLinearSolver {
        fn solve(&self, a: &DMatrix<f64>, b: &DVector<f64>) -> Result<DVector<f64>, OxiflowError> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            NalgebraDenseSolver.solve(a, b)
        }
    }

    fn make_config(t_end: f64, dt: f64) -> SolverConfiguration {
        SolverConfiguration::new(
            TimeConfiguration::new(t_end, StepControl::Fixed { dt }),
            IntegratorKind::BDF2,
        )
    }

    fn make_mesh(n: usize) -> Box<dyn Mesh> {
        Box::new(UniformGrid1D::new(n, 0.0, 1.0).unwrap())
    }

    // ── Basic correctness ─────────────────────────────────────────────────────

    #[test]
    fn history_depth_is_one() {
        assert_eq!(BDF2Solver::new().history_depth(), 1);
    }

    #[test]
    fn zero_derivative_field_stays_constant() {
        let scenario = Scenario::single(Box::new(ZeroDerivative), make_mesh(5));
        let config = make_config(1.0, 0.1);
        let result = BDF2Solver::new().solve(&scenario, &config).unwrap();
        for state in &result.states {
            let field = state.as_scalar_field().unwrap();
            for v in field.iter() {
                assert!((v - 2.5).abs() < 1e-12);
            }
        }
    }

    #[test]
    fn result_times_match_expected_steps() {
        let scenario = Scenario::single(Box::new(ZeroDerivative), make_mesh(3));
        let config = make_config(0.5, 0.1);
        let result = BDF2Solver::new().solve(&scenario, &config).unwrap();
        assert_eq!(result.states.len(), result.times.len());
        assert!((result.times[0] - 0.0).abs() < 1e-12);
        assert!((result.t_final().unwrap() - 0.5).abs() < 1e-9);
    }

    #[test]
    fn n_steps_is_correct() {
        let scenario = Scenario::single(Box::new(ZeroDerivative), make_mesh(2));
        let config = make_config(1.0, 0.25);
        let result = BDF2Solver::new().solve(&scenario, &config).unwrap();
        assert_eq!(result.n_steps, 4);
    }

    #[test]
    fn save_every_reduces_stored_states() {
        let scenario = Scenario::single(Box::new(ZeroDerivative), make_mesh(2));
        let config = SolverConfiguration::new(
            TimeConfiguration::new(1.0, StepControl::Fixed { dt: 0.1 }).saving_every(5),
            IntegratorKind::BDF2,
        );
        let result = BDF2Solver::new().solve(&scenario, &config).unwrap();
        assert_eq!(result.states.len(), 3);
    }

    // ── Startup phase (acceptance criterion, #44) ─────────────────────────────

    #[test]
    fn startup_step_matches_backward_euler_formula() {
        // First step has no history -- must fall back to exactly one
        // Backward Euler step: u^1 = u^0 / (1 + lambda*dt).
        let lambda = 3.0;
        let dt = 0.1;
        let scenario = Scenario::single(Box::new(ExponentialDecay { lambda }), make_mesh(2));
        let config = make_config(dt, dt); // exactly one step
        let result = BDF2Solver::new().solve(&scenario, &config).unwrap();

        let expected = 1.0 / (1.0 + lambda * dt);
        let field = result.states.last().unwrap().as_scalar_field().unwrap();
        for v in field.iter() {
            assert!((v - expected).abs() < 1e-9, "got {v}, expected {expected}");
        }
    }

    #[test]
    fn step_matches_one_iteration_of_solve() {
        let scenario = Scenario::single(Box::new(ExponentialDecay { lambda: 0.7 }), make_mesh(3));
        let config = make_config(0.1, 0.1); // exactly one step -- exercises startup

        let solver = BDF2Solver::new();
        let via_solve = solver.solve(&scenario, &config).unwrap();
        let final_via_solve = via_solve.states.last().unwrap().as_scalar_field().unwrap();

        let domain = scenario.single_domain().unwrap();
        let requirements = scenario.context_requirements();
        let chain =
            crate::solver::chain::build_calculator_chain(&requirements, &config.calculators)
                .unwrap();
        let mut u = domain.model.initial_state(domain.mesh.as_ref());
        let next = solver.step(domain, &chain, &mut u, &[], 0.0, 0.1).unwrap();
        let final_via_step = next.as_scalar_field().unwrap();

        assert_eq!(final_via_solve.len(), final_via_step.len());
        for i in 0..final_via_solve.len() {
            assert!((final_via_solve[i] - final_via_step[i]).abs() < 1e-15);
        }
    }

    // ── Post-startup recurrence ────────────────────────────────────────────────

    #[test]
    fn exponential_decay_matches_recurrence_after_startup() {
        // Independent reimplementation of the BDF2 recurrence (derived by
        // hand, see module docs) -- cross-checks the solver's output
        // rather than re-deriving the same formula from its own code.
        let lambda = 2.0;
        let dt = 0.1;
        let n_steps: usize = 10;
        let scenario = Scenario::single(Box::new(ExponentialDecay { lambda }), make_mesh(2));
        let config = make_config(n_steps as f64 * dt, dt);
        let result = BDF2Solver::new().solve(&scenario, &config).unwrap();

        let mut u_prev = 1.0_f64; // u^0
        let mut u_curr = u_prev / (1.0 + lambda * dt); // u^1, Backward Euler startup
        for _ in 1..n_steps {
            let u_next = (2.0 * u_curr - 0.5 * u_prev) / (1.5 + lambda * dt);
            u_prev = u_curr;
            u_curr = u_next;
        }

        let final_field = result.states.last().unwrap().as_scalar_field().unwrap();
        for v in final_field.iter() {
            assert!((v - u_curr).abs() < 1e-9, "got {v}, expected {u_curr}");
        }
    }

    // ── Stability ──────────────────────────────────────────────────────────────

    #[test]
    fn stable_for_very_stiff_problem_over_many_steps() {
        let lambda = 1.0e4;
        let dt = 0.1;
        let scenario = Scenario::single(Box::new(ExponentialDecay { lambda }), make_mesh(2));
        let config = make_config(2.0, dt);
        let result = BDF2Solver::new().solve(&scenario, &config).unwrap();

        for state in &result.states {
            let field = state.as_scalar_field().unwrap();
            for v in field.iter() {
                assert!(v.is_finite(), "value diverged: {v}");
                assert!(
                    v.abs() <= 1.0,
                    "expected strong damping (L-stable), got {v}"
                );
            }
        }
    }

    // ── LinearSolver substitution (DD-013) ────────────────────────────────────

    #[test]
    fn with_linear_solver_substitutes_backend() {
        let calls = Arc::new(AtomicUsize::new(0));
        let solver = BDF2Solver::new().with_linear_solver(Box::new(CountingLinearSolver {
            calls: calls.clone(),
        }));

        let scenario = Scenario::single(Box::new(ExponentialDecay { lambda: 1.0 }), make_mesh(2));
        let config = make_config(0.5, 0.1); // 5 steps: 1 startup + 4 BDF2

        solver.solve(&scenario, &config).unwrap();

        assert_eq!(
            calls.load(Ordering::SeqCst),
            5,
            "expected one linear solve per step, startup included"
        );
    }

    // ── Validation errors ─────────────────────────────────────────────────────

    #[test]
    fn negative_dt_returns_error() {
        let scenario = Scenario::single(Box::new(ZeroDerivative), make_mesh(2));
        let config = make_config(1.0, -0.1);
        assert!(BDF2Solver::new().solve(&scenario, &config).is_err());
    }

    #[test]
    fn t_end_before_t_start_returns_error() {
        let scenario = Scenario::single(Box::new(ZeroDerivative), make_mesh(2)).with_t_start(5.0);
        let config = make_config(1.0, 0.1);
        assert!(BDF2Solver::new().solve(&scenario, &config).is_err());
    }

    #[test]
    fn adaptive_step_control_returns_error() {
        let scenario = Scenario::single(Box::new(ZeroDerivative), make_mesh(2));
        let config = SolverConfiguration::new(
            TimeConfiguration::new(
                1.0,
                StepControl::Adaptive {
                    dt_init: 0.1,
                    dt_min: 1e-6,
                    dt_max: 1.0,
                    rtol: 1e-6,
                    atol: 1e-9,
                },
            ),
            IntegratorKind::BDF2,
        );
        assert!(BDF2Solver::new().solve(&scenario, &config).is_err());
    }

    #[test]
    fn missing_calculator_returns_error() {
        #[derive(Debug)]
        struct NeedsExternal;
        impl RequiresContext for NeedsExternal {
            fn required_variables(&self) -> Vec<ContextVariable> {
                vec![ContextVariable::External {
                    name: "missing".into(),
                }]
            }
        }
        impl PhysicalModel for NeedsExternal {
            fn compute_physics(
                &self,
                s: &ContextValue,
                _: &ComputeContext,
            ) -> Result<ContextValue, OxiflowError> {
                Ok(s.clone())
            }
            fn initial_state(&self, mesh: &dyn Mesh) -> ContextValue {
                ContextValue::ScalarField(DVector::from_element(mesh.n_dof(), 0.0))
            }
            fn name(&self) -> &str {
                "needs_external"
            }
        }

        let scenario = Scenario::single(Box::new(NeedsExternal), make_mesh(2));
        let config = make_config(1.0, 0.1);
        let err = BDF2Solver::new().solve(&scenario, &config).unwrap_err();
        assert!(matches!(err, OxiflowError::MissingCalculator(_)));
    }

    #[test]
    fn bdf2_step_history_length_mismatch_returns_error() {
        let scenario = Scenario::single(Box::new(ExponentialDecay { lambda: 1.0 }), make_mesh(3));
        let domain = scenario.single_domain().unwrap();
        let requirements = scenario.context_requirements();
        let chain = crate::solver::chain::build_calculator_chain(&requirements, &[]).unwrap();

        let mut state = domain.model.initial_state(domain.mesh.as_ref());
        let wrong_length_prev = ContextValue::ScalarField(DVector::from_element(2, 1.0)); // mesh has 3 nodes

        let err = bdf2_step(
            domain,
            &chain,
            &mut state,
            &wrong_length_prev,
            0.0,
            0.1,
            &NalgebraDenseSolver,
        )
        .unwrap_err();
        assert!(matches!(err, OxiflowError::InvalidDomain(_)));
    }
}
