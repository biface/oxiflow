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
//! The BDF2 update is the `ImplicitFormula::Bdf2` formula of
//! [`super::implicit`]: $\alpha = 3/2$, $c = \Delta t$, $u^{*} =
//! \frac{4u^n - u^{n-1}}{3}$ in [`super::newton`]'s residual convention.
//! It runs through `super::implicit::implicit_step_newton`, like the
//! theta method, so the Newton loop, the residual, the finite-difference
//! Jacobian and the solver's `NewtonParams` are those of the other
//! implicit solvers (DD-044, [#112](https://github.com/biface/oxiflow/issues/112)).
//!
//! ## Startup (acceptance criterion, #44)
//!
//! BDF2 needs `u^{n-1}`, which doesn't exist at the very first step. When
//! `history` is empty, [`BDF2Solver::step`] falls back to a single
//! Backward Euler step (`ImplicitFormula::Theta(1.0)`)
//! to produce `u^1` from `u^0` — the standard bootstrap for 2-step methods.
//! From the second step onward, the real BDF2 update runs.
//!
//! The bootstrap step runs the same iterated Newton loop as the BDF2
//! update, with the solver's own convergence criterion, Jacobian
//! strategy and iteration budget: a nonlinear model gets the budget it
//! was configured with from the first step on. With the default
//! configuration, the bootstrap step is the single frozen-Jacobian
//! correction it has always been.
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
//! ## State variants
//!
//! The update works on the flat vector of unknowns given by `StateLayout`
//! (`super::state_layout`), as the θ-method does: a `VectorField` state,
//! interleaved by node, is advanced like a `ScalarField`, and so is the
//! Backward Euler startup step (#192). For a `ScalarField`, results are
//! bit-identical to the former code.
//!
//! ## Banded path (feature `sparse`, DD-043)
//!
//! Like [`BackwardEulerSolver`](super::backward_euler::BackwardEulerSolver)
//! and [`CrankNicolsonSolver`](super::crank_nicolson::CrankNicolsonSolver),
//! the solver takes a banded linear system once a sparse solver
//! (`with_sparse_solver`) and a Jacobian half-bandwidth
//! (`with_jacobian_bandwidth`) are configured and the system has more
//! unknowns than `with_sparse_threshold` (default 100). The bootstrap
//! step and the BDF2 update both take it, through the same Newton loop
//! and with the same Newton parameters as the dense path (#133).
//!
//! ## Stability
//!
//! A-stable and, unlike Crank-Nicolson, L-stable — strong damping on
//! stiff modes, same qualitative behaviour as Backward Euler but 2nd
//! order accurate.

use crate::context::error::OxiflowError;
use crate::context::value::ContextValue;
use crate::context::ContextCalculator;
use crate::solver::linear::LinearSolver;
use crate::solver::methods::implicit::{ImplicitBackend, ImplicitFormula, JacobianDispatch};
use crate::solver::methods::newton::{JacobianStrategy, NewtonConvergence, NewtonParams};
use crate::solver::methods::SteppableSolver;
use crate::solver::scenario::{Domain, Scenario};
#[cfg(feature = "sparse")]
use crate::solver::sparse::SparseLinearSolver;
use crate::solver::{SimulationResult, Solver, SolverConfiguration};

/// BDF2 solver — implicit multi-step, 2nd order.
///
/// See [module docs](self) for the startup phase and the fixed-`dt`-only
/// scope at J4a.
///
/// # Examples
///
/// ```rust
/// use oxiflow::solver::methods::bdf2::BDF2Solver;
///
/// // Dense path (default).
/// let solver = BDF2Solver::new();
///
/// // Banded path (feature `sparse`, DD-043): the dense path stays
/// // untouched unless a sparse solver and a bandwidth are both configured.
/// # #[cfg(feature = "sparse")]
/// # {
/// use oxiflow::solver::sparse::FaerSparseSolver;
/// let banded = BDF2Solver::new()
///     .with_jacobian_bandwidth(3) // two interleaved components, stencil half-width 1
///     .with_sparse_solver(Box::new(FaerSparseSolver))
///     .with_sparse_threshold(100); // default
/// # }
/// ```
#[derive(Default)]
pub struct BDF2Solver {
    /// Linear-system backends (DD-013, DD-043): the dense solver and,
    /// with the `sparse` feature, the banded-path configuration, set by
    /// `with_linear_solver` and the `with_sparse_*`/`with_jacobian_bandwidth`
    /// builders, for the startup step and the BDF2 update alike.
    backend: ImplicitBackend,
    /// Newton configuration (DD-044, [#111](https://github.com/biface/oxiflow/issues/111)): convergence criterion,
    /// Jacobian refresh strategy and iteration budget, set by the
    /// `with_newton_convergence`/`with_jacobian_strategy`/
    /// `with_max_newton_iterations` builders. Default
    /// [`NewtonParams::default`]: a single frozen-Jacobian correction under
    /// [`stiff_jacobian_convergence`](super::implicit::stiff_jacobian_convergence),
    /// exact on affine problems; raise the
    /// budget (typically with [`JacobianStrategy::FullNewton`]) for a
    /// nonlinear model.
    newton: NewtonParams,
    /// Dispatch configuration of the finite-difference Jacobian assembly
    /// (DT-069, #198), for the startup step and the BDF2 update alike. Its
    /// threshold exists only with the `parallel` feature — see
    /// `with_parallel_threshold`.
    jacobian_dispatch: JacobianDispatch,
}

impl BDF2Solver {
    /// Creates a solver using the default `nalgebra` dense backend.
    pub fn new() -> Self {
        Self::default()
    }

    /// Substitutes the linear solver backend (DD-013).
    pub fn with_linear_solver(mut self, linear_solver: Box<dyn LinearSolver>) -> Self {
        self.backend.linear_solver = linear_solver;
        self
    }

    /// Configures the Newton convergence criterion (DD-044, [#112](https://github.com/biface/oxiflow/issues/112)), for the
    /// Backward Euler bootstrap step and the BDF2 update alike (see
    /// [module docs](self)).
    pub fn with_newton_convergence(mut self, newton_convergence: NewtonConvergence) -> Self {
        self.newton.convergence = newton_convergence;
        self
    }

    /// Configures the Jacobian refresh strategy (DD-044, [#112](https://github.com/biface/oxiflow/issues/112)).
    pub fn with_jacobian_strategy(mut self, jacobian_strategy: JacobianStrategy) -> Self {
        self.newton.jacobian_strategy = jacobian_strategy;
        self
    }

    /// Configures the Newton iteration budget (DD-044, [#112](https://github.com/biface/oxiflow/issues/112)) — see
    /// [`BackwardEulerSolver::with_max_newton_iterations`](super::backward_euler::BackwardEulerSolver::with_max_newton_iterations).
    pub fn with_max_newton_iterations(mut self, max_newton_iterations: usize) -> Self {
        self.newton.max_iterations = max_newton_iterations;
        self
    }

    /// Overrides the Rayon dispatch threshold of the finite-difference
    /// Jacobian assembly at construction time (DT-069, #198), in unknowns
    /// (nodes × components). The Jacobian's columns (dense path) or colors
    /// (banded path) are dispatched through Rayon once the system has at
    /// least this many unknowns; the Jacobian is bit-for-bit identical
    /// either way. Only exists when the crate is built with the `parallel`
    /// feature.
    ///
    /// # Panics
    ///
    /// Panics when `threshold == 0` — see `ParallelThreshold::set`.
    #[cfg(feature = "parallel")]
    pub fn with_parallel_threshold(self, threshold: usize) -> Self {
        self.jacobian_dispatch.set(threshold);
        self
    }

    /// Reconfigures the same threshold at runtime, through `&self` — no
    /// rebuild required. Scoped to this solver instance. Only exists when
    /// the crate is built with the `parallel` feature.
    ///
    /// # Panics
    ///
    /// Panics when `threshold == 0` — see `ParallelThreshold::set`.
    #[cfg(feature = "parallel")]
    pub fn set_parallel_threshold(&self, threshold: usize) {
        self.jacobian_dispatch.set(threshold);
    }

    /// Returns the effective dispatch threshold of the Jacobian assembly —
    /// the override if one was set, otherwise an indicative default of 256
    /// unknowns, to be replaced by a value measured on the target workload
    /// (`oxiflow_parallel` benchmarks). Only exists when the crate is built
    /// with the `parallel` feature.
    #[cfg(feature = "parallel")]
    pub fn parallel_threshold(&self) -> usize {
        self.jacobian_dispatch.threshold()
    }

    /// Configures the sparse backend (DD-043), for the startup step and
    /// the BDF2 update alike. Has no effect on its own — the banded path
    /// also requires [`Self::with_jacobian_bandwidth`] and a system larger
    /// than [`Self::with_sparse_threshold`] (default 100) at solve time;
    /// otherwise the dense path is used unchanged.
    #[cfg(feature = "sparse")]
    pub fn with_sparse_solver(mut self, sparse_solver: Box<dyn SparseLinearSolver>) -> Self {
        self.backend.sparse_solver = Some(sparse_solver);
        self
    }

    /// Number of unknowns (nodes × components) above which the banded path
    /// is used (default 100) — see [`Self::with_sparse_solver`].
    #[cfg(feature = "sparse")]
    pub fn with_sparse_threshold(mut self, sparse_threshold: usize) -> Self {
        self.backend.sparse_threshold = sparse_threshold;
        self
    }

    /// Declares the Jacobian's half-bandwidth (e.g.
    /// `scheme.stencil_radius()`, DD-039) — see [`Self::with_sparse_solver`].
    ///
    /// Counted in flat unknowns. For a `ScalarField` state this is the
    /// stencil half-width `s`; for a `VectorField` state of `d` components,
    /// whose unknowns are interleaved by node, it is `d·(s + 1) − 1` (#192).
    /// A band narrower than the Jacobian's leaves the Newton corrections
    /// inexact: the step reports [`OxiflowError::NewtonNotConverged`]
    /// rather than drifting from the dense path.
    #[cfg(feature = "sparse")]
    pub fn with_jacobian_bandwidth(mut self, jacobian_bandwidth: usize) -> Self {
        self.backend.jacobian_bandwidth = Some(jacobian_bandwidth);
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
        let formula = match history.first() {
            // Startup (#44 acceptance criterion): no u^{n-1} yet -- bootstrap
            // with one Backward Euler step to produce u^1 from u^0, under
            // the solver's own Newton configuration.
            None => ImplicitFormula::Theta(1.0),
            Some(previous) => ImplicitFormula::Bdf2 { previous },
        };
        self.backend.step(
            domain,
            chain,
            state,
            t,
            dt,
            formula,
            &self.newton,
            &self.jacobian_dispatch,
        )
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
    use crate::model::{CubicDecay, ExponentialDecay, ZeroDerivative};
    use crate::solver::config::{
        IntegratorKind, SolverConfiguration, StepControl, TimeConfiguration,
    };
    use crate::solver::linear::CountingLinearSolver;
    #[cfg(feature = "sparse")]
    use crate::solver::sparse::CountingSparseSolver;
    use nalgebra::DVector;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;

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
        let scenario = Scenario::single(Box::new(ZeroDerivative { initial: 2.5 }), make_mesh(5));
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
        let scenario = Scenario::single(Box::new(ZeroDerivative { initial: 2.5 }), make_mesh(3));
        let config = make_config(0.5, 0.1);
        let result = BDF2Solver::new().solve(&scenario, &config).unwrap();
        assert_eq!(result.states.len(), result.times.len());
        assert!((result.times[0] - 0.0).abs() < 1e-12);
        assert!((result.t_final().unwrap() - 0.5).abs() < 1e-9);
    }

    #[test]
    fn n_steps_is_correct() {
        let scenario = Scenario::single(Box::new(ZeroDerivative { initial: 2.5 }), make_mesh(2));
        let config = make_config(1.0, 0.25);
        let result = BDF2Solver::new().solve(&scenario, &config).unwrap();
        assert_eq!(result.n_steps, 4);
    }

    #[test]
    fn save_every_reduces_stored_states() {
        let scenario = Scenario::single(Box::new(ZeroDerivative { initial: 2.5 }), make_mesh(2));
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
        let scenario = Scenario::single(Box::new(ZeroDerivative { initial: 2.5 }), make_mesh(2));
        let config = make_config(1.0, -0.1);
        assert!(BDF2Solver::new().solve(&scenario, &config).is_err());
    }

    #[test]
    fn t_end_before_t_start_returns_error() {
        let scenario = Scenario::single(Box::new(ZeroDerivative { initial: 2.5 }), make_mesh(2))
            .with_t_start(5.0);
        let config = make_config(1.0, 0.1);
        assert!(BDF2Solver::new().solve(&scenario, &config).is_err());
    }

    #[test]
    fn adaptive_step_control_returns_error() {
        let scenario = Scenario::single(Box::new(ZeroDerivative { initial: 2.5 }), make_mesh(2));
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

        let err = BDF2Solver::new()
            .step(domain, &chain, &mut state, &[wrong_length_prev], 0.0, 0.1)
            .unwrap_err();
        assert!(
            matches!(&err, OxiflowError::InvalidDomain(message) if message.starts_with("history state:")),
            "got {err:?}"
        );
    }

    // ── Iterated Newton (DD-044, #112) ─────────────────────────────────────────

    /// Backward Euler root of `u + dt*u^3 = u_prev`, solved by scalar
    /// Newton independently of the crate's loop.
    fn backward_euler_cubic_decay(u_prev: f64, dt: f64) -> f64 {
        let mut u = u_prev;
        for _ in 0..100 {
            let g = u + dt * u.powi(3) - u_prev;
            let dg = 1.0 + 3.0 * dt * u.powi(2);
            u -= g / dg;
        }
        u
    }

    /// BDF2 root of `1.5*u + dt*u^3 = 2*u_n - 0.5*u_prev`, solved by scalar
    /// Newton independently of the crate's loop.
    fn bdf2_cubic_decay(u_n: f64, u_prev: f64, dt: f64) -> f64 {
        let rhs = 2.0 * u_n - 0.5 * u_prev;
        let mut u = u_n;
        for _ in 0..100 {
            let g = 1.5 * u + dt * u.powi(3) - rhs;
            let dg = 1.5 + 3.0 * dt * u.powi(2);
            u -= g / dg;
        }
        u
    }

    #[test]
    fn full_newton_resolves_nonaffine_correction_beyond_frozen_jacobian() {
        // f(u) = -u^3 -- genuinely nonlinear. `dt` is small enough for the
        // default single correction to pass its tolerance at both steps,
        // only approximately; FullNewton reaches the true discrete solution
        // at both steps, the bootstrap step included (J7 exit criterion,
        // #112).
        let dt = 0.01;
        let config = make_config(2.0 * dt, dt); // exactly two steps

        let scenario_default = Scenario::single(Box::new(CubicDecay), make_mesh(2));
        let default_result = BDF2Solver::new().solve(&scenario_default, &config).unwrap();

        let scenario_full = Scenario::single(Box::new(CubicDecay), make_mesh(2));
        let full_result = BDF2Solver::new()
            .with_newton_convergence(NewtonConvergence::default())
            .with_jacobian_strategy(JacobianStrategy::FullNewton)
            .with_max_newton_iterations(50)
            .solve(&scenario_full, &config)
            .unwrap();

        // u^1 comes from the Backward Euler bootstrap step, run with the
        // solver's configuration.
        let u0 = 1.0_f64;
        let u1_reference = backward_euler_cubic_decay(u0, dt);
        let u1_full = full_result.states[1].as_scalar_field().unwrap()[0];
        assert!(
            (u1_full - u1_reference).abs() < 1e-12,
            "FullNewton should converge the bootstrap step: got {u1_full}, expected {u1_reference}"
        );

        // Independently solved BDF2 equation for the real second step:
        // (3/2)*u2 - 2*u1 + (1/2)*u0 = dt*(-u2^3), i.e.
        // 1.5*u2 + dt*u2^3 = 2*u1 - 0.5*u0, using the *actual* u1 of the
        // FullNewton run, so this reference checks the second step alone.
        let u2_scalar = bdf2_cubic_decay(u1_full, u0, dt);

        let full_value = full_result.states[2].as_scalar_field().unwrap()[0];
        let default_value = default_result.states[2].as_scalar_field().unwrap()[0];

        assert!(
            (full_value - u2_scalar).abs() < 1e-6,
            "FullNewton should reach the converged discrete solution: got {full_value}, expected {u2_scalar}"
        );
        assert!(
            (default_value - full_value).abs() > 1e-7,
            "default single correction should measurably differ from the converged solution: \
             default={default_value}, full={full_value}"
        );
    }

    #[test]
    fn bootstrap_step_uses_the_configured_newton_parameters() {
        // At dt = 0.5, the Backward Euler bootstrap step of f(u) = -u^3
        // needs eight frozen-Jacobian corrections to pass the default
        // tolerance: every budget below that must fail and report the
        // configured budget, not a fixed single iteration.
        let dt = 0.5;
        let config = make_config(dt, dt); // the bootstrap step alone

        for budget in [1, 2] {
            let scenario = Scenario::single(Box::new(CubicDecay), make_mesh(2));
            let err = BDF2Solver::new()
                .with_max_newton_iterations(budget)
                .solve(&scenario, &config)
                .unwrap_err();
            assert!(
                matches!(err, OxiflowError::NewtonNotConverged { iterations, .. } if iterations == budget),
                "budget {budget}: expected NewtonNotConverged with {budget} iterations, got {err:?}"
            );
        }

        // A tolerance far below the default one: the bootstrap step only
        // reaches the reference if it honours the configured criterion
        // and strategy as well as the budget.
        let scenario = Scenario::single(Box::new(CubicDecay), make_mesh(2));
        let result = BDF2Solver::new()
            .with_newton_convergence(NewtonConvergence::ResidualOnly {
                tol_abs: 1e-13,
                tol_rel: 0.0,
            })
            .with_jacobian_strategy(JacobianStrategy::FullNewton)
            .with_max_newton_iterations(10)
            .solve(&scenario, &config)
            .unwrap();
        let u1 = result.states[1].as_scalar_field().unwrap()[0];
        let u1_reference = backward_euler_cubic_decay(1.0, dt);
        assert!(
            (u1 - u1_reference).abs() < 1e-12,
            "bootstrap step: got {u1}, expected {u1_reference}"
        );
    }

    // ── Banded path (DD-043, #133) ───────────────────────────────────────────

    /// Takes the banded path on every system: half-bandwidth 0 (the
    /// Jacobians of these scalar models are diagonal), threshold 0, and a
    /// sparse solver counting its calls in `sparse_calls`.
    #[cfg(feature = "sparse")]
    fn banded(solver: BDF2Solver, sparse_calls: &Arc<AtomicUsize>) -> BDF2Solver {
        solver
            .with_jacobian_bandwidth(0)
            .with_sparse_threshold(0)
            .with_sparse_solver(Box::new(CountingSparseSolver {
                calls: Arc::clone(sparse_calls),
            }))
    }

    #[cfg(feature = "sparse")]
    #[test]
    fn banded_path_runs_the_bootstrap_step_and_the_bdf2_update() {
        // Three steps, one correction each: the bootstrap step and both
        // BDF2 updates solve the banded system, never the dense one, and
        // give the results of the dense path.
        let dt = 0.1;
        let config = make_config(3.0 * dt, dt);
        let model = || Box::new(ExponentialDecay { lambda: 3.0 });

        let dense_calls = Arc::new(AtomicUsize::new(0));
        let sparse_calls = Arc::new(AtomicUsize::new(0));
        let solver = banded(
            BDF2Solver::new().with_linear_solver(Box::new(CountingLinearSolver {
                calls: Arc::clone(&dense_calls),
            })),
            &sparse_calls,
        );
        let banded_result = solver
            .solve(&Scenario::single(model(), make_mesh(4)), &config)
            .unwrap();
        assert_eq!(sparse_calls.load(Ordering::SeqCst), 3);
        assert_eq!(dense_calls.load(Ordering::SeqCst), 0);

        let dense_result = BDF2Solver::new()
            .solve(&Scenario::single(model(), make_mesh(4)), &config)
            .unwrap();
        // The systems are diagonal: both factorisations reduce to one
        // division per unknown, and the states agree bit for bit (measured).
        assert_eq!(banded_result.states, dense_result.states);
    }

    #[cfg(feature = "sparse")]
    #[test]
    fn banded_path_iterates_newton_to_the_root() {
        // f(u) = -u^3 at dt = 0.5, full Newton under a tolerance far below
        // the default one: the bootstrap step and the BDF2 update both
        // reach their roots on the banded path.
        let dt = 0.5;
        let sparse_calls = Arc::new(AtomicUsize::new(0));
        let solver = banded(
            BDF2Solver::new()
                .with_newton_convergence(NewtonConvergence::ResidualOnly {
                    tol_abs: 1e-13,
                    tol_rel: 0.0,
                })
                .with_jacobian_strategy(JacobianStrategy::FullNewton)
                .with_max_newton_iterations(10),
            &sparse_calls,
        );
        let result = solver
            .solve(
                &Scenario::single(Box::new(CubicDecay), make_mesh(2)),
                &make_config(2.0 * dt, dt),
            )
            .unwrap();

        let u1 = result.states[1].as_scalar_field().unwrap()[0];
        let u1_reference = backward_euler_cubic_decay(1.0, dt);
        assert!(
            (u1 - u1_reference).abs() < 1e-12,
            "bootstrap step: got {u1}, expected {u1_reference}"
        );
        let u2 = result.states[2].as_scalar_field().unwrap()[0];
        let u2_reference = bdf2_cubic_decay(u1, 1.0, dt);
        assert!(
            (u2 - u2_reference).abs() < 1e-12,
            "BDF2 update: got {u2}, expected {u2_reference}"
        );
        // One correction per step would make two banded solves; the loop
        // iterated on the banded system.
        assert!(sparse_calls.load(Ordering::SeqCst) > 2);
    }

    #[cfg(feature = "sparse")]
    #[test]
    fn banded_path_reports_the_configured_newton_budget() {
        // At dt = 0.5, two frozen-Jacobian corrections of f(u) = -u^3 do not
        // reach the default tolerance, neither at the bootstrap step nor
        // on a BDF2 update: both report the configured budget.
        let dt = 0.5;
        let sparse_calls = Arc::new(AtomicUsize::new(0));
        let solver = banded(
            BDF2Solver::new().with_max_newton_iterations(2),
            &sparse_calls,
        );
        let scenario = Scenario::single(Box::new(CubicDecay), make_mesh(2));

        let err = solver.solve(&scenario, &make_config(dt, dt)).unwrap_err();
        assert!(
            matches!(err, OxiflowError::NewtonNotConverged { iterations: 2, .. }),
            "bootstrap step: got {err:?}"
        );

        let domain = scenario.single_domain().unwrap();
        let chain =
            crate::solver::chain::build_calculator_chain(&scenario.context_requirements(), &[])
                .unwrap();
        let mut state = domain.model.initial_state(domain.mesh.as_ref());
        let previous = state.clone();
        let err = solver
            .step(domain, &chain, &mut state, &[previous], 0.0, dt)
            .unwrap_err();
        assert!(
            matches!(err, OxiflowError::NewtonNotConverged { iterations: 2, .. }),
            "BDF2 update: got {err:?}"
        );
    }

    #[cfg(feature = "sparse")]
    #[test]
    fn systems_up_to_the_sparse_threshold_stay_dense() {
        // 4 unknowns, threshold 4: the banded path needs strictly more
        // unknowns than the threshold.
        let dense_calls = Arc::new(AtomicUsize::new(0));
        let sparse_calls = Arc::new(AtomicUsize::new(0));
        let solver = banded(
            BDF2Solver::new().with_linear_solver(Box::new(CountingLinearSolver {
                calls: Arc::clone(&dense_calls),
            })),
            &sparse_calls,
        )
        .with_sparse_threshold(4);
        solver
            .solve(
                &Scenario::single(Box::new(ExponentialDecay { lambda: 3.0 }), make_mesh(4)),
                &make_config(0.2, 0.1),
            )
            .unwrap();
        assert_eq!(dense_calls.load(Ordering::SeqCst), 2);
        assert_eq!(sparse_calls.load(Ordering::SeqCst), 0);
    }
}
