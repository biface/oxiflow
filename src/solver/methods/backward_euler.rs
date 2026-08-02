//! # Module `solver::methods::backward_euler`
//!
//! Backward Euler integrator — implicit, 1st order (issue #43).
//!
//! ## Algorithm
//!
//! $$u^{n+1} = u^n + \Delta t \cdot f(u^{n+1}, t^{n+1})$$
//!
//! A thin wrapper around the shared generalised theta method
//! (`theta_method_step_newton`, θ=1, DD-044, [#111](https://github.com/biface/oxiflow/issues/111)) —
//! see that module's docs for the frozen-Jacobian default and the
//! iterated Newton path configurable via [`BackwardEulerSolver`]'s
//! builders.
//!
//! ## Stability
//!
//! Unconditionally A-stable for `f` affine in `u` — no CFL-style
//! restriction on `dt`, unlike the explicit methods. This is the whole
//! point: stiff problems (`λΔt ≫ 1`) that would blow up under
//! `ForwardEulerSolver` remain bounded here.
//!
//! ## Scope at J4a
//!
//! - Single-domain scenarios only — same restriction as the explicit
//!   solvers; see #40 for the dedicated multi-domain path.
//! - `StepControl::Fixed { dt }` only.
//! - See [`super::implicit`] for the boundary-condition interaction with
//!   the perturbed Jacobian — covered by a dedicated test since #113.

use crate::context::error::OxiflowError;
use crate::context::value::ContextValue;
use crate::context::ContextCalculator;
use crate::solver::linear::{LinearSolver, NalgebraDenseSolver};
#[cfg(feature = "sparse")]
use crate::solver::methods::implicit::theta_method_step_adaptive;
use crate::solver::methods::implicit::{stiff_jacobian_convergence, theta_method_step_newton};
use crate::solver::methods::newton::{JacobianStrategy, NewtonConvergence};
use crate::solver::methods::SteppableSolver;
use crate::solver::scenario::{Domain, Scenario};
#[cfg(feature = "sparse")]
use crate::solver::sparse::SparseLinearSolver;
use crate::solver::{SimulationResult, Solver, SolverConfiguration};

/// Backward Euler solver — implicit, 1st order.
///
/// # Examples
///
/// ```rust,ignore
/// use oxiflow::solver::methods::backward_euler::BackwardEulerSolver;
///
/// let solver = BackwardEulerSolver::new();
/// // Sparse backend (v0.6.0, DD-013 second phase, DD-043) — dense path
/// // stays untouched unless all three are configured explicitly:
/// // let solver = BackwardEulerSolver::new()
/// //     .with_jacobian_bandwidth(scheme.stencil_radius())
/// //     .with_sparse_solver(Box::new(FaerSparseSolver))
/// //     .with_sparse_threshold(100); // default
/// ```
pub struct BackwardEulerSolver {
    linear_solver: Box<dyn LinearSolver>,
    /// Newton convergence criterion (DD-044, [#111](https://github.com/biface/oxiflow/issues/111)) — default
    /// [`stiff_jacobian_convergence`] (deliberately looser than
    /// [`NewtonConvergence::default`] — see that function's docs for why
    /// zero-config construction needs it).
    newton_convergence: NewtonConvergence,
    /// Jacobian refresh strategy (DD-044, [#111](https://github.com/biface/oxiflow/issues/111)) — default
    /// [`JacobianStrategy::default`] (`ModifiedFrozen`).
    jacobian_strategy: JacobianStrategy,
    /// Newton iteration budget (DD-044, [#111](https://github.com/biface/oxiflow/issues/111)). Default `1`: exactly the
    /// pre-DD-044 single frozen-Jacobian correction — see
    /// [`super::implicit`]'s module docs for why this reproduces the
    /// historical behaviour exactly, not approximately, on affine
    /// problems. Raise this (and typically pair with
    /// [`JacobianStrategy::FullNewton`]) for genuinely nonlinear models.
    max_newton_iterations: usize,
    /// Sparse backend (DD-043) — `None` by default: the dense path
    /// ([`theta_method_step_newton`]) is unconditionally used unless this is
    /// explicitly configured via [`Self::with_sparse_solver`]. Not yet
    /// Newton-aware when configured — see [`super::implicit`]'s module
    /// docs for the known gap.
    #[cfg(feature = "sparse")]
    sparse_solver: Option<Box<dyn SparseLinearSolver>>,
    /// System size above which the sparse path is used, once a
    /// `sparse_solver` and a `jacobian_bandwidth` are both configured.
    /// Ignored (no effect) if `sparse_solver` is `None`.
    #[cfg(feature = "sparse")]
    sparse_threshold: usize,
    /// Half-bandwidth of the Jacobian — a HOW-side assertion supplied by
    /// the caller (e.g. `scheme.stencil_radius()`, DD-039), never read
    /// from `Domain`/`PhysicalModel` (DD-043, amendment 1). `None` by
    /// default: no assumption about locality, dense path only.
    #[cfg(feature = "sparse")]
    jacobian_bandwidth: Option<usize>,
}

impl Default for BackwardEulerSolver {
    fn default() -> Self {
        Self {
            linear_solver: Box::new(NalgebraDenseSolver),
            newton_convergence: stiff_jacobian_convergence(),
            jacobian_strategy: JacobianStrategy::default(),
            max_newton_iterations: 1,
            #[cfg(feature = "sparse")]
            sparse_solver: None,
            #[cfg(feature = "sparse")]
            sparse_threshold: 100,
            #[cfg(feature = "sparse")]
            jacobian_bandwidth: None,
        }
    }
}

impl BackwardEulerSolver {
    /// Creates a solver using the default `nalgebra` dense backend.
    pub fn new() -> Self {
        Self::default()
    }

    /// Substitutes the linear solver backend (DD-013).
    pub fn with_linear_solver(mut self, linear_solver: Box<dyn LinearSolver>) -> Self {
        self.linear_solver = linear_solver;
        self
    }

    /// Configures the Newton convergence criterion (DD-044, [#111](https://github.com/biface/oxiflow/issues/111)).
    pub fn with_newton_convergence(mut self, newton_convergence: NewtonConvergence) -> Self {
        self.newton_convergence = newton_convergence;
        self
    }

    /// Configures the Jacobian refresh strategy (DD-044, [#111](https://github.com/biface/oxiflow/issues/111)).
    pub fn with_jacobian_strategy(mut self, jacobian_strategy: JacobianStrategy) -> Self {
        self.jacobian_strategy = jacobian_strategy;
        self
    }

    /// Configures the Newton iteration budget (DD-044, [#111](https://github.com/biface/oxiflow/issues/111)) — see the
    /// field's own docs for why the default (`1`) reproduces the
    /// pre-DD-044 behaviour exactly on affine problems.
    pub fn with_max_newton_iterations(mut self, max_newton_iterations: usize) -> Self {
        self.max_newton_iterations = max_newton_iterations;
        self
    }

    /// Configures the sparse backend (DD-043). Has no effect on its own —
    /// the sparse path also requires [`Self::with_jacobian_bandwidth`] and
    /// a system larger than [`Self::with_sparse_threshold`] (default 100)
    /// at solve time; otherwise the dense path is used unchanged.
    #[cfg(feature = "sparse")]
    pub fn with_sparse_solver(mut self, sparse_solver: Box<dyn SparseLinearSolver>) -> Self {
        self.sparse_solver = Some(sparse_solver);
        self
    }

    /// System size above which the sparse path is used (default 100) —
    /// see [`Self::with_sparse_solver`].
    #[cfg(feature = "sparse")]
    pub fn with_sparse_threshold(mut self, sparse_threshold: usize) -> Self {
        self.sparse_threshold = sparse_threshold;
        self
    }

    /// Declares the Jacobian's half-bandwidth (e.g.
    /// `scheme.stencil_radius()`, DD-039) — see [`Self::with_sparse_solver`].
    #[cfg(feature = "sparse")]
    pub fn with_jacobian_bandwidth(mut self, jacobian_bandwidth: usize) -> Self {
        self.jacobian_bandwidth = Some(jacobian_bandwidth);
        self
    }
}

impl Solver for BackwardEulerSolver {
    fn solve(
        &self,
        scenario: &Scenario,
        config: &SolverConfiguration,
    ) -> Result<SimulationResult, OxiflowError> {
        self.solve_fixed_step(scenario, config)
    }
}

#[cfg(feature = "sparse")]
impl SteppableSolver for BackwardEulerSolver {
    fn step(
        &self,
        domain: &Domain,
        chain: &[&dyn ContextCalculator],
        state: &mut ContextValue,
        _history: &[ContextValue],
        t: f64,
        dt: f64,
    ) -> Result<ContextValue, OxiflowError> {
        // history_depth() defaults to 0 -- Backward Euler is a one-step
        // method, `_history` is always empty here and intentionally unused.
        //
        // Dispatch is gated on `sparse_solver` alone: with no solver
        // configured there is nothing to hand `theta_method_step_adaptive`,
        // so the dense path is taken directly rather than going through it
        // with a `None` bandwidth (equivalent result, one fewer branch to
        // reason about). The `n > sparse_threshold && bandwidth.is_some()`
        // condition is `theta_method_step_adaptive`'s own responsibility.
        match &self.sparse_solver {
            Some(sparse_solver) => theta_method_step_adaptive(
                domain,
                chain,
                state,
                t,
                dt,
                1.0,
                self.linear_solver.as_ref(),
                sparse_solver.as_ref(),
                self.sparse_threshold,
                self.jacobian_bandwidth,
            ),
            None => theta_method_step_newton(
                domain,
                chain,
                state,
                t,
                dt,
                1.0,
                self.linear_solver.as_ref(),
                self.newton_convergence,
                self.jacobian_strategy,
                self.max_newton_iterations,
            ),
        }
    }
}

#[cfg(not(feature = "sparse"))]
impl SteppableSolver for BackwardEulerSolver {
    fn step(
        &self,
        domain: &Domain,
        chain: &[&dyn ContextCalculator],
        state: &mut ContextValue,
        _history: &[ContextValue],
        t: f64,
        dt: f64,
    ) -> Result<ContextValue, OxiflowError> {
        // history_depth() defaults to 0 -- Backward Euler is a one-step
        // method, `_history` is always empty here and intentionally unused.
        theta_method_step_newton(
            domain,
            chain,
            state,
            t,
            dt,
            1.0,
            self.linear_solver.as_ref(),
            self.newton_convergence,
            self.jacobian_strategy,
            self.max_newton_iterations,
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
    use crate::solver::config::{
        IntegratorKind, SolverConfiguration, StepControl, TimeConfiguration,
    };
    use nalgebra::{DMatrix, DVector};
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

    /// Delegates to `NalgebraDenseSolver` but counts calls.
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
            IntegratorKind::BackwardEuler,
        )
    }

    fn make_mesh(n: usize) -> Box<dyn Mesh> {
        Box::new(UniformGrid1D::new(n, 0.0, 1.0).unwrap())
    }

    #[test]
    fn zero_derivative_field_stays_constant() {
        let scenario = Scenario::single(Box::new(ZeroDerivative), make_mesh(5));
        let config = make_config(1.0, 0.1);
        let result = BackwardEulerSolver::new()
            .solve(&scenario, &config)
            .unwrap();
        for state in &result.states {
            let field = state.as_scalar_field().unwrap();
            for v in field.iter() {
                assert!((v - 2.5).abs() < 1e-12);
            }
        }
    }

    #[test]
    fn exponential_decay_matches_analytical_over_many_steps() {
        let lambda = 2.0;
        let dt = 0.1;
        let n_steps = 20;
        let scenario = Scenario::single(Box::new(ExponentialDecay { lambda }), make_mesh(2));
        let config = make_config(n_steps as f64 * dt, dt);
        let result = BackwardEulerSolver::new()
            .solve(&scenario, &config)
            .unwrap();

        let expected = 1.0 / (1.0 + lambda * dt).powi(n_steps);
        let final_field = result.states.last().unwrap().as_scalar_field().unwrap();
        for v in final_field.iter() {
            assert!((v - expected).abs() < 1e-9, "got {v}, expected {expected}");
        }
    }

    #[test]
    fn result_times_match_expected_steps() {
        let scenario = Scenario::single(Box::new(ZeroDerivative), make_mesh(3));
        let config = make_config(0.5, 0.1);
        let result = BackwardEulerSolver::new()
            .solve(&scenario, &config)
            .unwrap();
        assert_eq!(result.states.len(), result.times.len());
        assert!((result.times[0] - 0.0).abs() < 1e-12);
        assert!((result.t_final().unwrap() - 0.5).abs() < 1e-9);
    }

    #[test]
    fn n_steps_is_correct() {
        let scenario = Scenario::single(Box::new(ZeroDerivative), make_mesh(2));
        let config = make_config(1.0, 0.25);
        let result = BackwardEulerSolver::new()
            .solve(&scenario, &config)
            .unwrap();
        assert_eq!(result.n_steps, 4);
    }

    #[test]
    fn save_every_reduces_stored_states() {
        let scenario = Scenario::single(Box::new(ZeroDerivative), make_mesh(2));
        let config = SolverConfiguration::new(
            TimeConfiguration::new(1.0, StepControl::Fixed { dt: 0.1 }).saving_every(5),
            IntegratorKind::BackwardEuler,
        );
        let result = BackwardEulerSolver::new()
            .solve(&scenario, &config)
            .unwrap();
        assert_eq!(result.states.len(), 3);
    }

    #[test]
    fn step_matches_one_iteration_of_solve() {
        let scenario = Scenario::single(Box::new(ExponentialDecay { lambda: 0.7 }), make_mesh(3));
        let config = make_config(0.1, 0.1);

        let solver = BackwardEulerSolver::new();
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

    #[test]
    fn stable_for_very_stiff_problem_over_many_steps() {
        let lambda = 1.0e4;
        let dt = 0.1;
        let scenario = Scenario::single(Box::new(ExponentialDecay { lambda }), make_mesh(2));
        let config = make_config(2.0, dt);
        let result = BackwardEulerSolver::new()
            .solve(&scenario, &config)
            .unwrap();

        for state in &result.states {
            let field = state.as_scalar_field().unwrap();
            for v in field.iter() {
                assert!(v.is_finite(), "value diverged: {v}");
                assert!(v.abs() <= 1.0, "expected monotonic damping, got {v}");
            }
        }
    }

    #[test]
    fn with_linear_solver_substitutes_backend() {
        let calls = Arc::new(AtomicUsize::new(0));
        let solver =
            BackwardEulerSolver::new().with_linear_solver(Box::new(CountingLinearSolver {
                calls: calls.clone(),
            }));

        let scenario = Scenario::single(Box::new(ExponentialDecay { lambda: 1.0 }), make_mesh(2));
        let config = make_config(0.5, 0.1);

        solver.solve(&scenario, &config).unwrap();

        assert_eq!(calls.load(Ordering::SeqCst), 5);
    }

    #[test]
    fn negative_dt_returns_error() {
        let scenario = Scenario::single(Box::new(ZeroDerivative), make_mesh(2));
        let config = make_config(1.0, -0.1);
        assert!(BackwardEulerSolver::new()
            .solve(&scenario, &config)
            .is_err());
    }

    #[test]
    fn t_end_before_t_start_returns_error() {
        let scenario = Scenario::single(Box::new(ZeroDerivative), make_mesh(2)).with_t_start(5.0);
        let config = make_config(1.0, 0.1);
        assert!(BackwardEulerSolver::new()
            .solve(&scenario, &config)
            .is_err());
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
        let err = BackwardEulerSolver::new()
            .solve(&scenario, &config)
            .unwrap_err();
        assert!(matches!(err, OxiflowError::MissingCalculator(_)));
    }

    // ── Iterated Newton (DD-044, #111) ─────────────────────────────────────────

    #[derive(Debug)]
    struct CubicDecay;

    impl RequiresContext for CubicDecay {
        fn required_variables(&self) -> Vec<ContextVariable> {
            vec![]
        }
    }

    impl PhysicalModel for CubicDecay {
        fn compute_physics(
            &self,
            state: &ContextValue,
            _ctx: &ComputeContext,
        ) -> Result<ContextValue, OxiflowError> {
            let u = state.as_scalar_field()?;
            Ok(ContextValue::ScalarField(u.map(|v| -v * v * v)))
        }

        fn initial_state(&self, mesh: &dyn Mesh) -> ContextValue {
            ContextValue::ScalarField(DVector::from_element(mesh.n_dof(), 1.0))
        }

        fn name(&self) -> &str {
            "cubic_decay_test"
        }
    }

    #[test]
    fn full_newton_resolves_nonaffine_correction_beyond_frozen_jacobian() {
        // f(u) = -u^3 -- genuinely nonlinear. Backward Euler's own
        // discrete equation for one step is u_next + dt*u_next^3 = u_n;
        // the default single frozen correction (max_newton_iterations=1,
        // the pre-DD-044 behaviour) is only a first-order approximation
        // of that equation's solution, not the solution itself -- under
        // the default tolerance this surfaces as an honest convergence
        // failure rather than a silently inaccurate "success" (nothing
        // about affine-ness is special-cased, see
        // `solver::methods::newton`'s docs). Configuring FullNewton with
        // enough iterations reaches the converged solution -- the J7
        // exit criterion (#111).
        let dt = 0.5; // deliberately large so the gap is clearly visible
        let config = make_config(dt, dt); // exactly one step

        let scenario_default = Scenario::single(Box::new(CubicDecay), make_mesh(2));
        let default_err = BackwardEulerSolver::new()
            .solve(&scenario_default, &config)
            .unwrap_err();
        assert!(
            matches!(default_err, OxiflowError::NewtonNotConverged { .. }),
            "default single correction should fail to converge on a genuinely \
             nonlinear model: got {default_err:?}"
        );

        let scenario_full = Scenario::single(Box::new(CubicDecay), make_mesh(2));
        // Explicit `NewtonConvergence::default()` (tol_abs=1e-8): the
        // solver's zero-config value is `stiff_jacobian_convergence()`
        // (tol_abs=1e-5, tuned for backward compatibility on stiff
        // affine problems, see that function's docs) -- this test wants
        // the tighter, general-purpose criterion to demonstrate genuine
        // convergence, not the loose one.
        let full_result = BackwardEulerSolver::new()
            .with_newton_convergence(NewtonConvergence::default())
            .with_jacobian_strategy(JacobianStrategy::FullNewton)
            .with_max_newton_iterations(50)
            .solve(&scenario_full, &config)
            .unwrap();
        let full_value = full_result
            .states
            .last()
            .unwrap()
            .as_scalar_field()
            .unwrap()[0];

        // Independently solved scalar equation: u_next + dt*u_next^3 = 1.0.
        let mut u_scalar = 1.0_f64;
        for _ in 0..100 {
            let g = u_scalar + dt * u_scalar.powi(3) - 1.0;
            let dg = 1.0 + 3.0 * dt * u_scalar.powi(2);
            u_scalar -= g / dg;
        }

        assert!(
            (full_value - u_scalar).abs() < 1e-6,
            "FullNewton should reach the converged discrete solution: got {full_value}, expected {u_scalar}"
        );
    }
}
