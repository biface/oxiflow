//! # Module `solver::methods::backward_euler`
//!
//! Backward Euler integrator — implicit, 1st order (issue #43).
//!
//! ## Algorithm
//!
//! $$u^{n+1} = u^n + \Delta t \cdot f(u^{n+1}, t^{n+1})$$
//!
//! A thin wrapper around the shared generalised theta method
//! ([`super::implicit::theta_method_step`], θ=1) — see that module's docs
//! for the frozen-Jacobian v1 scope and the path to a future iterated
//! Newton solver (DD-033).
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
//! - See [`super::implicit`] for the boundary-condition interaction
//!   caveat — not yet covered by a dedicated test.

use crate::context::error::OxiflowError;
use crate::context::value::ContextValue;
use crate::context::ContextCalculator;
use crate::solver::chain::build_calculator_chain;
use crate::solver::config::StepControl;
use crate::solver::linear::{LinearSolver, NalgebraDenseSolver};
use crate::solver::methods::implicit::theta_method_step;
use crate::solver::methods::{check_finite, SteppableSolver};
use crate::solver::scenario::{Domain, Scenario};
use crate::solver::{SimulationResult, Solver, SolverConfiguration};

use std::collections::HashMap;

/// Backward Euler solver — implicit, 1st order.
///
/// # Examples
///
/// ```rust,ignore
/// use oxiflow::solver::methods::backward_euler::BackwardEulerSolver;
///
/// let solver = BackwardEulerSolver::new();
/// // Or, once a sparse backend lands (v0.5.0, DD-013):
/// // let solver = BackwardEulerSolver::new().with_linear_solver(Box::new(FaerSparseSolver));
/// ```
pub struct BackwardEulerSolver {
    linear_solver: Box<dyn LinearSolver>,
}

impl Default for BackwardEulerSolver {
    fn default() -> Self {
        Self {
            linear_solver: Box::new(NalgebraDenseSolver),
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
}

impl Solver for BackwardEulerSolver {
    fn solve(
        &self,
        scenario: &Scenario,
        config: &SolverConfiguration,
    ) -> Result<SimulationResult, OxiflowError> {
        scenario.validate()?;
        let domain = scenario.single_domain()?;

        let dt = match &config.time.step_control {
            StepControl::Fixed { dt } => *dt,
            _ => {
                return Err(OxiflowError::InvalidDomain(
                    "BackwardEulerSolver only supports StepControl::Fixed at J4a".into(),
                ))
            }
        };

        let t_end = config.time.t_end;
        let t_start = scenario.t_start;

        if dt <= 0.0 {
            return Err(OxiflowError::InvalidDomain(
                "dt must be strictly positive".into(),
            ));
        }
        if t_end <= t_start {
            return Err(OxiflowError::InvalidDomain(
                "t_end must be greater than t_start".into(),
            ));
        }

        let requirements = scenario.context_requirements();
        let chain = build_calculator_chain(&requirements, &config.calculators)?;

        let mut u = domain.model.initial_state(domain.mesh.as_ref());

        let n_steps = ((t_end - t_start) / dt).round() as usize;
        let save_every = config.time.save_every.unwrap_or(1);
        let capacity = n_steps / save_every + 1;
        let mut states: Vec<ContextValue> = Vec::with_capacity(capacity);
        let mut times: Vec<f64> = Vec::with_capacity(capacity);

        states.push(u.clone());
        times.push(t_start);

        for step in 0..n_steps {
            let t = t_start + (step as f64) * dt;
            let t_next = t_start + ((step + 1) as f64) * dt;

            u = self.step(domain, &chain, &mut u, t, dt)?;

            check_finite(&u, t_next)?;

            if (step + 1) % save_every == 0 {
                states.push(u.clone());
                times.push(t_next);
            }
        }

        Ok(SimulationResult {
            states,
            times,
            n_steps,
            metadata: HashMap::new(),
        })
    }
}

impl SteppableSolver for BackwardEulerSolver {
    fn step(
        &self,
        domain: &Domain,
        chain: &[&dyn ContextCalculator],
        state: &mut ContextValue,
        t: f64,
        dt: f64,
    ) -> Result<ContextValue, OxiflowError> {
        theta_method_step(
            domain,
            chain,
            state,
            t,
            dt,
            1.0,
            self.linear_solver.as_ref(),
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
    use crate::solver::config::{IntegratorKind, SolverConfiguration, TimeConfiguration};
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
        let next = solver.step(domain, &chain, &mut u, 0.0, 0.1).unwrap();
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
}
