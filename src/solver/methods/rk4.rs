//! # Module `solver::methods::rk4`
//!
//! Classical Runge-Kutta 4 integrator — explicit, 4th order (issue #41).
//!
//! ## Algorithm
//!
//! At each time step, four stage derivatives are evaluated:
//!
//! $$
//! \begin{aligned}
//! k_1 &= f(t,\ u) \\
//! k_2 &= f(t + \tfrac{\Delta t}{2},\ u + \tfrac{\Delta t}{2} k_1) \\
//! k_3 &= f(t + \tfrac{\Delta t}{2},\ u + \tfrac{\Delta t}{2} k_2) \\
//! k_4 &= f(t + \Delta t,\ u + \Delta t\, k_3)
//! \end{aligned}
//! $$
//!
//! combined into the step:
//!
//! $$u^{n+1} = u^n + \frac{\Delta t}{6}\left(k_1 + 2k_2 + 2k_3 + k_4\right)$$
//!
//! where $f = \text{compute\_physics}(u, \text{ctx})$, evaluated on a state
//! that has had boundary conditions enforced for *that* stage — see
//! [`super::evaluate_derivative`].
//!
//! ## Boundary conditions across stages
//!
//! Each of the four stages is a separate derivative evaluation, so each one
//! gets its own boundary-condition application via
//! [`super::apply_boundary_conditions`] — on `u` itself for stage 1, and on
//! the transient intermediate states (`u + scale * k_i`) for stages 2-4. A
//! Dirichlet value enforced only once per outer step would otherwise drift
//! across the intermediate evaluations.
//!
//! `t` passed to [`crate::context::compute::ComputeContext`] varies per
//! stage (`t`, `t + dt/2`, `t + dt/2`, `t + dt`); `dt` itself is the
//! *outer* step size for all four stages — it identifies the configured
//! step, not a per-stage time delta.
//!
//! ## Scope at J4a
//!
//! - Single-domain scenarios only (`n_domains() == 1`) — same restriction
//!   as `ForwardEulerSolver`. Multi-domain `CouplingOperator` scenarios are
//!   out of scope; see #40 for the dedicated multi-domain proto.
//! - No `DiscreteOperator` (INV-2) — spatial schemes arrive at J4b.
//! - `StepControl::Fixed { dt }` only — adaptive step control is DoPri45's
//!   job (reserved, J4).
//!
//! ## Stability
//!
//! Classical RK4 has a larger stability region than forward Euler along the
//! imaginary axis, but is still explicit: the CFL condition
//! $\text{CFL} = v\,\Delta t / \Delta x \leq 1$ still bounds the usable
//! step size. The solver does not enforce this automatically.

use nalgebra::DVector;

use crate::context::error::OxiflowError;
use crate::context::value::ContextValue;
use crate::context::ContextCalculator;
use crate::solver::methods::{evaluate_derivative, SteppableSolver};
use crate::solver::scenario::{Domain, Scenario};
use crate::solver::{SimulationResult, Solver, SolverConfiguration};

/// Classical Runge-Kutta 4 solver — explicit, 4th order.
///
/// Implements the `Solver` trait for single-domain problems with fixed step
/// control. See [module documentation](self) for algorithm details.
///
/// # Examples
///
/// ```rust,ignore
/// use oxiflow::solver::methods::rk4::RK4Solver;
/// use oxiflow::solver::{Scenario, SolverConfiguration, TimeConfiguration, StepControl, IntegratorKind};
///
/// let scenario = Scenario::single(Box::new(my_model), Box::new(mesh));
/// let config = SolverConfiguration::new(
///     TimeConfiguration::new(100.0, StepControl::Fixed { dt: 0.1 }),
///     IntegratorKind::RK4,
/// );
/// let result = RK4Solver.solve(&scenario, &config).unwrap();
/// ```
pub struct RK4Solver;

impl Solver for RK4Solver {
    fn solve(
        &self,
        scenario: &Scenario,
        config: &SolverConfiguration,
    ) -> Result<SimulationResult, OxiflowError> {
        self.solve_fixed_step(scenario, config)
    }
}

impl SteppableSolver for RK4Solver {
    fn step(
        &self,
        domain: &Domain,
        chain: &[&dyn ContextCalculator],
        state: &mut ContextValue,
        _history: &[ContextValue],
        t: f64,
        dt: f64,
    ) -> Result<ContextValue, OxiflowError> {
        // history_depth() defaults to 0 -- RK4 is a one-step method,
        // `_history` is always empty here and intentionally unused.
        let half_dt = dt / 2.0;

        // Stage 1: k1 = f(t, u). BCs are applied to `state` itself here —
        // it is the persisted solution state, same contract as Euler.
        let k1 = evaluate_derivative(domain, chain, state, t, dt)?;

        // Stage 2: k2 = f(t + dt/2, u + dt/2 * k1)
        let mut u2 = combine(state, &k1, half_dt)?;
        let k2 = evaluate_derivative(domain, chain, &mut u2, t + half_dt, dt)?;

        // Stage 3: k3 = f(t + dt/2, u + dt/2 * k2)
        let mut u3 = combine(state, &k2, half_dt)?;
        let k3 = evaluate_derivative(domain, chain, &mut u3, t + half_dt, dt)?;

        // Stage 4: k4 = f(t + dt, u + dt * k3)
        let mut u4 = combine(state, &k3, dt)?;
        let k4 = evaluate_derivative(domain, chain, &mut u4, t + dt, dt)?;

        // Weighted combination: u_next = u + dt/6 * (k1 + 2k2 + 2k3 + k4)
        rk4_combine(state, &k1, &k2, &k3, &k4, dt)
    }
}

/// Computes `u + scale * k` for `ScalarField` states.
///
/// Returns `OxiflowError::TypeMismatch` if either operand is not
/// `ScalarField`, or `InvalidDomain` if their lengths differ.
fn combine(u: &ContextValue, k: &ContextValue, scale: f64) -> Result<ContextValue, OxiflowError> {
    let u_field = u.as_scalar_field()?;
    let k_field = k.as_scalar_field()?;

    if u_field.len() != k_field.len() {
        return Err(OxiflowError::InvalidDomain(format!(
            "state length {} != stage derivative length {}",
            u_field.len(),
            k_field.len()
        )));
    }

    Ok(ContextValue::ScalarField(u_field + k_field * scale))
}

/// Computes the RK4 weighted combination `u + dt/6 * (k1 + 2k2 + 2k3 + k4)`.
///
/// Returns `OxiflowError::InvalidDomain` if any stage derivative's length
/// differs from `u`'s.
fn rk4_combine(
    u: &ContextValue,
    k1: &ContextValue,
    k2: &ContextValue,
    k3: &ContextValue,
    k4: &ContextValue,
    dt: f64,
) -> Result<ContextValue, OxiflowError> {
    let u_field = u.as_scalar_field()?;
    let k1_field = k1.as_scalar_field()?;
    let k2_field = k2.as_scalar_field()?;
    let k3_field = k3.as_scalar_field()?;
    let k4_field = k4.as_scalar_field()?;

    let n = u_field.len();
    if k1_field.len() != n || k2_field.len() != n || k3_field.len() != n || k4_field.len() != n {
        return Err(OxiflowError::InvalidDomain(
            "RK4 stage derivative length mismatch with state".into(),
        ));
    }

    // Use fully-owned DVector operands throughout: avoids relying on
    // nalgebra's reference/owned operator-overload resolution across a long
    // expression chain — every `+`/`*` here is owned-op-owned, the most
    // basic and unambiguous combination.
    let k1_owned = k1_field.clone();
    let k2_owned = k2_field.clone();
    let k3_owned = k3_field.clone();
    let k4_owned = k4_field.clone();

    let weighted: DVector<f64> = k1_owned + k2_owned * 2.0 + k3_owned * 2.0 + k4_owned;
    let result: DVector<f64> = u_field.clone() + weighted * (dt / 6.0);

    Ok(ContextValue::ScalarField(result))
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::context::compute::ComputeContext;
    use crate::context::error::OxiflowError;
    use crate::context::value::ContextValue;
    use crate::context::variable::ContextVariable;
    use crate::mesh::{Mesh, UniformGrid1D};
    use crate::model::traits::{PhysicalModel, RequiresContext};
    use crate::solver::config::{
        IntegratorKind, SolverConfiguration, StepControl, TimeConfiguration,
    };
    use nalgebra::DVector;

    // ── Fixtures ──────────────────────────────────────────────────────────────

    /// Pure exponential decay: du/dt = -lambda * u
    /// Analytical solution: u(t) = u0 * exp(-lambda * t)
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

    /// Constant zero derivative — field stays unchanged.
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
            Ok(ContextValue::ScalarField(DVector::from_element(
                u.len(),
                0.0,
            )))
        }

        fn initial_state(&self, mesh: &dyn Mesh) -> ContextValue {
            ContextValue::ScalarField(DVector::from_element(mesh.n_dof(), 2.5))
        }

        fn name(&self) -> &str {
            "zero_derivative"
        }
    }

    fn make_config(t_end: f64, dt: f64) -> SolverConfiguration {
        SolverConfiguration::new(
            TimeConfiguration::new(t_end, StepControl::Fixed { dt }),
            IntegratorKind::RK4,
        )
    }

    fn make_mesh(n: usize) -> Box<dyn Mesh> {
        Box::new(UniformGrid1D::new(n, 0.0, 1.0).unwrap())
    }

    // ── Basic correctness ─────────────────────────────────────────────────────

    #[test]
    fn zero_derivative_field_stays_constant() {
        let scenario = Scenario::single(Box::new(ZeroDerivative), make_mesh(5));
        let config = make_config(1.0, 0.1);
        let result = RK4Solver.solve(&scenario, &config).unwrap();
        for state in &result.states {
            let field = state.as_scalar_field().unwrap();
            for v in field.iter() {
                assert!((v - 2.5).abs() < 1e-12);
            }
        }
    }

    #[test]
    fn exponential_decay_rk4_is_far_more_accurate_than_euler_order() {
        // At t=1, dt=0.1: a 1st-order method (Euler) has error ~O(dt) ~ 0.1.
        // RK4 should be many orders of magnitude more accurate at the same dt.
        let scenario = Scenario::single(Box::new(ExponentialDecay { lambda: 1.0 }), make_mesh(2));
        let config = make_config(1.0, 0.1);
        let result = RK4Solver.solve(&scenario, &config).unwrap();

        let final_state = result.states.last().unwrap().as_scalar_field().unwrap();
        let rk4_val = final_state[0];
        let analytical = (-1.0_f64).exp();

        let error = (rk4_val - analytical).abs();
        assert!(error < 1e-4, "RK4 error too large for dt=0.1: {}", error);
    }

    #[test]
    fn result_times_match_expected_steps() {
        let scenario = Scenario::single(Box::new(ZeroDerivative), make_mesh(3));
        let config = make_config(0.5, 0.1);
        let result = RK4Solver.solve(&scenario, &config).unwrap();

        assert_eq!(result.states.len(), result.times.len());
        assert!((result.times[0] - 0.0).abs() < 1e-12);
        assert!(result.t_final().unwrap() > 0.4);
    }

    #[test]
    fn n_steps_is_correct() {
        let scenario = Scenario::single(Box::new(ZeroDerivative), make_mesh(2));
        let config = make_config(1.0, 0.25);
        let result = RK4Solver.solve(&scenario, &config).unwrap();
        assert_eq!(result.n_steps, 4);
    }

    #[test]
    fn save_every_reduces_stored_states() {
        let scenario = Scenario::single(Box::new(ZeroDerivative), make_mesh(2));
        let config = SolverConfiguration::new(
            TimeConfiguration::new(1.0, StepControl::Fixed { dt: 0.1 }).saving_every(5),
            IntegratorKind::RK4,
        );
        let result = RK4Solver.solve(&scenario, &config).unwrap();
        assert_eq!(result.states.len(), 3);
    }

    // ── Floating-point time accumulation (chrom-rs regression) ───────────────

    #[test]
    fn time_accumulation_drift_is_real_and_exceeds_old_tolerance_at_scale() {
        // See euler.rs for the full rationale -- identical phenomenon,
        // independent of which integrator consumes `t`.
        let dt = 0.1_f64;
        let n = 10_000;

        let mut accumulated = 0.0_f64;
        for _ in 0..n {
            accumulated += dt;
        }
        let direct = (n as f64) * dt;
        let drift = (accumulated - direct).abs();

        assert!(
            drift > 1e-12,
            "expected measurable drift at n={n} steps, got {drift:.3e}"
        );

        let old_tolerance = dt * 1e-10;
        assert!(
            drift > old_tolerance,
            "drift {drift:.3e} should exceed the old boundary tolerance \
             {old_tolerance:.3e} at n={n} -- this is the scale where the \
             accumulating `while` loop became unsafe"
        );
    }

    #[test]
    fn step_count_and_final_time_are_exact_over_many_steps() {
        // Regression guard for the t_start + step*dt fix -- same rationale
        // and tolerance budget as the Euler counterpart. RK4 runs 4 stage
        // evaluations per step at this scale (400_000 total), still fast
        // for a trivial model.
        let dt = 0.1;
        let t_end = 10_000.0; // n_steps = 100_000
        let scenario = Scenario::single(Box::new(ZeroDerivative), make_mesh(2));
        let config = make_config(t_end, dt);
        let result = RK4Solver.solve(&scenario, &config).unwrap();

        assert_eq!(result.n_steps, 100_000);

        let final_time = *result.times.last().unwrap();
        assert!(
            (final_time - t_end).abs() < 1e-9,
            "final time {final_time} drifted too far from t_end={t_end}"
        );
    }

    // ── Order verification (acceptance criterion, #41) ───────────────────────

    #[test]
    fn rk4_error_reduces_by_16x_with_step_halving() {
        // 4th-order method: halving dt should reduce the error by ~2^4 = 16x.
        let lambda: f64 = 1.0;
        let t_end: f64 = 1.0;
        let analytical = (-(lambda * t_end)).exp();

        let error_at = |dt: f64| -> f64 {
            let scenario = Scenario::single(Box::new(ExponentialDecay { lambda }), make_mesh(2));
            let config = make_config(t_end, dt);
            let result = RK4Solver.solve(&scenario, &config).unwrap();
            let val = result.states.last().unwrap().as_scalar_field().unwrap()[0];
            (val - analytical).abs()
        };

        // Coarser steps than Euler's order test: RK4 error at very small dt
        // approaches f64 round-off, which would mask the 4th-order rate.
        let error_coarse = error_at(0.1);
        let error_fine = error_at(0.05);
        let ratio = error_coarse / error_fine;

        assert!(
            (13.0..19.0).contains(&ratio),
            "expected ~16x error reduction on dt halving, got {:.3}x (coarse={:.2e}, fine={:.2e})",
            ratio,
            error_coarse,
            error_fine
        );
    }

    // ── SteppableSolver (DD-031) ──────────────────────────────────────────────

    #[test]
    fn step_matches_one_iteration_of_solve() {
        let scenario = Scenario::single(Box::new(ExponentialDecay { lambda: 0.7 }), make_mesh(3));
        let config = make_config(0.1, 0.1); // exactly one step

        let via_solve = RK4Solver.solve(&scenario, &config).unwrap();
        let final_via_solve = via_solve.states.last().unwrap().as_scalar_field().unwrap();

        let domain = scenario.single_domain().unwrap();
        let requirements = scenario.context_requirements();
        let chain =
            crate::solver::chain::build_calculator_chain(&requirements, &config.calculators)
                .unwrap();
        let mut u = domain.model.initial_state(domain.mesh.as_ref());
        let next = RK4Solver
            .step(domain, &chain, &mut u, &[], 0.0, 0.1)
            .unwrap();
        let final_via_step = next.as_scalar_field().unwrap();

        assert_eq!(final_via_solve.len(), final_via_step.len());
        for i in 0..final_via_solve.len() {
            assert!(
                (final_via_solve[i] - final_via_step[i]).abs() < 1e-15,
                "solve() and step() diverged at index {i}: {} vs {}",
                final_via_solve[i],
                final_via_step[i]
            );
        }
    }

    // ── Boundary conditions ───────────────────────────────────────────────────

    #[test]
    fn boundary_condition_is_applied_at_every_stage() {
        use crate::boundary::{BoundaryCondition, BoundaryType};
        use crate::mesh::Mesh as MeshTrait;
        use crate::solver::scenario::Domain;
        use std::sync::atomic::{AtomicUsize, Ordering};
        use std::sync::Arc;

        /// Pins node 0 and counts how many times it was applied — used to
        /// confirm RK4 applies BCs once per stage (4 per step), not once
        /// per outer step.
        ///
        /// `Arc<AtomicUsize>`, not `Rc<Cell<usize>>` — `BoundaryCondition`
        /// requires `Send + Sync` since DD-037 (#45): a `Domain` can now be
        /// *owned* by `OperatorSplittingSolver`'s `SplitOperator`, which
        /// must itself be `Send + Sync` (`Solver: Send + Sync`).
        #[derive(Debug)]
        struct PinFirstNode {
            value: f64,
            calls: Arc<AtomicUsize>,
        }

        impl RequiresContext for PinFirstNode {
            fn required_variables(&self) -> Vec<ContextVariable> {
                vec![]
            }
        }

        impl BoundaryCondition for PinFirstNode {
            fn boundary_type(&self) -> BoundaryType {
                BoundaryType::Dirichlet
            }

            fn apply(
                &self,
                state: &mut DVector<f64>,
                _ctx: &ComputeContext,
                _mesh: &dyn MeshTrait,
            ) -> Result<(), OxiflowError> {
                state[0] = self.value;
                self.calls.fetch_add(1, Ordering::SeqCst);
                Ok(())
            }
        }

        let calls = Arc::new(AtomicUsize::new(0));
        let domain = Domain::new("pinned", Box::new(ZeroDerivative), make_mesh(3))
            .with_boundary_conditions(vec![Box::new(PinFirstNode {
                value: -7.0,
                calls: calls.clone(),
            })]);
        let scenario = Scenario::multi(vec![domain]).unwrap();
        let config = make_config(0.2, 0.1); // 2 steps

        let result = RK4Solver.solve(&scenario, &config).unwrap();

        let final_state = result.states.last().unwrap().as_scalar_field().unwrap();
        assert!((final_state[0] - (-7.0)).abs() < 1e-12);
        assert!((final_state[1] - 2.5).abs() < 1e-12);

        // 2 steps * 4 stages = 8 applications.
        assert_eq!(calls.load(Ordering::SeqCst), 8);
    }

    // ── Validation errors ─────────────────────────────────────────────────────

    #[test]
    fn negative_dt_returns_error() {
        let scenario = Scenario::single(Box::new(ZeroDerivative), make_mesh(2));
        let config = make_config(1.0, -0.1);
        assert!(RK4Solver.solve(&scenario, &config).is_err());
    }

    #[test]
    fn t_end_before_t_start_returns_error() {
        let scenario = Scenario::single(Box::new(ZeroDerivative), make_mesh(2)).with_t_start(5.0);
        let config = make_config(1.0, 0.1);
        assert!(RK4Solver.solve(&scenario, &config).is_err());
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
        let err = RK4Solver.solve(&scenario, &config).unwrap_err();
        assert!(matches!(err, OxiflowError::MissingCalculator(_)));
    }

    // ── combine / rk4_combine ─────────────────────────────────────────────────

    #[test]
    fn combine_computes_correctly() {
        let u = ContextValue::ScalarField(DVector::from_vec(vec![1.0, 2.0, 3.0]));
        let k = ContextValue::ScalarField(DVector::from_vec(vec![0.1, 0.2, 0.3]));
        let result = combine(&u, &k, 0.5).unwrap();
        let field = result.as_scalar_field().unwrap();
        assert!((field[0] - 1.05).abs() < 1e-12);
        assert!((field[1] - 2.10).abs() < 1e-12);
        assert!((field[2] - 3.15).abs() < 1e-12);
    }

    #[test]
    fn combine_mismatched_length_returns_error() {
        let u = ContextValue::ScalarField(DVector::from_element(3, 1.0));
        let k = ContextValue::ScalarField(DVector::from_element(2, 0.1));
        assert!(combine(&u, &k, 0.1).is_err());
    }

    #[test]
    fn rk4_combine_reduces_to_euler_when_stages_are_equal() {
        // If k1 == k2 == k3 == k4 == k, the weighted sum (k+2k+2k+k)/6 = k,
        // so RK4's update degenerates to the Euler update u + dt*k.
        let u = ContextValue::ScalarField(DVector::from_vec(vec![1.0, 2.0]));
        let k = ContextValue::ScalarField(DVector::from_vec(vec![0.5, -0.5]));
        let dt = 0.2;

        let rk4_result = rk4_combine(&u, &k, &k, &k, &k, dt).unwrap();
        let euler_result = combine(&u, &k, dt).unwrap();

        let rk4_field = rk4_result.as_scalar_field().unwrap();
        let euler_field = euler_result.as_scalar_field().unwrap();
        for i in 0..2 {
            assert!((rk4_field[i] - euler_field[i]).abs() < 1e-12);
        }
    }

    #[test]
    fn rk4_combine_mismatched_length_returns_error() {
        let u = ContextValue::ScalarField(DVector::from_element(3, 1.0));
        let k_ok = ContextValue::ScalarField(DVector::from_element(3, 0.1));
        let k_bad = ContextValue::ScalarField(DVector::from_element(2, 0.1));
        assert!(rk4_combine(&u, &k_ok, &k_ok, &k_bad, &k_ok, 0.1).is_err());
    }
}
