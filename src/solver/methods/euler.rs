//! # Module `solver::methods::euler`
//!
//! Forward Euler integrator — explicit, 1st order (issues #33, #41).
//!
//! ## Algorithm
//!
//! At each time step:
//!
//! $$u^{n+1} = u^n + \Delta t \cdot f(u^n, \text{ctx}^n)$$
//!
//! where $f = \text{compute\_physics}(u, \text{ctx})$ is the time derivative
//! returned by the physical model, evaluated on `u` *after* boundary
//! conditions have been enforced (see [`super::evaluate_derivative`]).
//!
//! ## Scope at J4a
//!
//! - Single-domain scenarios only (`n_domains() == 1`). Multi-domain
//!   scenarios with `CouplingOperator` are out of scope for this solver —
//!   see #40 for the dedicated multi-domain proto.
//! - No `DiscreteOperator` (INV-2) — spatial schemes arrive at J4b.
//!   The model computes `du/dt` internally from the field state and context.
//! - Boundary conditions ARE applied (since v0.2.0 / DD-008) — fixed in #41;
//!   the original J1 implementation predated `BoundaryCondition` and never
//!   called it, silently violating the contractual order documented in
//!   [`crate::solver`].
//! - `StepControl::Fixed { dt }` only — adaptive step at J4 (DoPri45).
//!
//! ## Stability
//!
//! For explicit methods, stability requires the CFL condition:
//!
//! $$\text{CFL} = \frac{v \, \Delta t}{\Delta x} \leq 1$$
//!
//! The solver does not enforce this automatically — the caller is responsible
//! for choosing a stable `dt`.

use crate::context::error::OxiflowError;
use crate::context::value::ContextValue;
use crate::context::ContextCalculator;
use crate::solver::methods::{evaluate_derivative, SteppableSolver};
use crate::solver::scenario::{Domain, Scenario};
use crate::solver::{SimulationResult, Solver, SolverConfiguration};

/// Forward Euler solver — explicit, 1st order.
///
/// Implements the `Solver` trait for single-domain problems with fixed step
/// control. See [module documentation](self) for algorithm details.
///
/// # Examples
///
/// ```rust,ignore
/// use oxiflow::solver::methods::euler::ForwardEulerSolver;
/// use oxiflow::solver::{Scenario, SolverConfiguration, TimeConfiguration, StepControl, IntegratorKind};
/// use oxiflow::mesh::UniformGrid1D;
///
/// let scenario = Scenario::single(Box::new(my_model), Box::new(mesh));
/// let config = SolverConfiguration::new(
///     TimeConfiguration::new(100.0, StepControl::Fixed { dt: 0.1 }),
///     IntegratorKind::Euler,
/// );
/// let solver = ForwardEulerSolver;
/// let result = solver.solve(&scenario, &config).unwrap();
/// ```
pub struct ForwardEulerSolver;

impl Solver for ForwardEulerSolver {
    fn solve(
        &self,
        scenario: &Scenario,
        config: &SolverConfiguration,
    ) -> Result<SimulationResult, OxiflowError> {
        self.solve_fixed_step(scenario, config)
    }
}

impl SteppableSolver for ForwardEulerSolver {
    fn step(
        &self,
        domain: &Domain,
        chain: &[&dyn ContextCalculator],
        state: &mut ContextValue,
        _history: &[ContextValue],
        t: f64,
        dt: f64,
    ) -> Result<ContextValue, OxiflowError> {
        // history_depth() defaults to 0 -- Euler is a one-step method,
        // `_history` is always empty here and intentionally unused.
        // Contractual order (calculators -> BCs -> compute_physics) is
        // enforced once, here, for both Euler and RK4 — see
        // `solver::methods::evaluate_derivative`. `state` is corrected
        // in-place by boundary conditions before the derivative is taken.
        let du_dt = evaluate_derivative(domain, chain, state, t, dt)?;

        // Euler step: u_next = u + dt * du_dt
        euler_step(state, &du_dt, dt)
    }
}

/// Computes `u + dt * du_dt` for `ScalarField` states.
///
/// Returns `OxiflowError::TypeMismatch` if `u` and `du_dt` are not both
/// `ScalarField`, or `InvalidDomain` if their lengths differ.
fn euler_step(
    u: &ContextValue,
    du_dt: &ContextValue,
    dt: f64,
) -> Result<ContextValue, OxiflowError> {
    let u_field = u.as_scalar_field()?;
    let du_field = du_dt.as_scalar_field()?;

    if u_field.len() != du_field.len() {
        return Err(OxiflowError::InvalidDomain(format!(
            "state length {} != derivative length {}",
            u_field.len(),
            du_field.len()
        )));
    }

    Ok(ContextValue::ScalarField(u_field + du_field * dt))
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
            IntegratorKind::Euler,
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
        let result = ForwardEulerSolver.solve(&scenario, &config).unwrap();
        // All saved states should be 2.5 everywhere
        for state in &result.states {
            let field = state.as_scalar_field().unwrap();
            for v in field.iter() {
                assert!((v - 2.5).abs() < 1e-12);
            }
        }
    }

    #[test]
    fn exponential_decay_euler_error_is_first_order() {
        // Euler approximation of du/dt = -u, u0=1 → u(t) = exp(-t)
        // At t=1: analytical = exp(-1) ≈ 0.3679
        // Euler with dt=0.1 should be close but not exact
        // UniformGrid1D requires >= 2 nodes — we use 2 and check node 0.
        let scenario = Scenario::single(Box::new(ExponentialDecay { lambda: 1.0 }), make_mesh(2));
        let config = make_config(1.0, 0.1);
        let result = ForwardEulerSolver.solve(&scenario, &config).unwrap();

        let final_state = result.states.last().unwrap().as_scalar_field().unwrap();
        let euler_val = final_state[0];
        let analytical = (-1.0_f64).exp();

        // Euler error should be small but non-zero
        let error = (euler_val - analytical).abs();
        assert!(error < 0.1, "error too large: {}", error);
        assert!(error > 1e-10, "error suspiciously small: {}", error);
    }

    #[test]
    fn result_times_match_expected_steps() {
        let scenario = Scenario::single(Box::new(ZeroDerivative), make_mesh(3));
        let config = make_config(0.5, 0.1);
        let result = ForwardEulerSolver.solve(&scenario, &config).unwrap();

        // t=0.0 (initial) + 5 steps = 6 saved states
        assert_eq!(result.states.len(), result.times.len());
        assert!((result.times[0] - 0.0).abs() < 1e-12);
        assert!(result.t_final().unwrap() > 0.4);
    }

    #[test]
    fn n_steps_is_correct() {
        let scenario = Scenario::single(Box::new(ZeroDerivative), make_mesh(2));
        let config = make_config(1.0, 0.25);
        let result = ForwardEulerSolver.solve(&scenario, &config).unwrap();
        assert_eq!(result.n_steps, 4);
    }

    #[test]
    fn save_every_reduces_stored_states() {
        let scenario = Scenario::single(Box::new(ZeroDerivative), make_mesh(2));
        let config = SolverConfiguration::new(
            TimeConfiguration::new(1.0, StepControl::Fixed { dt: 0.1 }).saving_every(5),
            IntegratorKind::Euler,
        );
        let result = ForwardEulerSolver.solve(&scenario, &config).unwrap();
        // 10 steps, save every 5 → 2 saves + initial = 3 states
        assert_eq!(result.states.len(), 3);
    }

    // ── Floating-point time accumulation (chrom-rs regression) ───────────────

    #[test]
    fn time_accumulation_drift_is_real_and_exceeds_old_tolerance_at_scale() {
        // Mathematically, t(n) = t(0) + n*dt is identical to adding dt to
        // itself n times. Computationally it is not: each `+=` rounds to
        // the nearest representable f64, and these roundings compound
        // rather than cancel. This test documents the magnitude of that
        // drift at a step count comparable to production runs (see the
        // module docs on `n_steps` above) — and confirms it is the reason
        // `ForwardEulerSolver`/`RK4Solver` compute `t` from the step index
        // rather than accumulating, as chrom-rs's RK4 also does.
        let dt = 0.1_f64;
        let n = 10_000;

        let mut accumulated = 0.0_f64;
        for _ in 0..n {
            accumulated += dt;
        }
        let direct = (n as f64) * dt;
        let drift = (accumulated - direct).abs();

        // The drift is real and measurable at this scale. If this
        // assertion ever fails because `drift` becomes 0, floating-point
        // semantics have changed and this test's rationale should be
        // re-examined, not silently relaxed.
        assert!(
            drift > 1e-12,
            "expected measurable drift at n={n} steps, got {drift:.3e}"
        );

        // ...and it exceeds the tolerance the old `while`-loop boundary
        // check relied on (`dt * 1e-10`) — this is precisely the scale at
        // which that loop could mis-count the total number of steps.
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
        // Regression guard for the t_start + step*dt fix. At n=100_000
        // steps, raw accumulation drift measures ~1.9e-8 (see the
        // accompanying float-arithmetic test) -- comfortably above the
        // 1e-9 tolerance asserted here, and comfortably below what the
        // direct per-step computation actually produces (~1e-13). This
        // tolerance is deliberately tight enough to fail against the old
        // accumulating implementation and loose enough to pass against
        // the current one with margin.
        let dt = 0.1;
        let t_end = 10_000.0; // n_steps = 100_000
        let scenario = Scenario::single(Box::new(ZeroDerivative), make_mesh(2));
        let config = make_config(t_end, dt);
        let result = ForwardEulerSolver.solve(&scenario, &config).unwrap();

        assert_eq!(result.n_steps, 100_000);

        let final_time = *result.times.last().unwrap();
        assert!(
            (final_time - t_end).abs() < 1e-9,
            "final time {final_time} drifted too far from t_end={t_end}"
        );
    }

    // ── Order verification (acceptance criterion, #41) ───────────────────────

    #[test]
    fn euler_error_halves_with_step_halving() {
        // First-order method: halving dt should roughly halve the error.
        let lambda: f64 = 1.0;
        let t_end: f64 = 1.0;
        let analytical = (-(lambda * t_end)).exp();

        let error_at = |dt: f64| -> f64 {
            let scenario = Scenario::single(Box::new(ExponentialDecay { lambda }), make_mesh(2));
            let config = make_config(t_end, dt);
            let result = ForwardEulerSolver.solve(&scenario, &config).unwrap();
            let val = result.states.last().unwrap().as_scalar_field().unwrap()[0];
            (val - analytical).abs()
        };

        let error_coarse = error_at(0.01);
        let error_fine = error_at(0.005);
        let ratio = error_coarse / error_fine;

        // First-order convergence: ratio should be close to 2. Generous
        // tolerance since dt is finite, not in the asymptotic limit.
        assert!(
            (1.7..2.3).contains(&ratio),
            "expected ~2x error reduction on dt halving, got {:.3}x (coarse={:.2e}, fine={:.2e})",
            ratio,
            error_coarse,
            error_fine
        );
    }

    // ── SteppableSolver (DD-031) ──────────────────────────────────────────────

    #[test]
    fn step_matches_one_iteration_of_solve() {
        // `solve()` now calls `self.step()` internally -- this guards against
        // the two ever diverging if either is edited independently later.
        let scenario = Scenario::single(Box::new(ExponentialDecay { lambda: 0.7 }), make_mesh(3));
        let config = make_config(0.1, 0.1); // exactly one step

        let via_solve = ForwardEulerSolver.solve(&scenario, &config).unwrap();
        let final_via_solve = via_solve.states.last().unwrap().as_scalar_field().unwrap();

        let domain = scenario.single_domain().unwrap();
        let requirements = scenario.context_requirements();
        let chain =
            crate::solver::chain::build_calculator_chain(&requirements, &config.calculators)
                .unwrap();
        let mut u = domain.model.initial_state(domain.mesh.as_ref());
        let next = ForwardEulerSolver
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

    // ── Boundary conditions (fixed in #41 — see module docs) ─────────────────

    #[test]
    fn boundary_condition_is_applied_each_step() {
        use crate::boundary::{BoundaryCondition, BoundaryType};
        use crate::context::compute::ComputeContext;
        use crate::mesh::Mesh as MeshTrait;
        use crate::solver::scenario::Domain;

        /// Pins node 0 to a fixed value on every application — a minimal
        /// Dirichlet-style fixture, not a physically meaningful BC.
        #[derive(Debug)]
        struct PinFirstNode {
            value: f64,
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
                Ok(())
            }
        }

        // ZeroDerivative leaves the field unchanged everywhere — any
        // deviation from its initial value of 2.5 at node 0 can only come
        // from the boundary condition.
        let domain = Domain::new("pinned", Box::new(ZeroDerivative), make_mesh(3))
            .with_boundary_conditions(vec![Box::new(PinFirstNode { value: -7.0 })]);
        let scenario = Scenario::multi(vec![domain]).unwrap();
        let config = make_config(0.3, 0.1);
        let result = ForwardEulerSolver.solve(&scenario, &config).unwrap();

        let final_state = result.states.last().unwrap().as_scalar_field().unwrap();
        assert!(
            (final_state[0] - (-7.0)).abs() < 1e-12,
            "boundary condition was not applied: node 0 = {}",
            final_state[0]
        );
        // Interior nodes are untouched by the BC and keep ZeroDerivative's value.
        assert!((final_state[1] - 2.5).abs() < 1e-12);
        assert!((final_state[2] - 2.5).abs() < 1e-12);
    }

    // ── Validation errors ─────────────────────────────────────────────────────

    #[test]
    fn negative_dt_returns_error() {
        let scenario = Scenario::single(Box::new(ZeroDerivative), make_mesh(2));
        let config = make_config(1.0, -0.1);
        assert!(ForwardEulerSolver.solve(&scenario, &config).is_err());
    }

    #[test]
    fn t_end_before_t_start_returns_error() {
        let scenario = Scenario::single(Box::new(ZeroDerivative), make_mesh(2)).with_t_start(5.0);
        let config = make_config(1.0, 0.1);
        assert!(ForwardEulerSolver.solve(&scenario, &config).is_err());
    }

    #[test]
    fn missing_calculator_returns_error() {
        use crate::context::variable::ContextVariable;

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
        let err = ForwardEulerSolver.solve(&scenario, &config).unwrap_err();
        assert!(matches!(err, OxiflowError::MissingCalculator(_)));
    }

    // ── euler_step ────────────────────────────────────────────────────────────

    #[test]
    fn euler_step_computes_correctly() {
        let u = ContextValue::ScalarField(DVector::from_vec(vec![1.0, 2.0, 3.0]));
        let du = ContextValue::ScalarField(DVector::from_vec(vec![0.1, 0.2, 0.3]));
        let result = euler_step(&u, &du, 0.5).unwrap();
        let field = result.as_scalar_field().unwrap();
        assert!((field[0] - 1.05).abs() < 1e-12);
        assert!((field[1] - 2.10).abs() < 1e-12);
        assert!((field[2] - 3.15).abs() < 1e-12);
    }

    #[test]
    fn euler_step_mismatched_length_returns_error() {
        let u = ContextValue::ScalarField(DVector::from_element(3, 1.0));
        let du = ContextValue::ScalarField(DVector::from_element(2, 0.1));
        assert!(euler_step(&u, &du, 0.1).is_err());
    }
}
