//! # Module `solver::methods::euler`
//!
//! Forward Euler integrator — explicit, 1st order (issue #33).
//!
//! ## Algorithm
//!
//! At each time step:
//!
//! $$u^{n+1} = u^n + \Delta t \cdot f(u^n, \text{ctx}^n)$$
//!
//! where $f = \text{compute\_physics}(u, \text{ctx})$ is the time derivative
//! returned by the physical model.
//!
//! ## Scope at J1
//!
//! - Single-domain scenarios only (`n_domains() == 1`).
//! - No `DiscreteOperator` (INV-2) — spatial schemes arrive at J4b.
//!   The model computes `du/dt` internally from the field state and context.
//! - No boundary conditions — `BoundaryCondition` arrives at J2.
//! - `StepControl::Fixed { dt }` only — adaptive step at J4.
//!
//! ## Stability
//!
//! For explicit methods, stability requires the CFL condition:
//!
//! $$\text{CFL} = \frac{v \, \Delta t}{\Delta x} \leq 1$$
//!
//! The solver does not enforce this automatically — the caller is responsible
//! for choosing a stable `dt`.

use std::collections::HashMap;

use crate::context::compute::ComputeContext;
use crate::context::error::OxiflowError;
use crate::context::value::ContextValue;
use crate::solver::chain::build_calculator_chain;
use crate::solver::config::StepControl;
use crate::solver::scenario::Scenario;
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
        // ── Pre-solve validation ───────────────────────────────────────────────
        scenario.validate()?;

        let domain = scenario.single_domain()?;

        let dt = match &config.time.step_control {
            StepControl::Fixed { dt } => *dt,
            _ => {
                return Err(OxiflowError::InvalidDomain(
                    "ForwardEulerSolver only supports StepControl::Fixed at J1".into(),
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

        // ── Build calculator chain ─────────────────────────────────────────────
        let requirements = scenario.context_requirements();
        let chain = build_calculator_chain(&requirements, &config.calculators)?;

        // ── Initial state ──────────────────────────────────────────────────────
        let mut u = domain.model.initial_state(domain.mesh.as_ref());

        // ── Result buffers ─────────────────────────────────────────────────────
        let save_every = config.time.save_every.unwrap_or(1);
        let capacity = ((t_end - t_start) / dt).ceil() as usize / save_every + 1;
        let mut states: Vec<ContextValue> = Vec::with_capacity(capacity);
        let mut times: Vec<f64> = Vec::with_capacity(capacity);

        // Save initial state
        states.push(u.clone());
        times.push(t_start);

        // ── Time loop ──────────────────────────────────────────────────────────
        let mut t = t_start;
        let mut step = 0usize;

        while t + dt <= t_end + dt * 1e-10 {
            // 1. Build ComputeContext for this step
            let mut ctx = ComputeContext::new(t, dt);

            // 2. Run calculators in priority order
            for calc in &chain {
                let value =
                    calc.compute(&u, &ctx)
                        .map_err(|e| OxiflowError::ComputationFailed {
                            variable: calc.provides(),
                            source: Box::new(e),
                        })?;
                ctx.insert(calc.provides(), value);
            }

            // 3. BCs — RESERVED J2

            // 4. Compute du/dt
            let du_dt = domain.model.compute_physics(&u, &ctx)?;

            // 5. Euler step: u_next = u + dt * du_dt
            u = euler_step(&u, &du_dt, dt)?;

            t += dt;
            step += 1;

            // Guard against NaN / Inf
            check_finite(&u, t)?;

            // Save according to frequency
            if step % save_every == 0 {
                states.push(u.clone());
                times.push(t);
            }
        }

        Ok(SimulationResult {
            states,
            times,
            n_steps: step,
            metadata: HashMap::new(),
        })
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

/// Checks that all nodal values are finite.
fn check_finite(u: &ContextValue, t: f64) -> Result<(), OxiflowError> {
    if let Ok(field) = u.as_scalar_field() {
        if field.iter().any(|v| !v.is_finite()) {
            return Err(OxiflowError::SolverDivergence {
                time: t,
                reason: "non-finite value detected in state vector".into(),
            });
        }
    }
    Ok(())
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
                vec![ContextVariable::External { name: "missing" }]
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
