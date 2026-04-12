//! # Solver pipeline — chromatography end-to-end
//!
//! This integration test validates the full core architecture pipeline:
//!
//! ```text
//! Scenario → build_calculator_chain → ComputeContext → compute_physics → ForwardEulerSolver
//! ```
//!
//! ## Physical model
//!
//! `GaussianInjection` models the temporal evolution of a concentration field
//! under a Gaussian injection at each node (simplified 0D per-node ODE):
//!
//! $$\frac{\partial c}{\partial t} = -\lambda \cdot c + A \cdot \exp\!\left(-\frac{(t - t_{\text{inj}})^2}{2\sigma^2}\right)$$
//!
//! At J1, spatial transport ($v \cdot \partial c / \partial z$, $D_{ax} \cdot \partial^2 c / \partial z^2$)
//! is not modelled — `DiscreteOperator` (INV-2) arrives at J4b. The test validates
//! that the engine pipeline runs correctly, not the spatial accuracy of the physics.
//!
//! ## Acceptance criteria
//!
//! - Solver runs without error from `t=0` to `t=t_end`
//! - `SimulationResult` contains saved states and times
//! - Peak concentration is reached and then decays
//! - `context_requirements()` declares `[Time]`, calculator chain is built correctly

use oxiflow::{
    context::{
        compute::ComputeContext,
        error::OxiflowError,
        value::ContextValue,
        variable::ContextVariable,
    },
    mesh::{Mesh, UniformGrid1D},
    model::traits::{PhysicalModel, RequiresContext},
    solver::{
        methods::euler::ForwardEulerSolver,
        scenario::Scenario,
        config::{IntegratorKind, SolverConfiguration, StepControl, TimeConfiguration},
        Solver,
    },
};
use nalgebra::DVector;

// ── Model ─────────────────────────────────────────────────────────────────────

/// Chromatographic injection model — J1 validation (simplified ODE per node).
///
/// Models concentration decay with a Gaussian injection source:
///
/// $$\frac{dc}{dt} = -\lambda \cdot c + c_{\max} \cdot \exp\!\left(-\frac{(t - t_{\text{inj}})^2}{2\sigma^2}\right)$$
///
/// Requires `Time` from the context. All other parameters are model-level constants.
struct GaussianInjection {
    /// First-order decay rate [1/s].
    lambda: f64,
    /// Peak injection time [s].
    t_inj: f64,
    /// Injection pulse width (std deviation) [s].
    sigma: f64,
    /// Peak concentration [mol/m³].
    c_max: f64,
}

impl RequiresContext for GaussianInjection {
    fn required_variables(&self) -> Vec<ContextVariable> {
        vec![ContextVariable::Time]
    }
}

impl PhysicalModel for GaussianInjection {
    fn compute_physics(
        &self,
        state: &ContextValue,
        ctx: &ComputeContext,
    ) -> Result<ContextValue, OxiflowError> {
        let t = ctx.time();
        let c = state.as_scalar_field()?;

        // Gaussian injection source term
        let injection = self.c_max
            * (-(t - self.t_inj).powi(2) / (2.0 * self.sigma.powi(2))).exp();

        // dc/dt = -lambda * c + injection (same at every node — 0D per-node ODE)
        let dc_dt = c.map(|ci| -self.lambda * ci + injection);

        Ok(ContextValue::ScalarField(dc_dt))
    }

    fn initial_state(&self, mesh: &dyn Mesh) -> ContextValue {
        // Start with near-zero concentration
        ContextValue::ScalarField(DVector::from_element(mesh.n_dof(), 0.0))
    }

    fn name(&self) -> &str {
        "gaussian_injection"
    }

    fn description(&self) -> Option<&str> {
        Some("J1 exit criterion — Gaussian injection chromatography (0D per-node ODE)")
    }
}

// ── Parameters ────────────────────────────────────────────────────────────────

const N_NODES: usize = 50;
const L_COLUMN: f64 = 0.25;  // [m]
const T_END: f64 = 60.0;     // [s]
const DT: f64 = 0.5;         // [s]
const LAMBDA: f64 = 0.05;    // [1/s]
const T_INJ: f64 = 10.0;     // [s]
const SIGMA: f64 = 2.0;      // [s]
const C_MAX: f64 = 1.0;      // [mol/m³]

fn make_scenario() -> Scenario {
    let model = Box::new(GaussianInjection {
        lambda: LAMBDA,
        t_inj: T_INJ,
        sigma: SIGMA,
        c_max: C_MAX,
    });
    let mesh = Box::new(UniformGrid1D::new(N_NODES, 0.0, L_COLUMN).unwrap());
    Scenario::single(model, mesh)
}

fn make_config() -> SolverConfiguration {
    SolverConfiguration::new(
        TimeConfiguration::new(T_END, StepControl::Fixed { dt: DT }),
        IntegratorKind::Euler,
    )
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[test]
fn core_architecture_solver_runs_to_completion() {
    let result = ForwardEulerSolver.solve(&make_scenario(), &make_config());
    assert!(result.is_ok(), "solver failed: {:?}", result.err());
}

#[test]
fn core_architecture_result_has_correct_structure() {
    let result = ForwardEulerSolver.solve(&make_scenario(), &make_config()).unwrap();

    // states and times have the same length
    assert_eq!(result.states.len(), result.times.len());
    assert!(!result.is_empty());

    // Initial state is saved at t=0
    assert!((result.times[0] - 0.0).abs() < 1e-10);

    // Final time is close to t_end
    let t_final = result.t_final().unwrap();
    assert!(t_final > T_END - DT, "t_final={} < t_end-dt={}", t_final, T_END - DT);
}

#[test]
fn core_architecture_initial_concentration_is_zero() {
    let result = ForwardEulerSolver.solve(&make_scenario(), &make_config()).unwrap();
    let initial = result.states[0].as_scalar_field().unwrap();
    assert_eq!(initial.len(), N_NODES);
    for v in initial.iter() {
        assert!(v.abs() < 1e-12, "initial concentration not zero: {}", v);
    }
}

#[test]
fn core_architecture_concentration_rises_then_falls() {
    let result = ForwardEulerSolver.solve(&make_scenario(), &make_config()).unwrap();

    // Sample node 0 concentration over time
    let concs: Vec<f64> = result.states.iter()
        .map(|s| s.as_scalar_field().unwrap()[0])
        .collect();

    // Should start near zero
    assert!(concs[0].abs() < 1e-10);

    // Should reach a peak above zero
    let peak = concs.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    assert!(peak > 0.01, "peak concentration too low: {}", peak);

    // Final concentration should be lower than peak (injection ended, decay continues)
    let final_c = *concs.last().unwrap();
    assert!(
        final_c < peak,
        "final concentration {} >= peak {}", final_c, peak
    );
}

#[test]
fn core_architecture_all_states_are_finite() {
    let result = ForwardEulerSolver.solve(&make_scenario(), &make_config()).unwrap();
    for (i, state) in result.states.iter().enumerate() {
        let field = state.as_scalar_field().unwrap();
        for (j, v) in field.iter().enumerate() {
            assert!(
                v.is_finite(),
                "non-finite value at step {} node {}: {}", i, j, v
            );
        }
    }
}

#[test]
fn core_architecture_n_steps_matches_time_span() {
    let result = ForwardEulerSolver.solve(&make_scenario(), &make_config()).unwrap();
    let expected_steps = (T_END / DT).ceil() as usize;
    assert_eq!(result.n_steps, expected_steps);
}

#[test]
fn core_architecture_context_requirements_declares_time() {
    let scenario = make_scenario();
    let reqs = scenario.context_requirements();
    assert!(
        reqs.contains(&ContextVariable::Time),
        "Time not in requirements: {:?}", reqs
    );
}

#[test]
fn core_architecture_no_calculator_needed_for_time() {
    // Time is a built-in variable — the chain should build without any user calculator
    let config = make_config(); // no calculators added
    let result = ForwardEulerSolver.solve(&make_scenario(), &config);
    assert!(
        result.is_ok(),
        "solver failed without Time calculator: {:?}", result.err()
    );
}
