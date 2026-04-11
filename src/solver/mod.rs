//! # Module `solver`
//!
//! Numerical solving orchestration — WHAT/HOW separation (issue #32).
//!
//! ## Responsibilities
//!
//! | Type | Role |
//! |---|---|
//! | [`scenario::Scenario`] | Declares the problem (WHAT) |
//! | [`config::SolverConfiguration`] | Configures solving (HOW) |
//! | [`Solver`] | Orchestrates execution |
//!
//! ## Contractual execution order
//!
//! `Solver::solve()` implementations must follow this order at each time step:
//!
//! 1. **Calculators** — populate `ComputeContext` in topological order
//! 2. **Boundary conditions** — apply to state using `ctx` (J2)
//! 3. **`compute_physics`** — compute `du/dt` from state + context
//! 4. **Integrate** — advance state by `dt`
//!
//! This order is a contract, not a convention. Deviating from it produces
//! silently incorrect results.

pub mod chain;
pub mod config;
pub mod methods;
pub mod scenario;

pub use config::{IntegratorKind, SolverConfiguration, StepControl, TimeConfiguration};
pub use scenario::{Domain, DomainId, Scenario};

use crate::context::error::OxiflowError;

// ── SimulationResult ──────────────────────────────────────────────────────────

/// Result of a completed simulation.
///
/// `states` and `times` have the same length. The save frequency is controlled
/// by `SolverConfiguration::time.save_every`.
///
/// # Examples
///
/// ```rust, ignore
/// use oxiflow::solver::SimulationResult;
/// use oxiflow::context::value::ContextValue;
/// use nalgebra::DVector;
///
/// let result = SimulationResult {
///     states: vec![ContextValue::ScalarField(DVector::from_element(10, 0.0))],
///     times:  vec![0.0],
///     n_steps: 1,
/// };
/// assert_eq!(result.states.len(), result.times.len());
/// ```
#[non_exhaustive]
pub struct SimulationResult {
    /// Saved field states at each recorded time.
    pub states: Vec<crate::context::value::ContextValue>,
    /// Simulation times corresponding to each saved state.
    pub times: Vec<f64>,
    /// Total number of time steps taken (may be larger than `states.len()`
    /// if `save_every > 1`).
    pub n_steps: usize,
    /// Solver metadata: timing, rejected steps, convergence info.
    ///
    /// Keys follow the convention `"solver.<key>"` (e.g. `"solver.rejected_steps"`).
    /// Empty at J1 — populated by adaptive integrators at J4 (DoPri45, BDF2).
    pub metadata: std::collections::HashMap<String, f64>,
}

impl SimulationResult {
    /// Returns the number of saved states.
    pub fn len(&self) -> usize {
        self.states.len()
    }

    /// Returns `true` if no states were saved.
    pub fn is_empty(&self) -> bool {
        self.states.is_empty()
    }

    /// Returns the final simulation time.
    pub fn t_final(&self) -> Option<f64> {
        self.times.last().copied()
    }
}

// ── Solver trait ──────────────────────────────────────────────────────────────

/// Orchestrates the time integration loop.
///
/// Implementations receive a `Scenario` (WHAT) and a `SolverConfiguration`
/// (HOW) and execute the contractual loop until `t_end`.
///
/// At J1, `Solver` implementations must:
/// - Verify `scenario.n_domains() == 1` (multi-domain requires J3)
/// - Build the calculator chain via `chain::build_calculator_chain()`
/// - Follow the contractual execution order
///
/// # Object safety
///
/// This trait is object-safe to support INV-4 (plugin-safe API, v2.0).
pub trait Solver: Send + Sync {
    /// Runs the simulation and returns the collected states.
    fn solve(
        &self,
        scenario: &Scenario,
        config: &SolverConfiguration,
    ) -> Result<SimulationResult, OxiflowError>;
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::context::value::ContextValue;
    use nalgebra::DVector;

    #[test]
    fn simulation_result_len() {
        let r = SimulationResult {
            states: vec![ContextValue::ScalarField(DVector::from_element(5, 0.0))],
            times: vec![1.0],
            n_steps: 10,
            metadata: std::collections::HashMap::new(),
        };
        assert_eq!(r.len(), 1);
        assert!(!r.is_empty());
    }

    #[test]
    fn simulation_result_t_final() {
        let r = SimulationResult {
            states: vec![
                ContextValue::ScalarField(DVector::from_element(5, 0.0)),
                ContextValue::ScalarField(DVector::from_element(5, 1.0)),
            ],
            times: vec![0.0, 1.0],
            n_steps: 2,
            metadata: std::collections::HashMap::new(),
        };
        assert_eq!(r.t_final(), Some(1.0));
    }

    #[test]
    fn empty_result() {
        let r = SimulationResult {
            states: vec![],
            times: vec![],
            n_steps: 0,
            metadata: std::collections::HashMap::new(),
        };
        assert!(r.is_empty());
        assert_eq!(r.t_final(), None);
    }

    #[test]
    fn solver_is_object_safe() {
        fn assert_object_safe<T: Solver + ?Sized>() {}
        assert_object_safe::<dyn Solver>();
    }
}
