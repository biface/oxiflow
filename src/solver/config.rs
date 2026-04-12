//! # Module `solver::config`
//!
//! Solving configuration — `SolverConfiguration` (HOW pole, DD-021, issue #32).
//!
//! ## Design
//!
//! `dt: f64` is never exposed directly in the public API. Instead, `StepControl`
//! encapsulates both fixed-step and adaptive-step strategies. `TimeConfiguration`
//! groups all temporal parameters. This prevents the breaking change that would
//! occur at J4 when adaptive integrators (DoPri45, BDF2) are introduced (DD-021).

use crate::context::calculator::ContextCalculator;

// ── StepControl ───────────────────────────────────────────────────────────────

/// Time step control strategy.
///
/// At J1, only `Fixed` is used. `Adaptive` is reserved for J4 (DoPri45, BDF2)
/// and added as a new variant — non-breaking for all J1/J2 code.
///
/// # Examples
///
/// ```rust
/// use oxiflow::solver::config::StepControl;
///
/// let fixed = StepControl::Fixed { dt: 0.01 };
/// assert_eq!(fixed.dt_initial(), 0.01);
/// ```
#[derive(Debug, Clone)]
#[non_exhaustive]
pub enum StepControl {
    /// Fixed time step — J1.
    ///
    /// Stability for explicit methods requires the CFL condition:
    ///
    /// $$\text{CFL} = \frac{v \, \Delta t}{\Delta x} \leq 1$$
    Fixed {
        /// Time step size.
        dt: f64,
    },

    /// Adaptive step-size control — RESERVED J4 (DoPri45, BDF2, DD-021).
    ///
    /// The integrator adjusts $\Delta t$ to keep the local truncation error
    /// within the tolerance band:
    ///
    /// $$\| e \| \leq \text{atol} + \text{rtol} \cdot \| u \|$$
    Adaptive {
        /// Initial time step guess.
        dt_init: f64,
        /// Minimum allowed time step — `OxiflowError::SolverDivergence` if reached.
        dt_min: f64,
        /// Maximum allowed time step.
        dt_max: f64,
        /// Relative tolerance.
        rtol: f64,
        /// Absolute tolerance.
        atol: f64,
    },
}

impl StepControl {
    /// Returns the initial `dt` regardless of strategy.
    pub fn dt_initial(&self) -> f64 {
        match self {
            Self::Fixed { dt } => *dt,
            Self::Adaptive { dt_init, .. } => *dt_init,
        }
    }

    /// Returns `true` if this is a fixed step strategy.
    pub fn is_fixed(&self) -> bool {
        matches!(self, Self::Fixed { .. })
    }

    /// Returns `true` if this is an adaptive step strategy.
    pub fn is_adaptive(&self) -> bool {
        matches!(self, Self::Adaptive { .. })
    }
}

// ── IntegratorKind ────────────────────────────────────────────────────────────

/// Temporal integration method.
///
/// At J1, only `Euler` and `RK4` are active. Other variants are
/// reserved for J4 (explicit, implicit, adaptive, IMEX).
///
/// # Examples
///
/// ```rust
/// use oxiflow::solver::config::IntegratorKind;
///
/// let method = IntegratorKind::Euler;
/// assert!(method.is_explicit());
/// ```
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum IntegratorKind {
    /// Forward Euler — explicit, 1st order — J1.
    Euler,
    /// Runge-Kutta 4 — explicit, 4th order — J1.
    RK4,
    // Reserved J4 — DoPri45, BackwardEuler, CrankNicolson, BDF2, IMEX
}

impl IntegratorKind {
    /// Returns `true` if the method is explicit.
    pub fn is_explicit(&self) -> bool {
        matches!(self, Self::Euler | Self::RK4)
    }
}

// ── TimeConfiguration ─────────────────────────────────────────────────────────

/// Temporal simulation parameters.
///
/// Groups `t_end`, step control strategy, and output frequency.
/// Decoupled from `SolverConfiguration` so that time parameters can be
/// modified independently of the integration method and calculators.
///
/// # Examples
///
/// ```rust
/// use oxiflow::solver::config::{TimeConfiguration, StepControl};
///
/// let time = TimeConfiguration::new(600.0, StepControl::Fixed { dt: 0.1 });
/// assert_eq!(time.t_end, 600.0);
/// assert_eq!(time.n_steps_estimate(), 6000);
/// ```
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct TimeConfiguration {
    /// End time of the simulation.
    pub t_end: f64,
    /// Step control strategy.
    pub step_control: StepControl,
    /// Save state every N steps into `SimulationResult`.
    ///
    /// `None` — save every step (default; suitable for short simulations).
    /// `Some(n)` — save every n-th step (avoids large result vectors).
    pub save_every: Option<usize>,
}

impl TimeConfiguration {
    /// Creates a time configuration with default save frequency (every step).
    pub fn new(t_end: f64, step_control: StepControl) -> Self {
        Self {
            t_end,
            step_control,
            save_every: None,
        }
    }

    /// Sets the save frequency.
    pub fn saving_every(mut self, n: usize) -> Self {
        self.save_every = Some(n);
        self
    }

    /// Estimates the number of steps for fixed step control.
    ///
    /// Returns 0 for adaptive step control (unknown a priori).
    pub fn n_steps_estimate(&self) -> usize {
        match &self.step_control {
            StepControl::Fixed { dt } => {
                if *dt > 0.0 {
                    (self.t_end / dt).ceil() as usize
                } else {
                    0
                }
            }
            StepControl::Adaptive { .. } => 0,
        }
    }
}

// ── SolverConfiguration ───────────────────────────────────────────────────────

/// Solving configuration — HOW pole.
///
/// Groups the integration method, temporal parameters, and context calculators.
/// `DiscreteOperator` (INV-2, J4b) is **not** a configuration field — it is
/// an implementation detail inside spatial `ContextCalculator`s.
///
/// # Examples
///
/// ```rust
/// use oxiflow::solver::config::{
///     SolverConfiguration, TimeConfiguration, StepControl, IntegratorKind,
/// };
///
/// let config = SolverConfiguration::new(
///     TimeConfiguration::new(600.0, StepControl::Fixed { dt: 0.1 }),
///     IntegratorKind::Euler,
/// );
/// assert!(config.calculators.is_empty());
/// assert_eq!(config.time.t_end, 600.0);
/// ```
#[non_exhaustive]
pub struct SolverConfiguration {
    /// Temporal parameters — t_end, step control, save frequency.
    pub time: TimeConfiguration,
    /// Temporal integration method.
    pub integrator: IntegratorKind,
    /// Context variable calculators provided by the user.
    ///
    /// The solver chains these in topological order to populate `ComputeContext`
    /// at each time step. Built-in calculators (Time, TimeStep) are always added
    /// automatically; only derived quantities need user-supplied calculators.
    pub calculators: Vec<Box<dyn ContextCalculator>>,
    // external_data: Option<Arc<dyn ExternalDataProvider>>  — RESERVED J2
    // parallel_threshold: Option<usize>                     — RESERVED J5 (DD-014)
}

impl SolverConfiguration {
    /// Creates a new solver configuration with no user calculators.
    pub fn new(time: TimeConfiguration, integrator: IntegratorKind) -> Self {
        Self {
            time,
            integrator,
            calculators: Vec::new(),
        }
    }

    /// Adds a context calculator (builder pattern).
    pub fn with_calculator(mut self, calc: Box<dyn ContextCalculator>) -> Self {
        self.calculators.push(calc);
        self
    }

    /// Adds multiple context calculators at once.
    pub fn with_calculators(mut self, calcs: Vec<Box<dyn ContextCalculator>>) -> Self {
        self.calculators.extend(calcs);
        self
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::context::compute::ComputeContext;
    use crate::context::error::OxiflowError;
    use crate::context::value::ContextValue;
    use crate::context::variable::ContextVariable;
    use crate::model::traits::RequiresContext;

    // ── StepControl ───────────────────────────────────────────────────────────

    #[test]
    fn fixed_dt_initial() {
        let sc = StepControl::Fixed { dt: 0.05 };
        assert_eq!(sc.dt_initial(), 0.05);
    }

    #[test]
    fn adaptive_dt_initial() {
        let sc = StepControl::Adaptive {
            dt_init: 0.01,
            dt_min: 1e-6,
            dt_max: 1.0,
            rtol: 1e-4,
            atol: 1e-6,
        };
        assert_eq!(sc.dt_initial(), 0.01);
    }

    #[test]
    fn is_fixed_and_is_adaptive() {
        assert!(StepControl::Fixed { dt: 0.01 }.is_fixed());
        assert!(!StepControl::Fixed { dt: 0.01 }.is_adaptive());
        let adaptive = StepControl::Adaptive {
            dt_init: 0.01,
            dt_min: 1e-6,
            dt_max: 1.0,
            rtol: 1e-4,
            atol: 1e-6,
        };
        assert!(adaptive.is_adaptive());
        assert!(!adaptive.is_fixed());
    }

    // ── IntegratorKind ────────────────────────────────────────────────────────

    #[test]
    fn euler_and_rk4_are_explicit() {
        assert!(IntegratorKind::Euler.is_explicit());
        assert!(IntegratorKind::RK4.is_explicit());
    }

    #[test]
    fn integrator_equality() {
        assert_eq!(IntegratorKind::Euler, IntegratorKind::Euler);
        assert_ne!(IntegratorKind::Euler, IntegratorKind::RK4);
    }

    // ── TimeConfiguration ─────────────────────────────────────────────────────

    #[test]
    fn n_steps_estimate_fixed() {
        let tc = TimeConfiguration::new(10.0, StepControl::Fixed { dt: 0.01 });
        assert_eq!(tc.n_steps_estimate(), 1000);
    }

    #[test]
    fn n_steps_estimate_adaptive_is_zero() {
        let tc = TimeConfiguration::new(
            10.0,
            StepControl::Adaptive {
                dt_init: 0.01,
                dt_min: 1e-6,
                dt_max: 1.0,
                rtol: 1e-4,
                atol: 1e-6,
            },
        );
        assert_eq!(tc.n_steps_estimate(), 0);
    }

    #[test]
    fn saving_every_builder() {
        let tc = TimeConfiguration::new(100.0, StepControl::Fixed { dt: 0.1 }).saving_every(10);
        assert_eq!(tc.save_every, Some(10));
    }

    #[test]
    fn default_save_every_is_none() {
        let tc = TimeConfiguration::new(1.0, StepControl::Fixed { dt: 0.1 });
        assert_eq!(tc.save_every, None);
    }

    // ── SolverConfiguration ───────────────────────────────────────────────────

    #[test]
    fn new_config_has_no_calculators() {
        let cfg = SolverConfiguration::new(
            TimeConfiguration::new(1.0, StepControl::Fixed { dt: 0.1 }),
            IntegratorKind::Euler,
        );
        assert!(cfg.calculators.is_empty());
    }

    #[test]
    fn with_calculator_adds_to_chain() {
        #[derive(Debug)]
        struct DummyCalc;
        impl RequiresContext for DummyCalc {
            fn required_variables(&self) -> Vec<ContextVariable> {
                vec![]
            }
        }
        impl crate::context::calculator::ContextCalculator for DummyCalc {
            fn provides(&self) -> ContextVariable {
                ContextVariable::Time
            }
            fn compute(
                &self,
                _: &ContextValue,
                ctx: &ComputeContext,
            ) -> Result<ContextValue, OxiflowError> {
                Ok(ContextValue::Scalar(ctx.time()))
            }
        }

        let cfg = SolverConfiguration::new(
            TimeConfiguration::new(1.0, StepControl::Fixed { dt: 0.1 }),
            IntegratorKind::RK4,
        )
        .with_calculator(Box::new(DummyCalc));

        assert_eq!(cfg.calculators.len(), 1);
        assert_eq!(cfg.integrator, IntegratorKind::RK4);
    }

    #[test]
    fn time_configuration_accessible() {
        let cfg = SolverConfiguration::new(
            TimeConfiguration::new(600.0, StepControl::Fixed { dt: 0.5 }),
            IntegratorKind::Euler,
        );
        assert_eq!(cfg.time.t_end, 600.0);
        assert_eq!(cfg.time.step_control.dt_initial(), 0.5);
    }
}
