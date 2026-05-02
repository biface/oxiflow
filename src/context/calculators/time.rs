//! # Module `context::calculators::time`
//!
//! Built-in calculators for temporal context variables.
//!
//! [`TimeCalculator`] and [`TimeStepCalculator`] are injected by the solver
//! before the user-defined calculator chain runs. They are registered here as
//! public types so that downstream models can inspect or test the chain without
//! coupling to solver internals.

use crate::context::calculator::ContextCalculator;
use crate::context::compute::ComputeContext;
use crate::context::error::OxiflowError;
use crate::context::value::ContextValue;
use crate::context::variable::ContextVariable;
use crate::model::traits::RequiresContext;

// ── TimeCalculator ────────────────────────────────────────────────────────────

/// Provides the current simulation time as `ContextVariable::Time`.
///
/// Priority 0 — runs first in every chain. No required variables.
///
/// # Examples
///
/// ```rust
/// use oxiflow::context::calculator::ContextCalculator;
/// use oxiflow::context::calculators::TimeCalculator;
/// use oxiflow::context::compute::ComputeContext;
/// use oxiflow::context::value::ContextValue;
/// use oxiflow::context::variable::ContextVariable;
///
/// let calc = TimeCalculator;
/// let ctx  = ComputeContext::new(3.14, 0.01);
///
/// assert_eq!(calc.provides(), ContextVariable::Time);
/// let val = calc.compute(&ContextValue::Scalar(0.0), &ctx).unwrap();
/// assert_eq!(val.as_scalar().unwrap(), 3.14);
/// ```
#[derive(Debug, Clone)]
pub struct TimeCalculator;

impl RequiresContext for TimeCalculator {
    fn required_variables(&self) -> Vec<ContextVariable> {
        vec![]
    }

    fn priority(&self) -> u32 {
        0
    }
}

impl ContextCalculator for TimeCalculator {
    fn provides(&self) -> ContextVariable {
        ContextVariable::Time
    }

    fn compute(
        &self,
        _state: &ContextValue,
        ctx: &ComputeContext,
    ) -> Result<ContextValue, OxiflowError> {
        Ok(ContextValue::Scalar(ctx.time()))
    }

    fn name(&self) -> &str {
        "time (built-in)"
    }
}

// ── TimeStepCalculator ────────────────────────────────────────────────────────

/// Provides the current time step as `ContextVariable::TimeStep`.
///
/// Priority 0 — runs first in every chain. No required variables.
///
/// # Examples
///
/// ```rust
/// use oxiflow::context::calculator::ContextCalculator;
/// use oxiflow::context::calculators::TimeStepCalculator;
/// use oxiflow::context::compute::ComputeContext;
/// use oxiflow::context::value::ContextValue;
/// use oxiflow::context::variable::ContextVariable;
///
/// let calc = TimeStepCalculator;
/// let ctx  = ComputeContext::new(0.0, 0.05);
///
/// assert_eq!(calc.provides(), ContextVariable::TimeStep);
/// let val = calc.compute(&ContextValue::Scalar(0.0), &ctx).unwrap();
/// assert_eq!(val.as_scalar().unwrap(), 0.05);
/// ```
#[derive(Debug, Clone)]
pub struct TimeStepCalculator;

impl RequiresContext for TimeStepCalculator {
    fn required_variables(&self) -> Vec<ContextVariable> {
        vec![]
    }

    fn priority(&self) -> u32 {
        0
    }
}

impl ContextCalculator for TimeStepCalculator {
    fn provides(&self) -> ContextVariable {
        ContextVariable::TimeStep
    }

    fn compute(
        &self,
        _state: &ContextValue,
        ctx: &ComputeContext,
    ) -> Result<ContextValue, OxiflowError> {
        Ok(ContextValue::Scalar(ctx.time_step()))
    }

    fn name(&self) -> &str {
        "time_step (built-in)"
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn ctx(t: f64, dt: f64) -> ComputeContext {
        ComputeContext::new(t, dt)
    }

    // ── TimeCalculator ────────────────────────────────────────────────────────

    #[test]
    fn time_calculator_provides_time() {
        assert_eq!(TimeCalculator.provides(), ContextVariable::Time);
    }

    #[test]
    fn time_calculator_returns_current_time() {
        let result = TimeCalculator
            .compute(&ContextValue::Scalar(0.0), &ctx(2.5, 0.01))
            .unwrap();
        assert_eq!(result.as_scalar().unwrap(), 2.5);
    }

    #[test]
    fn time_calculator_priority_is_zero() {
        assert_eq!(TimeCalculator.priority(), 0);
    }

    #[test]
    fn time_calculator_has_no_required_variables() {
        assert!(TimeCalculator.required_variables().is_empty());
    }

    #[test]
    fn time_calculator_name() {
        assert_eq!(TimeCalculator.name(), "time (built-in)");
    }

    // ── TimeStepCalculator ────────────────────────────────────────────────────

    #[test]
    fn timestep_calculator_provides_timestep() {
        assert_eq!(TimeStepCalculator.provides(), ContextVariable::TimeStep);
    }

    #[test]
    fn timestep_calculator_returns_current_dt() {
        let result = TimeStepCalculator
            .compute(&ContextValue::Scalar(0.0), &ctx(0.0, 0.05))
            .unwrap();
        assert_eq!(result.as_scalar().unwrap(), 0.05);
    }

    #[test]
    fn timestep_calculator_priority_is_zero() {
        assert_eq!(TimeStepCalculator.priority(), 0);
    }

    #[test]
    fn timestep_calculator_has_no_required_variables() {
        assert!(TimeStepCalculator.required_variables().is_empty());
    }

    #[test]
    fn timestep_calculator_name() {
        assert_eq!(TimeStepCalculator.name(), "time_step (built-in)");
    }

    // ── Object safety ─────────────────────────────────────────────────────────

    #[test]
    fn both_are_object_safe() {
        let calcs: Vec<Box<dyn ContextCalculator>> =
            vec![Box::new(TimeCalculator), Box::new(TimeStepCalculator)];
        assert_eq!(calcs[0].provides(), ContextVariable::Time);
        assert_eq!(calcs[1].provides(), ContextVariable::TimeStep);
    }
}
