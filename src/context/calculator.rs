//! # Module `context::calculator`
//!
//! Trait `ContextCalculator` — computes a single context variable (issue #32, DD-020).
//!
//! A calculator is the symmetric counterpart of `RequiresContext`: where a model
//! *declares* what it needs, a calculator *provides* what it computes.
//! The solver chains calculators in topological order (DD-009, J2) and feeds
//! results into `ComputeContext` before each time step.
//!
//! ## Placement
//!
//! Calculators live in `SolverConfiguration` — they are part of HOW the problem is
//! solved, not WHAT the problem is. `DiscreteOperator` (INV-2, J4b) is an
//! implementation detail inside spatial calculators, never exposed at configuration level.

use crate::context::compute::ComputeContext;
use crate::context::error::OxiflowError;
use crate::context::value::ContextValue;
use crate::context::variable::ContextVariable;
use crate::model::traits::RequiresContext;

/// Computes one context variable and injects it into `ComputeContext`.
///
/// The solver calls calculators in topological order — a calculator that
/// `depends_on()` variable X is guaranteed to run after the calculator that
/// `provides()` X. Within a dependency tier, execution order is determined
/// by `priority()` (inherited from `RequiresContext`).
///
/// # Contract
///
/// - `provides()` must return a stable `ContextVariable` — the same value
///   across calls for a given calculator instance.
/// - `compute()` receives the current field state and a **partially populated**
///   `ComputeContext` (variables with lower priority already resolved).
/// - `compute()` must not modify `ctx` directly — the solver inserts the result.
///
/// # DiscreteOperator (INV-2, J4b)
///
/// Spatial calculators (gradient, Laplacian, flux) may internally hold a
/// `DiscreteOperator` implementation. This is an implementation detail of the
/// calculator — `DiscreteOperator` is never exposed in `SolverConfiguration`.
///
/// # Examples
///
/// ```rust
/// use oxiflow::context::calculator::ContextCalculator;
/// use oxiflow::context::variable::ContextVariable;
/// use oxiflow::context::value::ContextValue;
/// use oxiflow::context::compute::ComputeContext;
/// use oxiflow::context::error::OxiflowError;
/// use oxiflow::model::traits::RequiresContext;
///
/// #[derive(Debug)]
/// struct TimeCalculator;
///
/// impl RequiresContext for TimeCalculator {
///     fn required_variables(&self) -> Vec<ContextVariable> { vec![] }
///     fn priority(&self) -> u32 { 0 }
/// }
///
/// impl ContextCalculator for TimeCalculator {
///     fn provides(&self) -> ContextVariable { ContextVariable::Time }
///     fn compute(&self, _state: &ContextValue, ctx: &ComputeContext)
///         -> Result<ContextValue, OxiflowError>
///     {
///         Ok(ContextValue::Scalar(ctx.time()))
///     }
///     fn name(&self) -> &str { "time" }
/// }
/// ```
pub trait ContextCalculator: RequiresContext + Send + Sync + std::fmt::Debug {
    /// The context variable this calculator produces.
    ///
    /// Must be stable across calls. The solver uses this to match calculators
    /// against `RequiresContext::required_variables()` declarations.
    fn provides(&self) -> ContextVariable;

    /// Computes the value for `provides()` given the current field state and
    /// the partially populated context.
    ///
    /// Variables declared in `depends_on()` are guaranteed to be present in
    /// `ctx` when this method is called.
    fn compute(
        &self,
        state: &ContextValue,
        ctx: &ComputeContext,
    ) -> Result<ContextValue, OxiflowError>;

    /// Human-readable name for logging and error messages.
    fn name(&self) -> &str {
        "unnamed calculator"
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── Fixtures ──────────────────────────────────────────────────────────────

    /// Returns the current time as a Scalar — priority 0.
    #[derive(Debug)]
    struct TimeCalculator;

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
            "time"
        }
    }

    /// Returns a fixed external scalar — depends on nothing.
    #[derive(Debug)]
    struct ConstantCalculator {
        var: ContextVariable,
        value: f64,
    }

    impl RequiresContext for ConstantCalculator {
        fn required_variables(&self) -> Vec<ContextVariable> {
            vec![]
        }
    }

    impl ContextCalculator for ConstantCalculator {
        fn provides(&self) -> ContextVariable {
            self.var.clone()
        }
        fn compute(
            &self,
            _state: &ContextValue,
            _ctx: &ComputeContext,
        ) -> Result<ContextValue, OxiflowError> {
            Ok(ContextValue::Scalar(self.value))
        }
    }

    /// Depends on Time — must run after TimeCalculator.
    #[derive(Debug)]
    struct TimeDependentCalculator;

    impl RequiresContext for TimeDependentCalculator {
        fn required_variables(&self) -> Vec<ContextVariable> {
            vec![ContextVariable::Time]
        }
        fn depends_on(&self) -> Vec<ContextVariable> {
            vec![ContextVariable::Time]
        }
        fn priority(&self) -> u32 {
            50
        }
    }

    impl ContextCalculator for TimeDependentCalculator {
        fn provides(&self) -> ContextVariable {
            ContextVariable::External {
                name: "double_time",
            }
        }
        fn compute(
            &self,
            _state: &ContextValue,
            ctx: &ComputeContext,
        ) -> Result<ContextValue, OxiflowError> {
            let t = ctx.time();
            Ok(ContextValue::Scalar(t * 2.0))
        }
    }

    // ── provides() ───────────────────────────────────────────────────────────

    #[test]
    fn provides_returns_declared_variable() {
        assert_eq!(TimeCalculator.provides(), ContextVariable::Time);
    }

    #[test]
    fn provides_is_stable_across_calls() {
        let calc = ConstantCalculator {
            var: ContextVariable::TimeStep,
            value: 0.01,
        };
        assert_eq!(calc.provides(), calc.provides());
    }

    // ── compute() ────────────────────────────────────────────────────────────

    #[test]
    fn time_calculator_returns_current_time() {
        let ctx = ComputeContext::new(3.14, 0.01);
        let result = TimeCalculator
            .compute(&ContextValue::Scalar(0.0), &ctx)
            .unwrap();
        assert_eq!(result.as_scalar().unwrap(), 3.14);
    }

    #[test]
    fn constant_calculator_returns_fixed_value() {
        let calc = ConstantCalculator {
            var: ContextVariable::External { name: "D_ax" },
            value: 1.5e-4,
        };
        let ctx = ComputeContext::new(0.0, 0.01);
        let result = calc.compute(&ContextValue::Scalar(0.0), &ctx).unwrap();
        assert!((result.as_scalar().unwrap() - 1.5e-4).abs() < 1e-12);
    }

    #[test]
    fn time_dependent_calculator_reads_from_ctx() {
        let mut ctx = ComputeContext::new(5.0, 0.01);
        ctx.insert(ContextVariable::Time, ContextValue::Scalar(5.0));
        let result = TimeDependentCalculator
            .compute(&ContextValue::Scalar(0.0), &ctx)
            .unwrap();
        assert_eq!(result.as_scalar().unwrap(), 10.0);
    }

    // ── name() ───────────────────────────────────────────────────────────────

    #[test]
    fn name_returns_provided_string() {
        assert_eq!(TimeCalculator.name(), "time");
    }

    #[test]
    fn default_name_is_unnamed() {
        let calc = ConstantCalculator {
            var: ContextVariable::TimeStep,
            value: 0.01,
        };
        assert_eq!(calc.name(), "unnamed calculator");
    }

    // ── RequiresContext integration ───────────────────────────────────────────

    #[test]
    fn time_calculator_has_no_requirements() {
        assert!(TimeCalculator.required_variables().is_empty());
        assert_eq!(TimeCalculator.priority(), 0);
    }

    #[test]
    fn time_dependent_requires_time() {
        let calc = TimeDependentCalculator;
        assert!(calc.required_variables().contains(&ContextVariable::Time));
        assert!(calc.depends_on().contains(&ContextVariable::Time));
        assert_eq!(calc.priority(), 50);
    }

    // ── Object safety ─────────────────────────────────────────────────────────

    #[test]
    fn trait_is_object_safe() {
        let calcs: Vec<Box<dyn ContextCalculator>> = vec![
            Box::new(TimeCalculator),
            Box::new(ConstantCalculator {
                var: ContextVariable::TimeStep,
                value: 0.01,
            }),
        ];
        assert_eq!(calcs[0].provides(), ContextVariable::Time);
        assert_eq!(calcs[1].provides(), ContextVariable::TimeStep);
    }
}
