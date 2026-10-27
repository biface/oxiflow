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

use std::sync::Arc;

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
/// - The solver calls [`compute_shared()`](Self::compute_shared), which defaults
///   to `compute()`. A calculator that keeps its value between evaluations
///   overrides it to hand that value over without a copy (DT-068, #195).
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

    /// Computes the value for `provides()` as a shared allocation.
    ///
    /// This is the method the solver calls at each derivative evaluation.
    /// The default wraps the result of [`compute()`](Self::compute) in an
    /// `Arc`: the value is moved, not copied, at the cost of one `Arc`
    /// allocation per evaluation.
    ///
    /// # When to override
    ///
    /// Override it when the calculator keeps a value from one evaluation to
    /// the next (for instance a coefficient field evaluated once on the mesh).
    /// Returning a clone of the kept `Arc` hands over the same allocation to
    /// every evaluation, where `compute()` would have to copy the whole value.
    ///
    /// # Contract
    ///
    /// An override returns the same value as `compute()` for the same `state`
    /// and `ctx`. `compute()` stays required, and remains the method to call
    /// outside the solver.
    ///
    /// This method is not a parallel-dispatch capability and carries no
    /// configuration, so it is consistent with P3 of DA-048 (#140): an
    /// implementer that ignores it is unaffected.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use std::sync::Arc;
    ///
    /// use nalgebra::DVector;
    /// use oxiflow::context::{
    ///     ComputeContext, ContextCalculator, ContextValue, ContextVariable, OxiflowError,
    /// };
    /// use oxiflow::model::RequiresContext;
    ///
    /// /// Coefficient field evaluated once, then shared with every evaluation.
    /// #[derive(Debug)]
    /// struct FixedCoefficient {
    ///     field: Arc<ContextValue>,
    /// }
    ///
    /// impl RequiresContext for FixedCoefficient {
    ///     fn required_variables(&self) -> Vec<ContextVariable> { vec![] }
    /// }
    ///
    /// impl ContextCalculator for FixedCoefficient {
    ///     fn provides(&self) -> ContextVariable {
    ///         ContextVariable::External { name: "c".into() }
    ///     }
    ///     fn compute(&self, _state: &ContextValue, _ctx: &ComputeContext)
    ///         -> Result<ContextValue, OxiflowError>
    ///     {
    ///         Ok(self.field.as_ref().clone())
    ///     }
    ///     fn compute_shared(&self, _state: &ContextValue, _ctx: &ComputeContext)
    ///         -> Result<Arc<ContextValue>, OxiflowError>
    ///     {
    ///         Ok(Arc::clone(&self.field))
    ///     }
    /// }
    ///
    /// let calc = FixedCoefficient {
    ///     field: Arc::new(ContextValue::ScalarField(DVector::from_element(5, 0.5))),
    /// };
    /// let state = ContextValue::ScalarField(DVector::zeros(5));
    /// let ctx = ComputeContext::new(0.0, 0.01);
    ///
    /// let first = calc.compute_shared(&state, &ctx)?;
    /// let second = calc.compute_shared(&state, &ctx)?;
    /// assert!(Arc::ptr_eq(&first, &second));
    /// assert_eq!(*first, calc.compute(&state, &ctx)?);
    /// # Ok::<(), OxiflowError>(())
    /// ```
    fn compute_shared(
        &self,
        state: &ContextValue,
        ctx: &ComputeContext,
    ) -> Result<Arc<ContextValue>, OxiflowError> {
        self.compute(state, ctx).map(Arc::new)
    }

    /// Human-readable name for logging and error messages.
    fn name(&self) -> &str {
        "unnamed calculator"
    }
}

// ── Test fixture shared across the crate ──────────────────────────────────────

/// Calculator that keeps one value and hands over the same allocation at every
/// call of `compute_shared` — the case DT-068 (#195) is about.
///
/// Shared by the unit tests of this module and of `solver::methods`.
#[cfg(test)]
#[derive(Debug)]
pub(crate) struct CachedValueCalculator {
    /// Variable provided.
    pub(crate) var: ContextVariable,
    /// Value kept between calls.
    pub(crate) value: Arc<ContextValue>,
}

#[cfg(test)]
impl RequiresContext for CachedValueCalculator {
    fn required_variables(&self) -> Vec<ContextVariable> {
        vec![]
    }
}

#[cfg(test)]
impl ContextCalculator for CachedValueCalculator {
    fn provides(&self) -> ContextVariable {
        self.var.clone()
    }
    fn compute(
        &self,
        _state: &ContextValue,
        _ctx: &ComputeContext,
    ) -> Result<ContextValue, OxiflowError> {
        Ok(self.value.as_ref().clone())
    }
    fn compute_shared(
        &self,
        _state: &ContextValue,
        _ctx: &ComputeContext,
    ) -> Result<Arc<ContextValue>, OxiflowError> {
        Ok(Arc::clone(&self.value))
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
                name: "double_time".into(),
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
        let ctx = ComputeContext::new(std::f64::consts::PI, 0.01);
        let result = TimeCalculator
            .compute(&ContextValue::Scalar(0.0), &ctx)
            .unwrap();
        assert_eq!(result.as_scalar().unwrap(), std::f64::consts::PI);
    }

    #[test]
    fn constant_calculator_returns_fixed_value() {
        let calc = ConstantCalculator {
            var: ContextVariable::External {
                name: "D_ax".into(),
            },
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

    // ── compute_shared() ─────────────────────────────────────────────────────

    #[test]
    fn default_compute_shared_returns_the_value_of_compute() {
        let ctx = ComputeContext::new(2.5, 0.01);
        let state = ContextValue::Scalar(0.0);
        let shared = TimeCalculator.compute_shared(&state, &ctx).unwrap();
        assert_eq!(*shared, TimeCalculator.compute(&state, &ctx).unwrap());
    }

    #[test]
    fn default_compute_shared_propagates_errors_of_compute() {
        #[derive(Debug)]
        struct FailingCalculator;
        impl RequiresContext for FailingCalculator {
            fn required_variables(&self) -> Vec<ContextVariable> {
                vec![]
            }
        }
        impl ContextCalculator for FailingCalculator {
            fn provides(&self) -> ContextVariable {
                ContextVariable::Time
            }
            fn compute(
                &self,
                _state: &ContextValue,
                ctx: &ComputeContext,
            ) -> Result<ContextValue, OxiflowError> {
                ctx.scalar(ContextVariable::TimeStep)
                    .map(ContextValue::Scalar)
            }
        }
        let ctx = ComputeContext::new(0.0, 0.01);
        let err = FailingCalculator
            .compute_shared(&ContextValue::Scalar(0.0), &ctx)
            .unwrap_err();
        assert!(matches!(err, OxiflowError::MissingCalculator(_)));
    }

    #[test]
    fn overridden_compute_shared_hands_over_the_same_allocation() {
        let calc = CachedValueCalculator {
            var: ContextVariable::External { name: "c".into() },
            value: Arc::new(ContextValue::ScalarField(nalgebra::DVector::from_element(
                8, 1.5,
            ))),
        };
        let ctx = ComputeContext::new(0.0, 0.01);
        let state = ContextValue::Scalar(0.0);
        let first = calc.compute_shared(&state, &ctx).unwrap();
        let second = calc.compute_shared(&state, &ctx).unwrap();
        assert!(Arc::ptr_eq(&first, &second));
        assert!(Arc::ptr_eq(&first, &calc.value));
        // Same value as compute(), as the contract requires.
        assert_eq!(*first, calc.compute(&state, &ctx).unwrap());
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
