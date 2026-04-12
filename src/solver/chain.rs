//! # Module `solver::chain`
//!
//! Calculator chain — validation and priority-based ordering (issue #33).
//!
//! ## Responsibility
//!
//! `build_calculator_chain` verifies that every context variable required by the
//! scenario has a corresponding calculator in `SolverConfiguration`, then returns
//! the calculators sorted by ascending priority for execution.
//!
//! ## Ordering strategy (J1)
//!
//! At J1, ordering is by `priority()` (ascending). Calculators with lower
//! priority numbers run first — reserved ranges:
//! - **0–49**: system variables (Time, TimeStep) — injected directly by the solver
//! - **50–99**: external data providers
//! - **100+**: derived quantities (default)
//!
//! Full topological ordering via Kahn's algorithm is reserved for J2 (DD-009),
//! when calculators may declare `depends_on()` dependencies on each other.
//!
//! ## Built-in variables
//!
//! `Time` and `TimeStep` are always available in `ComputeContext` — the solver
//! injects them directly via `ComputeContext::new(t, dt)` before running the
//! chain. No calculator is needed for these variables.

use crate::context::calculator::ContextCalculator;
use crate::context::error::OxiflowError;
use crate::context::variable::ContextVariable;

/// Built-in variables provided directly by the solver — no calculator needed.
///
/// `Time` and `TimeStep` are injected via `ComputeContext::new(t, dt)` before
/// the calculator chain runs. Declaring them as requirements is valid; checking
/// them against the calculator list would be a false negative.
const BUILTIN_VARIABLES: &[ContextVariable] = &[ContextVariable::Time, ContextVariable::TimeStep];

/// Validates requirements against provided calculators and returns an
/// execution-ordered chain.
///
/// # Validation
///
/// Every variable in `requirements` must be either:
/// - a built-in variable (`Time`, `TimeStep`), or
/// - covered by exactly one calculator via `provides()`.
///
/// A variable covered by multiple calculators is accepted — the last one
/// in priority order wins (consistent with `ComputeContext::insert` overwrite).
///
/// # Ordering (J1)
///
/// Calculators are sorted by ascending `priority()`. Within the same priority,
/// original registration order is preserved (stable sort).
///
/// # Errors
///
/// Returns `OxiflowError::MissingCalculator` for the first uncovered
/// non-builtin required variable.
///
/// # Examples
///
/// ```rust
/// use oxiflow::solver::chain::build_calculator_chain;
/// use oxiflow::context::variable::ContextVariable;
/// use oxiflow::context::value::ContextValue;
/// use oxiflow::context::compute::ComputeContext;
/// use oxiflow::context::error::OxiflowError;
/// use oxiflow::context::calculator::ContextCalculator;
/// use oxiflow::model::traits::RequiresContext;
///
/// #[derive(Debug)]
/// struct TimeCalc;
/// impl RequiresContext for TimeCalc {
///     fn required_variables(&self) -> Vec<ContextVariable> { vec![] }
///     fn priority(&self) -> u32 { 0 }
/// }
/// impl ContextCalculator for TimeCalc {
///     fn provides(&self) -> ContextVariable { ContextVariable::Time }
///     fn compute(&self, _: &ContextValue, ctx: &ComputeContext)
///         -> Result<ContextValue, OxiflowError>
///     { Ok(ContextValue::Scalar(ctx.time())) }
/// }
///
/// let requirements = vec![ContextVariable::Time];
/// let calculators: Vec<Box<dyn ContextCalculator>> = vec![Box::new(TimeCalc)];
/// let chain = build_calculator_chain(&requirements, &calculators).unwrap();
/// assert_eq!(chain.len(), 1);
/// ```
pub fn build_calculator_chain<'a>(
    requirements: &[ContextVariable],
    calculators: &'a [Box<dyn ContextCalculator>],
) -> Result<Vec<&'a dyn ContextCalculator>, OxiflowError> {
    // Check every requirement is covered
    for req in requirements {
        if is_builtin(req) {
            continue;
        }
        let covered = calculators.iter().any(|c| &c.provides() == req);
        if !covered {
            return Err(OxiflowError::MissingCalculator(req.clone()));
        }
    }

    // Sort by priority (stable — preserves registration order within same priority)
    let mut chain: Vec<&dyn ContextCalculator> = calculators.iter().map(|c| c.as_ref()).collect();
    chain.sort_by_key(|c| c.priority());

    Ok(chain)
}

fn is_builtin(var: &ContextVariable) -> bool {
    BUILTIN_VARIABLES.contains(var)
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::context::compute::ComputeContext;
    use crate::context::value::ContextValue;
    use crate::model::traits::RequiresContext;

    // ── Fixtures ──────────────────────────────────────────────────────────────

    #[derive(Debug)]
    struct NamedCalc {
        provides: ContextVariable,
        priority: u32,
    }

    impl RequiresContext for NamedCalc {
        fn required_variables(&self) -> Vec<ContextVariable> {
            vec![]
        }
        fn priority(&self) -> u32 {
            self.priority
        }
    }

    impl ContextCalculator for NamedCalc {
        fn provides(&self) -> ContextVariable {
            self.provides.clone()
        }
        fn compute(
            &self,
            _state: &ContextValue,
            ctx: &ComputeContext,
        ) -> Result<ContextValue, OxiflowError> {
            Ok(ContextValue::Scalar(ctx.time()))
        }
    }

    fn make_calc(provides: ContextVariable, priority: u32) -> Box<dyn ContextCalculator> {
        Box::new(NamedCalc { provides, priority })
    }

    // ── Validation ────────────────────────────────────────────────────────────

    #[test]
    fn empty_requirements_with_no_calculators_succeeds() {
        let chain = build_calculator_chain(&[], &[]).unwrap();
        assert!(chain.is_empty());
    }

    #[test]
    fn builtin_time_requires_no_calculator() {
        let requirements = vec![ContextVariable::Time, ContextVariable::TimeStep];
        let chain = build_calculator_chain(&requirements, &[]).unwrap();
        assert!(chain.is_empty());
    }

    #[test]
    fn satisfied_requirement_succeeds() {
        let requirements = vec![ContextVariable::External { name: "D_ax" }];
        let calcs = vec![make_calc(ContextVariable::External { name: "D_ax" }, 100)];
        let chain = build_calculator_chain(&requirements, &calcs).unwrap();
        assert_eq!(chain.len(), 1);
    }

    #[test]
    fn missing_calculator_returns_error() {
        let requirements = vec![ContextVariable::External { name: "missing" }];
        let err = build_calculator_chain(&requirements, &[]).unwrap_err();
        assert!(matches!(err, OxiflowError::MissingCalculator(_)));
    }

    #[test]
    fn missing_calculator_error_names_the_variable() {
        let var = ContextVariable::SpatialGradient {
            dimension: 0,
            component: None,
        };
        let requirements = vec![var.clone()];
        let err = build_calculator_chain(&requirements, &[]).unwrap_err();
        assert!(matches!(err, OxiflowError::MissingCalculator(v) if v == var));
    }

    #[test]
    fn duplicate_calculator_for_same_variable_is_accepted() {
        let var = ContextVariable::External { name: "v" };
        let requirements = vec![var.clone()];
        let calcs = vec![make_calc(var.clone(), 100), make_calc(var.clone(), 50)];
        assert!(build_calculator_chain(&requirements, &calcs).is_ok());
    }

    #[test]
    fn extra_calculators_beyond_requirements_are_included() {
        // Calculators for unrequired variables are still executed
        let requirements = vec![ContextVariable::External { name: "a" }];
        let calcs = vec![
            make_calc(ContextVariable::External { name: "a" }, 100),
            make_calc(ContextVariable::External { name: "b" }, 100),
        ];
        let chain = build_calculator_chain(&requirements, &calcs).unwrap();
        assert_eq!(chain.len(), 2);
    }

    // ── Ordering ──────────────────────────────────────────────────────────────

    #[test]
    fn chain_sorted_by_ascending_priority() {
        let calcs = vec![
            make_calc(ContextVariable::External { name: "c" }, 200),
            make_calc(ContextVariable::External { name: "a" }, 50),
            make_calc(ContextVariable::External { name: "b" }, 100),
        ];
        let chain = build_calculator_chain(&[], &calcs).unwrap();
        assert_eq!(chain[0].priority(), 50);
        assert_eq!(chain[1].priority(), 100);
        assert_eq!(chain[2].priority(), 200);
    }

    #[test]
    fn stable_sort_preserves_registration_order_within_same_priority() {
        let calcs = vec![
            make_calc(ContextVariable::External { name: "first" }, 100),
            make_calc(ContextVariable::External { name: "second" }, 100),
        ];
        let chain = build_calculator_chain(&[], &calcs).unwrap();
        assert_eq!(
            chain[0].provides(),
            ContextVariable::External { name: "first" }
        );
        assert_eq!(
            chain[1].provides(),
            ContextVariable::External { name: "second" }
        );
    }

    #[test]
    fn mixed_builtin_and_user_requirements() {
        let requirements = vec![
            ContextVariable::Time,
            ContextVariable::External { name: "D_ax" },
        ];
        let calcs = vec![make_calc(ContextVariable::External { name: "D_ax" }, 100)];
        let chain = build_calculator_chain(&requirements, &calcs).unwrap();
        assert_eq!(chain.len(), 1);
    }
}
