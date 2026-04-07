//! # Module `model::traits`
//!
//! Core traits for physical model components (issue #29, DD-005).
//!
//! ## `RequiresContext`
//!
//! Standalone trait implemented by any component that needs context variables
//! during computation: physical models, boundary conditions (J2), source terms,
//! coupling operators (J3).
//!
//! `required_variables()` is **mandatory** — every implementor must declare its
//! context needs explicitly. Silent omissions are a compiler error, not a runtime
//! surprise. This enforces the "declarative before implicit" principle (DD-005).

use crate::context::variable::ContextVariable;

/// Declares context variable requirements for any engine component.
///
/// Implemented by physical models, boundary conditions (J2), source terms, and
/// coupling operators (J3). The solver aggregates declarations from all components
/// before solving and guarantees every required variable is available.
///
/// # Mandatory method
///
/// `required_variables()` has no default. Every implementor must declare its
/// required context variables explicitly — even if that declaration is `vec![]`.
///
/// # Examples
///
/// ```rust
/// use oxiflow::model::traits::RequiresContext;
/// use oxiflow::context::variable::ContextVariable;
///
/// struct ChromatographyModel;
///
/// impl RequiresContext for ChromatographyModel {
///     fn required_variables(&self) -> Vec<ContextVariable> {
///         vec![
///             ContextVariable::Time,
///             ContextVariable::SpatialGradient { dimension: 0 },
///         ]
///     }
/// }
///
/// struct PureAdvection;
///
/// impl RequiresContext for PureAdvection {
///     fn required_variables(&self) -> Vec<ContextVariable> {
///         vec![]
///     }
/// }
/// ```
pub trait RequiresContext {
    /// Returns the context variables this component **must** have available.
    ///
    /// The solver raises `OxiflowError::MissingCalculator` before solving if
    /// any required variable has no registered calculator.
    ///
    /// No default implementation — every component must declare explicitly.
    fn required_variables(&self) -> Vec<ContextVariable>;

    /// Returns context variables this component uses if available, but can work
    /// without. The solver provides them when calculators are registered.
    ///
    /// Default: no optional variables.
    fn optional_variables(&self) -> Vec<ContextVariable> {
        vec![]
    }

    /// Returns variables this component's calculator depends on.
    ///
    /// Used by the solver to build the topological execution order (J2, DD-009).
    /// Declaring dependencies prevents silent ordering bugs when calculators
    /// depend on each other's output.
    ///
    /// Default: no dependencies.
    fn depends_on(&self) -> Vec<ContextVariable> {
        vec![]
    }

    /// Execution priority within the calculator chain.
    ///
    /// Lower values run first. Reserved values:
    /// - `0`   — system variables (Time, TimeStep)
    /// - `50`  — external data providers
    /// - `100` — default for derived quantities
    ///
    /// Used as fallback when no topological ordering is declared via `depends_on()`.
    fn priority(&self) -> u32 {
        100
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── Test fixtures ─────────────────────────────────────────────────────────

    struct FullModel;
    impl RequiresContext for FullModel {
        fn required_variables(&self) -> Vec<ContextVariable> {
            vec![
                ContextVariable::Time,
                ContextVariable::SpatialGradient { dimension: 0 },
            ]
        }
        fn optional_variables(&self) -> Vec<ContextVariable> {
            vec![ContextVariable::External { name: "T_amb" }]
        }
        fn depends_on(&self) -> Vec<ContextVariable> {
            vec![ContextVariable::Time]
        }
        fn priority(&self) -> u32 {
            200
        }
    }

    struct MinimalModel;
    impl RequiresContext for MinimalModel {
        fn required_variables(&self) -> Vec<ContextVariable> {
            vec![]
        }
    }

    struct DefaultsModel;
    impl RequiresContext for DefaultsModel {
        fn required_variables(&self) -> Vec<ContextVariable> {
            vec![ContextVariable::TimeStep]
        }
    }

    // ── required_variables ────────────────────────────────────────────────────

    #[test]
    fn required_variables_returns_declared_variables() {
        let vars = FullModel.required_variables();
        assert_eq!(vars.len(), 2);
        assert!(vars.contains(&ContextVariable::Time));
        assert!(vars.contains(&ContextVariable::SpatialGradient { dimension: 0 }));
    }

    #[test]
    fn required_variables_empty_vec_is_valid() {
        assert!(MinimalModel.required_variables().is_empty());
    }

    #[test]
    fn required_variables_single_variable() {
        let vars = DefaultsModel.required_variables();
        assert_eq!(vars.len(), 1);
        assert_eq!(vars[0], ContextVariable::TimeStep);
    }

    // ── optional_variables ────────────────────────────────────────────────────

    #[test]
    fn optional_variables_returns_declared() {
        let vars = FullModel.optional_variables();
        assert_eq!(vars.len(), 1);
        assert_eq!(vars[0], ContextVariable::External { name: "T_amb" });
    }

    #[test]
    fn optional_variables_default_is_empty() {
        assert!(MinimalModel.optional_variables().is_empty());
        assert!(DefaultsModel.optional_variables().is_empty());
    }

    // ── depends_on ────────────────────────────────────────────────────────────

    #[test]
    fn depends_on_returns_declared_dependencies() {
        let deps = FullModel.depends_on();
        assert_eq!(deps.len(), 1);
        assert_eq!(deps[0], ContextVariable::Time);
    }

    #[test]
    fn depends_on_default_is_empty() {
        assert!(MinimalModel.depends_on().is_empty());
        assert!(DefaultsModel.depends_on().is_empty());
    }

    // ── priority ──────────────────────────────────────────────────────────────

    #[test]
    fn priority_returns_custom_value() {
        assert_eq!(FullModel.priority(), 200);
    }

    #[test]
    fn priority_default_is_100() {
        assert_eq!(MinimalModel.priority(), 100);
        assert_eq!(DefaultsModel.priority(), 100);
    }

    // ── Object safety ─────────────────────────────────────────────────────────

    #[test]
    fn trait_is_object_safe() {
        let model: Box<dyn RequiresContext> = Box::new(FullModel);
        assert_eq!(model.required_variables().len(), 2);
        assert_eq!(model.priority(), 200);
    }

    #[test]
    fn solver_can_aggregate_all_required_variables() {
        let components: Vec<Box<dyn RequiresContext>> = vec![
            Box::new(FullModel),
            Box::new(MinimalModel),
            Box::new(DefaultsModel),
        ];
        let all_required: Vec<ContextVariable> = components
            .iter()
            .flat_map(|c| c.required_variables())
            .collect();
        // FullModel(2) + MinimalModel(0) + DefaultsModel(1) = 3
        assert_eq!(all_required.len(), 3);
    }

    // ── Send + Sync ───────────────────────────────────────────────────────────

    #[test]
    fn implementors_are_send_and_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<FullModel>();
        assert_send_sync::<MinimalModel>();
        assert_send_sync::<DefaultsModel>();
    }
}
