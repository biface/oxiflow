//! # Module `context::error`
//!
//! Main error type of the oxiflow engine (DD-004, issue #28).
//!
//! `OxiflowError` is a typed enum built with `thiserror` covering all engine
//! failure cases. Every public function returns `Result<_, OxiflowError>` —
//! never `Result<_, String>`.
//!
//! ## Design rationale
//!
//! chrom-rs used `Result<_, String>` throughout. String errors are not
//! matchable programmatically: downstream code cannot distinguish a missing
//! calculator from a solver divergence without parsing strings. `OxiflowError`
//! makes every failure case a first-class type.

use crate::context::variable::ContextVariable;

/// Typed error enum for all oxiflow engine failures.
///
/// Each variant corresponds to a distinct, matchable failure mode.
/// The `source` field in `ComputationFailed` preserves the original error
/// for error-chain display while keeping the variant matchable.
///
/// # Examples
///
/// ```rust
/// use oxiflow::context::error::OxiflowError;
/// use oxiflow::context::variable::ContextVariable;
///
/// let err = OxiflowError::MissingCalculator(ContextVariable::Time);
/// assert!(matches!(err, OxiflowError::MissingCalculator(ContextVariable::Time)));
///
/// let err = OxiflowError::TypeMismatch {
///     expected: "Scalar",
///     actual:   "Vector",
/// };
/// assert!(matches!(err, OxiflowError::TypeMismatch { .. }));
/// ```
#[derive(Debug, thiserror::Error)]
pub enum OxiflowError {
    /// No calculator registered for the required variable.
    #[error("missing calculator for variable: {0}")]
    MissingCalculator(ContextVariable),

    /// A calculator returned an error while computing a variable.
    #[error("computation failed for {variable}: {source}")]
    ComputationFailed {
        variable: ContextVariable,
        #[source]
        source: Box<dyn std::error::Error + Send + Sync>,
    },

    /// A circular dependency was detected among calculators.
    #[error("circular dependency detected involving: {0}")]
    CircularDependency(ContextVariable),

    /// A context accessor was called with the wrong `ContextValue` variant.
    #[error("type mismatch: expected {expected}, actual {actual}")]
    TypeMismatch {
        expected: &'static str,
        actual: &'static str,
    },

    /// The mesh or domain configuration is invalid.
    #[error("invalid domain: {0}")]
    InvalidDomain(String),

    /// An external data source returned an error or is unavailable.
    #[error("external data error: {0}")]
    ExternalData(String),

    /// The solver produced a non-finite state and cannot continue.
    #[error("solver divergence at t={time:.4e}: {reason}")]
    SolverDivergence { time: f64, reason: String },
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn missing_calculator_matches_variable() {
        let err = OxiflowError::MissingCalculator(ContextVariable::Time);
        assert!(matches!(
            err,
            OxiflowError::MissingCalculator(ContextVariable::Time)
        ));
    }

    #[test]
    fn missing_calculator_display_contains_variable() {
        let err = OxiflowError::MissingCalculator(ContextVariable::TimeStep);
        assert!(err.to_string().contains("TimeStep"));
    }

    #[test]
    fn computation_failed_is_matchable() {
        let source: Box<dyn std::error::Error + Send + Sync> =
            Box::new(std::io::Error::other("calculator error"));
        let err = OxiflowError::ComputationFailed {
            variable: ContextVariable::SpatialGradient {
                dimension: 0,
                component: None,
            },
            source,
        };
        assert!(matches!(err, OxiflowError::ComputationFailed { .. }));
    }

    #[test]
    fn computation_failed_display_contains_variable_and_source() {
        let source: Box<dyn std::error::Error + Send + Sync> =
            Box::new(std::io::Error::other("overflow"));
        let err = OxiflowError::ComputationFailed {
            variable: ContextVariable::SpatialGradient {
                dimension: 1,
                component: None,
            },
            source,
        };
        let msg = err.to_string();
        assert!(msg.contains("SpatialGradient"));
        assert!(msg.contains("overflow"));
    }

    #[test]
    fn circular_dependency_matches_variable() {
        let err = OxiflowError::CircularDependency(ContextVariable::External { name: "flux" });
        assert!(matches!(err, OxiflowError::CircularDependency(_)));
    }

    #[test]
    fn circular_dependency_display_contains_variable() {
        let err = OxiflowError::CircularDependency(ContextVariable::Time);
        assert!(err.to_string().contains("Time"));
    }

    #[test]
    fn type_mismatch_fields_are_accessible() {
        let err = OxiflowError::TypeMismatch {
            expected: "Scalar",
            actual: "Vector",
        };
        assert!(matches!(
            err,
            OxiflowError::TypeMismatch {
                expected: "Scalar",
                actual: "Vector"
            }
        ));
    }

    #[test]
    fn type_mismatch_display_contains_both_types() {
        let err = OxiflowError::TypeMismatch {
            expected: "Matrix",
            actual: "ScalarField",
        };
        let msg = err.to_string();
        assert!(msg.contains("Matrix"));
        assert!(msg.contains("ScalarField"));
    }

    #[test]
    fn invalid_domain_display_contains_reason() {
        let err = OxiflowError::InvalidDomain("n_points must be > 1".into());
        assert!(err.to_string().contains("n_points must be > 1"));
    }

    #[test]
    fn external_data_display_contains_reason() {
        let err = OxiflowError::ExternalData("file not found".into());
        assert!(err.to_string().contains("file not found"));
    }

    #[test]
    fn solver_divergence_display_contains_time_and_reason() {
        let err = OxiflowError::SolverDivergence {
            time: 1.23e-4,
            reason: "NaN detected in state vector".into(),
        };
        let msg = err.to_string();
        assert!(msg.contains("NaN detected"));
        assert!(msg.contains("1.23"));
    }

    #[test]
    fn solver_divergence_time_formatted_scientific() {
        let err = OxiflowError::SolverDivergence {
            time: 0.001,
            reason: "diverged".into(),
        };
        assert!(err.to_string().contains("e-"));
    }

    #[test]
    fn all_variants_implement_debug() {
        let variants: Vec<Box<dyn std::fmt::Debug>> = vec![
            Box::new(OxiflowError::MissingCalculator(ContextVariable::Time)),
            Box::new(OxiflowError::CircularDependency(ContextVariable::TimeStep)),
            Box::new(OxiflowError::TypeMismatch {
                expected: "Scalar",
                actual: "Boolean",
            }),
            Box::new(OxiflowError::InvalidDomain("test".into())),
            Box::new(OxiflowError::ExternalData("test".into())),
            Box::new(OxiflowError::SolverDivergence {
                time: 0.0,
                reason: "test".into(),
            }),
        ];
        for v in &variants {
            assert!(!format!("{:?}", v).is_empty());
        }
    }
}
