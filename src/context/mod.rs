//! # Module `context`
//!
//! Context variables and calculators management.
//!
//! This module gathers the fundamental types enabling a physical model to declare
//! its context variable requirements and the engine to resolve them before solving.
//!
//! ## Submodules
//!
//! - [`variable`] — Type `ContextVariable`: typed key for context variables
//! - [`error`]    — Type `OxiflowError`: typed engine error enum (DD-004)
//! - [`value`]    — Type `ContextValue`: typed value enum (DD-003)
//! - [`compute`]  — Type `ComputeContext` and trait `RequiresContext` (DD-005, DD-006)

pub mod compute;
pub mod error;
pub mod value;
pub mod variable;

// ── Re-exports ────────────────────────────────────────────────────────────────

pub use compute::ComputeContext;
pub use error::OxiflowError;
pub use value::ContextValue;
pub use variable::ContextVariable;
