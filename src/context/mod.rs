//! # Module `context`
//!
//! Context variables and calculators management.
//!
//! This module gathers the fundamental types enabling a physical model to declare
//! its context variable requirements and the engine to resolve them before solving.
//!
//! ## Submodules
//!
//! - [`compute`] — Type `ComputeContext` and trait `RequiresContext` (DD-005, DD-006)
//! - [`error`]   — Type `OxiflowError` and variant (DD-004)

pub mod compute;
pub mod error;
