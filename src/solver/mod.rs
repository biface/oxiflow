//! # Module `solver`
//!
//! Numerical solving orchestration — WHAT/HOW separation.
//!
//! ## Responsibilities
//!
//! | Type | role |
//! |---|---|
//! | [`scenario`] | Declares and validates the problem |
//! | [`config`] | Configures solving parameters |
//! | [`chain`] | Composes and sequences steps |
//! | [`methods`] | Temporal integration methods |
//!
//! `Scenario` validates configuration consistency *before* solving. Any configuration
//! error raises an `OxiflowError` at startup, never a panic mid-computation.

pub mod chain;
pub mod config;
pub mod methods;
pub mod scenario;
