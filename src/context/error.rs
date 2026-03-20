//! # Module `context::error`
//!
//! Main error type of the oxiflow engine.
//!
//! `OxiflowError` is a typed enum built with `thiserror` covering all engine failure
//! cases: missing variable, type mismatch, circular dependency, solver divergence
//! (DD-004, J1).
