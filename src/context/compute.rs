//! # Module `context::compute`
//!
//! Type-safe computation context and variable requirements declaration.
//!
//! ## `RequiresContext` (DD-005)
//!
//! Trait separate from `PhysicalModel`: a model declares its required, optional
//! variables and dependencies. The engine guarantees their availability before solving.
//!
//! ## `ComputeContext` (DD-006)
//!
//! Type-safe API from v0.2 — context variable access is compile-time guaranteed or
//! fails explicitly via `OxiflowError` at startup, never mid-computation.
