//! oxiflow — Generic PDE solver engine
//!
//! Solves problems governed by the canonical form:
//!
//! ```text
//! ∂u/∂t + ∇·F(u, ∇u) = S(u, x, t)
//! ```
//!
//! # Architecture
//!
//! - [`Scenario`]            — declares the problem (WHAT)
//! - [`SolverConfiguration`] — configures resolution (HOW)
//! - [`Solver`]              — executes the numerical integration
//!
//! # Modules
//!
//! Modules are declared but not yet implemented (v0.0.5 skeleton).
//! Implementation begins at v0.1.0 — Core Architecture.

// ── Modules (skeleton — implementation starts at v0.1.0) ────────────────────

pub mod context;
pub mod model;
pub mod mesh;
pub mod solver;
pub mod boundary;
pub mod coupling;
pub mod operators;

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    // Placeholder test — ensures llvm-cov generates a valid profdata
    // even when no module is implemented yet.
    // Remove once the first J1 tests are in place.
    #[test]
    fn placeholder() {}
}
