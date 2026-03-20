//! # oxiflow
//!
//! Generic engine for solving partial differential equations of the form:
//!
//! ```text
//! ∂u/∂t + ∇·F(u, ∇u) = S(u, x, t)
//! ```
//!
//! where `u` is a physical field (concentration, temperature, velocity…),
//! `F` a flux (advective, diffusive, dispersive) and `S` a source or reaction term.
//!
//! ## Architecture — separation WHAT/HOW
//!
//! The engine enforces three strictly separated responsibility levels:
//!
//! | Type | Role |
//! |---|---|
//! | `Scenario` | Declares the problem |
//! | `SolverConfiguration` | Configures the solving |
//! | `Solver` | Orchestrates execution |
//!
//! ## Modules
//!
//! - [`context`]   — Variables and calculators (DD-003–DD-006)
//! - [`mesh`]      — Mesh abstraction (INV-1, DD-007)
//! - [`model`]     — Physical models
//! - [`boundary`]  — Boundary conditions (DD-008)
//! - [`solver`]    — Numerical orchestration
//! - [`operators`] — Discrete operators (INV-2, DD-012)
//! - [`coupling`]  — Multi-domain coupling (INV-3, DD-011)
//!
//! ## Invariants FEM anticipated
//!
//! v0.x abstractions do not presuppose a structured grid, ensuring forward compatibility
//! with the FEM support planned at v2.0.
//!
//! | Invariant | Description | Active from |
//! |---|---|---|
//! | INV-1 | Abstract `Mesh` — zero `dx`/`nx` in public API | v0.1.0 |
//! | INV-2 | `DiscreteOperator<M: Mesh>` — integrators decoupled from the scheme | v0.5.0 |
//! | INV-3 | `CouplingOperator` — explicit inter-domain coupling | v0.3.0 |
//! | INV-4 | Plugin-safe API — object-safe traits from external crates | v2.0.0 |

pub mod boundary;
pub mod context;
pub mod coupling;
pub mod mesh;
pub mod model;
pub mod operators;
pub mod solver;

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    // Placeholder test — ensures llvm-cov generates a valid profdata
    // even when no module is implemented yet.
    // Remove once the first J1 tests are in place.
    #[test]
    fn placeholder() {}
}