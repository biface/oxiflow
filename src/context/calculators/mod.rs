//! # Module `context::calculators`
//!
//! Built-in [`ContextCalculator`] implementations for common PDE quantities.
//!
//! These calculators cover the most frequent context variables in
//! transport–reaction–diffusion problems. Each implements [`ContextCalculator`]
//! and can be registered directly in [`SolverConfiguration`].
//!
//! ## Available calculators
//!
//! | Type | Provides | Notes |
//! |---|---|---|
//! | [`TimeCalculator`] | `ContextVariable::Time` | Current simulation time |
//! | [`TimeStepCalculator`] | `ContextVariable::TimeStep` | Current `dt` |
//! | [`FDGradientCalculator`] | `ContextVariable::SpatialGradient` | FD gradient (fwd/ctr/bwd) |
//! | [`FDLaplacianCalculator`] | `ContextVariable::External { name }` | FD Laplacian |
//! | [`TrapezoidalIntegral`] | `ContextVariable::External { name }` | Spatial quadrature |
//! | [`ExternalTabulated`] | `ContextVariable::External { name }` | Tabulated f(t) interpolation |
//!
//! ## Mesh ownership
//!
//! Spatial calculators ([`FDGradientCalculator`], [`FDLaplacianCalculator`],
//! [`TrapezoidalIntegral`]) hold an [`Arc<dyn Mesh>`] internally. This is an
//! implementation detail consistent with INV-1 (DD-007): the mesh never appears
//! in the public `ContextCalculator` API. At v0.5.0, `Arc<dyn Mesh>` will be
//! replaced by a concrete [`DiscreteOperator`] (INV-2, DD-012), with zero change
//! to the `ContextCalculator` trait signature.
//!
//! [`ContextCalculator`]: crate::context::calculator::ContextCalculator
//! [`SolverConfiguration`]: crate::solver::config::SolverConfiguration
//! [`Arc<dyn Mesh>`]: std::sync::Arc
//! [`DiscreteOperator`]: crate::operators

pub mod integral;
pub mod spatial;
pub mod tabulated;
pub mod time;

// ── Re-exports ────────────────────────────────────────────────────────────────

pub use integral::TrapezoidalIntegral;
pub use spatial::{FDGradientCalculator, FDLaplacianCalculator};
pub use tabulated::{ExternalTabulated, Interpolation};
pub use time::{TimeCalculator, TimeStepCalculator};

// ── FDScheme ──────────────────────────────────────────────────────────────────

/// Finite-difference stencil for first-order spatial derivatives.
///
/// Used by [`FDGradientCalculator`]. At interior nodes all three schemes are
/// fully applicable; at boundary nodes the calculator falls back to the nearest
/// available one-sided stencil automatically.
///
/// | Variant | Stencil | Order | Boundary fallback |
/// |---|---|---|---|
/// | `Forward` | `(u[i+1] − u[i]) / dx` | 1st | `Backward` at last node |
/// | `Backward` | `(u[i] − u[i−1]) / dx` | 1st | `Forward` at first node |
/// | `Central` | `(u[i+1] − u[i−1]) / (2 dx)` | 2nd | One-sided at boundaries |
///
/// # Examples
///
/// ```rust
/// use oxiflow::context::calculators::FDScheme;
///
/// let scheme = FDScheme::Central;
/// ```
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FDScheme {
    /// Forward difference: `(u[i+1] − u[i]) / dx` — 1st-order accurate.
    Forward,
    /// Backward difference: `(u[i] − u[i−1]) / dx` — 1st-order accurate.
    Backward,
    /// Central difference: `(u[i+1] − u[i−1]) / (2 dx)` — 2nd-order accurate.
    Central,
}
