//! # Module `operators`
//!
//! Spatial discretization operators — INV-2 invariant.
//!
//! ## Core principle (INV-2)
//!
//! Spatial discretization is decoupled from the rest of the engine through two
//! sibling traits — never a single generic-over-everything trait, and never a
//! scheme called directly from a consumer:
//!
//! - [`DiscreteOperator`] — raw differential quantities (`∂u/∂x`, `∂²u/∂x²`) that
//!   never need physical parameters or context access; `dx` comes from the mesh
//!   alone. Consumed transparently through [`crate::context::ContextCalculator`]
//!   (e.g. `FDGradientCalculator`) — a model declares a required context
//!   variable and never knows which scheme produced it (DD-012).
//! - [`FluxDivergenceOperator`] — the complete `∇·F(u, ∇u)` term, which does need
//!   physical parameters (advection velocity, diffusion coefficient) that may be
//!   space/time-varying. Carries `ctx: &ComputeContext` and `RequiresContext`
//!   directly on `apply()`. Consumed by `DiscretizedModel<Op>` (DD-038, DD-039).
//!
//! `FluxDivergenceOperator` is a **sibling** trait, not a sub-trait of
//! `DiscreteOperator`: their `apply()` signatures diverge (with/without `ctx`),
//! so a sub-trait relationship could not cleanly reuse the parent method.
//! Forcing `ctx`/`RequiresContext` onto `DiscreteOperator` for every scheme was
//! considered and rejected (DD-012, amendment 2) — it would have imposed that
//! capability on FD, which never needs it.
//!
//! **Selection criterion for future schemes** (FEM, spectral methods, J20):
//! does the scheme need per-call access to physical context, or does it reduce
//! to a local differentiation from mesh data alone? The former implements
//! `FluxDivergenceOperator`; the latter implements `DiscreteOperator`.
//!
//! ## Planned implementations
//!
//! | Type | Scheme | Trait | Milestone |
//! |---|---|---|---|
//! | [`fd::UpwindGradient`], [`fd::CenteredGradient`], [`fd::CenteredLaplacian`] | FD | `DiscreteOperator` | v0.5.0 (#47) ✅ |
//! | `ConservativeFV` | FV | `FluxDivergenceOperator` | v0.5.0 (#48) |
//! | `Weno3`, `Weno5` | WENO + limiters | `FluxDivergenceOperator` | v0.5.0 (#49) |
//! | `FiniteElement` | FEM P1/P2 | TBD | J7 — v2.0 |

pub mod fd;

use crate::context::compute::ComputeContext;
use crate::context::error::OxiflowError;
use crate::context::value::ContextValue;
use crate::mesh::Mesh;
use crate::model::traits::RequiresContext;

/// Raw spatial differential operator (FD family) — INV-2.
///
/// Computes a single differential quantity (`∂u/∂x`, `∂²u/∂x²`, ...) from a
/// field and a mesh, with no dependency on physical parameters or runtime
/// context: `dx` comes entirely from `mesh`. Consumed transparently through
/// [`crate::context::ContextCalculator`] (DD-012) — never called directly by a
/// temporal integrator.
///
/// Uses an associated type `MeshType`, not a generic parameter, per DD-012
/// amendment 1: each scheme implementation targets exactly one mesh family, so
/// a generic `DiscreteOperator<M: Mesh>` would add flexibility no consumer
/// needs, at the cost of object-safety (INV-4).
///
/// Distinct from [`FluxDivergenceOperator`] (DD-039), which computes the
/// complete `∇·F(u, ∇u)` term and does need context access — see the module
/// documentation above for the selection criterion between the two.
pub trait DiscreteOperator: Send + Sync {
    /// The mesh family this scheme is implemented for.
    type MeshType: Mesh;

    /// Applies the scheme to `field` on `mesh`, returning the differential
    /// quantity the scheme computes.
    fn apply(
        &self,
        field: &ContextValue,
        mesh: &Self::MeshType,
    ) -> Result<ContextValue, OxiflowError>;
}

/// Complete flux-divergence spatial operator (FV/WENO family) — DD-039.
///
/// Computes the full `∇·F(u, ∇u)` term of the canonical PDE form, which —
/// unlike [`DiscreteOperator`] — requires knowledge of the physics of `F`
/// (advection velocity, diffusion coefficient, ...), potentially varying in
/// space or time. Carries [`RequiresContext`] and receives `ctx` directly on
/// `apply()`.
///
/// A **sibling** trait to `DiscreteOperator`, not a sub-trait: the two
/// `apply()` signatures diverge (with/without `ctx`), so a sub-trait
/// relationship could not reuse the parent method cleanly (DD-039, option B
/// rejected). Adding `ctx`/`RequiresContext` to `DiscreteOperator` itself was
/// also rejected (DD-012, amendment 2; DD-039, option A) — it would force
/// that capability onto FD schemes, which never need it.
///
/// Two consumption modes fall out of `RequiresContext` for free, with no new
/// mechanism: a scheme with only constant physical parameters returns
/// `required_variables() -> vec![]` and reads its own fields directly in
/// `apply()`; a scheme with space/time-varying parameters declares them as
/// [`crate::context::ContextVariable`]s in `required_variables()` and reads
/// them from `ctx`, resolved and ordered beforehand by `chain.rs` (DD-009)
/// exactly as for any `PhysicalModel`/`BoundaryCondition`.
///
/// Consumed by `DiscretizedModel<Op>` (DD-038), which binds `Op:
/// FluxDivergenceOperator` — an FD scheme cannot be plugged in there; that
/// becomes a compile error, not a silent runtime bug.
pub trait FluxDivergenceOperator: RequiresContext + Send + Sync {
    /// The mesh family this scheme is implemented for.
    type MeshType: Mesh;

    /// Applies the scheme to `field` on `mesh`, resolving any physical
    /// parameter declared in [`RequiresContext::required_variables`] from
    /// `ctx`, and returns `∇·F(u, ∇u)`.
    fn apply(
        &self,
        field: &ContextValue,
        mesh: &Self::MeshType,
        ctx: &ComputeContext,
    ) -> Result<ContextValue, OxiflowError>;
}
