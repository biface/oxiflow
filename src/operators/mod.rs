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
//! | [`fv::FVCenteredFlux`], [`fv::FVUpwindFlux`] | FV | `FluxDivergenceOperator` | v0.5.0 (#48) ✅ |
//! | [`weno::WENO3`], [`weno::WENO5`] | WENO | `FluxDivergenceOperator` | v0.5.0 (#49) ✅ |
//! | [`limiters::LimitedFlux`] (`MinMod`/`VanLeer`/`Superbee`) | MUSCL | `FluxDivergenceOperator` | v0.5.0 (#99) ✅ |
//! | [`adaptive::AdaptiveFlux`] (WENO3/limiter blend, Péclet-gated) | — | `FluxDivergenceOperator` | v0.5.0 (#99) ✅ |
//! | `FiniteElement` | FEM P1/P2 | TBD | J7 — v2.0 |

pub mod adaptive;
pub mod fd;
pub mod fv;
pub mod limiters;
pub mod weno;

use crate::boundary::BoundaryCondition;
use crate::context::compute::ComputeContext;
use crate::context::error::OxiflowError;
use crate::context::value::ContextValue;
use crate::mesh::Mesh;
use crate::model::traits::RequiresContext;
use nalgebra::DVector;

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

// ── FluxBoundary ──────────────────────────────────────────────────────────────

/// Boundary treatment shared by cell/face-based [`FluxDivergenceOperator`]
/// families (`operators::fv`, `operators::weno`) — both face the same
/// question: a finite `n`-node mesh only delimits `n−1` interior cells (FV)
/// or lacks enough neighbors for a wide stencil at the edges (WENO) unless
/// the domain wraps around.
///
/// Three families of treatment exist: periodic wrap (exact conservation by
/// telescoping for FV; the only one implemented so far); ghost cells via the
/// existing [`crate::boundary::BoundaryCondition`] system (a real
/// integration, scheduled later this sprint, after `#49`); and FD-style
/// one-sided truncation at the boundary (breaks FV's conservation property).
/// Rather than picking one silently, `FluxBoundary` makes the choice an
/// explicit constructor parameter on every operator that needs it.
/// `#[non_exhaustive]` (DD-022) signals this is an extension point, not a
/// closed set — only [`FluxBoundary::Periodic`] exists today.
#[non_exhaustive]
#[derive(Debug, Clone)]
pub enum FluxBoundary {
    /// The domain wraps around: the last cell/node's right neighbor is the
    /// first cell/node. Gives an exact discrete conservation property for
    /// FV — the sum of per-cell flux divergences telescopes to zero, since
    /// every internal face flux is counted once as a `+` contribution (its
    /// left cell) and once as a `−` contribution (its right cell).
    Periodic,

    /// One-sided (decentered) fallback at the two boundary cells — no wrap,
    /// no invented value outside the domain.
    ///
    /// Each boundary cell is assigned the same stencil formula already
    /// computed for the nearest cell that has a full interior stencil,
    /// rather than reconstructing a face flux across the domain edge — the
    /// same posture `operators::fd`'s `CenteredLaplacian`/`UpwindGradient`
    /// already take at their own boundary nodes (DD-012): reuse an
    /// available one-sided formula instead of inventing an exterior point.
    /// Breaks FV's exact conservation property (`Periodic`'s telescoping
    /// sum no longer holds, since the outermost face fluxes are each
    /// counted only once, not twice) — a documented compromise (DD-042),
    /// not an oversight.
    ///
    /// Requires enough cells for at least one cell with a full stencil to
    /// exist: 3 for `operators::fv` (2-point stencil, one interior cell to
    /// borrow from), 3 for `WENO3` (needs 1 cell of margin each side), 5
    /// for `WENO5` (needs 2 cells of margin each side) — checked explicitly
    /// by each operator, not silently producing a degenerate result.
    Truncation,

    /// Ghost-cell fallback (DD-042): the two boundary cells' missing
    /// neighbors are supplied by the real [`crate::boundary::BoundaryCondition`]
    /// already registered on the `Domain` for that edge, via
    /// [`crate::boundary::BoundaryCondition::ghost_value`] — the one variant
    /// of `FluxBoundary` that references actual boundary physics, unlike
    /// `Periodic`/`Truncation`, which are purely numerical stencil-closure
    /// conventions.
    ///
    /// `Arc` rather than `Box`: the same `BoundaryCondition` instance is
    /// typically already owned by `Domain` (`Vec<Box<dyn BoundaryCondition>>`)
    /// and shared here, not duplicated.
    ///
    /// If the referenced BC does not override `ghost_value()` (still
    /// returns `None`), `apply()` fails explicitly via
    /// [`OxiflowError::PreconditionFailed`] rather than silently
    /// substituting an approximation — see DD-042 for why no generic
    /// per-`BoundaryType` fallback is attempted here.
    GhostCell(
        std::sync::Arc<dyn crate::boundary::BoundaryCondition>,
        std::sync::Arc<dyn crate::boundary::BoundaryCondition>,
    ),
}

// ── check_cfl ─────────────────────────────────────────────────────────────────

/// Checks the explicit advective CFL stability condition `|v|·dt/dx ≤ 1`.
///
/// Shared by `operators::fv` and `operators::weno` — both are explicit
/// advective schemes subject to the same necessary stability condition
/// (documented for Lax–Wendroff-type schemes). Covers the *advective*
/// condition only; a diffusive term's own stability limit (Fourier number
/// `D·dt/dx² ≤ 0.5`) is a separate concern, not checked here — conceptually
/// the solver's step-size control's responsibility, not a spatial
/// operator's, if ever enforced.
pub(crate) fn check_cfl(
    context: &'static str,
    velocity: f64,
    dt: f64,
    dx: f64,
) -> Result<(), OxiflowError> {
    let cfl = velocity.abs() * dt / dx;
    if cfl > 1.0 {
        return Err(OxiflowError::PreconditionFailed {
            context,
            message: format!(
                "CFL condition violated: |v|*dt/dx = {cfl:.6} > 1.0 \
                 (v = {velocity}, dt = {dt}, dx = {dx}) — explicit advection \
                 scheme is unstable at this step size"
            ),
        });
    }
    Ok(())
}

// ── Shared wide-stencil divergence helpers ──────────────────────────────────
//
// Originally introduced in `operators::weno` (#49) for its multi-point
// stencils, relocated here `pub(crate)` once `operators::limiters` (#99)
// turned out to need the exact same boundary-handling logic for its
// 3-point MUSCL stencil — a second concrete consumer, not a speculative
// generalization (same posture as DD-042's own "no anticipation without a
// second concrete need").

/// Periodic index `i + k` wrapped into `[0, n)`, `k` possibly negative.
pub(crate) fn wrap(i: usize, k: isize, n: usize) -> usize {
    (i as isize + k).rem_euclid(n as isize) as usize
}

/// Computes the divergence of a wide-stencil face flux for a periodic
/// domain — analogous to `operators::fv`'s two-point `periodic_divergence`,
/// generalized to a `face_flux` that reads an arbitrary window of `u`
/// rather than just the two immediately adjacent values.
///
/// `face_flux(u, n, i)` evaluates the flux at the face between node `i` and
/// node `(i+1) % n`.
pub(crate) fn periodic_wide_divergence(
    u: &DVector<f64>,
    dx: f64,
    min_nodes: usize,
    context: &'static str,
    face_flux: impl Fn(&DVector<f64>, usize, usize) -> f64,
) -> Result<DVector<f64>, OxiflowError> {
    let n = u.len();
    if n < min_nodes {
        return Err(OxiflowError::InvalidDomain(format!(
            "{context} requires at least {min_nodes} nodes, got {n}"
        )));
    }

    let mut div = DVector::zeros(n);
    for i in 0..n {
        let flux_right = face_flux(u, n, i);
        let flux_left = face_flux(u, n, (i + n - 1) % n);
        div[i] = (flux_right - flux_left) / dx;
    }
    Ok(div)
}

/// Computes the one-sided (decentered) divergence for a non-periodic domain
/// — [`FluxBoundary::Truncation`], wide-stencil generalization of
/// `operators::fv`'s `truncated_divergence`.
///
/// `face_flux(u, n, k)` reads offsets in `[-margin_left, margin_right]`
/// relative to `k` (asymmetric for upwind-biased stencils — e.g. WENO3's
/// left-biased reconstruction reads `{k−1, k, k+1}`, its right-biased one
/// reads `{k, k+1, k+2}`; the caller computes `margin_left`/`margin_right`
/// from the reconstruction actually selected, not a fixed value, so the
/// boundary zone is no wider than the stencil in use actually requires). A
/// cell `i`'s divergence reads `face_flux(i)` (right face) and
/// `face_flux(i−1)` (left face); the safe (non-wrapping) range for both is
/// `[margin_left+1, n−1−margin_right]`. Cells outside that range — lacking a
/// full stencil on at least one side — are assigned the same value already
/// computed for the nearest safe cell, the same posture as
/// `operators::fv::truncated_divergence` and `operators::fd`'s boundary
/// stencils. The two boundary zones are generally of different sizes
/// (`margin_left+1` cells on the left, `margin_right` cells on the right) —
/// an inherent consequence of `face(i)` and `face(i−1)` both needing to be
/// safe for cell `i`, not an asymmetry bug.
pub(crate) fn truncated_wide_divergence(
    u: &DVector<f64>,
    dx: f64,
    margin_left: usize,
    margin_right: usize,
    context: &'static str,
    face_flux: impl Fn(&DVector<f64>, usize, usize) -> f64,
) -> Result<DVector<f64>, OxiflowError> {
    let n = u.len();
    let min_nodes = margin_left + margin_right + 2;
    if n < min_nodes {
        return Err(OxiflowError::InvalidDomain(format!(
            "{context} with FluxBoundary::Truncation requires at least {min_nodes} nodes \
             for this upwind direction (margin_left={margin_left}, margin_right={margin_right}), \
             got {n}"
        )));
    }

    let safe_start = margin_left + 1;
    let safe_end = n - 1 - margin_right; // inclusive

    let mut div = DVector::zeros(n);
    for i in safe_start..=safe_end {
        let flux_right = face_flux(u, n, i);
        let flux_left = face_flux(u, n, i - 1);
        div[i] = (flux_right - flux_left) / dx;
    }
    for i in 0..safe_start {
        div[i] = div[safe_start];
    }
    for i in (safe_end + 1)..n {
        div[i] = div[safe_end];
    }
    Ok(div)
}

/// Builds a ghost-padded copy of `u` for [`FluxBoundary::GhostCell`] —
/// `margin_left` ghost values prepended, `u` in the middle, `margin_right`
/// ghost values appended, so that every real cell's stencil (however wide)
/// finds all its neighbors in-bounds with plain indexing — no wraparound.
///
/// Each ghost value comes from [`BoundaryCondition::ghost_value`] at the
/// depth matching its distance from the domain edge, paired with the real
/// interior value at the *symmetric* depth (method of images — DD-042,
/// amendment 1): depth `k` pairs with interior index `k−1` from that edge.
/// Every layer is derived directly from a real interior value, never from a
/// previously computed ghost layer, so a Robin condition's exactness does
/// not erode with depth.
pub(crate) fn ghost_padded_field(
    u: &DVector<f64>,
    dx: f64,
    margins: (usize, usize),
    bcs: (&dyn BoundaryCondition, &dyn BoundaryCondition),
    context: &'static str,
) -> Result<Vec<f64>, OxiflowError> {
    let (margin_left, margin_right) = margins;
    let (left_bc, right_bc) = bcs;
    let n = u.len();
    let mut extended = Vec::with_capacity(n + margin_left + margin_right);

    // Left ghosts, deepest first, so the array reads left-to-right in
    // increasing physical position once `u` follows.
    for depth in (1..=margin_left).rev() {
        let interior_at_depth = u[depth - 1];
        let g = left_bc
            .ghost_value(depth, interior_at_depth, dx)
            .ok_or_else(|| OxiflowError::PreconditionFailed {
                context,
                message: format!(
                    "left boundary condition ({:?}) does not supply a ghost value at depth \
                     {depth} — FluxBoundary::GhostCell requires an exact value, not a generic \
                     fallback",
                    left_bc.boundary_type()
                ),
            })?;
        extended.push(g);
    }

    extended.extend(u.iter().copied());

    for depth in 1..=margin_right {
        let interior_at_depth = u[n - depth];
        let g = right_bc
            .ghost_value(depth, interior_at_depth, dx)
            .ok_or_else(|| OxiflowError::PreconditionFailed {
                context,
                message: format!(
                    "right boundary condition ({:?}) does not supply a ghost value at depth \
                     {depth} — FluxBoundary::GhostCell requires an exact value, not a generic \
                     fallback",
                    right_bc.boundary_type()
                ),
            })?;
        extended.push(g);
    }

    Ok(extended)
}

/// Computes the divergence for [`FluxBoundary::GhostCell`] by delegating to
/// `face_flux` over a [`ghost_padded_field`] — every real cell's stencil,
/// including the two boundary ones, reads only in-bounds data, so no
/// separate boundary-cell code path is needed here (unlike
/// [`truncated_wide_divergence`]): the padding does the work.
pub(crate) fn ghost_cell_wide_divergence(
    u: &DVector<f64>,
    dx: f64,
    margins: (usize, usize),
    bcs: (&dyn BoundaryCondition, &dyn BoundaryCondition),
    context: &'static str,
    face_flux: impl Fn(&DVector<f64>, usize, usize) -> f64,
) -> Result<DVector<f64>, OxiflowError> {
    let n = u.len();
    let margin_left = margins.0;
    let extended = ghost_padded_field(u, dx, margins, bcs, context)?;
    let extended = DVector::from_vec(extended);
    let m = extended.len();

    let mut div = DVector::zeros(n);
    for i in 0..n {
        let k = i + margin_left;
        let flux_right = face_flux(&extended, m, k);
        let flux_left = face_flux(&extended, m, k - 1);
        div[i] = (flux_right - flux_left) / dx;
    }
    Ok(div)
}
