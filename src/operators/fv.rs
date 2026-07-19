//! # Module `operators::fv`
//!
//! Finite-volume [`FluxDivergenceOperator`] implementations for
//! [`UniformGrid1D`] — `FVCenteredFlux` (2nd-order centered advection),
//! `FVUpwindFlux` (1st-order upwind advection, robust for strong advection)
//! (#48, DD-039, DD-040).
//!
//! Both compute the full flux `F(u, ∇u) = v·u − D·∂u/∂x` (advection, Fick's
//! law diffusion) — the diffusive term is always centered in both schemes;
//! only the advective term differs (centered vs upwind). Only advection
//! needs directional bias for stability; diffusion is symmetric.
//!
//! ## Cell semantics (DD-040)
//!
//! `UniformGrid1D`/`Mesh` stay purely nodal (INV-1) — no cell-center API was
//! added. `u[i]` is interpreted as the average of the cell whose faces sit
//! at nodes `i` and `i+1`; a face flux is therefore evaluated directly at a
//! mesh node, not at a separately-computed cell center.
//!
//! ## `FluxBoundary` — an explicit choice, not a hidden assumption
//!
//! A finite `n`-node mesh only delimits `n−1` interior cells unless the
//! domain wraps around — matching the number of state values to the number
//! of cells requires *some* boundary convention. [`FluxBoundary`] (defined
//! in [`crate::operators`], shared with `operators::weno` — both families
//! face the same boundary question) makes this an explicit constructor
//! parameter rather than a hidden assumption; see its documentation for the
//! alternatives considered.
//!
//! ## CFL check
//!
//! Both schemes check the explicit advective stability condition
//! `|v|·dt/dx ≤ 1` via the shared [`crate::operators::check_cfl`] (the
//! condition documented for Lax–Wendroff-type explicit advection schemes)
//! inside `apply()`, using `ctx.time_step()` — failing explicitly via
//! [`OxiflowError::PreconditionFailed`] rather than silently returning an
//! unstable result. This covers the *advective* stability condition only;
//! the diffusive term has its own explicit-stability limit (Fourier number
//! `D·dt/dx² ≤ 0.5`), not checked here — out of `#48`'s scope, and belongs
//! conceptually to the solver's step-size control, not a spatial operator,
//! if ever enforced.
//!
//! [`UniformGrid1D`]: crate::mesh::structured::UniformGrid1D

use nalgebra::DVector;

use crate::boundary::BoundaryCondition;
use crate::context::compute::ComputeContext;
use crate::context::error::OxiflowError;
use crate::context::value::ContextValue;
use crate::context::variable::ContextVariable;
use crate::mesh::structured::UniformGrid1D;
use crate::mesh::Mesh;
use crate::model::traits::RequiresContext;
use crate::operators::{check_cfl, FluxBoundary, FluxDivergenceOperator};

// ── Shared helpers ────────────────────────────────────────────────────────────

/// Computes the periodic flux divergence for a given face-flux function.
///
/// `face_flux(left, right)` evaluates the total flux `F` at the face between
/// two neighboring cells, given their (cell-average) state values. The
/// divergence at cell `i` is `(F_{i+1/2} − F_{i−1/2}) / dx`, with periodic
/// wrap-around indexing (`FluxBoundary::Periodic`, see module documentation).
fn periodic_divergence(
    u: &DVector<f64>,
    dx: f64,
    face_flux: impl Fn(f64, f64) -> f64,
) -> Result<DVector<f64>, OxiflowError> {
    let n = u.len();
    if n < 2 {
        return Err(OxiflowError::InvalidDomain(format!(
            "finite-volume flux requires at least 2 cells, got {n}"
        )));
    }

    let mut div = DVector::zeros(n);
    for i in 0..n {
        let next = (i + 1) % n;
        let prev = (i + n - 1) % n;
        let flux_right = face_flux(u[i], u[next]);
        let flux_left = face_flux(u[prev], u[i]);
        div[i] = (flux_right - flux_left) / dx;
    }
    Ok(div)
}

/// Computes the one-sided (decentered) flux divergence for a non-periodic
/// domain — [`FluxBoundary::Truncation`].
///
/// Interior cells use the same centered two-face formula as
/// [`periodic_divergence`]. The two boundary cells (`0` and `n−1`) have no
/// face across the domain edge; rather than inventing one, each is assigned
/// the same formula already computed for its nearest interior neighbor
/// (`1` and `n−2` respectively) — see [`FluxBoundary::Truncation`]'s
/// documentation for why this mirrors `operators::fd`'s existing boundary
/// posture. Requires at least 3 cells: 2 boundary cells plus 1 interior cell
/// to borrow from.
fn truncated_divergence(
    u: &DVector<f64>,
    dx: f64,
    face_flux: impl Fn(f64, f64) -> f64,
) -> Result<DVector<f64>, OxiflowError> {
    let n = u.len();
    if n < 3 {
        return Err(OxiflowError::InvalidDomain(format!(
            "finite-volume flux with FluxBoundary::Truncation requires at least 3 cells, got {n}"
        )));
    }

    let mut div = DVector::zeros(n);
    for i in 1..n - 1 {
        let flux_right = face_flux(u[i], u[i + 1]);
        let flux_left = face_flux(u[i - 1], u[i]);
        div[i] = (flux_right - flux_left) / dx;
    }
    div[0] = div[1];
    div[n - 1] = div[n - 2];
    Ok(div)
}

/// Computes the flux divergence using ghost-cell values supplied by real
/// [`BoundaryCondition`]s — [`FluxBoundary::GhostCell`].
///
/// Interior cells use the same centered two-face formula as
/// [`periodic_divergence`]/[`truncated_divergence`]. The two boundary cells
/// use a ghost value from `left_bc`/`right_bc` (via
/// [`BoundaryCondition::ghost_value`], depth 1 — FV's 2-point stencil never
/// needs more) as the missing neighbor, instead of wrapping (`Periodic`) or
/// reusing an interior formula (`Truncation`) — see
/// [`FluxBoundary::GhostCell`]'s documentation for why only this variant
/// references real boundary physics. Fails explicitly if either BC does not
/// override `ghost_value()` (still returns `None`) — no generic fallback is
/// substituted (DD-042).
fn ghost_cell_divergence(
    u: &DVector<f64>,
    dx: f64,
    left_bc: &dyn BoundaryCondition,
    right_bc: &dyn BoundaryCondition,
    context: &'static str,
    face_flux: impl Fn(f64, f64) -> f64,
) -> Result<DVector<f64>, OxiflowError> {
    let n = u.len();
    if n < 2 {
        return Err(OxiflowError::InvalidDomain(format!(
            "finite-volume flux requires at least 2 cells, got {n}"
        )));
    }

    let ghost_left =
        left_bc
            .ghost_value(1, u[0], dx)
            .ok_or_else(|| OxiflowError::PreconditionFailed {
                context,
                message: format!(
                    "left boundary condition ({:?}) does not supply a ghost value at depth 1 — \
                 FluxBoundary::GhostCell requires an exact ghost value, not a generic fallback",
                    left_bc.boundary_type()
                ),
            })?;
    let ghost_right =
        right_bc
            .ghost_value(1, u[n - 1], dx)
            .ok_or_else(|| OxiflowError::PreconditionFailed {
                context,
                message: format!(
                    "right boundary condition ({:?}) does not supply a ghost value at depth 1 — \
                 FluxBoundary::GhostCell requires an exact ghost value, not a generic fallback",
                    right_bc.boundary_type()
                ),
            })?;

    let mut div = DVector::zeros(n);
    for i in 1..n - 1 {
        let flux_right = face_flux(u[i], u[i + 1]);
        let flux_left = face_flux(u[i - 1], u[i]);
        div[i] = (flux_right - flux_left) / dx;
    }
    div[0] = (face_flux(u[0], u[1]) - face_flux(ghost_left, u[0])) / dx;
    div[n - 1] = (face_flux(u[n - 1], ghost_right) - face_flux(u[n - 2], u[n - 1])) / dx;
    Ok(div)
}

// ── FVCenteredFlux ────────────────────────────────────────────────────────────

/// 2nd-order centered advective flux + centered Fick's-law diffusive flux.
///
/// `F(u, ∇u) = v·u − D·∂u/∂x`, with the advective term reconstructed as the
/// average of the two neighboring cell values at each face.
#[derive(Debug, Clone)]
pub struct FVCenteredFlux {
    velocity: f64,
    diffusion: f64,
    boundary: FluxBoundary,
}

impl FVCenteredFlux {
    /// Creates a centered-flux FV operator.
    pub fn new(velocity: f64, diffusion: f64, boundary: FluxBoundary) -> Self {
        Self {
            velocity,
            diffusion,
            boundary,
        }
    }
}

impl RequiresContext for FVCenteredFlux {
    fn required_variables(&self) -> Vec<ContextVariable> {
        // Constant velocity/diffusion — nothing to resolve from ComputeContext.
        // A space/time-varying case would declare them here instead (see the
        // FluxDivergenceOperator "advanced case" pattern in DD-039).
        vec![]
    }
}

impl FluxDivergenceOperator for FVCenteredFlux {
    type MeshType = UniformGrid1D;

    fn apply(
        &self,
        field: &ContextValue,
        mesh: &Self::MeshType,
        ctx: &ComputeContext,
    ) -> Result<ContextValue, OxiflowError> {
        let u = field.as_scalar_field()?;
        let dx = mesh.characteristic_length();
        check_cfl("FVCenteredFlux", self.velocity, ctx.time_step(), dx)?;

        let v = self.velocity;
        let d = self.diffusion;
        let face_flux = |left: f64, right: f64| v * (left + right) / 2.0 - d * (right - left) / dx;

        let div = match &self.boundary {
            FluxBoundary::Periodic => periodic_divergence(u, dx, face_flux)?,
            FluxBoundary::Truncation => truncated_divergence(u, dx, face_flux)?,
            FluxBoundary::GhostCell(left_bc, right_bc) => ghost_cell_divergence(
                u,
                dx,
                left_bc.as_ref(),
                right_bc.as_ref(),
                "FVCenteredFlux",
                face_flux,
            )?,
        };
        Ok(ContextValue::ScalarField(div))
    }

    fn stencil_radius(&self) -> usize {
        // Two-point face flux — one neighbor on each side.
        1
    }
}

// ── FVUpwindFlux ──────────────────────────────────────────────────────────────

/// 1st-order upwind advective flux + centered Fick's-law diffusive flux.
///
/// `F(u, ∇u) = v·u − D·∂u/∂x`, with the advective term taken from the
/// upstream cell (direction of `v`) at each face — robust for strong
/// advection where `FVCenteredFlux` would oscillate.
#[derive(Debug, Clone)]
pub struct FVUpwindFlux {
    velocity: f64,
    diffusion: f64,
    boundary: FluxBoundary,
}

impl FVUpwindFlux {
    /// Creates an upwind-flux FV operator.
    pub fn new(velocity: f64, diffusion: f64, boundary: FluxBoundary) -> Self {
        Self {
            velocity,
            diffusion,
            boundary,
        }
    }
}

impl RequiresContext for FVUpwindFlux {
    fn required_variables(&self) -> Vec<ContextVariable> {
        vec![]
    }
}

impl FluxDivergenceOperator for FVUpwindFlux {
    type MeshType = UniformGrid1D;

    fn apply(
        &self,
        field: &ContextValue,
        mesh: &Self::MeshType,
        ctx: &ComputeContext,
    ) -> Result<ContextValue, OxiflowError> {
        let u = field.as_scalar_field()?;
        let dx = mesh.characteristic_length();
        check_cfl("FVUpwindFlux", self.velocity, ctx.time_step(), dx)?;

        let v = self.velocity;
        let d = self.diffusion;
        let face_flux = |left: f64, right: f64| {
            let advective = if v >= 0.0 { v * left } else { v * right };
            advective - d * (right - left) / dx
        };

        let div = match &self.boundary {
            FluxBoundary::Periodic => periodic_divergence(u, dx, face_flux)?,
            FluxBoundary::Truncation => truncated_divergence(u, dx, face_flux)?,
            FluxBoundary::GhostCell(left_bc, right_bc) => ghost_cell_divergence(
                u,
                dx,
                left_bc.as_ref(),
                right_bc.as_ref(),
                "FVUpwindFlux",
                face_flux,
            )?,
        };
        Ok(ContextValue::ScalarField(div))
    }

    fn stencil_radius(&self) -> usize {
        // Two-point face flux — one neighbor on each side.
        1
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    fn mesh(n: usize) -> UniformGrid1D {
        UniformGrid1D::new(n, 0.0, 1.0).unwrap()
    }

    fn ctx(dt: f64) -> ComputeContext {
        ComputeContext::new(0.0, dt)
    }

    // ── Test fixtures: BoundaryCondition with/without ghost_value() ───────────

    /// Fixed-value ghost cell (Dirichlet-like), ignoring `boundary_state`/
    /// `neighbor_state`/`dx` — enough to exercise `FluxBoundary::GhostCell`'s
    /// wiring without depending on a real (not-yet-integrated) BC like
    /// `DanckwertsInlet`.
    #[derive(Debug)]
    struct FixedGhost(f64);

    impl RequiresContext for FixedGhost {
        fn required_variables(&self) -> Vec<crate::context::variable::ContextVariable> {
            vec![]
        }
    }

    impl crate::boundary::BoundaryCondition for FixedGhost {
        fn boundary_type(&self) -> crate::boundary::BoundaryType {
            crate::boundary::BoundaryType::Dirichlet
        }
        fn apply(
            &self,
            _state: &mut DVector<f64>,
            _ctx: &ComputeContext,
            _mesh: &dyn Mesh,
        ) -> Result<(), OxiflowError> {
            Ok(())
        }
        fn ghost_value(&self, _depth: usize, _interior_at_depth: f64, _dx: f64) -> Option<f64> {
            Some(self.0)
        }
    }

    /// A BC that does not override `ghost_value()` — stays at the trait's
    /// `None` default, to exercise `FluxBoundary::GhostCell`'s explicit
    /// failure path.
    #[derive(Debug)]
    struct NoGhost;

    impl RequiresContext for NoGhost {
        fn required_variables(&self) -> Vec<crate::context::variable::ContextVariable> {
            vec![]
        }
    }

    impl crate::boundary::BoundaryCondition for NoGhost {
        fn boundary_type(&self) -> crate::boundary::BoundaryType {
            crate::boundary::BoundaryType::Neumann
        }
        fn apply(
            &self,
            _state: &mut DVector<f64>,
            _ctx: &ComputeContext,
            _mesh: &dyn Mesh,
        ) -> Result<(), OxiflowError> {
            Ok(())
        }
    }

    // ── Conservation property (#48 acceptance criterion) ──────────────────────

    #[test]
    fn centered_flux_conserves_sum_for_periodic_problem() {
        let m = mesh(6);
        let dx = m.characteristic_length();
        let u = ContextValue::ScalarField(DVector::from_vec(vec![0.2, 1.3, -0.7, 2.1, 0.0, -1.5]));
        let op = FVCenteredFlux::new(0.8, 0.05, FluxBoundary::Periodic);
        // dt small enough to satisfy CFL for this test's purpose.
        let result = op.apply(&u, &m, &ctx(0.1 * dx)).unwrap();
        let sum: f64 = result.as_scalar_field().unwrap().iter().sum();
        assert!(sum.abs() < 1e-10, "expected 0, got {sum}");
    }

    #[test]
    fn upwind_flux_conserves_sum_for_periodic_problem() {
        let m = mesh(6);
        let dx = m.characteristic_length();
        let u = ContextValue::ScalarField(DVector::from_vec(vec![0.2, 1.3, -0.7, 2.1, 0.0, -1.5]));
        let op = FVUpwindFlux::new(-0.4, 0.02, FluxBoundary::Periodic);
        let result = op.apply(&u, &m, &ctx(0.1 * dx)).unwrap();
        let sum: f64 = result.as_scalar_field().unwrap().iter().sum();
        assert!(sum.abs() < 1e-10, "expected 0, got {sum}");
    }

    // ── Pure diffusion (v = 0) matches the hand-derived stencil ───────────────

    #[test]
    fn centered_flux_pure_diffusion_matches_laplacian_stencil() {
        let m = mesh(4);
        let dx = m.characteristic_length();
        let values = vec![0.0, 1.0, 0.0, -1.0];
        let u = ContextValue::ScalarField(DVector::from_vec(values.clone()));
        let d = 0.5;
        let op = FVCenteredFlux::new(0.0, d, FluxBoundary::Periodic);
        let result = op.apply(&u, &m, &ctx(0.01)).unwrap();
        let div = result.as_scalar_field().unwrap();

        let n = values.len();
        for i in 0..n {
            let next = (i + 1) % n;
            let prev = (i + n - 1) % n;
            // Derived in the module documentation: F(l, r) = -D(r-l)/dx  ⇒
            // div[i] = -D * (u[prev] - 2u[i] + u[next]) / dx².
            let expected = -d * (values[prev] - 2.0 * values[i] + values[next]) / (dx * dx);
            assert!(
                (div[i] - expected).abs() < 1e-10,
                "node {i}: expected {expected}, got {}",
                div[i]
            );
        }
    }

    // ── CFL check ───────────────────────────────────────────────────────────

    #[test]
    fn centered_flux_rejects_cfl_violation() {
        let m = mesh(5); // dx = 0.25
        let op = FVCenteredFlux::new(10.0, 0.0, FluxBoundary::Periodic);
        let u = ContextValue::ScalarField(DVector::from_element(5, 1.0));
        // |v|*dt/dx = 10 * 1.0 / 0.25 = 40 ≫ 1
        let err = op.apply(&u, &m, &ctx(1.0)).unwrap_err();
        assert!(matches!(err, OxiflowError::PreconditionFailed { .. }));
    }

    #[test]
    fn upwind_flux_accepts_cfl_within_bounds() {
        let m = mesh(5); // dx = 0.25
        let op = FVUpwindFlux::new(1.0, 0.0, FluxBoundary::Periodic);
        let u = ContextValue::ScalarField(DVector::from_element(5, 1.0));
        // |v|*dt/dx = 1.0 * 0.2 / 0.25 = 0.8 ≤ 1
        assert!(op.apply(&u, &m, &ctx(0.2)).is_ok());
    }

    // ── Truncation boundary treatment (#DD-042 amendment scope) ───────────────

    #[test]
    fn centered_flux_truncation_boundary_matches_nearest_interior_formula() {
        let m = mesh(6);
        let dx = m.characteristic_length();
        let u = ContextValue::ScalarField(DVector::from_vec(vec![0.2, 1.3, -0.7, 2.1, 0.0, -1.5]));
        let op = FVCenteredFlux::new(0.8, 0.05, FluxBoundary::Truncation);
        let result = op.apply(&u, &m, &ctx(0.1 * dx)).unwrap();
        let div = result.as_scalar_field().unwrap().clone();
        // Boundary cells reuse their nearest interior neighbor's stencil
        // value — see FluxBoundary::Truncation's documentation.
        assert_eq!(div[0], div[1]);
        assert_eq!(div[div.len() - 1], div[div.len() - 2]);
    }

    #[test]
    fn upwind_flux_truncation_rejects_field_below_three_cells() {
        let m = mesh(5);
        let op = FVUpwindFlux::new(0.5, 0.0, FluxBoundary::Truncation);
        let u = ContextValue::ScalarField(DVector::from_vec(vec![1.0, 2.0]));
        let err = op.apply(&u, &m, &ctx(0.001)).unwrap_err();
        assert!(matches!(err, OxiflowError::InvalidDomain(_)));
    }

    #[test]
    fn centered_flux_truncation_does_not_conserve_sum_in_general() {
        // Documented tradeoff (FluxBoundary::Truncation): unlike Periodic,
        // the outermost face fluxes are each counted once, not twice —
        // no telescoping, so the sum is generally non-zero for a non-trivial
        // field. Asserting non-conservation here guards against silently
        // reintroducing exact conservation by accident later.
        let m = mesh(6);
        let dx = m.characteristic_length();
        let u = ContextValue::ScalarField(DVector::from_vec(vec![0.2, 1.3, -0.7, 2.1, 0.0, -1.5]));
        let op = FVCenteredFlux::new(0.8, 0.05, FluxBoundary::Truncation);
        let result = op.apply(&u, &m, &ctx(0.1 * dx)).unwrap();
        let sum: f64 = result.as_scalar_field().unwrap().iter().sum();
        assert!(sum.abs() > 1e-6, "expected non-zero sum, got {sum}");
    }

    // ── GhostCell boundary treatment (DD-042) ──────────────────────────────

    #[test]
    fn centered_flux_ghost_cell_matches_hand_derived_value() {
        // Pure diffusion (v=0) so the face flux reduces to -D*(right-left)/dx
        // — easy to hand-verify against a known ghost value.
        let m = mesh(4);
        let dx = m.characteristic_length();
        let values = vec![0.0, 1.0, 0.0, -1.0];
        let u = ContextValue::ScalarField(DVector::from_vec(values.clone()));
        let d = 0.5;
        let left_bc = Arc::new(FixedGhost(2.0));
        let right_bc = Arc::new(FixedGhost(1.0));
        let op = FVCenteredFlux::new(0.0, d, FluxBoundary::GhostCell(left_bc, right_bc));
        let result = op.apply(&u, &m, &ctx(0.01)).unwrap();
        let div = result.as_scalar_field().unwrap();

        // div[0] = -D * ((values[1]-values[0]) - (values[0]-ghost_left)) / dx²
        let expected_0 = -d * ((values[1] - values[0]) - (values[0] - 2.0)) / (dx * dx);
        assert!((div[0] - expected_0).abs() < 1e-10, "got {}", div[0]);

        let n = values.len();
        let expected_n = -d * ((1.0 - values[n - 1]) - (values[n - 1] - values[n - 2])) / (dx * dx);
        assert!(
            (div[n - 1] - expected_n).abs() < 1e-10,
            "got {}",
            div[n - 1]
        );
    }

    #[test]
    fn centered_flux_ghost_cell_fails_explicitly_when_bc_has_no_ghost_value() {
        let m = mesh(4);
        let u = ContextValue::ScalarField(DVector::from_element(4, 1.0));
        let left_bc = Arc::new(NoGhost);
        let right_bc = Arc::new(FixedGhost(0.0));
        let op = FVCenteredFlux::new(0.0, 0.1, FluxBoundary::GhostCell(left_bc, right_bc));
        let err = op.apply(&u, &m, &ctx(0.001)).unwrap_err();
        assert!(matches!(err, OxiflowError::PreconditionFailed { .. }));
    }

    #[test]
    fn upwind_flux_ghost_cell_accepts_minimum_two_cells() {
        let m = mesh(2);
        let u = ContextValue::ScalarField(DVector::from_vec(vec![1.0, 2.0]));
        let left_bc = Arc::new(FixedGhost(0.0));
        let right_bc = Arc::new(FixedGhost(3.0));
        let op = FVUpwindFlux::new(0.5, 0.0, FluxBoundary::GhostCell(left_bc, right_bc));
        assert!(op.apply(&u, &m, &ctx(0.01)).is_ok());
    }

    // ── InvalidDomain on undersized field ──────────────────────────────────

    #[test]
    fn centered_flux_rejects_single_cell_field() {
        let m = mesh(5);
        let op = FVCenteredFlux::new(0.0, 0.0, FluxBoundary::Periodic);
        let u = ContextValue::ScalarField(DVector::from_vec(vec![1.0]));
        let err = op.apply(&u, &m, &ctx(0.001)).unwrap_err();
        assert!(matches!(err, OxiflowError::InvalidDomain(_)));
    }

    // ── RequiresContext ─────────────────────────────────────────────────────

    #[test]
    fn constant_parameters_require_no_context_variables() {
        let op = FVCenteredFlux::new(1.0, 0.1, FluxBoundary::Periodic);
        assert!(op.required_variables().is_empty());
    }
}
