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

// ── FVCenteredFlux ────────────────────────────────────────────────────────────

/// 2nd-order centered advective flux + centered Fick's-law diffusive flux.
///
/// `F(u, ∇u) = v·u − D·∂u/∂x`, with the advective term reconstructed as the
/// average of the two neighboring cell values at each face.
#[derive(Debug, Clone, Copy)]
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

        let div = match self.boundary {
            FluxBoundary::Periodic => periodic_divergence(u, dx, face_flux)?,
        };
        Ok(ContextValue::ScalarField(div))
    }
}

// ── FVUpwindFlux ──────────────────────────────────────────────────────────────

/// 1st-order upwind advective flux + centered Fick's-law diffusive flux.
///
/// `F(u, ∇u) = v·u − D·∂u/∂x`, with the advective term taken from the
/// upstream cell (direction of `v`) at each face — robust for strong
/// advection where `FVCenteredFlux` would oscillate.
#[derive(Debug, Clone, Copy)]
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

        let div = match self.boundary {
            FluxBoundary::Periodic => periodic_divergence(u, dx, face_flux)?,
        };
        Ok(ContextValue::ScalarField(div))
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn mesh(n: usize) -> UniformGrid1D {
        UniformGrid1D::new(n, 0.0, 1.0).unwrap()
    }

    fn ctx(dt: f64) -> ComputeContext {
        ComputeContext::new(0.0, dt)
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
