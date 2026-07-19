//! # Module `operators::adaptive`
//!
//! Adaptive WENO/limiter selection — [`AdaptiveFlux`] (#99, second half; the
//! mechanical half — `operators::limiters` — landed separately).
//!
//! ## What "local Péclet" actually means here
//!
//! The originating issue asked for a scheme that "activates the limiter
//! when local Péclet exceeds a threshold". Taken literally, that has no
//! meaning in this module's scope: the mesh Péclet number
//! `Pe = |v|·dx/D` is *constant* across the whole domain when velocity and
//! diffusion are themselves constant (the case both `operators::weno` and
//! `operators::limiters` are scoped to) — there is nothing "local" for it to
//! vary over. Resolved as follows (agreed before implementation, not
//! reinterpreted afterward):
//!
//! - **Péclet is a global on/off gate.** If `Pe ≤ peclet_threshold`
//!   (diffusion-dominated), the limiter never engages — every face uses
//!   `WENO3` alone, matching how a diffusion-dominated problem behaves: the
//!   diffusive term already damps the oscillations WENO's nonlinear
//!   weighting exists to suppress, so blending in a limiter buys nothing and
//!   only costs accuracy where WENO alone is already 3rd order.
//! - **The actual per-face, per-cell decision is smoothness-driven**,
//!   reusing [`WENO3::smoothness`] — a normalized `[0, 1)` indicator already
//!   available from the WENO-Z weighting formula, not a separate detector
//!   built for this module. `0` (smooth substencils) keeps the face at pure
//!   WENO; approaching `1` (a kink/discontinuity straddles the stencil)
//!   blends toward the limiter's reconstruction instead.
//!
//! Concretely, per face `i`:
//! ```text
//! gate  = smoothness(i)  if Pe > peclet_threshold, else 0
//! F(i)  = (1 − gate)·F_weno(i) + gate·F_limited(i)
//! ```
//! `gate = 0` everywhere recovers `WENO3` exactly (both the Péclet gate and
//! a genuinely smooth field produce this). `gate → 1` recovers
//! `LimitedFlux` at that face. Because `smoothness` only ever reaches `1` in
//! the limit and is otherwise a continuous function of the data, the blend
//! never has a hard discontinuous switch between the two schemes — avoiding
//! a spurious source of oscillation the switch itself could otherwise
//! introduce.
//!
//! ## Reuses `WENO3` and `LimitedFlux` directly — no reconstruction math
//! duplicated here
//!
//! `AdaptiveFlux` holds one `WENO3` and one `LimitedFlux` internally,
//! constructed with the same `velocity`/`diffusion`/`boundary`, and calls
//! their (already tested) `face_flux`/`smoothness`. This module contributes
//! only the blending policy above — same posture as `operators::limiters`
//! not re-deriving `operators::fv`'s flux formulas.
//!
//! ## `FluxBoundary` and stencil footprint
//!
//! `LimitedFlux`'s stencil is deliberately identical to `WENO3`'s (see
//! `operators::limiters`'s module doc) — the blended face flux therefore has
//! exactly `WENO3`'s footprint too, so `AdaptiveFlux` reuses the exact same
//! `Truncation`/`GhostCell` margins as `WENO3` (see its `apply()` for the
//! derivation).
//!
//! ## Scope note
//!
//! `WENO5` is not wired into this module — no concrete need for it here yet
//! (`WENO3` already demonstrates the blending policy; extending to `WENO5`
//! is additive, not a redesign, if a concrete need for the extra order
//! arises).
//!
//! [`WENO3::smoothness`]: crate::operators::weno::WENO3

use nalgebra::DVector;

use crate::context::compute::ComputeContext;
use crate::context::error::OxiflowError;
use crate::context::value::ContextValue;
use crate::context::variable::ContextVariable;
use crate::mesh::structured::UniformGrid1D;
use crate::mesh::Mesh;
use crate::model::traits::RequiresContext;
use crate::operators::limiters::{LimitedFlux, Limiter};
use crate::operators::weno::WENO3;
use crate::operators::{
    check_cfl, ghost_cell_wide_divergence, periodic_wide_divergence, truncated_wide_divergence,
    FluxBoundary, FluxDivergenceOperator,
};

/// Adaptive blend of [`WENO3`] and [`LimitedFlux`], gated by a global
/// mesh Péclet number and driven per-face by WENO's own smoothness
/// indicator — see the module documentation for the full rationale.
#[derive(Debug, Clone)]
pub struct AdaptiveFlux {
    velocity: f64,
    diffusion: f64,
    peclet_threshold: f64,
    weno: WENO3,
    limited: LimitedFlux,
    boundary: FluxBoundary,
}

impl AdaptiveFlux {
    /// Creates an adaptive WENO/limiter operator.
    ///
    /// `peclet_threshold` gates the mechanism: below it, every face uses
    /// `WENO3` alone (see the module documentation for why). `boundary` is
    /// shared by the internal `WENO3`/`LimitedFlux` instances — both need
    /// identical treatment since their blended result is used as a single
    /// consistent face flux.
    pub fn new(
        velocity: f64,
        diffusion: f64,
        limiter: Limiter,
        peclet_threshold: f64,
        boundary: FluxBoundary,
    ) -> Self {
        Self {
            velocity,
            diffusion,
            peclet_threshold,
            weno: WENO3::new(velocity, diffusion, boundary.clone()),
            limited: LimitedFlux::new(velocity, diffusion, limiter, boundary.clone()),
            boundary,
        }
    }

    /// Global mesh Péclet number `|v|·dx/D`. `D = 0.0` (pure advection)
    /// yields `+∞` via ordinary float division — always above any finite
    /// threshold, so the mechanism always engages for pure advection
    /// without a special case.
    fn peclet(&self, dx: f64) -> f64 {
        self.velocity.abs() * dx / self.diffusion
    }

    fn face_flux(&self, dx: f64, u: &DVector<f64>, n: usize, i: usize, gate_active: bool) -> f64 {
        let f_weno = self.weno.face_flux(dx, u, n, i);
        if !gate_active {
            return f_weno;
        }
        let gate = self.weno.smoothness(u, n, i);
        let f_limited = self.limited.face_flux(dx, u, n, i);
        (1.0 - gate) * f_weno + gate * f_limited
    }
}

impl RequiresContext for AdaptiveFlux {
    fn required_variables(&self) -> Vec<ContextVariable> {
        vec![]
    }
}

impl FluxDivergenceOperator for AdaptiveFlux {
    type MeshType = UniformGrid1D;

    fn apply(
        &self,
        field: &ContextValue,
        mesh: &Self::MeshType,
        ctx: &ComputeContext,
    ) -> Result<ContextValue, OxiflowError> {
        let u = field.as_scalar_field()?;
        let dx = mesh.characteristic_length();
        check_cfl("AdaptiveFlux", self.velocity, ctx.time_step(), dx)?;

        let gate_active = self.peclet(dx) > self.peclet_threshold;
        let face = |u: &DVector<f64>, n: usize, i: usize| self.face_flux(dx, u, n, i, gate_active);

        // Same stencil footprint and margins as WENO3 — see the module doc.
        let div = match &self.boundary {
            FluxBoundary::Periodic => periodic_wide_divergence(u, dx, 3, "AdaptiveFlux", face)?,
            FluxBoundary::Truncation => {
                let (margin_left, margin_right) =
                    if self.velocity >= 0.0 { (1, 1) } else { (0, 2) };
                truncated_wide_divergence(u, dx, margin_left, margin_right, "AdaptiveFlux", face)?
            }
            FluxBoundary::GhostCell(left_bc, right_bc) => {
                let margins = if self.velocity >= 0.0 { (2, 1) } else { (1, 2) };
                ghost_cell_wide_divergence(
                    u,
                    dx,
                    margins,
                    (left_bc.as_ref(), right_bc.as_ref()),
                    "AdaptiveFlux",
                    face,
                )?
            }
        };
        Ok(ContextValue::ScalarField(div))
    }

    fn stencil_radius(&self) -> usize {
        // Same footprint as WENO3 (see module doc) — delegate rather than
        // repeat the literal, so the two can never drift apart.
        self.weno.stencil_radius()
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

    // ── Péclet gate ─────────────────────────────────────────────────────────

    #[test]
    fn below_peclet_threshold_matches_pure_weno3_exactly() {
        // Small velocity, large diffusion => low Péclet => gate never
        // engages, regardless of how steep the data is.
        let m = mesh(8);
        let dx = m.characteristic_length();
        let u = ContextValue::ScalarField(DVector::from_vec(vec![
            0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0,
        ]));
        let velocity = 0.1;
        let diffusion = 10.0;
        let op = AdaptiveFlux::new(
            velocity,
            diffusion,
            Limiter::Superbee,
            1000.0, // threshold far above this problem's Péclet number
            FluxBoundary::Periodic,
        );
        let weno = WENO3::new(velocity, diffusion, FluxBoundary::Periodic);

        let adaptive_result = op.apply(&u, &m, &ctx(0.001 * dx)).unwrap();
        let weno_result = weno.apply(&u, &m, &ctx(0.001 * dx)).unwrap();
        let adaptive = adaptive_result.as_scalar_field().unwrap();
        let expected = weno_result.as_scalar_field().unwrap();
        for i in 0..adaptive.len() {
            assert!(
                (adaptive[i] - expected[i]).abs() < 1e-12,
                "cell {i}: adaptive={}, pure WENO3={}",
                adaptive[i],
                expected[i]
            );
        }
    }

    #[test]
    fn above_peclet_threshold_diverges_from_pure_weno3_on_a_step() {
        // Large velocity, tiny diffusion => high Péclet => gate can engage;
        // a step function gives WENO3::smoothness a large value near the
        // discontinuity, so the blend should differ measurably from pure
        // WENO3 there.
        let m = mesh(8);
        let dx = m.characteristic_length();
        let u = ContextValue::ScalarField(DVector::from_vec(vec![
            0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0,
        ]));
        let velocity = 1.0;
        let diffusion = 1e-6;
        let op = AdaptiveFlux::new(
            velocity,
            diffusion,
            Limiter::Superbee,
            1.0, // low threshold — this problem's Péclet is far above it
            FluxBoundary::Periodic,
        );
        let weno = WENO3::new(velocity, diffusion, FluxBoundary::Periodic);

        let adaptive_result = op.apply(&u, &m, &ctx(0.0001 * dx)).unwrap();
        let weno_result = weno.apply(&u, &m, &ctx(0.0001 * dx)).unwrap();
        let adaptive = adaptive_result.as_scalar_field().unwrap();
        let pure_weno = weno_result.as_scalar_field().unwrap();
        let max_diff = (0..adaptive.len())
            .map(|i| (adaptive[i] - pure_weno[i]).abs())
            .fold(0.0, f64::max);
        assert!(
            max_diff > 1e-8,
            "expected a measurable difference from pure WENO3 near the step, got max diff {max_diff}"
        );
    }

    #[test]
    fn peclet_is_infinite_for_pure_advection_and_always_gates() {
        let op = AdaptiveFlux::new(1.0, 0.0, Limiter::MinMod, 1e6, FluxBoundary::Periodic);
        assert!(op.peclet(0.1).is_infinite());
    }

    // ── Conservation property (periodic) — holds regardless of the blend ──

    #[test]
    fn conserves_sum_for_periodic_problem() {
        let m = mesh(6);
        let dx = m.characteristic_length();
        let u = ContextValue::ScalarField(DVector::from_vec(vec![0.2, 1.3, -0.7, 2.1, 0.0, -1.5]));
        let op = AdaptiveFlux::new(0.8, 0.05, Limiter::VanLeer, 0.5, FluxBoundary::Periodic);
        let result = op.apply(&u, &m, &ctx(0.1 * dx)).unwrap();
        let sum: f64 = result.as_scalar_field().unwrap().iter().sum();
        assert!(sum.abs() < 1e-9, "expected ~0, got {sum}");
    }

    // ── FluxBoundary::Truncation / GhostCell wiring (same margins as WENO3) ─

    #[test]
    fn truncation_boundary_does_not_panic_and_matches_weno3_margins() {
        let m = mesh(7);
        let dx = m.characteristic_length();
        let u =
            ContextValue::ScalarField(DVector::from_vec(vec![0.2, 1.3, -0.7, 2.1, 0.0, -1.5, 0.6]));
        let op = AdaptiveFlux::new(1.0, 1e-6, Limiter::MinMod, 1.0, FluxBoundary::Truncation);
        assert!(op.apply(&u, &m, &ctx(0.01 * dx)).is_ok());
    }

    #[test]
    fn truncation_rejects_field_below_minimum_for_direction() {
        let m = mesh(3);
        let op = AdaptiveFlux::new(1.0, 1e-6, Limiter::MinMod, 1.0, FluxBoundary::Truncation);
        let u = ContextValue::ScalarField(DVector::from_element(3, 1.0));
        let err = op.apply(&u, &m, &ctx(0.001)).unwrap_err();
        assert!(matches!(err, OxiflowError::InvalidDomain(_)));
    }

    // ── CFL check ───────────────────────────────────────────────────────────

    #[test]
    fn rejects_cfl_violation() {
        let m = mesh(5);
        let op = AdaptiveFlux::new(10.0, 0.0, Limiter::VanLeer, 1.0, FluxBoundary::Periodic);
        let u = ContextValue::ScalarField(DVector::from_element(5, 1.0));
        let err = op.apply(&u, &m, &ctx(1.0)).unwrap_err();
        assert!(matches!(err, OxiflowError::PreconditionFailed { .. }));
    }
}
