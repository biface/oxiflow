//! # Module `operators::limiters`
//!
//! Flux-limited (MUSCL-style) advective flux + centered diffusive flux for
//! [`UniformGrid1D`] — [`Limiter`] (`MinMod`, `VanLeer`, `Superbee`) and
//! [`LimitedFlux`] (#99, split out of `#49` at closing time — DD-012,
//! DD-039, DD-041).
//!
//! ## Why a limiter, alongside FV and WENO
//!
//! `operators::fv::FVUpwindFlux` is 1st order (diffusive on sharp fronts);
//! `operators::fv::FVCenteredFlux` and `operators::weno` are higher order
//! but need a smoothness-driven blend (WENO's nonlinear weights) or, here, a
//! slope limiter to stay non-oscillatory near a discontinuity while
//! recovering 2nd order where the solution is smooth. A limiter is
//! considerably cheaper than WENO's multi-substencil reconstruction —
//! useful on its own for advection-dominated regions where WENO's extra
//! cost buys little, and as the "steep" half of the adaptive WENO/limiter
//! blend (a separate, not-yet-designed piece of work — see the tracking
//! note at the end of this doc).
//!
//! ## MUSCL reconstruction, not a flux blend
//!
//! Two equivalent formulations exist in the literature: blending a low- and
//! a high-order *flux* with `φ(r)` (Sweby, 1984), or directly building a
//! limited-slope *reconstruction* of the face value. This module uses the
//! latter — one reconstructed face value per direction, not two flux
//! evaluations combined — since it avoids re-deriving/re-exposing
//! `operators::fv`'s low/high flux formulas as a dependency of this module;
//! the two are mathematically identical for this scalar `F = v·u` advective
//! term.
//!
//! For `v ≥ 0` (window `{a, b, c} = {u[i−1], u[i], u[i+1]}`, the same
//! offsets `operators::weno`'s `weno3_left` uses):
//! ```text
//! r = (b − a) / (c − b)                  (upstream slope / downstream slope)
//! u_face = b + 0.5·φ(r)·(c − b)
//! ```
//! `v < 0` mirrors this (window `{b, c, d} = {u[i], u[i+1], u[i+2]}`, same
//! offsets as `weno3_right`):
//! ```text
//! r = (d − c) / (c − b)
//! u_face = c − 0.5·φ(r)·(c − b)
//! ```
//! `φ(r) = 0` recovers pure upwind (1st order, always non-oscillatory);
//! `φ(r) = 1` recovers a centered/Lax–Wendroff-type correction (2nd order,
//! can oscillate) — every limiter here interpolates between the two based
//! on the local ratio of consecutive slopes `r`. The diffusive term is
//! always the direct centered difference between the two nodes adjacent to
//! the face, exactly as in `operators::fv`/`operators::weno`.
//!
//! ## Cell semantics
//!
//! Same cell-average interpretation as `operators::fv` (DD-040) and
//! `operators::weno` (DD-041) — `u[i]` is the average of `u` over the cell
//! whose faces sit at nodes `i` and `i+1`.
//!
//! ## `r = 0/0` (locally constant field)
//!
//! `ratio` (the helper computing `r`) returns `0.0` when the denominator is smaller than a fixed
//! epsilon, rather than propagating a `NaN`. This is safe in both cases it
//! covers: a genuinely constant local field also has a numerator near zero,
//! so `c − b ≈ 0` makes the face value insensitive to `φ`'s exact value
//! regardless; a genuine local extremum (numerator non-negligible, opposite
//! or discontinuous slope) is exactly where every limiter here is designed
//! to clip toward pure upwind — every `Limiter::phi` maps `r ≤ 0` to `0`, so
//! defaulting the ill-conditioned ratio to `0.0` lands on the same
//! non-oscillatory choice a genuine extremum would demand, not an arbitrary
//! one.
//!
//! ## `FluxBoundary` and CFL
//!
//! Reuses [`crate::operators::FluxBoundary`], [`crate::operators::check_cfl`],
//! and the same wide-stencil boundary helpers `operators::weno` uses
//! (`periodic_wide_divergence`, `truncated_wide_divergence`,
//! `ghost_cell_wide_divergence`, relocated to `operators` — see their
//! module-level doc there) — `LimitedFlux`'s stencil footprint is exactly
//! `WENO3`'s (`{i−1, i, i+1}` for `v ≥ 0`, `{i, i+1, i+2}` for `v < 0`), so
//! the same margins apply: `Truncation` uses `(1, 1)`/`(0, 2)`;
//! `GhostCell` uses `(2, 1)`/`(1, 2)` (one extra layer on the left — see
//! `operators::weno`'s `GhostCell` match arm for why).
//!
//! ## Adaptive WENO/limiter selection — not in this module
//!
//! The originating issue (`#99`) also calls for an adaptive scheme that
//! picks WENO where the solution is smooth and a limiter where it is steep.
//! That selection logic is intentionally **not** implemented here — it
//! needs its own design pass (candidate approach: reuse `operators::weno`'s
//! already-computed smoothness indicators `β` per cell, gated by a
//! global Péclet-number threshold that decides whether the mechanism
//! engages at all, since velocity/diffusion are constant in this module's
//! scope and a genuinely *local* Péclet number has no meaning here). Tracked
//! as follow-up work, not deferred silently.
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
use crate::operators::{
    check_cfl, ghost_cell_wide_divergence, periodic_wide_divergence, truncated_wide_divergence,
    wrap, FluxBoundary, FluxDivergenceOperator,
};

/// Ratio of consecutive slopes with a safe `0/0` fallback — see the module
/// documentation ("`r = 0/0`") for why `0.0` is the correct default, not
/// merely a convenient one.
fn ratio(numer: f64, denom: f64) -> f64 {
    const EPS: f64 = 1e-12;
    if denom.abs() < EPS {
        0.0
    } else {
        numer / denom
    }
}

// ── Limiter ───────────────────────────────────────────────────────────────────

/// Slope limiter function `φ(r)`, `r` the ratio of consecutive slopes.
///
/// All three satisfy `φ(r) = 0` for `r ≤ 0` (no correction at a local
/// extremum or discontinuous slope — pure upwind, non-oscillatory) and
/// `φ(1) = 1` (exact for a locally linear field, 2nd order there). They
/// differ in how much high-order correction they allow for `r > 1`:
/// `MinMod` caps at `1` (most diffusive/restrictive), `Superbee` reaches `2`
/// fastest (least diffusive, most compressive — can steepen contact
/// discontinuities), `VanLeer` sits strictly between the two — a smooth,
/// differentiable compromise. This ordering
/// (`MinMod ≤ VanLeer ≤ Superbee` for `r > 1`) is `#99`'s acceptance
/// criterion, checked directly in the tests below.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Limiter {
    /// `φ(r) = max(0, min(1, r))` — most diffusive of the three.
    MinMod,
    /// `φ(r) = (r + |r|) / (1 + |r|)` — smooth, strictly between `MinMod`
    /// and `Superbee` for `r > 1`.
    VanLeer,
    /// `φ(r) = max(0, min(2r, 1), min(r, 2))` — least diffusive, saturates
    /// at `2`.
    Superbee,
}

impl Limiter {
    /// Evaluates `φ(r)`.
    pub fn phi(&self, r: f64) -> f64 {
        match self {
            Limiter::MinMod => r.clamp(0.0, 1.0),
            Limiter::VanLeer => (r + r.abs()) / (1.0 + r.abs()),
            Limiter::Superbee => {
                let a = (2.0 * r).min(1.0);
                let b = r.min(2.0);
                a.max(b).max(0.0)
            }
        }
    }
}

// ── LimitedFlux ───────────────────────────────────────────────────────────────

/// Flux-limited (MUSCL) advective flux + centered Fick's-law diffusive
/// flux, 2nd order where smooth, non-oscillatory near discontinuities.
///
/// `F(u, ∇u) = v·u − D·∂u/∂x` — see the module documentation for the
/// reconstruction formula and boundary treatment.
#[derive(Debug, Clone)]
pub struct LimitedFlux {
    velocity: f64,
    diffusion: f64,
    limiter: Limiter,
    boundary: FluxBoundary,
}

impl LimitedFlux {
    /// Creates a flux-limited operator using `limiter`.
    pub fn new(velocity: f64, diffusion: f64, limiter: Limiter, boundary: FluxBoundary) -> Self {
        Self {
            velocity,
            diffusion,
            limiter,
            boundary,
        }
    }

    fn face_flux(&self, dx: f64, u: &DVector<f64>, n: usize, i: usize) -> f64 {
        let v = self.velocity;
        let u_face = if v >= 0.0 {
            let (a, b, c) = (u[wrap(i, -1, n)], u[i], u[wrap(i, 1, n)]);
            let r = ratio(b - a, c - b);
            b + 0.5 * self.limiter.phi(r) * (c - b)
        } else {
            let (b, c, d) = (u[i], u[wrap(i, 1, n)], u[wrap(i, 2, n)]);
            let r = ratio(d - c, c - b);
            c - 0.5 * self.limiter.phi(r) * (c - b)
        };
        let diffusive = self.diffusion * (u[wrap(i, 1, n)] - u[i]) / dx;
        v * u_face - diffusive
    }
}

impl RequiresContext for LimitedFlux {
    fn required_variables(&self) -> Vec<ContextVariable> {
        // Constant velocity/diffusion — nothing to resolve from ComputeContext
        // (the "simple case" of DD-039, same posture as FV/WENO).
        vec![]
    }
}

impl FluxDivergenceOperator for LimitedFlux {
    type MeshType = UniformGrid1D;

    fn apply(
        &self,
        field: &ContextValue,
        mesh: &Self::MeshType,
        ctx: &ComputeContext,
    ) -> Result<ContextValue, OxiflowError> {
        let u = field.as_scalar_field()?;
        let dx = mesh.characteristic_length();
        check_cfl("LimitedFlux", self.velocity, ctx.time_step(), dx)?;

        let div = match &self.boundary {
            FluxBoundary::Periodic => {
                periodic_wide_divergence(u, dx, 3, "LimitedFlux", |u, n, i| {
                    self.face_flux(dx, u, n, i)
                })?
            }
            FluxBoundary::Truncation => {
                // Same stencil footprint as WENO3 — see the module doc.
                let (margin_left, margin_right) =
                    if self.velocity >= 0.0 { (1, 1) } else { (0, 2) };
                truncated_wide_divergence(
                    u,
                    dx,
                    margin_left,
                    margin_right,
                    "LimitedFlux",
                    |u, n, i| self.face_flux(dx, u, n, i),
                )?
            }
            FluxBoundary::GhostCell(left_bc, right_bc) => {
                let margins = if self.velocity >= 0.0 { (2, 1) } else { (1, 2) };
                ghost_cell_wide_divergence(
                    u,
                    dx,
                    margins,
                    (left_bc.as_ref(), right_bc.as_ref()),
                    "LimitedFlux",
                    |u, n, i| self.face_flux(dx, u, n, i),
                )?
            }
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

    // ── Limiter properties (#99 acceptance criterion) ──────────────────────

    #[test]
    fn all_limiters_vanish_at_nonpositive_r() {
        for limiter in [Limiter::MinMod, Limiter::VanLeer, Limiter::Superbee] {
            for r in [-5.0, -1.0, 0.0] {
                assert_eq!(limiter.phi(r), 0.0, "{limiter:?} at r={r}");
            }
        }
    }

    #[test]
    fn all_limiters_are_exact_at_r_equals_one() {
        for limiter in [Limiter::MinMod, Limiter::VanLeer, Limiter::Superbee] {
            assert!((limiter.phi(1.0) - 1.0).abs() < 1e-12, "{limiter:?} at r=1");
        }
    }

    #[test]
    fn minmod_is_most_diffusive_superbee_is_least() {
        // For r > 1, MinMod caps at 1 (smallest correction — most diffusive),
        // Superbee grows fastest toward 2 (largest correction — least
        // diffusive), VanLeer sits strictly between the two.
        for r in [1.5, 2.0, 3.0, 10.0] {
            let minmod = Limiter::MinMod.phi(r);
            let vanleer = Limiter::VanLeer.phi(r);
            let superbee = Limiter::Superbee.phi(r);
            assert!(
                minmod <= vanleer && vanleer <= superbee,
                "r={r}: expected MinMod({minmod}) <= VanLeer({vanleer}) <= Superbee({superbee})"
            );
        }
    }

    #[test]
    fn minmod_caps_at_one_superbee_caps_at_two() {
        assert_eq!(Limiter::MinMod.phi(100.0), 1.0);
        assert_eq!(Limiter::Superbee.phi(100.0), 2.0);
    }

    // ── r = 0/0 safety ──────────────────────────────────────────────────────

    #[test]
    fn constant_field_produces_finite_zero_divergence() {
        let m = mesh(6);
        let dx = m.characteristic_length();
        let u = ContextValue::ScalarField(DVector::from_element(6, 2.5));
        for limiter in [Limiter::MinMod, Limiter::VanLeer, Limiter::Superbee] {
            let op = LimitedFlux::new(0.7, 0.1, limiter, FluxBoundary::Periodic);
            let result = op.apply(&u, &m, &ctx(0.01 * dx)).unwrap();
            for &v in result.as_scalar_field().unwrap().iter() {
                assert!(v.is_finite(), "{limiter:?}: non-finite divergence");
                assert!(v.abs() < 1e-10, "{limiter:?}: expected ~0, got {v}");
            }
        }
    }

    // ── Conservation property (periodic) ───────────────────────────────────

    #[test]
    fn limited_flux_conserves_sum_for_periodic_problem() {
        let m = mesh(6);
        let dx = m.characteristic_length();
        let u = ContextValue::ScalarField(DVector::from_vec(vec![0.2, 1.3, -0.7, 2.1, 0.0, -1.5]));
        let op = LimitedFlux::new(0.8, 0.05, Limiter::VanLeer, FluxBoundary::Periodic);
        let result = op.apply(&u, &m, &ctx(0.1 * dx)).unwrap();
        let sum: f64 = result.as_scalar_field().unwrap().iter().sum();
        assert!(sum.abs() < 1e-10, "expected 0, got {sum}");
    }

    // ── Stays within data bounds on a step (non-oscillatory, TVD spirit) ───

    #[test]
    fn superbee_reconstruction_stays_within_data_bounds_on_step() {
        let dx = 1.0; // irrelevant here: only the reconstructed face value is checked
        let values = vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0];
        let uv = DVector::from_vec(values);
        let n = uv.len();
        let op = LimitedFlux::new(1.0, 0.0, Limiter::Superbee, FluxBoundary::Periodic);
        let face = op.face_flux(dx, &uv, n, 2); // face straddling the step
                                                // diffusion = 0 here, so face_flux = v * u_face = u_face (v = 1.0).
        assert!(
            (-0.05..=1.05).contains(&face),
            "expected reconstruction within [0, 1] (± tolerance), got {face}"
        );
    }

    // ── FluxBoundary::Truncation ────────────────────────────────────────────

    #[test]
    fn truncation_left_biased_boundary_matches_nearest_safe_cell() {
        let m = mesh(7);
        let dx = m.characteristic_length();
        let u =
            ContextValue::ScalarField(DVector::from_vec(vec![0.2, 1.3, -0.7, 2.1, 0.0, -1.5, 0.6]));
        let op = LimitedFlux::new(1.0, 0.0, Limiter::VanLeer, FluxBoundary::Truncation);
        let div = op.apply(&u, &m, &ctx(0.01 * dx)).unwrap();
        let div = div.as_scalar_field().unwrap();
        assert_eq!(div[0], div[1]);
        assert_eq!(div[6], div[5]);
    }

    #[test]
    fn truncation_rejects_field_below_minimum_for_direction() {
        let m = mesh(3);
        let op = LimitedFlux::new(1.0, 0.0, Limiter::MinMod, FluxBoundary::Truncation);
        let u = ContextValue::ScalarField(DVector::from_element(3, 1.0));
        let err = op.apply(&u, &m, &ctx(0.001)).unwrap_err();
        assert!(matches!(err, OxiflowError::InvalidDomain(_)));
    }

    // ── FluxBoundary::GhostCell ─────────────────────────────────────────────

    #[derive(Debug)]
    struct DirichletGhost(f64);

    impl RequiresContext for DirichletGhost {
        fn required_variables(&self) -> Vec<ContextVariable> {
            vec![]
        }
    }

    impl crate::boundary::BoundaryCondition for DirichletGhost {
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
        fn ghost_value(&self, _depth: usize, interior_at_depth: f64, _dx: f64) -> Option<f64> {
            Some(2.0 * self.0 - interior_at_depth)
        }
    }

    #[test]
    fn ghost_cell_left_biased_matches_hand_built_extended_field() {
        use std::sync::Arc;

        let values = vec![0.2, 1.3, -0.7, 2.1, 0.0];
        let m = mesh(values.len());
        let dx = m.characteristic_length();
        let u = ContextValue::ScalarField(DVector::from_vec(values.clone()));
        let g = 0.5;
        let left_bc = Arc::new(DirichletGhost(g));
        let right_bc = Arc::new(DirichletGhost(g));
        let op = LimitedFlux::new(
            1.0,
            0.0,
            Limiter::VanLeer,
            FluxBoundary::GhostCell(left_bc, right_bc),
        );
        let div = op.apply(&u, &m, &ctx(0.001 * dx)).unwrap();
        let div = div.as_scalar_field().unwrap();

        let n = values.len();
        let ghost_m2 = 2.0 * g - values[1];
        let ghost_m1 = 2.0 * g - values[0];
        let ghost_p1 = 2.0 * g - values[n - 1];
        let mut extended = vec![ghost_m2, ghost_m1];
        extended.extend(values.iter().copied());
        extended.push(ghost_p1);
        let ext = DVector::from_vec(extended);
        let m_ext = ext.len();

        let probe = LimitedFlux::new(1.0, 0.0, Limiter::VanLeer, FluxBoundary::Periodic);
        let face = |i: usize| probe.face_flux(dx, &ext, m_ext, i);
        let expected_0 = (face(2) - face(1)) / dx;
        let expected_last = (face(2 + n - 1) - face(2 + n - 2)) / dx;
        assert!((div[0] - expected_0).abs() < 1e-10, "got {}", div[0]);
        assert!(
            (div[n - 1] - expected_last).abs() < 1e-10,
            "got {}",
            div[n - 1]
        );
    }

    #[test]
    fn ghost_cell_fails_explicitly_when_bc_has_no_ghost_value() {
        use std::sync::Arc;

        #[derive(Debug)]
        struct NoGhostBC;
        impl RequiresContext for NoGhostBC {
            fn required_variables(&self) -> Vec<ContextVariable> {
                vec![]
            }
        }
        impl crate::boundary::BoundaryCondition for NoGhostBC {
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

        let m = mesh(5);
        let u = ContextValue::ScalarField(DVector::from_element(5, 1.0));
        let left_bc = Arc::new(NoGhostBC);
        let right_bc = Arc::new(DirichletGhost(0.0));
        let op = LimitedFlux::new(
            1.0,
            0.0,
            Limiter::MinMod,
            FluxBoundary::GhostCell(left_bc, right_bc),
        );
        let err = op.apply(&u, &m, &ctx(0.001)).unwrap_err();
        assert!(matches!(err, OxiflowError::PreconditionFailed { .. }));
    }

    // ── CFL check ───────────────────────────────────────────────────────────

    #[test]
    fn rejects_cfl_violation() {
        let m = mesh(5); // dx = 0.25
        let op = LimitedFlux::new(10.0, 0.0, Limiter::VanLeer, FluxBoundary::Periodic);
        let u = ContextValue::ScalarField(DVector::from_element(5, 1.0));
        let err = op.apply(&u, &m, &ctx(1.0)).unwrap_err();
        assert!(matches!(err, OxiflowError::PreconditionFailed { .. }));
    }

    // ── RequiresContext ─────────────────────────────────────────────────────

    #[test]
    fn constant_parameters_require_no_context_variables() {
        let op = LimitedFlux::new(1.0, 0.1, Limiter::MinMod, FluxBoundary::Periodic);
        assert!(op.required_variables().is_empty());
    }
}
