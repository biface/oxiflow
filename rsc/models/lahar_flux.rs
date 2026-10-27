//! Lahar multi-field model, flux part: `ShallowLayerFlux`, the
//! `FluxDivergenceOperator` of the `(m, p, s)` system (Rusanov flux,
//! MinMod-limited MUSCL reconstruction).
//!
//! `ShallowLayerFlux::divergence` takes one treatment per domain edge
//! (`EdgeTreatment`): the truncation posture of `apply`, a reflective wall,
//! or a face left to a coupling operator. `ShallowLayerFluxWithEdges` wraps
//! the operator with chosen treatments, for the coupled-grid example.
//!
//! Single definition shared by `examples/lahar_multifield.rs`,
//! `examples/lahar_multifield_plot.rs`, `tests/lahar_multifield_analytical.rs`
//! and `benches/lahar_multifield.rs`, each of which includes this file as a
//! module with `#[path]`. The model (DD-032, Woodhouse, Hogg & Phillips) is
//! documented in `examples/lahar_multifield.rs`.
//!
//! Requires `rsc/models/lahar_physics.rs` as a sibling module named
//! `lahar_physics`.

use super::lahar_physics::{LaharPhysics, COL_M, COL_P, COL_S};
use nalgebra::DMatrix;
use oxiflow::{
    context::{
        compute::ComputeContext, error::OxiflowError, value::ContextValue,
        variable::ContextVariable,
    },
    mesh::{Mesh, UniformGrid1D},
    model::RequiresContext,
    operators::{limiters::Limiter, FluxDivergenceOperator},
};

// ── Flux operator ─────────────────────────────────────────────────────────────

/// `FluxDivergenceOperator` for the coupled `(m, p, s)` shallow-layer
/// system -- Rusanov numerical flux, MinMod-limited MUSCL reconstruction.
/// `b`'s own flux-divergence contribution is always zero (no spatial flux
/// term for bed elevation in the governing equations).
pub(crate) struct ShallowLayerFlux {
    pub(crate) physics: LaharPhysics,
    pub(crate) limiter: Limiter,
    /// Rayon dispatch threshold for the divergence loop (two
    /// [`ShallowLayerFlux::rusanov_face`] evaluations per node --
    /// primitives + physical flux each side) -- default measured at
    /// v0.9.0, see [`default_flux_parallel_threshold`]. Per-
    /// instance, not global (DD-048 P1/P2/P3): this site's own coupling
    /// granularity (one node's flux touches only its immediate face
    /// neighbours) has nothing to do with `LaharSource`'s per-node cost
    /// profile, so each gets its own threshold rather than sharing one.
    #[cfg(feature = "parallel")]
    pub(crate) parallel_threshold: usize,
}

/// Measured: `benches/lahar_multifield.rs`'s `flux_dispatch_diagnostic` put the real crossover
/// between 1,000 and 10,000 -- but 1,000 itself flips sign between runs
/// (session: 1.18x slower in one run, 1.09x faster in another, on the
/// same machine), so the default sits at 10,000, past that noisy
/// boundary, where the win is consistent and stable (~3-4x from 10,000
/// through 1,000,000), measured at v0.9.0 (#53). At v0.9.0 the diagnostic
/// timed a copy of the divergence loop; it now times this operator's
/// `apply`, so the numbers above hold until the next measurement.
#[cfg(feature = "parallel")]
pub(crate) fn default_flux_parallel_threshold() -> usize {
    10_000
}

impl ShallowLayerFlux {
    fn physical_flux(&self, m: f64, p: f64, s: f64, h: f64, u: f64, psi: f64) -> [f64; 3] {
        self.physics.physical_flux(m, p, s, h, u, psi)
    }

    /// `r = (b-a)/(c-b)`, epsilon-guarded -- reproduces
    /// `operators::limiters::ratio` (private there, so restated here
    /// rather than relying on visibility this example doesn't have).
    fn ratio(numer: f64, denom: f64) -> f64 {
        const EPS: f64 = 1.0e-12;
        if denom.abs() < EPS {
            0.0
        } else {
            numer / denom
        }
    }

    /// One MinMod-limited slope per cell, extrapolated to both edges --
    /// see `examples/lahar_multifield.rs`'s module doc for why this is
    /// simpler than `LimitedFlux`'s
    /// own velocity-direction-dependent window. Domain-edge cells (`i==0`
    /// or `i==n-1`) fall back to first order (no neighbour on one side).
    fn reconstruct(&self, col: &[f64], i: usize, n: usize) -> (f64, f64) {
        if i == 0 || i == n - 1 {
            return (col[i], col[i]);
        }
        let (a, b, c) = (col[i - 1], col[i], col[i + 1]);
        let r = Self::ratio(b - a, c - b);
        let slope = self.limiter.phi(r) * (c - b);
        (b - 0.5 * slope, b + 0.5 * slope)
    }

    fn rusanov_face(&self, left: [f64; 3], right: [f64; 3]) -> Result<[f64; 3], OxiflowError> {
        let (hl, ul, psil) = self.physics.primitives(left[0], left[1], left[2])?;
        let (hr, ur, psir) = self.physics.primitives(right[0], right[1], right[2])?;
        let fl = self.physical_flux(left[0], left[1], left[2], hl, ul, psil);
        let fr = self.physical_flux(right[0], right[1], right[2], hr, ur, psir);
        let s_max = self
            .physics
            .wave_speed(hl, ul)
            .max(self.physics.wave_speed(hr, ur));
        let mut face = [0.0; 3];
        for (k, entry) in face.iter_mut().enumerate() {
            *entry = 0.5 * (fl[k] + fr[k]) - 0.5 * s_max * (right[k] - left[k]);
        }
        Ok(face)
    }
}

impl RequiresContext for ShallowLayerFlux {
    fn required_variables(&self) -> Vec<ContextVariable> {
        // Constant rho_f/rho_s -- nothing to resolve from ComputeContext,
        // the "simple case" of DD-039 (same posture as FV/WENO).
        vec![]
    }
}

impl ShallowLayerFlux {
    /// Flux divergence of the `(m, p, s)` system, with one treatment per
    /// domain edge (see [`EdgeTreatment`]). [`FluxDivergenceOperator::apply`]
    /// is this method with [`EdgeTreatment::NeighbourDivergence`] on both
    /// edges; [`ShallowLayerFluxWithEdges`] selects the treatments.
    pub(crate) fn divergence(
        &self,
        field: &ContextValue,
        mesh: &UniformGrid1D,
        ctx: &ComputeContext,
        edges: Edges,
    ) -> Result<ContextValue, OxiflowError> {
        let state = field.as_vector_field()?;
        let n = state.nrows();
        if n < 3 {
            return Err(OxiflowError::InvalidDomain(format!(
                "ShallowLayerFlux requires at least 3 nodes, got {n}"
            )));
        }
        let dx = mesh.characteristic_length();
        let dt = ctx.time_step();

        // CFL check reproduced directly -- operators::check_cfl is
        // pub(crate), not visible from an example. Uses the raw
        // (unreconstructed) cell wave speeds as a cheap upper bound.
        let mut max_speed = 0.0_f64;
        let mut max_at = 0usize;
        for i in 0..n {
            let (h, u, _) =
                self.physics
                    .primitives(state[(i, COL_M)], state[(i, COL_P)], state[(i, COL_S)])?;
            let speed = self.physics.wave_speed(h, u);
            if speed > max_speed {
                max_speed = speed;
                max_at = i;
            }
        }
        if max_speed * dt / dx > 1.0 {
            let (h, u, psi) = self.physics.primitives(
                state[(max_at, COL_M)],
                state[(max_at, COL_P)],
                state[(max_at, COL_S)],
            )?;
            return Err(OxiflowError::PreconditionFailed {
                context: "ShallowLayerFlux::apply",
                message: format!(
                    "CFL condition violated: max_wave_speed*dt/dx = {:.6} > 1.0 \
                     (max_wave_speed = {max_speed}, dt = {dt}, dx = {dx}) at node {max_at} \
                     (x = {:.3}, t = {:.6}): h={h}, u={u}, psi={psi}, m={}, p={}, s={}",
                    max_speed * dt / dx,
                    mesh.coordinates(max_at)[0],
                    ctx.time(),
                    state[(max_at, COL_M)],
                    state[(max_at, COL_P)],
                    state[(max_at, COL_S)],
                ),
            });
        }

        // Reconstructed left/right edge value per cell, per component.
        let mut left_edge = vec![[0.0_f64; 3]; n];
        let mut right_edge = vec![[0.0_f64; 3]; n];
        for (k, col_idx) in [COL_M, COL_P, COL_S].into_iter().enumerate() {
            let col: Vec<f64> = (0..n).map(|i| state[(i, col_idx)]).collect();
            for i in 0..n {
                let (l, r) = self.reconstruct(&col, i, n);
                left_edge[i][k] = l;
                right_edge[i][k] = r;
            }
        }

        // Real per-node cost here: two `rusanov_face` calls (each a
        // `primitives` solve + `physical_flux` evaluation per side) --
        // independent across `i`, reading only the already-computed
        // `left_edge`/`right_edge` immutably. Parallel results collected
        // into a `Vec` first, then copied sequentially into `div` --
        // the same DMatrix column-major consequence `langmuir_multi`
        // (#138) documents for its own row-parallel assembly.
        let compute_divergence = |i: usize| -> Result<[f64; 3], OxiflowError> {
            let right_face = self.rusanov_face(right_edge[i], left_edge[i + 1])?;
            let left_face = self.rusanov_face(right_edge[i - 1], left_edge[i])?;
            let mut d = [0.0; 3];
            for k in 0..3 {
                d[k] = (right_face[k] - left_face[k]) / dx;
            }
            Ok(d)
        };

        #[cfg(feature = "parallel")]
        let interior: Vec<[f64; 3]> = if (n - 2) >= self.parallel_threshold {
            use rayon::prelude::*;
            (1..n - 1)
                .into_par_iter()
                .map(compute_divergence)
                .collect::<Result<Vec<_>, _>>()?
        } else {
            (1..n - 1)
                .map(compute_divergence)
                .collect::<Result<Vec<_>, _>>()?
        };
        #[cfg(not(feature = "parallel"))]
        let interior: Vec<[f64; 3]> = (1..n - 1)
            .map(compute_divergence)
            .collect::<Result<Vec<_>, _>>()?;

        let mut div = DMatrix::zeros(n, 4); // column COL_B stays 0.
        for (offset, d) in interior.into_iter().enumerate() {
            let i = offset + 1;
            for (k, val) in d.into_iter().enumerate() {
                div[(i, k)] = val;
            }
        }
        // Edge cells, one treatment per side. `None`: truncation posture
        // (operators::fv/limiters' own documented choice, reproduced
        // here), the edge cell reuses the nearest interior cell's
        // divergence rather than inventing an exterior value. `Some`: the
        // edge cell is updated by its inner face and the given outer face.
        // Edge cells are reconstructed at first order, so their edge
        // values are the cell values.
        let mirror = |u: [f64; 3]| [u[0], -u[1], u[2]];
        let left_outer = match edges.left {
            EdgeTreatment::NeighbourDivergence => None,
            EdgeTreatment::Wall => Some(self.rusanov_face(mirror(left_edge[0]), left_edge[0])?),
            EdgeTreatment::Exchanged => Some([0.0; 3]),
        };
        match left_outer {
            None => {
                for k in 0..3 {
                    div[(0, k)] = div[(1, k)];
                }
            }
            Some(outer) => {
                let inner = self.rusanov_face(right_edge[0], left_edge[1])?;
                for k in 0..3 {
                    div[(0, k)] = (inner[k] - outer[k]) / dx;
                }
            }
        }
        let right_outer = match edges.right {
            EdgeTreatment::NeighbourDivergence => None,
            EdgeTreatment::Wall => {
                Some(self.rusanov_face(right_edge[n - 1], mirror(right_edge[n - 1]))?)
            }
            EdgeTreatment::Exchanged => Some([0.0; 3]),
        };
        match right_outer {
            None => {
                for k in 0..3 {
                    div[(n - 1, k)] = div[(n - 2, k)];
                }
            }
            Some(outer) => {
                let inner = self.rusanov_face(right_edge[n - 2], left_edge[n - 1])?;
                for k in 0..3 {
                    div[(n - 1, k)] = (outer[k] - inner[k]) / dx;
                }
            }
        }

        Ok(ContextValue::VectorField(div))
    }
}

impl FluxDivergenceOperator for ShallowLayerFlux {
    type MeshType = UniformGrid1D;

    fn apply(
        &self,
        field: &ContextValue,
        mesh: &Self::MeshType,
        ctx: &ComputeContext,
    ) -> Result<ContextValue, OxiflowError> {
        let edges = Edges {
            left: EdgeTreatment::NeighbourDivergence,
            right: EdgeTreatment::NeighbourDivergence,
        };
        self.divergence(field, mesh, ctx, edges)
    }

    fn stencil_radius(&self) -> usize {
        1
    }
}

// ── Edge treatments ───────────────────────────────────────────────────────────

/// How [`ShallowLayerFlux::divergence`] updates a domain-edge cell.
// Only the coupled-grid example selects treatments other than
// `NeighbourDivergence`; the other consumers of this file use `apply`.
#[allow(dead_code)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum EdgeTreatment {
    /// The edge cell reuses its interior neighbour's divergence (truncation
    /// posture of [`FluxDivergenceOperator::apply`]).
    NeighbourDivergence,
    /// Reflective wall: the outer face is the Rusanov flux between the edge
    /// cell and its mirror image (momentum reversed). No mass or solids
    /// cross it; a layer at rest stays at rest.
    Wall,
    /// The outer face carries no flux of its own: a coupling operator
    /// supplies the whole exchange across it.
    Exchanged,
}

/// One [`EdgeTreatment`] per domain edge.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Edges {
    pub(crate) left: EdgeTreatment,
    pub(crate) right: EdgeTreatment,
}

/// [`ShallowLayerFlux`] with chosen edge treatments, as a
/// `FluxDivergenceOperator`.
// Used by the coupled-grid example only (see `EdgeTreatment`).
#[allow(dead_code)]
pub(crate) struct ShallowLayerFluxWithEdges {
    pub(crate) flux: ShallowLayerFlux,
    pub(crate) edges: Edges,
}

impl RequiresContext for ShallowLayerFluxWithEdges {
    fn required_variables(&self) -> Vec<ContextVariable> {
        self.flux.required_variables()
    }
}

impl FluxDivergenceOperator for ShallowLayerFluxWithEdges {
    type MeshType = UniformGrid1D;

    fn apply(
        &self,
        field: &ContextValue,
        mesh: &Self::MeshType,
        ctx: &ComputeContext,
    ) -> Result<ContextValue, OxiflowError> {
        self.flux.divergence(field, mesh, ctx, self.edges)
    }

    fn stencil_radius(&self) -> usize {
        self.flux.stencil_radius()
    }
}
