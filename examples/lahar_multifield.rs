//! # Example: multi-field lahar benchmark model (#139, feeds #53)
//!
//! Implements DD-032 (#85): a physically representative lahar model per
//! Woodhouse, Hogg & Phillips (LaharFlowPaper.pdf, cited in DD-032),
//! reduced to one spatial dimension. Four coupled conservation quantities,
//! not three -- see the state layout note below.
//!
//! ## Governing equations (paper eq. 2, 4-6, 11-21, reduced to 1D)
//!
//! Conserved state per node: `(m, p, s, b)` where `m = rho_bar*h` (mass
//! depth), `p = rho_bar*h*u_bar` (momentum depth), `s = h*psi_bar` (solid
//! volume depth), `b` = bed elevation. This departs from #139's literal
//! `(h, h*u_bar, c_bar)` listing in two ways, both discussed and confirmed
//! in session:
//!
//! - `b(x,t)` is tracked as a fourth evolving quantity, not held fixed --
//!   the published model is genuinely four coupled equations (mass,
//!   momentum, solids, bed evolution), not three; #139's three-field
//!   description omits the bed because DD-032's own summary did too.
//! - The conserved momentum-equation variables carry the bulk density
//!   `rho_bar = rho_f*(1-psi_bar) + rho_s*psi_bar`, not raw `h`/`h*u_bar`.
//!   With `rho_s/rho_f = 2`, dropping this weight would misrepresent the
//!   bulking behaviour DD-032 explicitly cites as the model's signature
//!   effect. `h`, `u_bar`, `psi_bar`, `c_bar` are recovered by diagnostic
//!   (see [`LaharPhysics::primitives`]) wherever needed.
//!
//! ```text
//! d(m)/dt + d(m*u)/dx           = -rho_b*db/dt + Q(x,t)
//! d(p)/dt + d(p*u + 0.5*rho_bar*g*h^2)/dx
//!                                = -rho_bar*g*h*db/dx - tau_xz - u_b*rho_b*db/dt
//! d(s)/dt + d(s*u)/dx           = -(D - E) + psi_src*Q(x,t)
//! db/dt                         = (D - E) / psi_b
//! ```
//!
//! `Q(x,t)` is a constant-rate top-hat volumetric source density over a
//! fixed interval `[x_lo, x_hi]`, matching the paper's own single-phase
//! validation setup (Sec. 4.1: "release ... at a constant rate", total
//! flux `q = 1 m^2/s` over `|x| < 0.5`) -- `Q = q / (x_hi - x_lo)` so the
//! integral of `Q` over the source strip equals `q`. Injected as a mass
//! rate `rho_f*Q` here since the validation scenario in
//! `tests/lahar_multifield_analytical.rs` is pure water (`psi_src = 0`
//! everywhere, including at the source).
//!
//! ## Basal traction (paper eq. 8-12)
//!
//! Concentration-weighted blend of Chezy drag (dilute) and Pouliquen-
//! Forterre granular friction (dense):
//!
//! ```text
//! tau_xz = [Cd*|u|^2*(1-f(psi)) + mu(h,u)*g*h*f(psi)] * rho_bar*u/|u|
//! f(psi) = 0.5*(1 + tanh(alpha*(psi - psi0)))
//! ```
//!
//! `mu(h,u)` (paper eq. 10), `Fr = |u|/sqrt(g*h)`:
//!
//! ```text
//! Fr > beta:      mu = mu_stop(h*beta/Fr)
//! 0 < Fr < beta:  mu = (Fr/beta)^gamma * (mu_stop(h) - mu_start(h)) + mu_start(h)
//! Fr ~ 0:         mu = min(mu_start(h), |db/dx - dh/dx|)
//! ```
//!
//! with `mu_stop(h) = tan(d1) + (tan(d2)-tan(d1))/(h/L + 1)`, `mu_start`
//! identical with `d3` in place of `d1` (paper adopts `d1 ~= d3 - 1 deg`
//! to cut one free parameter). `L = 3*d/2` (paper's own choice, not a free
//! parameter). The `Fr ~ 0` branch needs `dh/dx`, computed locally by
//! finite difference on the diagnostic `h` array (`h` is derived from the
//! state, not a state column, so no `FDGradientCalculator` can target it
//! directly -- `LaharSource` stores `dx` at construction and does the
//! difference itself).
//!
//! ## Erosion / deposition (paper eq. 15-21)
//!
//! Shields threshold `theta_c` (Soulsby 1997), entrainment `E` above
//! threshold, hindered-settling deposition `D` (Richardson-Zaki exponent,
//! Soulsby 1997 single-grain speed) -- see [`shields_theta_critical`],
//! [`erosion_rate`], [`deposition_rate`] for the exact formulas.
//!
//! ## Numerical scheme
//!
//! [`ShallowLayerFlux`] implements [`FluxDivergenceOperator`] directly (no
//! new trait -- confirmed in session that the existing
//! `FluxDivergenceOperator`/`DiscretizedModel`/`CompositeModel` stack
//! already carries `ContextValue::VectorField` end to end, a side effect
//! of #138's `ContextValue` generalization): local Lax-Friedrichs
//! (Rusanov) numerical flux on `(m, p, s)`, `b`'s own flux held at zero
//! (no spatial flux term for bed elevation). Face states come from a
//! MinMod-limited MUSCL reconstruction, reusing
//! [`oxiflow::operators::limiters::Limiter::phi`] directly -- but
//! **simplified from `LimitedFlux`'s own velocity-direction-dependent
//! window**: one symmetric slope per cell, extrapolated to both edges,
//! since Rusanov's own `-0.5*s_max*(U_R-U_L)` dissipation term supplies
//! the upwind bias here, not the reconstruction. Boundary cells use
//! first-order (unreconstructed) values and the divergence there is
//! copied from the nearest interior cell -- the same `Truncation`-style
//! posture `operators::fv`/`operators::limiters` document, reproduced
//! locally since their `pub(crate)` helpers aren't visible from an
//! example.
//!
//! `check_cfl` (operators::mod) is also `pub(crate)` -- reproduced here as
//! a direct wave-speed check (`|u| + sqrt(g*h)`, shallow-water
//! eigenvalues) rather than the constant-velocity form FV/WENO use.
//!
//! [`LaharSource`] (an ordinary [`PhysicalModel`], not a flux term)
//! computes friction, erosion/deposition, bed evolution, and the top-hat
//! source for all four columns, requiring `d(b)/dx` via an ordinary
//! [`FDGradientCalculator`] on component 3 -- no new machinery, the same
//! per-component gradient mechanism #138 generalized.
//!
//! Composed as `CompositeModel::new(vec![LaharSource, DiscretizedModel<ShallowLayerFlux>])`
//! -- `LaharSource` first, so `CompositeModel::initial_state` (which uses
//! the first operand) returns the real physical initial condition (dry
//! bed, flat topography), not `DiscretizedModel`'s own placeholder zero
//! `ScalarField`.
//!
//! `ForwardEulerSolver` -- the only integrator wired to
//! `ContextValue::add_scaled` for `VectorField` state (still true after
//! #136-#138).
//!
//! ## Parameter grouping
//!
//! [`FrictionParams`], [`ErosionDepositionParams`], and [`SourceInjection`]
//! group related constructor/function arguments into structs rather than
//! passing them individually -- the session's own carried-over lesson
//! (`clippy::too_many_arguments`, threshold 7: group related parameters
//! at design time rather than reaching for `#[allow]`).
//!
//! ## What's NOT attempted here
//!
//! - Reproducing the paper's own Sec. 4.2 worked example (eroding flow on
//!   an inclined plane) exactly -- it develops secondary fronts and
//!   roll-wave instabilities with no closed-form check, only qualitative
//!   figures. The analytical validation instead uses the *simpler*
//!   Sec. 4.1 single-phase (non-eroding) case, which has closed-form
//!   front-position scaling laws for both drag regimes.
//! - Two-dimensional topography, curvilinear/terrain-following
//!   coordinates, or any `y`-direction term -- out of scope for a 1D
//!   `UniformGrid1D` benchmark.

use std::sync::Arc;

use nalgebra::DMatrix;
use oxiflow::{
    context::{
        calculators::{FDGradientCalculator, FDScheme},
        compute::ComputeContext,
        error::OxiflowError,
        value::ContextValue,
        variable::ContextVariable,
    },
    mesh::{Mesh, UniformGrid1D},
    model::{CompositeModel, DiscretizedModel, PhysicalModel, RequiresContext},
    operators::{limiters::Limiter, FluxDivergenceOperator},
    solver::{
        config::{IntegratorKind, StepControl, TimeConfiguration},
        methods::ForwardEulerSolver,
        scenario::Scenario,
        Solver, SolverConfiguration,
    },
};

// ── State layout ─────────────────────────────────────────────────────────────

/// Column index of `m = rho_bar*h` in the 4-column `VectorField` state.
pub const COL_M: usize = 0;
/// Column index of `p = rho_bar*h*u_bar`.
pub const COL_P: usize = 1;
/// Column index of `s = h*psi_bar` (solid volume depth).
pub const COL_S: usize = 2;
/// Column index of `b` (bed elevation).
pub const COL_B: usize = 3;

pub const G: f64 = 9.81;
/// Depth below which a node is treated as dry -- guards the `s/h`,
/// `p/m` diagnostics against division by (near-)zero at a wetting front,
/// a standard regularisation for shallow-water schemes.
const H_MIN: f64 = 1.0e-6;

// ── Shared density/diagnostic physics ────────────────────────────────────────

/// Density-mixing physics shared by [`ShallowLayerFlux`] and
/// [`LaharSource`] -- kept as its own small `Copy` type rather than
/// duplicated fields, since both need exactly the same `rho_bar`/
/// `primitives` logic.
#[derive(Debug, Clone, Copy)]
pub struct LaharPhysics {
    pub rho_f: f64,
    pub rho_s: f64,
}

impl LaharPhysics {
    pub fn rho_bar(&self, psi: f64) -> f64 {
        self.rho_f * (1.0 - psi) + self.rho_s * psi
    }

    /// Recovers `(h, u, psi)` from conservative `(m, p, s)` -- paper eq. 1,
    /// solved for `h` given `m = rho_f*h + (rho_s-rho_f)*h*psi = rho_f*h +
    /// (rho_s-rho_f)*s`.
    ///
    /// # Errors
    ///
    /// `OxiflowError::PreconditionFailed` if the recovered depth is
    /// meaningfully negative (beyond `H_MIN` numerical noise) -- a real
    /// solver-integrity problem, not a normal dry cell.
    pub fn primitives(&self, m: f64, p: f64, s: f64) -> Result<(f64, f64, f64), OxiflowError> {
        let delta_rho = self.rho_s - self.rho_f;
        let h = (m - delta_rho * s) / self.rho_f;
        if h < -H_MIN {
            return Err(OxiflowError::PreconditionFailed {
                context: "LaharPhysics::primitives",
                message: format!("non-physical negative depth h={h} recovered from m={m}, s={s}"),
            });
        }
        let h = h.max(0.0);
        if h <= H_MIN {
            // Dry cell: force zero velocity rather than dividing by a
            // near-zero m -- a tiny residual in p (a normal Rusanov/MUSCL
            // undershoot at a wet/dry front) would otherwise be amplified
            // into a spurious, arbitrarily large velocity (observed in
            // session as a CFL blow-up with max_wave_speed in the
            // thousands of m/s). Standard shallow-water dry-front
            // regularisation.
            return Ok((h, 0.0, 0.0));
        }
        let psi = (s / h).clamp(0.0, 0.999);
        let u = p / m.max(1.0e-9);
        Ok((h, u, psi))
    }

    pub fn wave_speed(&self, h: f64, u: f64) -> f64 {
        u.abs() + (G * h).sqrt()
    }
}

// ── Grouped parameters (clippy::too_many_arguments, see module doc) ──────────

/// Basal-friction parameters (paper eq. 8-12) -- Chezy coefficient plus
/// the full Pouliquen-Forterre granular closure and the concentration
/// switching function.
#[derive(Debug, Clone, Copy)]
pub struct FrictionParams {
    pub cd: f64,
    pub tan_d1: f64,
    pub tan_d2: f64,
    pub tan_d3: f64,
    pub length_l: f64,
    pub beta: f64,
    pub gamma: f64,
    pub alpha: f64,
    pub psi0: f64,
}

/// Erosion/deposition parameters (paper eq. 15-21).
#[derive(Debug, Clone, Copy)]
pub struct ErosionDepositionParams {
    pub epsilon: f64,
    pub nu: f64,
    pub grain_d: f64,
    pub psi_b: f64,
}

/// Top-hat volumetric injection `Q(x,t)` (paper Sec. 4.1/2.3).
#[derive(Debug, Clone, Copy)]
pub struct SourceInjection {
    /// Total volumetric rate (m^2/s in 1D, matching the paper's `q`) --
    /// spread as a density `rate/(x_hi-x_lo)` over the strip.
    pub rate: f64,
    pub x_lo: f64,
    pub x_hi: f64,
    /// Solids volume fraction of the injected material -- 0.0 for the
    /// pure-water analytical validation case.
    pub psi: f64,
}

impl SourceInjection {
    fn density(&self, x: f64) -> f64 {
        if x >= self.x_lo && x <= self.x_hi {
            self.rate / (self.x_hi - self.x_lo)
        } else {
            0.0
        }
    }
}

// ── Closure functions (paper eq. 8-21) ───────────────────────────────────────

/// Pouliquen-Forterre `mu_stop`/`mu_start` (paper eq. "232"/"233"),
/// `tan_d` is `tan_d1` for `mu_stop`, `tan_d3` for `mu_start`.
pub fn mu_branch(h: f64, tan_d: f64, tan_d2: f64, length_l: f64) -> f64 {
    tan_d + (tan_d2 - tan_d) / (h / length_l + 1.0)
}

/// Full Pouliquen-Forterre granular friction coefficient (paper eq. 10).
pub fn pouliquen_mu(h: f64, u: f64, p: &FrictionParams, slope_diff: f64) -> f64 {
    if h <= H_MIN {
        return 0.0;
    }
    let fr = u.abs() / (G * h).sqrt();
    if fr < 1.0e-9 {
        // Static branch -- paper eq. 10, third case.
        let mu_start = mu_branch(h, p.tan_d3, p.tan_d2, p.length_l);
        mu_start.min(slope_diff.abs())
    } else if fr > p.beta {
        mu_branch(h * p.beta / fr, p.tan_d1, p.tan_d2, p.length_l)
    } else {
        let mu_stop_h = mu_branch(h, p.tan_d1, p.tan_d2, p.length_l);
        let mu_start_h = mu_branch(h, p.tan_d3, p.tan_d2, p.length_l);
        (fr / p.beta).powf(p.gamma) * (mu_stop_h - mu_start_h) + mu_start_h
    }
}

/// Concentration switching function `f(psi)` (paper eq. 12).
pub fn switching_function(psi: f64, psi0: f64, alpha: f64) -> f64 {
    0.5 * (1.0 + (alpha * (psi - psi0)).tanh())
}

/// Bulk basal traction `tau_xz` (paper eq. 11), sign carried by `u`.
pub fn basal_traction(
    h: f64,
    u: f64,
    psi: f64,
    rho_bar: f64,
    p: &FrictionParams,
    slope_diff: f64,
) -> f64 {
    if u.abs() < 1.0e-12 {
        return 0.0;
    }
    let f_psi = switching_function(psi, p.psi0, p.alpha);
    let mu = pouliquen_mu(h, u, p, slope_diff);
    let fluid_term = p.cd * u * u * (1.0 - f_psi);
    let granular_term = mu * G * h * f_psi;
    (fluid_term + granular_term) * rho_bar * u.signum()
}

/// Critical Shields stress `theta_c` (paper eq. 16, Soulsby 1997);
/// `r_2_3` is `R^(2/3)`, `R` the paper's dimensionless grain-size group.
pub fn shields_theta_critical(r_2_3: f64) -> f64 {
    0.30 / (1.0 + 1.2 * r_2_3) + 0.055 * (1.0 - (-0.02 * r_2_3).exp())
}

/// Erosion rate `E` (paper eq. 17) -- zero below the critical Shields
/// stress, `epsilon*sqrt(g'*d)*(theta-theta_c)^1.5` above it.
pub fn erosion_rate(theta: f64, theta_c: f64, epsilon: f64, g_prime: f64, d: f64) -> f64 {
    if theta <= theta_c {
        0.0
    } else {
        epsilon * (g_prime * d).sqrt() * (theta - theta_c).powf(1.5)
    }
}

/// Single-grain settling speed (paper eq. 21, Soulsby 1997).
pub fn settling_speed_single_grain(nu: f64, d: f64, r_star: f64) -> f64 {
    (nu / d) * ((10.36_f64.powi(2) + 1.049 * r_star.powi(2)).sqrt() - 10.36)
}

/// Richardson-Zaki exponent `n` (paper eq. 20), `re_s` is the settling
/// particle Reynolds number `sqrt(g*d^3)/nu`.
pub fn richardson_zaki_n(re_s: f64) -> f64 {
    (4.7 + 0.41 * re_s.powf(0.75)) / (1.0 + 0.175 * re_s.powf(0.75))
}

/// Hindered settling speed `w_s` (paper eq. 19) and deposition flux
/// `D = psi*w_s` (paper eq. 18), combined since `D` is always what's
/// actually needed.
pub fn deposition_rate(psi: f64, psi_star: f64, nu: f64, d: f64, r_star: f64) -> f64 {
    let re_s = (G * d.powi(3)).sqrt() / nu;
    let n = richardson_zaki_n(re_s);
    let ws0 = settling_speed_single_grain(nu, d, r_star);
    let ws = (1.0 - psi).powf(2.7 - 0.15 * n) * (1.0 - psi / psi_star).powf(0.62 * n - 1.46) * ws0;
    psi * ws
}

// ── Flux operator ─────────────────────────────────────────────────────────────

/// `FluxDivergenceOperator` for the coupled `(m, p, s)` shallow-layer
/// system -- Rusanov numerical flux, MinMod-limited MUSCL reconstruction.
/// `b`'s own flux-divergence contribution is always zero (no spatial flux
/// term for bed elevation in the governing equations).
pub struct ShallowLayerFlux {
    pub physics: LaharPhysics,
    pub limiter: Limiter,
    /// Rayon dispatch threshold for the divergence loop (two
    /// [`ShallowLayerFlux::rusanov_face`] evaluations per node --
    /// primitives + physical flux each side) -- unmeasured back-of-
    /// envelope default, see [`default_flux_parallel_threshold`]. Per-
    /// instance, not global (DD-048 P1/P2/P3): this site's own coupling
    /// granularity (one node's flux touches only its immediate face
    /// neighbours) has nothing to do with `LaharSource`'s per-node cost
    /// profile, so each gets its own threshold rather than sharing one.
    #[cfg(feature = "parallel")]
    pub parallel_threshold: usize,
}

/// Measured: `flux_dispatch_diagnostic` below puts the real crossover
/// between 1,000 and 10,000 -- but 1,000 itself flips sign between runs
/// (session: 1.18x slower in one run, 1.09x faster in another, on the
/// same machine), so the default sits at 10,000, past that noisy
/// boundary, where the win is consistent and stable (~3-4x from 10,000
/// through 1,000,000) -- see BENCHMARKS.md.
#[cfg(feature = "parallel")]
pub fn default_flux_parallel_threshold() -> usize {
    10_000
}

impl ShallowLayerFlux {
    fn physical_flux(&self, m: f64, p: f64, s: f64, h: f64, u: f64, psi: f64) -> [f64; 3] {
        let rho_bar = self.physics.rho_bar(psi);
        [m * u, p * u + 0.5 * rho_bar * G * h * h, s * u]
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
    /// see the module doc for why this is simpler than `LimitedFlux`'s
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

impl FluxDivergenceOperator for ShallowLayerFlux {
    type MeshType = UniformGrid1D;

    fn apply(
        &self,
        field: &ContextValue,
        mesh: &Self::MeshType,
        ctx: &ComputeContext,
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
        // Truncation posture (operators::fv/limiters' own documented
        // choice, reproduced here): boundary cells reuse the nearest
        // interior cell's divergence rather than inventing an exterior
        // value.
        for k in 0..3 {
            div[(0, k)] = div[(1, k)];
            div[(n - 1, k)] = div[(n - 2, k)];
        }

        Ok(ContextValue::VectorField(div))
    }

    fn stencil_radius(&self) -> usize {
        1
    }
}

// ── Source term (friction, erosion/deposition, bed evolution, injection) ─────

/// Ordinary `PhysicalModel` -- friction, erosion/deposition, bed
/// evolution, and the top-hat source `Q(x,t)`. No spatial flux; composed
/// alongside `DiscretizedModel<ShallowLayerFlux>` via `CompositeModel`.
///
/// Stores `dx` and each node's `x` coordinate at construction time:
/// `compute_physics(&self, state, ctx)` receives neither the mesh nor a
/// coordinate accessor (per `PhysicalModel`'s own signature), so both are
/// captured once in [`LaharSource::new`] rather than recomputed per call.
pub struct LaharSource {
    pub physics: LaharPhysics,
    pub friction: FrictionParams,
    pub erosion: ErosionDepositionParams,
    pub injection: SourceInjection,
    /// Initial (flat, per Sec. 4.1's "non-erodible substrate") bed
    /// elevation.
    pub initial_b: f64,
    dx: f64,
    x_coords: Vec<f64>,
    /// Rayon dispatch threshold for the per-node friction/erosion/
    /// deposition loop -- unmeasured back-of-envelope default, see
    /// [`default_source_parallel_threshold`]. Per-instance (DD-048
    /// P1/P2/P3), independent of `ShallowLayerFlux`'s own threshold:
    /// this site's per-node cost (three-branch Pouliquen-Forterre
    /// friction plus the Shields/settling-velocity closures) has nothing
    /// to do with the flux operator's cost profile.
    #[cfg(feature = "parallel")]
    parallel_threshold: usize,
}

/// Measured: `source_dispatch_diagnostic` below puts the real crossover
/// between 100 and 1,000, consistent across runs (2.71x, then 2.90x
/// faster at 1,000 -- no sign flip): the same profile as
/// `langmuir_multi`'s own measured threshold (#138), sitting right at
/// the crossover rather than carrying a larger, unjustified safety
/// margin -- see BENCHMARKS.md.
#[cfg(feature = "parallel")]
pub fn default_source_parallel_threshold() -> usize {
    1_000
}

impl LaharSource {
    pub fn new(
        mesh: &UniformGrid1D,
        physics: LaharPhysics,
        friction: FrictionParams,
        erosion: ErosionDepositionParams,
        injection: SourceInjection,
        initial_b: f64,
        #[cfg(feature = "parallel")] parallel_threshold: usize,
    ) -> Self {
        let x_coords = (0..mesh.n_dof()).map(|i| mesh.coordinates(i)[0]).collect();
        Self {
            physics,
            friction,
            erosion,
            injection,
            initial_b,
            dx: mesh.characteristic_length(),
            x_coords,
            #[cfg(feature = "parallel")]
            parallel_threshold,
        }
    }

    fn rho_b(&self) -> f64 {
        self.physics.rho_f * (1.0 - self.erosion.psi_b) + self.physics.rho_s * self.erosion.psi_b
    }

    fn g_prime(&self) -> f64 {
        G * (self.physics.rho_s - self.physics.rho_f) / self.physics.rho_f
    }

    /// Paper's dimensionless grain-size group `R` (distinct from `R^(2/3)`
    /// used directly in the Shields closure).
    fn r_dimensionless(&self) -> f64 {
        let d = self.erosion.grain_d;
        let re_p = d * (self.g_prime() * d).sqrt() / self.erosion.nu;
        re_p * ((self.physics.rho_s - self.physics.rho_f) / self.physics.rho_f).sqrt()
    }
}

impl RequiresContext for LaharSource {
    fn required_variables(&self) -> Vec<ContextVariable> {
        vec![ContextVariable::SpatialGradient {
            dimension: 0,
            component: Some(COL_B),
        }]
    }
}

impl PhysicalModel for LaharSource {
    fn compute_physics(
        &self,
        state: &ContextValue,
        ctx: &ComputeContext,
    ) -> Result<ContextValue, OxiflowError> {
        let s = state.as_vector_field()?;
        let n = s.nrows();
        let dt = ctx.time_step();

        let db_dx = ctx
            .external(ContextVariable::SpatialGradient {
                dimension: 0,
                component: Some(COL_B),
            })?
            .as_scalar_field()?
            .clone();

        // Diagnostic h/u/psi per node -- h feeds a local dh/dx (Fr~0
        // branch only), computed here since h is derived from (m, s),
        // not a state column an FDGradientCalculator could target.
        let mut h = vec![0.0_f64; n];
        let mut u = vec![0.0_f64; n];
        let mut psi = vec![0.0_f64; n];
        for i in 0..n {
            let (hi, ui, psii) =
                self.physics
                    .primitives(s[(i, COL_M)], s[(i, COL_P)], s[(i, COL_S)])?;
            h[i] = hi;
            u[i] = ui;
            psi[i] = psii;
        }

        let r_star = self.r_dimensionless();
        let r_2_3 = r_star.powf(2.0 / 3.0);
        let theta_c = shields_theta_critical(r_2_3);
        let rho_b = self.rho_b();
        let g_prime = self.g_prime();
        let psi_star = 0.6; // random-close-packing fraction (paper's own psi*)

        let mut out = DMatrix::zeros(n, 4);
        let compute_node = |i: usize| -> [f64; 4] {
            let dh_dx = if i == 0 || i == n - 1 {
                0.0 // no interior neighbour on one side; Fr~0 there falls
                    // back to mu_start alone (slope_diff = |db/dx|).
            } else {
                (h[i + 1] - h[i - 1]) / (2.0 * self.dx)
            };
            let slope_diff = (db_dx[i] - dh_dx).abs();

            let rho_bar = self.physics.rho_bar(psi[i]);
            let tau_xz = basal_traction(h[i], u[i], psi[i], rho_bar, &self.friction, slope_diff);

            let theta = if h[i] > H_MIN {
                tau_xz.abs()
                    / ((self.physics.rho_s - self.physics.rho_f) * G * self.erosion.grain_d)
            } else {
                0.0
            };
            let erosion = erosion_rate(
                theta,
                theta_c,
                self.erosion.epsilon,
                g_prime,
                self.erosion.grain_d,
            );
            let deposition = if h[i] > H_MIN {
                deposition_rate(
                    psi[i],
                    psi_star,
                    self.erosion.nu,
                    self.erosion.grain_d,
                    r_star,
                )
            } else {
                0.0
            };
            let db_dt = (deposition - erosion) / self.erosion.psi_b;
            let u_b = tau_xz.signum() * (tau_xz.abs() / rho_bar.max(1.0e-9)).sqrt();

            // Explicit-Euler friction overshoot guard: at a thin/shallow
            // front, tau_xz's magnitude can demand a momentum change
            // bigger than the momentum actually present, which would
            // reverse the flow's direction and accelerate it the other
            // way within a single step -- diagnosed in session as a
            // genuine instability (max_wave_speed in the thousands of
            // m/s a few nodes past the source edge, ~40 steps in), not a
            // CFL/advective problem (the CFL check above already covers
            // that separately). Friction can only ever brake flow to a
            // stop within one step, never past it -- clip the momentum
            // contribution accordingly rather than trusting the raw
            // explicit rate. Loses accuracy only in the (rare, already
            // near-stationary) cells where this actually triggers.
            let p_i = s[(i, COL_P)];
            let raw_friction_dp = -tau_xz;
            let friction_dp =
                if p_i * raw_friction_dp < 0.0 && (raw_friction_dp * dt).abs() > p_i.abs() {
                    -p_i / dt
                } else {
                    raw_friction_dp
                };

            let injection = self.injection.density(self.x_coords[i]);

            [
                -rho_b * db_dt + self.physics.rho_f * injection,
                -rho_bar * G * h[i] * db_dx[i] + friction_dp - u_b * rho_b * db_dt,
                -(deposition - erosion) + self.injection.psi * injection,
                db_dt,
            ]
        };

        // Real per-node cost here: the three-branch Pouliquen-Forterre
        // friction closure plus the Shields/settling-velocity closures --
        // independent across nodes (reads only the already-computed
        // h/u/psi/db_dx arrays and self, all shared immutable). Parallel
        // results collected into a `Vec` first, then copied sequentially
        // into `out` -- the same DMatrix column-major consequence
        // `langmuir_multi` (#138) documents for its own row-parallel
        // assembly.
        #[cfg(feature = "parallel")]
        let rows: Vec<[f64; 4]> = if n >= self.parallel_threshold {
            use rayon::prelude::*;
            (0..n).into_par_iter().map(compute_node).collect()
        } else {
            (0..n).map(compute_node).collect()
        };
        #[cfg(not(feature = "parallel"))]
        let rows: Vec<[f64; 4]> = (0..n).map(compute_node).collect();

        for (i, row) in rows.into_iter().enumerate() {
            for (k, val) in row.into_iter().enumerate() {
                out[(i, k)] = val;
            }
        }

        Ok(ContextValue::VectorField(out))
    }

    fn initial_state(&self, mesh: &dyn Mesh) -> ContextValue {
        // Dry bed everywhere (m=p=s=0): the flow builds up entirely from
        // the source term, matching Sec. 4.1's "release ... at a constant
        // rate onto a flat substrate" (no pre-existing layer).
        let n = mesh.n_dof();
        let mut state = DMatrix::zeros(n, 4);
        for i in 0..n {
            state[(i, COL_B)] = self.initial_b;
        }
        ContextValue::VectorField(state)
    }

    fn name(&self) -> &str {
        "lahar_multifield_source"
    }

    fn description(&self) -> Option<&str> {
        Some(
            "friction, erosion/deposition, bed evolution and source injection for the lahar \
             model (#139)",
        )
    }
}

// ── Validation scenario (paper Sec. 4.1, single-phase, non-erodible) ─────────

/// Depth threshold used to locate the flow front `x_f(t)` in a diagnostic
/// `m` array -- consistent with `H_MIN`'s own "dry" definition.
const FRONT_THRESHOLD: f64 = 1.0e-4;

/// Which drag regime a run isolates -- see [`build_source`] and
/// [`run_and_report`]'s analytical-formula selection.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ValidationRegime {
    /// Chezy-drag-dominated asymptote (paper Fig. 4, `x_f ~ (g*q^2/Cd)^(1/5) * t^(4/5)`).
    ChezyDragDominated,
    /// Constant-Coulomb-friction asymptote (`x_f ~ q*t/sqrt(mu)`) -- known
    /// (session-confirmed) to hold only on average across a slump/stall
    /// cycle, not pointwise.
    CoulombConstant,
    /// Early inertia-pressure-gradient balance (paper Sec. 4, "initially
    /// the front propagation for each flow is similar ... drag has
    /// little effect", `x_f ~ (g*q)^(1/3) * t`) -- friction-independent,
    /// so reuses the Chezy-only friction config without engaging its
    /// drag-dominated asymptote (only valid at short `t`, before drag
    /// takes over).
    EarlyInertial,
}

/// Builds the flat-bed, non-erodible, pure-water configuration paper
/// Sec. 4.1 validates against, isolating one drag regime at a time by
/// choosing degenerate friction parameters -- not a special code path,
/// the same `LaharSource`/`ShallowLayerFlux` used for the full model.
///
/// `ChezyDragDominated`/`EarlyInertial`: kills the granular branch
/// entirely (`tan_d1=tan_d2=tan_d3=0`, so `mu` is identically zero
/// regardless of `psi`), isolating Chezy drag -- the paper's own first
/// comparison curve (Fig. 4, `Cd=0.04`); reused as-is for `EarlyInertial`
/// since the paper states drag has little effect in that regime anyway.
/// `CoulombConstant`: kills the Chezy term (`cd=0`) and forces
/// `f(psi)=1` for all `psi>=0` via `psi0=-10.0` (so the switching
/// function never falls back toward the fluid term even at `psi=0`),
/// with a constant granular coefficient `mu=0.1`
/// (`tan_d1=tan_d2=tan_d3=0.1`, collapsing `mu_branch` to that constant
/// regardless of `h`) -- the paper's second comparison curve (Fig. 4,
/// constant Coulomb `mu=0.1`).
fn build_source(mesh: &UniformGrid1D, regime: ValidationRegime, source_rate: f64) -> LaharSource {
    let physics = LaharPhysics {
        rho_f: 1000.0,
        rho_s: 2000.0,
    };
    let friction = if regime == ValidationRegime::CoulombConstant {
        FrictionParams {
            cd: 0.0,
            tan_d1: 0.1,
            tan_d2: 0.1,
            tan_d3: 0.1,
            length_l: 1.0, // unused: mu_branch collapses to 0.1 regardless of h/L.
            beta: 0.136,
            gamma: 1.0e-3,
            alpha: 5.0,
            psi0: -10.0,
        }
    } else {
        FrictionParams {
            cd: 0.04,
            tan_d1: 0.0,
            tan_d2: 0.0,
            tan_d3: 0.0,
            length_l: 1.0, // unused: mu_branch(h,0,0,L) = 0 for any L.
            beta: 0.136,
            gamma: 1.0e-3,
            alpha: 5.0,
            psi0: 0.4,
        }
    };
    let erosion = ErosionDepositionParams {
        epsilon: 0.0, // non-erodible substrate (paper Sec. 4.1)
        nu: 1.2e-6,
        grain_d: 1.0e-3,
        psi_b: 0.6,
    };
    let injection = SourceInjection {
        rate: source_rate,
        x_lo: -0.5,
        x_hi: 0.5,
        psi: 0.0, // pure water (paper Sec. 4.1: "either water or grains")
    };
    LaharSource::new(
        mesh,
        physics,
        friction,
        erosion,
        injection,
        0.0,
        #[cfg(feature = "parallel")]
        default_source_parallel_threshold(),
    )
}

/// Rightmost `x` where the diagnostic depth `h = m/rho_f` (pure water:
/// `psi=0` everywhere, so `h = m/rho_f` exactly, no need for the full
/// `LaharPhysics::primitives` solve) exceeds [`FRONT_THRESHOLD`].
fn front_position(m_column: &nalgebra::DVector<f64>, x_coords: &[f64], rho_f: f64) -> f64 {
    m_column
        .iter()
        .zip(x_coords)
        .filter(|(&m, _)| m / rho_f > FRONT_THRESHOLD)
        .map(|(_, &x)| x.abs())
        .fold(0.0, f64::max)
}

/// Runs one drag-regime configuration and prints its simulated front
/// position against the paper's own closed-form scaling law (Sec. 4.1,
/// Fig. 4's drag-dominated asymptote) at the final time.
fn run_and_report(
    regime: ValidationRegime,
    n_points: usize,
    x_half_width: f64,
    t_end: f64,
) -> Result<(), OxiflowError> {
    const DT: f64 = 5.0e-4; // conservative for CFL near h~0, unchanged from the working run
    const Q: f64 = 1.0; // paper's own q = 1 m^2/s (Sec. 4.1)
    const CD: f64 = 0.04;
    const MU: f64 = 0.1;

    let source_mesh = UniformGrid1D::new(n_points, -x_half_width, x_half_width)
        .map_err(OxiflowError::InvalidDomain)?;
    let source = build_source(&source_mesh, regime, Q);

    let flux_mesh: Arc<UniformGrid1D> = Arc::new(
        UniformGrid1D::new(n_points, -x_half_width, x_half_width)
            .map_err(OxiflowError::InvalidDomain)?,
    );
    let flux = ShallowLayerFlux {
        physics: LaharPhysics {
            rho_f: 1000.0,
            rho_s: 2000.0,
        },
        limiter: Limiter::MinMod,
        #[cfg(feature = "parallel")]
        parallel_threshold: default_flux_parallel_threshold(),
    };
    let discretized = DiscretizedModel::new(Arc::new(flux), flux_mesh, "lahar_flux");

    let composite = CompositeModel::new(
        vec![Box::new(source), Box::new(discretized)],
        "lahar_multifield",
    )?;

    let grad_mesh: Arc<dyn Mesh> = Arc::new(
        UniformGrid1D::new(n_points, -x_half_width, x_half_width)
            .map_err(OxiflowError::InvalidDomain)?,
    );
    let config = SolverConfiguration::new(
        TimeConfiguration::new(t_end, StepControl::Fixed { dt: DT }),
        IntegratorKind::Euler,
    )
    .with_calculator(Box::new(FDGradientCalculator::new(
        grad_mesh,
        0,
        Some(COL_B),
        FDScheme::Central,
    )));

    let scenario_mesh = UniformGrid1D::new(n_points, -x_half_width, x_half_width)
        .map_err(OxiflowError::InvalidDomain)?;
    let x_coords: Vec<f64> = (0..n_points)
        .map(|i| scenario_mesh.coordinates(i)[0])
        .collect();

    let solver = ForwardEulerSolver;
    let result = solver.solve(
        &Scenario::single(Box::new(composite), Box::new(scenario_mesh)),
        &config,
    )?;

    let final_state = result.states.last().expect("at least one recorded state");
    let m_column = final_state.as_vector_field()?.column(COL_M).into_owned();
    let xf_simulated = front_position(&m_column, &x_coords, 1000.0);

    // Trajectory diagnostic -- prints x_f(t) at a handful of points so a
    // stalled front (genuine for constant-Coulomb friction, per the
    // paper's own "series of slumps ... periods where it does not move")
    // can be told apart from a bug that stops all progress from the
    // start.
    let n_snapshots = 20usize;
    let stride = (result.states.len() / n_snapshots).max(1);
    for (idx, (t, state)) in result.times.iter().zip(&result.states).enumerate() {
        if idx % stride == 0 {
            let m_col = state.as_vector_field()?.column(COL_M).into_owned();
            let xf = front_position(&m_col, &x_coords, 1000.0);
            println!("  t={t:.3}: x_f={xf:.3}");
        }
    }

    let (regime_name, xf_analytical) = match regime {
        ValidationRegime::ChezyDragDominated => (
            "Chezy (drag-dominated)",
            (9.81 * Q * Q / CD).powf(0.2) * t_end.powf(0.8),
        ),
        ValidationRegime::CoulombConstant => ("Coulomb (constant mu)", Q * t_end / MU.sqrt()),
        ValidationRegime::EarlyInertial => ("Early inertial", (9.81 * Q).powf(1.0 / 3.0) * t_end),
    };
    let rel_err = (xf_simulated - xf_analytical).abs() / xf_analytical;

    println!(
        "{regime_name}: x_f simulated = {xf_simulated:.3} m, x_f analytical = {xf_analytical:.3} m, \
         relative error = {:.1}%",
        rel_err * 100.0
    );

    Ok(())
}

fn main() {
    // Chezy, drag-dominated regime: unchanged from the validated 10s run.
    if let Err(e) = run_and_report(ValidationRegime::ChezyDragDominated, 1001, 50.0, 10.0) {
        eprintln!("run failed: {e}");
    }
    // Coulomb (constant mu), drag-dominated asymptote: session-confirmed
    // to progress in slumps separated by genuine stalls (paper's own
    // documented behaviour for constant mu), not a smooth approach to
    // the analytical rate -- kept for the trajectory diagnostic, not
    // expected to match pointwise even at 60s.
    if let Err(e) = run_and_report(ValidationRegime::CoulombConstant, 2001, 100.0, 60.0) {
        eprintln!("run failed: {e}");
    }
    // Early inertial regime: friction-independent by the paper's own
    // account ("drag has little effect"), so this isolates the
    // flux/pressure-gradient machinery specifically, avoiding the
    // slump/stall complexity entirely. Short domain/time -- the paper's
    // own figure marks this regime at t=2s.
    if let Err(e) = run_and_report(ValidationRegime::EarlyInertial, 501, 10.0, 1.5) {
        eprintln!("run failed: {e}");
    }
}
