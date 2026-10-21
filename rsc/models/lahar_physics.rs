//! Lahar multi-field model, shared part: state layout, density mixing and
//! primitive recovery (`LaharPhysics`), grouped parameters, the friction and
//! erosion/deposition closures, and the flow-front diagnostic.
//!
//! Single definition shared by `examples/lahar_multifield.rs`,
//! `examples/lahar_multifield_plot.rs`, `tests/lahar_multifield_analytical.rs`
//! and `benches/lahar_multifield.rs`, each of which includes this file as a
//! module with `#[path]`. The model (DD-032, Woodhouse, Hogg & Phillips) is
//! documented in `examples/lahar_multifield.rs`.
//!
//! `rsc/models/lahar_flux.rs` and `rsc/models/lahar_source.rs` import from
//! this file through `super::lahar_physics`: a consumer that includes either
//! of them declares this file as a sibling module named `lahar_physics`.

use nalgebra::DVector;
use oxiflow::context::error::OxiflowError;

// ── State layout ─────────────────────────────────────────────────────────────

/// Column index of `m = rho_bar*h` in the 4-column `VectorField` state.
pub(crate) const COL_M: usize = 0;
/// Column index of `p = rho_bar*h*u_bar`.
pub(crate) const COL_P: usize = 1;
/// Column index of `s = h*psi_bar` (solid volume depth).
pub(crate) const COL_S: usize = 2;
/// Column index of `b` (bed elevation).
pub(crate) const COL_B: usize = 3;

pub(crate) const G: f64 = 9.81;
/// Depth below which a node is treated as dry -- guards the `s/h`,
/// `p/m` diagnostics against division by (near-)zero at a wetting front,
/// a standard regularisation for shallow-water schemes.
pub(crate) const H_MIN: f64 = 1.0e-6;

// ── Shared density/diagnostic physics ────────────────────────────────────────

/// Density-mixing physics shared by [`ShallowLayerFlux`] and
/// [`LaharSource`] -- kept as its own small `Copy` type rather than
/// duplicated fields, since both need exactly the same `rho_bar`/
/// `primitives` logic.
#[derive(Debug, Clone, Copy)]
pub(crate) struct LaharPhysics {
    pub(crate) rho_f: f64,
    pub(crate) rho_s: f64,
}

impl LaharPhysics {
    pub(crate) fn rho_bar(&self, psi: f64) -> f64 {
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
    pub(crate) fn primitives(
        &self,
        m: f64,
        p: f64,
        s: f64,
    ) -> Result<(f64, f64, f64), OxiflowError> {
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

    pub(crate) fn wave_speed(&self, h: f64, u: f64) -> f64 {
        u.abs() + (G * h).sqrt()
    }
}

// ── Grouped parameters (clippy::too_many_arguments, see the example's module doc)

/// Basal-friction parameters (paper eq. 8-12) -- Chezy coefficient plus
/// the full Pouliquen-Forterre granular closure and the concentration
/// switching function.
#[derive(Debug, Clone, Copy)]
pub(crate) struct FrictionParams {
    pub(crate) cd: f64,
    pub(crate) tan_d1: f64,
    pub(crate) tan_d2: f64,
    pub(crate) tan_d3: f64,
    pub(crate) length_l: f64,
    pub(crate) beta: f64,
    pub(crate) gamma: f64,
    pub(crate) alpha: f64,
    pub(crate) psi0: f64,
}

/// Erosion/deposition parameters (paper eq. 15-21).
#[derive(Debug, Clone, Copy)]
pub(crate) struct ErosionDepositionParams {
    pub(crate) epsilon: f64,
    pub(crate) nu: f64,
    pub(crate) grain_d: f64,
    pub(crate) psi_b: f64,
}

/// Top-hat volumetric injection `Q(x,t)` (paper Sec. 4.1/2.3).
#[derive(Debug, Clone, Copy)]
pub(crate) struct SourceInjection {
    /// Total volumetric rate (m^2/s in 1D, matching the paper's `q`) --
    /// spread as a density `rate/(x_hi-x_lo)` over the strip.
    pub(crate) rate: f64,
    pub(crate) x_lo: f64,
    pub(crate) x_hi: f64,
    /// Solids volume fraction of the injected material -- 0.0 for the
    /// pure-water analytical validation case.
    pub(crate) psi: f64,
}

impl SourceInjection {
    pub(crate) fn density(&self, x: f64) -> f64 {
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
pub(crate) fn mu_branch(h: f64, tan_d: f64, tan_d2: f64, length_l: f64) -> f64 {
    tan_d + (tan_d2 - tan_d) / (h / length_l + 1.0)
}

/// Full Pouliquen-Forterre granular friction coefficient (paper eq. 10).
pub(crate) fn pouliquen_mu(h: f64, u: f64, p: &FrictionParams, slope_diff: f64) -> f64 {
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
pub(crate) fn switching_function(psi: f64, psi0: f64, alpha: f64) -> f64 {
    0.5 * (1.0 + (alpha * (psi - psi0)).tanh())
}

/// Bulk basal traction `tau_xz` (paper eq. 11), sign carried by `u`.
pub(crate) fn basal_traction(
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
pub(crate) fn shields_theta_critical(r_2_3: f64) -> f64 {
    0.30 / (1.0 + 1.2 * r_2_3) + 0.055 * (1.0 - (-0.02 * r_2_3).exp())
}

/// Erosion rate `E` (paper eq. 17) -- zero below the critical Shields
/// stress, `epsilon*sqrt(g'*d)*(theta-theta_c)^1.5` above it.
pub(crate) fn erosion_rate(theta: f64, theta_c: f64, epsilon: f64, g_prime: f64, d: f64) -> f64 {
    if theta <= theta_c {
        0.0
    } else {
        epsilon * (g_prime * d).sqrt() * (theta - theta_c).powf(1.5)
    }
}

/// Single-grain settling speed (paper eq. 21, Soulsby 1997).
pub(crate) fn settling_speed_single_grain(nu: f64, d: f64, r_star: f64) -> f64 {
    (nu / d) * ((10.36_f64.powi(2) + 1.049 * r_star.powi(2)).sqrt() - 10.36)
}

/// Richardson-Zaki exponent `n` (paper eq. 20), `re_s` is the settling
/// particle Reynolds number `sqrt(g*d^3)/nu`.
pub(crate) fn richardson_zaki_n(re_s: f64) -> f64 {
    (4.7 + 0.41 * re_s.powf(0.75)) / (1.0 + 0.175 * re_s.powf(0.75))
}

/// Hindered settling speed `w_s` (paper eq. 19) and deposition flux
/// `D = psi*w_s` (paper eq. 18), combined since `D` is always what's
/// actually needed.
pub(crate) fn deposition_rate(psi: f64, psi_star: f64, nu: f64, d: f64, r_star: f64) -> f64 {
    let re_s = (G * d.powi(3)).sqrt() / nu;
    let n = richardson_zaki_n(re_s);
    let ws0 = settling_speed_single_grain(nu, d, r_star);
    let ws = (1.0 - psi).powf(2.7 - 0.15 * n) * (1.0 - psi / psi_star).powf(0.62 * n - 1.46) * ws0;
    psi * ws
}

// ── Flow-front diagnostic ─────────────────────────────────────────────────────

/// Depth threshold used to locate the flow front `x_f(t)` in a diagnostic
/// `m` array -- consistent with `H_MIN`'s own "dry" definition.
// Used by the examples and the test, not by the benchmark, which does
// not track the front.
#[allow(dead_code)]
pub(crate) const FRONT_THRESHOLD: f64 = 1.0e-4;

/// Rightmost `x` where the diagnostic depth `h = m/rho_f` (pure water:
/// `psi=0` everywhere, so `h = m/rho_f` exactly, no need for the full
/// `LaharPhysics::primitives` solve) exceeds [`FRONT_THRESHOLD`].
// Unused in the benchmark, see `FRONT_THRESHOLD`.
#[allow(dead_code)]
pub(crate) fn front_position(m_column: &DVector<f64>, x_coords: &[f64], rho_f: f64) -> f64 {
    m_column
        .iter()
        .zip(x_coords)
        .filter(|(&m, _)| m / rho_f > FRONT_THRESHOLD)
        .map(|(_, &x)| x.abs())
        .fold(0.0, f64::max)
}
