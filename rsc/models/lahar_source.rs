//! Lahar multi-field model, source part: `LaharSource` (friction,
//! erosion/deposition, bed evolution, top-hat injection) and the paper's
//! Sec. 4.1 validation configuration (`ValidationRegime`, `build_source`).
//!
//! Single definition shared by `examples/lahar_multifield.rs`,
//! `examples/lahar_multifield_plot.rs`, `examples/lahar_lake_wave.rs`,
//! `tests/lahar_multifield_analytical.rs` and `benches/lahar_multifield.rs`,
//! each of which includes this file as a module with `#[path]`. The model (DD-032, Woodhouse, Hogg & Phillips) is
//! documented in `examples/lahar_multifield.rs`.
//!
//! Requires `rsc/models/lahar_physics.rs` as a sibling module named
//! `lahar_physics`.

use super::lahar_physics::{
    basal_traction, deposition_rate, erosion_rate, shields_theta_critical, ErosionDepositionParams,
    FrictionParams, LaharPhysics, SourceInjection, COL_B, COL_M, COL_P, COL_S, G, H_MIN,
};
use nalgebra::DMatrix;
use oxiflow::{
    context::{
        compute::ComputeContext, error::OxiflowError, value::ContextValue,
        variable::ContextVariable,
    },
    mesh::{Mesh, UniformGrid1D},
    model::{PhysicalModel, RequiresContext},
};

// ── Source term (friction, erosion/deposition, bed evolution, injection) ─────

/// Ordinary `PhysicalModel` -- friction, erosion/deposition, bed
/// evolution, and the top-hat source `Q(x,t)`. No spatial flux; composed
/// alongside `DiscretizedModel<ShallowLayerFlux>` via `CompositeModel`.
///
/// Stores `dx` and each node's `x` coordinate at construction time:
/// `compute_physics(&self, state, ctx)` receives neither the mesh nor a
/// coordinate accessor (per `PhysicalModel`'s own signature), so both are
/// captured once in [`LaharSource::new`] rather than recomputed per call.
pub(crate) struct LaharSource {
    pub(crate) physics: LaharPhysics,
    pub(crate) friction: FrictionParams,
    pub(crate) erosion: ErosionDepositionParams,
    pub(crate) injection: SourceInjection,
    /// Initial (flat, per Sec. 4.1's "non-erodible substrate") bed
    /// elevation.
    pub(crate) initial_b: f64,
    dx: f64,
    x_coords: Vec<f64>,
    /// Rayon dispatch threshold for the per-node friction/erosion/
    /// deposition loop -- default measured at v0.9.0, see
    /// [`default_source_parallel_threshold`]. Per-instance (DD-048
    /// P1/P2/P3), independent of `ShallowLayerFlux`'s own threshold:
    /// this site's per-node cost (three-branch Pouliquen-Forterre
    /// friction plus the Shields/settling-velocity closures) has nothing
    /// to do with the flux operator's cost profile.
    #[cfg(feature = "parallel")]
    parallel_threshold: usize,
}

/// Measured: `benches/lahar_multifield.rs`'s `source_dispatch_diagnostic` put the real crossover
/// between 100 and 1,000, consistent across runs (2.71x, then 2.90x
/// faster at 1,000 -- no sign flip): the same profile as
/// `langmuir_multi`'s own measured threshold (#138), sitting right at
/// the crossover rather than carrying a larger, unjustified safety
/// margin, measured at v0.9.0 (#53). At v0.9.0 the diagnostic timed a
/// partial copy of the per-node loop; it now times this model's
/// `compute_physics`, so the numbers above hold until the next
/// measurement.
#[cfg(feature = "parallel")]
pub(crate) fn default_source_parallel_threshold() -> usize {
    1_000
}

impl LaharSource {
    pub(crate) fn new(
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
            // Explicit-Euler deposition overshoot guard: in a thin layer,
            // `D*dt` can exceed the solids actually present and drive `s`
            // and `h` negative within a single step. A step can deposit at
            // most the solids present, never more -- the same posture as
            // the friction guard below. `D` is zero for pure water, so the
            // guard never acts there.
            let deposition = if h[i] > H_MIN {
                deposition_rate(
                    psi[i],
                    psi_star,
                    self.erosion.nu,
                    self.erosion.grain_d,
                    r_star,
                )
                .min(s[(i, COL_S)].max(0.0) / dt)
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
                -rho_b * db_dt + self.physics.rho_bar(self.injection.psi) * injection,
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

/// Which drag regime a run isolates -- see [`build_source`], and the
/// analytical-formula selection in `examples/lahar_multifield.rs`
/// (`run_and_report`) and `tests/lahar_multifield_analytical.rs`
/// (`run_regime`).
// The plot example and the benchmark run the Chezy-drag-dominated regime
// only; the other variants are used by the main example and the test.
#[allow(dead_code)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ValidationRegime {
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
// The coupled-grid example builds its own configuration and does not call
// this function.
#[allow(dead_code)]
pub(crate) fn build_source(
    mesh: &UniformGrid1D,
    regime: ValidationRegime,
    source_rate: f64,
) -> LaharSource {
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
