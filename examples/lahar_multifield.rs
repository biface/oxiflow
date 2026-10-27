//! # Example: multi-field lahar benchmark model (#139, feeds #53)
//!
//! Implements DD-032 (#85): a physically representative lahar model per
//! Woodhouse, Hogg & Phillips (LaharFlowPaper.pdf, cited in DD-032),
//! reduced to one spatial dimension. Four coupled conservation quantities,
//! not three -- see the state layout note below.
//!
//! The model is defined once in `rsc/models/`, in three files included
//! below as modules of the same names: `lahar_physics.rs` (state layout,
//! density mixing, closures), `lahar_flux.rs` (`ShallowLayerFlux`) and
//! `lahar_source.rs` (`LaharSource` and the Sec. 4.1 validation
//! configuration). They are shared with the plot example, the analytical
//! test and the benchmark.
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

use oxiflow::{
    context::{
        calculators::{FDGradientCalculator, FDScheme},
        error::OxiflowError,
    },
    mesh::{Mesh, UniformGrid1D},
    model::{CompositeModel, DiscretizedModel},
    operators::limiters::Limiter,
    solver::{
        config::{IntegratorKind, StepControl, TimeConfiguration},
        methods::ForwardEulerSolver,
        scenario::Scenario,
        Solver, SolverConfiguration,
    },
};

#[path = "../rsc/models/lahar_flux.rs"]
mod lahar_flux;
#[path = "../rsc/models/lahar_physics.rs"]
mod lahar_physics;
#[path = "../rsc/models/lahar_source.rs"]
mod lahar_source;

#[cfg(feature = "parallel")]
use lahar_flux::default_flux_parallel_threshold;
use lahar_flux::ShallowLayerFlux;
use lahar_physics::{front_position, LaharPhysics, COL_B, COL_M};
use lahar_source::{build_source, ValidationRegime};

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

    let source_mesh = UniformGrid1D::new(n_points, -x_half_width, x_half_width)?;
    let source = build_source(&source_mesh, regime, Q);

    let flux_mesh: Arc<UniformGrid1D> =
        Arc::new(UniformGrid1D::new(n_points, -x_half_width, x_half_width)?);
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

    let grad_mesh: Arc<dyn Mesh> =
        Arc::new(UniformGrid1D::new(n_points, -x_half_width, x_half_width)?);
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

    let scenario_mesh = UniformGrid1D::new(n_points, -x_half_width, x_half_width)?;
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
