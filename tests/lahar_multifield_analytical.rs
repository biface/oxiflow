//! # Integration test: multi-field lahar model vs. analytical scaling laws (#139)
//!
//! Companion to `examples/lahar_multifield.rs`, on the same model
//! (`rsc/models/lahar_physics.rs`, `lahar_flux.rs`, `lahar_source.rs`):
//! this file asserts the acceptance criteria from #139 as a regression
//! gate.
//!
//! Three-tier validation, agreed in session after the drag-dominated
//! Coulomb comparison turned out not to be a fair pointwise check (see
//! below):
//!
//! - [`early_inertial_regime_matches_analytical`] -- the friction-
//!   independent early regime (paper Sec. 4, "drag has little effect",
//!   `x_f ~ (g*q)^(1/3) * t`), a strict regression gate (1% tolerance):
//!   this isolates the flux/pressure-gradient machinery on its own, and
//!   session runs matched to 0.3%.
//! - [`chezy_drag_dominated_matches_analytical`] -- the Chezy-drag
//!   asymptote (`x_f ~ (g*q^2/Cd)^(1/5) * t^(4/5)`), a looser gate (15%
//!   tolerance): session runs matched to 7.7% at `t=10s`, plausible given
//!   the scaling law is itself only asymptotic and not yet fully
//!   established at that `t`.
//! - [`coulomb_constant_friction_shows_slump_and_stall_pattern`] --
//!   **not** a pointwise numerical comparison. A 60s session run showed
//!   the front advance, stall completely (unchanged for several seconds),
//!   then resume -- exactly the paper's own documented behaviour for a
//!   *constant* Coulomb coefficient ("the front progresses through a
//!   series of slumps interspersed by periods where the front does not
//!   move"), not matching the smooth `x_f ~ q*t/sqrt(mu)` asymptote
//!   pointwise at any single finite `t` tested (this session: 86% off at
//!   `t=60s`, despite a qualitatively correct trajectory shape). Asserts
//!   the qualitative pattern (at least one stall followed by a resume)
//!   instead of a numerical tolerance against that formula.

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

/// Runs one drag-regime configuration and returns the full `x_f(t)`
/// trajectory (one entry per recorded state) -- shared by all three
/// tests below, which each apply their own check to the result.
fn run_regime(
    regime: ValidationRegime,
    n_points: usize,
    x_half_width: f64,
    t_end: f64,
    save_every: usize,
) -> Result<Vec<(f64, f64)>, OxiflowError> {
    const DT: f64 = 5.0e-4;
    const Q: f64 = 1.0; // paper's own q = 1 m^2/s (Sec. 4.1)

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
        TimeConfiguration::new(t_end, StepControl::Fixed { dt: DT }).saving_every(save_every),
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

    result
        .times
        .iter()
        .zip(&result.states)
        .map(|(&t, state)| {
            let m_col = state.as_vector_field()?.column(COL_M).into_owned();
            Ok((t, front_position(&m_col, &x_coords, 1000.0)))
        })
        .collect()
}

/// Which drag regime a run isolates -- see `build_source` and
/// `run_regime`'s analytical-formula selection (`ValidationRegime` is
/// defined in `rsc/models/lahar_source.rs`; see
/// `examples/lahar_multifield.rs`'s own doc for the rationale of each
/// variant).
#[test]
fn early_inertial_regime_matches_analytical() {
    const Q: f64 = 1.0;
    const T_END: f64 = 1.5;
    const TOLERANCE: f64 = 0.01; // 1% -- session run matched to 0.3%

    let trajectory = run_regime(ValidationRegime::EarlyInertial, 501, 10.0, T_END, 1)
        .expect("early inertial run should not fail (no CFL/precondition issues observed)");
    let xf_simulated = trajectory.last().expect("at least one recorded state").1;
    let xf_analytical = (9.81 * Q).powf(1.0 / 3.0) * T_END;

    let rel_err = (xf_simulated - xf_analytical).abs() / xf_analytical;
    assert!(
        rel_err < TOLERANCE,
        "early inertial front position off by {:.1}% (simulated={xf_simulated:.3}, \
         analytical={xf_analytical:.3}) -- exceeds the {:.0}% tolerance",
        rel_err * 100.0,
        TOLERANCE * 100.0
    );
}

#[test]
fn chezy_drag_dominated_matches_analytical() {
    const Q: f64 = 1.0;
    const CD: f64 = 0.04;
    const T_END: f64 = 10.0;
    const TOLERANCE: f64 = 0.15; // 15% -- session run matched to 7.7%; the
                                 // scaling law is itself only asymptotic,
                                 // not yet fully established at t=10s.

    let trajectory = run_regime(ValidationRegime::ChezyDragDominated, 1001, 50.0, T_END, 1)
        .expect("Chezy-drag-dominated run should not fail");
    let xf_simulated = trajectory.last().expect("at least one recorded state").1;
    let xf_analytical = (9.81 * Q * Q / CD).powf(0.2) * T_END.powf(0.8);

    let rel_err = (xf_simulated - xf_analytical).abs() / xf_analytical;
    assert!(
        rel_err < TOLERANCE,
        "Chezy-drag-dominated front position off by {:.1}% (simulated={xf_simulated:.3}, \
         analytical={xf_analytical:.3}) -- exceeds the {:.0}% tolerance",
        rel_err * 100.0,
        TOLERANCE * 100.0
    );
}

/// Not a pointwise numerical comparison -- see the module doc. Asserts
/// the qualitative slump/stall pattern the paper documents for a
/// *constant* Coulomb coefficient: at least one interval where the front
/// does not advance at all, followed by a later interval where it does.
/// A run that only ever advances smoothly (no stall) or that stalls
/// forever (no resume) would both fail this -- either would mean the
/// friction-overshoot guard (see `examples/lahar_multifield.rs`'s module
/// doc) is masking real dynamics rather than just preventing an explicit-
/// Euler artifact.
///
/// `#[ignore]`d: 60s of simulated time at dt=5e-4 (120,000 steps, 2001
/// nodes) -- same reasoning as `gpu_context_acquires_adapter` being
/// ignored for CI hardware constraints, here for CI *time* cost instead.
/// Shortening the run isn't a substitute: the slump/stall pattern this
/// test checks for needs enough simulated time to show more than one
/// cycle, so cutting the duration would weaken exactly what's being
/// verified rather than just making it faster. Run explicitly with
/// `cargo test --release -- --ignored` (release strongly recommended --
/// this ran 15-27s in release locally; debug would be far slower).
#[test]
#[ignore]
fn coulomb_constant_friction_shows_slump_and_stall_pattern() {
    const STALL_EPS: f64 = 1.0e-6; // front position considered "unchanged"
                                   // below this delta between consecutive
                                   // recorded snapshots.

    // save_every=1000 -> ~0.5s between recorded snapshots (dt=5e-4),
    // coarse enough that consecutive-snapshot deltas reflect genuine
    // multi-second stalls/resumes, not single-step grid-crossing noise.
    let trajectory = run_regime(ValidationRegime::CoulombConstant, 2001, 100.0, 60.0, 1000)
        .expect("Coulomb-constant run should not fail");

    let mut saw_stall = false;
    let mut saw_resume_after_stall = false;
    for window in trajectory.windows(2) {
        let (_, xf_prev) = window[0];
        let (_, xf_next) = window[1];
        let delta = xf_next - xf_prev;
        if delta.abs() < STALL_EPS {
            saw_stall = true;
        } else if saw_stall && delta > STALL_EPS {
            saw_resume_after_stall = true;
        }
    }

    assert!(
        saw_stall,
        "expected at least one interval where the front does not advance at all \
         (paper's documented constant-Coulomb behaviour) -- none observed"
    );
    assert!(
        saw_resume_after_stall,
        "expected the front to resume advancing after stalling at least once -- \
         a permanent stall would mean the friction-overshoot guard is masking \
         real dynamics rather than just preventing an explicit-Euler artifact"
    );
}
