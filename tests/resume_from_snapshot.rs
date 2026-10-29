//! # Resuming a run from a `SimulationSnapshot`
//!
//! Checks the restore contract of
//! [DT-025](https://github.com/biface/oxiflow/issues/66) as the
//! `model::initial_condition` documentation states it: a run snapshotted
//! after N steps and resumed with `PrescribedInitialState` and
//! `Scenario::with_t_start` for M more steps matches an uninterrupted N + M
//! run. States are identical for one-step integrators and a model that does
//! not read the time; times agree to rounding; a multistep integrator (BDF2)
//! restarts on its bootstrap step and does not reproduce the uninterrupted
//! run.

use nalgebra::DVector;
use oxiflow::context::{ComputeContext, ContextValue, ContextVariable, OxiflowError};
use oxiflow::mesh::{Mesh, UniformGrid1D};
use oxiflow::model::{PhysicalModel, PrescribedInitialState, RequiresContext};
use oxiflow::solver::methods::{BDF2Solver, ForwardEulerSolver, RK4Solver};
use oxiflow::solver::snapshot::SimulationSnapshot;
use oxiflow::solver::{
    IntegratorKind, Scenario, SimulationResult, Solver, SolverConfiguration, StepControl,
    TimeConfiguration,
};

const N_NODES: usize = 8;
/// Not a power of two, so that `t_start + k·dt` rounds differently from
/// `(N + k)·dt`.
const DT: f64 = 0.1;
const N_FIRST: usize = 7;
const N_REST: usize = 9;

/// `du_i/dt = -u_i^3`, from a different value at every node: nonlinear and
/// autonomous (it never reads the time).
struct CubicDecay;

impl RequiresContext for CubicDecay {
    fn required_variables(&self) -> Vec<ContextVariable> {
        vec![]
    }
}

impl PhysicalModel for CubicDecay {
    fn compute_physics(
        &self,
        state: &ContextValue,
        _ctx: &ComputeContext,
    ) -> Result<ContextValue, OxiflowError> {
        Ok(ContextValue::ScalarField(
            state.as_scalar_field()?.map(|v| -v * v * v),
        ))
    }

    fn initial_state(&self, mesh: &dyn Mesh) -> ContextValue {
        ContextValue::ScalarField(DVector::from_fn(mesh.n_dof(), |i, _| 1.0 + 0.25 * i as f64))
    }

    fn name(&self) -> &str {
        "cubic_decay"
    }
}

fn mesh() -> Box<dyn Mesh> {
    Box::new(UniformGrid1D::new(N_NODES, 0.0, 1.0).unwrap())
}

fn config(t_end: f64, integrator: IntegratorKind) -> SolverConfiguration {
    SolverConfiguration::new(
        TimeConfiguration::new(t_end, StepControl::Fixed { dt: DT }),
        integrator,
    )
}

/// Runs N + M steps in one go, then N steps, a snapshot, and M steps resumed
/// from it. Returns the uninterrupted run and the resumed one.
fn uninterrupted_and_resumed(
    solver: &dyn Solver,
    integrator: IntegratorKind,
) -> (SimulationResult, SimulationResult) {
    let t_end = (N_FIRST + N_REST) as f64 * DT;
    let whole = solver
        .solve(
            &Scenario::single(Box::new(CubicDecay), mesh()),
            &config(t_end, integrator.clone()),
        )
        .unwrap();

    let first_config = config(N_FIRST as f64 * DT, integrator.clone());
    let first = solver
        .solve(
            &Scenario::single(Box::new(CubicDecay), mesh()),
            &first_config,
        )
        .unwrap();
    assert_eq!(first.n_steps, N_FIRST);

    let snapshot = SimulationSnapshot {
        time_config: first_config.time.clone(),
        integrator: integrator.clone(),
        state: first.states.last().unwrap().clone(),
        t: first.t_final().unwrap(),
        n_steps: first.n_steps,
        error: None,
        partial: None,
    };

    let model = PrescribedInitialState::new(Box::new(CubicDecay), snapshot.state.clone());
    let resumed_scenario = Scenario::single(Box::new(model), mesh()).with_t_start(snapshot.t);
    let resumed = solver
        .solve(&resumed_scenario, &config(t_end, integrator))
        .unwrap();
    assert_eq!(resumed.n_steps, N_REST);
    (whole, resumed)
}

/// Resumed states equal the tail of the uninterrupted run bit for bit;
/// resumed times agree with it to rounding. The tolerance is 4 ε relative;
/// the largest difference measured at these settings is 0.77 ε.
fn assert_resume_reproduces(whole: &SimulationResult, resumed: &SimulationResult) {
    let tail_states = &whole.states[N_FIRST..];
    let tail_times = &whole.times[N_FIRST..];
    assert_eq!(resumed.states.len(), tail_states.len());
    assert_eq!(resumed.states, tail_states);
    for (resumed_t, whole_t) in resumed.times.iter().zip(tail_times) {
        assert!(
            (resumed_t - whole_t).abs() <= 4.0 * f64::EPSILON * whole_t.abs(),
            "time {resumed_t} vs {whole_t}"
        );
    }
}

#[test]
fn a_forward_euler_run_resumed_from_a_snapshot_reproduces_the_uninterrupted_run() {
    let (whole, resumed) = uninterrupted_and_resumed(&ForwardEulerSolver, IntegratorKind::Euler);
    assert_resume_reproduces(&whole, &resumed);
}

#[test]
fn an_rk4_run_resumed_from_a_snapshot_reproduces_the_uninterrupted_run() {
    let (whole, resumed) = uninterrupted_and_resumed(&RK4Solver, IntegratorKind::RK4);
    assert_resume_reproduces(&whole, &resumed);
}

#[test]
fn the_resumed_times_are_not_all_bit_identical_to_the_uninterrupted_ones() {
    // Guards the documented caveat: if this ever fails, the loop no longer
    // computes `t_start + k·dt`, and the documentation must say so.
    let (whole, resumed) = uninterrupted_and_resumed(&ForwardEulerSolver, IntegratorKind::Euler);
    assert_ne!(resumed.times, whole.times[N_FIRST..]);
}

#[test]
fn a_bdf2_run_resumed_from_a_snapshot_restarts_on_its_bootstrap_step() {
    // Guards the documented limitation: the snapshot carries no history.
    // Iterated Newton: the default single iteration does not converge on
    // this nonlinear model at this `dt`.
    let (whole, resumed) = uninterrupted_and_resumed(
        &BDF2Solver::new().with_max_newton_iterations(20),
        IntegratorKind::BDF2,
    );
    assert_eq!(resumed.states[0], whole.states[N_FIRST]);
    assert_ne!(resumed.states[1], whole.states[N_FIRST + 1]);
}
