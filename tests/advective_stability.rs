//! # Integration test: advective stability limit ([DA-073](https://github.com/biface/oxiflow/issues/229), [#228](https://github.com/biface/oxiflow/issues/228))
//!
//! The advective stability condition belongs to the explicit integrators,
//! not to the flux schemes. Through the public API only, these tests check
//! that each implicit integrator runs every flux scheme of the crate above
//! a Courant number of 1, that forward Euler and RK4 refuse such a step
//! with an error naming the solver, and that DoPri45 shortens its step to
//! the limit instead of failing. The model reaches the solvers wrapped in
//! `PrescribedInitialState` and `CompositeModel`, which must forward the
//! wave speed of the scheme.

use std::f64::consts::PI;
use std::sync::Arc;

use nalgebra::DVector;
use oxiflow::{
    context::variable::ContextVariable,
    context::{compute::ComputeContext, error::OxiflowError, value::ContextValue},
    mesh::{Mesh, UniformGrid1D},
    model::{
        CompositeModel, DiscretizedModel, PhysicalModel, PrescribedInitialState, RequiresContext,
    },
    operators::{
        adaptive::AdaptiveFlux,
        fv::{FVCenteredFlux, FVUpwindFlux},
        limiters::{LimitedFlux, Limiter},
        weno::{WENO3, WENO5},
        FluxBoundary, FluxDivergenceOperator,
    },
    solver::{
        config::{IntegratorKind, StepControl, TimeConfiguration},
        methods::{
            BDF2Solver, BackwardEulerSolver, CrankNicolsonSolver, DoPri45Solver,
            ForwardEulerSolver, JacobianStrategy, RK4Solver,
        },
        Scenario, Solver, SolverConfiguration,
    },
};

/// 51 nodes on (0, 1), periodic: `dx = 0.02`.
const N_POINTS: usize = 51;
const VELOCITY: f64 = 1.0;
/// Courant number `VELOCITY · DT / dx = 2`.
const DT: f64 = 0.04;
const T_END: f64 = 0.2;
/// Newton budget of the implicit integrators (see the implicit test).
const NEWTON_ITERATIONS: usize = 20;

fn grid() -> UniformGrid1D {
    UniformGrid1D::new(N_POINTS, 0.0, 1.0).unwrap()
}

fn dx() -> f64 {
    grid().characteristic_length()
}

/// `u₀(x) = 1 + ½·sin(2πx)`: smooth and periodic.
fn initial_state() -> ContextValue {
    let mesh = grid();
    ContextValue::ScalarField(DVector::from_fn(N_POINTS, |i, _| {
        1.0 + 0.5 * (2.0 * PI * mesh.coordinates(i)[0]).sin()
    }))
}

fn discretized<Op>(scheme: Op, name: &str) -> Box<dyn PhysicalModel>
where
    Op: FluxDivergenceOperator<MeshType = UniformGrid1D> + 'static,
{
    let model = DiscretizedModel::new(Arc::new(scheme), Arc::new(grid()), name);
    Box::new(PrescribedInitialState::new(
        Box::new(model),
        initial_state(),
    ))
}

/// Every flux scheme of the crate, pure advection at `VELOCITY`, periodic.
fn flux_models() -> Vec<Box<dyn PhysicalModel>> {
    let p = || FluxBoundary::Periodic;
    vec![
        discretized(FVCenteredFlux::new(VELOCITY, 0.0, p()), "fv_centered"),
        discretized(FVUpwindFlux::new(VELOCITY, 0.0, p()), "fv_upwind"),
        discretized(WENO3::new(VELOCITY, 0.0, p()), "weno3"),
        discretized(WENO5::new(VELOCITY, 0.0, p()), "weno5"),
        discretized(
            LimitedFlux::new(VELOCITY, 0.0, Limiter::VanLeer, p()),
            "limited",
        ),
        discretized(
            AdaptiveFlux::new(VELOCITY, 0.0, Limiter::MinMod, 1.0, p()),
            "adaptive",
        ),
    ]
}

fn scenario(model: Box<dyn PhysicalModel>) -> Scenario {
    Scenario::single(model, Box::new(grid()))
}

fn fixed(integrator: IntegratorKind) -> SolverConfiguration {
    SolverConfiguration::new(
        TimeConfiguration::new(T_END, StepControl::Fixed { dt: DT }),
        integrator,
    )
}

fn mass(state: &ContextValue) -> f64 {
    state.as_scalar_field().unwrap().sum()
}

// ── Implicit integrators: no advective limit ──────────────────────────────────

#[test]
fn implicit_integrators_run_every_flux_scheme_above_courant_one() {
    // WENO and the limiters are nonlinear in u: the default single Newton
    // iteration with a frozen Jacobian does not converge at this step, so
    // full Newton with a larger budget is set, as the documentation of
    // `NewtonNotConverged` advises (DD-044).
    let solvers: Vec<(&str, Box<dyn Solver>, IntegratorKind)> = vec![
        (
            "backward_euler",
            Box::new(
                BackwardEulerSolver::new()
                    .with_jacobian_strategy(JacobianStrategy::FullNewton)
                    .with_max_newton_iterations(NEWTON_ITERATIONS),
            ),
            IntegratorKind::BackwardEuler,
        ),
        (
            "crank_nicolson",
            Box::new(
                CrankNicolsonSolver::new()
                    .with_jacobian_strategy(JacobianStrategy::FullNewton)
                    .with_max_newton_iterations(NEWTON_ITERATIONS),
            ),
            IntegratorKind::CrankNicolson,
        ),
        (
            "bdf2",
            Box::new(
                BDF2Solver::new()
                    .with_jacobian_strategy(JacobianStrategy::FullNewton)
                    .with_max_newton_iterations(NEWTON_ITERATIONS),
            ),
            IntegratorKind::BDF2,
        ),
    ];
    let mass_0 = mass(&initial_state());
    for (solver_name, solver, kind) in &solvers {
        for model in flux_models() {
            let model_name = model.name().to_string();
            let result = solver
                .solve(&scenario(model), &fixed(kind.clone()))
                .unwrap_or_else(|e| panic!("{solver_name} with {model_name}: {e}"));
            assert_eq!(result.n_steps, 5, "{solver_name} with {model_name}");
            let last = result.states.last().unwrap();
            assert!(
                last.as_scalar_field()
                    .unwrap()
                    .iter()
                    .all(|v| v.is_finite()),
                "{solver_name} with {model_name}: non-finite state"
            );
            // Periodic flux divergences telescope: the mass is kept.
            let drift = (mass(last) - mass_0).abs() / mass_0;
            assert!(
                drift < 1e-6,
                "{solver_name} with {model_name}: drift {drift}"
            );
        }
    }
}

// ── Explicit integrators with a fixed step: refused ───────────────────────────

fn assert_refused_by(err: OxiflowError, solver_context: &str) {
    match err {
        OxiflowError::PreconditionFailed { context, message } => {
            assert_eq!(context, solver_context);
            assert!(message.contains("Courant number 2."), "{message}");
        }
        other => panic!("expected PreconditionFailed from {solver_context}, got {other:?}"),
    }
}

#[test]
fn forward_euler_and_rk4_refuse_a_step_above_courant_one() {
    for model in flux_models() {
        let err = ForwardEulerSolver
            .solve(&scenario(model), &fixed(IntegratorKind::Euler))
            .unwrap_err();
        assert_refused_by(err, "ForwardEulerSolver::step");
    }
    for model in flux_models() {
        let err = RK4Solver
            .solve(&scenario(model), &fixed(IntegratorKind::RK4))
            .unwrap_err();
        assert_refused_by(err, "RK4Solver::step");
    }
}

#[test]
fn forward_euler_runs_at_the_limit_of_courant_one() {
    let config = SolverConfiguration::new(
        TimeConfiguration::new(T_END, StepControl::Fixed { dt: 0.5 * DT }),
        IntegratorKind::Euler,
    );
    let model = discretized(
        FVUpwindFlux::new(VELOCITY, 0.0, FluxBoundary::Periodic),
        "fv_upwind",
    );
    assert!(ForwardEulerSolver.solve(&scenario(model), &config).is_ok());
}

/// A source term without advection: declares no wave speed.
struct UniformSource;

impl RequiresContext for UniformSource {
    fn required_variables(&self) -> Vec<ContextVariable> {
        vec![]
    }
}

impl PhysicalModel for UniformSource {
    fn compute_physics(
        &self,
        state: &ContextValue,
        _ctx: &ComputeContext,
    ) -> Result<ContextValue, OxiflowError> {
        let n = state.as_scalar_field()?.len();
        Ok(ContextValue::ScalarField(DVector::from_element(n, 0.1)))
    }

    fn initial_state(&self, _mesh: &dyn Mesh) -> ContextValue {
        initial_state()
    }

    fn name(&self) -> &str {
        "uniform_source"
    }
}

#[test]
fn a_composite_model_carries_the_speed_of_its_flux_term() {
    // Two advective terms at |v| = 1 and a source: speed 2, so a step at
    // Courant number 1 for one term is refused for the sum.
    let composite = CompositeModel::new(
        vec![
            Box::new(UniformSource),
            discretized(
                FVUpwindFlux::new(VELOCITY, 0.0, FluxBoundary::Periodic),
                "upwind_a",
            ),
            discretized(
                FVUpwindFlux::new(-VELOCITY, 0.0, FluxBoundary::Periodic),
                "upwind_b",
            ),
        ],
        "advection_and_source",
    )
    .unwrap();
    let config = SolverConfiguration::new(
        TimeConfiguration::new(T_END, StepControl::Fixed { dt: 0.5 * DT }),
        IntegratorKind::Euler,
    );
    let err = ForwardEulerSolver
        .solve(&scenario(Box::new(composite)), &config)
        .unwrap_err();
    assert_refused_by(err, "ForwardEulerSolver::step");
}

// ── DoPri45: shortens its step ───────────────────────────────────────────────

#[test]
fn dopri45_shortens_its_steps_to_the_advective_limit() {
    let config = SolverConfiguration::new(
        TimeConfiguration::new(
            T_END,
            StepControl::Adaptive {
                dt_init: DT,
                dt_min: 1e-8,
                dt_max: 1.0,
                rtol: 1e-3,
                atol: 1e-6,
            },
        ),
        IntegratorKind::DoPri45,
    );
    let model = discretized(
        FVUpwindFlux::new(VELOCITY, 0.0, FluxBoundary::Periodic),
        "fv_upwind",
    );
    let result = DoPri45Solver.solve(&scenario(model), &config).unwrap();

    assert!(result.metadata["solver.cfl_limited_steps"] >= 1.0);
    let dt_limit = dx() / VELOCITY;
    for pair in result.times.windows(2) {
        assert!(pair[1] - pair[0] <= dt_limit * (1.0 + 1e-12));
    }
    assert!((result.times.last().unwrap() - T_END).abs() < 1e-12);
}
