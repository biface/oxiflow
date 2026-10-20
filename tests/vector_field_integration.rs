//! # Integration test: time integration of multi-component states (#192)
//!
//! Every integrator advances a `VectorField` state (n nodes × d components)
//! the same way it advances a `ScalarField`, and divergence detection sees a
//! non-finite value in any component. The reference problem is a pointwise
//! linear system `du/dt = A u` at every node, whose exact discrete and
//! continuous solutions are known.

use std::sync::Mutex;

use nalgebra::{DMatrix, Matrix2, Vector2};
use oxiflow::{
    context::{
        compute::ComputeContext, error::OxiflowError, value::ContextValue,
        variable::ContextVariable, ContextCalculator,
    },
    mesh::{Mesh, UniformGrid1D},
    model::traits::{PhysicalModel, RequiresContext},
    solver::{
        config::{IntegratorKind, StepControl, TimeConfiguration},
        methods::{DoPri45Solver, ForwardEulerSolver, RK4Solver, SteppableSolver},
        scenario::{Domain, Scenario},
        snapshot::SimulationSnapshot,
        SimulationResult, Solver, SolverConfiguration,
    },
};

// ── Reference model ───────────────────────────────────────────────────────────

/// `du/dt = A u` at every node, for a two-component state `u = (u₀, u₁)`.
/// Node `i` of a mesh on `[0, 1]` starts at `(1 + x_i, 2 − x_i)`, so every
/// row of the state is distinct.
struct LinearSystem2 {
    matrix: Matrix2<f64>,
}

impl LinearSystem2 {
    fn new(matrix: Matrix2<f64>) -> Self {
        Self { matrix }
    }

    fn initial_row(x: f64) -> Vector2<f64> {
        Vector2::new(1.0 + x, 2.0 - x)
    }
}

impl RequiresContext for LinearSystem2 {
    fn required_variables(&self) -> Vec<ContextVariable> {
        vec![]
    }
}

impl PhysicalModel for LinearSystem2 {
    fn compute_physics(
        &self,
        state: &ContextValue,
        _ctx: &ComputeContext,
    ) -> Result<ContextValue, OxiflowError> {
        let u = state.as_vector_field()?;
        if u.ncols() != 2 {
            return Err(OxiflowError::InvalidDomain(format!(
                "expected 2 components, got {}",
                u.ncols()
            )));
        }
        // Rows are nodes: (A uᵢ)ᵀ = uᵢᵀ Aᵀ for every row at once.
        let a_t = DMatrix::from_column_slice(2, 2, self.matrix.transpose().as_slice());
        Ok(ContextValue::VectorField(u * a_t))
    }

    fn initial_state(&self, mesh: &dyn Mesh) -> ContextValue {
        let n = mesh.n_dof();
        let mut u = DMatrix::zeros(n, 2);
        for i in 0..n {
            let row = Self::initial_row(mesh.coordinates(i)[0]);
            u[(i, 0)] = row[0];
            u[(i, 1)] = row[1];
        }
        ContextValue::VectorField(u)
    }

    fn name(&self) -> &str {
        "linear_system_2"
    }
}

// ── Helpers ───────────────────────────────────────────────────────────────────

const N_NODES: usize = 5;

fn mesh() -> UniformGrid1D {
    UniformGrid1D::new(N_NODES, 0.0, 1.0).unwrap()
}

fn scenario(matrix: Matrix2<f64>) -> Scenario {
    Scenario::single(Box::new(LinearSystem2::new(matrix)), Box::new(mesh()))
}

fn fixed_step(t_end: f64, dt: f64, integrator: IntegratorKind) -> SolverConfiguration {
    SolverConfiguration::new(
        TimeConfiguration::new(t_end, StepControl::Fixed { dt }),
        integrator,
    )
}

/// Largest absolute gap between the final state and `expected(x)` per node.
fn max_error(result: &SimulationResult, expected: impl Fn(f64) -> Vector2<f64>) -> f64 {
    let last = result.states.last().unwrap().as_vector_field().unwrap();
    let mesh = mesh();
    assert_eq!(last.shape(), (N_NODES, 2));
    (0..N_NODES)
        .flat_map(|i| {
            let e = expected(mesh.coordinates(i)[0]);
            [(last[(i, 0)] - e[0]).abs(), (last[(i, 1)] - e[1]).abs()]
        })
        .fold(0.0, f64::max)
}

/// Coupled system with distinct real eigenvalues (−1 and −3).
fn coupled_matrix() -> Matrix2<f64> {
    Matrix2::new(-2.0, 1.0, 1.0, -2.0)
}

/// `exp(t A)` for [`coupled_matrix`], from its eigen-decomposition
/// (eigenvectors `(1, 1)` and `(1, −1)`).
fn coupled_exact_propagator(t: f64) -> Matrix2<f64> {
    let (slow, fast) = ((-t).exp(), (-3.0 * t).exp());
    Matrix2::new(slow + fast, slow - fast, slow - fast, slow + fast) * 0.5
}

/// `n_steps` applications of the one-step matrix `step`.
fn power(step: Matrix2<f64>, n_steps: usize) -> Matrix2<f64> {
    (0..n_steps).fold(Matrix2::identity(), |p, _| step * p)
}

// ── Forward Euler ─────────────────────────────────────────────────────────────

#[test]
fn forward_euler_matches_its_discrete_solution_on_a_coupled_system() {
    // Forward Euler is exactly u_{k+1} = (I + dt A) u_k.
    let (dt, n_steps) = (0.01, 50);
    let a = coupled_matrix();
    let result = ForwardEulerSolver
        .solve(
            &scenario(a),
            &fixed_step(dt * n_steps as f64, dt, IntegratorKind::Euler),
        )
        .unwrap();
    let propagator = power(Matrix2::identity() + a * dt, n_steps);
    let error = max_error(&result, |x| propagator * LinearSystem2::initial_row(x));
    assert!(error < 1e-12, "max error = {error:e}");
}

// ── Classical Runge–Kutta ─────────────────────────────────────────────────────

#[test]
fn rk4_matches_its_discrete_solution_on_a_coupled_system() {
    // On a linear system, one RK4 step is the degree-4 Taylor polynomial of
    // exp(dt A).
    let (dt, n_steps) = (0.05, 20);
    let a = coupled_matrix();
    let result = RK4Solver
        .solve(
            &scenario(a),
            &fixed_step(dt * n_steps as f64, dt, IntegratorKind::RK4),
        )
        .unwrap();
    let h = a * dt;
    let step = Matrix2::identity() + h + h * h / 2.0 + h * h * h / 6.0 + h * h * h * h / 24.0;
    let propagator = power(step, n_steps);
    let error = max_error(&result, |x| propagator * LinearSystem2::initial_row(x));
    assert!(error < 1e-12, "max error = {error:e}");
}

#[test]
fn rk4_converges_to_the_exact_solution_at_fourth_order() {
    let a = coupled_matrix();
    let t_end = 1.0;
    let error_at = |dt: f64| {
        let result = RK4Solver
            .solve(&scenario(a), &fixed_step(t_end, dt, IntegratorKind::RK4))
            .unwrap();
        max_error(&result, |x| {
            coupled_exact_propagator(t_end) * LinearSystem2::initial_row(x)
        })
    };
    let (coarse, fine) = (error_at(0.1), error_at(0.05));
    let order = (coarse / fine).log2();
    assert!((3.8..4.2).contains(&order), "observed order = {order:.3}");
}

// ── Dormand–Prince 4(5) ───────────────────────────────────────────────────────

#[test]
fn dopri45_follows_the_exact_solution_within_its_tolerance() {
    let a = coupled_matrix();
    let t_end = 2.0;
    let config = SolverConfiguration::new(
        TimeConfiguration::new(
            t_end,
            StepControl::Adaptive {
                dt_init: 0.1,
                dt_min: 1e-8,
                dt_max: 0.5,
                rtol: 1e-8,
                atol: 1e-10,
            },
        ),
        IntegratorKind::DoPri45,
    );
    let result = DoPri45Solver.solve(&scenario(a), &config).unwrap();
    assert!((result.times.last().unwrap() - t_end).abs() < 1e-12);
    assert!(result.metadata["solver.accepted_steps"] > 1.0);
    let error = max_error(&result, |x| {
        coupled_exact_propagator(t_end) * LinearSystem2::initial_row(x)
    });
    assert!(error < 1e-7, "max error = {error:e}");
}

// ── Divergence detection ──────────────────────────────────────────────────────

/// Forward Euler that keeps the snapshot handed to `on_divergence`.
#[derive(Default)]
struct RecordingEuler {
    snapshot: Mutex<Option<SimulationSnapshot>>,
}

impl Solver for RecordingEuler {
    fn solve(
        &self,
        scenario: &Scenario,
        config: &SolverConfiguration,
    ) -> Result<SimulationResult, OxiflowError> {
        self.solve_fixed_step(scenario, config)
    }

    fn on_divergence(&self, snapshot: &SimulationSnapshot) {
        *self.snapshot.lock().unwrap() = Some(snapshot.clone());
    }
}

impl SteppableSolver for RecordingEuler {
    fn step(
        &self,
        domain: &Domain,
        chain: &[&dyn ContextCalculator],
        state: &mut ContextValue,
        history: &[ContextValue],
        t: f64,
        dt: f64,
    ) -> Result<ContextValue, OxiflowError> {
        ForwardEulerSolver.step(domain, chain, state, history, t, dt)
    }
}

#[test]
fn diverging_vector_field_raises_solver_divergence_with_a_snapshot() {
    // Growth factor 1 + 1e200 per step: the state overflows to infinity on
    // the second step. Before #192 this run ended with Ok and infinite states.
    let solver = RecordingEuler::default();
    let result = solver.solve(
        &scenario(Matrix2::identity() * 1e200),
        &fixed_step(10.0, 1.0, IntegratorKind::Euler),
    );
    assert!(
        matches!(result, Err(OxiflowError::SolverDivergence { .. })),
        "expected SolverDivergence, got {result:?}"
    );

    let snapshot = solver
        .snapshot
        .into_inner()
        .unwrap()
        .expect("on_divergence must be called");
    let state = snapshot.state.as_vector_field().unwrap();
    assert_eq!(state.shape(), (N_NODES, 2));
    assert!(state.iter().any(|v| !v.is_finite()));
    assert_eq!(snapshot.n_steps, 2);
    let partial = snapshot.partial.expect("partial result must be attached");
    assert!(partial.states.iter().all(|s| s
        .as_vector_field()
        .unwrap()
        .iter()
        .all(|v| v.is_finite())));
}
