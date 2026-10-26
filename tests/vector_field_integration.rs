//! # Integration test: time integration of multi-component states (#192)
//!
//! Every integrator (Forward Euler, RK4, DoPri45, Backward Euler,
//! Crank–Nicolson, BDF2, Strang splitting) advances a `VectorField` state
//! (n nodes × d components) the same way it advances a `ScalarField`, and
//! divergence detection sees a non-finite value in any component. The
//! reference problem is a pointwise linear system `du/dt = A u` at every
//! node, whose exact discrete and continuous solutions are known; a
//! reaction–diffusion system checks the banded Jacobian (feature `sparse`).
//! Boundary conditions applied through `apply()` stay `ScalarField`-only.

use std::sync::Mutex;

use nalgebra::{DMatrix, DVector, Matrix2, Vector2};
use oxiflow::{
    boundary::{BoundaryCondition, BoundaryType},
    context::{
        compute::ComputeContext, error::OxiflowError, value::ContextValue,
        variable::ContextVariable, ContextCalculator,
    },
    mesh::{Mesh, UniformGrid1D},
    model::traits::{PhysicalModel, RequiresContext},
    solver::{
        config::{IntegratorKind, StepControl, TimeConfiguration},
        methods::{
            BDF2Solver, BackwardEulerSolver, CrankNicolsonSolver, DoPri45Solver,
            ForwardEulerSolver, OperatorSplittingSolver, RK4Solver, SteppableSolver,
        },
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

// ── θ-method: Backward Euler and Crank–Nicolson ──────────────────────────────

/// Tolerance of the implicit runs against their discrete solution: the
/// single Newton correction is exact for a linear model, up to the
/// finite-difference Jacobian's cancellation error.
const IMPLICIT_TOLERANCE: f64 = 1e-8;

#[test]
fn backward_euler_matches_its_discrete_solution_on_a_coupled_system() {
    // Backward Euler is exactly u_{k+1} = (I − dt A)⁻¹ u_k on a linear system.
    let (dt, n_steps) = (0.1, 10);
    let a = coupled_matrix();
    let result = BackwardEulerSolver::new()
        .solve(
            &scenario(a),
            &fixed_step(dt * n_steps as f64, dt, IntegratorKind::BackwardEuler),
        )
        .unwrap();
    let step = (Matrix2::identity() - a * dt).try_inverse().unwrap();
    let propagator = power(step, n_steps);
    let error = max_error(&result, |x| propagator * LinearSystem2::initial_row(x));
    assert!(error < IMPLICIT_TOLERANCE, "max error = {error:e}");
}

#[test]
fn crank_nicolson_matches_its_discrete_solution_on_a_coupled_system() {
    // Crank–Nicolson is u_{k+1} = (I − dt/2 A)⁻¹ (I + dt/2 A) u_k.
    let (dt, n_steps) = (0.1, 10);
    let a = coupled_matrix();
    let result = CrankNicolsonSolver::new()
        .solve(
            &scenario(a),
            &fixed_step(dt * n_steps as f64, dt, IntegratorKind::CrankNicolson),
        )
        .unwrap();
    let half = a * (dt / 2.0);
    let step = (Matrix2::identity() - half).try_inverse().unwrap() * (Matrix2::identity() + half);
    let propagator = power(step, n_steps);
    let error = max_error(&result, |x| propagator * LinearSystem2::initial_row(x));
    assert!(error < IMPLICIT_TOLERANCE, "max error = {error:e}");
}

// ── BDF2 ──────────────────────────────────────────────────────────────────────

#[test]
fn bdf2_matches_its_discrete_solution_on_a_coupled_system() {
    // Startup: one Backward Euler step, u₁ = (I − dt A)⁻¹ u₀; then
    // u_{k+1} = (3/2 I − dt A)⁻¹ (2 u_k − 1/2 u_{k−1}).
    let (dt, n_steps) = (0.1, 10);
    let a = coupled_matrix();
    let result = BDF2Solver::new()
        .solve(
            &scenario(a),
            &fixed_step(dt * n_steps as f64, dt, IntegratorKind::BDF2),
        )
        .unwrap();
    let startup = (Matrix2::identity() - a * dt).try_inverse().unwrap();
    let implicit = (Matrix2::identity() * 1.5 - a * dt).try_inverse().unwrap();
    let error = max_error(&result, |x| {
        let mut previous = LinearSystem2::initial_row(x);
        let mut current = startup * previous;
        for _ in 1..n_steps {
            let next = implicit * (current * 2.0 - previous * 0.5);
            previous = current;
            current = next;
        }
        current
    });
    assert!(error < IMPLICIT_TOLERANCE, "max error = {error:e}");
}

// ── Operator splitting (Strang) ───────────────────────────────────────────────

/// `A = A₁ + A₂` with non-commuting parts, so the splitting is not exact.
fn split_matrices() -> (Matrix2<f64>, Matrix2<f64>) {
    (
        Matrix2::new(-1.0, 1.0, 0.0, -0.5),
        Matrix2::new(-1.0, 0.0, 1.0, -1.5),
    )
}

fn split_domain(id: &str, matrix: Matrix2<f64>) -> Domain {
    Domain::new(id, Box::new(LinearSystem2::new(matrix)), Box::new(mesh()))
}

#[test]
fn strang_splitting_matches_its_discrete_solution_on_a_coupled_system() {
    // Explicit half-steps on A₁ around an implicit full step on A₂:
    // one outer step is (I + dt/2 A₁) (I − dt A₂)⁻¹ (I + dt/2 A₁).
    let (dt, n_steps) = (0.05, 20);
    let (a1, a2) = split_matrices();
    let solver = OperatorSplittingSolver::strang(
        split_domain("explicit", a1),
        Box::new(ForwardEulerSolver),
        split_domain("implicit", a2),
        Box::new(BackwardEulerSolver::new()),
    )
    .unwrap();
    let result = solver
        .solve(
            &scenario(a1 + a2),
            &fixed_step(dt * n_steps as f64, dt, IntegratorKind::Imex),
        )
        .unwrap();
    let half = Matrix2::identity() + a1 * (dt / 2.0);
    let full = (Matrix2::identity() - a2 * dt).try_inverse().unwrap();
    let propagator = power(half * full * half, n_steps);
    let error = max_error(&result, |x| propagator * LinearSystem2::initial_row(x));
    assert!(error < IMPLICIT_TOLERANCE, "max error = {error:e}");
}

// ── Boundary conditions ───────────────────────────────────────────────────────

/// Neumann condition that leaves the state as is.
#[derive(Debug)]
struct NoOpCondition;

impl RequiresContext for NoOpCondition {
    fn required_variables(&self) -> Vec<ContextVariable> {
        vec![]
    }
}

impl BoundaryCondition for NoOpCondition {
    fn boundary_type(&self) -> BoundaryType {
        BoundaryType::Neumann
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

#[test]
fn boundary_condition_apply_refuses_a_vector_field_state() {
    // `apply()` takes a single field: a VectorField state with conditions
    // attached is refused, never silently half-applied. Its boundaries go
    // through `ghost_value()` at the operator level instead.
    let model = LinearSystem2::new(coupled_matrix());
    let mut state = model.initial_state(&mesh());
    let domain = Domain::new("with_bc", Box::new(model), Box::new(mesh()))
        .with_boundary_conditions(vec![Box::new(NoOpCondition)]);
    let result = ForwardEulerSolver.step(&domain, &[], &mut state, &[], 0.0, 0.1);
    assert!(
        matches!(result, Err(OxiflowError::TypeMismatch { .. })),
        "expected TypeMismatch, got {result:?}"
    );
}

// ── Banded Jacobian on a VectorField (feature `sparse`) ──────────────────────

/// Two-component reaction–diffusion: `∂u_c/∂t = D_c ∂²u_c/∂x² + (A u)_c`,
/// second-order centred differences (stencil half-width 1) with zero-flux
/// ends. Couples neighbouring nodes *and* the two components of a node, so
/// its Jacobian in node-interleaved order has half-bandwidth
/// `d·(s + 1) − 1 = 3`.
#[cfg(feature = "sparse")]
struct ReactionDiffusion2 {
    diffusion: [f64; 2],
    reaction: Matrix2<f64>,
    step: f64,
}

#[cfg(feature = "sparse")]
impl RequiresContext for ReactionDiffusion2 {
    fn required_variables(&self) -> Vec<ContextVariable> {
        vec![]
    }
}

#[cfg(feature = "sparse")]
impl PhysicalModel for ReactionDiffusion2 {
    fn compute_physics(
        &self,
        state: &ContextValue,
        _ctx: &ComputeContext,
    ) -> Result<ContextValue, OxiflowError> {
        let u = state.as_vector_field()?;
        let n = u.nrows();
        let mut du = DMatrix::zeros(n, 2);
        for i in 0..n {
            // Zero flux: mirror the neighbour across each end.
            let left = if i == 0 { 1 } else { i - 1 };
            let right = if i == n - 1 { n - 2 } else { i + 1 };
            for c in 0..2 {
                let laplacian =
                    (u[(left, c)] - 2.0 * u[(i, c)] + u[(right, c)]) / (self.step * self.step);
                let reaction =
                    self.reaction[(c, 0)] * u[(i, 0)] + self.reaction[(c, 1)] * u[(i, 1)];
                du[(i, c)] = self.diffusion[c] * laplacian + reaction;
            }
        }
        Ok(ContextValue::VectorField(du))
    }

    fn initial_state(&self, mesh: &dyn Mesh) -> ContextValue {
        let n = mesh.n_dof();
        let mut u = DMatrix::zeros(n, 2);
        for i in 0..n {
            let x = mesh.coordinates(i)[0];
            u[(i, 0)] = (std::f64::consts::PI * x).cos();
            u[(i, 1)] = x * x;
        }
        ContextValue::VectorField(u)
    }

    fn name(&self) -> &str {
        "reaction_diffusion_2"
    }
}

/// Final state of 5 Backward Euler steps on [`ReactionDiffusion2`], 30 nodes
/// (60 unknowns), with the sparse path forced when `bandwidth` is given.
#[cfg(feature = "sparse")]
fn reaction_diffusion_backward_euler(
    bandwidth: Option<usize>,
) -> Result<DMatrix<f64>, OxiflowError> {
    use oxiflow::solver::sparse::FaerSparseSolver;

    let n_nodes = 30;
    let mesh = UniformGrid1D::new(n_nodes, 0.0, 1.0).unwrap();
    let model = ReactionDiffusion2 {
        diffusion: [1e-2, 5e-3],
        reaction: Matrix2::new(-1.0, 0.5, 0.3, -2.0),
        step: mesh.characteristic_length(),
    };
    let scenario = Scenario::single(Box::new(model), Box::new(mesh));
    let mut solver = BackwardEulerSolver::new();
    if let Some(k) = bandwidth {
        solver = solver
            .with_jacobian_bandwidth(k)
            .with_sparse_solver(Box::new(FaerSparseSolver))
            .with_sparse_threshold(0);
    }
    let result = solver.solve(
        &scenario,
        &fixed_step(0.05, 0.01, IntegratorKind::BackwardEuler),
    )?;
    Ok(result
        .states
        .last()
        .unwrap()
        .as_vector_field()
        .unwrap()
        .clone())
}

#[cfg(feature = "sparse")]
#[test]
fn banded_path_with_bandwidth_d_times_s_plus_one_minus_one_matches_dense_path() {
    let dense = reaction_diffusion_backward_euler(None).unwrap();
    let banded = reaction_diffusion_backward_euler(Some(3)).unwrap();
    let gap = (&dense - &banded).amax();
    assert!(gap < 1e-10, "dense vs banded gap = {gap:e}");
}

#[cfg(feature = "sparse")]
#[test]
fn banded_path_with_the_scalar_stencil_width_misses_the_component_coupling() {
    // k = s = 1 is the ScalarField rule: on 2 interleaved components it
    // drops entries of the Jacobian. The model is affine, so the single
    // default correction is exact with the full Jacobian; with the
    // truncated one it is not, and the Newton loop reports it instead of
    // returning a state that drifts from the dense path. This is why the
    // bandwidth is counted in flat unknowns.
    let err = reaction_diffusion_backward_euler(Some(1)).unwrap_err();
    assert!(
        matches!(err, OxiflowError::NewtonNotConverged { iterations: 1, .. }),
        "too-narrow band: got {err:?}"
    );
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
