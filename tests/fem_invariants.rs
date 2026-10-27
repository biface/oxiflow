//! # Integration test: FEM invariants INV-1 and INV-3 (#172)
//!
//! Regression gate for the parts of the FEM invariants that hold today, written
//! from outside the crate on purpose: every `Mesh` here is implemented outside
//! it (`tests/common/meshes.rs`), the way a downstream crate (or the J11
//! unstructured meshes) would do it, without access to the engine's internals.
//!
//! - **INV-1 (abstract `Mesh`)** — the solver core (`Scenario`, the context
//!   pipeline, `ForwardEulerSolver`) runs unchanged on a non-uniform 1D mesh
//!   and on a 2D mesh; a coordinate-based calculator (`TrapezoidalIntegral`) is
//!   exact on a non-uniform 1D mesh and refuses, explicitly, a mesh dimension it
//!   does not support.
//! - **INV-3 (`CouplingOperator`)** — a single-domain solver carries no
//!   inter-domain logic: it refuses a multi-domain scenario instead of running
//!   it.
//!
//! Components that accept any `Mesh` but are only correct on a uniform 1D grid
//! (findings of the #172 audit) are guarded in the tests of the component
//! itself: `tests/fd_calculators.rs` and `tests/danckwerts.rs`.

mod common;

use std::sync::Arc;

use common::meshes::{ScatteredMesh2D, StretchedGrid1D};
use nalgebra::DVector;
use oxiflow::{
    context::{
        calculator::ContextCalculator, calculators::TrapezoidalIntegral, compute::ComputeContext,
        error::OxiflowError, value::ContextValue, variable::ContextVariable,
    },
    mesh::{Mesh, UniformGrid1D},
    model::traits::{PhysicalModel, RequiresContext},
    solver::{
        config::{IntegratorKind, StepControl, TimeConfiguration},
        methods::ForwardEulerSolver,
        scenario::{Domain, DomainId, Scenario},
        Solver, SolverConfiguration,
    },
};

// ── Pointwise model ───────────────────────────────────────────────────────────

/// `du/dt = -k(x) u` with `k(x) = 1 + Σ x_d` and `u(x, 0) = 1 + Σ x_d`.
///
/// Exact solution `u(x, t) = u(x, 0) exp(-k(x) t)`. The model never reads a
/// spacing or a neighbour: every spatial datum comes from
/// `Mesh::coordinates()`, which is what INV-1 promises any mesh can provide.
/// The rates are computed at construction from the same mesh, because
/// `compute_physics` has no mesh argument (coordinate-dependent coefficients
/// are the subject of DD-066, #182).
struct CoordinateDecay {
    rates: DVector<f64>,
}

fn coordinate_sum(mesh: &dyn Mesh, i: usize) -> f64 {
    mesh.coordinates(i).iter().sum()
}

impl CoordinateDecay {
    fn new(mesh: &dyn Mesh) -> Self {
        let rates = DVector::from_iterator(
            mesh.n_dof(),
            (0..mesh.n_dof()).map(|i| 1.0 + coordinate_sum(mesh, i)),
        );
        Self { rates }
    }

    fn exact(&self, mesh: &dyn Mesh, i: usize, t: f64) -> f64 {
        (1.0 + coordinate_sum(mesh, i)) * (-self.rates[i] * t).exp()
    }
}

impl RequiresContext for CoordinateDecay {
    fn required_variables(&self) -> Vec<ContextVariable> {
        vec![]
    }
}

impl PhysicalModel for CoordinateDecay {
    fn compute_physics(
        &self,
        state: &ContextValue,
        _ctx: &ComputeContext,
    ) -> Result<ContextValue, OxiflowError> {
        let u = state.as_scalar_field()?;
        Ok(ContextValue::ScalarField(-self.rates.component_mul(u)))
    }

    fn initial_state(&self, mesh: &dyn Mesh) -> ContextValue {
        ContextValue::ScalarField(DVector::from_iterator(
            mesh.n_dof(),
            (0..mesh.n_dof()).map(|i| 1.0 + coordinate_sum(mesh, i)),
        ))
    }

    fn name(&self) -> &str {
        "coordinate_decay"
    }
}

// ── Helpers ───────────────────────────────────────────────────────────────────

const T_END: f64 = 0.5;
const DT: f64 = 1e-4;
// Forward Euler, local rate k ≤ 3, dt = 1e-4: global error O(k² dt t) ≈ 2e-4
// in relative terms — 1e-3 leaves a comfortable margin.
const RELATIVE_TOLERANCE: f64 = 1e-3;

fn euler_config() -> SolverConfiguration {
    SolverConfiguration::new(
        TimeConfiguration::new(T_END, StepControl::Fixed { dt: DT }),
        IntegratorKind::Euler,
    )
}

/// Runs `CoordinateDecay` on `mesh` and returns the worst relative error
/// against the exact solution at `T_END`.
fn max_relative_error(mesh: Box<dyn Mesh>, reference: &dyn Mesh) -> f64 {
    let model = CoordinateDecay::new(reference);
    let checker = CoordinateDecay::new(reference);
    let scenario = Scenario::single(Box::new(model), mesh);
    let result = ForwardEulerSolver
        .solve(&scenario, &euler_config())
        .expect("solver core must run on any Mesh implementation (INV-1)");
    let last = result.states.last().unwrap().as_scalar_field().unwrap();
    assert_eq!(last.len(), reference.n_dof());
    (0..reference.n_dof())
        .map(|i| {
            let exact = checker.exact(reference, i, T_END);
            ((last[i] - exact) / exact).abs()
        })
        .fold(0.0, f64::max)
}

fn integral_variable() -> ContextVariable {
    ContextVariable::External {
        name: "integral".into(),
    }
}

// ── INV-1 ─────────────────────────────────────────────────────────────────────

#[test]
fn inv1_solver_core_runs_on_non_uniform_1d_mesh() {
    let reference = StretchedGrid1D::new(41, 1.1);
    let error = max_relative_error(Box::new(StretchedGrid1D::new(41, 1.1)), &reference);
    assert!(
        error < RELATIVE_TOLERANCE,
        "max relative error = {error:e}, expected < {RELATIVE_TOLERANCE:e}"
    );
}

#[test]
fn inv1_solver_core_runs_on_2d_mesh() {
    let reference = ScatteredMesh2D::new(6);
    let error = max_relative_error(Box::new(ScatteredMesh2D::new(6)), &reference);
    assert!(
        error < RELATIVE_TOLERANCE,
        "max relative error = {error:e}, expected < {RELATIVE_TOLERANCE:e}"
    );
}

#[test]
fn inv1_coordinate_based_integral_is_exact_on_non_uniform_1d_mesh() {
    // The trapezoidal rule is exact for a linear integrand on any node
    // distribution: ∫₀¹ (2x + 1) dx = 2.
    let mesh = Arc::new(StretchedGrid1D::new(17, 1.3));
    let u = DVector::from_iterator(
        mesh.n_dof(),
        (0..mesh.n_dof()).map(|i| 2.0 * mesh.coordinates(i)[0] + 1.0),
    );
    let calc = TrapezoidalIntegral::new(mesh, integral_variable());
    let value = calc
        .compute(&ContextValue::ScalarField(u), &ComputeContext::new(0.0, DT))
        .unwrap()
        .as_scalar()
        .unwrap();
    assert!(
        (value - 2.0).abs() < 1e-12,
        "integral = {value}, expected 2"
    );
}

#[test]
fn inv1_dimension_limited_calculator_refuses_unsupported_mesh() {
    // A component that only supports some meshes must say so explicitly
    // rather than return a silently wrong value — the behaviour the audit's
    // fix issues align the other 1D-only components on.
    let mesh = Arc::new(ScatteredMesh2D::new(4));
    let u = DVector::from_element(mesh.n_dof(), 1.0);
    let calc = TrapezoidalIntegral::new(mesh, integral_variable());
    let result = calc.compute(&ContextValue::ScalarField(u), &ComputeContext::new(0.0, DT));
    assert!(
        matches!(result, Err(OxiflowError::PreconditionFailed { .. })),
        "expected PreconditionFailed on a 2D mesh, got {result:?}"
    );
}

// ── INV-3 ─────────────────────────────────────────────────────────────────────

#[test]
fn inv3_single_domain_solver_refuses_multi_domain_scenario() {
    let make_domain = |id: &str| {
        let mesh = UniformGrid1D::new(5, 0.0, 1.0).unwrap();
        let model = CoordinateDecay::new(&mesh);
        Domain::new(DomainId::new(id), Box::new(model), Box::new(mesh))
    };
    let scenario = Scenario::multi(vec![make_domain("a"), make_domain("b")]).unwrap();
    let result = ForwardEulerSolver.solve(&scenario, &euler_config());
    assert!(
        matches!(result, Err(OxiflowError::InvalidDomain(_))),
        "a single-domain solver must not run a multi-domain scenario"
    );
}
