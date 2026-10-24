//! # Integration test: meshes supported by the Danckwerts conditions (#187)
//!
//! `DanckwertsInlet` and `DanckwertsOutlet` locate the boundary by DOF index
//! (node `0`, node `n − 1`), which an ordered 1D grid alone makes meaningful.
//! These tests check, from outside the crate, that both conditions refuse a
//! 2D mesh with `PreconditionFailed` and still apply on a 1D mesh whose nodes
//! are ordered but not equally spaced.

mod common;

use common::meshes::{ScatteredMesh2D, StretchedGrid1D};
use nalgebra::DVector;
use oxiflow::{
    boundary::{
        danckwerts::{DanckwertsInlet, DanckwertsOutlet},
        BoundaryCondition,
    },
    context::{
        compute::ComputeContext, error::OxiflowError, value::ContextValue,
        variable::ContextVariable,
    },
    mesh::Mesh,
};

// ── Helpers ───────────────────────────────────────────────────────────────────

const DISPERSION: f64 = 1e-3;
const VELOCITY: f64 = 1e-2;
const FEED: f64 = 2.0;

fn feed_variable() -> ContextVariable {
    ContextVariable::External {
        name: "feed_concentration".into(),
    }
}

fn inlet() -> DanckwertsInlet {
    DanckwertsInlet::new(DISPERSION, VELOCITY, feed_variable())
}

/// Context holding everything `DanckwertsInlet` reads, so that a refusal can
/// only come from the mesh.
fn full_context() -> ComputeContext {
    let mut ctx = ComputeContext::new(0.0, 1e-3);
    ctx.insert(feed_variable(), ContextValue::Scalar(FEED));
    ctx
}

/// `u_i = i + 1`, so every node holds a distinct value.
fn ramp(mesh: &dyn Mesh) -> DVector<f64> {
    DVector::from_iterator(mesh.n_dof(), (0..mesh.n_dof()).map(|i| (i + 1) as f64))
}

fn assert_refused(result: Result<(), OxiflowError>, what: &str) {
    assert!(
        matches!(result, Err(OxiflowError::PreconditionFailed { .. })),
        "{what}: expected PreconditionFailed, got {result:?}"
    );
}

// ── Refused meshes ────────────────────────────────────────────────────────────

#[test]
fn inlet_refuses_2d_mesh() {
    let mesh = ScatteredMesh2D::new(5);
    let mut state = ramp(&mesh);
    let before = state.clone();
    let result = inlet().apply(&mut state, &full_context(), &mesh);
    assert_refused(result, "DanckwertsInlet on a 2D mesh");
    assert_eq!(
        state, before,
        "a refused condition must leave the state as is"
    );
}

#[test]
fn outlet_refuses_2d_mesh() {
    let mesh = ScatteredMesh2D::new(5);
    let mut state = ramp(&mesh);
    let before = state.clone();
    let result = DanckwertsOutlet::new().apply(&mut state, &full_context(), &mesh);
    assert_refused(result, "DanckwertsOutlet on a 2D mesh");
    assert_eq!(
        state, before,
        "a refused condition must leave the state as is"
    );
}

// ── Accepted meshes ───────────────────────────────────────────────────────────

#[test]
fn both_conditions_apply_on_ordered_non_uniform_1d_mesh() {
    // The inlet reads the first spacing, x1 - x0, and nothing else of the
    // geometry: an ordered 1D mesh is enough, uniform or not.
    let mesh = StretchedGrid1D::new(9, 1.2);
    let n = mesh.n_dof();
    let ctx = full_context();
    let mut state = ramp(&mesh);

    inlet().apply(&mut state, &ctx, &mesh).unwrap();
    DanckwertsOutlet::new()
        .apply(&mut state, &ctx, &mesh)
        .unwrap();

    let kappa = DISPERSION / (VELOCITY * (mesh.coordinates(1)[0] - mesh.coordinates(0)[0]));
    let expected_inlet = (FEED + kappa * 2.0) / (1.0 + kappa);
    assert!((state[0] - expected_inlet).abs() < 1e-12);
    assert_eq!(state[n - 1], state[n - 2]);
    assert!((1..n - 1).all(|i| state[i] == (i + 1) as f64));
}
