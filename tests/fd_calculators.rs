//! # Integration test: meshes supported by the FD spatial calculators (#186)
//!
//! `FDGradientCalculator` and `FDLaplacianCalculator` hold an `Arc<dyn Mesh>`
//! but their stencils are only correct on a 1D mesh with equally spaced nodes.
//! These tests check, from outside the crate, that both calculators refuse
//! every other mesh with `PreconditionFailed` instead of returning a wrong
//! field, and that a mesh implemented downstream with uniform nodes is still
//! accepted.

mod common;

use std::sync::Arc;

use common::meshes::{ScatteredMesh2D, StretchedGrid1D};
use nalgebra::DVector;
use oxiflow::{
    context::{
        calculator::ContextCalculator,
        calculators::{FDGradientCalculator, FDLaplacianCalculator, FDScheme},
        compute::ComputeContext,
        error::OxiflowError,
        value::ContextValue,
        variable::ContextVariable,
    },
    mesh::{Mesh, UniformGrid1D},
};

// ── Helpers ───────────────────────────────────────────────────────────────────

/// `u(x) = x²` sampled on the first coordinate of every node.
fn squared_first_coordinate(mesh: &dyn Mesh) -> ContextValue {
    ContextValue::ScalarField(DVector::from_iterator(
        mesh.n_dof(),
        (0..mesh.n_dof()).map(|i| mesh.coordinates(i)[0].powi(2)),
    ))
}

fn ctx() -> ComputeContext {
    ComputeContext::new(0.0, 1e-3)
}

fn laplacian_variable() -> ContextVariable {
    ContextVariable::External {
        name: "laplacian".into(),
    }
}

fn gradient(mesh: Arc<dyn Mesh>, dimension: usize) -> Result<ContextValue, OxiflowError> {
    let state = squared_first_coordinate(&*mesh);
    FDGradientCalculator::new(mesh, dimension, None, FDScheme::Central).compute(&state, &ctx())
}

fn laplacian(mesh: Arc<dyn Mesh>) -> Result<ContextValue, OxiflowError> {
    let state = squared_first_coordinate(&*mesh);
    FDLaplacianCalculator::new(mesh, laplacian_variable()).compute(&state, &ctx())
}

fn assert_refused(result: Result<ContextValue, OxiflowError>, what: &str) {
    assert!(
        matches!(result, Err(OxiflowError::PreconditionFailed { .. })),
        "{what}: expected PreconditionFailed, got {result:?}"
    );
}

/// Uniform 1D mesh implemented downstream: same nodes as `UniformGrid1D`,
/// written with nothing but the public `Mesh` trait.
#[derive(Debug)]
struct DownstreamUniform1D {
    nodes: Vec<[f64; 1]>,
    step: f64,
}

impl DownstreamUniform1D {
    fn new(n_points: usize) -> Self {
        let step = 1.0 / (n_points - 1) as f64;
        let nodes = (0..n_points).map(|i| [i as f64 * step]).collect();
        Self { nodes, step }
    }
}

impl Mesh for DownstreamUniform1D {
    fn n_dof(&self) -> usize {
        self.nodes.len()
    }
    fn coordinates(&self, i: usize) -> &[f64] {
        &self.nodes[i]
    }
    fn spatial_dimension(&self) -> usize {
        1
    }
    fn characteristic_length(&self) -> f64 {
        self.step
    }
}

// ── Refused meshes ────────────────────────────────────────────────────────────

#[test]
fn gradient_refuses_non_uniform_1d_mesh() {
    assert_refused(
        gradient(Arc::new(StretchedGrid1D::new(41, 1.1)), 0),
        "gradient on a stretched 1D mesh",
    );
}

#[test]
fn laplacian_refuses_non_uniform_1d_mesh() {
    // The audit (#172) measured 2.4 to 125 here for an exact value of 2.
    assert_refused(
        laplacian(Arc::new(StretchedGrid1D::new(41, 1.1))),
        "laplacian on a stretched 1D mesh",
    );
}

#[test]
fn gradient_refuses_2d_mesh_in_every_direction() {
    for dimension in 0..2 {
        assert_refused(
            gradient(Arc::new(ScatteredMesh2D::new(6)), dimension),
            &format!("gradient along dimension {dimension} on a 2D mesh"),
        );
    }
}

#[test]
fn laplacian_refuses_2d_mesh() {
    assert_refused(
        laplacian(Arc::new(ScatteredMesh2D::new(6))),
        "laplacian on a 2D mesh",
    );
}

#[test]
fn gradient_refuses_dimension_beyond_a_1d_mesh() {
    assert_refused(
        gradient(Arc::new(UniformGrid1D::new(11, 0.0, 1.0).unwrap()), 1),
        "gradient along dimension 1 on a 1D mesh",
    );
}

// ── Accepted meshes ───────────────────────────────────────────────────────────

#[test]
fn downstream_uniform_1d_mesh_gives_the_same_fields_as_uniform_grid() {
    let downstream: Arc<dyn Mesh> = Arc::new(DownstreamUniform1D::new(11));
    let reference: Arc<dyn Mesh> = Arc::new(UniformGrid1D::new(11, 0.0, 1.0).unwrap());

    let grad = gradient(downstream.clone(), 0).unwrap();
    let grad_ref = gradient(reference.clone(), 0).unwrap();
    let lap = laplacian(downstream).unwrap();
    let lap_ref = laplacian(reference).unwrap();

    let close = |a: &DVector<f64>, b: &DVector<f64>| (a - b).amax() < 1e-12;
    assert!(close(
        grad.as_scalar_field().unwrap(),
        grad_ref.as_scalar_field().unwrap()
    ));
    assert!(close(
        lap.as_scalar_field().unwrap(),
        lap_ref.as_scalar_field().unwrap()
    ));
    // u = x² → ∇²u = 2, exact at interior nodes on a uniform grid.
    let lap = lap.as_scalar_field().unwrap();
    assert!((1..lap.len() - 1).all(|i| (lap[i] - 2.0).abs() < 1e-9));
}
