//! # Test meshes implemented outside the crate
//!
//! Meshes written the way a downstream crate (or the J11 unstructured
//! meshes) would write them, with nothing but the public `Mesh` trait. Shared
//! by the integration tests that check how a component behaves on a mesh
//! other than `UniformGrid1D`.

use oxiflow::mesh::Mesh;

/// Non-uniform 1D mesh on `[0, 1]` with geometric spacing: each cell is
/// `ratio` times wider than the previous one.
#[derive(Debug)]
pub struct StretchedGrid1D {
    nodes: Vec<[f64; 1]>,
    h_min: f64,
}

impl StretchedGrid1D {
    pub fn new(n_points: usize, ratio: f64) -> Self {
        assert!(n_points >= 2 && ratio > 1.0);
        let n_cells = (n_points - 1) as i32;
        // First cell width so that the geometric series of widths sums to 1.
        let h0 = (ratio - 1.0) / (ratio.powi(n_cells) - 1.0);
        let mut x = 0.0;
        let mut nodes = Vec::with_capacity(n_points);
        nodes.push([0.0]);
        for k in 0..n_cells {
            x += h0 * ratio.powi(k);
            nodes.push([x]);
        }
        // Pin the last node to the exact right end (removes rounding drift).
        nodes[n_points - 1] = [1.0];
        Self { nodes, h_min: h0 }
    }
}

impl Mesh for StretchedGrid1D {
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
        self.h_min
    }
}

/// 2D node cloud on a jittered `side × side` lattice of the unit square — no
/// connectivity, which is all a pointwise model needs.
#[derive(Debug)]
pub struct ScatteredMesh2D {
    nodes: Vec<[f64; 2]>,
    h: f64,
}

impl ScatteredMesh2D {
    pub fn new(side: usize) -> Self {
        assert!(side >= 2);
        let h = 1.0 / (side - 1) as f64;
        let mut nodes = Vec::with_capacity(side * side);
        for j in 0..side {
            for i in 0..side {
                // Deterministic jitter, a quarter of the spacing at most.
                let jx = 0.25 * h * ((i * 7 + j * 3) % 5) as f64 / 4.0;
                let jy = 0.25 * h * ((i * 2 + j * 5) % 5) as f64 / 4.0;
                nodes.push([i as f64 * h + jx, j as f64 * h + jy]);
            }
        }
        Self { nodes, h }
    }
}

impl Mesh for ScatteredMesh2D {
    fn n_dof(&self) -> usize {
        self.nodes.len()
    }
    fn coordinates(&self, i: usize) -> &[f64] {
        &self.nodes[i]
    }
    fn spatial_dimension(&self) -> usize {
        2
    }
    fn characteristic_length(&self) -> f64 {
        self.h
    }
}
