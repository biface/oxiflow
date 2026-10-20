//! # Module `mesh`
//!
//! Spatial mesh abstraction — INV-1 invariant (DD-007, DD-019, issue #31).
//!
//! ## Core principle (INV-1)
//!
//! No public engine API exposes `dx`, `nx` or raw indices as first-class spatial
//! parameters. All spatial references go through the `Mesh` trait. This guarantees
//! forward compatibility with FEM unstructured meshes (v0.11.0 (J11)) — zero breaking change
//! on existing code when `unstructured/` is added.
//!
//! ## Module hierarchy (DD-019)
//!
//! ```text
//! src/mesh/
//! ├── mod.rs              — Mesh trait + public re-exports
//! ├── structured/
//! │   └── mod.rs          — UniformGrid1D (J1); nD generalisation at v0.11.5 (DD-051)
//! └── unstructured/       — RESERVED v0.11.0 (J11) (UnstructuredMesh2D, TetrahedralMesh3D)
//! ```
//!
//! ## Implementations
//!
//! | Type | Family | Description | Milestone |
//! |---|---|---|---|
//! | [`UniformGrid1D`] | structured | 1D uniform grid, FD/FV | J1 — v0.1.0 |
//! | `UnstructuredMesh2D` | unstructured | 2D triangular mesh, FEM | J11 — v0.11.0 |
//! | `TetrahedralMesh3D` | unstructured | 3D tetrahedral mesh, FEM | J11 — v0.11.0 |

pub mod structured;

// unstructured/ — RESERVED v0.11.0 (J11)
// pub mod unstructured;

// ── Public re-exports ─────────────────────────────────────────────────────────

pub use structured::UniformGrid1D;

// ── Mesh trait ────────────────────────────────────────────────────────────────

/// Abstract spatial mesh — INV-1 invariant.
///
/// All spatial information consumed by the engine passes through this trait.
/// No implementation detail (`dx`, `nx`, node table) leaks into the public API.
///
/// # Object safety
///
/// This trait is object-safe: it can be used as `Box<dyn Mesh>` and `&dyn Mesh`.
/// Required for INV-4 (plugin-safe API, v0.30.0 / J30, DA-017).
///
/// # Implementing `Mesh`
///
/// A minimal implementation for a 1D uniform grid:
///
/// ```rust
/// use oxiflow::mesh::Mesh;
///
/// struct MyGrid { n: usize, dx: f64 }
///
/// impl Mesh for MyGrid {
///     fn n_dof(&self) -> usize { self.n }
///     fn coordinates(&self, i: usize) -> &[f64] {
///         // In practice, return a slice into a pre-computed node table.
///         // This example uses a workaround for illustration only.
///         std::slice::from_ref(Box::leak(Box::new(i as f64 * self.dx)))
///     }
///     fn spatial_dimension(&self) -> usize { 1 }
///     fn characteristic_length(&self) -> f64 { self.dx }
/// }
/// ```
///
/// # INV-1 compliance
///
/// Implementations must **not** expose `dx`, `nx` or raw indices in any public
/// method beyond those defined here. Spatial parameters are internal details.
pub trait Mesh: Send + Sync {
    /// Total number of degrees of freedom (nodes) in the mesh.
    ///
    /// For a 1D uniform grid: `n_dof() == n_points`.
    /// For a 2D FEM mesh: `n_dof()` is the number of mesh nodes.
    fn n_dof(&self) -> usize;

    /// Coordinates of node `i` as a slice of length `spatial_dimension()`.
    ///
    /// Returns `&[f64]` — zero allocation. Callers must not store the returned
    /// reference beyond the lifetime of the mesh. Implementors typically return
    /// a slice into a pre-computed node coordinate table.
    ///
    /// # Panics
    ///
    /// Implementations may panic if `i >= n_dof()`. Use `n_dof()` to guard
    /// iteration bounds.
    fn coordinates(&self, i: usize) -> &[f64];

    /// Number of spatial dimensions of this mesh.
    ///
    /// - `1` for 1D grids (chromatography column, 1D heat transfer)
    /// - `2` for 2D meshes (surface flow, 2D diffusion)
    /// - `3` for 3D meshes (volumetric FEM)
    fn spatial_dimension(&self) -> usize;

    /// A representative spatial length scale of the mesh.
    ///
    /// Used by integrators and operators for stability estimates (CFL condition,
    /// Péclet number). For a uniform 1D grid: `dx`. For unstructured meshes:
    /// the minimum element diameter or average edge length.
    fn characteristic_length(&self) -> f64;
}

// ── Uniform spacing ───────────────────────────────────────────────────────────

/// Relative tolerance of [`uniform_spacing`] on each node gap.
///
/// `UniformGrid1D` builds its nodes as `x_start + i·dx`, so a gap differs from
/// `dx` by about `ε·|x| / dx` (ε the `f64` machine epsilon): below `1e-6` as
/// long as `|x| / dx < 1e9`. A spacing defect of `1e-6` relative is also far
/// below the truncation error of the stencils that rely on this check.
pub(crate) const SPACING_TOLERANCE: f64 = 1e-6;

/// Node spacing of a 1D mesh whose nodes are equally spaced in index order.
///
/// Returns `Some(characteristic_length())` when the mesh is one-dimensional,
/// has at least two nodes, and every gap `x[i+1] − x[i]` lies within
/// [`SPACING_TOLERANCE`] (relative) of `characteristic_length()`. Returns
/// `None` otherwise, including when `characteristic_length()` is not a
/// positive finite number.
///
/// The gaps are compared with `characteristic_length()`, not with their mean:
/// callers use the returned value as the stencil step, so a mesh with uniform
/// nodes but a different declared length must be rejected as well. A positive
/// reference also rules out nodes in decreasing order.
///
/// Cost is `O(n_dof)`; callers on a hot path resolve it once and store the
/// result, which is sound as long as a mesh does not change after
/// construction.
pub(crate) fn uniform_spacing(mesh: &dyn Mesh) -> Option<f64> {
    let n = mesh.n_dof();
    let step = mesh.characteristic_length();
    if mesh.spatial_dimension() != 1 || n < 2 || !step.is_finite() || step <= 0.0 {
        return None;
    }
    let bound = SPACING_TOLERANCE * step;
    let uniform = (0..n - 1).all(|i| {
        let gap = mesh.coordinates(i + 1)[0] - mesh.coordinates(i)[0];
        (gap - step).abs() <= bound
    });
    uniform.then_some(step)
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// 1D mesh built from an explicit node list and a declared length.
    struct NodeList {
        nodes: Vec<[f64; 1]>,
        length: f64,
    }

    impl NodeList {
        fn new(xs: &[f64], length: f64) -> Self {
            Self {
                nodes: xs.iter().map(|&x| [x]).collect(),
                length,
            }
        }
    }

    impl Mesh for NodeList {
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
            self.length
        }
    }

    /// Two nodes in 2D, uniform in x if projected.
    struct TwoNodes2D([[f64; 2]; 2]);

    impl Mesh for TwoNodes2D {
        fn n_dof(&self) -> usize {
            2
        }
        fn coordinates(&self, i: usize) -> &[f64] {
            &self.0[i]
        }
        fn spatial_dimension(&self) -> usize {
            2
        }
        fn characteristic_length(&self) -> f64 {
            1.0
        }
    }

    #[test]
    fn uniform_grid_returns_its_step() {
        let grid = UniformGrid1D::new(11, -3.0, 2.0).unwrap();
        assert_eq!(uniform_spacing(&grid), Some(grid.characteristic_length()));
    }

    #[test]
    fn uniform_grid_far_from_origin_is_accepted() {
        // |x| / dx = 1e7: rounding of x_start + i·dx stays within tolerance.
        let grid = UniformGrid1D::new(1001, 1.0e4, 1.0e4 + 1.0).unwrap();
        assert!(uniform_spacing(&grid).is_some());
    }

    #[test]
    fn non_uniform_nodes_are_rejected() {
        let mesh = NodeList::new(&[0.0, 0.25, 0.5, 1.0], 0.25);
        assert_eq!(uniform_spacing(&mesh), None);
    }

    #[test]
    fn length_inconsistent_with_nodes_is_rejected() {
        let mesh = NodeList::new(&[0.0, 0.25, 0.5, 0.75], 0.5);
        assert_eq!(uniform_spacing(&mesh), None);
    }

    #[test]
    fn decreasing_nodes_are_rejected() {
        let mesh = NodeList::new(&[0.75, 0.5, 0.25, 0.0], 0.25);
        assert_eq!(uniform_spacing(&mesh), None);
    }

    #[test]
    fn degenerate_length_is_rejected() {
        for length in [0.0, -0.25, f64::NAN, f64::INFINITY] {
            let mesh = NodeList::new(&[0.0, 0.25], length);
            assert_eq!(uniform_spacing(&mesh), None, "length = {length}");
        }
    }

    #[test]
    fn single_node_is_rejected() {
        let mesh = NodeList::new(&[0.0], 1.0);
        assert_eq!(uniform_spacing(&mesh), None);
    }

    #[test]
    fn non_1d_mesh_is_rejected() {
        let mesh = TwoNodes2D([[0.0, 0.0], [1.0, 0.0]]);
        assert_eq!(uniform_spacing(&mesh), None);
    }
}
