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
//!
//! ## Boundary regions (DA-071)
//!
//! The mesh, not the boundary conditions, knows where its boundary is.
//! [`BoundaryRegion`] names a region (today a [`BoundaryRegion::Face`] of a
//! structured grid), [`Mesh::boundary_nodes`] lists its nodes and
//! [`Mesh::boundary_neighbours`] pairs each with its first interior
//! neighbour. Both methods have a default for 1D meshes whose nodes are
//! ordered along the axis.

pub mod structured;

// unstructured/ — RESERVED v0.11.0 (J11)
// pub mod unstructured;

// ── Public re-exports ─────────────────────────────────────────────────────────

pub use structured::UniformGrid1D;

use crate::context::error::OxiflowError;

// ── Boundary regions (DA-071) ─────────────────────────────────────────────────

/// A region of the boundary of a mesh (DA-071, #203).
///
/// Boundary conditions designate the nodes they act on through a region,
/// which the mesh resolves with [`Mesh::boundary_nodes`] and
/// [`Mesh::boundary_neighbours`], rather than through node indices (INV-1).
///
/// `#[non_exhaustive]`: the boundaries of unstructured meshes, identified by
/// node tags (Gmsh physical groups, DD-050), will be a further variant.
///
/// # Examples
///
/// ```rust
/// use oxiflow::mesh::{BoundaryRegion, FaceSide, Mesh, UniformGrid1D};
///
/// let grid = UniformGrid1D::new(5, 0.0, 1.0)?;
/// let outlet = BoundaryRegion::Face { axis: 0, side: FaceSide::Upper };
/// assert_eq!(grid.boundary_nodes(&outlet)?, vec![4]);
///
/// let neighbours = grid.boundary_neighbours(&outlet)?;
/// assert_eq!((neighbours[0].node, neighbours[0].inner), (4, 3));
/// assert_eq!(neighbours[0].distance, 0.25);
/// # Ok::<(), oxiflow::context::OxiflowError>(())
/// ```
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum BoundaryRegion {
    /// Face of a structured grid normal to `axis` (0 for x, 1 for y, 2 for
    /// z), on the given `side`. On a 1D grid, the two ends.
    Face {
        /// Axis the face is normal to.
        axis: usize,
        /// Which of the two faces normal to `axis`.
        side: FaceSide,
    },
}

/// Which of the two faces normal to an axis (DA-071).
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum FaceSide {
    /// The face where the coordinate along the axis is smallest.
    Lower,
    /// The face where the coordinate along the axis is largest.
    Upper,
}

/// A node of a boundary region with its first neighbour along the inward
/// normal, as returned by [`Mesh::boundary_neighbours`] (DA-071).
///
/// `#[non_exhaustive]` so that further fields (a normal, a weight) can be
/// added; a mesh implemented outside the crate builds it with
/// [`BoundaryNode::new`].
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BoundaryNode {
    /// Index of the boundary node.
    pub node: usize,
    /// Index of its first neighbour along the inward normal.
    pub inner: usize,
    /// Distance between the two nodes, positive.
    pub distance: f64,
}

impl BoundaryNode {
    /// A boundary node, its inner neighbour and their distance.
    pub fn new(node: usize, inner: usize, distance: f64) -> Self {
        Self {
            node,
            inner,
            distance,
        }
    }
}

/// Checks that `region` is a face the default boundary methods resolve: a
/// face normal to axis 0 of a 1D mesh with at least `min_nodes` nodes.
/// Returns the node count and the face side.
fn default_face(
    mesh: &(impl Mesh + ?Sized),
    region: &BoundaryRegion,
    min_nodes: usize,
    context: &'static str,
) -> Result<(usize, FaceSide), OxiflowError> {
    let refuse = |message: String| OxiflowError::PreconditionFailed { context, message };
    let BoundaryRegion::Face { axis, side } = region;
    let dimension = mesh.spatial_dimension();
    if dimension != 1 || *axis != 0 {
        return Err(refuse(format!(
            "{region:?} is not defined by this mesh: the default implementation covers \
             axis 0 of 1D meshes only, got spatial_dimension = {dimension}"
        )));
    }
    let n = mesh.n_dof();
    if n < min_nodes {
        return Err(refuse(format!(
            "{region:?} needs at least {min_nodes} node(s), the mesh has {n}"
        )));
    }
    Ok((n, *side))
}

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

    /// Nodes of the boundary `region` (DA-071, #203).
    ///
    /// The default implementation covers the two faces of a 1D mesh, normal
    /// to axis 0, assuming nodes ordered along the axis: `Lower` is node `0`,
    /// `Upper` node `n_dof() − 1`. Meshes of higher dimension, or whose
    /// nodes are not in that order, override it.
    ///
    /// # Errors
    ///
    /// `OxiflowError::PreconditionFailed` for a region the mesh does not
    /// define (with the default: another axis, a mesh that is not 1D) or a
    /// mesh without nodes.
    fn boundary_nodes(&self, region: &BoundaryRegion) -> Result<Vec<usize>, OxiflowError> {
        let (n, side) = default_face(self, region, 1, "Mesh::boundary_nodes")?;
        Ok(match side {
            FaceSide::Lower => vec![0],
            FaceSide::Upper => vec![n - 1],
        })
    }

    /// Nodes of the boundary `region`, each with its first neighbour along
    /// the inward normal and their distance (DA-071, #203) — what a
    /// condition written on a structured grid needs, such as a one-sided
    /// difference across the boundary.
    ///
    /// The default implementation covers the two faces of a 1D mesh, normal
    /// to axis 0, assuming nodes ordered along the axis: `Lower` is node `0`
    /// with neighbour `1`, `Upper` node `n_dof() − 1` with neighbour
    /// `n_dof() − 2`. Only the order of that boundary pair is checked (its
    /// distance must be positive), not the order of the whole mesh.
    ///
    /// Unstructured meshes have no such neighbour: they keep a refusal here,
    /// a condition on them being imposed weakly instead.
    ///
    /// # Errors
    ///
    /// `OxiflowError::PreconditionFailed` for a region the mesh does not
    /// define, a mesh with fewer than two nodes, or a boundary pair whose
    /// distance is not positive (nodes not ordered along the axis).
    fn boundary_neighbours(
        &self,
        region: &BoundaryRegion,
    ) -> Result<Vec<BoundaryNode>, OxiflowError> {
        let context = "Mesh::boundary_neighbours";
        let (n, side) = default_face(self, region, 2, context)?;
        let (node, inner) = match side {
            FaceSide::Lower => (0, 1),
            FaceSide::Upper => (n - 1, n - 2),
        };
        let distance = (self.coordinates(node)[0] - self.coordinates(inner)[0]).abs();
        let ordered = match side {
            FaceSide::Lower => self.coordinates(inner)[0] > self.coordinates(node)[0],
            FaceSide::Upper => self.coordinates(node)[0] > self.coordinates(inner)[0],
        };
        if !ordered {
            return Err(OxiflowError::PreconditionFailed {
                context,
                message: format!(
                    "{region:?}: nodes {node} and {inner} are not ordered along axis 0 \
                     (x = {} and {})",
                    self.coordinates(node)[0],
                    self.coordinates(inner)[0]
                ),
            });
        }
        Ok(vec![BoundaryNode::new(node, inner, distance)])
    }
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

// ── Test mesh ─────────────────────────────────────────────────────────────────

/// Mesh built from explicit node coordinates, for unit tests in any
/// dimension `D`; `characteristic_length()` returns the declared `length`.
#[cfg(test)]
pub(crate) struct NodeCloud {
    dimension: usize,
    n_nodes: usize,
    coordinates: Vec<f64>,
    length: f64,
}

#[cfg(test)]
impl NodeCloud {
    /// One node per entry of `nodes`, in index order.
    pub(crate) fn new<const D: usize>(nodes: &[[f64; D]], length: f64) -> Self {
        Self {
            dimension: D,
            n_nodes: nodes.len(),
            coordinates: nodes.concat(),
            length,
        }
    }
}

#[cfg(test)]
impl Mesh for NodeCloud {
    fn n_dof(&self) -> usize {
        self.n_nodes
    }
    fn coordinates(&self, i: usize) -> &[f64] {
        &self.coordinates[i * self.dimension..(i + 1) * self.dimension]
    }
    fn spatial_dimension(&self) -> usize {
        self.dimension
    }
    fn characteristic_length(&self) -> f64 {
        self.length
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── Boundary regions (DA-071) ────────────────────────────────────────────

    const LOWER: BoundaryRegion = BoundaryRegion::Face {
        axis: 0,
        side: FaceSide::Lower,
    };
    const UPPER: BoundaryRegion = BoundaryRegion::Face {
        axis: 0,
        side: FaceSide::Upper,
    };

    #[test]
    fn faces_of_a_uniform_grid() {
        let grid = UniformGrid1D::new(5, 0.0, 1.0).unwrap();
        assert_eq!(grid.boundary_nodes(&LOWER).unwrap(), vec![0]);
        assert_eq!(grid.boundary_nodes(&UPPER).unwrap(), vec![4]);
        assert_eq!(
            grid.boundary_neighbours(&LOWER).unwrap(),
            vec![BoundaryNode::new(0, 1, 0.25)]
        );
        assert_eq!(
            grid.boundary_neighbours(&UPPER).unwrap(),
            vec![BoundaryNode::new(4, 3, 0.25)]
        );
    }

    #[test]
    fn faces_of_a_non_uniform_mesh_use_the_boundary_spacing() {
        let mesh = NodeCloud::new(&[[0.0], [0.1], [0.4], [1.0]], 0.1);
        assert_eq!(
            mesh.boundary_neighbours(&LOWER).unwrap(),
            vec![BoundaryNode::new(0, 1, 0.1)]
        );
        assert_eq!(
            mesh.boundary_neighbours(&UPPER).unwrap(),
            vec![BoundaryNode::new(3, 2, 0.6)]
        );
    }

    #[test]
    fn a_single_node_is_both_faces_but_has_no_neighbour() {
        let mesh = NodeCloud::new(&[[0.5]], 1.0);
        assert_eq!(mesh.boundary_nodes(&LOWER).unwrap(), vec![0]);
        assert_eq!(mesh.boundary_nodes(&UPPER).unwrap(), vec![0]);
        assert!(matches!(
            mesh.boundary_neighbours(&LOWER),
            Err(OxiflowError::PreconditionFailed {
                context: "Mesh::boundary_neighbours",
                ..
            })
        ));
    }

    fn refused(result: Result<impl std::fmt::Debug, OxiflowError>, context: &str) -> bool {
        matches!(result, Err(OxiflowError::PreconditionFailed { context: c, .. }) if c == context)
    }

    #[test]
    fn the_default_refuses_what_it_does_not_cover() {
        let empty = NodeCloud::new::<1>(&[], 0.0);
        let plane = NodeCloud::new(&[[0.0, 0.0], [1.0, 0.0]], 1.0);
        let axis_1 = BoundaryRegion::Face {
            axis: 1,
            side: FaceSide::Lower,
        };
        let line = UniformGrid1D::new(3, 0.0, 1.0).unwrap();
        for (mesh, region) in [
            (&empty as &dyn Mesh, LOWER),
            (&plane, LOWER),
            (&line, axis_1),
        ] {
            assert!(refused(
                mesh.boundary_nodes(&region),
                "Mesh::boundary_nodes"
            ));
            assert!(refused(
                mesh.boundary_neighbours(&region),
                "Mesh::boundary_neighbours"
            ));
        }
    }

    #[test]
    fn a_boundary_pair_out_of_order_is_refused() {
        let reversed = NodeCloud::new(&[[1.0], [0.5], [0.0]], 0.5);
        assert!(refused(
            reversed.boundary_neighbours(&LOWER),
            "Mesh::boundary_neighbours"
        ));
        assert!(refused(
            reversed.boundary_neighbours(&UPPER),
            "Mesh::boundary_neighbours"
        ));
        let repeated = NodeCloud::new(&[[0.0], [0.0], [1.0]], 0.5);
        assert!(refused(
            repeated.boundary_neighbours(&LOWER),
            "Mesh::boundary_neighbours"
        ));
    }

    /// Two faces per axis of a 2D lattice, numbered row by row: a mesh that
    /// overrides the default.
    struct Lattice2D;

    impl Mesh for Lattice2D {
        fn n_dof(&self) -> usize {
            4
        }
        fn coordinates(&self, i: usize) -> &[f64] {
            const NODES: [[f64; 2]; 4] = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]];
            &NODES[i]
        }
        fn spatial_dimension(&self) -> usize {
            2
        }
        fn characteristic_length(&self) -> f64 {
            1.0
        }
        fn boundary_nodes(&self, region: &BoundaryRegion) -> Result<Vec<usize>, OxiflowError> {
            let BoundaryRegion::Face { axis, side } = region;
            Ok(match (axis, side) {
                (0, FaceSide::Lower) => vec![0, 2],
                (0, FaceSide::Upper) => vec![1, 3],
                (_, FaceSide::Lower) => vec![0, 1],
                (_, FaceSide::Upper) => vec![2, 3],
            })
        }
    }

    #[test]
    fn an_override_replaces_the_default() {
        let face = BoundaryRegion::Face {
            axis: 1,
            side: FaceSide::Upper,
        };
        assert_eq!(Lattice2D.boundary_nodes(&face).unwrap(), vec![2, 3]);
        assert!(refused(
            Lattice2D.boundary_neighbours(&face),
            "Mesh::boundary_neighbours"
        ));
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
        let mesh = NodeCloud::new(&[[0.0], [0.25], [0.5], [1.0]], 0.25);
        assert_eq!(uniform_spacing(&mesh), None);
    }

    #[test]
    fn length_inconsistent_with_nodes_is_rejected() {
        let mesh = NodeCloud::new(&[[0.0], [0.25], [0.5], [0.75]], 0.5);
        assert_eq!(uniform_spacing(&mesh), None);
    }

    #[test]
    fn decreasing_nodes_are_rejected() {
        let mesh = NodeCloud::new(&[[0.75], [0.5], [0.25], [0.0]], 0.25);
        assert_eq!(uniform_spacing(&mesh), None);
    }

    #[test]
    fn degenerate_length_is_rejected() {
        for length in [0.0, -0.25, f64::NAN, f64::INFINITY] {
            let mesh = NodeCloud::new(&[[0.0], [0.25]], length);
            assert_eq!(uniform_spacing(&mesh), None, "length = {length}");
        }
    }

    #[test]
    fn single_node_is_rejected() {
        let mesh = NodeCloud::new(&[[0.0]], 1.0);
        assert_eq!(uniform_spacing(&mesh), None);
    }

    #[test]
    fn non_1d_mesh_is_rejected() {
        let mesh = NodeCloud::new(&[[0.0, 0.0], [1.0, 0.0]], 1.0);
        assert_eq!(uniform_spacing(&mesh), None);
    }
}
