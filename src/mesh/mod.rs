//! # Module `mesh`
//!
//! Spatial mesh abstraction — INV-1 invariant (DD-007, DD-019, issue #31).
//!
//! ## Core principle (INV-1)
//!
//! No public engine API exposes `dx`, `nx` or raw indices as first-class spatial
//! parameters. All spatial references go through the `Mesh` trait. This guarantees
//! forward compatibility with FEM unstructured meshes at J7 — zero breaking change
//! on existing code when `unstructured/` is added.
//!
//! ## Module hierarchy (DD-019)
//!
//! ```text
//! src/mesh/
//! ├── mod.rs              — Mesh trait + public re-exports
//! ├── structured/
//! │   └── mod.rs          — UniformGrid1D (J1); UniformGrid2D (J4b+)
//! └── unstructured/       — RESERVED J7 (UnstructuredMesh2D, TetrahedralMesh3D)
//! ```
//!
//! ## Implementations
//!
//! | Type | Family | Description | Milestone |
//! |---|---|---|---|
//! | [`UniformGrid1D`] | structured | 1D uniform grid, FD/FV | J1 — v0.1.0 |
//! | `UnstructuredMesh2D` | unstructured | 2D triangular mesh, FEM | J7 — v2.0.0 |
//! | `TetrahedralMesh3D` | unstructured | 3D tetrahedral mesh, FEM | J7 — v2.0.0 |

pub mod structured;

// unstructured/ — RESERVED J7
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
/// Required for INV-4 (plugin-safe API, J7).
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
