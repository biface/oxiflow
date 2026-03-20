//! # Module `mesh`
//!
//! Spatial mesh abstraction — INV-1 invariant.
//!
//! ## Core principle (INV-1)
//!
//! No public engine API exposes `dx`, `nx` or raw indices as first-class spatial
//! parameters. All spatial references go through the `Mesh` trait, of which
//! `UniformGrid1D` will be the first implementation (J1, v0.2) and unstructured
//! FEM meshes the second (J7, v2.0).
//!
//! ## Planned implementations
//!
//! | Type | Description | Milestone |
//! |---|---|---|
//! | `UniformGrid1D` | 1D uniform grid | J1 — v0.2 |
//! | `UnstructuredMesh2D` | 2D triangular mesh | J7 — v2.0 |
//! | `TetrahedralMesh3D` | 3D tetrahedral mesh | J7 — v2.0 |
