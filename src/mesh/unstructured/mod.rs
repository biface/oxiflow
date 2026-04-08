//! # Module `mesh::unstructured`
//!
//! Unstructured mesh implementations — FEM triangular and tetrahedral meshes.
//!
//! **RESERVED — J7 (v2.0.0)**
//!
//! This module is intentionally empty at J1. Implementations will be added
//! at J7 when FEM support lands (DD-007, DD-019):
//!
//! - `UnstructuredMesh2D` — 2D triangular mesh (Gmsh/Triangle reader)
//! - `TetrahedralMesh3D` — 3D tetrahedral mesh
//!
//! Adding these types here at J7 requires zero changes to `mesh::structured`
//! or any existing `Mesh` implementor — that is the point of DD-019.
