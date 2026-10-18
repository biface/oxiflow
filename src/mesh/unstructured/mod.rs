//! # Module `mesh::unstructured`
//!
//! Unstructured mesh implementations — FEM triangular and tetrahedral meshes.
//!
//! **RESERVED — v0.11.0 (J11)** (DA-049 #150, DD-050 #151)
//!
//! This module is intentionally empty until the FEM mesh foundations land at
//! v0.11.0 (J11) (DD-007, DD-019; DA-049 #150, DD-050 #151):
//!
//! - `UnstructuredMesh2D` — 2D triangular mesh (Gmsh import at v0.12.0, DT-052 #153)
//! - `TetrahedralMesh3D` — 3D tetrahedral mesh
//!
//! Adding these types here requires zero changes to `mesh::structured`
//! or any existing `Mesh` implementor — that is the point of DD-019.
