//! # Module `mesh::spec`
//!
//! External, serialisable description of a mesh — the mesh part of the
//! WHAT-axis configuration ([DT-027](https://github.com/biface/oxiflow/issues/75)
//! amendment 1, [#108](https://github.com/biface/oxiflow/issues/108)).
//!
//! [`MeshSpec`] names a mesh type and its construction parameters. It is
//! consumed by [`DomainSpec`](crate::solver::DomainSpec), which builds the
//! mesh once and shares it between the domain and the components that need
//! it (a flux scheme). Its fields are the parameters of the constructor, not
//! the fields of the mesh: `x_start` and `x_end` are inputs of
//! [`UniformGrid1D::new`](super::UniformGrid1D::new), which keeps the node
//! spacing instead.
//!
//! The set of variants is closed and known at compile time; further meshes
//! (the unstructured and nD meshes of the v0.11 milestones) are added as new
//! variants ([DD-072](https://github.com/biface/oxiflow/issues/215)).

/// Which mesh to build, and with what parameters.
///
/// With the `serde` feature, a mesh is written as its variant name around
/// its parameters:
///
/// ```text
/// mesh:
///   UniformGrid1D: { n_points: 200, x_start: 0.0, x_end: 0.25 }
/// ```
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Deserialize))]
pub enum MeshSpec {
    /// A [`UniformGrid1D`](super::UniformGrid1D) of `n_points` nodes from
    /// `x_start` to `x_end`; see [`UniformGrid1D::new`](super::UniformGrid1D::new)
    /// for the constraints on the parameters.
    UniformGrid1D {
        /// Number of nodes, at least 2.
        n_points: usize,
        /// Coordinate of the first node.
        x_start: f64,
        /// Coordinate of the last node, greater than `x_start`.
        x_end: f64,
    },
}

#[cfg(all(test, feature = "serde"))]
mod tests {
    use super::*;

    #[test]
    fn a_uniform_grid_reads_from_its_parameters() {
        let json = r#"{ "UniformGrid1D": { "n_points": 11, "x_start": 0.0, "x_end": 2.5 } }"#;
        let spec: MeshSpec = serde_json::from_str(json).unwrap();
        assert_eq!(
            spec,
            MeshSpec::UniformGrid1D {
                n_points: 11,
                x_start: 0.0,
                x_end: 2.5
            }
        );
    }

    #[test]
    fn an_unknown_mesh_is_refused() {
        let json = r#"{ "Hexahedral": { "n_points": 11 } }"#;
        assert!(serde_json::from_str::<MeshSpec>(json).is_err());
    }
}
