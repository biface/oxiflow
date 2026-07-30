//! # Module `solver`
//!
//! Numerical solving orchestration — WHAT/HOW separation (issue #32).
//!
//! ## Responsibilities
//!
//! | Type | Role |
//! |---|---|
//! | [`scenario::Scenario`] | Declares the problem (WHAT) |
//! | [`config::SolverConfiguration`] | Configures solving (HOW) |
//! | [`Solver`] | Orchestrates execution |
//!
//! ## Contractual execution order
//!
//! `Solver::solve()` implementations must follow this order at each time step:
//!
//! 1. **Calculators** — populate `ComputeContext` in topological order
//! 2. **Boundary conditions** — apply to state using `ctx` (J2)
//! 3. **`compute_physics`** — compute `du/dt` from state + context
//! 4. **Integrate** — advance state by `dt`
//!
//! This order is a contract, not a convention. Deviating from it produces
//! silently incorrect results.

pub mod chain;
pub mod config;
pub mod linear;
pub mod methods;
pub mod orchestrator;
pub mod scenario;
pub mod snapshot;
#[cfg(feature = "sparse")]
pub mod sparse;
#[cfg(feature = "sparse")]
pub mod spec;

pub use config::{IntegratorKind, SolverConfiguration, StepControl, TimeConfiguration};
pub use scenario::{Domain, DomainId, Scenario};
pub use snapshot::SimulationSnapshot;
#[cfg(feature = "sparse")]
pub use spec::IntegratorSpec;

use crate::context::error::OxiflowError;

// ── SimulationResult ──────────────────────────────────────────────────────────

/// Result of a completed simulation.
///
/// `states` and `times` have the same length. The save frequency is controlled
/// by `SolverConfiguration::time.save_every`.
///
/// # Examples
///
/// ```rust, ignore
/// use oxiflow::solver::SimulationResult;
/// use oxiflow::context::value::ContextValue;
/// use nalgebra::DVector;
///
/// let result = SimulationResult {
///     states: vec![ContextValue::ScalarField(DVector::from_element(10, 0.0))],
///     times:  vec![0.0],
///     n_steps: 1,
/// };
/// assert_eq!(result.states.len(), result.times.len());
/// ```
#[non_exhaustive]
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct SimulationResult {
    /// Saved field states at each recorded time.
    pub states: Vec<crate::context::value::ContextValue>,
    /// Simulation times corresponding to each saved state.
    pub times: Vec<f64>,
    /// Total number of time steps taken (may be larger than `states.len()`
    /// if `save_every > 1`).
    pub n_steps: usize,
    /// Solver metadata: timing, rejected steps, convergence info.
    ///
    /// Keys follow the convention `"solver.<key>"` (e.g. `"solver.rejected_steps"`).
    /// Empty at J1 — populated by adaptive integrators at J4 (DoPri45, BDF2).
    pub metadata: std::collections::HashMap<String, f64>,
}

impl SimulationResult {
    /// Returns the number of saved states.
    pub fn len(&self) -> usize {
        self.states.len()
    }

    /// Returns `true` if no states were saved.
    pub fn is_empty(&self) -> bool {
        self.states.is_empty()
    }

    /// Returns the final simulation time.
    pub fn t_final(&self) -> Option<f64> {
        self.times.last().copied()
    }

    /// Writes the **final** saved state to an XML VTK unstructured grid
    /// (`.vtu`), for visualisation in ParaView/VisIt (DD-027, issue #78).
    ///
    /// Node coordinates come from `mesh`; nodes are connected by `Line`
    /// cells in `Mesh` node order (`n_dof() - 1` segments), matching the
    /// only mesh currently implemented (`UniformGrid1D`). Only `mesh.
    /// spatial_dimension() == 1` is supported for this first increment —
    /// coordinates are padded with zeros to the 3 components VTK requires.
    ///
    /// Exports the last entry of `states` only — not the full time series.
    /// A `.pvd` time-series collection is future work, not part of this
    /// increment (see #78's acceptance criteria, scoped to a single file).
    ///
    /// Requires the `vtk` feature.
    ///
    /// # Errors
    ///
    /// Returns [`OxiflowError::Persistence`] if: `states` is empty; `mesh`
    /// has `spatial_dimension() != 1`; the last state is not a
    /// [`crate::context::value::ContextValue::ScalarField`]; its length
    /// doesn't match `mesh.n_dof()`; or the underlying `vtkio` export
    /// fails.
    #[cfg(feature = "vtk")]
    pub fn write_vtk(
        &self,
        mesh: &dyn crate::mesh::Mesh,
        path: &std::path::Path,
    ) -> Result<(), OxiflowError> {
        use crate::context::value::ContextValue;
        use vtkio::model::{
            Attribute, Attributes, ByteOrder, CellType, Cells, DataSet, IOBuffer,
            UnstructuredGridPiece, Version, VertexNumbers, Vtk,
        };

        if mesh.spatial_dimension() != 1 {
            return Err(OxiflowError::Persistence(format!(
                "write_vtk currently only supports 1D meshes (Line cells), \
                 got spatial_dimension() = {}",
                mesh.spatial_dimension()
            )));
        }

        let n = mesh.n_dof();
        if n == 0 {
            return Err(OxiflowError::Persistence(
                "write_vtk: mesh has zero degrees of freedom".to_string(),
            ));
        }

        let last_state = self.states.last().ok_or_else(|| {
            OxiflowError::Persistence("write_vtk: SimulationResult has no saved states".to_string())
        })?;
        let values: Vec<f64> = match last_state {
            ContextValue::ScalarField(v) => v.iter().copied().collect(),
            other => {
                return Err(OxiflowError::Persistence(format!(
                    "write_vtk currently only supports ScalarField states, got {other:?}"
                )));
            }
        };
        if values.len() != n {
            return Err(OxiflowError::Persistence(format!(
                "write_vtk: state has {} values, mesh has {n} degrees of freedom",
                values.len()
            )));
        }

        // Node coordinates, padded to 3 components (VTK always stores 3D points).
        let mut points = Vec::with_capacity(n * 3);
        for i in 0..n {
            let c = mesh.coordinates(i);
            points.push(c[0]);
            points.push(0.0);
            points.push(0.0);
        }

        // Line cells connecting consecutive nodes; a single Vertex if n == 1.
        let (connectivity, offsets, types) = if n >= 2 {
            let mut connectivity = Vec::with_capacity((n - 1) * 2);
            let mut offsets = Vec::with_capacity(n - 1);
            for i in 0..n - 1 {
                connectivity.push(i as u64);
                connectivity.push((i + 1) as u64);
                offsets.push(((i + 1) * 2) as u64);
            }
            (connectivity, offsets, vec![CellType::Line; n - 1])
        } else {
            (vec![0u64], vec![1u64], vec![CellType::Vertex])
        };

        let mut attributes = Attributes::new();
        attributes
            .point
            .push(Attribute::scalars("field", 1).with_data(IOBuffer::F64(values)));

        let vtk = Vtk {
            version: Version::new((2, 0)),
            title: String::from("oxiflow SimulationResult"),
            byte_order: ByteOrder::BigEndian,
            file_path: Some(path.to_path_buf()),
            data: DataSet::inline(UnstructuredGridPiece {
                points: IOBuffer::F64(points),
                cells: Cells {
                    cell_verts: VertexNumbers::XML {
                        connectivity,
                        offsets,
                    },
                    types,
                },
                data: attributes,
            }),
        };

        vtk.export(path)
            .map_err(|e| OxiflowError::Persistence(format!("VTK export failed: {e}")))
    }
}

// ── Solver trait ──────────────────────────────────────────────────────────────

/// Orchestrates the time integration loop.
///
/// Implementations receive a `Scenario` (WHAT) and a `SolverConfiguration`
/// (HOW) and execute the contractual loop until `t_end`.
///
/// `Solver` implementations must:
/// - Verify `scenario.n_domains() == 1`
/// - Build the calculator chain via `chain::build_calculator_chain()`
/// - Follow the contractual execution order
///
/// `Solver` is deliberately kept single-domain (DD-031) — coupled
/// multi-domain scenarios go through [`orchestrator::MultiDomainOrchestrator`]
/// instead, which drives one [`methods::SteppableSolver`] per domain and
/// invokes `CouplingOperator`s between steps. This keeps each domain free
/// to use a different integrator, and leaves `Solver`/`SimulationResult`
/// unchanged for the well-tested single-domain path.
///
/// # Object safety
///
/// This trait is object-safe to support INV-4 (plugin-safe API, v2.0).
pub trait Solver: Send + Sync {
    /// Runs the simulation and returns the collected states.
    fn solve(
        &self,
        scenario: &Scenario,
        config: &SolverConfiguration,
    ) -> Result<SimulationResult, OxiflowError>;

    /// Called automatically right before a divergence-class error is
    /// returned, with a [`SimulationSnapshot`] capturing the state at the
    /// point of failure (DD-025 Option B, issue #71). Two distinct
    /// failure modes are routed through this hook:
    ///
    /// - [`OxiflowError::SolverDivergence`] — a non-finite value appeared
    ///   in the field itself (checked after every step by
    ///   [`methods`]'s internal finiteness check).
    /// - [`OxiflowError::NewtonNotConverged`] — the iterated Newton
    ///   solver ([`methods::newton`]) exhausted its iteration budget
    ///   without satisfying its convergence criterion (DD-044, v0.7.0).
    ///   This is a failure of the *nonlinear correction* to converge, not
    ///   necessarily a non-finite state — the last iterate may still be
    ///   finite, just not accepted as converged.
    ///
    /// Both cases reuse the same [`SimulationSnapshot`] checkpoint
    /// infrastructure; implementations that override this hook to
    /// persist or log should not assume the snapshot's `error` field
    /// always describes a non-finite value.
    ///
    /// Default implementation is a no-op — override to persist (e.g. via
    /// [`snapshot::write_snapshot`]) or log. There is no `checkpoint()`
    /// counterpart on this trait; see the [`snapshot`] module docs for why
    /// explicit checkpoints are constructed directly by caller code instead.
    fn on_divergence(&self, snapshot: &snapshot::SimulationSnapshot) {
        let _ = snapshot;
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::context::value::ContextValue;
    use nalgebra::DVector;

    #[test]
    fn simulation_result_len() {
        let r = SimulationResult {
            states: vec![ContextValue::ScalarField(DVector::from_element(5, 0.0))],
            times: vec![1.0],
            n_steps: 10,
            metadata: std::collections::HashMap::new(),
        };
        assert_eq!(r.len(), 1);
        assert!(!r.is_empty());
    }

    #[test]
    fn simulation_result_t_final() {
        let r = SimulationResult {
            states: vec![
                ContextValue::ScalarField(DVector::from_element(5, 0.0)),
                ContextValue::ScalarField(DVector::from_element(5, 1.0)),
            ],
            times: vec![0.0, 1.0],
            n_steps: 2,
            metadata: std::collections::HashMap::new(),
        };
        assert_eq!(r.t_final(), Some(1.0));
    }

    #[test]
    fn empty_result() {
        let r = SimulationResult {
            states: vec![],
            times: vec![],
            n_steps: 0,
            metadata: std::collections::HashMap::new(),
        };
        assert!(r.is_empty());
        assert_eq!(r.t_final(), None);
    }

    #[test]
    fn solver_is_object_safe() {
        fn assert_object_safe<T: Solver + ?Sized>() {}
        assert_object_safe::<dyn Solver>();
    }

    // ── write_vtk ─────────────────────────────────────────────────────────────

    #[cfg(feature = "vtk")]
    #[test]
    fn write_vtk_round_trip() {
        use crate::mesh::structured::UniformGrid1D;

        let mesh = UniformGrid1D::new(5, 0.0, 4.0).unwrap();
        let result = SimulationResult {
            states: vec![ContextValue::ScalarField(DVector::from_vec(vec![
                0.0, 1.0, 2.0, 1.0, 0.0,
            ]))],
            times: vec![0.4],
            n_steps: 4,
            metadata: std::collections::HashMap::new(),
        };

        let path = std::env::temp_dir().join("oxiflow_test_write_vtk_round_trip.vtu");
        result.write_vtk(&mesh, &path).unwrap();

        // Round-trip: re-parse the file and check the point-data values match.
        let reimported = vtkio::model::Vtk::import(&path).unwrap();
        let vtkio::model::DataSet::UnstructuredGrid { pieces, .. } = reimported.data else {
            panic!("expected an UnstructuredGrid dataset");
        };
        let piece = pieces[0]
            .load_piece_data(None)
            .expect("failed to load piece data");
        assert_eq!(piece.points.len(), 5 * 3); // n_dof * 3 components
        let field = piece
            .data
            .point
            .iter()
            .find(|a| a.name() == "field")
            .expect("missing 'field' point attribute");
        let vtkio::model::Attribute::DataArray(da) = field else {
            panic!("expected a DataArray attribute");
        };
        let values: Vec<f64> = da.data.clone().cast_into().unwrap();
        assert_eq!(values, vec![0.0, 1.0, 2.0, 1.0, 0.0]);

        std::fs::remove_file(&path).ok();
    }

    #[cfg(feature = "vtk")]
    #[test]
    fn write_vtk_rejects_2d_mesh() {
        struct FakeMesh2D;
        impl crate::mesh::Mesh for FakeMesh2D {
            fn n_dof(&self) -> usize {
                1
            }
            fn coordinates(&self, _i: usize) -> &[f64] {
                &[0.0, 0.0]
            }
            fn spatial_dimension(&self) -> usize {
                2
            }
            fn characteristic_length(&self) -> f64 {
                1.0
            }
        }

        let result = SimulationResult {
            states: vec![ContextValue::ScalarField(DVector::from_element(1, 0.0))],
            times: vec![0.0],
            n_steps: 0,
            metadata: std::collections::HashMap::new(),
        };
        let path = std::env::temp_dir().join("oxiflow_test_write_vtk_rejects_2d.vtu");
        let result = result.write_vtk(&FakeMesh2D, &path);
        assert!(matches!(result, Err(OxiflowError::Persistence(_))));
    }

    #[cfg(feature = "vtk")]
    #[test]
    fn write_vtk_rejects_empty_states() {
        use crate::mesh::structured::UniformGrid1D;

        let mesh = UniformGrid1D::new(3, 0.0, 1.0).unwrap();
        let result = SimulationResult {
            states: vec![],
            times: vec![],
            n_steps: 0,
            metadata: std::collections::HashMap::new(),
        };
        let path = std::env::temp_dir().join("oxiflow_test_write_vtk_rejects_empty.vtu");
        let outcome = result.write_vtk(&mesh, &path);
        assert!(matches!(outcome, Err(OxiflowError::Persistence(_))));
    }
}
