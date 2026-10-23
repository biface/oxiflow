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
#[cfg(feature = "gpu")]
pub mod gpu;
pub mod linear;
pub mod methods;
pub mod orchestrator;
#[cfg(feature = "parallel")]
pub mod parallel;
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
/// ```rust
/// use oxiflow::mesh::UniformGrid1D;
/// use oxiflow::solver::methods::ForwardEulerSolver;
/// use oxiflow::solver::{IntegratorKind, Scenario, SimulationResult, Solver,
///                       SolverConfiguration, StepControl, TimeConfiguration};
/// # use nalgebra::DVector;
/// # use oxiflow::context::{ComputeContext, ContextValue, ContextVariable, OxiflowError};
/// # use oxiflow::mesh::Mesh;
/// # use oxiflow::model::{PhysicalModel, RequiresContext};
/// # struct Decay;
/// # impl RequiresContext for Decay {
/// #     fn required_variables(&self) -> Vec<ContextVariable> { vec![] }
/// # }
/// # impl PhysicalModel for Decay {
/// #     fn compute_physics(
/// #         &self,
/// #         state: &ContextValue,
/// #         _ctx: &ComputeContext,
/// #     ) -> Result<ContextValue, OxiflowError> {
/// #         Ok(ContextValue::ScalarField(-state.as_scalar_field()?.clone()))
/// #     }
/// #     fn initial_state(&self, mesh: &dyn Mesh) -> ContextValue {
/// #         ContextValue::ScalarField(DVector::from_element(mesh.n_dof(), 1.0))
/// #     }
/// #     fn name(&self) -> &str { "decay" }
/// # }
/// // `SimulationResult` is `#[non_exhaustive]`: it is obtained from a solver,
/// // not built by hand.
/// let scenario = Scenario::single(Box::new(Decay), Box::new(UniformGrid1D::new(10, 0.0, 1.0)?));
/// let config = SolverConfiguration::new(
///     TimeConfiguration::new(1.0, StepControl::Fixed { dt: 0.1 }),
///     IntegratorKind::Euler,
/// );
/// let result: SimulationResult = ForwardEulerSolver.solve(&scenario, &config)?;
/// assert_eq!(result.states.len(), result.times.len());
/// assert_eq!(result.times[0], 0.0);
/// # Ok::<(), OxiflowError>(())
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
    /// (`.vtu`), for visualisation in ParaView/VisIt (DT-027, issue #78).
    ///
    /// Node coordinates come from `mesh`; nodes are connected by `Line`
    /// cells in `Mesh` node order (`n_dof() - 1` segments), matching the
    /// only mesh currently implemented (`UniformGrid1D`). Only `mesh.
    /// spatial_dimension() == 1` is supported for this first increment —
    /// coordinates are padded with zeros to the 3 components VTK requires.
    /// 2D/3D meshes are planned for v0.11.0.
    ///
    /// The state is written as point data, one scalar `Float64` array per
    /// component:
    ///
    /// | Final state | Point-data arrays |
    /// |---|---|
    /// | `ScalarField` (n values) | `field` |
    /// | `VectorField` (n × d) | `field_0`, `field_1`, …, `field_{d-1}` (column `c` → `field_c`) |
    ///
    /// As a state, a `VectorField` holds d unknowns per node (several
    /// species, several physical fields), with d unrelated to the mesh
    /// dimension: a 1D two-species model has d = 2. One array per component
    /// gives each its own colour map and range in ParaView, where a single
    /// `NumberOfComponents = d` array would be shown as a geometric vector
    /// (magnitude by default). The first array is declared as the
    /// active scalars (`<PointData Scalars="…">`).
    ///
    /// The file is UTF-8 XML; data arrays use VTK's inline `binary` format
    /// (little-endian bytes, base64-encoded, `UInt64` size header), so every
    /// `f64` is stored bit for bit — NaN and infinities included, which lets
    /// a diverged state be inspected in ParaView. It is written by the crate
    /// itself, with no third-party VTK dependency (#191).
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
    /// has `spatial_dimension() != 1` or zero degrees of freedom; the last
    /// state is neither a [`crate::context::value::ContextValue::ScalarField`]
    /// nor a [`crate::context::value::ContextValue::VectorField`]; a
    /// `ScalarField` length or a `VectorField` row count doesn't match
    /// `mesh.n_dof()`; a `VectorField` has no column; or the file cannot be
    /// created or written.
    #[cfg(feature = "vtk")]
    pub fn write_vtk(
        &self,
        mesh: &dyn crate::mesh::Mesh,
        path: &std::path::Path,
    ) -> Result<(), OxiflowError> {
        use crate::context::value::ContextValue;
        use std::io::Write;

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
        // Un tableau `(nom, valeurs)` par composante de l'état.
        let fields: Vec<(String, &[f64])> = match last_state {
            ContextValue::ScalarField(v) => {
                if v.len() != n {
                    return Err(OxiflowError::Persistence(format!(
                        "write_vtk: state has {} values, mesh has {n} degrees of freedom",
                        v.len()
                    )));
                }
                vec![("field".to_string(), v.as_slice())]
            }
            ContextValue::VectorField(m) => {
                if m.nrows() != n {
                    return Err(OxiflowError::Persistence(format!(
                        "write_vtk: VectorField state has {} rows, mesh has {n} degrees of freedom",
                        m.nrows()
                    )));
                }
                if m.ncols() == 0 {
                    return Err(OxiflowError::Persistence(
                        "write_vtk: VectorField state has no component".to_string(),
                    ));
                }
                // `DMatrix` est stocké par colonnes (column-major) : la
                // composante `c` occupe la tranche contiguë `[c·n, (c+1)·n)`.
                m.as_slice()
                    .chunks_exact(n)
                    .enumerate()
                    .map(|(c, column)| (format!("field_{c}"), column))
                    .collect()
            }
            other => {
                return Err(OxiflowError::Persistence(format!(
                    "write_vtk only supports ScalarField and VectorField states, got {other:?}"
                )));
            }
        };
        let x: Vec<f64> = (0..n).map(|i| mesh.coordinates(i)[0]).collect();

        let io_err =
            |e: std::io::Error| OxiflowError::Persistence(format!("VTK export failed: {e}"));
        let file = std::fs::File::create(path).map_err(io_err)?;
        let mut w = std::io::BufWriter::new(file);
        write_vtu_1d(&mut w, &x, &fields)
            .and_then(|()| w.flush())
            .map_err(io_err)
    }
}

/// Écrit une grille VTK XML `UnstructuredGrid` 1D, tableaux au format
/// `binary` en ligne — corps de [`SimulationResult::write_vtk`], qui a déjà
/// validé les entrées (`x.len() >= 1`, au moins un tableau, chacun de
/// longueur `x.len()`).
///
/// `fields` : tableaux scalaires de `PointData`, écrits dans l'ordre donné ;
/// le premier est déclaré comme scalaires actifs (`Scalars`).
///
/// Nœuds reliés par des cellules `Line` (type VTK 3) dans l'ordre de `x` ;
/// un seul `Vertex` (type 1) si `x.len() == 1`. Coordonnées complétées par
/// des zéros (VTK stocke toujours 3 composantes).
#[cfg(feature = "vtk")]
fn write_vtu_1d<W: std::io::Write, S: AsRef<str>>(
    w: &mut W,
    x: &[f64],
    fields: &[(S, &[f64])],
) -> std::io::Result<()> {
    let n = x.len();
    let (n_cells, nodes_per_cell, cell_type) = if n >= 2 { (n - 1, 2, 3u8) } else { (1, 1, 1u8) };

    let points: Vec<u8> = x
        .iter()
        .flat_map(|&xi| [xi, 0.0, 0.0])
        .flat_map(f64::to_le_bytes)
        .collect();
    let connectivity: Vec<u8> = if n >= 2 {
        (0..n as i64 - 1)
            .flat_map(|i| [i, i + 1])
            .flat_map(i64::to_le_bytes)
            .collect()
    } else {
        0i64.to_le_bytes().to_vec()
    };
    let offsets: Vec<u8> = (1..=n_cells as i64)
        .flat_map(|c| (c * nodes_per_cell).to_le_bytes())
        .collect();

    writeln!(w, r#"<?xml version="1.0" encoding="UTF-8"?>"#)?;
    writeln!(
        w,
        r#"<VTKFile type="UnstructuredGrid" version="1.0" byte_order="LittleEndian" header_type="UInt64">"#
    )?;
    writeln!(w, "  <UnstructuredGrid>")?;
    writeln!(
        w,
        r#"    <Piece NumberOfPoints="{n}" NumberOfCells="{n_cells}">"#
    )?;
    let active = fields.first().map_or("", |(name, _)| name.as_ref());
    writeln!(w, r#"      <PointData Scalars="{active}">"#)?;
    for (name, values) in fields {
        let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        write_data_array(w, "Float64", name.as_ref(), 1, &bytes)?;
    }
    writeln!(w, "      </PointData>")?;
    writeln!(w, "      <Points>")?;
    write_data_array(w, "Float64", "Points", 3, &points)?;
    writeln!(w, "      </Points>")?;
    writeln!(w, "      <Cells>")?;
    write_data_array(w, "Int64", "connectivity", 1, &connectivity)?;
    write_data_array(w, "Int64", "offsets", 1, &offsets)?;
    write_data_array(w, "UInt8", "types", 1, &vec![cell_type; n_cells])?;
    writeln!(w, "      </Cells>")?;
    writeln!(w, "    </Piece>")?;
    writeln!(w, "  </UnstructuredGrid>")?;
    writeln!(w, "</VTKFile>")
}

/// Écrit un `DataArray` au format `binary` en ligne : en-tête `UInt64`
/// (taille des données en octets, petit-boutiste) suivi des données, encodés
/// ensemble en base64 — la forme non compressée que VTK lit, et celle que
/// produisait `vtkio`.
#[cfg(feature = "vtk")]
fn write_data_array<W: std::io::Write>(
    w: &mut W,
    scalar_type: &str,
    name: &str,
    components: usize,
    bytes: &[u8],
) -> std::io::Result<()> {
    let mut payload = Vec::with_capacity(8 + bytes.len());
    payload.extend_from_slice(&(bytes.len() as u64).to_le_bytes());
    payload.extend_from_slice(bytes);
    writeln!(
        w,
        r#"        <DataArray type="{scalar_type}" Name="{name}" NumberOfComponents="{components}" format="binary">"#
    )?;
    writeln!(w, "          {}", base64_encode(&payload))?;
    writeln!(w, "        </DataArray>")
}

/// Encodage base64 standard (RFC 4648 §4, alphabet `A–Z a–z 0–9 + /`, avec
/// remplissage `=`), seul encodage requis par le format `binary` de VTK.
#[cfg(feature = "vtk")]
fn base64_encode(bytes: &[u8]) -> String {
    const ALPHABET: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut out = String::with_capacity(bytes.len().div_ceil(3) * 4);
    for chunk in bytes.chunks(3) {
        let b = [
            chunk[0],
            *chunk.get(1).unwrap_or(&0),
            *chunk.get(2).unwrap_or(&0),
        ];
        let triple = u32::from(b[0]) << 16 | u32::from(b[1]) << 8 | u32::from(b[2]);
        for k in 0..4 {
            if k <= chunk.len() {
                out.push(ALPHABET[(triple >> (18 - 6 * k) & 0x3f) as usize] as char);
            } else {
                out.push('=');
            }
        }
    }
    out
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
/// This trait is object-safe to support INV-4 (plugin-safe API, v0.30.0 / J30, DA-017).
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

        // Aller-retour : relecture par un parseur XML (quick-xml, dev-dependency),
        // décodage base64 indépendant et contrôle de chaque `DataArray` et des
        // attributs de `Piece`.
        let text = std::fs::read_to_string(&path).unwrap();
        let (piece, arrays) = parse_vtu(&text);
        assert_eq!(piece, (5, 4)); // NumberOfPoints, NumberOfCells

        assert_eq!(arrays["field"], vec![0.0, 1.0, 2.0, 1.0, 0.0]);
        let points = &arrays["Points"];
        assert_eq!(points.len(), 5 * 3); // n_dof * 3 components
        let x: Vec<f64> = points.chunks(3).map(|p| p[0]).collect();
        assert_eq!(x, vec![0.0, 1.0, 2.0, 3.0, 4.0]);
        assert!(points.chunks(3).all(|p| p[1] == 0.0 && p[2] == 0.0));
        assert_eq!(
            arrays["connectivity"],
            vec![0.0, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0]
        );
        assert_eq!(arrays["offsets"], vec![2.0, 4.0, 6.0, 8.0]);
        assert_eq!(arrays["types"], vec![3.0; 4]); // VTK_LINE

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

    /// Relit un `.vtu` écrit par `write_vtk` : renvoie `(NumberOfPoints,
    /// NumberOfCells)` de la `Piece` et, pour chaque `DataArray`, ses valeurs
    /// (converties en `f64`) indexées par l'attribut `Name`. Vérifie au passage
    /// le format `binary`, l'en-tête de taille `UInt64` et la longueur des
    /// données selon le type déclaré. Échoue si le XML est mal formé.
    #[cfg(feature = "vtk")]
    fn parse_vtu(text: &str) -> ((usize, usize), std::collections::HashMap<String, Vec<f64>>) {
        use quick_xml::events::Event;
        use quick_xml::Reader;

        let attr = |e: &quick_xml::events::BytesStart, key: &str| -> String {
            let a = e
                .try_get_attribute(key)
                .unwrap()
                .expect("missing attribute");
            String::from_utf8(a.value.into_owned()).unwrap()
        };

        let mut reader = Reader::from_str(text);
        let mut piece = (0, 0);
        let mut arrays = std::collections::HashMap::new();
        let mut current: Option<(String, String)> = None; // (Name, type)
        loop {
            match reader.read_event().expect("malformed XML") {
                Event::Start(e) if e.name().as_ref() == b"Piece" => {
                    piece = (
                        attr(&e, "NumberOfPoints").parse().unwrap(),
                        attr(&e, "NumberOfCells").parse().unwrap(),
                    );
                }
                Event::Start(e) if e.name().as_ref() == b"DataArray" => {
                    assert_eq!(attr(&e, "format"), "binary");
                    current = Some((attr(&e, "Name"), attr(&e, "type")));
                }
                Event::Text(t) => {
                    if let Some((name, scalar_type)) = current.take() {
                        let raw = base64_decode(t.decode().unwrap().trim());
                        let (header, data) = raw.split_at(8);
                        let size = u64::from_le_bytes(header.try_into().unwrap());
                        assert_eq!(size as usize, data.len(), "size header of {name}");
                        let values: Vec<f64> = match scalar_type.as_str() {
                            "Float64" => data
                                .chunks_exact(8)
                                .map(|c| f64::from_le_bytes(c.try_into().unwrap()))
                                .collect(),
                            "Int64" => data
                                .chunks_exact(8)
                                .map(|c| i64::from_le_bytes(c.try_into().unwrap()) as f64)
                                .collect(),
                            "UInt8" => data.iter().map(|&b| f64::from(b)).collect(),
                            other => panic!("unexpected DataArray type {other}"),
                        };
                        arrays.insert(name, values);
                    }
                }
                Event::Eof => break,
                _ => {}
            }
        }
        (piece, arrays)
    }

    /// Décodeur base64 indépendant de `base64_encode`, pour que les tests ne
    /// valident pas l'encodeur par lui-même.
    #[cfg(feature = "vtk")]
    fn base64_decode(text: &str) -> Vec<u8> {
        let sextet = |c: u8| -> u32 {
            match c {
                b'A'..=b'Z' => u32::from(c - b'A'),
                b'a'..=b'z' => u32::from(c - b'a') + 26,
                b'0'..=b'9' => u32::from(c - b'0') + 52,
                b'+' => 62,
                b'/' => 63,
                _ => panic!("invalid base64 character {:?}", c as char),
            }
        };
        let bytes = text.as_bytes();
        assert_eq!(bytes.len() % 4, 0, "base64 length must be a multiple of 4");
        let mut out = Vec::with_capacity(bytes.len() / 4 * 3);
        for quad in bytes.chunks(4) {
            let pad = quad.iter().rev().take_while(|&&c| c == b'=').count();
            let n = quad[..4 - pad]
                .iter()
                .fold(0u32, |acc, &c| acc << 6 | sextet(c))
                << (6 * pad);
            out.extend_from_slice(&n.to_be_bytes()[1..4 - pad]);
        }
        out
    }

    #[cfg(feature = "vtk")]
    #[test]
    fn base64_encode_matches_rfc4648_vectors() {
        // RFC 4648 §10.
        for (input, expected) in [
            ("", ""),
            ("f", "Zg=="),
            ("fo", "Zm8="),
            ("foo", "Zm9v"),
            ("foob", "Zm9vYg=="),
            ("fooba", "Zm9vYmE="),
            ("foobar", "Zm9vYmFy"),
        ] {
            assert_eq!(super::base64_encode(input.as_bytes()), expected);
            assert_eq!(base64_decode(expected), input.as_bytes());
        }
        let all: Vec<u8> = (0..=255).collect();
        assert_eq!(base64_decode(&super::base64_encode(&all)), all);
    }

    #[cfg(feature = "vtk")]
    #[test]
    fn write_vtk_values_round_trip_bit_for_bit() {
        use crate::mesh::structured::UniformGrid1D;

        // Valeurs non finies comprises : un état divergé doit rester
        // inspectable dans ParaView. Comparaison bit à bit (NaN != NaN).
        let values = vec![
            1.0 / 3.0,
            -2.0e-300,
            f64::MIN_POSITIVE,
            -0.0,
            f64::NAN,
            f64::INFINITY,
            f64::NEG_INFINITY,
        ];
        let mesh = UniformGrid1D::new(values.len(), 0.0, 1.0).unwrap();
        let result = SimulationResult {
            states: vec![ContextValue::ScalarField(DVector::from_vec(values.clone()))],
            times: vec![1.0],
            n_steps: 1,
            metadata: std::collections::HashMap::new(),
        };
        let path = std::env::temp_dir().join("oxiflow_test_write_vtk_bits.vtu");
        result.write_vtk(&mesh, &path).unwrap();
        let (_, arrays) = parse_vtu(&std::fs::read_to_string(&path).unwrap());
        let bits = |v: &[f64]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
        assert_eq!(bits(&arrays["field"]), bits(&values));
        std::fs::remove_file(&path).ok();
    }

    #[cfg(feature = "vtk")]
    #[test]
    fn write_vtk_vector_field_one_array_per_component_bit_for_bit() {
        use crate::mesh::structured::UniformGrid1D;
        use nalgebra::DMatrix;

        // Deux composantes, valeurs non finies dans chacune : comparaison bit
        // à bit composante par composante (NaN != NaN).
        let c0 = vec![1.0 / 3.0, f64::NAN, -0.0, 4.0];
        let c1 = vec![f64::INFINITY, 2.0e-300, f64::NEG_INFINITY, -1.5];
        let state =
            DMatrix::from_columns(&[DVector::from_vec(c0.clone()), DVector::from_vec(c1.clone())]);
        let mesh = UniformGrid1D::new(4, 0.0, 3.0).unwrap();
        let result = SimulationResult {
            states: vec![ContextValue::VectorField(state)],
            times: vec![1.0],
            n_steps: 1,
            metadata: std::collections::HashMap::new(),
        };
        let path = std::env::temp_dir().join("oxiflow_test_write_vtk_vector_field.vtu");
        result.write_vtk(&mesh, &path).unwrap();
        let text = std::fs::read_to_string(&path).unwrap();
        std::fs::remove_file(&path).ok();

        let (piece, arrays) = parse_vtu(&text);
        assert_eq!(piece, (4, 3));
        let bits = |v: &[f64]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
        assert_eq!(bits(&arrays["field_0"]), bits(&c0));
        assert_eq!(bits(&arrays["field_1"]), bits(&c1));
        assert!(!arrays.contains_key("field"));
        assert!(text.contains(r#"<PointData Scalars="field_0">"#));
        for c in 0..2 {
            assert!(text.contains(&format!(r#"Name="field_{c}" NumberOfComponents="1""#)));
        }
    }

    #[cfg(feature = "vtk")]
    #[test]
    fn write_vtk_rejects_vector_field_row_mismatch() {
        use crate::mesh::structured::UniformGrid1D;

        let mesh = UniformGrid1D::new(3, 0.0, 1.0).unwrap();
        let result = SimulationResult {
            states: vec![ContextValue::VectorField(nalgebra::DMatrix::zeros(4, 2))],
            times: vec![0.0],
            n_steps: 0,
            metadata: std::collections::HashMap::new(),
        };
        let path = std::env::temp_dir().join("oxiflow_test_write_vtk_vector_rows.vtu");
        let outcome = result.write_vtk(&mesh, &path);
        assert!(matches!(outcome, Err(OxiflowError::Persistence(m)) if m.contains("4 rows")));
    }

    #[cfg(feature = "vtk")]
    #[test]
    fn write_vtk_rejects_vector_field_without_component() {
        use crate::mesh::structured::UniformGrid1D;

        let mesh = UniformGrid1D::new(3, 0.0, 1.0).unwrap();
        let result = SimulationResult {
            states: vec![ContextValue::VectorField(nalgebra::DMatrix::zeros(3, 0))],
            times: vec![0.0],
            n_steps: 0,
            metadata: std::collections::HashMap::new(),
        };
        let path = std::env::temp_dir().join("oxiflow_test_write_vtk_vector_empty.vtu");
        let outcome = result.write_vtk(&mesh, &path);
        assert!(matches!(outcome, Err(OxiflowError::Persistence(m)) if m.contains("no component")));
    }

    #[cfg(feature = "vtk")]
    #[test]
    fn write_vtk_rejects_scalar_field_length_mismatch_and_other_variants() {
        use crate::mesh::structured::UniformGrid1D;

        let mesh = UniformGrid1D::new(3, 0.0, 1.0).unwrap();
        let path = std::env::temp_dir().join("oxiflow_test_write_vtk_rejects_state.vtu");
        for state in [
            ContextValue::ScalarField(DVector::from_element(2, 0.0)),
            ContextValue::Scalar(1.0),
        ] {
            let result = SimulationResult {
                states: vec![state],
                times: vec![0.0],
                n_steps: 0,
                metadata: std::collections::HashMap::new(),
            };
            let outcome = result.write_vtk(&mesh, &path);
            assert!(matches!(outcome, Err(OxiflowError::Persistence(_))));
        }
    }

    #[cfg(feature = "vtk")]
    #[test]
    fn write_vtu_single_node_writes_a_vertex() {
        // `UniformGrid1D` exige n >= 2 : le cas d'un nœud unique (autre `Mesh`)
        // se teste directement sur l'écrivain interne.
        let mut buf = Vec::new();
        super::write_vtu_1d(&mut buf, &[0.5], &[("field", &[7.0][..])]).unwrap();
        let text = String::from_utf8(buf).unwrap();
        let (piece, arrays) = parse_vtu(&text);
        assert_eq!(piece, (1, 1));
        assert_eq!(arrays["connectivity"], vec![0.0]);
        assert_eq!(arrays["offsets"], vec![1.0]);
        assert_eq!(arrays["types"], vec![1.0]); // VTK_VERTEX
    }

    #[cfg(feature = "vtk")]
    #[test]
    fn write_vtk_reports_io_failure() {
        use crate::mesh::structured::UniformGrid1D;

        let mesh = UniformGrid1D::new(2, 0.0, 1.0).unwrap();
        let result = SimulationResult {
            states: vec![ContextValue::ScalarField(DVector::from_element(2, 0.0))],
            times: vec![0.0],
            n_steps: 0,
            metadata: std::collections::HashMap::new(),
        };
        // Un répertoire existant ne peut pas être ouvert comme fichier.
        let outcome = result.write_vtk(&mesh, &std::env::temp_dir());
        assert!(
            matches!(outcome, Err(OxiflowError::Persistence(m)) if m.starts_with("VTK export failed"))
        );
    }
}
