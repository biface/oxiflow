//! # Module `solver::snapshot`
//!
//! Serialisable snapshot of a simulation's physical state, for post-mortem
//! divergence analysis and normal pause/resume (DD-025 Option B, issue #71 ;
//! generalised by DD-029, issue #77).
//!
//! ## Two-phase restore contract (DD-025)
//!
//! `SimulationSnapshot` captures only the **physical state** (`ContextValue`,
//! the temporal configuration in effect, and the integration method used) —
//! never `Scenario`/`Domain`, which hold `Box<dyn Trait>` fields and are not
//! serialisable. Restoring a run is a two-phase contract: oxiflow supplies
//! the physical state via this type; the caller reconstructs
//! `Domain`/`Scenario`/`SolverConfiguration` themselves and re-injects the
//! state as the initial condition.
//!
//! ## Two triggers, one type (DD-029)
//!
//! - **Automatic** — [`Solver::on_divergence`](super::Solver::on_divergence)
//!   is called with a snapshot right before a [`OxiflowError::SolverDivergence`]
//!   is returned. Default implementation is a no-op; override to persist.
//! - **Explicit** — normal pause/resume checkpointing. There is no
//!   `Solver::checkpoint()` method: every field of `SimulationSnapshot` is
//!   already public, and the live physical state (`ContextValue`, `t`,
//!   `n_steps`) only exists in the caller's own loop when driving
//!   [`SteppableSolver::step`](super::methods::SteppableSolver::step)
//!   directly — `Solver::solve()` is a single blocking call with no
//!   externally reachable state to hand a trait method. Callers construct
//!   `SimulationSnapshot` directly instead (see example below).
//!
//! # Examples
//!
//! ```rust
//! use oxiflow::solver::snapshot::SimulationSnapshot;
//! use oxiflow::solver::config::{IntegratorKind, StepControl, TimeConfiguration};
//! use oxiflow::context::value::ContextValue;
//! use nalgebra::DVector;
//!
//! // Explicit checkpoint, constructed directly by caller code driving its
//! // own `SteppableSolver::step()` loop — no trait method involved.
//! let snapshot = SimulationSnapshot {
//!     time_config: TimeConfiguration::new(600.0, StepControl::Fixed { dt: 0.1 }),
//!     integrator:  IntegratorKind::BackwardEuler,
//!     state:       ContextValue::ScalarField(DVector::from_element(10, 0.0)),
//!     t:           12.3,
//!     n_steps:     123,
//!     error:       None,
//!     partial:     None,
//! };
//! assert_eq!(snapshot.t, 12.3);
//! ```

#[cfg(feature = "serde")]
use crate::context::error::OxiflowError;
use crate::context::value::ContextValue;
use crate::solver::config::{IntegratorKind, TimeConfiguration};
use crate::solver::SimulationResult;

// ── SimulationSnapshot ──────────────────────────────────────────────────────────

/// Serialisable snapshot of a simulation's physical state.
///
/// See the [module docs](self) for the two-phase restore contract and the
/// two triggers (`on_divergence` automatic, explicit checkpoint constructed
/// directly).
///
/// Deliberately **not** `#[non_exhaustive]`: the whole point of not having a
/// `Solver::checkpoint()` method is that callers construct this struct
/// directly via its public fields (see module docs) — `#[non_exhaustive]`
/// would block exactly that from outside this crate.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct SimulationSnapshot {
    /// Serialisable temporal configuration in effect when the snapshot was
    /// taken.
    pub time_config: TimeConfiguration,
    /// Integration method in effect when the snapshot was taken.
    pub integrator: IntegratorKind,
    /// Physical state at snapshot time.
    pub state: ContextValue,
    /// Simulation time at snapshot.
    pub t: f64,
    /// Number of time steps taken up to the snapshot.
    pub n_steps: usize,
    /// Populated on failure — error message for post-mortem analysis.
    /// `None` for a normal explicit checkpoint.
    pub error: Option<String>,
    /// Partial result up to the snapshot point, if available.
    pub partial: Option<SimulationResult>,
}

// ── File format ──────────────────────────────────────────────────────────────

/// On-disk format for [`write_snapshot`] / [`read_snapshot`].
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SnapshotFormat {
    /// Human-readable JSON (`serde_json`) — intended for debugging.
    Json,
    /// Native binary format (`bincode` 1.x) — fidelity and speed, the
    /// recommended default for `on_divergence`/checkpoint round-trips.
    Bincode,
}

/// Writes `snapshot` to `path` in the given [`SnapshotFormat`].
///
/// Requires the `serde` feature.
///
/// # Errors
///
/// Returns [`OxiflowError::Persistence`] if the file cannot be created or
/// written, or if serialisation fails.
#[cfg(feature = "serde")]
pub fn write_snapshot(
    snapshot: &SimulationSnapshot,
    path: &std::path::Path,
    format: SnapshotFormat,
) -> Result<(), OxiflowError> {
    match format {
        SnapshotFormat::Json => {
            let file = std::fs::File::create(path)
                .map_err(|e| OxiflowError::Persistence(format!("cannot create {path:?}: {e}")))?;
            serde_json::to_writer_pretty(file, snapshot)
                .map_err(|e| OxiflowError::Persistence(format!("JSON serialisation failed: {e}")))
        }
        SnapshotFormat::Bincode => {
            let bytes = bincode::serialize(snapshot).map_err(|e| {
                OxiflowError::Persistence(format!("bincode serialisation failed: {e}"))
            })?;
            std::fs::write(path, bytes)
                .map_err(|e| OxiflowError::Persistence(format!("cannot write {path:?}: {e}")))
        }
    }
}

/// Reads a [`SimulationSnapshot`] previously written by [`write_snapshot`].
///
/// Requires the `serde` feature.
///
/// # Errors
///
/// Returns [`OxiflowError::Persistence`] if the file cannot be read or if
/// deserialisation fails.
#[cfg(feature = "serde")]
pub fn read_snapshot(
    path: &std::path::Path,
    format: SnapshotFormat,
) -> Result<SimulationSnapshot, OxiflowError> {
    match format {
        SnapshotFormat::Json => {
            let file = std::fs::File::open(path)
                .map_err(|e| OxiflowError::Persistence(format!("cannot open {path:?}: {e}")))?;
            serde_json::from_reader(file)
                .map_err(|e| OxiflowError::Persistence(format!("JSON deserialisation failed: {e}")))
        }
        SnapshotFormat::Bincode => {
            let bytes = std::fs::read(path)
                .map_err(|e| OxiflowError::Persistence(format!("cannot read {path:?}: {e}")))?;
            bincode::deserialize(&bytes).map_err(|e| {
                OxiflowError::Persistence(format!("bincode deserialisation failed: {e}"))
            })
        }
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::solver::config::StepControl;
    use nalgebra::DVector;

    fn sample_snapshot() -> SimulationSnapshot {
        SimulationSnapshot {
            time_config: TimeConfiguration::new(10.0, StepControl::Fixed { dt: 0.1 }),
            integrator: IntegratorKind::BackwardEuler,
            state: ContextValue::ScalarField(DVector::from_element(5, 1.5)),
            t: 3.4,
            n_steps: 34,
            error: Some("non-finite value detected in state vector".to_string()),
            partial: None,
        }
    }

    #[test]
    fn snapshot_fields_accessible() {
        let s = sample_snapshot();
        assert_eq!(s.t, 3.4);
        assert_eq!(s.n_steps, 34);
        assert!(s.error.is_some());
    }

    #[cfg(feature = "serde")]
    #[test]
    fn json_round_trip() {
        let dir = tempdir();
        let path = dir.join("snapshot.json");
        let original = sample_snapshot();

        write_snapshot(&original, &path, SnapshotFormat::Json).unwrap();
        let restored = read_snapshot(&path, SnapshotFormat::Json).unwrap();

        assert_eq!(restored.t, original.t);
        assert_eq!(restored.n_steps, original.n_steps);
        assert_eq!(restored.error, original.error);
        std::fs::remove_file(&path).ok();
    }

    #[cfg(feature = "serde")]
    #[test]
    fn bincode_round_trip() {
        let dir = tempdir();
        let path = dir.join("snapshot.bin");
        let original = sample_snapshot();

        write_snapshot(&original, &path, SnapshotFormat::Bincode).unwrap();
        let restored = read_snapshot(&path, SnapshotFormat::Bincode).unwrap();

        assert_eq!(restored.t, original.t);
        assert_eq!(restored.n_steps, original.n_steps);
        assert_eq!(restored.error, original.error);
        std::fs::remove_file(&path).ok();
    }

    #[cfg(feature = "serde")]
    #[test]
    fn read_snapshot_missing_file_is_persistence_error() {
        let path = std::path::PathBuf::from("/nonexistent/path/snapshot.json");
        let err = read_snapshot(&path, SnapshotFormat::Json).unwrap_err();
        assert!(matches!(err, OxiflowError::Persistence(_)));
    }

    /// Returns the system temp dir — no external crate needed for this.
    #[cfg(feature = "serde")]
    fn tempdir() -> std::path::PathBuf {
        std::env::temp_dir()
    }
}
