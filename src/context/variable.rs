//! # Module `context::variable`
//!
//! Typed keys identifying context variables required or produced by calculators.
//!
//! `ContextVariable` is the key type of the context map
//! (`HashMap<ContextVariable, ContextValue>`). It must satisfy `Hash + Eq`
//! to serve as a map key — which rules out `f64` fields (DD-003).
//!
//! ## Design note — no `position: f64`
//!
//! chrom-rs stored gradients as point-wise scalars keyed by `(dimension, position)`.
//! oxiflow stores the complete gradient field under a single key
//! `SpatialGradient { dimension }` → `ContextValue::ScalarField`.
//! Node-level access is the operator's responsibility (INV-2, J5).

/// Typed key identifying a context variable in the compute context.
///
/// All variants implement `Hash + Eq` so that `ContextVariable` can serve as
/// a `HashMap` key without workarounds. In particular, no `f64` field appears
/// here — the full spatial gradient field is stored as `ContextValue::ScalarField`.
///
/// # Examples
///
/// ```rust
/// use oxiflow::context::variable::ContextVariable;
///
/// let t   = ContextVariable::Time;
/// let dt  = ContextVariable::TimeStep;
/// let gx  = ContextVariable::SpatialGradient { dimension: 0 };
/// let ext = ContextVariable::External { name: "ambient_temperature" };
///
/// assert_ne!(t, dt);
/// assert_ne!(
///     ContextVariable::SpatialGradient { dimension: 0 },
///     ContextVariable::SpatialGradient { dimension: 1 },
/// );
/// ```
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum ContextVariable {
    /// Current simulation time `t`.
    Time,

    /// Current time step `dt`.
    TimeStep,

    /// Spatial gradient of the primary field along `dimension`.
    ///
    /// The associated `ContextValue` is a `ScalarField` containing one gradient
    /// value per mesh node — not a point-wise scalar.
    ///
    /// `dimension = 0` → ∂u/∂x, `dimension = 1` → ∂u/∂y, etc.
    SpatialGradient {
        /// Spatial dimension index (0-based).
        dimension: usize,
    },

    /// External scalar provided by the user (e.g. ambient temperature, feed concentration).
    ///
    /// The `name` is a static string and participates in `Hash + Eq`.
    External {
        /// Unique name of the external variable.
        name: &'static str,
    },
}

impl std::fmt::Display for ContextVariable {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Time => write!(f, "Time"),
            Self::TimeStep => write!(f, "TimeStep"),
            Self::SpatialGradient { dimension } => {
                write!(f, "SpatialGradient(dim={})", dimension)
            }
            Self::External { name } => write!(f, "External({})", name),
        }
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    // ── Construction & equality ───────────────────────────────────────────────

    #[test]
    fn time_equals_time() {
        assert_eq!(ContextVariable::Time, ContextVariable::Time);
    }

    #[test]
    fn time_differs_from_timestep() {
        assert_ne!(ContextVariable::Time, ContextVariable::TimeStep);
    }

    #[test]
    fn spatial_gradient_same_dimension_equal() {
        let a = ContextVariable::SpatialGradient { dimension: 0 };
        let b = ContextVariable::SpatialGradient { dimension: 0 };
        assert_eq!(a, b);
    }

    #[test]
    fn spatial_gradient_different_dimension_not_equal() {
        let a = ContextVariable::SpatialGradient { dimension: 0 };
        let b = ContextVariable::SpatialGradient { dimension: 1 };
        assert_ne!(a, b);
    }

    #[test]
    fn external_same_name_equal() {
        let a = ContextVariable::External {
            name: "temperature",
        };
        let b = ContextVariable::External {
            name: "temperature",
        };
        assert_eq!(a, b);
    }

    #[test]
    fn external_different_name_not_equal() {
        let a = ContextVariable::External {
            name: "temperature",
        };
        let b = ContextVariable::External { name: "pressure" };
        assert_ne!(a, b);
    }

    // ── Clone ─────────────────────────────────────────────────────────────────

    #[test]
    fn clone_preserves_equality() {
        let vars = [
            ContextVariable::Time,
            ContextVariable::TimeStep,
            ContextVariable::SpatialGradient { dimension: 2 },
            ContextVariable::External { name: "feed" },
        ];
        for v in &vars {
            assert_eq!(v.clone(), *v);
        }
    }

    // ── Hash (usable as HashMap key) ──────────────────────────────────────────

    #[test]
    fn usable_as_hashmap_key() {
        let mut map: HashMap<ContextVariable, f64> = HashMap::new();
        map.insert(ContextVariable::Time, 1.5);
        map.insert(ContextVariable::TimeStep, 0.01);
        map.insert(ContextVariable::SpatialGradient { dimension: 0 }, 0.3);
        map.insert(ContextVariable::External { name: "T_amb" }, 298.15);

        assert_eq!(map[&ContextVariable::Time], 1.5);
        assert_eq!(map[&ContextVariable::TimeStep], 0.01);
        assert_eq!(map[&ContextVariable::SpatialGradient { dimension: 0 }], 0.3);
        assert_eq!(map[&ContextVariable::External { name: "T_amb" }], 298.15);
    }

    #[test]
    fn gradient_dimensions_are_distinct_keys() {
        let mut map: HashMap<ContextVariable, f64> = HashMap::new();
        map.insert(ContextVariable::SpatialGradient { dimension: 0 }, 1.0);
        map.insert(ContextVariable::SpatialGradient { dimension: 1 }, 2.0);

        assert_eq!(map[&ContextVariable::SpatialGradient { dimension: 0 }], 1.0);
        assert_eq!(map[&ContextVariable::SpatialGradient { dimension: 1 }], 2.0);
    }

    // ── Display ───────────────────────────────────────────────────────────────

    #[test]
    fn display_time() {
        assert_eq!(format!("{}", ContextVariable::Time), "Time");
    }

    #[test]
    fn display_timestep() {
        assert_eq!(format!("{}", ContextVariable::TimeStep), "TimeStep");
    }

    #[test]
    fn display_spatial_gradient() {
        let v = ContextVariable::SpatialGradient { dimension: 1 };
        assert_eq!(format!("{}", v), "SpatialGradient(dim=1)");
    }

    #[test]
    fn display_external() {
        let v = ContextVariable::External { name: "T_amb" };
        assert_eq!(format!("{}", v), "External(T_amb)");
    }

    // ── Debug ─────────────────────────────────────────────────────────────────

    #[test]
    fn debug_is_non_empty() {
        let s = format!("{:?}", ContextVariable::SpatialGradient { dimension: 0 });
        assert!(s.contains("SpatialGradient"));
        assert!(s.contains('0'));
    }
}
