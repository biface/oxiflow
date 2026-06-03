//! # Module `context::quantity`
//!
//! Type-safe physical quantity identifiers for multi-component fields (DD-010).
//!
//! ## Design
//!
//! [`PhysicalQuantity`] identifies a physical field by both its *kind*
//! (concentration, temperature, …) and its *component index* for multi-component
//! systems. This avoids the flat-enum anti-pattern seen in chrom-rs, where
//! `PhysicalQuantity::Concentration` was a single key for all species — forcing
//! an implicit column-ordering convention on the state matrix.
//!
//! | Variant | Component | Typical use |
//! |---------|-----------|-------------|
//! | `Concentration { component }` | species index | chromatography, Nernst-Planck |
//! | `Temperature` | — (always scalar) | heat conduction, thermo-mechanical coupling |
//! | `Pressure` | — (always scalar) | Darcy, Saint-Venant hydrostatic |
//! | `Velocity { component }` | spatial dimension | Saint-Venant `hv`, structural dynamics |
//! | `Custom { name, component }` | user-defined | `WaterDepth`, `MagneticFlux`, … |
//!
//! ## Convention
//!
//! `component: 0` is the canonical index for single-component models. Use the
//! constructor helpers (`PhysicalQuantity::concentration()`) to avoid boilerplate
//! in J1/J2 code.
//!
//! ## Serialisation
//!
//! Under `--features serde`, `PhysicalQuantity` serialises to a flat JSON object:
//!
//! ```json
//! { "quantity": "Concentration", "component": 0 }
//! { "quantity": "Temperature",   "component": 0 }
//! { "quantity": "WaterDepth",    "component": 0 }
//! ```
//!
//! `Temperature` and `Pressure` (intrinsically scalar) always serialise with
//! `"component": 0` for uniformity. `Custom` serialises the `name` field directly
//! as `"quantity"` — no `"Custom(...)"` wrapper — for maximum readability by
//! external tools.

use std::borrow::Cow;
use std::fmt;

// ── PhysicalQuantity ──────────────────────────────────────────────────────────

/// Type-safe identifier for a physical field in a multi-component system.
///
/// Used as part of the composite key `(DomainId, PhysicalQuantity)` in
/// [`MultiDomainState`](super::state::MultiDomainState).
///
/// # Examples
///
/// ```rust
/// use oxiflow::context::quantity::PhysicalQuantity;
///
/// // Single-component convenience constructors (component: 0)
/// let c   = PhysicalQuantity::concentration();
/// let t   = PhysicalQuantity::temperature();
/// let p   = PhysicalQuantity::pressure();
/// let v   = PhysicalQuantity::velocity();
///
/// // Multi-component — explicit index
/// let c1  = PhysicalQuantity::Concentration { component: 1 };
/// let hv  = PhysicalQuantity::Velocity { component: 0 };
///
/// // Custom field
/// let h   = PhysicalQuantity::custom("WaterDepth");
/// let b   = PhysicalQuantity::custom("MagneticFlux");
///
/// assert_ne!(c, c1);
/// assert_eq!(c, PhysicalQuantity::Concentration { component: 0 });
/// ```
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum PhysicalQuantity {
    /// Species concentration — multi-component chromatography, Nernst-Planck,
    /// reaction-diffusion.
    ///
    /// `component: 0` for single-species models (J1/J2 default).
    Concentration {
        /// Species / component index (0-based).
        component: usize,
    },

    /// Temperature field — heat conduction, thermo-mechanical coupling.
    ///
    /// Intrinsically scalar: no `component` field. Serialises with
    /// `"component": 0` for format uniformity.
    Temperature,

    /// Pressure field — Darcy flow, Saint-Venant hydrostatic pressure.
    ///
    /// Intrinsically scalar: no `component` field. Serialises with
    /// `"component": 0` for format uniformity.
    Pressure,

    /// Velocity component — Saint-Venant momentum `hv`, structural dynamics.
    ///
    /// `component` indexes the spatial dimension (0 = axial for 1-D models).
    Velocity {
        /// Spatial dimension index (0-based).
        component: usize,
    },

    /// User-defined physical quantity.
    ///
    /// Use for fields that do not map to a standard variant: water depth `h`,
    /// magnetic flux density `B`, void fraction `θ`, etc.
    ///
    /// `name` is a `Cow<'static, str>` — zero allocation for compile-time
    /// literals, `Deserialize`-compatible for dynamic names (same pattern as
    /// `ContextVariable::External`).
    Custom {
        /// Field name. Serialises directly as the `"quantity"` JSON key.
        name: Cow<'static, str>,
        /// Component index (0-based). Use `0` for scalar custom fields.
        component: usize,
    },
}

impl PhysicalQuantity {
    // ── Convenience constructors ──────────────────────────────────────────────

    /// Concentration for single-component models (`component: 0`).
    #[inline]
    pub fn concentration() -> Self {
        Self::Concentration { component: 0 }
    }

    /// Temperature (scalar — no component index).
    #[inline]
    pub fn temperature() -> Self {
        Self::Temperature
    }

    /// Pressure (scalar — no component index).
    #[inline]
    pub fn pressure() -> Self {
        Self::Pressure
    }

    /// Velocity for 1-D models (`component: 0`).
    #[inline]
    pub fn velocity() -> Self {
        Self::Velocity { component: 0 }
    }

    /// Custom scalar field identified by `name` (`component: 0`).
    ///
    /// # Examples
    ///
    /// ```rust
    /// use oxiflow::context::quantity::PhysicalQuantity;
    ///
    /// let h = PhysicalQuantity::custom("WaterDepth");
    /// assert_eq!(h, PhysicalQuantity::Custom {
    ///     name: "WaterDepth".into(),
    ///     component: 0,
    /// });
    /// ```
    #[inline]
    pub fn custom(name: impl Into<Cow<'static, str>>) -> Self {
        Self::Custom {
            name: name.into(),
            component: 0,
        }
    }

    // ── Accessors ─────────────────────────────────────────────────────────────

    /// Returns the component index.
    ///
    /// `Temperature` and `Pressure` always return `0` (intrinsically scalar).
    ///
    /// # Examples
    ///
    /// ```rust
    /// use oxiflow::context::quantity::PhysicalQuantity;
    ///
    /// assert_eq!(PhysicalQuantity::concentration().component(), 0);
    /// assert_eq!(PhysicalQuantity::Concentration { component: 2 }.component(), 2);
    /// assert_eq!(PhysicalQuantity::temperature().component(), 0);
    /// ```
    pub fn component(&self) -> usize {
        match self {
            Self::Concentration { component } => *component,
            Self::Temperature => 0,
            Self::Pressure => 0,
            Self::Velocity { component } => *component,
            Self::Custom { component, .. } => *component,
        }
    }

    /// Returns the quantity name as a string slice.
    ///
    /// For `Custom`, returns the user-provided name. For standard variants,
    /// returns the variant name. This is the value written to the `"quantity"`
    /// field in the serialised JSON.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use oxiflow::context::quantity::PhysicalQuantity;
    ///
    /// assert_eq!(PhysicalQuantity::concentration().kind_str(), "Concentration");
    /// assert_eq!(PhysicalQuantity::temperature().kind_str(), "Temperature");
    /// assert_eq!(PhysicalQuantity::custom("WaterDepth").kind_str(), "WaterDepth");
    /// ```
    pub fn kind_str(&self) -> &str {
        match self {
            Self::Concentration { .. } => "Concentration",
            Self::Temperature => "Temperature",
            Self::Pressure => "Pressure",
            Self::Velocity { .. } => "Velocity",
            Self::Custom { name, .. } => name.as_ref(),
        }
    }
}

impl fmt::Display for PhysicalQuantity {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Concentration { component } => write!(f, "Concentration({component})"),
            Self::Temperature => write!(f, "Temperature"),
            Self::Pressure => write!(f, "Pressure"),
            Self::Velocity { component } => write!(f, "Velocity({component})"),
            Self::Custom { name, component } => write!(f, "{name}({component})"),
        }
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    // ── Constructors ──────────────────────────────────────────────────────────

    #[test]
    fn concentration_default_is_component_zero() {
        assert_eq!(
            PhysicalQuantity::concentration(),
            PhysicalQuantity::Concentration { component: 0 }
        );
    }

    #[test]
    fn temperature_equals_temperature() {
        assert_eq!(
            PhysicalQuantity::temperature(),
            PhysicalQuantity::Temperature
        );
    }

    #[test]
    fn pressure_equals_pressure() {
        assert_eq!(PhysicalQuantity::pressure(), PhysicalQuantity::Pressure);
    }

    #[test]
    fn velocity_default_is_component_zero() {
        assert_eq!(
            PhysicalQuantity::velocity(),
            PhysicalQuantity::Velocity { component: 0 }
        );
    }

    #[test]
    fn custom_default_is_component_zero() {
        assert_eq!(
            PhysicalQuantity::custom("WaterDepth"),
            PhysicalQuantity::Custom {
                name: "WaterDepth".into(),
                component: 0,
            }
        );
    }

    // ── Equality & distinctness ───────────────────────────────────────────────

    #[test]
    fn different_components_are_not_equal() {
        let c0 = PhysicalQuantity::Concentration { component: 0 };
        let c1 = PhysicalQuantity::Concentration { component: 1 };
        assert_ne!(c0, c1);
    }

    #[test]
    fn different_kinds_are_not_equal() {
        assert_ne!(
            PhysicalQuantity::concentration(),
            PhysicalQuantity::temperature()
        );
        assert_ne!(
            PhysicalQuantity::concentration(),
            PhysicalQuantity::pressure()
        );
        assert_ne!(
            PhysicalQuantity::concentration(),
            PhysicalQuantity::velocity()
        );
    }

    #[test]
    fn custom_different_names_not_equal() {
        assert_ne!(
            PhysicalQuantity::custom("WaterDepth"),
            PhysicalQuantity::custom("MagneticFlux")
        );
    }

    #[test]
    fn custom_same_name_different_component_not_equal() {
        let a = PhysicalQuantity::Custom {
            name: "B".into(),
            component: 0,
        };
        let b = PhysicalQuantity::Custom {
            name: "B".into(),
            component: 1,
        };
        assert_ne!(a, b);
    }

    // ── component() accessor ──────────────────────────────────────────────────

    #[test]
    fn component_returns_correct_index() {
        assert_eq!(
            PhysicalQuantity::Concentration { component: 2 }.component(),
            2
        );
        assert_eq!(PhysicalQuantity::Temperature.component(), 0);
        assert_eq!(PhysicalQuantity::Pressure.component(), 0);
        assert_eq!(PhysicalQuantity::Velocity { component: 1 }.component(), 1);
        assert_eq!(
            PhysicalQuantity::Custom {
                name: "B".into(),
                component: 3
            }
            .component(),
            3
        );
    }

    // ── kind_str() accessor ───────────────────────────────────────────────────

    #[test]
    fn kind_str_standard_variants() {
        assert_eq!(
            PhysicalQuantity::concentration().kind_str(),
            "Concentration"
        );
        assert_eq!(PhysicalQuantity::temperature().kind_str(), "Temperature");
        assert_eq!(PhysicalQuantity::pressure().kind_str(), "Pressure");
        assert_eq!(PhysicalQuantity::velocity().kind_str(), "Velocity");
    }

    #[test]
    fn kind_str_custom_returns_name() {
        assert_eq!(
            PhysicalQuantity::custom("WaterDepth").kind_str(),
            "WaterDepth"
        );
    }

    // ── Display ───────────────────────────────────────────────────────────────

    #[test]
    fn display_concentration() {
        assert_eq!(
            format!("{}", PhysicalQuantity::Concentration { component: 1 }),
            "Concentration(1)"
        );
    }

    #[test]
    fn display_temperature_and_pressure() {
        assert_eq!(format!("{}", PhysicalQuantity::Temperature), "Temperature");
        assert_eq!(format!("{}", PhysicalQuantity::Pressure), "Pressure");
    }

    #[test]
    fn display_velocity() {
        assert_eq!(
            format!("{}", PhysicalQuantity::Velocity { component: 0 }),
            "Velocity(0)"
        );
    }

    #[test]
    fn display_custom() {
        assert_eq!(
            format!(
                "{}",
                PhysicalQuantity::Custom {
                    name: "WaterDepth".into(),
                    component: 0
                }
            ),
            "WaterDepth(0)"
        );
    }

    // ── Hash (usable as HashMap key) ──────────────────────────────────────────

    #[test]
    fn usable_as_hashmap_key() {
        let mut map: HashMap<PhysicalQuantity, f64> = HashMap::new();
        map.insert(PhysicalQuantity::concentration(), 1.0);
        map.insert(PhysicalQuantity::Concentration { component: 1 }, 2.0);
        map.insert(PhysicalQuantity::temperature(), 298.15);
        map.insert(PhysicalQuantity::custom("WaterDepth"), 0.5);

        assert_eq!(map[&PhysicalQuantity::concentration()], 1.0);
        assert_eq!(map[&PhysicalQuantity::Concentration { component: 1 }], 2.0);
        assert_eq!(map[&PhysicalQuantity::temperature()], 298.15);
        assert_eq!(map[&PhysicalQuantity::custom("WaterDepth")], 0.5);
    }

    #[test]
    fn three_component_scenario() {
        // Validates the 3-component acceptance criterion from #38
        let mut map: HashMap<PhysicalQuantity, &str> = HashMap::new();
        map.insert(PhysicalQuantity::Concentration { component: 0 }, "Malic");
        map.insert(PhysicalQuantity::Concentration { component: 1 }, "Citric");
        map.insert(PhysicalQuantity::Concentration { component: 2 }, "Tartaric");

        assert_eq!(
            map[&PhysicalQuantity::Concentration { component: 0 }],
            "Malic"
        );
        assert_eq!(
            map[&PhysicalQuantity::Concentration { component: 1 }],
            "Citric"
        );
        assert_eq!(
            map[&PhysicalQuantity::Concentration { component: 2 }],
            "Tartaric"
        );
        assert_eq!(map.len(), 3);
    }

    // ── Clone ─────────────────────────────────────────────────────────────────

    #[test]
    fn clone_preserves_equality() {
        let variants = [
            PhysicalQuantity::Concentration { component: 0 },
            PhysicalQuantity::Temperature,
            PhysicalQuantity::Pressure,
            PhysicalQuantity::Velocity { component: 1 },
            PhysicalQuantity::Custom {
                name: "B".into(),
                component: 0,
            },
        ];
        for v in &variants {
            assert_eq!(v.clone(), *v);
        }
    }
}
