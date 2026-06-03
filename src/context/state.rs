//! # Module `context::state`
//!
//! Multi-domain state container for coupled simulations (DD-010, DD-011).
//!
//! ## Design
//!
//! [`MultiDomainState`] maps a composite key `(DomainId, PhysicalQuantity)`
//! to a [`ContextValue`] field. This is **Option B** from the design review
//! (2026-06-01): the key is fully typed and self-documenting — no implicit
//! column-ordering convention is required.
//!
//! ```text
//! ("column", Concentration { component: 0 }) → ScalarField([0.0, 1.2, …])
//! ("column", Concentration { component: 1 }) → ScalarField([0.0, 0.3, …])
//! ("lake",   Velocity      { component: 0 }) → ScalarField([0.1, 0.2, …])
//! ```
//!
//! ## Serialisation (feature `serde`)
//!
//! `MultiDomainState` serialises to an array of explicit entry objects for
//! maximum readability by external tools (Python, CLI, wiki):
//!
//! ```json
//! {
//!   "states": [
//!     { "domain": "column", "quantity": "Concentration", "component": 0, "value": [0.0, 1.2] },
//!     { "domain": "lake",   "quantity": "Velocity",      "component": 0, "value": [0.1, 0.2] }
//!   ]
//! }
//! ```
//!
//! `Custom` quantities serialise the `name` field directly as `"quantity"` —
//! no `"Custom(...)"` wrapper. `Temperature` and `Pressure` always serialise
//! with `"component": 0`.
//!
//! ## Memory layout
//!
//! Field values are stored as [`ContextValue::ScalarField`] (`DVector<f64>`,
//! column-major) or [`ContextValue::VectorField`] (`DMatrix<f64>`, column-major
//! — nalgebra default). This satisfies DD-026 INV-GPU-1 and INV-GPU-4.
//!
// GPU-READY: bytemuck::Pod candidate (DD-026 INV-GPU-5) — pending v0.5.0 (DD-013)

use std::collections::HashMap;

use crate::context::error::OxiflowError;
use crate::context::quantity::PhysicalQuantity;
use crate::context::value::ContextValue;
use crate::solver::scenario::DomainId;

// ── MultiDomainState ──────────────────────────────────────────────────────────

/// Multi-domain physical state container.
///
/// Maps `(DomainId, PhysicalQuantity)` composite keys to [`ContextValue`]
/// fields. Used by [`CouplingOperator`](crate::coupling) to exchange state
/// between physical domains without implicit ordering conventions (DD-010,
/// DD-011, INV-3).
///
/// # Examples
///
/// ```rust
/// use oxiflow::context::state::MultiDomainState;
/// use oxiflow::context::quantity::PhysicalQuantity;
/// use oxiflow::context::value::ContextValue;
/// use oxiflow::solver::scenario::DomainId;
/// use nalgebra::DVector;
///
/// let mut state = MultiDomainState::new();
///
/// state.set(
///     DomainId::new("column"),
///     PhysicalQuantity::concentration(),
///     ContextValue::ScalarField(DVector::from_vec(vec![0.0, 1.0, 2.0])),
/// );
///
/// let field = state.get(&DomainId::new("column"), &PhysicalQuantity::concentration());
/// assert!(field.is_some());
/// ```
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(into = "MultiDomainStateSerde"))]
#[cfg_attr(feature = "serde", serde(try_from = "MultiDomainStateSerde"))]
pub struct MultiDomainState {
    /// Internal map: (domain, quantity) → field value.
    ///
    /// Column-major memory layout for all field payloads (DD-026 INV-GPU-4).
    states: HashMap<(DomainId, PhysicalQuantity), ContextValue>,
}

impl MultiDomainState {
    // ── Constructors ──────────────────────────────────────────────────────────

    /// Creates an empty state container.
    pub fn new() -> Self {
        Self {
            states: HashMap::new(),
        }
    }

    // ── Accessors ─────────────────────────────────────────────────────────────

    /// Returns a reference to the field for `(domain, quantity)`, if present.
    pub fn get(&self, domain: &DomainId, quantity: &PhysicalQuantity) -> Option<&ContextValue> {
        self.states.get(&(domain.clone(), quantity.clone()))
    }

    /// Returns a mutable reference to the field for `(domain, quantity)`, if present.
    pub fn get_mut(
        &mut self,
        domain: &DomainId,
        quantity: &PhysicalQuantity,
    ) -> Option<&mut ContextValue> {
        self.states.get_mut(&(domain.clone(), quantity.clone()))
    }

    /// Inserts or replaces the field for `(domain, quantity)`.
    pub fn set(&mut self, domain: DomainId, quantity: PhysicalQuantity, value: ContextValue) {
        self.states.insert((domain, quantity), value);
    }

    /// Removes the field for `(domain, quantity)`, returning it if present.
    pub fn remove(
        &mut self,
        domain: &DomainId,
        quantity: &PhysicalQuantity,
    ) -> Option<ContextValue> {
        self.states.remove(&(domain.clone(), quantity.clone()))
    }

    /// Returns `true` if no fields are stored.
    pub fn is_empty(&self) -> bool {
        self.states.is_empty()
    }

    /// Returns the number of stored fields.
    pub fn len(&self) -> usize {
        self.states.len()
    }

    /// Returns an iterator over all `((domain, quantity), value)` entries.
    pub fn iter(&self) -> impl Iterator<Item = (&DomainId, &PhysicalQuantity, &ContextValue)> {
        self.states
            .iter()
            .map(|((domain, quantity), value)| (domain, quantity, value))
    }

    // ── Domain helpers ────────────────────────────────────────────────────────

    /// Returns all fields belonging to `domain`.
    pub fn domain_fields<'a>(
        &'a self,
        domain: &'a DomainId,
    ) -> impl Iterator<Item = (&'a PhysicalQuantity, &'a ContextValue)> + 'a {
        self.states
            .iter()
            .filter(move |((d, _), _)| d == domain)
            .map(|((_, q), v)| (q, v))
    }

    /// Returns `true` if `domain` has at least one field stored.
    pub fn contains_domain(&self, domain: &DomainId) -> bool {
        self.states.keys().any(|(d, _)| d == domain)
    }

    // ── Validation ────────────────────────────────────────────────────────────

    /// Verifies that all fields for a domain have the same number of DOFs.
    ///
    /// Returns `OxiflowError::InvalidDomain` if inconsistent field lengths are
    /// detected for `domain`.
    pub fn validate_domain_consistency(&self, domain: &DomainId) -> Result<(), OxiflowError> {
        let lengths: Vec<usize> = self
            .domain_fields(domain)
            .filter_map(|(_, v)| match v {
                ContextValue::ScalarField(f) => Some(f.len()),
                ContextValue::VectorField(f) => Some(f.nrows()),
                _ => None,
            })
            .collect();

        if lengths.windows(2).any(|w| w[0] != w[1]) {
            return Err(OxiflowError::InvalidDomain(format!(
                "domain '{}' has fields with inconsistent DOF counts: {:?}",
                domain, lengths
            )));
        }
        Ok(())
    }
}

impl Default for MultiDomainState {
    fn default() -> Self {
        Self::new()
    }
}

// ── Serde support ─────────────────────────────────────────────────────────────

/// Serialisation entry — one per `(domain, quantity)` pair.
///
/// Produces the human-readable JSON format consumed by external tools:
///
/// ```json
/// { "domain": "column", "quantity": "Concentration", "component": 0, "value": [...] }
/// ```
#[cfg(feature = "serde")]
#[derive(serde::Serialize, serde::Deserialize)]
struct MultiDomainStateEntry {
    domain: String,
    quantity: String,
    component: usize,
    value: ContextValue,
}

/// Newtype wrapper for serde — holds the flat array of entries.
#[cfg(feature = "serde")]
#[derive(serde::Serialize, serde::Deserialize)]
struct MultiDomainStateSerde {
    states: Vec<MultiDomainStateEntry>,
}

#[cfg(feature = "serde")]
impl From<MultiDomainState> for MultiDomainStateSerde {
    fn from(mds: MultiDomainState) -> Self {
        let states = mds
            .states
            .into_iter()
            .map(|((domain, quantity), value)| MultiDomainStateEntry {
                domain: domain.as_str().to_string(),
                quantity: quantity.kind_str().to_string(),
                component: quantity.component(),
                value,
            })
            .collect();
        Self { states }
    }
}

#[cfg(feature = "serde")]
impl TryFrom<MultiDomainStateSerde> for MultiDomainState {
    type Error = OxiflowError;

    fn try_from(raw: MultiDomainStateSerde) -> Result<Self, Self::Error> {
        let mut mds = MultiDomainState::new();
        for entry in raw.states {
            let domain = DomainId::new(entry.domain);
            let quantity = parse_quantity(&entry.quantity, entry.component)?;
            mds.set(domain, quantity, entry.value);
        }
        Ok(mds)
    }
}

/// Reconstructs a [`PhysicalQuantity`] from its serialised `kind_str` and
/// `component` index.
#[cfg(feature = "serde")]
fn parse_quantity(kind: &str, component: usize) -> Result<PhysicalQuantity, OxiflowError> {
    match kind {
        "Concentration" => Ok(PhysicalQuantity::Concentration { component }),
        "Temperature" => Ok(PhysicalQuantity::Temperature),
        "Pressure" => Ok(PhysicalQuantity::Pressure),
        "Velocity" => Ok(PhysicalQuantity::Velocity { component }),
        name => Ok(PhysicalQuantity::Custom {
            name: name.to_string().into(),
            component,
        }),
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::DVector;

    fn scalar_field(values: Vec<f64>) -> ContextValue {
        ContextValue::ScalarField(DVector::from_vec(values))
    }

    fn column() -> DomainId {
        DomainId::new("column")
    }
    fn lake() -> DomainId {
        DomainId::new("lake")
    }

    // ── Basic get/set/remove ──────────────────────────────────────────────────

    #[test]
    fn set_and_get() {
        let mut state = MultiDomainState::new();
        state.set(
            column(),
            PhysicalQuantity::concentration(),
            scalar_field(vec![1.0, 2.0]),
        );
        assert!(state
            .get(&column(), &PhysicalQuantity::concentration())
            .is_some());
    }

    #[test]
    fn get_absent_returns_none() {
        let state = MultiDomainState::new();
        assert!(state
            .get(&column(), &PhysicalQuantity::concentration())
            .is_none());
    }

    #[test]
    fn remove_returns_value() {
        let mut state = MultiDomainState::new();
        state.set(
            column(),
            PhysicalQuantity::concentration(),
            scalar_field(vec![1.0]),
        );
        let removed = state.remove(&column(), &PhysicalQuantity::concentration());
        assert!(removed.is_some());
        assert!(state.is_empty());
    }

    #[test]
    fn set_overwrites_existing() {
        let mut state = MultiDomainState::new();
        state.set(
            column(),
            PhysicalQuantity::concentration(),
            scalar_field(vec![1.0]),
        );
        state.set(
            column(),
            PhysicalQuantity::concentration(),
            scalar_field(vec![9.0]),
        );
        assert_eq!(state.len(), 1);
    }

    // ── Multi-component ───────────────────────────────────────────────────────

    #[test]
    fn three_components_are_distinct_keys() {
        let mut state = MultiDomainState::new();
        for k in 0..3 {
            state.set(
                column(),
                PhysicalQuantity::Concentration { component: k },
                scalar_field(vec![k as f64]),
            );
        }
        assert_eq!(state.len(), 3);
        for k in 0..3 {
            let v = state.get(&column(), &PhysicalQuantity::Concentration { component: k });
            assert!(v.is_some());
        }
    }

    // ── Multi-domain ──────────────────────────────────────────────────────────

    #[test]
    fn two_domains_same_quantity_are_distinct() {
        let mut state = MultiDomainState::new();
        state.set(
            column(),
            PhysicalQuantity::concentration(),
            scalar_field(vec![1.0]),
        );
        state.set(
            lake(),
            PhysicalQuantity::concentration(),
            scalar_field(vec![2.0]),
        );
        assert_eq!(state.len(), 2);
        assert!(state
            .get(&column(), &PhysicalQuantity::concentration())
            .is_some());
        assert!(state
            .get(&lake(), &PhysicalQuantity::concentration())
            .is_some());
    }

    #[test]
    fn contains_domain() {
        let mut state = MultiDomainState::new();
        state.set(
            column(),
            PhysicalQuantity::concentration(),
            scalar_field(vec![1.0]),
        );
        assert!(state.contains_domain(&column()));
        assert!(!state.contains_domain(&lake()));
    }

    #[test]
    fn domain_fields_filters_correctly() {
        let mut state = MultiDomainState::new();
        state.set(
            column(),
            PhysicalQuantity::concentration(),
            scalar_field(vec![1.0]),
        );
        state.set(
            column(),
            PhysicalQuantity::temperature(),
            scalar_field(vec![298.0]),
        );
        state.set(
            lake(),
            PhysicalQuantity::concentration(),
            scalar_field(vec![0.5]),
        );

        let col_id = column();
        let col_fields: Vec<_> = state.domain_fields(&col_id).collect();
        assert_eq!(col_fields.len(), 2);

        let lake_id = lake();
        let lake_fields: Vec<_> = state.domain_fields(&lake_id).collect();
        assert_eq!(lake_fields.len(), 1);
    }

    // ── validate_domain_consistency ───────────────────────────────────────────

    #[test]
    fn consistent_domain_passes_validation() {
        let mut state = MultiDomainState::new();
        state.set(
            column(),
            PhysicalQuantity::Concentration { component: 0 },
            scalar_field(vec![1.0, 2.0, 3.0]),
        );
        state.set(
            column(),
            PhysicalQuantity::Concentration { component: 1 },
            scalar_field(vec![0.1, 0.2, 0.3]),
        );
        assert!(state.validate_domain_consistency(&column()).is_ok());
    }

    #[test]
    fn inconsistent_domain_fails_validation() {
        let mut state = MultiDomainState::new();
        state.set(
            column(),
            PhysicalQuantity::Concentration { component: 0 },
            scalar_field(vec![1.0, 2.0]),
        );
        state.set(
            column(),
            PhysicalQuantity::Concentration { component: 1 },
            scalar_field(vec![0.1, 0.2, 0.3]),
        );
        assert!(state.validate_domain_consistency(&column()).is_err());
    }

    // ── Default ───────────────────────────────────────────────────────────────

    #[test]
    fn default_is_empty() {
        let state = MultiDomainState::default();
        assert!(state.is_empty());
        assert_eq!(state.len(), 0);
    }

    // ── Serde round-trip ──────────────────────────────────────────────────────

    #[cfg(feature = "serde")]
    #[test]
    fn serde_roundtrip_standard_variants() {
        let mut state = MultiDomainState::new();
        state.set(
            column(),
            PhysicalQuantity::concentration(),
            scalar_field(vec![1.0, 2.0]),
        );
        state.set(
            lake(),
            PhysicalQuantity::temperature(),
            scalar_field(vec![298.15]),
        );

        let json = serde_json::to_string(&state).unwrap();
        let restored: MultiDomainState = serde_json::from_str(&json).unwrap();

        assert!(restored
            .get(&column(), &PhysicalQuantity::concentration())
            .is_some());
        assert!(restored
            .get(&lake(), &PhysicalQuantity::temperature())
            .is_some());
    }

    #[cfg(feature = "serde")]
    #[test]
    fn serde_roundtrip_custom_variant() {
        let mut state = MultiDomainState::new();
        state.set(
            lake(),
            PhysicalQuantity::custom("WaterDepth"),
            scalar_field(vec![3.5, 4.0]),
        );

        let json = serde_json::to_string(&state).unwrap();
        let restored: MultiDomainState = serde_json::from_str(&json).unwrap();

        assert!(restored
            .get(&lake(), &PhysicalQuantity::custom("WaterDepth"))
            .is_some());
    }

    #[cfg(feature = "serde")]
    #[test]
    fn serde_json_format_is_human_readable() {
        let mut state = MultiDomainState::new();
        state.set(
            DomainId::new("col"),
            PhysicalQuantity::Concentration { component: 0 },
            scalar_field(vec![1.0]),
        );

        let json = serde_json::to_string(&state).unwrap();
        assert!(json.contains("\"domain\":\"col\""));
        assert!(json.contains("\"quantity\":\"Concentration\""));
        assert!(json.contains("\"component\":0"));
    }
}
