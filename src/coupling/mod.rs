//! # Module `coupling`
//!
//! Multi-physics inter-domain coupling — INV-3 invariant ([DA-011](https://github.com/biface/oxiflow/issues/11)).
//!
//! ## Core principle (INV-3)
//!
//! All interactions between distinct physical domains (lahar/lake, fluid/solid,
//! column/reservoir) must go through the [`CouplingOperator`] trait with
//! [`DomainId`] and [`Interface`]. No coupling logic is coded directly in the
//! [`Solver`](crate::solver::Solver) ([DA-011](https://github.com/biface/oxiflow/issues/11), J3).
//!
//! ## Types
//!
//! | Type | Role |
//! |------|------|
//! | [`Interface`] | Describes the boundary shared by two domains |
//! | [`CouplingContext`] | Context of a coupling — one `ComputeContext` per side of its `Interface` |
//! | [`CouplingOperator`] | Trait — exchanges state across an `Interface` |
//!
//! ## Object safety
//!
//! `CouplingOperator` is object-safe to support INV-4 (plugin-safe API, v0.30.0 / J30, [DA-017](https://github.com/biface/oxiflow/issues/17)).
//! It can be stored as `Box<dyn CouplingOperator>` in `Scenario` at J3.

use crate::context::compute::ComputeContext;
use crate::context::error::OxiflowError;
use crate::context::state::MultiDomainState;
use crate::model::traits::RequiresContext;
use crate::solver::scenario::DomainId;

// ── Interface ─────────────────────────────────────────────────────────────────

/// Boundary shared between two physical domains.
///
/// An `Interface` identifies the two domains it connects and carries an
/// optional label for diagnostics and serialisation. It does not encode any
/// geometry — spatial discretisation of the interface is the responsibility
/// of the concrete [`CouplingOperator`] implementation.
///
/// # Examples
///
/// ```rust
/// use oxiflow::coupling::Interface;
/// use oxiflow::solver::scenario::DomainId;
///
/// let iface = Interface::new(
///     DomainId::new("lahar"),
///     DomainId::new("lake"),
/// );
/// assert_eq!(iface.source().as_str(), "lahar");
/// assert_eq!(iface.target().as_str(), "lake");
/// assert!(iface.label().is_none());
///
/// let labelled = Interface::new(
///     DomainId::new("column"),
///     DomainId::new("reservoir"),
/// ).with_label("column-reservoir");
/// assert_eq!(labelled.label(), Some("column-reservoir"));
/// ```
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Interface {
    /// Source domain — the domain whose state is read by the coupling.
    source: DomainId,
    /// Target domain — the domain whose state is written by the coupling.
    target: DomainId,
    /// Optional human-readable label for diagnostics.
    label: Option<String>,
}

impl Interface {
    /// Creates an interface between `source` and `target` with no label.
    pub fn new(source: DomainId, target: DomainId) -> Self {
        Self {
            source,
            target,
            label: None,
        }
    }

    /// Attaches a label to this interface.
    pub fn with_label(mut self, label: impl Into<String>) -> Self {
        self.label = Some(label.into());
        self
    }

    /// Returns the source domain identifier.
    pub fn source(&self) -> &DomainId {
        &self.source
    }

    /// Returns the target domain identifier.
    pub fn target(&self) -> &DomainId {
        &self.target
    }

    /// Returns the optional label, if any.
    pub fn label(&self) -> Option<&str> {
        self.label.as_deref()
    }
}

impl std::fmt::Display for Interface {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match &self.label {
            Some(l) => write!(f, "Interface({} → {} [{}])", self.source, self.target, l),
            None => write!(f, "Interface({} → {})", self.source, self.target),
        }
    }
}

// ── CouplingContext ───────────────────────────────────────────────────────────

/// Context a [`CouplingOperator`] receives: one [`ComputeContext`] per side
/// of its [`Interface`] ([DA-011](https://github.com/biface/oxiflow/issues/11)
/// amendment 1, [#231](https://github.com/biface/oxiflow/issues/231)).
///
/// A calculator is evaluated on one domain's state, and a coupling reads
/// two domains: [`source`](Self::source) holds the values computed on the
/// source domain's state, [`target`](Self::target) those computed on the
/// target domain's state. The orchestrator fills each side with the
/// variables the coupling requires (and the calculators these depend on),
/// at the time of the exchange; a coupling with no requirement gets two
/// contexts without calculator values.
///
/// Both sides share the time and the time step, read through
/// [`time`](Self::time) and [`time_step`](Self::time_step).
///
/// # Examples
///
/// ```rust
/// use nalgebra::DVector;
/// use oxiflow::context::{ContextValue, ContextVariable};
/// use oxiflow::coupling::CouplingContext;
///
/// let slope = ContextVariable::SpatialGradient { dimension: 0, component: None };
/// let mut ctx = CouplingContext::new(2.0, 0.1);
/// ctx.source_mut().insert(
///     slope.clone(),
///     ContextValue::ScalarField(DVector::from_vec(vec![1.0, 1.0])),
/// );
/// assert_eq!(ctx.time_step(), 0.1);
/// assert_eq!(ctx.source().gradient(0)?[0], 1.0);
/// assert!(ctx.target().try_get(slope).is_none());
/// # Ok::<(), oxiflow::context::OxiflowError>(())
/// ```
pub struct CouplingContext {
    /// Values computed on the source domain's state.
    source: ComputeContext,
    /// Values computed on the target domain's state.
    target: ComputeContext,
}

impl CouplingContext {
    /// A context at `time` with step `time_step`, holding no calculator
    /// value on either side.
    ///
    /// Values are added through [`source_mut`](Self::source_mut) and
    /// [`target_mut`](Self::target_mut): this is how a unit test of a
    /// coupling builds the context the orchestrator would pass.
    pub fn new(time: f64, time_step: f64) -> Self {
        Self {
            source: ComputeContext::new(time, time_step),
            target: ComputeContext::new(time, time_step),
        }
    }

    /// Assembles the two sides the orchestrator evaluated, both at the same
    /// time and step.
    pub(crate) fn from_sides(source: ComputeContext, target: ComputeContext) -> Self {
        debug_assert_eq!(source.time(), target.time());
        debug_assert_eq!(source.time_step(), target.time_step());
        Self { source, target }
    }

    /// Time of the exchange.
    pub fn time(&self) -> f64 {
        self.source.time()
    }

    /// Time step of the exchange.
    pub fn time_step(&self) -> f64 {
        self.source.time_step()
    }

    /// Values computed on the source domain's state.
    pub fn source(&self) -> &ComputeContext {
        &self.source
    }

    /// Values computed on the target domain's state.
    pub fn target(&self) -> &ComputeContext {
        &self.target
    }

    /// Source side, to insert values in a unit test of a coupling.
    pub fn source_mut(&mut self) -> &mut ComputeContext {
        &mut self.source
    }

    /// Target side, to insert values in a unit test of a coupling.
    pub fn target_mut(&mut self) -> &mut ComputeContext {
        &mut self.target
    }
}

// ── CouplingOperator ──────────────────────────────────────────────────────────

/// Exchanges physical state between two domains across an [`Interface`].
///
/// Implementors read from one or more `(DomainId, PhysicalQuantity)` entries
/// in `states` and return an updated [`MultiDomainState`] with the coupled
/// values written to the target domain. The operator must not modify source
/// domain entries.
///
/// The variables the operator declares through [`RequiresContext`] reach
/// it in [`CouplingContext`], computed on each side of the interface.
///
/// # INV-3 contract
///
/// All inter-domain coupling must go through this trait. No coupling logic
/// may be coded directly in [`Solver`](crate::solver::Solver) implementations.
///
/// # Object safety
///
/// This trait is object-safe — it can be used as `Box<dyn CouplingOperator>`.
///
/// # Examples
///
/// ```rust
/// use oxiflow::coupling::{CouplingContext, CouplingOperator, Interface};
/// use oxiflow::context::state::MultiDomainState;
/// use oxiflow::context::error::OxiflowError;
/// use oxiflow::context::variable::ContextVariable;
/// use oxiflow::model::traits::RequiresContext;
/// use oxiflow::solver::DomainId;
///
/// struct MassTransfer { interface: Interface }
///
/// impl RequiresContext for MassTransfer {
///     fn required_variables(&self) -> Vec<ContextVariable> { vec![] }
/// }
///
/// impl CouplingOperator for MassTransfer {
///     fn apply(
///         &self,
///         states: &MultiDomainState,
///         _ctx: &CouplingContext,
///         _interface: &Interface,
///     ) -> Result<MultiDomainState, OxiflowError> {
///         Ok(states.clone())   // stub — real impl transfers mass
///     }
///     fn interface(&self) -> &Interface { &self.interface }
/// }
///
/// let coupling: Box<dyn CouplingOperator> = Box::new(MassTransfer {
///     interface: Interface::new(DomainId::new("lahar"), DomainId::new("lake")),
/// });
/// assert_eq!(coupling.interface().source(), &DomainId::new("lahar"));
/// ```
pub trait CouplingOperator: RequiresContext + Send + Sync {
    /// Applies the coupling and returns the updated multi-domain state.
    ///
    /// The returned `MultiDomainState` contains the target domain entries
    /// with coupled values. Implementations must not alter source domain
    /// entries. `ctx` holds, for each side of `interface`, the variables
    /// the operator requires.
    fn apply(
        &self,
        states: &MultiDomainState,
        ctx: &CouplingContext,
        interface: &Interface,
    ) -> Result<MultiDomainState, OxiflowError>;

    /// Returns the interface this operator acts on.
    fn interface(&self) -> &Interface;
}

// ── Test coupling ─────────────────────────────────────────────────────────────
//
// Shared by the unit tests of the crate that need a coupling but not an
// exchange. Defined once here rather than in every test module.

/// Coupling that returns the state unchanged, with no context requirement.
#[cfg(test)]
pub(crate) struct PassthroughCoupling {
    pub(crate) interface: Interface,
}

#[cfg(test)]
impl RequiresContext for PassthroughCoupling {
    fn required_variables(&self) -> Vec<crate::context::variable::ContextVariable> {
        vec![]
    }
}

#[cfg(test)]
impl CouplingOperator for PassthroughCoupling {
    fn apply(
        &self,
        states: &MultiDomainState,
        _ctx: &CouplingContext,
        _interface: &Interface,
    ) -> Result<MultiDomainState, OxiflowError> {
        Ok(states.clone())
    }
    fn interface(&self) -> &Interface {
        &self.interface
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn lahar() -> DomainId {
        DomainId::new("lahar")
    }
    fn lake() -> DomainId {
        DomainId::new("lake")
    }

    // ── Interface construction ────────────────────────────────────────────────

    #[test]
    fn interface_new() {
        let iface = Interface::new(lahar(), lake());
        assert_eq!(iface.source(), &lahar());
        assert_eq!(iface.target(), &lake());
        assert!(iface.label().is_none());
    }

    #[test]
    fn interface_with_label() {
        let iface = Interface::new(lahar(), lake()).with_label("lahar-lake");
        assert_eq!(iface.label(), Some("lahar-lake"));
    }

    #[test]
    fn interface_display_no_label() {
        let iface = Interface::new(lahar(), lake());
        assert_eq!(format!("{}", iface), "Interface(lahar → lake)");
    }

    #[test]
    fn interface_display_with_label() {
        let iface = Interface::new(lahar(), lake()).with_label("lahar-lake");
        assert_eq!(format!("{}", iface), "Interface(lahar → lake [lahar-lake])");
    }

    #[test]
    fn interface_clone_equals_original() {
        let iface = Interface::new(lahar(), lake()).with_label("test");
        assert_eq!(iface.clone(), iface);
    }

    // ── Object safety ─────────────────────────────────────────────────────────

    #[test]
    fn coupling_operator_is_object_safe() {
        fn assert_object_safe<T: CouplingOperator + ?Sized>() {}
        assert_object_safe::<dyn CouplingOperator>();
    }

    // ── Stub implementation ───────────────────────────────────────────────────

    use crate::context::quantity::PhysicalQuantity;
    use crate::context::value::ContextValue;
    use nalgebra::DVector;

    #[test]
    fn passthrough_coupling_returns_unchanged_state() {
        let iface = Interface::new(lahar(), lake());
        let op = PassthroughCoupling {
            interface: iface.clone(),
        };

        let mut states = MultiDomainState::new();
        states.set(
            lahar(),
            PhysicalQuantity::concentration(),
            ContextValue::ScalarField(DVector::from_vec(vec![1.0, 2.0])),
        );

        let ctx = CouplingContext::new(0.0, 0.1);
        let result = op.apply(&states, &ctx, &iface).unwrap();

        assert!(result
            .get(&lahar(), &PhysicalQuantity::concentration())
            .is_some());
    }

    #[test]
    fn coupling_operator_interface_accessor() {
        let iface = Interface::new(lahar(), lake());
        let op = PassthroughCoupling {
            interface: iface.clone(),
        };
        assert_eq!(op.interface(), &iface);
    }
}
