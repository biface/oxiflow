//! # Module `boundary`
//!
//! Boundary conditions — [`BoundaryCondition`] trait (DD-008, issue #34)
//! and built-in implementations (issue #35).
//!
//! ## Classification
//!
//! Two orthogonal enums characterise every boundary condition:
//!
//! | Enum | Axis | Purpose |
//! |---|---|---|
//! | [`BoundaryType`] | Mathematical | Nature of the constraint (Dirichlet, Neumann, Robin, Periodic) |
//! | [`BoundaryLocation`] | Geometric | Position in the domain (Inlet, Wall, Interface, …) |
//!
//! `boundary_type()` is required on the trait. `location()` is optional
//! (default `None`) — relevant for structured 1D cases but essential for
//! complex meshes, FEM, and multi-phase problems.
//!
//! ## Available implementations
//!
//! | Type | `BoundaryType` | `BoundaryLocation` |
//! |---|---|---|
//! | [`DanckwertsInlet`] | `Robin` | `Some(Inlet)` |
//! | [`DanckwertsOutlet`] | `Neumann` | `Some(Outlet)` |
//!
//! ## Role
//!
//! A boundary condition constrains the primary field $u$ at the boundary of a
//! spatial domain. At every derivative evaluation (DD-070, #200), the solver
//! runs the calculators providing the variables the conditions require, then
//! applies the conditions, then runs the remaining calculators and calls
//! `PhysicalModel::compute_physics`: the calculators that read the state (a
//! gradient, a Laplacian) see the constrained state.
//!
//! ## Constrained nodes (DD-070)
//!
//! A condition that sets node values declares them through
//! [`BoundaryCondition::constrained_nodes`]. These nodes are algebraic, not
//! differential: the solver sets the derivative to zero there, so neither an
//! integrator nor a Newton iteration moves them, and the solver drivers
//! re-apply every condition to the state at the end of each step and to the
//! initial state. Every saved state therefore satisfies the conditions,
//! including a value that depends on time or on neighbouring nodes.
//!
//! ## RequiresContext integration
//!
//! `BoundaryCondition` is a supertrait of [`RequiresContext`], giving every
//! boundary condition the full four-method interface:
//!
//! | Method | Default | Purpose |
//! |---|---|---|
//! | `required_variables()` | — (required) | Variables that must be present in `ctx` |
//! | `optional_variables()` | `vec![]` | Variables used if present |
//! | `depends_on()` | `vec![]` | Ordering relative to other BCs (reserved) |
//! | `priority()` | `100` | Execution order when multiple BCs are registered |
//!
//! ## Execution order
//!
//! Multiple boundary conditions on the same domain are applied in ascending
//! `priority()` order. `depends_on()` is available but not consumed by the
//! engine — it is reserved for complex BC ordering, with no milestone assigned.
//!
//! ## Object safety
//!
//! `BoundaryCondition` is object-safe. The solver stores BCs as
//! `Vec<Box<dyn BoundaryCondition>>` inside [`Domain`].
//!
//! [`ComputeContext`]: crate::context::ComputeContext
//! [`RequiresContext`]: crate::model::RequiresContext
//! [`Domain`]: crate::solver::scenario::Domain
//! [`DanckwertsInlet`]: crate::boundary::DanckwertsInlet
//! [`DanckwertsOutlet`]: crate::boundary::DanckwertsOutlet

use std::borrow::Cow;

use crate::context::compute::ComputeContext;
use crate::context::error::OxiflowError;
use crate::mesh::Mesh;
use crate::model::traits::RequiresContext;
use nalgebra::DVector;

pub mod danckwerts;
pub use danckwerts::{DanckwertsInlet, DanckwertsOutlet};

pub mod injection;
pub use injection::{dirac_initial_state, InjectionInlet, TemporalInjection};

// ── BoundaryType ──────────────────────────────────────────────────────────────

/// Mathematical classification of a boundary condition.
///
/// Describes the nature of the constraint imposed on the primary field $u$:
///
/// | Variant | Constraint | Example |
/// |---|---|---|
/// | `Dirichlet` | $u = g$ on $\partial\Omega$ | Fixed concentration at inlet |
/// | `Neumann` | $\nabla u \cdot \mathbf{n} = g$ | Zero flux at outlet |
/// | `Robin` | $\alpha u + \beta \nabla u \cdot \mathbf{n} = g$ | Danckwerts inlet |
/// | `Periodic` | $u(x_0) = u(x_L)$ | Periodic column simulation |
///
/// # Serialisation
///
/// Implements `Serialize` / `Deserialize` under the `serde` feature flag.
///
/// # Examples
///
/// ```rust
/// use oxiflow::boundary::BoundaryType;
///
/// assert_eq!(BoundaryType::Robin.clone(), BoundaryType::Robin);
/// assert_ne!(BoundaryType::Dirichlet, BoundaryType::Neumann);
/// ```
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum BoundaryType {
    /// Prescribed value: $u = g$ on the boundary.
    Dirichlet,
    /// Prescribed normal flux: $\nabla u \cdot \mathbf{n} = g$.
    ///
    /// `g = 0` gives the homogeneous Neumann (no-flux, insulation) condition.
    Neumann,
    /// Linear combination: $\alpha u + \beta \nabla u \cdot \mathbf{n} = g$.
    ///
    /// Generalises Dirichlet ($\beta = 0$) and Neumann ($\alpha = 0$).
    /// Used for Danckwerts inlet, convective heat transfer, impedance BCs.
    Robin,
    /// Periodic link: $u(x_0) = u(x_L)$ with matching normal flux.
    Periodic,
}

// ── BoundaryLocation ──────────────────────────────────────────────────────────

/// Geometric location of a boundary condition within the domain.
///
/// Describes *where* in the spatial domain the condition is applied. Optional
/// for simple 1D structured problems, but essential for complex meshes (FEM,
/// unstructured grids), multi-phase flows, and inter-domain coupling.
///
/// # Extensibility
///
/// `#[non_exhaustive]` allows future variants at minor versions without
/// breaking downstream code. The `Custom` variant covers user-defined locations
/// with a `Cow<'static, str>` key, consistent with
/// [`ContextVariable::External`](crate::context::variable::ContextVariable::External).
///
/// # Serialisation
///
/// Implements `Serialize` / `Deserialize` under the `serde` feature flag.
///
/// # Examples
///
/// ```rust
/// use oxiflow::boundary::BoundaryLocation;
///
/// let loc = BoundaryLocation::Inlet;
/// assert_ne!(loc, BoundaryLocation::Outlet);
///
/// let custom = BoundaryLocation::Custom("membrane_left".into());
/// assert!(matches!(custom, BoundaryLocation::Custom(_)));
/// ```
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum BoundaryLocation {
    /// Upstream entry — convective flux enters here.
    Inlet,
    /// Downstream exit — convective flux leaves here.
    Outlet,
    /// Solid wall — no-slip or no-flux depending on the physics.
    Wall,
    /// Symmetry plane — zero normal gradient and zero normal velocity.
    Symmetry,
    /// Periodic face — linked to another `Periodic` face of the domain.
    Periodic,
    /// Interface between two distinct materials or porous media.
    ///
    /// Used for heterogeneous columns, composite materials, soil layers.
    Interface,
    /// Phase interface — boundary between two thermodynamic phases.
    ///
    /// Relevant for Stefan problems, solidification fronts, boiling,
    /// condensation. The interface position may evolve during the simulation.
    PhaseInterface,
    /// Coupling interface between two domains (INV-3, J3+).
    ///
    /// Managed by `CouplingOperator` — reserved from v0.3.0.
    CouplingInterface,
    /// User-defined location not covered by built-in variants.
    Custom(Cow<'static, str>),
}

// ── BoundaryCondition ─────────────────────────────────────────────────────────

/// Constrains the primary field $u$ at the boundary of a spatial domain.
///
/// A boundary condition receives the current field state and the fully populated
/// [`ComputeContext`], then modifies `state` in-place to enforce the constraint.
///
/// # Contract
///
/// - `apply` is called at every derivative evaluation, after the calculators
///   providing the variables declared in `required_variables()` and before
///   the other calculators and `PhysicalModel::compute_physics` (DD-070). The
///   solver drivers also call it on the initial state and at the end of each
///   step.
/// - `apply` must not modify `ctx` or `mesh` — only `state`.
/// - Variables declared in `required_variables()` are guaranteed to be present
///   in `ctx` when `apply` is called. They are computed on the state *before*
///   the conditions are applied, so a condition should only require variables
///   that do not depend on the state (time, a feed concentration): a gradient
///   read there would come from the unconstrained state.
/// - `apply` only modifies the nodes returned by
///   [`constrained_nodes`](Self::constrained_nodes), if it declares any.
///
/// # State variants
///
/// `apply` receives a single field, so it applies to `ScalarField` states
/// only: a domain whose state is a `VectorField` and that has boundary
/// conditions attached fails with `OxiflowError::TypeMismatch` at the first
/// derivative evaluation. The boundaries of a `VectorField` state are
/// closed at the operator level instead, through
/// [`Self::ghost_value`] (as in the `langmuir_multi` and `lahar_multifield`
/// examples). Conditions applied component by component belong with the
/// mixed finite-element spaces (#150).
///
/// # Classification
///
/// Every implementation must declare its mathematical nature via `boundary_type()`.
/// The geometric location may be declared via `location()` (default `None`) —
/// essential for complex meshes, FEM, and multi-phase problems.
///
/// # Serialisation
///
/// `BoundaryCondition` does not implement `serde::Serialize` / `serde::Deserialize`.
/// Concrete implementations may hold arbitrary state that cannot be serialised
/// generically. If persistence is required, provide a dedicated configuration
/// type and reconstruct the BC from it — see `SimulationSnapshot` (DD-025).
///
/// # Examples
///
/// ```rust
/// use oxiflow::boundary::{BoundaryCondition, BoundaryType};
/// use oxiflow::context::compute::ComputeContext;
/// use oxiflow::context::error::OxiflowError;
/// use oxiflow::context::variable::ContextVariable;
/// use oxiflow::mesh::{Mesh, UniformGrid1D};
/// use oxiflow::model::traits::RequiresContext;
/// use nalgebra::DVector;
///
/// #[derive(Debug)]
/// struct ZeroFluxBC;
///
/// impl RequiresContext for ZeroFluxBC {
///     fn required_variables(&self) -> Vec<ContextVariable> { vec![] }
/// }
///
/// impl BoundaryCondition for ZeroFluxBC {
///     fn boundary_type(&self) -> BoundaryType { BoundaryType::Neumann }
///
///     fn apply(
///         &self,
///         _state: &mut DVector<f64>,
///         _ctx: &ComputeContext,
///         _mesh: &dyn Mesh,
///     ) -> Result<(), OxiflowError> { Ok(()) }
/// }
///
/// let bc = ZeroFluxBC;
/// assert_eq!(bc.boundary_type(), BoundaryType::Neumann);
/// assert!(bc.location().is_none());
/// ```
///
/// [`RequiresContext`]: crate::model::RequiresContext
/// [`ComputeContext`]: crate::context::ComputeContext
///
/// # `Send + Sync` (DD-037)
///
/// Added alongside DD-037 (#45): [`crate::solver::methods::imex::SplitOperator`]
/// is the first place in the engine where a `Domain` is *owned* by a type
/// that must itself be `Send + Sync` (`Solver: Send + Sync`) rather than
/// only borrowed as `&Domain`. Without this bound, `Box<dyn BoundaryCondition>`
/// — and therefore `Domain` — would not be `Send + Sync`, and
/// `OperatorSplittingSolver` could not implement `Solver`. Both concrete
/// implementations (`DanckwertsInlet`, `DanckwertsOutlet`) already satisfy
/// this trivially (plain `f64`/`ContextVariable` fields or no fields at
/// all) — non-breaking in practice, though technically a new bound on the
/// trait for any future external implementor.
pub trait BoundaryCondition: RequiresContext + std::fmt::Debug + Send + Sync {
    /// Returns the mathematical type of this boundary condition.
    ///
    /// Required — every implementation must declare whether it enforces a
    /// Dirichlet, Neumann, Robin, or Periodic constraint.
    fn boundary_type(&self) -> BoundaryType;

    /// Returns the geometric location of this boundary condition, if known.
    ///
    /// Optional — defaults to `None` for simple or location-agnostic BCs.
    /// Override for complex meshes, FEM, and multi-phase problems where the
    /// location carries semantic meaning for the solver or post-processor.
    fn location(&self) -> Option<BoundaryLocation> {
        None
    }

    /// Applies the boundary condition to `state`.
    ///
    /// Called at every derivative evaluation and by the solver drivers (see
    /// the *Contract* section). Modifies `state` in-place to enforce the
    /// constraint.
    ///
    /// # Parameters
    ///
    /// - `state` — current field $u$ as a DOF vector; modified in-place.
    /// - `ctx`   — context holding the time, the time step and the variables
    ///   declared in `required_variables()`.
    /// - `mesh`  — spatial discretisation of the domain.
    ///
    /// # Errors
    ///
    /// Returns [`OxiflowError`] if the constraint cannot be applied.
    fn apply(
        &self,
        state: &mut DVector<f64>,
        ctx: &ComputeContext,
        mesh: &dyn Mesh,
    ) -> Result<(), OxiflowError>;

    /// Indices of the nodes [`apply`](Self::apply) sets, for `mesh` (DD-070,
    /// #200).
    ///
    /// The solver treats these nodes as algebraic: it sets the derivative to
    /// zero there at every evaluation, so that the integrators and the Newton
    /// iterations of the implicit methods leave them alone (their rows of the
    /// implicit system become identity rows), and the condition alone fixes
    /// their value, within a step as well as at its end.
    ///
    /// Defaults to an empty list: a condition that declares nothing keeps the
    /// derivative the model gives its nodes. The solver drivers still restore
    /// it at the end of each step, but its nodes move within the step: a
    /// direct call to `SteppableSolver::step` returns them moved, and the
    /// error estimate of an adaptive method or the residual of a Newton
    /// iteration includes that movement. Return only nodes `apply` actually
    /// sets — a declared node that `apply` leaves alone would keep its
    /// initial value.
    ///
    /// # Errors reported by the solver
    ///
    /// An index outside the state is reported as
    /// `OxiflowError::PreconditionFailed` at the first derivative evaluation.
    fn constrained_nodes(&self, mesh: &dyn Mesh) -> Vec<usize> {
        let _ = mesh;
        Vec::new()
    }

    /// Ghost-cell value for this boundary condition, at reflection depth
    /// `depth` (DD-042, amendment 1).
    ///
    /// `FluxBoundary::GhostCell` (`operators::fv`, `operators::weno`) needs
    /// values *outside* the real domain to close its face-flux computation
    /// at the boundary — something `apply()` above does not provide, since
    /// it only constrains real DOFs. `depth` counts ghost layers outward
    /// from the boundary (`1` = nearest, `2` = next, ...) — `operators::fv`'s
    /// 2-point stencil only ever needs `depth = 1`; `operators::weno`'s
    /// wider stencils need up to `2` (`WENO3`) or `3` (`WENO5`) layers
    /// depending on the upwind direction selected at the call site.
    ///
    /// `interior_at_depth` is the real interior value at the *symmetric*
    /// depth — the method-of-images convention: ghost layer `k` mirrors
    /// interior node `k−1` (so `depth = 1` pairs with the nearest interior
    /// neighbor, `depth = 2` with the next one, etc.). Each layer is derived
    /// directly from this real interior value, not from a previously
    /// computed ghost layer — deliberately, so a Robin condition (e.g.
    /// [`DanckwertsInlet`]) stays exact at every depth instead of eroding
    /// through repeated extrapolation. A chained scheme (each ghost derived
    /// from the previous one) was considered and rejected for exactly this
    /// reason — see DD-042, amendment 1.
    ///
    /// Defaults to `None` — non-breaking for every existing or external
    /// implementation. A generic Dirichlet/Neumann formula is deliberately
    /// *not* supplied here as a fallback: a Robin condition has no generic
    /// ghost value independent of its own α/β coefficients, and silently
    /// approximating one for it would be wrong, not merely imprecise — see
    /// DD-042 for the two concrete consumers (lahar-lake, chromatography)
    /// that motivated this method and why a generic-by-`BoundaryType`
    /// fallback was rejected in favor of per-implementation exactness.
    ///
    /// A `FluxBoundary::GhostCell` referencing a BC that does not override
    /// this method (still returns `None`), or that cannot supply the
    /// `depth` a caller needs, must fail explicitly
    /// (`OxiflowError::PreconditionFailed`) rather than substitute an
    /// approximation of its own.
    ///
    /// Takes `ctx` (added alongside [`InjectionInlet`]) so a ghost value can
    /// depend on simulation time — the two motivating cases at DD-042
    /// (lahar-lake, Danckwerts) were both steady-state, so the original
    /// signature had no way to express a time-varying boundary; a
    /// competitive-adsorption inlet (chromatography) needs exactly that
    /// (an injection profile evaluated at `ctx.time()`), not a
    /// per-implementation special case bolted on separately.
    fn ghost_value(
        &self,
        depth: usize,
        interior_at_depth: f64,
        dx: f64,
        ctx: &ComputeContext,
    ) -> Option<f64> {
        let _ = (depth, interior_at_depth, dx, ctx);
        None
    }
}

// ── Test condition ────────────────────────────────────────────────────────────

/// Test condition setting some nodes to a value that may depend on time,
/// declared as constrained nodes (DD-070) or not, and counting its
/// applications. Shared by the solver tests.
#[cfg(test)]
pub(crate) struct PinnedNodes {
    nodes: Vec<usize>,
    value: Box<dyn Fn(f64) -> f64 + Send + Sync>,
    declared: Vec<usize>,
    calls: std::sync::Arc<std::sync::atomic::AtomicUsize>,
}

#[cfg(test)]
impl PinnedNodes {
    /// Sets `nodes` to `value(t)` without declaring them, as a condition
    /// written before DD-070.
    pub(crate) fn new(nodes: &[usize], value: impl Fn(f64) -> f64 + Send + Sync + 'static) -> Self {
        Self {
            nodes: nodes.to_vec(),
            value: Box::new(value),
            declared: Vec::new(),
            calls: Default::default(),
        }
    }

    /// Declares the pinned nodes through `constrained_nodes`.
    pub(crate) fn declared(self) -> Self {
        let nodes = self.nodes.clone();
        self.declaring(&nodes)
    }

    /// Declares `nodes` through `constrained_nodes`, whatever `apply` sets —
    /// for the tests of an inconsistent declaration.
    pub(crate) fn declaring(mut self, nodes: &[usize]) -> Self {
        self.declared = nodes.to_vec();
        self
    }

    /// Counter of the calls to `apply`.
    pub(crate) fn calls(&self) -> std::sync::Arc<std::sync::atomic::AtomicUsize> {
        std::sync::Arc::clone(&self.calls)
    }
}

#[cfg(test)]
impl std::fmt::Debug for PinnedNodes {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PinnedNodes")
            .field("nodes", &self.nodes)
            .field("declared", &self.declared)
            .finish()
    }
}

#[cfg(test)]
impl RequiresContext for PinnedNodes {
    fn required_variables(&self) -> Vec<crate::context::variable::ContextVariable> {
        vec![]
    }
}

#[cfg(test)]
impl BoundaryCondition for PinnedNodes {
    fn boundary_type(&self) -> BoundaryType {
        BoundaryType::Dirichlet
    }

    fn apply(
        &self,
        state: &mut DVector<f64>,
        ctx: &ComputeContext,
        _mesh: &dyn Mesh,
    ) -> Result<(), OxiflowError> {
        let value = (self.value)(ctx.time());
        for &i in &self.nodes {
            state[i] = value;
        }
        self.calls.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        Ok(())
    }

    fn constrained_nodes(&self, _mesh: &dyn Mesh) -> Vec<usize> {
        self.declared.clone()
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::context::variable::ContextVariable;
    use crate::mesh::structured::UniformGrid1D;

    // ── Fixtures ──────────────────────────────────────────────────────────────

    #[derive(Debug)]
    struct ZeroFluxBC;

    impl RequiresContext for ZeroFluxBC {
        fn required_variables(&self) -> Vec<ContextVariable> {
            vec![]
        }
    }

    impl BoundaryCondition for ZeroFluxBC {
        fn boundary_type(&self) -> BoundaryType {
            BoundaryType::Neumann
        }
        fn apply(
            &self,
            _state: &mut DVector<f64>,
            _ctx: &ComputeContext,
            _mesh: &dyn Mesh,
        ) -> Result<(), OxiflowError> {
            Ok(())
        }
    }

    #[derive(Debug)]
    struct FixedInletBC {
        value: f64,
    }

    impl RequiresContext for FixedInletBC {
        fn required_variables(&self) -> Vec<ContextVariable> {
            vec![]
        }
        fn priority(&self) -> u32 {
            50
        }
    }

    impl BoundaryCondition for FixedInletBC {
        fn boundary_type(&self) -> BoundaryType {
            BoundaryType::Dirichlet
        }
        fn location(&self) -> Option<BoundaryLocation> {
            Some(BoundaryLocation::Inlet)
        }
        fn apply(
            &self,
            state: &mut DVector<f64>,
            _ctx: &ComputeContext,
            _mesh: &dyn Mesh,
        ) -> Result<(), OxiflowError> {
            if !state.is_empty() {
                state[0] = self.value;
            }
            Ok(())
        }
    }

    #[derive(Debug)]
    struct TimeDependentBC;

    impl RequiresContext for TimeDependentBC {
        fn required_variables(&self) -> Vec<ContextVariable> {
            vec![ContextVariable::Time]
        }
    }

    impl BoundaryCondition for TimeDependentBC {
        fn boundary_type(&self) -> BoundaryType {
            BoundaryType::Dirichlet
        }
        fn apply(
            &self,
            state: &mut DVector<f64>,
            ctx: &ComputeContext,
            _mesh: &dyn Mesh,
        ) -> Result<(), OxiflowError> {
            if !state.is_empty() {
                state[0] = ctx.time();
            }
            Ok(())
        }
    }

    fn make_mesh() -> UniformGrid1D {
        UniformGrid1D::new(5, 0.0, 1.0).unwrap()
    }

    fn make_state(n: usize) -> DVector<f64> {
        DVector::from_element(n, 0.0)
    }

    // ── BoundaryType ──────────────────────────────────────────────────────────

    #[test]
    fn boundary_type_variants_are_distinct() {
        assert_ne!(BoundaryType::Dirichlet, BoundaryType::Neumann);
        assert_ne!(BoundaryType::Neumann, BoundaryType::Robin);
        assert_ne!(BoundaryType::Robin, BoundaryType::Periodic);
    }

    #[test]
    fn boundary_type_clone_preserves_equality() {
        assert_eq!(BoundaryType::Robin.clone(), BoundaryType::Robin);
    }

    #[test]
    fn boundary_type_debug_non_empty() {
        for t in [
            BoundaryType::Dirichlet,
            BoundaryType::Neumann,
            BoundaryType::Robin,
            BoundaryType::Periodic,
        ] {
            assert!(!format!("{:?}", t).is_empty());
        }
    }

    // ── BoundaryLocation ──────────────────────────────────────────────────────

    #[test]
    fn boundary_location_variants_are_distinct() {
        assert_ne!(BoundaryLocation::Inlet, BoundaryLocation::Outlet);
        assert_ne!(BoundaryLocation::Wall, BoundaryLocation::Symmetry);
        assert_ne!(
            BoundaryLocation::Interface,
            BoundaryLocation::PhaseInterface
        );
        assert_ne!(
            BoundaryLocation::PhaseInterface,
            BoundaryLocation::CouplingInterface,
        );
    }

    #[test]
    fn boundary_location_custom_equality_by_name() {
        let a = BoundaryLocation::Custom("membrane".into());
        let b = BoundaryLocation::Custom("membrane".into());
        let c = BoundaryLocation::Custom("wall_left".into());
        assert_eq!(a, b);
        assert_ne!(a, c);
    }

    #[test]
    fn boundary_location_clone_preserves_equality() {
        let loc = BoundaryLocation::Custom("top".into());
        assert_eq!(loc.clone(), loc);
    }

    #[test]
    fn boundary_location_debug_non_empty() {
        let locations = [
            BoundaryLocation::Inlet,
            BoundaryLocation::Outlet,
            BoundaryLocation::Wall,
            BoundaryLocation::Symmetry,
            BoundaryLocation::Periodic,
            BoundaryLocation::Interface,
            BoundaryLocation::PhaseInterface,
            BoundaryLocation::CouplingInterface,
            BoundaryLocation::Custom("test".into()),
        ];
        for loc in &locations {
            assert!(!format!("{:?}", loc).is_empty());
        }
    }

    // ── Object safety ─────────────────────────────────────────────────────────

    #[test]
    fn trait_is_object_safe() {
        let bcs: Vec<Box<dyn BoundaryCondition>> =
            vec![Box::new(ZeroFluxBC), Box::new(FixedInletBC { value: 1.0 })];
        assert_eq!(bcs.len(), 2);
    }

    #[test]
    fn boxed_bc_can_be_applied() {
        let bc: Box<dyn BoundaryCondition> = Box::new(FixedInletBC {
            value: std::f64::consts::PI,
        });
        let mesh = make_mesh();
        let mut state = make_state(mesh.n_dof());
        let ctx = ComputeContext::new(0.0, 0.01);
        assert!(bc.apply(&mut state, &ctx, &mesh).is_ok());
        assert!((state[0] - std::f64::consts::PI).abs() < 1e-12);
    }

    // ── boundary_type() ───────────────────────────────────────────────────────

    #[test]
    fn boundary_type_returned_correctly() {
        assert_eq!(ZeroFluxBC.boundary_type(), BoundaryType::Neumann);
        assert_eq!(
            FixedInletBC { value: 0.0 }.boundary_type(),
            BoundaryType::Dirichlet
        );
        assert_eq!(TimeDependentBC.boundary_type(), BoundaryType::Dirichlet);
    }

    #[test]
    fn boundary_type_accessible_via_dyn() {
        let bc: &dyn BoundaryCondition = &ZeroFluxBC;
        assert_eq!(bc.boundary_type(), BoundaryType::Neumann);
    }

    // ── location() ────────────────────────────────────────────────────────────

    #[test]
    fn location_default_is_none() {
        assert!(ZeroFluxBC.location().is_none());
        assert!(TimeDependentBC.location().is_none());
    }

    #[test]
    fn location_override_returned_correctly() {
        assert_eq!(
            FixedInletBC { value: 0.0 }.location(),
            Some(BoundaryLocation::Inlet)
        );
    }

    #[test]
    fn location_accessible_via_dyn() {
        let bc: &dyn BoundaryCondition = &FixedInletBC { value: 0.0 };
        assert_eq!(bc.location(), Some(BoundaryLocation::Inlet));
    }

    // ── RequiresContext supertrait ─────────────────────────────────────────────

    #[test]
    fn required_variables_forwarded_via_dyn() {
        let bc: &dyn BoundaryCondition = &TimeDependentBC;
        assert!(bc.required_variables().contains(&ContextVariable::Time));
    }

    #[test]
    fn optional_variables_default_is_empty() {
        assert!(ZeroFluxBC.optional_variables().is_empty());
    }

    #[test]
    fn depends_on_default_is_empty() {
        assert!(ZeroFluxBC.depends_on().is_empty());
    }

    #[test]
    fn priority_default_is_100() {
        assert_eq!(ZeroFluxBC.priority(), 100);
    }

    #[test]
    fn priority_can_be_overridden() {
        assert_eq!(FixedInletBC { value: 0.0 }.priority(), 50);
    }

    // ── apply() ───────────────────────────────────────────────────────────────

    #[test]
    fn zero_flux_bc_does_not_modify_state() {
        let mesh = make_mesh();
        let mut state = DVector::from_element(mesh.n_dof(), 2.0);
        let ctx = ComputeContext::new(0.0, 0.01);
        ZeroFluxBC.apply(&mut state, &ctx, &mesh).unwrap();
        assert!(state.iter().all(|&v| (v - 2.0).abs() < 1e-12));
    }

    #[test]
    fn fixed_inlet_bc_sets_first_dof() {
        let mesh = make_mesh();
        let mut state = make_state(mesh.n_dof());
        let ctx = ComputeContext::new(0.0, 0.01);
        FixedInletBC { value: 42.0 }
            .apply(&mut state, &ctx, &mesh)
            .unwrap();
        assert!((state[0] - 42.0).abs() < 1e-12);
        assert!(state.iter().skip(1).all(|&v| v == 0.0));
    }

    #[test]
    fn time_dependent_bc_reads_context_time() {
        let mesh = make_mesh();
        let mut state = make_state(mesh.n_dof());
        let ctx = ComputeContext::new(3.5, 0.01);
        TimeDependentBC.apply(&mut state, &ctx, &mesh).unwrap();
        assert!((state[0] - 3.5).abs() < 1e-12);
    }

    // ── ghost_value() (DD-042) ─────────────────────────────────────────────

    #[test]
    fn ghost_value_defaults_to_none() {
        let ctx = ComputeContext::new(0.0, 0.01);
        assert_eq!(ZeroFluxBC.ghost_value(1, 2.0, 0.1, &ctx), None);
        assert_eq!(
            FixedInletBC { value: 0.0 }.ghost_value(2, 2.0, 0.1, &ctx),
            None
        );
    }

    // ── Debug supertrait ──────────────────────────────────────────────────────

    #[test]
    fn debug_output_is_non_empty() {
        let bc: &dyn BoundaryCondition = &ZeroFluxBC;
        assert!(!format!("{:?}", bc).is_empty());
    }
}
