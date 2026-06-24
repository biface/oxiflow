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
//! spatial domain. It is applied **after** the context calculators have run and
//! **before** `PhysicalModel::compute_physics` is called, so the full
//! [`ComputeContext`] is available when `apply` executes.
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
//! | `depends_on()` | `vec![]` | Ordering relative to other BCs (J3+) |
//! | `priority()` | `100` | Execution order when multiple BCs are registered |
//!
//! ## Execution order
//!
//! Multiple boundary conditions on the same domain are applied in ascending
//! `priority()` order. `depends_on()` is available but not consumed by the
//! engine at J2 — it is reserved for complex BC ordering at J3+.
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
/// - `apply` is called once per time step, after context calculators and before
///   `PhysicalModel::compute_physics`.
/// - `apply` must not modify `ctx` or `mesh` — only `state`.
/// - Variables declared in `required_variables()` are guaranteed to be present
///   in `ctx` when `apply` is called.
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
    /// Called once per time step, after all context calculators have run.
    /// Modifies `state` in-place to enforce the constraint.
    ///
    /// # Parameters
    ///
    /// - `state` — current field $u$ as a DOF vector; modified in-place.
    /// - `ctx`   — fully populated context (time, step, spatial quantities).
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
        let bc: Box<dyn BoundaryCondition> = Box::new(FixedInletBC { value: 3.14 });
        let mesh = make_mesh();
        let mut state = make_state(mesh.n_dof());
        let ctx = ComputeContext::new(0.0, 0.01);
        assert!(bc.apply(&mut state, &ctx, &mesh).is_ok());
        assert!((state[0] - 3.14).abs() < 1e-12);
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

    // ── Debug supertrait ──────────────────────────────────────────────────────

    #[test]
    fn debug_output_is_non_empty() {
        let bc: &dyn BoundaryCondition = &ZeroFluxBC;
        assert!(!format!("{:?}", bc).is_empty());
    }
}
