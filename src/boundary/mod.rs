//! # Module `boundary`
//!
//! Boundary conditions — [`BoundaryCondition`] trait (DD-008, issue #34).
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
//! The solver aggregates `required_variables()` from all boundary conditions into
//! the global requirements list, validated by `build_calculator_chain` before
//! the first time step.
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

use crate::context::compute::ComputeContext;
use crate::context::error::OxiflowError;
use crate::mesh::Mesh;
use crate::model::traits::RequiresContext;
use nalgebra::DVector;

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
/// # Execution order
///
/// Multiple BCs on the same domain are applied in ascending `priority()` order
/// (inherited from [`RequiresContext`]). `depends_on()` is reserved for J3+.
///
/// # Examples
///
/// ```rust
/// use oxiflow::boundary::BoundaryCondition;
/// use oxiflow::context::compute::ComputeContext;
/// use oxiflow::context::error::OxiflowError;
/// use oxiflow::context::variable::ContextVariable;
/// use oxiflow::mesh::{Mesh, UniformGrid1D};
/// use oxiflow::model::traits::RequiresContext;
/// use nalgebra::DVector;
///
/// /// Homogeneous Neumann BC — zero flux at both ends.
/// #[derive(Debug)]
/// struct ZeroFluxBC;
///
/// impl RequiresContext for ZeroFluxBC {
///     fn required_variables(&self) -> Vec<ContextVariable> { vec![] }
/// }
///
/// impl BoundaryCondition for ZeroFluxBC {
///     fn apply(
///         &self,
///         _state: &mut DVector<f64>,
///         _ctx: &ComputeContext,
///         _mesh: &dyn Mesh,
///     ) -> Result<(), OxiflowError> {
///         // Zero-flux: no modification needed (natural BC in weak form).
///         Ok(())
///     }
/// }
///
/// let mesh = UniformGrid1D::new(10, 0.0, 1.0).unwrap();
/// let mut state = DVector::from_element(mesh.n_dof(), 1.0);
/// let ctx = ComputeContext::new(0.0, 0.01);
/// let bc = ZeroFluxBC;
/// assert!(bc.apply(&mut state, &ctx, &mesh).is_ok());
/// ```
///
/// [`RequiresContext`]: crate::model::RequiresContext
/// [`ComputeContext`]: crate::context::ComputeContext
pub trait BoundaryCondition: RequiresContext + std::fmt::Debug {
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
    /// Returns [`OxiflowError`] if the constraint cannot be applied
    /// (e.g., a required context value is missing or the state is inconsistent).
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

    /// Zero-flux (homogeneous Neumann) — no context requirements, no-op apply.
    #[derive(Debug)]
    struct ZeroFluxBC;

    impl RequiresContext for ZeroFluxBC {
        fn required_variables(&self) -> Vec<ContextVariable> {
            vec![]
        }
    }

    impl BoundaryCondition for ZeroFluxBC {
        fn apply(
            &self,
            _state: &mut DVector<f64>,
            _ctx: &ComputeContext,
            _mesh: &dyn Mesh,
        ) -> Result<(), OxiflowError> {
            Ok(())
        }
    }

    /// Fixed inlet value — sets state[0] to a constant.
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

    /// BC that requires Time from context.
    #[derive(Debug)]
    struct TimeDependentBC;

    impl RequiresContext for TimeDependentBC {
        fn required_variables(&self) -> Vec<ContextVariable> {
            vec![ContextVariable::Time]
        }
    }

    impl BoundaryCondition for TimeDependentBC {
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

    // ── RequiresContext supertrait ─────────────────────────────────────────────

    #[test]
    fn required_variables_forwarded_via_dyn() {
        let bc: &dyn BoundaryCondition = &TimeDependentBC;
        assert!(bc.required_variables().contains(&ContextVariable::Time));
    }

    #[test]
    fn optional_variables_default_is_empty() {
        let bc = ZeroFluxBC;
        assert!(bc.optional_variables().is_empty());
    }

    #[test]
    fn depends_on_default_is_empty() {
        let bc = ZeroFluxBC;
        assert!(bc.depends_on().is_empty());
    }

    #[test]
    fn priority_default_is_100() {
        let bc = ZeroFluxBC;
        assert_eq!(bc.priority(), 100);
    }

    #[test]
    fn priority_can_be_overridden() {
        let bc = FixedInletBC { value: 0.0 };
        assert_eq!(bc.priority(), 50);
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
        // Other DOFs unchanged
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
