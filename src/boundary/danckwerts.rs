//! # Module `boundary::danckwerts`
//!
//! Danckwerts boundary conditions for convection-diffusion problems (issue #35).
//!
//! ## Physical context
//!
//! Danckwerts conditions arise from a mass balance at the boundaries of a porous
//! medium column (chromatography, fixed-bed reactors, dispersion in soil). They
//! account for both convective and dispersive fluxes at the inlet and impose zero
//! dispersive flux at the outlet.
//!
//! ## Inlet — Robin condition
//!
//! At $x = 0$, the total mass flux is continuous:
//!
//! $$v \, u(0, t) - D \left.\frac{\partial u}{\partial x}\right|_{x=0}
//!   = v \, u_{\text{feed}}(t)$$
//!
//! Rearranged into an explicit assignment for the inlet node:
//!
//! $$u(0, t) = u_{\text{feed}}(t) + \frac{D}{v}
//!   \left.\frac{\partial u}{\partial x}\right|_{x=0}$$
//!
//! The feed concentration $u_{\text{feed}}(t)$ is read from [`ComputeContext`] via
//! a typed [`ContextVariable`] key — it can be constant, time-dependent, or
//! produced by any user-supplied [`ContextCalculator`].
//!
//! ## Outlet — Neumann condition
//!
//! At $x = L$, the dispersive flux vanishes:
//!
//! $$\left.\frac{\partial u}{\partial x}\right|_{x=L} = 0$$
//!
//! Applied as a first-order finite-difference approximation: the last node value
//! is set equal to its immediate predecessor, enforcing zero gradient.
//!
//! ## Souplesse
//!
//! The physical parameters $D$ (dispersion) and $v$ (velocity) are struct fields —
//! they characterise the medium and do not change during a simulation. The feed
//! concentration is deliberately read from context via a [`ContextVariable`] key,
//! allowing any dynamics: constant injection, step injection, gradient elution, or
//! a user-computed profile that itself depends on `Time` or other variables.
//!
//! ## Serialisation
//!
//! Both types implement `serde::Serialize` / `serde::Deserialize` under the `serde`
//! feature flag. All fields are either `f64` or [`ContextVariable`] — both
//! already serialisable.
//!
//! [`ComputeContext`]: crate::context::compute::ComputeContext
//! [`ContextCalculator`]: crate::context::calculator::ContextCalculator
//! [`ContextVariable`]: crate::context::variable::ContextVariable

use crate::boundary::BoundaryCondition;
use crate::context::compute::ComputeContext;
use crate::context::error::OxiflowError;
use crate::context::variable::ContextVariable;
use crate::mesh::Mesh;
use crate::model::traits::RequiresContext;
use nalgebra::DVector;

// ── DanckwertsInlet ───────────────────────────────────────────────────────────

/// Danckwerts inlet condition — Robin BC at $x = 0$.
///
/// Enforces mass-flux continuity at the column inlet:
///
/// $$u(0, t) = u_{\text{feed}}(t) + \frac{D}{v}
///   \left.\frac{\partial u}{\partial x}\right|_{x=0}$$
///
/// # Parameters
///
/// - `dispersion` — axial dispersion coefficient $D$ $[\text{m}^2 \cdot \text{s}^{-1}]$.
/// - `velocity` — interstitial velocity $v$ $[\text{m} \cdot \text{s}^{-1}]$.
/// - `feed_variable` — typed key used to read $u_{\text{feed}}(t)$ from
///   [`ComputeContext`]. Any `ContextVariable` is accepted — typically an
///   [`External`](crate::context::variable::ContextVariable::External) variable
///   produced by a user-supplied calculator (constant, step, gradient elution…).
///
/// # Requirements
///
/// Declares two required context variables:
///
/// - `ContextVariable::SpatialGradient { dimension: 0, component: None }` —
///   the nodal gradient field $\partial u / \partial x$ computed by a gradient
///   calculator registered in `SolverConfiguration`.
/// - `feed_variable` — the feed concentration $u_{\text{feed}}(t)$.
///
/// The solver validates that both are covered before the first time step.
///
/// # Serialisation
///
/// Implements `Serialize` / `Deserialize` under the `serde` feature flag.
///
/// # Examples
///
/// ```rust
/// use oxiflow::boundary::danckwerts::DanckwertsInlet;
/// use oxiflow::context::variable::ContextVariable;
/// use oxiflow::model::RequiresContext;
///
/// let feed_var = ContextVariable::External { name: "feed_concentration".into() };
/// let inlet = DanckwertsInlet::new(1e-7, 1e-3, feed_var);
/// assert_eq!(inlet.priority(), 50);
/// ```
///
/// [`ComputeContext`]: crate::context::compute::ComputeContext
#[non_exhaustive]
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct DanckwertsInlet {
    /// Axial dispersion coefficient $D$.
    pub dispersion: f64,
    /// Interstitial velocity $v$.
    pub velocity: f64,
    /// Context key used to read the feed concentration $u_{\text{feed}}(t)$.
    pub feed_variable: ContextVariable,
}

impl DanckwertsInlet {
    /// Creates a new `DanckwertsInlet`.
    ///
    /// # Parameters
    ///
    /// - `dispersion` — axial dispersion coefficient $D$.
    /// - `velocity`   — interstitial velocity $v$ (must be non-zero at apply time).
    /// - `feed_variable` — context key for $u_{\text{feed}}(t)$.
    pub fn new(dispersion: f64, velocity: f64, feed_variable: ContextVariable) -> Self {
        Self {
            dispersion,
            velocity,
            feed_variable,
        }
    }
}

impl RequiresContext for DanckwertsInlet {
    fn required_variables(&self) -> Vec<ContextVariable> {
        vec![
            ContextVariable::SpatialGradient {
                dimension: 0,
                component: None,
            },
            self.feed_variable.clone(),
        ]
    }

    fn priority(&self) -> u32 {
        50
    }
}

impl BoundaryCondition for DanckwertsInlet {
    fn boundary_type(&self) -> crate::boundary::BoundaryType {
        crate::boundary::BoundaryType::Robin
    }

    fn location(&self) -> Option<crate::boundary::BoundaryLocation> {
        Some(crate::boundary::BoundaryLocation::Inlet)
    }

    /// Applies the Danckwerts inlet condition to `state[0]`.
    ///
    /// Reads `∂u/∂x|_{x=0}` from the gradient field at node 0 and
    /// $u_{\text{feed}}$ from the context variable declared at construction.
    ///
    /// # Errors
    ///
    /// - [`OxiflowError::MissingCalculator`] if the gradient field or the feed
    ///   variable is absent from the context.
    /// - [`OxiflowError::TypeMismatch`] if the feed variable is not a `Scalar`.
    /// - [`OxiflowError::PreconditionFailed`] if the mesh has no nodes or if
    ///   `velocity` is zero (division by zero in $D/v$).
    fn apply(
        &self,
        state: &mut DVector<f64>,
        ctx: &ComputeContext,
        mesh: &dyn Mesh,
    ) -> Result<(), OxiflowError> {
        if mesh.n_dof() == 0 {
            return Err(OxiflowError::PreconditionFailed {
                context: "DanckwertsInlet",
                message: "mesh has no degrees of freedom".into(),
            });
        }
        if self.velocity == 0.0 {
            return Err(OxiflowError::PreconditionFailed {
                context: "DanckwertsInlet",
                message: "velocity must be non-zero".into(),
            });
        }

        let gradient = ctx.gradient(0)?;
        let du_dx_0 = gradient[0];
        let u_feed = ctx.external(self.feed_variable.clone())?.as_scalar()?;

        state[0] = u_feed + (self.dispersion / self.velocity) * du_dx_0;
        Ok(())
    }
}

// ── DanckwertsOutlet ──────────────────────────────────────────────────────────

/// Danckwerts outlet condition — Neumann BC at $x = L$.
///
/// Enforces zero dispersive flux at the column outlet:
///
/// $$\left.\frac{\partial u}{\partial x}\right|_{x=L} = 0$$
///
/// Applied as a first-order finite-difference approximation:
///
/// $$u_{n-1} = u_{n-2}$$
///
/// No context variables are required — the condition depends only on the
/// current field state and the mesh geometry.
///
/// # Serialisation
///
/// Implements `Serialize` / `Deserialize` under the `serde` feature flag.
///
/// # Examples
///
/// ```rust
/// use oxiflow::boundary::danckwerts::DanckwertsOutlet;
/// use oxiflow::model::RequiresContext;
///
/// let outlet = DanckwertsOutlet::new();
/// assert_eq!(outlet.priority(), 60);
/// ```
#[non_exhaustive]
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct DanckwertsOutlet;

impl DanckwertsOutlet {
    /// Creates a new `DanckwertsOutlet`.
    pub fn new() -> Self {
        Self
    }
}

impl Default for DanckwertsOutlet {
    fn default() -> Self {
        Self::new()
    }
}

impl RequiresContext for DanckwertsOutlet {
    fn required_variables(&self) -> Vec<ContextVariable> {
        vec![]
    }

    fn priority(&self) -> u32 {
        60
    }
}

impl BoundaryCondition for DanckwertsOutlet {
    fn boundary_type(&self) -> crate::boundary::BoundaryType {
        crate::boundary::BoundaryType::Neumann
    }

    fn location(&self) -> Option<crate::boundary::BoundaryLocation> {
        Some(crate::boundary::BoundaryLocation::Outlet)
    }

    /// Applies the Danckwerts outlet condition.
    ///
    /// Sets `state[n-1] = state[n-2]` — first-order approximation of zero
    /// gradient at the outlet. Requires at least two nodes.
    ///
    /// # Errors
    ///
    /// - [`OxiflowError::PreconditionFailed`] if the mesh has fewer than two nodes.
    fn apply(
        &self,
        state: &mut DVector<f64>,
        _ctx: &ComputeContext,
        mesh: &dyn Mesh,
    ) -> Result<(), OxiflowError> {
        let n = mesh.n_dof();
        if n < 2 {
            return Err(OxiflowError::PreconditionFailed {
                context: "DanckwertsOutlet",
                message: "mesh must have at least 2 nodes".into(),
            });
        }
        state[n - 1] = state[n - 2];
        Ok(())
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::context::value::ContextValue;
    use crate::mesh::structured::UniformGrid1D;
    use nalgebra::DVector;

    // ── Fixtures ──────────────────────────────────────────────────────────────

    fn feed_var() -> ContextVariable {
        ContextVariable::External {
            name: "feed_concentration".into(),
        }
    }

    fn make_inlet(d: f64, v: f64) -> DanckwertsInlet {
        DanckwertsInlet::new(d, v, feed_var())
    }

    fn make_mesh(n: usize) -> UniformGrid1D {
        UniformGrid1D::new(n, 0.0, 1.0).unwrap()
    }

    fn make_ctx_with_gradient_and_feed(gradient_at_0: f64, u_feed: f64) -> ComputeContext {
        let mut ctx = ComputeContext::new(0.0, 0.01);
        let n = 5;
        let mut grad = DVector::zeros(n);
        grad[0] = gradient_at_0;
        ctx.insert(
            ContextVariable::SpatialGradient {
                dimension: 0,
                component: None,
            },
            ContextValue::ScalarField(grad),
        );
        ctx.insert(feed_var(), ContextValue::Scalar(u_feed));
        ctx
    }

    // ── boundary_type() and location() ───────────────────────────────────────

    #[test]
    fn inlet_boundary_type_is_robin() {
        assert_eq!(
            make_inlet(1e-7, 1e-3).boundary_type(),
            crate::boundary::BoundaryType::Robin
        );
    }

    #[test]
    fn inlet_location_is_inlet() {
        assert_eq!(
            make_inlet(1e-7, 1e-3).location(),
            Some(crate::boundary::BoundaryLocation::Inlet)
        );
    }

    #[test]
    fn outlet_boundary_type_is_neumann() {
        assert_eq!(
            DanckwertsOutlet::new().boundary_type(),
            crate::boundary::BoundaryType::Neumann
        );
    }

    #[test]
    fn outlet_location_is_outlet() {
        assert_eq!(
            DanckwertsOutlet::new().location(),
            Some(crate::boundary::BoundaryLocation::Outlet)
        );
    }

    // ── RequiresContext — DanckwertsInlet ─────────────────────────────────────

    #[test]
    fn inlet_required_variables_contains_gradient_and_feed() {
        let inlet = make_inlet(1e-7, 1e-3);
        let reqs = inlet.required_variables();
        assert!(reqs.contains(&ContextVariable::SpatialGradient {
            dimension: 0,
            component: None,
        }));
        assert!(reqs.contains(&feed_var()));
        assert_eq!(reqs.len(), 2);
    }

    #[test]
    fn inlet_optional_variables_default_empty() {
        assert!(make_inlet(1e-7, 1e-3).optional_variables().is_empty());
    }

    #[test]
    fn inlet_depends_on_default_empty() {
        assert!(make_inlet(1e-7, 1e-3).depends_on().is_empty());
    }

    #[test]
    fn inlet_priority_is_50() {
        assert_eq!(make_inlet(1e-7, 1e-3).priority(), 50);
    }

    // ── RequiresContext — DanckwertsOutlet ────────────────────────────────────

    #[test]
    fn outlet_required_variables_is_empty() {
        assert!(DanckwertsOutlet::new().required_variables().is_empty());
    }

    #[test]
    fn outlet_priority_is_60() {
        assert_eq!(DanckwertsOutlet::new().priority(), 60);
    }

    // ── apply — DanckwertsInlet ───────────────────────────────────────────────

    #[test]
    fn inlet_apply_zero_gradient_equals_u_feed() {
        // When ∂u/∂x|₀ = 0, state[0] = u_feed exactly.
        let inlet = make_inlet(1e-7, 1e-3);
        let mesh = make_mesh(5);
        let ctx = make_ctx_with_gradient_and_feed(0.0, 1.0);
        let mut state = DVector::zeros(5);
        inlet.apply(&mut state, &ctx, &mesh).unwrap();
        assert!((state[0] - 1.0).abs() < 1e-12);
    }

    #[test]
    fn inlet_apply_formula_is_correct() {
        // u(0) = u_feed + (D/v) * du_dx_0
        // With D=1e-4, v=1e-2, du_dx_0=0.5, u_feed=0.8:
        // u(0) = 0.8 + (1e-4/1e-2)*0.5 = 0.8 + 0.005 = 0.805
        let inlet = make_inlet(1e-4, 1e-2);
        let mesh = make_mesh(5);
        let ctx = make_ctx_with_gradient_and_feed(0.5, 0.8);
        let mut state = DVector::zeros(5);
        inlet.apply(&mut state, &ctx, &mesh).unwrap();
        let expected = 0.8 + (1e-4 / 1e-2) * 0.5;
        assert!((state[0] - expected).abs() < 1e-12);
    }

    #[test]
    fn inlet_apply_does_not_modify_other_nodes() {
        let inlet = make_inlet(1e-7, 1e-3);
        let mesh = make_mesh(5);
        let ctx = make_ctx_with_gradient_and_feed(0.0, 1.0);
        let mut state = DVector::from_element(5, 2.0);
        inlet.apply(&mut state, &ctx, &mesh).unwrap();
        // Only node 0 is changed.
        for i in 1..5 {
            assert!((state[i] - 2.0).abs() < 1e-12, "node {i} was modified");
        }
    }

    #[test]
    fn inlet_apply_missing_gradient_returns_error() {
        let inlet = make_inlet(1e-7, 1e-3);
        let mesh = make_mesh(5);
        let mut ctx = ComputeContext::new(0.0, 0.01);
        ctx.insert(feed_var(), ContextValue::Scalar(1.0));
        // Gradient not inserted.
        let mut state = DVector::zeros(5);
        let err = inlet.apply(&mut state, &ctx, &mesh).unwrap_err();
        assert!(matches!(err, OxiflowError::MissingCalculator(_)));
    }

    #[test]
    fn inlet_apply_missing_feed_returns_error() {
        let inlet = make_inlet(1e-7, 1e-3);
        let mesh = make_mesh(5);
        let mut ctx = ComputeContext::new(0.0, 0.01);
        let grad = ContextValue::ScalarField(DVector::zeros(5));
        ctx.insert(
            ContextVariable::SpatialGradient {
                dimension: 0,
                component: None,
            },
            grad,
        );
        // Feed not inserted.
        let mut state = DVector::zeros(5);
        let err = inlet.apply(&mut state, &ctx, &mesh).unwrap_err();
        assert!(matches!(err, OxiflowError::MissingCalculator(_)));
    }

    #[test]
    fn inlet_apply_zero_velocity_returns_error() {
        let inlet = make_inlet(1e-7, 0.0);
        let mesh = make_mesh(5);
        let ctx = make_ctx_with_gradient_and_feed(0.0, 1.0);
        let mut state = DVector::zeros(5);
        let err = inlet.apply(&mut state, &ctx, &mesh).unwrap_err();
        assert!(matches!(err, OxiflowError::PreconditionFailed { .. }));
    }

    #[test]
    fn inlet_apply_empty_mesh_returns_error() {
        let inlet = make_inlet(1e-7, 1e-3);
        // UniformGrid1D requires n >= 2, so we use a DummyMesh for this edge case.
        struct EmptyMesh;
        impl Mesh for EmptyMesh {
            fn n_dof(&self) -> usize {
                0
            }
            fn coordinates(&self, _: usize) -> &[f64] {
                &[]
            }
            fn spatial_dimension(&self) -> usize {
                1
            }
            fn characteristic_length(&self) -> f64 {
                0.0
            }
        }
        impl std::fmt::Debug for EmptyMesh {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                write!(f, "EmptyMesh")
            }
        }
        let mesh = EmptyMesh;
        let ctx = make_ctx_with_gradient_and_feed(0.0, 1.0);
        let mut state = DVector::zeros(0);
        let err = inlet.apply(&mut state, &ctx, &mesh).unwrap_err();
        assert!(matches!(err, OxiflowError::PreconditionFailed { .. }));
    }

    // ── apply — DanckwertsOutlet ──────────────────────────────────────────────

    #[test]
    fn outlet_apply_sets_last_node_to_predecessor() {
        let mesh = make_mesh(5);
        let ctx = ComputeContext::new(0.0, 0.01);
        let mut state = DVector::from_vec(vec![1.0, 2.0, 3.0, 4.0, 9.9]);
        DanckwertsOutlet::new()
            .apply(&mut state, &ctx, &mesh)
            .unwrap();
        assert!((state[4] - state[3]).abs() < 1e-12);
        assert!((state[4] - 4.0).abs() < 1e-12);
    }

    #[test]
    fn outlet_apply_does_not_modify_other_nodes() {
        let mesh = make_mesh(5);
        let ctx = ComputeContext::new(0.0, 0.01);
        let values = vec![1.0, 2.0, 3.0, 4.0, 9.9];
        let mut state = DVector::from_vec(values.clone());
        DanckwertsOutlet::new()
            .apply(&mut state, &ctx, &mesh)
            .unwrap();
        for i in 0..4 {
            assert!(
                (state[i] - values[i]).abs() < 1e-12,
                "node {i} was modified"
            );
        }
    }

    #[test]
    fn outlet_apply_two_node_mesh() {
        let mesh = make_mesh(2);
        let ctx = ComputeContext::new(0.0, 0.01);
        let mut state = DVector::from_vec(vec![3.0, 0.0]);
        DanckwertsOutlet::new()
            .apply(&mut state, &ctx, &mesh)
            .unwrap();
        assert!((state[1] - 3.0).abs() < 1e-12);
    }

    #[test]
    fn outlet_apply_single_node_returns_error() {
        struct SingleMesh;
        impl Mesh for SingleMesh {
            fn n_dof(&self) -> usize {
                1
            }
            fn coordinates(&self, _: usize) -> &[f64] {
                &[0.0]
            }
            fn spatial_dimension(&self) -> usize {
                1
            }
            fn characteristic_length(&self) -> f64 {
                0.0
            }
        }
        impl std::fmt::Debug for SingleMesh {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                write!(f, "SingleMesh")
            }
        }
        let mesh = SingleMesh;
        let ctx = ComputeContext::new(0.0, 0.01);
        let mut state = DVector::from_element(1, 1.0);
        let err = DanckwertsOutlet::new()
            .apply(&mut state, &ctx, &mesh)
            .unwrap_err();
        assert!(matches!(err, OxiflowError::PreconditionFailed { .. }));
    }

    // ── Object safety ─────────────────────────────────────────────────────────

    #[test]
    fn both_bcs_are_object_safe() {
        let bcs: Vec<Box<dyn BoundaryCondition>> = vec![
            Box::new(make_inlet(1e-7, 1e-3)),
            Box::new(DanckwertsOutlet::new()),
        ];
        assert_eq!(bcs.len(), 2);
    }

    #[test]
    fn debug_inlet_is_non_empty() {
        let s = format!("{:?}", make_inlet(1e-7, 1e-3));
        assert!(s.contains("DanckwertsInlet"));
        assert!(s.contains("dispersion"));
    }

    #[test]
    fn debug_outlet_is_non_empty() {
        assert!(!format!("{:?}", DanckwertsOutlet::new()).is_empty());
    }
}
