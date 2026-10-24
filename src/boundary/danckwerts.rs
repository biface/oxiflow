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
//! Discretised with the one-sided difference between the inlet node and its
//! neighbour, $(u_1 - u_0)/\Delta x$ with $\Delta x = x_1 - x_0$, and solved
//! for the inlet node (DD-070, #201):
//!
//! $$u_0 = \frac{u_{\text{feed}}(t) + \kappa \, u_1}{1 + \kappa},
//!   \qquad \kappa = \frac{D}{v \, \Delta x}$$
//!
//! The inlet value follows from its neighbour alone, not from a gradient
//! computed on the state before the condition was applied. The former form,
//! $u_0 = u_{\text{feed}} + (D/v)\,\partial_x u|_0$ with the gradient read
//! from the context, iterated $u_0$ with a factor $-\kappa$ from one
//! evaluation to the next and diverged as soon as $\kappa > 1$, that is
//! $(n - 1)/\text{Pe} > 1$ on a uniform grid with $\text{Pe} = vL/D$.
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
//! ## Supported meshes
//!
//! Both conditions locate the boundary by DOF index: the inlet is node `0`
//! and reads node `1`, the outlet node `n − 1` copies node `n − 2`. That
//! holds on an ordered 1D grid only — on an unstructured mesh (#151) node `0` is an
//! arbitrary node. Both `apply()` methods therefore return
//! `PreconditionFailed` when `mesh.spatial_dimension() != 1` (#187), as
//! `TrapezoidalIntegral` does. On a 1D mesh, nodes are assumed ordered from
//! inlet to outlet.
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

/// Refuses a mesh that is not 1D: both conditions locate the boundary by
/// DOF index, which only an ordered 1D grid makes meaningful.
fn require_1d(mesh: &dyn Mesh, context: &'static str) -> Result<(), OxiflowError> {
    let dimension = mesh.spatial_dimension();
    if dimension == 1 {
        Ok(())
    } else {
        Err(OxiflowError::PreconditionFailed {
            context,
            message: format!("only 1D meshes are supported, got spatial_dimension = {dimension}"),
        })
    }
}

// ── DanckwertsInlet ───────────────────────────────────────────────────────────

/// Danckwerts inlet condition — Robin BC at $x = 0$.
///
/// Enforces mass-flux continuity at the column inlet,
/// $v\,u(0, t) - D\,\partial_x u|_{x=0} = v\,u_{\text{feed}}(t)$, through its
/// one-sided discretisation solved for node 0 (see the module
/// documentation):
///
/// $$u_0 = \frac{u_{\text{feed}}(t) + \kappa \, u_1}{1 + \kappa},
///   \qquad \kappa = \frac{D}{v \, (x_1 - x_0)}$$
///
/// Node 0 is declared as a constrained node (DD-070): the solver keeps its
/// derivative at zero and the condition alone sets it.
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
/// Declares one required context variable, `feed_variable` — the feed
/// concentration $u_{\text{feed}}(t)$, which does not depend on the state.
/// The solver validates that it is covered before the first time step. No
/// gradient calculator is needed.
///
/// # Supported meshes
///
/// 1D meshes with at least two nodes, ordered from inlet (node `0`) to
/// outlet; see the module documentation. `apply()` refuses any other spatial
/// dimension, a single node, and a node `1` that is not downstream of node
/// `0`.
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
        vec![self.feed_variable.clone()]
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
    /// Reads $u_1$ from `state`, $x_0$ and $x_1$ from `mesh`, and
    /// $u_{\text{feed}}$ from the context variable declared at construction.
    ///
    /// # Errors
    ///
    /// - [`OxiflowError::MissingCalculator`] if the feed variable is absent
    ///   from the context.
    /// - [`OxiflowError::TypeMismatch`] if the feed variable is not a `Scalar`.
    /// - [`OxiflowError::PreconditionFailed`] if the mesh is not 1D, has fewer
    ///   than two nodes, if node `1` is not downstream of node `0`, if
    ///   `velocity` is zero, or if $1 + \kappa = 0$.
    fn apply(
        &self,
        state: &mut DVector<f64>,
        ctx: &ComputeContext,
        mesh: &dyn Mesh,
    ) -> Result<(), OxiflowError> {
        require_1d(mesh, "DanckwertsInlet")?;
        let refuse = |message: String| OxiflowError::PreconditionFailed {
            context: "DanckwertsInlet",
            message,
        };
        if mesh.n_dof() < 2 {
            return Err(refuse(format!(
                "mesh must have at least 2 nodes, got {}",
                mesh.n_dof()
            )));
        }
        if self.velocity == 0.0 {
            return Err(refuse("velocity must be non-zero".into()));
        }
        let dx = mesh.coordinates(1)[0] - mesh.coordinates(0)[0];
        if dx.is_nan() || dx <= 0.0 {
            return Err(refuse(format!(
                "node 1 must be downstream of node 0, got x1 - x0 = {dx}"
            )));
        }
        let kappa = self.dispersion / (self.velocity * dx);
        if 1.0 + kappa == 0.0 {
            return Err(refuse(format!(
                "1 + D / (v dx) is zero (D = {}, v = {}, dx = {dx})",
                self.dispersion, self.velocity
            )));
        }

        let u_feed = ctx.external(self.feed_variable.clone())?.as_scalar()?;
        state[0] = (u_feed + kappa * state[1]) / (1.0 + kappa);
        Ok(())
    }

    /// Node 0, set by [`apply`](Self::apply) (DD-070).
    fn constrained_nodes(&self, mesh: &dyn Mesh) -> Vec<usize> {
        if mesh.n_dof() == 0 {
            Vec::new()
        } else {
            vec![0]
        }
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
/// # Supported meshes
///
/// 1D meshes whose last node (`n − 1`) is the outlet and whose node `n − 2`
/// is its neighbour; see the module documentation. `apply()` refuses any
/// other spatial dimension.
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
    /// - [`OxiflowError::PreconditionFailed`] if the mesh is not 1D or has
    ///   fewer than two nodes.
    fn apply(
        &self,
        state: &mut DVector<f64>,
        _ctx: &ComputeContext,
        mesh: &dyn Mesh,
    ) -> Result<(), OxiflowError> {
        require_1d(mesh, "DanckwertsOutlet")?;
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

    /// The last node, set by [`apply`](Self::apply) (DD-070).
    fn constrained_nodes(&self, mesh: &dyn Mesh) -> Vec<usize> {
        mesh.n_dof().checked_sub(1).into_iter().collect()
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::context::value::ContextValue;
    use crate::mesh::structured::UniformGrid1D;
    use crate::mesh::NodeCloud;
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

    fn make_ctx_with_feed(u_feed: f64) -> ComputeContext {
        let mut ctx = ComputeContext::new(0.0, 0.01);
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
    fn inlet_requires_the_feed_only() {
        // DD-070: no state-dependent variable, so no gradient.
        assert_eq!(
            make_inlet(1e-7, 1e-3).required_variables(),
            vec![feed_var()]
        );
    }

    #[test]
    fn inlet_declares_node_zero() {
        let inlet = make_inlet(1e-7, 1e-3);
        assert_eq!(inlet.constrained_nodes(&make_mesh(5)), vec![0]);
        assert!(inlet
            .constrained_nodes(&NodeCloud::new::<1>(&[], 0.0))
            .is_empty());
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

    /// `v u0 - D (u1 - u0) / dx - v u_feed`: zero when the discrete Robin
    /// condition holds.
    fn robin_residual(inlet: &DanckwertsInlet, state: &DVector<f64>, dx: f64, u_feed: f64) -> f64 {
        inlet.velocity * state[0]
            - inlet.dispersion * (state[1] - state[0]) / dx
            - inlet.velocity * u_feed
    }

    #[test]
    fn inlet_apply_equals_u_feed_on_a_flat_profile() {
        // u1 = u_feed gives u0 = u_feed, whatever kappa.
        let inlet = make_inlet(1e-4, 1e-2);
        let mut state = DVector::from_element(5, 0.8);
        state[0] = 0.0;
        inlet
            .apply(&mut state, &make_ctx_with_feed(0.8), &make_mesh(5))
            .unwrap();
        assert!((state[0] - 0.8).abs() < 1e-15);
    }

    #[test]
    fn inlet_apply_solves_the_discrete_robin_relation() {
        // D = 1e-3, v = 1e-2, dx = 0.25: kappa = 0.4.
        // u0 = (0.8 + 0.4 * 2.0) / 1.4.
        let inlet = make_inlet(1e-3, 1e-2);
        let mut state = DVector::from_element(5, 2.0);
        inlet
            .apply(&mut state, &make_ctx_with_feed(0.8), &make_mesh(5))
            .unwrap();
        assert!((state[0] - (0.8 + 0.4 * 2.0) / 1.4).abs() < 1e-15);
        assert!(robin_residual(&inlet, &state, 0.25, 0.8).abs() < 1e-15);
    }

    #[test]
    fn inlet_apply_does_not_depend_on_the_previous_inlet_value() {
        // The former form iterated u0 with a factor -kappa; this one does
        // not read u0 at all.
        let inlet = make_inlet(1.0, 1e-2); // kappa = 400 on dx = 0.25
        let mut a = DVector::from_element(5, 2.0);
        let mut b = a.clone();
        b[0] = 1e6;
        let ctx = make_ctx_with_feed(0.8);
        inlet.apply(&mut a, &ctx, &make_mesh(5)).unwrap();
        inlet.apply(&mut b, &ctx, &make_mesh(5)).unwrap();
        assert_eq!(a[0].to_bits(), b[0].to_bits());
        assert!(robin_residual(&inlet, &a, 0.25, 0.8).abs() < 1e-12);
    }

    #[test]
    fn inlet_apply_does_not_modify_other_nodes() {
        let inlet = make_inlet(1e-7, 1e-3);
        let ctx = make_ctx_with_feed(1.0);
        let mut state = DVector::from_element(5, 2.0);
        inlet.apply(&mut state, &ctx, &make_mesh(5)).unwrap();
        for i in 1..5 {
            assert!((state[i] - 2.0).abs() < 1e-12, "node {i} was modified");
        }
    }

    #[test]
    fn inlet_apply_missing_feed_returns_error() {
        let inlet = make_inlet(1e-7, 1e-3);
        let ctx = ComputeContext::new(0.0, 0.01);
        let mut state = DVector::zeros(5);
        let err = inlet.apply(&mut state, &ctx, &make_mesh(5)).unwrap_err();
        assert!(matches!(err, OxiflowError::MissingCalculator(_)));
    }

    #[test]
    fn inlet_apply_refuses_a_single_node_or_a_reversed_mesh() {
        let inlet = make_inlet(1e-7, 1e-3);
        let ctx = make_ctx_with_feed(1.0);
        let single = NodeCloud::new(&[[0.0]], 0.0);
        let reversed = NodeCloud::new(&[[1.0], [0.5], [0.0]], 0.5);
        for (mesh, n) in [(&single, 1), (&reversed, 3)] {
            let mut state = DVector::from_element(n, 2.0);
            let err = inlet.apply(&mut state, &ctx, mesh).unwrap_err();
            assert!(
                matches!(err, OxiflowError::PreconditionFailed { .. }),
                "{err:?}"
            );
            assert_eq!(state, DVector::from_element(n, 2.0));
        }
    }

    #[test]
    fn inlet_apply_zero_velocity_returns_error() {
        let inlet = make_inlet(1e-7, 0.0);
        let mesh = make_mesh(5);
        let ctx = make_ctx_with_feed(1.0);
        let mut state = DVector::zeros(5);
        let err = inlet.apply(&mut state, &ctx, &mesh).unwrap_err();
        assert!(matches!(err, OxiflowError::PreconditionFailed { .. }));
    }

    #[test]
    fn inlet_apply_empty_mesh_returns_error() {
        let inlet = make_inlet(1e-7, 1e-3);
        // UniformGrid1D requires n >= 2: a node cloud without nodes covers this edge case.
        let mesh = NodeCloud::new::<1>(&[], 0.0);
        let ctx = make_ctx_with_feed(1.0);
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
    fn outlet_declares_its_last_node() {
        // DD-070: the node `apply` sets is declared to the solver.
        let outlet = DanckwertsOutlet::new();
        assert_eq!(outlet.constrained_nodes(&make_mesh(5)), vec![4]);
        assert!(outlet
            .constrained_nodes(&NodeCloud::new::<1>(&[], 0.0))
            .is_empty());
    }

    #[test]
    fn outlet_apply_single_node_returns_error() {
        let mesh = NodeCloud::new(&[[0.0]], 0.0);
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
