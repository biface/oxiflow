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
//! Each condition acts on a boundary region of the mesh (DA-071, #204): by
//! default the lower face of axis 0 for the inlet, the upper face for the
//! outlet — the two ends of a 1D column, flow towards increasing `x`.
//! [`DanckwertsInlet::on`] and [`DanckwertsOutlet::on`] choose another face,
//! for a flow towards decreasing `x` for instance. The mesh resolves the
//! region with [`Mesh::boundary_neighbours`]: every node of the face is set
//! from its first interior neighbour along the normal. A mesh that does not
//! define the region refuses it with `PreconditionFailed`; with the default
//! implementation of `Mesh`, that is any mesh that is not 1D (#187), and a
//! 1D mesh whose boundary nodes are not ordered along the axis.
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
//! feature flag. All fields are `f64`, [`ContextVariable`] or
//! [`BoundaryRegion`] — all serialisable. An inlet serialised before DA-071,
//! without a region, reads back on the lower face of axis 0. The outlet,
//! formerly a unit value, now serialises as a map holding its region; an
//! empty map reads back on the upper face.
//!
//! [`ComputeContext`]: crate::context::compute::ComputeContext
//! [`ContextCalculator`]: crate::context::calculator::ContextCalculator
//! [`ContextVariable`]: crate::context::variable::ContextVariable

use crate::boundary::BoundaryCondition;
use crate::context::compute::ComputeContext;
use crate::context::error::OxiflowError;
use crate::context::variable::ContextVariable;
use crate::mesh::{BoundaryRegion, FaceSide, Mesh};
use crate::model::traits::RequiresContext;
use nalgebra::DVector;

/// Default region of the inlet: the lower face of axis 0.
fn lower_face() -> BoundaryRegion {
    BoundaryRegion::Face {
        axis: 0,
        side: FaceSide::Lower,
    }
}

/// Default region of the outlet: the upper face of axis 0.
fn upper_face() -> BoundaryRegion {
    BoundaryRegion::Face {
        axis: 0,
        side: FaceSide::Upper,
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
/// Written for node 0 and its neighbour 1 of a 1D column, the default
/// region; on any region, each boundary node and its interior neighbour
/// play these roles. The nodes of the region are declared as constrained
/// nodes (DD-070): the solver keeps their derivative at zero and the
/// condition alone sets them.
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
/// Any mesh defining the region through [`Mesh::boundary_neighbours`]; see
/// the module documentation. With the default implementation of `Mesh`: 1D
/// meshes with at least two nodes, whose inlet pair is ordered along the
/// axis.
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
    /// Boundary region the condition acts on (DA-071) — the lower face of
    /// axis 0 unless set with [`Self::on`].
    #[cfg_attr(feature = "serde", serde(default = "lower_face"))]
    pub region: BoundaryRegion,
}

impl DanckwertsInlet {
    /// Creates a new `DanckwertsInlet`.
    ///
    /// # Parameters
    ///
    /// - `dispersion` — axial dispersion coefficient $D$.
    /// - `velocity`   — interstitial velocity $v$ (must be non-zero at apply time).
    /// - `feed_variable` — context key for $u_{\text{feed}}(t)$.
    ///
    /// The condition acts on the lower face of axis 0 — node `0` of a 1D
    /// column; [`Self::on`] chooses another face.
    pub fn new(dispersion: f64, velocity: f64, feed_variable: ContextVariable) -> Self {
        Self {
            dispersion,
            velocity,
            feed_variable,
            region: lower_face(),
        }
    }

    /// Sets the boundary region the condition acts on (DA-071) — the upper
    /// face of axis 0 for a flow towards decreasing `x`, for instance.
    pub fn on(mut self, region: BoundaryRegion) -> Self {
        self.region = region;
        self
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

    /// Applies the Danckwerts inlet condition to every node of its region.
    ///
    /// For each boundary node, reads its interior neighbour from `state` and
    /// their distance from [`Mesh::boundary_neighbours`], and
    /// $u_{\text{feed}}$ from the context variable declared at construction.
    ///
    /// # Errors
    ///
    /// - [`OxiflowError::MissingCalculator`] if the feed variable is absent
    ///   from the context.
    /// - [`OxiflowError::TypeMismatch`] if the feed variable is not a `Scalar`.
    /// - [`OxiflowError::PreconditionFailed`] if the mesh does not define the
    ///   region (with the default implementation of `Mesh`: a mesh that is
    ///   not 1D, has fewer than two nodes, or whose boundary pair is not
    ///   ordered along the axis), if `velocity` is zero, or if
    ///   $1 + \kappa = 0$.
    fn apply(
        &self,
        state: &mut DVector<f64>,
        ctx: &ComputeContext,
        mesh: &dyn Mesh,
    ) -> Result<(), OxiflowError> {
        let refuse = |message: String| OxiflowError::PreconditionFailed {
            context: "DanckwertsInlet",
            message,
        };
        let boundary = mesh.boundary_neighbours(&self.region)?;
        if self.velocity == 0.0 {
            return Err(refuse("velocity must be non-zero".into()));
        }
        let u_feed = ctx.external(self.feed_variable.clone())?.as_scalar()?;

        // Every kappa is checked before any node is set: a refused condition
        // leaves the state as is.
        let mut kappas = Vec::with_capacity(boundary.len());
        for b in &boundary {
            let kappa = self.dispersion / (self.velocity * b.distance);
            if 1.0 + kappa == 0.0 {
                return Err(refuse(format!(
                    "1 + D / (v dx) is zero at node {} (D = {}, v = {}, dx = {})",
                    b.node, self.dispersion, self.velocity, b.distance
                )));
            }
            kappas.push(kappa);
        }
        for (b, kappa) in boundary.iter().zip(kappas) {
            state[b.node] = (u_feed + kappa * state[b.inner]) / (1.0 + kappa);
        }
        Ok(())
    }

    /// The nodes of its region, set by [`apply`](Self::apply) (DD-070). A
    /// region the mesh does not define gives no node here; `apply` reports
    /// the error.
    fn constrained_nodes(&self, mesh: &dyn Mesh) -> Vec<usize> {
        mesh.boundary_nodes(&self.region).unwrap_or_default()
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
/// written for the last node of a 1D column, the default region; on any
/// region, each boundary node takes the value of its interior neighbour.
/// The nodes of the region are declared as constrained nodes (DD-070).
///
/// No context variables are required — the condition depends only on the
/// current field state and the mesh geometry.
///
/// # Supported meshes
///
/// Any mesh defining the region through [`Mesh::boundary_neighbours`]; see
/// the module documentation. With the default implementation of `Mesh`: 1D
/// meshes with at least two nodes, whose outlet pair is ordered along the
/// axis.
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
#[cfg_attr(feature = "serde", serde(default))]
pub struct DanckwertsOutlet {
    /// Boundary region the condition acts on (DA-071) — the upper face of
    /// axis 0 unless set with [`Self::on`].
    pub region: BoundaryRegion,
}

impl DanckwertsOutlet {
    /// Creates a new `DanckwertsOutlet`, acting on the upper face of axis 0
    /// — node `n − 1` of a 1D column; [`Self::on`] chooses another face.
    pub fn new() -> Self {
        Self {
            region: upper_face(),
        }
    }

    /// Sets the boundary region the condition acts on (DA-071) — the lower
    /// face of axis 0 for a flow towards decreasing `x`, for instance.
    pub fn on(mut self, region: BoundaryRegion) -> Self {
        self.region = region;
        self
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

    /// Applies the Danckwerts outlet condition to every node of its region.
    ///
    /// Sets each boundary node to the value of its first interior neighbour
    /// ([`Mesh::boundary_neighbours`]) — first-order approximation of a zero
    /// gradient across the outlet; on a 1D column, `state[n-1] = state[n-2]`.
    ///
    /// # Errors
    ///
    /// - [`OxiflowError::PreconditionFailed`] if the mesh does not define the
    ///   region (with the default implementation of `Mesh`: a mesh that is
    ///   not 1D, has fewer than two nodes, or whose boundary pair is not
    ///   ordered along the axis).
    fn apply(
        &self,
        state: &mut DVector<f64>,
        _ctx: &ComputeContext,
        mesh: &dyn Mesh,
    ) -> Result<(), OxiflowError> {
        for b in mesh.boundary_neighbours(&self.region)? {
            state[b.node] = state[b.inner];
        }
        Ok(())
    }

    /// The nodes of its region, set by [`apply`](Self::apply) (DD-070). A
    /// region the mesh does not define gives no node here; `apply` reports
    /// the error.
    fn constrained_nodes(&self, mesh: &dyn Mesh) -> Vec<usize> {
        mesh.boundary_nodes(&self.region).unwrap_or_default()
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

    // ── Boundary regions (DA-071) ────────────────────────────────────────────

    #[test]
    fn conditions_on_the_opposite_faces_for_a_flow_towards_decreasing_x() {
        use crate::mesh::{BoundaryRegion, FaceSide};
        let face = |side| BoundaryRegion::Face { axis: 0, side };
        let mesh = make_mesh(5); // dx = 0.25
        let inlet = make_inlet(1e-3, 1e-2).on(face(FaceSide::Upper)); // kappa = 0.4
        let outlet = DanckwertsOutlet::new().on(face(FaceSide::Lower));
        assert_eq!(inlet.constrained_nodes(&mesh), vec![4]);
        assert_eq!(outlet.constrained_nodes(&mesh), vec![0]);

        let mut state = DVector::from_vec(vec![9.0, 1.0, 2.0, 3.0, 9.0]);
        inlet
            .apply(&mut state, &make_ctx_with_feed(0.8), &mesh)
            .unwrap();
        outlet
            .apply(&mut state, &make_ctx_with_feed(0.8), &mesh)
            .unwrap();
        assert!((state[4] - (0.8 + 0.4 * 3.0) / 1.4).abs() < 1e-15);
        assert_eq!(state[0], 1.0);
        assert_eq!(state.rows(1, 3).as_slice(), &[1.0, 2.0, 3.0]);
    }

    #[test]
    fn new_conditions_act_on_the_two_ends_of_axis_0() {
        use crate::mesh::{BoundaryRegion, FaceSide};
        assert_eq!(
            make_inlet(1e-7, 1e-3).region,
            BoundaryRegion::Face {
                axis: 0,
                side: FaceSide::Lower
            }
        );
        assert_eq!(
            DanckwertsOutlet::new().region,
            BoundaryRegion::Face {
                axis: 0,
                side: FaceSide::Upper
            }
        );
    }

    #[cfg(feature = "serde")]
    #[test]
    fn serialised_conditions_without_a_region_read_back_on_their_default_face() {
        let mut inlet = serde_json::to_value(make_inlet(1e-7, 1e-3)).unwrap();
        inlet.as_object_mut().unwrap().remove("region").unwrap();
        let inlet: DanckwertsInlet = serde_json::from_value(inlet).unwrap();
        assert_eq!(inlet.region, make_inlet(1e-7, 1e-3).region);

        let outlet: DanckwertsOutlet = serde_json::from_str("{}").unwrap();
        assert_eq!(outlet.region, DanckwertsOutlet::new().region);
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
