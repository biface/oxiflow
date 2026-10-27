//! # Module `boundary::dirichlet`
//!
//! Dirichlet boundary condition on a boundary region of the mesh (#202).
//!
//! ## Condition
//!
//! Every node of the region holds a prescribed value, constant or given as a
//! function of time:
//!
//! $$u = g(t) \quad \text{on the region}$$
//!
//! The time is read from [`ComputeContext::time`]; the condition requires no
//! context variable.
//!
//! ## Region
//!
//! The condition acts on a [`BoundaryRegion`] the mesh resolves with
//! [`Mesh::boundary_nodes`] (DA-071): with the default implementation of
//! `Mesh`, either face of a 1D mesh, normal to axis 0. A mesh that does not
//! define the region refuses it with `PreconditionFailed`. Only the nodes of
//! the region are needed, not their interior neighbours: unlike the
//! Danckwerts conditions, a Dirichlet condition also applies on a 1D mesh of
//! a single node.
//!
//! ## Constrained nodes (DD-070)
//!
//! The nodes of the region are declared through
//! [`BoundaryCondition::constrained_nodes`]: the solver keeps their
//! derivative at zero and the solver drivers re-apply the condition to the
//! initial state and at the end of each step, at the exact time of the saved
//! state. Every saved state holds `g(t)` at those nodes, bit for bit.
//!
//! ## Location
//!
//! [`BoundaryCondition::location`] keeps its default, `None`: the region
//! already designates where the condition acts, and a
//! [`BoundaryLocation`](crate::boundary::BoundaryLocation) (`Inlet`,
//! `Wall`…) carries a physical role a held value does not imply.
//!
//! ## Serialisation
//!
//! Not serialisable: the time-dependent value is a closure. A configuration
//! that needs persistence rebuilds the condition from its own description,
//! as `BoundaryCondition` recommends.
//!
//! [`ComputeContext::time`]: crate::context::compute::ComputeContext::time

use std::sync::Arc;

use crate::boundary::{BoundaryCondition, BoundaryType};
use crate::context::compute::ComputeContext;
use crate::context::error::OxiflowError;
use crate::context::variable::ContextVariable;
use crate::mesh::{BoundaryRegion, Mesh};
use crate::model::traits::RequiresContext;
use nalgebra::DVector;

/// Prescribed value of a [`Dirichlet`] condition.
///
/// `Arc` rather than `Box`: a clone of the condition shares the closure
/// instead of copying it, which a `dyn Fn` could not do.
#[derive(Clone)]
enum PrescribedValue {
    /// The same value at every time.
    Constant(f64),
    /// `g(t)`, evaluated at the context's time.
    OfTime(Arc<dyn Fn(f64) -> f64 + Send + Sync>),
}

impl PrescribedValue {
    fn at(&self, t: f64) -> f64 {
        match self {
            Self::Constant(value) => *value,
            Self::OfTime(g) => g(t),
        }
    }
}

impl std::fmt::Debug for PrescribedValue {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Constant(value) => f.debug_tuple("Constant").field(value).finish(),
            Self::OfTime(_) => f.write_str("OfTime(<fn>)"),
        }
    }
}

/// Dirichlet condition — every node of a boundary region holds a prescribed
/// value $g(t)$, constant or a function of time (#202).
///
/// The nodes of the region are declared as constrained nodes (DD-070): the
/// solver keeps their derivative at zero, and every saved state holds the
/// value $g(t)$ of its own time. See the module documentation.
///
/// Cloning a condition built with [`Self::of_t`] shares its function.
///
/// # Supported meshes
///
/// Any mesh defining the region through [`Mesh::boundary_nodes`]. With the
/// default implementation of `Mesh`: 1D meshes with at least one node.
///
/// # Examples
///
/// ```rust
/// use oxiflow::boundary::{BoundaryCondition, BoundaryType, Dirichlet};
/// use oxiflow::context::compute::ComputeContext;
/// use oxiflow::mesh::{BoundaryRegion, FaceSide, UniformGrid1D};
/// use nalgebra::DVector;
///
/// let mesh = UniformGrid1D::new(5, 0.0, 1.0).unwrap();
/// let lower = BoundaryRegion::Face { axis: 0, side: FaceSide::Lower };
/// let upper = BoundaryRegion::Face { axis: 0, side: FaceSide::Upper };
///
/// // T = 1 at x = 0, T = 1 + t/2 at x = 1.
/// let hot = Dirichlet::constant(lower, 1.0);
/// let ramp = Dirichlet::of_t(upper, |t| 1.0 + 0.5 * t);
/// assert_eq!(hot.boundary_type(), BoundaryType::Dirichlet);
///
/// let mut state = DVector::zeros(5);
/// let ctx = ComputeContext::new(2.0, 1e-3);
/// hot.apply(&mut state, &ctx, &mesh).unwrap();
/// ramp.apply(&mut state, &ctx, &mesh).unwrap();
/// assert_eq!(state.as_slice(), &[1.0, 0.0, 0.0, 0.0, 2.0]);
/// assert_eq!(ramp.constrained_nodes(&mesh), vec![4]);
/// ```
#[derive(Debug, Clone)]
pub struct Dirichlet {
    /// Boundary region the condition acts on (DA-071).
    pub region: BoundaryRegion,
    value: PrescribedValue,
}

impl Dirichlet {
    /// Creates a condition holding `value` on `region`, at every time.
    ///
    /// A non-finite `value` is reported by [`apply`](BoundaryCondition::apply).
    pub fn constant(region: BoundaryRegion, value: f64) -> Self {
        Self {
            region,
            value: PrescribedValue::Constant(value),
        }
    }

    /// Creates a condition holding `value(t)` on `region`, `t` being the
    /// context's time at each application.
    ///
    /// `value` should be a pure function of `t`: the solver may evaluate it
    /// several times at the same time, within a step and at its end. A
    /// non-finite result is reported by [`apply`](BoundaryCondition::apply).
    pub fn of_t<F>(region: BoundaryRegion, value: F) -> Self
    where
        F: Fn(f64) -> f64 + Send + Sync + 'static,
    {
        Self {
            region,
            value: PrescribedValue::OfTime(Arc::new(value)),
        }
    }
}

impl RequiresContext for Dirichlet {
    /// None: the time comes from [`ComputeContext::time`].
    fn required_variables(&self) -> Vec<ContextVariable> {
        vec![]
    }
}

impl BoundaryCondition for Dirichlet {
    fn boundary_type(&self) -> BoundaryType {
        BoundaryType::Dirichlet
    }

    /// Sets every node of the region to $g(t)$, `t` being `ctx.time()`.
    ///
    /// Both checks come before any node is set: a refused condition leaves
    /// the state as is.
    ///
    /// # Errors
    ///
    /// - [`OxiflowError::PreconditionFailed`] if the mesh does not define the
    ///   region (with the default implementation of `Mesh`: a mesh that is
    ///   not 1D, or has no node).
    /// - [`OxiflowError::ExternalData`] if $g(t)$ is not finite, naming the
    ///   region and the time.
    fn apply(
        &self,
        state: &mut DVector<f64>,
        ctx: &ComputeContext,
        mesh: &dyn Mesh,
    ) -> Result<(), OxiflowError> {
        let nodes = mesh.boundary_nodes(&self.region)?;
        let t = ctx.time();
        let value = self.value.at(t);
        if !value.is_finite() {
            return Err(OxiflowError::ExternalData(format!(
                "Dirichlet condition on {:?}: non-finite value {value} at t = {t}",
                self.region
            )));
        }
        for i in nodes {
            state[i] = value;
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
    use crate::mesh::structured::UniformGrid1D;
    use crate::mesh::{FaceSide, NodeCloud};

    const LOWER: BoundaryRegion = BoundaryRegion::Face {
        axis: 0,
        side: FaceSide::Lower,
    };
    const UPPER: BoundaryRegion = BoundaryRegion::Face {
        axis: 0,
        side: FaceSide::Upper,
    };

    fn grid() -> UniformGrid1D {
        UniformGrid1D::new(5, 0.0, 1.0).unwrap()
    }

    /// `u_i = i + 1`, so every node holds a distinct value.
    fn ramp(n: usize) -> DVector<f64> {
        DVector::from_iterator(n, (0..n).map(|i| (i + 1) as f64))
    }

    fn at(t: f64) -> ComputeContext {
        ComputeContext::new(t, 1e-3)
    }

    // ── apply() ───────────────────────────────────────────────────────────────

    #[test]
    fn a_constant_value_is_set_on_either_face_only() {
        let mesh = grid();
        let mut state = ramp(5);
        Dirichlet::constant(LOWER, -1.0)
            .apply(&mut state, &at(0.0), &mesh)
            .unwrap();
        assert_eq!(state.as_slice(), &[-1.0, 2.0, 3.0, 4.0, 5.0]);
        Dirichlet::constant(UPPER, 7.5)
            .apply(&mut state, &at(0.0), &mesh)
            .unwrap();
        assert_eq!(state.as_slice(), &[-1.0, 2.0, 3.0, 4.0, 7.5]);
    }

    #[test]
    fn a_time_dependent_value_is_read_at_the_context_time() {
        let mesh = grid();
        let g = |t: f64| 1.0 + t * t;
        let bc = Dirichlet::of_t(UPPER, g);
        for t in [0.0, 0.3, 2.0] {
            let mut state = ramp(5);
            bc.apply(&mut state, &at(t), &mesh).unwrap();
            assert_eq!(state[4].to_bits(), g(t).to_bits(), "t = {t}");
            assert_eq!(&state.as_slice()[..4], &[1.0, 2.0, 3.0, 4.0]);
        }
    }

    #[test]
    fn a_single_node_mesh_is_held_from_either_face() {
        let mesh = NodeCloud::new(&[[0.5]], 1.0);
        for region in [LOWER, UPPER] {
            let mut state = DVector::from_element(1, 0.0);
            Dirichlet::constant(region, 3.0)
                .apply(&mut state, &at(0.0), &mesh)
                .unwrap();
            assert_eq!(state[0], 3.0, "{region:?}");
        }
    }

    // ── Refusals ──────────────────────────────────────────────────────────────

    fn assert_precondition_failed(result: Result<(), OxiflowError>, what: &str) {
        assert!(
            matches!(result, Err(OxiflowError::PreconditionFailed { .. })),
            "{what}: expected PreconditionFailed, got {result:?}"
        );
    }

    #[test]
    fn a_2d_mesh_is_refused_through_the_region() {
        let mesh = NodeCloud::new(&[[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], 1.0);
        let mut state = ramp(3);
        for region in [LOWER, UPPER] {
            let result = Dirichlet::constant(region, 0.0).apply(&mut state, &at(0.0), &mesh);
            assert_precondition_failed(result, &format!("{region:?}"));
        }
        assert_eq!(state.as_slice(), &[1.0, 2.0, 3.0]);
    }

    #[test]
    fn an_empty_mesh_is_refused_through_the_region() {
        let mesh = NodeCloud::new::<1>(&[], 0.0);
        let mut state = DVector::zeros(0);
        let result = Dirichlet::constant(LOWER, 0.0).apply(&mut state, &at(0.0), &mesh);
        assert_precondition_failed(result, "empty mesh");
    }

    #[test]
    fn a_non_finite_value_is_refused_with_its_time() {
        let mesh = grid();
        let cases = [
            ("NaN constant", Dirichlet::constant(LOWER, f64::NAN)),
            (
                "infinite at t = 2",
                Dirichlet::of_t(UPPER, |t| if t >= 2.0 { f64::INFINITY } else { t }),
            ),
        ];
        for (what, bc) in cases {
            let mut state = ramp(5);
            let result = bc.apply(&mut state, &at(2.0), &mesh);
            let Err(OxiflowError::ExternalData(message)) = result else {
                panic!("{what}: expected ExternalData, got {result:?}");
            };
            assert!(message.contains("t = 2"), "{what}: {message}");
            assert!(message.contains("Face"), "{what}: {message}");
            assert_eq!(state, ramp(5), "{what}: state modified");
        }
    }

    // ── constrained_nodes() ───────────────────────────────────────────────────

    #[test]
    fn constrained_nodes_are_the_nodes_of_the_region() {
        let mesh = grid();
        assert_eq!(
            Dirichlet::constant(LOWER, 0.0).constrained_nodes(&mesh),
            vec![0]
        );
        assert_eq!(
            Dirichlet::constant(UPPER, 0.0).constrained_nodes(&mesh),
            vec![4]
        );
    }

    #[test]
    fn a_refused_region_declares_no_node() {
        let plane = NodeCloud::new(&[[0.0, 0.0], [1.0, 0.0]], 1.0);
        let empty = NodeCloud::new::<1>(&[], 0.0);
        assert!(Dirichlet::constant(LOWER, 0.0)
            .constrained_nodes(&plane)
            .is_empty());
        assert!(Dirichlet::constant(LOWER, 0.0)
            .constrained_nodes(&empty)
            .is_empty());
    }

    // ── Classification and requirements ───────────────────────────────────────

    #[test]
    fn classified_as_dirichlet_without_location_or_requirement() {
        let bc = Dirichlet::of_t(LOWER, |t| t);
        assert_eq!(bc.boundary_type(), BoundaryType::Dirichlet);
        assert!(bc.location().is_none());
        assert!(bc.required_variables().is_empty());
        assert!(bc.optional_variables().is_empty());
        assert_eq!(bc.priority(), 100);
    }

    #[test]
    fn usable_as_a_boxed_boundary_condition() {
        let bcs: Vec<Box<dyn BoundaryCondition>> = vec![
            Box::new(Dirichlet::constant(LOWER, 1.0)),
            Box::new(Dirichlet::of_t(UPPER, |t| t)),
        ];
        let mesh = grid();
        let mut state = ramp(5);
        for bc in &bcs {
            bc.apply(&mut state, &at(0.25), &mesh).unwrap();
        }
        assert_eq!(state.as_slice(), &[1.0, 2.0, 3.0, 4.0, 0.25]);
    }

    // ── Clone and Debug ───────────────────────────────────────────────────────

    #[test]
    fn a_clone_shares_the_function_and_keeps_the_region() {
        let calls = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let counter = Arc::clone(&calls);
        let bc = Dirichlet::of_t(UPPER, move |t| {
            counter.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            2.0 * t
        });
        let copy = bc.clone();
        assert_eq!(copy.region, UPPER);
        let mesh = grid();
        let mut state = ramp(5);
        copy.apply(&mut state, &at(1.5), &mesh).unwrap();
        bc.apply(&mut state, &at(1.5), &mesh).unwrap();
        assert_eq!(state[4], 3.0);
        assert_eq!(calls.load(std::sync::atomic::Ordering::SeqCst), 2);
    }

    #[test]
    fn debug_shows_the_region_and_the_kind_of_value() {
        let constant = format!("{:?}", Dirichlet::constant(LOWER, 1.25));
        assert!(constant.contains("Lower"), "{constant}");
        assert!(constant.contains("Constant(1.25)"), "{constant}");
        let of_t = format!("{:?}", Dirichlet::of_t(UPPER, |t| t));
        assert!(of_t.contains("Upper"), "{of_t}");
        assert!(of_t.contains("OfTime(<fn>)"), "{of_t}");
    }
}
