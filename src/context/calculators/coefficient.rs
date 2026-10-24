//! # Module `context::calculators::coefficient`
//!
//! Coefficient fields c(x) and c(x, t) evaluated on the mesh nodes (DD-066,
//! #182).
//!
//! [`CoefficientField`] evaluates a user function on `Mesh::coordinates(i)`
//! for every node `i` and publishes the result as a `ScalarField` under
//! `ContextVariable::Coefficient { name }`. It is the space-dependent sibling
//! of `ExternalTabulated`, which stays a function of time only and publishes
//! a `Scalar`.
//!
//! ## Sharing without copy (DT-068, #195)
//!
//! The calculator overrides `ContextCalculator::compute_shared`, so the
//! solver receives the field it keeps instead of a copy:
//!
//! - c(x) is evaluated once, at construction, and every evaluation gets the
//!   same allocation;
//! - c(x, t) keeps its last time and value: evaluations at the same time
//!   (the stages of a step at one time, the n + 1 evaluations of a dense
//!   Jacobian) share one computation.
//!
//! Both caches assume the mesh does not change after construction. This is
//! an implementation detail, not a public contract: invalidation on a moving
//! or adapted mesh is for v0.11.0 and later (DD-066).
//!
//! The cache of c(x, t) is never locked while the field is computed. The
//! solver may evaluate derivatives inside Rayon tasks (DT-069, #198), and a
//! worker waiting for its own parallel loop can run another such task: if
//! that task blocked on a lock held by the same worker, it would never be
//! released. Concurrent misses at the same time therefore compute the same
//! field independently; the finite-difference Jacobians do not cause them,
//! since they evaluate the unperturbed state before dispatching columns.
//!
//! ## Parallel dispatch (DA-048, #140)
//!
//! With the `parallel` feature, the nodes are evaluated through Rayon once
//! the mesh has at least as many nodes as the calculator's threshold, and
//! each value is written directly into the output vector (no intermediate
//! collection, see #144). The threshold governs c(x, t); c(x) is evaluated
//! once, at construction, with the default threshold. Each node is
//! evaluated by the same function with the same arguments in both modes, so
//! the sequential and parallel fields are bit-for-bit identical.

use std::borrow::Cow;
use std::sync::{Arc, PoisonError, RwLock};

use nalgebra::DVector;

use crate::context::calculator::ContextCalculator;
use crate::context::compute::ComputeContext;
use crate::context::error::OxiflowError;
use crate::context::value::ContextValue;
use crate::context::variable::ContextVariable;
use crate::mesh::Mesh;
use crate::model::traits::RequiresContext;
#[cfg(feature = "parallel")]
use crate::solver::parallel::ParallelThreshold;

/// Default Rayon dispatch threshold of [`CoefficientField`], in nodes — an
/// indicative value. One node costs one call of the user function, often
/// heavier than one stencil point of the FD operators
/// ([`crate::operators::fd::default_parallel_threshold`], 49 999) and much
/// lighter than one derivative evaluation of the Jacobian assembly (256).
/// Measure the crossover on the target function and mesh, and set it with
/// [`CoefficientField::with_parallel_threshold`].
#[cfg(feature = "parallel")]
pub(crate) fn default_parallel_threshold() -> usize {
    10_000
}

/// Space-time coefficient function, as stored by [`CoefficientField::of_x_t`].
type SpaceTimeFunction = dyn Fn(&[f64], f64) -> f64 + Send + Sync;

/// What the calculator evaluates, and what it keeps between evaluations.
enum Source {
    /// c(x): the field evaluated at construction.
    Space(Arc<ContextValue>),
    /// c(x, t): the function, and the last time with its field.
    SpaceTime {
        function: Box<SpaceTimeFunction>,
        last: RwLock<Option<(f64, Arc<ContextValue>)>>,
    },
}

/// Coefficient field c(x) or c(x, t), evaluated on every mesh node (DD-066).
///
/// The function receives `Mesh::coordinates(i)`, a slice of length
/// `spatial_dimension()`, so a coefficient written for a 1D grid keeps its
/// meaning when the mesh gains dimensions. Any mesh is accepted: only its
/// coordinates are read.
///
/// The result is a `ScalarField` of `n_dof()` values, published under
/// [`ContextVariable::Coefficient`] with the name given at construction.
/// Models read it with [`ComputeContext::external`] or
/// [`ComputeContext::try_get`], then index the node they need.
///
/// # Function contract
///
/// The function must be pure — its result depends only on its arguments —
/// because nodes may be evaluated in parallel and in any order, and values
/// are reused between evaluations at the same time.
///
/// # Errors
///
/// A non-finite value (NaN or infinity) is an [`OxiflowError::ExternalData`]
/// naming the first such node (lowest index), its coordinates and, for
/// c(x, t), the time: at construction for c(x) ([`Self::of_x`] returns a
/// `Result`), at evaluation for c(x, t). A state that is not a `ScalarField`
/// or `VectorField` is a [`OxiflowError::TypeMismatch`]; one whose node
/// count differs from the mesh's `n_dof()` is a
/// [`OxiflowError::PreconditionFailed`].
///
/// # GPU
///
/// `GPU-READY: no` for the source: a user closure does not translate into a
/// shader. The output is a contiguous `ScalarField`, computed on the CPU and
/// handed over like any other context value (INV-GPU-1 to INV-GPU-5 review
/// under DD-066).
///
/// # Examples
///
/// ```rust
/// use std::sync::Arc;
///
/// use nalgebra::DVector;
/// use oxiflow::context::calculators::CoefficientField;
/// use oxiflow::context::{
///     ComputeContext, ContextCalculator, ContextValue, ContextVariable, OxiflowError,
/// };
/// use oxiflow::mesh::{Mesh, UniformGrid1D};
///
/// let mesh: Arc<dyn Mesh> = Arc::new(UniformGrid1D::new(5, 0.0, 1.0)?);
/// let state = ContextValue::ScalarField(DVector::zeros(5));
///
/// // c(x): porosity varying along a column, evaluated once.
/// let porosity = CoefficientField::of_x(Arc::clone(&mesh), "porosity", |x| 0.4 + 0.1 * x[0])?;
/// assert_eq!(
///     porosity.provides(),
///     ContextVariable::Coefficient { name: "porosity".into() }
/// );
/// let ctx = ComputeContext::new(0.0, 0.01);
/// let field = porosity.compute(&state, &ctx)?;
/// assert_eq!(field.as_scalar_field()?[4], 0.5);
///
/// // c(x, t): a source decaying in time, evaluated at the context's time.
/// let source = CoefficientField::of_x_t(mesh, "source", |x, t| x[0] * (-t).exp());
/// let ctx = ComputeContext::new(2.0, 0.01);
/// let field = source.compute(&state, &ctx)?;
/// assert_eq!(field.as_scalar_field()?[4], (-2.0_f64).exp());
/// # Ok::<(), OxiflowError>(())
/// ```
pub struct CoefficientField {
    mesh: Arc<dyn Mesh>,
    name: Cow<'static, str>,
    source: Source,
    /// Rayon dispatch threshold (DD-014, DD-048) — see `ParallelThreshold`.
    #[cfg(feature = "parallel")]
    parallel_threshold: ParallelThreshold,
}

impl CoefficientField {
    /// Creates a coefficient c(x), evaluated once on every node of `mesh`.
    ///
    /// `f` receives the coordinates of a node. It is not kept today; the
    /// `Send + Sync + 'static` bounds leave room for re-evaluating it on a
    /// changed mesh (DD-066) without a breaking change.
    ///
    /// # Errors
    ///
    /// [`OxiflowError::ExternalData`] if `f` returns a non-finite value on
    /// some node: the first such node is reported, with its coordinates.
    pub fn of_x<F>(
        mesh: Arc<dyn Mesh>,
        name: impl Into<Cow<'static, str>>,
        f: F,
    ) -> Result<Self, OxiflowError>
    where
        F: Fn(&[f64]) -> f64 + Send + Sync + 'static,
    {
        let name = name.into();
        #[cfg(feature = "parallel")]
        let parallel = mesh.n_dof() >= default_parallel_threshold();
        #[cfg(not(feature = "parallel"))]
        let parallel = false;

        let field = evaluate(&*mesh, &name, None, parallel, f)?;
        Ok(Self {
            mesh,
            name,
            source: Source::Space(Arc::new(field)),
            #[cfg(feature = "parallel")]
            parallel_threshold: ParallelThreshold::unset(),
        })
    }

    /// Creates a coefficient c(x, t), evaluated on every node of `mesh` at
    /// the time of each evaluation context.
    ///
    /// `f` receives the coordinates of a node and the time. Non-finite
    /// values are reported at evaluation, since they depend on t.
    pub fn of_x_t<F>(mesh: Arc<dyn Mesh>, name: impl Into<Cow<'static, str>>, f: F) -> Self
    where
        F: Fn(&[f64], f64) -> f64 + Send + Sync + 'static,
    {
        Self {
            mesh,
            name: name.into(),
            source: Source::SpaceTime {
                function: Box::new(f),
                last: RwLock::new(None),
            },
            #[cfg(feature = "parallel")]
            parallel_threshold: ParallelThreshold::unset(),
        }
    }

    /// Overrides the Rayon dispatch threshold at construction time (DD-014,
    /// DD-048). It governs c(x, t); c(x) has already been evaluated. Only
    /// exists when the crate is built with the `parallel` feature.
    ///
    /// # Panics
    ///
    /// Panics when `threshold == 0` — see `ParallelThreshold::set`.
    #[cfg(feature = "parallel")]
    pub fn with_parallel_threshold(self, threshold: usize) -> Self {
        self.parallel_threshold.set(threshold);
        self
    }

    /// Reconfigures the Rayon dispatch threshold at runtime, through `&self`.
    /// Scoped to this instance. Only exists when the crate is built with the
    /// `parallel` feature.
    ///
    /// # Panics
    ///
    /// Panics when `threshold == 0` — see `ParallelThreshold::set`.
    #[cfg(feature = "parallel")]
    pub fn set_parallel_threshold(&self, threshold: usize) {
        self.parallel_threshold.set(threshold);
    }

    /// Returns the effective Rayon dispatch threshold, in nodes: the
    /// configured override if one was set, otherwise an indicative default
    /// of 10 000 to be replaced by a measured value. Only exists when the
    /// crate is built with the `parallel` feature.
    #[cfg(feature = "parallel")]
    pub fn parallel_threshold(&self) -> usize {
        self.parallel_threshold.get(default_parallel_threshold())
    }

    /// Whether `n` nodes are evaluated through Rayon: `n >= threshold`, as
    /// at every dispatch site of the crate. Always `false` without the
    /// `parallel` feature.
    fn is_parallel_for(&self, n: usize) -> bool {
        #[cfg(feature = "parallel")]
        {
            n >= self.parallel_threshold()
        }
        #[cfg(not(feature = "parallel"))]
        {
            let _ = n;
            false
        }
    }

    /// Checks that `state` is a field with one row per mesh node.
    fn check_state(&self, state: &ContextValue) -> Result<(), OxiflowError> {
        let nodes = match state {
            ContextValue::ScalarField(v) => v.len(),
            ContextValue::VectorField(m) => m.nrows(),
            other => {
                return Err(OxiflowError::TypeMismatch {
                    expected: "ScalarField or VectorField",
                    actual: other.variant_name(),
                })
            }
        };
        let n_dof = self.mesh.n_dof();
        if nodes != n_dof {
            return Err(OxiflowError::PreconditionFailed {
                context: "CoefficientField",
                message: format!(
                    "coefficient field '{}': the state has {nodes} nodes, the mesh has \
                     n_dof() = {n_dof}",
                    self.name
                ),
            });
        }
        Ok(())
    }
}

/// Evaluates `f` on every node of `mesh` and checks that every value is
/// finite. `time` is only used in the error message.
fn evaluate<F>(
    mesh: &dyn Mesh,
    name: &str,
    time: Option<f64>,
    parallel: bool,
    f: F,
) -> Result<ContextValue, OxiflowError>
where
    F: Fn(&[f64]) -> f64 + Sync,
{
    let mut values = DVector::<f64>::zeros(mesh.n_dof());
    fill(values.as_mut_slice(), mesh, parallel, &f);

    match first_non_finite(values.as_slice(), parallel) {
        None => Ok(ContextValue::ScalarField(values)),
        Some(i) => {
            let at_time = time.map_or_else(String::new, |t| format!(", t = {t}"));
            Err(OxiflowError::ExternalData(format!(
                "coefficient field '{name}': non-finite value {} at node {i}, \
                 coordinates {:?}{at_time}",
                values[i],
                mesh.coordinates(i)
            )))
        }
    }
}

/// Writes `f(coordinates(i))` into `slots[i]`, through Rayon when
/// `parallel`, in order otherwise.
fn fill<F>(slots: &mut [f64], mesh: &dyn Mesh, parallel: bool, f: &F)
where
    F: Fn(&[f64]) -> f64 + Sync,
{
    #[cfg(feature = "parallel")]
    if parallel {
        use rayon::prelude::*;
        slots
            .par_iter_mut()
            .enumerate()
            .for_each(|(i, slot)| *slot = f(mesh.coordinates(i)));
        return;
    }
    #[cfg(not(feature = "parallel"))]
    debug_assert!(
        !parallel,
        "no parallel dispatch without the `parallel` feature"
    );

    for (i, slot) in slots.iter_mut().enumerate() {
        *slot = f(mesh.coordinates(i));
    }
}

/// Index of the first non-finite value, the lowest one in both modes.
fn first_non_finite(values: &[f64], parallel: bool) -> Option<usize> {
    #[cfg(feature = "parallel")]
    if parallel {
        use rayon::prelude::*;
        return values.par_iter().position_first(|v| !v.is_finite());
    }
    #[cfg(not(feature = "parallel"))]
    debug_assert!(
        !parallel,
        "no parallel dispatch without the `parallel` feature"
    );

    values.iter().position(|v| !v.is_finite())
}

impl std::fmt::Debug for CoefficientField {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let kind = match self.source {
            Source::Space(_) => "c(x)",
            Source::SpaceTime { .. } => "c(x, t)",
        };
        let mut s = f.debug_struct("CoefficientField");
        s.field("name", &self.name).field("kind", &kind);
        #[cfg(feature = "parallel")]
        s.field("parallel_threshold", &self.parallel_threshold());
        s.field("mesh_n_dof", &self.mesh.n_dof()).finish()
    }
}

impl RequiresContext for CoefficientField {
    fn required_variables(&self) -> Vec<ContextVariable> {
        vec![]
    }

    // External data, same rank as `ExternalTabulated`.
    fn priority(&self) -> u32 {
        50
    }
}

impl ContextCalculator for CoefficientField {
    fn provides(&self) -> ContextVariable {
        ContextVariable::Coefficient {
            name: self.name.clone(),
        }
    }

    /// Returns a copy of the field `compute_shared` hands over.
    fn compute(
        &self,
        state: &ContextValue,
        ctx: &ComputeContext,
    ) -> Result<ContextValue, OxiflowError> {
        self.compute_shared(state, ctx).map(Arc::unwrap_or_clone)
    }

    /// Hands over the kept field: the one evaluated at construction for
    /// c(x); for c(x, t), the one of the last evaluation if it was at the
    /// same time (compared bit for bit), otherwise a new one, which is kept.
    fn compute_shared(
        &self,
        state: &ContextValue,
        ctx: &ComputeContext,
    ) -> Result<Arc<ContextValue>, OxiflowError> {
        self.check_state(state)?;
        match &self.source {
            Source::Space(field) => Ok(Arc::clone(field)),
            Source::SpaceTime { function, last } => {
                let t = ctx.time();
                // The lock is only held to read or replace the pair, never
                // while evaluating (see the module documentation), so it
                // cannot be poisoned by a panic of `function`.
                let cached = last
                    .read()
                    .unwrap_or_else(PoisonError::into_inner)
                    .as_ref()
                    .filter(|(time, _)| time.to_bits() == t.to_bits())
                    .map(|(_, field)| Arc::clone(field));
                if let Some(field) = cached {
                    return Ok(field);
                }

                let parallel = self.is_parallel_for(self.mesh.n_dof());
                let field = Arc::new(evaluate(&*self.mesh, &self.name, Some(t), parallel, |x| {
                    function(x, t)
                })?);
                *last.write().unwrap_or_else(PoisonError::into_inner) =
                    Some((t, Arc::clone(&field)));
                Ok(field)
            }
        }
    }

    fn name(&self) -> &str {
        "coefficient_field (built-in)"
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mesh::{NodeCloud, UniformGrid1D};
    use nalgebra::DMatrix;
    use std::sync::atomic::{AtomicUsize, Ordering};

    fn grid(n: usize) -> Arc<dyn Mesh> {
        Arc::new(UniformGrid1D::new(n, 0.0, 1.0).unwrap())
    }

    fn scalar_state(n: usize) -> ContextValue {
        ContextValue::ScalarField(DVector::zeros(n))
    }

    fn values(value: &ContextValue) -> Vec<f64> {
        value.as_scalar_field().unwrap().iter().copied().collect()
    }

    fn bits(value: &ContextValue) -> Vec<u64> {
        values(value).iter().map(|v| v.to_bits()).collect()
    }

    // ── Exact values ──────────────────────────────────────────────────────────

    #[test]
    fn constant_coefficient_on_every_node() {
        let calc = CoefficientField::of_x(grid(7), "k", |_| 2.5).unwrap();
        let field = calc
            .compute(&scalar_state(7), &ComputeContext::new(0.0, 0.1))
            .unwrap();
        assert_eq!(values(&field), vec![2.5; 7]);
    }

    #[test]
    fn linear_coefficient_matches_the_node_coordinates() {
        let mesh = grid(11);
        let calc = CoefficientField::of_x(Arc::clone(&mesh), "k", |x| 1.0 + 2.0 * x[0]).unwrap();
        let field = calc
            .compute(&scalar_state(11), &ComputeContext::new(0.0, 0.1))
            .unwrap();
        let expected: Vec<f64> = (0..11)
            .map(|i| 1.0 + 2.0 * mesh.coordinates(i)[0])
            .collect();
        assert_eq!(values(&field), expected);
        assert_eq!(values(&field)[10], 3.0);
    }

    #[test]
    fn space_time_coefficient_uses_the_context_time() {
        let mesh = grid(5);
        let calc = CoefficientField::of_x_t(Arc::clone(&mesh), "q", |x, t| x[0] * t + 1.0);
        let field = calc
            .compute(&scalar_state(5), &ComputeContext::new(0.5, 0.1))
            .unwrap();
        let expected: Vec<f64> = (0..5).map(|i| mesh.coordinates(i)[0] * 0.5 + 1.0).collect();
        assert_eq!(values(&field), expected);
    }

    #[test]
    fn coordinates_of_a_two_dimensional_mesh_are_passed_whole() {
        let mesh: Arc<dyn Mesh> = Arc::new(NodeCloud::new(
            &[[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]],
            1.0,
        ));
        let calc = CoefficientField::of_x(mesh, "k", |x| x[0] + 10.0 * x[1]).unwrap();
        let field = calc
            .compute(&scalar_state(4), &ComputeContext::new(0.0, 0.1))
            .unwrap();
        assert_eq!(values(&field), vec![0.0, 1.0, 10.0, 11.0]);
    }

    #[test]
    fn vector_field_state_with_one_row_per_node_is_accepted() {
        let calc = CoefficientField::of_x(grid(4), "k", |_| 1.0).unwrap();
        let state = ContextValue::VectorField(DMatrix::zeros(4, 3));
        let field = calc
            .compute(&state, &ComputeContext::new(0.0, 0.1))
            .unwrap();
        assert_eq!(values(&field), vec![1.0; 4]);
    }

    // ── Errors ────────────────────────────────────────────────────────────────

    #[test]
    fn non_finite_space_coefficient_fails_at_construction_on_the_first_node() {
        let err =
            CoefficientField::of_x(grid(11), "k", |x| if x[0] > 0.55 { f64::NAN } else { 1.0 })
                .unwrap_err();
        let OxiflowError::ExternalData(message) = err else {
            panic!("expected ExternalData, got {err:?}");
        };
        assert!(message.contains("'k'"), "{message}");
        assert!(message.contains("node 6,"), "{message}");
        assert!(message.contains("[0.6"), "{message}");
        assert!(!message.contains("t ="), "{message}");
    }

    #[test]
    fn infinite_value_is_reported_as_well() {
        let result = CoefficientField::of_x(grid(3), "k", |_| f64::INFINITY);
        assert!(matches!(result, Err(OxiflowError::ExternalData(_))));
    }

    #[test]
    fn non_finite_space_time_coefficient_fails_at_evaluation_with_its_time() {
        let calc = CoefficientField::of_x_t(grid(5), "q", |_, t| 1.0 / (t - 1.0));
        let state = scalar_state(5);
        assert!(calc.compute(&state, &ComputeContext::new(0.0, 0.1)).is_ok());

        for _ in 0..2 {
            let err = calc
                .compute(&state, &ComputeContext::new(1.0, 0.1))
                .unwrap_err();
            let OxiflowError::ExternalData(message) = err else {
                panic!("expected ExternalData, got {err:?}");
            };
            assert!(message.contains("node 0,"), "{message}");
            assert!(message.contains("t = 1"), "{message}");
        }
    }

    #[test]
    fn state_with_another_node_count_is_rejected() {
        let calc = CoefficientField::of_x(grid(5), "k", |_| 1.0).unwrap();
        let ctx = ComputeContext::new(0.0, 0.1);
        for state in [
            scalar_state(4),
            ContextValue::VectorField(DMatrix::zeros(6, 2)),
        ] {
            let result = calc.compute_shared(&state, &ctx);
            assert!(
                matches!(
                    result,
                    Err(OxiflowError::PreconditionFailed {
                        context: "CoefficientField",
                        ..
                    })
                ),
                "{result:?}"
            );
        }
    }

    #[test]
    fn state_that_is_not_a_field_is_rejected() {
        let calc = CoefficientField::of_x_t(grid(5), "q", |_, _| 1.0);
        let result = calc.compute(&ContextValue::Scalar(1.0), &ComputeContext::new(0.0, 0.1));
        assert!(matches!(
            result,
            Err(OxiflowError::TypeMismatch {
                actual: "Scalar",
                ..
            })
        ));
    }

    // ── Sharing and caching ───────────────────────────────────────────────────

    #[test]
    fn space_coefficient_hands_over_the_same_allocation() {
        let calc = CoefficientField::of_x(grid(5), "k", |x| x[0]).unwrap();
        let state = scalar_state(5);
        let first = calc
            .compute_shared(&state, &ComputeContext::new(0.0, 0.1))
            .unwrap();
        let later = calc
            .compute_shared(&state, &ComputeContext::new(3.0, 0.1))
            .unwrap();
        assert!(Arc::ptr_eq(&first, &later));
    }

    #[test]
    fn space_time_coefficient_is_evaluated_once_per_time() {
        let calls = Arc::new(AtomicUsize::new(0));
        let counter = Arc::clone(&calls);
        let calc = CoefficientField::of_x_t(grid(5), "q", move |x, t| {
            counter.fetch_add(1, Ordering::Relaxed);
            x[0] + t
        });
        let state = scalar_state(5);

        let first = calc
            .compute_shared(&state, &ComputeContext::new(0.5, 0.1))
            .unwrap();
        let again = calc
            .compute_shared(&state, &ComputeContext::new(0.5, 0.2))
            .unwrap();
        assert!(Arc::ptr_eq(&first, &again));
        assert_eq!(calls.load(Ordering::Relaxed), 5);

        let next = calc
            .compute_shared(&state, &ComputeContext::new(0.75, 0.1))
            .unwrap();
        assert!(!Arc::ptr_eq(&first, &next));
        assert_eq!(calls.load(Ordering::Relaxed), 10);
        assert_eq!(values(&next)[0], 0.75);

        // Going back to an earlier time evaluates again, with its values.
        let back = calc
            .compute_shared(&state, &ComputeContext::new(0.5, 0.1))
            .unwrap();
        assert_eq!(bits(&back), bits(&first));
        assert_eq!(calls.load(Ordering::Relaxed), 15);
    }

    #[test]
    fn compute_returns_the_value_compute_shared_hands_over() {
        let state = scalar_state(6);
        let ctx = ComputeContext::new(0.25, 0.1);
        let calcs = [
            CoefficientField::of_x(grid(6), "k", |x| x[0] * x[0]).unwrap(),
            CoefficientField::of_x_t(grid(6), "q", |x, t| x[0] - t),
        ];
        for calc in &calcs {
            let shared = calc.compute_shared(&state, &ctx).unwrap();
            assert_eq!(calc.compute(&state, &ctx).unwrap(), *shared);
        }
    }

    // ── Calculator metadata ───────────────────────────────────────────────────

    #[test]
    fn metadata_of_the_calculator() {
        let calc = CoefficientField::of_x_t(grid(3), "porosity", |_, _| 0.4);
        assert_eq!(
            calc.provides(),
            ContextVariable::Coefficient {
                name: "porosity".into()
            }
        );
        assert!(calc.required_variables().is_empty());
        assert_eq!(calc.priority(), 50);
        assert_eq!(calc.name(), "coefficient_field (built-in)");
        let debug = format!("{calc:?}");
        assert!(
            debug.contains("porosity") && debug.contains("c(x, t)"),
            "{debug}"
        );
    }

    // ── Parallel dispatch ─────────────────────────────────────────────────────

    #[cfg(feature = "parallel")]
    mod parallel {
        use super::*;

        #[test]
        fn threshold_defaults_and_overrides() {
            let calc = CoefficientField::of_x_t(grid(3), "q", |_, _| 1.0);
            assert_eq!(calc.parallel_threshold(), default_parallel_threshold());
            calc.set_parallel_threshold(7);
            assert_eq!(calc.parallel_threshold(), 7);
            let calc = calc.with_parallel_threshold(9);
            assert_eq!(calc.parallel_threshold(), 9);
            assert!(!calc.is_parallel_for(8));
            assert!(calc.is_parallel_for(9));
        }

        #[test]
        fn sequential_and_parallel_fields_are_bit_identical() {
            let n = 2_001;
            let state = scalar_state(n);
            let ctx = ComputeContext::new(0.3, 0.1);
            let f = |x: &[f64], t: f64| (x[0] * 7.0).sin() * (-t * x[0]).exp() + x[0].ln_1p();
            let sequential =
                CoefficientField::of_x_t(grid(n), "q", f).with_parallel_threshold(usize::MAX);
            let parallel = CoefficientField::of_x_t(grid(n), "q", f).with_parallel_threshold(1);
            assert!(!sequential.is_parallel_for(n) && parallel.is_parallel_for(n));
            assert_eq!(
                bits(&sequential.compute(&state, &ctx).unwrap()),
                bits(&parallel.compute(&state, &ctx).unwrap())
            );
        }

        #[test]
        fn parallel_evaluation_reports_the_first_non_finite_node() {
            let n = 2_001;
            let calc =
                CoefficientField::of_x_t(
                    grid(n),
                    "q",
                    |x, _| {
                        if x[0] >= 0.2502 {
                            f64::NAN
                        } else {
                            0.0
                        }
                    },
                )
                .with_parallel_threshold(1);
            let err = calc
                .compute(&scalar_state(n), &ComputeContext::new(0.0, 0.1))
                .unwrap_err();
            let OxiflowError::ExternalData(message) = err else {
                panic!("expected ExternalData, got {err:?}");
            };
            assert!(message.contains("node 501,"), "{message}");
        }
    }
}
