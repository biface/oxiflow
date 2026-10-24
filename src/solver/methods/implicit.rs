//! # Module `solver::methods::implicit`
//!
//! Shared machinery for implicit (theta-method) integrators — DD-033
//! ([#86](https://github.com/biface/oxiflow/issues/86)).
//!
//! [`BackwardEulerSolver`](super::backward_euler::BackwardEulerSolver) (θ=1)
//! and [`CrankNicolsonSolver`](super::crank_nicolson::CrankNicolsonSolver)
//! (θ=0.5) are thin wrappers around `theta_method_step_newton` (DD-044,
//! [#111](https://github.com/biface/oxiflow/issues/111)), which itself relies on `finite_difference_jacobian` and the
//! shared [`super::newton`] loop. `theta_method_step` keeps the
//! original, non-configurable entry point for direct callers — see below.
//!
//! ## Why a single shared function for two methods
//!
//! The generalised theta method is:
//!
//! $$u^{n+1} = u^n + \Delta t\left[(1-\theta) f(u^n) + \theta f(u^{n+1})\right]$$
//!
//! Linearising the residual $g(u) = u - u^n - \Delta t[(1-\theta)f(u^n) +
//! \theta f(u)]$ at $u = u^n$ gives $g(u^n) = -\Delta t \cdot f(u^n)$ for
//! **any** $\theta$ — the right-hand side of the linear correction doesn't
//! depend on θ at all. Only the system matrix does
//! ($I - \theta \Delta t J_f$). One function, one parameter.
//!
//! ## v1 scope — frozen Jacobian, single correction (DD-033,
//! [#86](https://github.com/biface/oxiflow/issues/86)) by default
//!
//! `theta_method_step` performs exactly **one** Newton-style correction,
//! with the Jacobian frozen at $u^n$ — this is exact when `compute_physics`
//! is affine in `u` and a first-order approximation otherwise, and stays
//! the behaviour of `theta_method_step` itself (used directly by
//! callers that don't need more) and of
//! [`BackwardEulerSolver`](super::backward_euler::BackwardEulerSolver)/
//! [`CrankNicolsonSolver`](super::crank_nicolson::CrankNicolsonSolver)
//! when left unconfigured.
//!
//! `theta_method_step_newton` (DD-044, [#111](https://github.com/biface/oxiflow/issues/111)) is the genuinely iterated
//! alternative DD-033 anticipated: it routes through the shared
//! [`super::newton`] loop with caller-supplied convergence/Jacobian-
//! strategy/iteration-budget configuration.
//! `BackwardEulerSolver`/`CrankNicolsonSolver` call it directly (with
//! `with_newton_convergence`/`with_jacobian_strategy`/
//! `with_max_newton_iterations` builders); `theta_method_step` itself
//! keeps its original signature, plus the Jacobian dispatch configuration
//! (DT-069, #198), by delegating to it with
//! `k_max = 1`/[`JacobianStrategy::ModifiedFrozen`]
//! baked in, reproducing its historical output exactly (see
//! [`super::newton`]'s own docs for why no special-casing is needed for
//! this to hold on affine problems).
//!
//! ## Known gap — sparse path not yet Newton-aware
//!
//! `theta_method_step_adaptive` (the sparse/banded dispatch, DD-043)
//! still performs a single frozen-Jacobian correction unconditionally —
//! it is not wired through [`super::newton`] at this increment. Composing
//! the two was verified, not implemented, per [#111](https://github.com/biface/oxiflow/issues/111)'s stated scope:
//! configuring both a sparse backend *and* a non-default Newton budget on
//! the same solver silently keeps the sparse path single-shot. Tracked as
//! a follow-up, mirroring the existing BDF2/sparse gap.
//!
//! ## State variants
//!
//! Newton and both finite-difference Jacobians work on the flat vector of
//! unknowns given by `StateLayout` (`super::state_layout`): a `ScalarField`
//! as is, a `VectorField` interleaved by node, so a `VectorField` state is
//! advanced like a `ScalarField` (#192). The banded path therefore takes a
//! half-bandwidth counted in flat unknowns, `d·(s + 1) − 1` for `d`
//! components and a stencil of half-width `s`. For a `ScalarField`,
//! results are bit-identical to the former code.
//!
//! ## Boundary conditions × the perturbed Jacobian
//!
//! `finite_difference_jacobian` perturbs each component of the state and
//! re-evaluates `evaluate_derivative`, which re-applies boundary
//! conditions on every call. A Dirichlet-constrained node will have its
//! perturbation overwritten before `compute_physics` sees it — physically
//! correct (a BC-constrained node isn't a free unknown): the constrained
//! node's own *column* in the assembled Jacobian is exactly zero, while
//! its *row* still reflects genuine physics-based coupling to
//! unconstrained neighbours. Verified by
//! `tests::dirichlet_constrained_column_is_exactly_zero_row_keeps_real_coupling`
//! (#113) — previously documented as a known untested limitation (DD-033,
//! [#86](https://github.com/biface/oxiflow/issues/86), v0.4.0).
//!
//! ## Parallel assembly (DT-069, #198)
//!
//! With the `parallel` feature, both finite-difference Jacobians dispatch
//! their independent evaluations through Rayon — one per column for the
//! dense one, one per color for the banded one — once the system has at
//! least as many unknowns as the solver's dispatch threshold. Each Rayon
//! job perturbs its own work state and writes its columns in place, in the
//! final matrix. Every column is computed from the same operands with the
//! same operations in both modes, so the sequential and parallel Jacobians
//! are bit-for-bit identical. The threshold belongs to each implicit
//! solver (`JacobianDispatch`); without the feature it does not exist and
//! the assembly is sequential.

use nalgebra::{DMatrix, DVector};

use super::evaluate_derivative;
use super::newton::{self, JacobianStrategy, NewtonConvergence};
use super::state_layout::StateLayout;
use crate::context::error::OxiflowError;
use crate::context::value::ContextValue;
use crate::context::ContextCalculator;
use crate::solver::linear::LinearSolver;
#[cfg(feature = "parallel")]
use crate::solver::parallel::ParallelThreshold;
use crate::solver::scenario::Domain;
#[cfg(feature = "sparse")]
use crate::solver::sparse::SparseLinearSolver;
#[cfg(feature = "sparse")]
use faer::sparse::{SparseColMat, SymbolicSparseColMat};

/// Finite-difference step size for [`finite_difference_jacobian`].
///
/// Not scaled by state magnitude (v1 simplification) — fine for the O(1)
/// stiff linear test problems this targets; revisit if used on
/// significantly different magnitude scales.
const FD_EPSILON: f64 = 1e-7;

/// Default Rayon dispatch threshold of the Jacobian assembly, in unknowns
/// (DT-069, #198) — an indicative value, shared by the three implicit
/// solvers since they share this assembly. Each dispatched task is a full
/// derivative evaluation, much heavier than one stencil point, so the
/// crossover is expected far below
/// [`crate::operators::fd::default_parallel_threshold`] (49 999). Measure it
/// on the target workload with the `oxiflow_parallel` benchmarks and set
/// it through the solver's `with_parallel_threshold`.
#[cfg(feature = "parallel")]
pub(crate) fn default_parallel_threshold() -> usize {
    256
}

/// Dispatch configuration of the Jacobian assembly, held by each implicit
/// solver and passed down to [`finite_difference_jacobian`] and, with the
/// `sparse` feature, `banded_finite_difference_jacobian` (DT-069, #198).
///
/// Holds a `ParallelThreshold` only with the `parallel` feature: without
/// it the type is empty, no threshold is representable, and the assembly
/// is sequential. Passing this configuration rather than a bare number
/// keeps the threshold out of the feature-agnostic signatures, without one
/// version of every θ-method function per feature.
#[derive(Debug, Default)]
pub(crate) struct JacobianDispatch {
    #[cfg(feature = "parallel")]
    threshold: ParallelThreshold,
}

impl JacobianDispatch {
    /// Whether the assembly of a system of `n` unknowns goes through Rayon:
    /// `n >= threshold`, the comparison used by every dispatch site of the
    /// crate. Always `false` without the `parallel` feature.
    pub(crate) fn is_parallel_for(&self, n: usize) -> bool {
        #[cfg(feature = "parallel")]
        {
            n >= self.threshold()
        }
        #[cfg(not(feature = "parallel"))]
        {
            let _ = n;
            false
        }
    }

    /// Effective threshold: the override if one was set, otherwise
    /// [`default_parallel_threshold`].
    #[cfg(feature = "parallel")]
    pub(crate) fn threshold(&self) -> usize {
        self.threshold.get(default_parallel_threshold())
    }

    /// Overrides the threshold.
    ///
    /// # Panics
    ///
    /// Panics when `threshold == 0` — see `ParallelThreshold::set`.
    #[cfg(feature = "parallel")]
    pub(crate) fn set(&self, threshold: usize) {
        self.threshold.set(threshold);
    }
}

/// Runs `fill` on every item, each call with a work state cloned from
/// `state` that `fill` refills and perturbs itself.
///
/// Through Rayon when `parallel`, with one work state per Rayon job
/// (`try_for_each_init` initialises once per job, not once per thread):
/// of the order of the number of threads, not of items;
/// otherwise in order, with one work state for all items. Items must be
/// independent of one another: each writes only its own columns, from
/// its own work state, so the result does not depend on the dispatch.
fn try_for_each_with_work<T, F>(
    items: Vec<T>,
    parallel: bool,
    state: &ContextValue,
    fill: F,
) -> Result<(), OxiflowError>
where
    T: Send,
    F: Fn(&mut ContextValue, T) -> Result<(), OxiflowError> + Sync,
{
    #[cfg(feature = "parallel")]
    if parallel {
        use rayon::prelude::*;
        return items
            .into_par_iter()
            .try_for_each_init(|| state.clone(), |work, item| fill(work, item));
    }
    #[cfg(not(feature = "parallel"))]
    debug_assert!(
        !parallel,
        "no parallel dispatch without the `parallel` feature"
    );

    let mut work = state.clone();
    items.into_iter().try_for_each(|item| fill(&mut work, item))
}

/// Estimates $\partial f/\partial u$ at `state`, `t`, by forward
/// differences.
///
/// For `f` affine in `u` (the case these implicit methods are validated
/// against), forward differences are exact regardless of step size — no
/// truncation error from the linear approximation itself.
///
/// Rows and columns follow the flat ordering of [`StateLayout`]: for a
/// `VectorField` state, unknowns are interleaved by node (#192).
///
/// A work state, cloned from `state`, is refilled before each column and
/// perturbed in place (#197): the boundary conditions modify the evaluated
/// state in place and do not report which nodes they changed, so a full
/// refill per column is needed, but no allocation is. Columns are
/// dispatched according to `dispatch` — see
/// [parallel assembly](self#parallel-assembly-dt-069-198).
///
/// See [boundary conditions × the perturbed Jacobian](self#boundary-conditions--the-perturbed-jacobian).
pub(crate) fn finite_difference_jacobian(
    domain: &Domain,
    chain: &[&dyn ContextCalculator],
    state: &ContextValue,
    t: f64,
    dt: f64,
    dispatch: &JacobianDispatch,
) -> Result<DMatrix<f64>, OxiflowError> {
    let layout = StateLayout::of(state)?;
    let base = layout.values(state)?;
    let n = layout.len();

    let mut base_state = state.clone();
    let f0 = evaluate_derivative(domain, chain, &mut base_state, t, dt)?;
    let f0_field = layout.flatten(&f0)?.into_owned();

    let mut jacobian = DMatrix::<f64>::zeros(n, n);
    if n == 0 {
        return Ok(jacobian);
    }

    // Column-major storage: column j is the j-th chunk of n values.
    let columns: Vec<(usize, &mut [f64])> =
        jacobian.as_mut_slice().chunks_mut(n).enumerate().collect();

    try_for_each_with_work(
        columns,
        dispatch.is_parallel_for(n),
        state,
        |work, (j, column)| {
            let values = layout.values_mut(work)?;
            values.copy_from_slice(base);
            values[layout.storage_index(j)] += FD_EPSILON;

            let f_j = evaluate_derivative(domain, chain, work, t, dt)?;
            let f_j_values = layout.values(&f_j)?;

            for (i, entry) in column.iter_mut().enumerate() {
                *entry = (f_j_values[layout.storage_index(i)] - f0_field[i]) / FD_EPSILON;
            }
            Ok(())
        },
    )?;

    Ok(jacobian)
}

/// Convergence criterion guaranteeing [`theta_method_step`]'s
/// single-correction contract succeeds on the stiffest affine problems
/// this crate currently tests (`λ = 1e4`-scale exponential decay).
///
/// Deliberately distinct from [`NewtonConvergence::default`] (`tol_abs =
/// 1e-8`): on such problems, [`finite_difference_jacobian`]'s fixed step
/// size suffers catastrophic-cancellation error whose absolute magnitude
/// scales roughly as `λ · ε_machine / FD_EPSILON` — empirically up to
/// ~1e-6 for the stiffest cases tested. The single correction itself is
/// still mathematically exact for affine `f`; this is a
/// finite-difference precision floor, not a nonlinearity signal, and
/// [`NewtonConvergence::default`] remains the right general-purpose
/// starting point for callers configuring the loop themselves — it just
/// isn't loose enough to guarantee success on this specific,
/// already-tested edge of the stiffness range. Used here and by
/// [`BackwardEulerSolver::default`](super::backward_euler::BackwardEulerSolver)/
/// [`CrankNicolsonSolver::default`](super::crank_nicolson::CrankNicolsonSolver)
/// — not by `NewtonConvergence`'s own `Default` impl — precisely so that
/// zero-config solver construction keeps reproducing history exactly,
/// without changing what a fresh, explicitly-configured
/// `NewtonConvergence::default()` means elsewhere.
pub(crate) fn stiff_jacobian_convergence() -> NewtonConvergence {
    NewtonConvergence::ResidualOnly {
        tol_abs: 1e-5,
        tol_rel: 1e-6,
    }
}

/// Performs one step of the generalised theta method, with the Jacobian
/// frozen at `state` (one Newton-style correction, not iterated — see
/// [module docs](self)).
///
/// `theta = 1.0` is Backward Euler; `theta = 0.5` is Crank-Nicolson.
///
/// `state` is mutated in-place by boundary condition application — same
/// contract as the explicit solvers (see
/// [`crate::solver::methods::evaluate_derivative`]): callers should not
/// assume `state` is left unchanged after this call.
///
/// Delegates to [`theta_method_step_newton`] with the pre-DD-044 defaults
/// (`k_max = 1`, [`JacobianStrategy::ModifiedFrozen`],
/// [`stiff_jacobian_convergence`]) baked in — kept at its original
/// signature (DD-044, #110) for direct callers that don't need the
/// configurable path, plus the Jacobian dispatch configuration of the
/// calling solver (DT-069, #198).
#[allow(clippy::too_many_arguments)]
pub(crate) fn theta_method_step(
    domain: &Domain,
    chain: &[&dyn ContextCalculator],
    state: &mut ContextValue,
    t: f64,
    dt: f64,
    theta: f64,
    linear_solver: &dyn LinearSolver,
    dispatch: &JacobianDispatch,
) -> Result<ContextValue, OxiflowError> {
    theta_method_step_newton(
        domain,
        chain,
        state,
        t,
        dt,
        theta,
        linear_solver,
        stiff_jacobian_convergence(),
        JacobianStrategy::default(),
        1,
        dispatch,
    )
}

/// Same contract as [`theta_method_step`], routed through the generic
/// iterated Newton loop ([`newton::solve`]) with caller-supplied
/// convergence/Jacobian-strategy/iteration-budget configuration (DD-044,
/// #111) — [`BackwardEulerSolver`](super::backward_euler::BackwardEulerSolver)
/// and [`CrankNicolsonSolver`](super::crank_nicolson::CrankNicolsonSolver)
/// call this directly, passing their own configured fields; their
/// defaults reproduce [`theta_method_step`]'s call above exactly.
///
/// Follows [`newton`]'s residual convention with $\alpha = 1$, $c = \theta
/// \Delta t$, $u^{*} = u^n + \Delta t(1-\theta)f(u^n)$ — see that
/// module's docs. `f(u^n)` is evaluated once, at time `t` (the known
/// term); every subsequent candidate `f`/Jacobian evaluation inside the
/// Newton loop is at time `t + dt` (the implicit/target time), matching
/// [`theta_method_step`]'s original convention.
#[allow(clippy::too_many_arguments)]
pub(crate) fn theta_method_step_newton(
    domain: &Domain,
    chain: &[&dyn ContextCalculator],
    state: &mut ContextValue,
    t: f64,
    dt: f64,
    theta: f64,
    linear_solver: &dyn LinearSolver,
    newton_convergence: NewtonConvergence,
    jacobian_strategy: JacobianStrategy,
    max_iterations: usize,
    dispatch: &JacobianDispatch,
) -> Result<ContextValue, OxiflowError> {
    // f(u^n, t) -- BCs applied in-place to `state` itself, consistent with
    // the explicit solvers' contract (same as `theta_method_step`).
    let f_n = evaluate_derivative(domain, chain, state, t, dt)?;
    // Newton works on the flat vector of unknowns (#192).
    let layout = StateLayout::of(state)?;
    let u_n_field = layout.flatten(state)?.into_owned();
    let f_n_field = layout.flatten(&f_n)?.into_owned();

    let u0 = u_n_field.clone();
    let u_star: DVector<f64> = u_n_field + f_n_field * (dt * (1.0 - theta));
    let coeff = theta * dt;
    let t_target = t + dt;

    let u_next = newton::solve(
        u0,
        &u_star,
        1.0,
        coeff,
        move |candidate: &DVector<f64>| {
            let mut candidate_state = layout.rebuild(candidate.clone())?;
            let f = evaluate_derivative(domain, chain, &mut candidate_state, t_target, dt)?;
            Ok(layout.flatten(&f)?.into_owned())
        },
        move |candidate: &DVector<f64>| {
            let candidate_state = layout.rebuild(candidate.clone())?;
            finite_difference_jacobian(domain, chain, &candidate_state, t_target, dt, dispatch)
        },
        linear_solver,
        &newton::NewtonParams {
            convergence: newton_convergence,
            jacobian_strategy,
            max_iterations,
        },
    )?;

    layout.rebuild(u_next)
}

// ── Sparse / banded path (DD-013 second phase, DD-043) ─────────────────────────
//
// `#[cfg(feature = "sparse")]` only. Bandwidth is always an explicit
// parameter here, supplied by the caller (the solver's own HOW-side
// configuration, e.g. `BackwardEulerSolver::with_jacobian_bandwidth`) — it
// is never read from `Domain`/`PhysicalModel`. See DD-043, amendment 1, for
// why `PhysicalModel::jacobian_bandwidth` was rejected: bandwidth is a
// property of the numerical scheme (HOW), not of the physical model (WHAT).

/// Assembles $\partial f/\partial u$ within a band of half-width
/// `bandwidth` around the diagonal, directly in compressed-column form, in
/// the flat ordering of [`StateLayout`] (#192): for a `VectorField` of d
/// components and a stencil of half-width `s`, the band half-width is
/// `d·(s + 1) − 1`.
///
/// Columns are grouped by Curtis-Powell-Reid (CPR) coloring: for a
/// genuinely banded Jacobian, columns `2*bandwidth + 1` or more apart never
/// share a nonzero row, so all columns of the same color are perturbed
/// **simultaneously** in a single [`evaluate_derivative`] call —
/// `2*bandwidth + 1` evaluations, instead of [`finite_difference_jacobian`]'s
/// `n`. As there, a work state is refilled and perturbed in place for each
/// color (#197), and colors are dispatched according to `dispatch` — the
/// threshold is compared with the number of unknowns `n`, not of colors.
///
/// The structure is known in advance (DT-069, #198): column `j` holds rows
/// `max(0, j − bandwidth)` to `min(n, j + bandwidth + 1)`, in increasing
/// order, diagonal included. `col_ptr` and `row_idx` are built from `n` and
/// `bandwidth`; each column of `values` is written once, in place, with the
/// same arithmetic as [`finite_difference_jacobian`] — no triplets, no
/// sort. Every entry of the band is stored, zeros included.
#[cfg(feature = "sparse")]
pub(crate) fn banded_finite_difference_jacobian(
    domain: &Domain,
    chain: &[&dyn ContextCalculator],
    state: &ContextValue,
    t: f64,
    dt: f64,
    bandwidth: usize,
    dispatch: &JacobianDispatch,
) -> Result<SparseColMat<usize, f64>, OxiflowError> {
    let layout = StateLayout::of(state)?;
    let base = layout.values(state)?;
    let n = layout.len();

    let mut base_state = state.clone();
    let f0 = evaluate_derivative(domain, chain, &mut base_state, t, dt)?;
    let f0_field = layout.flatten(&f0)?.into_owned();

    let rows_of_column = |j: usize| j.saturating_sub(bandwidth)..(j + bandwidth + 1).min(n);
    let mut col_ptr = Vec::with_capacity(n + 1);
    let mut row_idx = Vec::new();
    col_ptr.push(0);
    for j in 0..n {
        row_idx.extend(rows_of_column(j));
        col_ptr.push(row_idx.len());
    }
    let mut values = vec![0.0; row_idx.len()];

    // One group of (column, column storage) per color; column `j` has
    // color `j % n_colors`, and colors from `n` on have no column.
    let n_colors = 2 * bandwidth + 1;
    let mut colors: Vec<Vec<(usize, &mut [f64])>> =
        (0..n_colors.min(n)).map(|_| Vec::new()).collect();
    let mut rest = values.as_mut_slice();
    for j in 0..n {
        let (column, tail) = std::mem::take(&mut rest).split_at_mut(col_ptr[j + 1] - col_ptr[j]);
        colors[j % n_colors].push((j, column));
        rest = tail;
    }

    try_for_each_with_work(colors, dispatch.is_parallel_for(n), state, |work, color| {
        let values = layout.values_mut(work)?;
        values.copy_from_slice(base);
        for (j, _) in &color {
            values[layout.storage_index(*j)] += FD_EPSILON;
        }

        let f_c = evaluate_derivative(domain, chain, work, t, dt)?;
        let f_c_values = layout.values(&f_c)?;

        // The bands of the columns of one color ([j-bandwidth,
        // j+bandwidth]) never overlap: each row's response is attributable
        // to exactly one column.
        for (j, column) in color {
            for (i, entry) in rows_of_column(j).zip(column.iter_mut()) {
                *entry = (f_c_values[layout.storage_index(i)] - f0_field[i]) / FD_EPSILON;
            }
        }
        Ok(())
    })?;

    let symbolic = SymbolicSparseColMat::new_checked(n, n, col_ptr, None, row_idx);
    Ok(SparseColMat::new(symbolic, values))
}

/// Same contract as [`theta_method_step`], with an additional sparse path:
/// when `jacobian_bandwidth` is `Some(k)` **and** `n > sparse_threshold`,
/// assembles `I - theta*dt*J` directly in sparse (banded) form via
/// [`banded_finite_difference_jacobian`] and solves it with `sparse_solver`. Falls
/// back to [`theta_method_step`] unchanged in every other case — `None`
/// bandwidth, or a small enough system that the dense path is cheaper
/// regardless of bandwidth.
///
/// `jacobian_bandwidth`/`sparse_threshold`/`sparse_solver`/`dispatch` are
/// supplied by the caller's own HOW-side configuration (DD-043, amendment
/// 1; DT-069) — never read from `Domain` or `PhysicalModel`. `dispatch`
/// applies to both paths.
#[cfg(feature = "sparse")]
#[allow(clippy::too_many_arguments)]
pub(crate) fn theta_method_step_adaptive(
    domain: &Domain,
    chain: &[&dyn ContextCalculator],
    state: &mut ContextValue,
    t: f64,
    dt: f64,
    theta: f64,
    dense_solver: &dyn LinearSolver,
    sparse_solver: &dyn SparseLinearSolver,
    sparse_threshold: usize,
    jacobian_bandwidth: Option<usize>,
    dispatch: &JacobianDispatch,
) -> Result<ContextValue, OxiflowError> {
    let layout = StateLayout::of(state)?;
    let n = layout.len();

    match jacobian_bandwidth {
        Some(k) if n > sparse_threshold => {
            let f_n = evaluate_derivative(domain, chain, state, t, dt)?;
            let u_n_field = layout.flatten(state)?.into_owned();
            let f_n_field = layout.flatten(&f_n)?.into_owned();

            // Jacobian frozen at the (now BC-corrected) u^n, evaluated at
            // t + dt — same convention as theta_method_step.
            let mut system_matrix =
                banded_finite_difference_jacobian(domain, chain, state, t + dt, dt, k, dispatch)?;

            // I - theta*dt*J, folded in place into the values of J. Column j
            // starts at row max(0, j - k) and always holds its diagonal
            // entry, at offset j - max(0, j - k).
            let coeff = -theta * dt;
            for j in 0..n {
                let column = system_matrix.val_of_col_mut(j);
                for value in column.iter_mut() {
                    *value *= coeff;
                }
                column[j - j.saturating_sub(k)] += 1.0;
            }

            let rhs = f_n_field * dt;
            let delta_u = sparse_solver.solve(&system_matrix, &rhs)?;

            let u_next: DVector<f64> = u_n_field + delta_u;
            layout.rebuild(u_next)
        }
        _ => theta_method_step(domain, chain, state, t, dt, theta, dense_solver, dispatch),
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::context::compute::ComputeContext;
    use crate::context::variable::ContextVariable;
    use crate::mesh::{Mesh, UniformGrid1D};
    use crate::model::traits::{PhysicalModel, RequiresContext};
    use crate::solver::chain::build_calculator_chain;
    use crate::solver::linear::NalgebraDenseSolver;
    use crate::solver::scenario::Scenario;

    #[derive(Debug)]
    struct ExponentialDecay {
        lambda: f64,
    }

    impl RequiresContext for ExponentialDecay {
        fn required_variables(&self) -> Vec<ContextVariable> {
            vec![]
        }
    }

    impl PhysicalModel for ExponentialDecay {
        fn compute_physics(
            &self,
            state: &ContextValue,
            _ctx: &ComputeContext,
        ) -> Result<ContextValue, OxiflowError> {
            let u = state.as_scalar_field()?;
            Ok(ContextValue::ScalarField(u.map(|v| -self.lambda * v)))
        }

        fn initial_state(&self, mesh: &dyn Mesh) -> ContextValue {
            ContextValue::ScalarField(DVector::from_element(mesh.n_dof(), 1.0))
        }

        fn name(&self) -> &str {
            "exponential_decay"
        }
    }

    fn make_mesh(n: usize) -> Box<dyn Mesh> {
        Box::new(UniformGrid1D::new(n, 0.0, 1.0).unwrap())
    }

    /// Two-component `VectorField` model: a reaction matrix couples the
    /// components of each node, and each component diffuses to its
    /// neighbours (zero outside the domain). Affine in `u`. In the
    /// node-interleaved order its Jacobian has half-bandwidth
    /// `d·(s + 1) − 1 = 3` (d = 2, s = 1). Also used by `sparse_tests`.
    #[derive(Debug)]
    pub(super) struct CoupledReactionDiffusion {
        pub(super) reaction: [[f64; 2]; 2],
        pub(super) diffusion: f64,
    }

    impl CoupledReactionDiffusion {
        pub(super) const COMPONENTS: usize = 2;

        /// Exact Jacobian entry between unknown `(node, component)` pairs.
        fn exact_entry(&self, (i, c): (usize, usize), (m, k): (usize, usize)) -> f64 {
            let reaction = if m == i { self.reaction[c][k] } else { 0.0 };
            let diffusion = match (k == c, i.abs_diff(m)) {
                (true, 0) => -2.0 * self.diffusion,
                (true, 1) => self.diffusion,
                _ => 0.0,
            };
            reaction + diffusion
        }
    }

    impl RequiresContext for CoupledReactionDiffusion {
        fn required_variables(&self) -> Vec<ContextVariable> {
            vec![]
        }
    }

    impl PhysicalModel for CoupledReactionDiffusion {
        fn compute_physics(
            &self,
            state: &ContextValue,
            _ctx: &ComputeContext,
        ) -> Result<ContextValue, OxiflowError> {
            let u = state.as_vector_field()?;
            let n = u.nrows();
            let out = DMatrix::from_fn(n, Self::COMPONENTS, |i, c| {
                let reaction = self.reaction[c][0] * u[(i, 0)] + self.reaction[c][1] * u[(i, 1)];
                let left = if i > 0 { u[(i - 1, c)] } else { 0.0 };
                let right = if i + 1 < n { u[(i + 1, c)] } else { 0.0 };
                reaction + self.diffusion * (left - 2.0 * u[(i, c)] + right)
            });
            Ok(ContextValue::VectorField(out))
        }

        fn initial_state(&self, mesh: &dyn Mesh) -> ContextValue {
            ContextValue::VectorField(DMatrix::from_fn(mesh.n_dof(), Self::COMPONENTS, |i, c| {
                1.0 + 0.1 * i as f64 - 0.3 * c as f64
            }))
        }

        fn name(&self) -> &str {
            "coupled_reaction_diffusion_test"
        }
    }

    pub(super) fn coupled_reaction_diffusion() -> CoupledReactionDiffusion {
        CoupledReactionDiffusion {
            reaction: [[-1.5, 0.4], [0.7, -0.9]],
            diffusion: 0.6,
        }
    }

    #[test]
    fn jacobian_of_a_vector_field_follows_the_node_interleaved_order() {
        let nodes = 5;
        let d = CoupledReactionDiffusion::COMPONENTS;
        let model = coupled_reaction_diffusion();
        let expected =
            |row: usize, col: usize| model.exact_entry((row / d, row % d), (col / d, col % d));
        let scenario = Scenario::single(Box::new(coupled_reaction_diffusion()), make_mesh(nodes));
        let domain = scenario.single_domain().unwrap();
        let chain = build_calculator_chain(&scenario.context_requirements(), &[]).unwrap();
        let state = domain.model.initial_state(domain.mesh.as_ref());

        let jac = finite_difference_jacobian(
            domain,
            &chain,
            &state,
            0.0,
            0.1,
            &JacobianDispatch::default(),
        )
        .unwrap();

        assert_eq!(jac.shape(), (nodes * d, nodes * d));
        for row in 0..nodes * d {
            for col in 0..nodes * d {
                let exact = expected(row, col);
                if exact == 0.0 {
                    // The perturbed unknown does not enter this row: same
                    // inputs, same operations, hence exactly zero.
                    assert_eq!(jac[(row, col)], 0.0, "({row}, {col})");
                } else {
                    assert!(
                        (jac[(row, col)] - exact).abs() < 1e-6,
                        "({row}, {col}): {} vs {exact}",
                        jac[(row, col)]
                    );
                }
            }
        }
    }

    // ── Jacobian dispatch (DT-069, #198) ─────────────────────────────────────

    /// Dispatch configuration with the given threshold.
    #[cfg(feature = "parallel")]
    pub(super) fn dispatch_with_threshold(threshold: usize) -> JacobianDispatch {
        let dispatch = JacobianDispatch::default();
        dispatch.set(threshold);
        dispatch
    }

    #[cfg(feature = "parallel")]
    #[test]
    fn jacobian_dispatch_compares_the_number_of_unknowns_with_its_threshold() {
        let dispatch = JacobianDispatch::default();
        let default = default_parallel_threshold();
        assert_eq!(dispatch.threshold(), default);
        assert!(!dispatch.is_parallel_for(default - 1));
        assert!(dispatch.is_parallel_for(default));
        dispatch.set(3);
        assert_eq!(dispatch.threshold(), 3);
        assert!(!dispatch.is_parallel_for(2));
        assert!(dispatch.is_parallel_for(3));
    }

    #[cfg(feature = "parallel")]
    #[test]
    #[should_panic(expected = "parallel threshold must be at least 1")]
    fn jacobian_dispatch_rejects_a_zero_threshold() {
        JacobianDispatch::default().set(0);
    }

    #[cfg(not(feature = "parallel"))]
    #[test]
    fn jacobian_dispatch_is_sequential_without_the_parallel_feature() {
        assert!(!JacobianDispatch::default().is_parallel_for(usize::MAX));
    }

    #[cfg(feature = "parallel")]
    #[test]
    fn dense_jacobian_is_bit_identical_in_sequential_and_parallel_dispatch() {
        let scenarios = [
            Scenario::single(Box::new(ExponentialDecay { lambda: 2.0 }), make_mesh(9)),
            Scenario::single(Box::new(coupled_reaction_diffusion()), make_mesh(6)),
        ];
        for scenario in &scenarios {
            let domain = scenario.single_domain().unwrap();
            let chain = build_calculator_chain(&scenario.context_requirements(), &[]).unwrap();
            let state = domain.model.initial_state(domain.mesh.as_ref());
            let jacobian = |threshold| {
                let dispatch = dispatch_with_threshold(threshold);
                finite_difference_jacobian(domain, &chain, &state, 0.0, 0.1, &dispatch).unwrap()
            };
            let (parallel, sequential) = (jacobian(1), jacobian(usize::MAX));
            assert_eq!(parallel.shape(), sequential.shape());
            assert!(parallel
                .iter()
                .zip(sequential.iter())
                .all(|(a, b)| a.to_bits() == b.to_bits()));
        }
    }

    /// `du/dt = -u`, recording at each evaluation whether it ran on a Rayon
    /// worker thread.
    #[cfg(feature = "parallel")]
    #[derive(Debug)]
    struct WorkerRecorder {
        on_worker: std::sync::Arc<std::sync::Mutex<Vec<bool>>>,
    }

    #[cfg(feature = "parallel")]
    impl RequiresContext for WorkerRecorder {
        fn required_variables(&self) -> Vec<ContextVariable> {
            vec![]
        }
    }

    #[cfg(feature = "parallel")]
    impl PhysicalModel for WorkerRecorder {
        fn compute_physics(
            &self,
            state: &ContextValue,
            _ctx: &ComputeContext,
        ) -> Result<ContextValue, OxiflowError> {
            self.on_worker
                .lock()
                .unwrap()
                .push(rayon::current_thread_index().is_some());
            Ok(ContextValue::ScalarField(
                state.as_scalar_field()?.map(|v| -v),
            ))
        }

        fn initial_state(&self, mesh: &dyn Mesh) -> ContextValue {
            ContextValue::ScalarField(DVector::from_element(mesh.n_dof(), 1.0))
        }

        fn name(&self) -> &str {
            "worker_recorder"
        }
    }

    /// Runs `step` once on a domain of `n` unknowns and reports whether any
    /// derivative evaluation ran on a Rayon worker thread.
    #[cfg(feature = "parallel")]
    fn evaluated_on_a_worker(
        n: usize,
        step: impl Fn(
            &Domain,
            &[&dyn ContextCalculator],
            &mut ContextValue,
        ) -> Result<ContextValue, OxiflowError>,
    ) -> bool {
        let on_worker = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
        let model = WorkerRecorder {
            on_worker: std::sync::Arc::clone(&on_worker),
        };
        let scenario = Scenario::single(Box::new(model), make_mesh(n));
        let domain = scenario.single_domain().unwrap();
        let chain = build_calculator_chain(&scenario.context_requirements(), &[]).unwrap();
        let mut state = domain.model.initial_state(domain.mesh.as_ref());
        step(domain, &chain, &mut state).unwrap();
        let recorded = on_worker.lock().unwrap();
        assert!(!recorded.is_empty());
        recorded.iter().any(|&on_worker| on_worker)
    }

    #[cfg(feature = "parallel")]
    #[test]
    fn implicit_solvers_hand_their_threshold_to_the_jacobian_assembly() {
        use crate::solver::methods::backward_euler::BackwardEulerSolver;
        use crate::solver::methods::bdf2::BDF2Solver;
        use crate::solver::methods::crank_nicolson::CrankNicolsonSolver;
        use crate::solver::methods::SteppableSolver;

        let n = 8;
        let history = [ContextValue::ScalarField(DVector::from_element(n, 1.0))];
        for threshold in [1, usize::MAX] {
            let parallel = threshold == 1;
            let be = BackwardEulerSolver::new().with_parallel_threshold(threshold);
            let cn = CrankNicolsonSolver::new().with_parallel_threshold(threshold);
            let bdf2 = BDF2Solver::new().with_parallel_threshold(threshold);
            let cases: [(&str, &dyn SteppableSolver, &[ContextValue]); 4] = [
                ("Backward Euler", &be, &[]),
                ("Crank-Nicolson", &cn, &[]),
                ("BDF2 startup", &bdf2, &[]),
                ("BDF2 update", &bdf2, &history),
            ];
            for (name, solver, history) in cases {
                let on_worker = evaluated_on_a_worker(n, |domain, chain, state| {
                    solver.step(domain, chain, state, history, 0.0, 0.1)
                });
                assert_eq!(on_worker, parallel, "{name}, threshold {threshold}");
            }

            #[cfg(feature = "sparse")]
            {
                use crate::solver::sparse::FaerSparseSolver;
                // Banded path: bandwidth declared, sparse threshold 0.
                let be = BackwardEulerSolver::new()
                    .with_sparse_solver(Box::new(FaerSparseSolver))
                    .with_jacobian_bandwidth(1)
                    .with_sparse_threshold(0)
                    .with_parallel_threshold(threshold);
                let cn = CrankNicolsonSolver::new()
                    .with_sparse_solver(Box::new(FaerSparseSolver))
                    .with_jacobian_bandwidth(1)
                    .with_sparse_threshold(0)
                    .with_parallel_threshold(threshold);
                let cases: [(&str, &dyn SteppableSolver); 2] = [
                    ("banded Backward Euler", &be),
                    ("banded Crank-Nicolson", &cn),
                ];
                for (name, solver) in cases {
                    let on_worker = evaluated_on_a_worker(n, |domain, chain, state| {
                        solver.step(domain, chain, state, &[], 0.0, 0.1)
                    });
                    assert_eq!(on_worker, parallel, "{name}, threshold {threshold}");
                }
            }
        }
    }

    #[cfg(feature = "parallel")]
    #[test]
    fn implicit_solvers_expose_their_jacobian_dispatch_threshold() {
        use crate::solver::methods::backward_euler::BackwardEulerSolver;
        use crate::solver::methods::bdf2::BDF2Solver;
        use crate::solver::methods::crank_nicolson::CrankNicolsonSolver;

        fn check<S>(
            new: fn() -> S,
            with: fn(S, usize) -> S,
            set: fn(&S, usize),
            get: fn(&S) -> usize,
        ) {
            assert_eq!(get(&new()), default_parallel_threshold());
            let solver = with(new(), 7);
            assert_eq!(get(&solver), 7);
            set(&solver, 9);
            assert_eq!(get(&solver), 9);
        }
        check(
            BackwardEulerSolver::new,
            BackwardEulerSolver::with_parallel_threshold,
            BackwardEulerSolver::set_parallel_threshold,
            BackwardEulerSolver::parallel_threshold,
        );
        check(
            CrankNicolsonSolver::new,
            CrankNicolsonSolver::with_parallel_threshold,
            CrankNicolsonSolver::set_parallel_threshold,
            CrankNicolsonSolver::parallel_threshold,
        );
        check(
            BDF2Solver::new,
            BDF2Solver::with_parallel_threshold,
            BDF2Solver::set_parallel_threshold,
            BDF2Solver::parallel_threshold,
        );
    }

    #[test]
    fn jacobian_of_linear_decay_is_minus_lambda_identity() {
        let lambda = 2.5;
        let scenario = Scenario::single(Box::new(ExponentialDecay { lambda }), make_mesh(3));
        let domain = scenario.single_domain().unwrap();
        let requirements = scenario.context_requirements();
        let chain = build_calculator_chain(&requirements, &[]).unwrap();

        let state = domain.model.initial_state(domain.mesh.as_ref());
        let jac = finite_difference_jacobian(
            domain,
            &chain,
            &state,
            0.0,
            0.1,
            &JacobianDispatch::default(),
        )
        .unwrap();

        for i in 0..3 {
            for j in 0..3 {
                let expected = if i == j { -lambda } else { 0.0 };
                assert!(
                    (jac[(i, j)] - expected).abs() < 1e-4,
                    "jac[{i},{j}] = {} (expected {expected})",
                    jac[(i, j)]
                );
            }
        }
    }

    #[test]
    fn backward_euler_theta_one_matches_analytical_for_linear_decay() {
        // For du/dt = -lambda*u, backward Euler gives:
        // u^{n+1} = u^n / (1 + lambda*dt)
        let lambda = 3.0;
        let dt = 0.1;
        let scenario = Scenario::single(Box::new(ExponentialDecay { lambda }), make_mesh(2));
        let domain = scenario.single_domain().unwrap();
        let requirements = scenario.context_requirements();
        let chain = build_calculator_chain(&requirements, &[]).unwrap();

        let mut state = domain.model.initial_state(domain.mesh.as_ref());
        let next = theta_method_step(
            domain,
            &chain,
            &mut state,
            0.0,
            dt,
            1.0,
            &NalgebraDenseSolver,
            &JacobianDispatch::default(),
        )
        .unwrap();

        let expected = 1.0 / (1.0 + lambda * dt);
        let field = next.as_scalar_field().unwrap();
        for v in field.iter() {
            assert!((v - expected).abs() < 1e-9, "got {v}, expected {expected}");
        }
    }

    #[test]
    fn crank_nicolson_theta_half_matches_analytical_for_linear_decay() {
        // For du/dt = -lambda*u, Crank-Nicolson gives:
        // u^{n+1} = u^n * (1 - lambda*dt/2) / (1 + lambda*dt/2)
        let lambda = 3.0;
        let dt = 0.1;
        let scenario = Scenario::single(Box::new(ExponentialDecay { lambda }), make_mesh(2));
        let domain = scenario.single_domain().unwrap();
        let requirements = scenario.context_requirements();
        let chain = build_calculator_chain(&requirements, &[]).unwrap();

        let mut state = domain.model.initial_state(domain.mesh.as_ref());
        let next = theta_method_step(
            domain,
            &chain,
            &mut state,
            0.0,
            dt,
            0.5,
            &NalgebraDenseSolver,
            &JacobianDispatch::default(),
        )
        .unwrap();

        let expected = (1.0 - lambda * dt / 2.0) / (1.0 + lambda * dt / 2.0);
        let field = next.as_scalar_field().unwrap();
        for v in field.iter() {
            assert!((v - expected).abs() < 1e-9, "got {v}, expected {expected}");
        }
    }

    #[test]
    fn backward_euler_stable_for_very_stiff_problem() {
        // lambda*dt = 1000 -- far beyond any explicit method's stability
        // limit. Backward Euler must remain bounded and well-behaved.
        let lambda = 1.0e4;
        let dt = 0.1; // lambda*dt = 1000
        let scenario = Scenario::single(Box::new(ExponentialDecay { lambda }), make_mesh(2));
        let domain = scenario.single_domain().unwrap();
        let requirements = scenario.context_requirements();
        let chain = build_calculator_chain(&requirements, &[]).unwrap();

        let mut state = domain.model.initial_state(domain.mesh.as_ref());
        let next = theta_method_step(
            domain,
            &chain,
            &mut state,
            0.0,
            dt,
            1.0,
            &NalgebraDenseSolver,
            &JacobianDispatch::default(),
        )
        .unwrap();

        let field = next.as_scalar_field().unwrap();
        for v in field.iter() {
            assert!(v.is_finite(), "value diverged: {v}");
            assert!(v.abs() < 1.0, "expected strong damping, got {v}");
        }
    }

    // ── Dirichlet BC × perturbed Jacobian (#113) ───────────────────────────────

    use crate::boundary::{BoundaryCondition, BoundaryType};

    /// Discrete tridiagonal diffusion with a spike initial condition —
    /// genuine spatial coupling to neighbours, needed to distinguish
    /// "legitimate physics coupling" from "spurious coupling introduced by
    /// BC re-application" in the test below. A local copy rather than
    /// reusing `sparse_tests::TridiagonalDiffusion`: that fixture only
    /// exists under `feature = "sparse"`, and this test needs no sparse
    /// machinery.
    #[derive(Debug)]
    struct TridiagonalDiffusion {
        diffusion: f64,
    }

    impl RequiresContext for TridiagonalDiffusion {
        fn required_variables(&self) -> Vec<ContextVariable> {
            vec![]
        }
    }

    impl PhysicalModel for TridiagonalDiffusion {
        fn compute_physics(
            &self,
            state: &ContextValue,
            _ctx: &ComputeContext,
        ) -> Result<ContextValue, OxiflowError> {
            let u = state.as_scalar_field()?;
            let n = u.len();
            let d = self.diffusion;
            let out = DVector::from_fn(n, |i, _| {
                let left = if i > 0 { u[i - 1] } else { 0.0 };
                let right = if i + 1 < n { u[i + 1] } else { 0.0 };
                d * (left - 2.0 * u[i] + right)
            });
            Ok(ContextValue::ScalarField(out))
        }

        fn initial_state(&self, mesh: &dyn Mesh) -> ContextValue {
            let n = mesh.n_dof();
            ContextValue::ScalarField(DVector::from_fn(
                n,
                |i, _| if i == n / 2 { 1.0 } else { 0.0 },
            ))
        }

        fn name(&self) -> &str {
            "tridiagonal_diffusion_dirichlet_test"
        }
    }

    /// Fixes a single node to a constant value — a minimal Dirichlet BC.
    /// No built-in `BoundaryCondition` implementation in `crate::boundary`
    /// enforces a plain Dirichlet constraint today (only `DanckwertsInlet`/
    /// `DanckwertsOutlet`, Robin/Neumann), so this test defines its own.
    #[derive(Debug)]
    struct FixedDirichlet {
        node_index: usize,
        value: f64,
    }

    impl RequiresContext for FixedDirichlet {
        fn required_variables(&self) -> Vec<ContextVariable> {
            vec![]
        }
    }

    impl BoundaryCondition for FixedDirichlet {
        fn boundary_type(&self) -> BoundaryType {
            BoundaryType::Dirichlet
        }

        fn apply(
            &self,
            state: &mut DVector<f64>,
            _ctx: &ComputeContext,
            _mesh: &dyn Mesh,
        ) -> Result<(), OxiflowError> {
            state[self.node_index] = self.value;
            Ok(())
        }
    }

    #[test]
    fn dirichlet_constrained_column_is_exactly_zero_row_keeps_real_coupling() {
        // DD-033's known limitation, closed by #113: `finite_difference_jacobian`
        // perturbs each state component and re-evaluates `evaluate_derivative`,
        // which re-applies boundary conditions on every call. A
        // Dirichlet-constrained node's perturbation is therefore overwritten
        // before `compute_physics` ever sees it -- physically correct (a
        // BC-constrained node isn't a free unknown), verified here rather
        // than merely documented.
        let n = 5;
        let constrained = n / 2; // the model's initial spike location
        let diffusion = 0.7;

        let domain = Domain::new(
            "dirichlet_test",
            Box::new(TridiagonalDiffusion { diffusion }),
            make_mesh(n),
        )
        .with_boundary_conditions(vec![Box::new(FixedDirichlet {
            node_index: constrained,
            value: 0.0,
        })]);
        let scenario = Scenario::multi(vec![domain]).unwrap();
        let domain = scenario.single_domain().unwrap();
        let requirements = scenario.context_requirements();
        let chain = build_calculator_chain(&requirements, &[]).unwrap();

        let state = domain.model.initial_state(domain.mesh.as_ref());
        let jac = finite_difference_jacobian(
            domain,
            &chain,
            &state,
            0.0,
            0.01,
            &JacobianDispatch::default(),
        )
        .unwrap();

        // Column: perturbing the constrained node itself must leave every
        // row's output exactly unchanged -- the perturbation never
        // survives BC re-application, so both the base and perturbed
        // evaluations see bit-identical states, giving f_j - f0 = 0
        // exactly, not merely small.
        for i in 0..n {
            assert_eq!(
                jac[(i, constrained)],
                0.0,
                "column {constrained} (the constrained node) must be exactly \
                 zero at row {i} -- got {}",
                jac[(i, constrained)]
            );
        }

        // Row: the constrained node's own output still genuinely depends
        // on its unconstrained neighbours through the model's real spatial
        // coupling -- legitimate physics, not spurious coupling, and must
        // not be zeroed out by the fix the column check above verifies.
        let left = constrained - 1;
        let right = constrained + 1;
        assert!(
            (jac[(constrained, left)] - diffusion).abs() < 1e-6,
            "expected left-neighbour coupling ~{diffusion}, got {}",
            jac[(constrained, left)]
        );
        assert!(
            (jac[(constrained, right)] - diffusion).abs() < 1e-6,
            "expected right-neighbour coupling ~{diffusion}, got {}",
            jac[(constrained, right)]
        );
    }
}

#[cfg(all(test, feature = "sparse"))]
mod sparse_tests {
    use super::*;
    use crate::context::compute::ComputeContext;
    use crate::context::variable::ContextVariable;
    use crate::mesh::{Mesh, UniformGrid1D};
    use crate::model::traits::{PhysicalModel, RequiresContext};
    use crate::solver::chain::build_calculator_chain;
    use crate::solver::linear::NalgebraDenseSolver;
    use crate::solver::scenario::Scenario;
    use crate::solver::sparse::FaerSparseSolver;

    /// Discrete tridiagonal diffusion — `du_i/dt = D*(u[i-1] - 2u[i] +
    /// u[i+1])`, zero outside the domain. Not a real `BoundaryCondition` —
    /// this fixture exists only to exercise the banded path with a
    /// genuinely local, bandwidth-1 coupling (real Dirichlet/Neumann
    /// boundary handling is covered separately, see the module's own docs
    /// on boundary conditions × the perturbed Jacobian, and
    /// `tests::dirichlet_constrained_column_is_exactly_zero_row_keeps_real_coupling`,
    /// #113).
    #[derive(Debug)]
    struct TridiagonalDiffusion {
        diffusion: f64,
    }

    impl RequiresContext for TridiagonalDiffusion {
        fn required_variables(&self) -> Vec<ContextVariable> {
            vec![]
        }
    }

    impl PhysicalModel for TridiagonalDiffusion {
        fn compute_physics(
            &self,
            state: &ContextValue,
            _ctx: &ComputeContext,
        ) -> Result<ContextValue, OxiflowError> {
            let u = state.as_scalar_field()?;
            let n = u.len();
            let d = self.diffusion;
            let out = DVector::from_fn(n, |i, _| {
                let left = if i > 0 { u[i - 1] } else { 0.0 };
                let right = if i + 1 < n { u[i + 1] } else { 0.0 };
                d * (left - 2.0 * u[i] + right)
            });
            Ok(ContextValue::ScalarField(out))
        }

        fn initial_state(&self, mesh: &dyn Mesh) -> ContextValue {
            let n = mesh.n_dof();
            ContextValue::ScalarField(DVector::from_fn(
                n,
                |i, _| if i == n / 2 { 1.0 } else { 0.0 },
            ))
        }

        fn name(&self) -> &str {
            "tridiagonal_diffusion_test"
        }
    }

    /// A [`SparseLinearSolver`] that panics if ever called — used to prove
    /// the dense path was actually taken (threshold/bandwidth conditions
    /// not met), not just that the result happens to look dense-like.
    struct PanicIfCalled;

    impl SparseLinearSolver for PanicIfCalled {
        fn solve(
            &self,
            _a: &faer::sparse::SparseColMat<usize, f64>,
            _b: &DVector<f64>,
        ) -> Result<DVector<f64>, OxiflowError> {
            panic!("sparse solver called when the dense path should have been taken");
        }
    }

    fn make_mesh(n: usize) -> Box<dyn Mesh> {
        Box::new(UniformGrid1D::new(n, 0.0, 1.0).unwrap())
    }

    /// Builds the dense Jacobian and the banded one of `scenario` at its
    /// initial state, and checks that they are equal bit for bit.
    ///
    /// Inside the band, a row of a color evaluation reads exactly the
    /// unknowns it reads in the dense evaluation of that column (the other
    /// perturbed columns of the color are outside its stencil), so every
    /// entry is computed from the same operands with the same operations.
    /// Outside the band the dense entries are exactly zero for the same
    /// reason, and the banded matrix stores nothing.
    fn assert_banded_jacobian_equals_dense(scenario: &Scenario, bandwidth: usize) {
        let domain = scenario.single_domain().unwrap();
        let chain = build_calculator_chain(&scenario.context_requirements(), &[]).unwrap();
        let state = domain.model.initial_state(domain.mesh.as_ref());

        let dense = finite_difference_jacobian(
            domain,
            &chain,
            &state,
            0.0,
            0.01,
            &JacobianDispatch::default(),
        )
        .unwrap();
        let banded = banded_finite_difference_jacobian(
            domain,
            &chain,
            &state,
            0.0,
            0.01,
            bandwidth,
            &JacobianDispatch::default(),
        )
        .unwrap();
        let n = dense.nrows();
        assert_eq!((banded.nrows(), banded.ncols()), (n, n));

        let mut stored = 0;
        for entry in banded.as_ref().triplet_iter() {
            let (i, j) = (entry.row, entry.col);
            assert!(i.abs_diff(j) <= bandwidth, "({i},{j}) outside the band");
            assert_eq!(
                entry.val.to_bits(),
                dense[(i, j)].to_bits(),
                "({i},{j}): banded {} vs dense {}",
                entry.val,
                dense[(i, j)]
            );
            stored += 1;
        }
        let band_size: usize = (0..n)
            .map(|j| (j + bandwidth + 1).min(n) - j.saturating_sub(bandwidth))
            .sum();
        assert_eq!(stored, band_size);

        for i in 0..n {
            for j in 0..n {
                if i.abs_diff(j) > bandwidth {
                    assert_eq!(dense[(i, j)], 0.0, "dense ({i},{j}) outside the band");
                }
            }
        }
    }

    #[test]
    fn banded_jacobian_equals_dense_jacobian_on_a_scalar_field() {
        let scenario = Scenario::single(
            Box::new(TridiagonalDiffusion { diffusion: 0.7 }),
            make_mesh(12),
        );
        assert_banded_jacobian_equals_dense(&scenario, 1);
    }

    #[cfg(feature = "parallel")]
    #[test]
    fn banded_jacobian_is_bit_identical_in_sequential_and_parallel_dispatch() {
        use super::tests::{coupled_reaction_diffusion, dispatch_with_threshold};

        let cases = [
            (
                Scenario::single(
                    Box::new(TridiagonalDiffusion { diffusion: 0.7 }),
                    make_mesh(12),
                ),
                1,
            ),
            (
                Scenario::single(Box::new(coupled_reaction_diffusion()), make_mesh(6)),
                3,
            ),
        ];
        for (scenario, bandwidth) in &cases {
            let domain = scenario.single_domain().unwrap();
            let chain = build_calculator_chain(&scenario.context_requirements(), &[]).unwrap();
            let state = domain.model.initial_state(domain.mesh.as_ref());
            let jacobian = |threshold| {
                let dispatch = dispatch_with_threshold(threshold);
                banded_finite_difference_jacobian(
                    domain, &chain, &state, 0.0, 0.01, *bandwidth, &dispatch,
                )
                .unwrap()
            };
            let (parallel, sequential) = (jacobian(1), jacobian(usize::MAX));
            assert_eq!(
                parallel.symbolic().col_ptr(),
                sequential.symbolic().col_ptr()
            );
            assert_eq!(
                parallel.symbolic().row_idx(),
                sequential.symbolic().row_idx()
            );
            assert!(parallel
                .val()
                .iter()
                .zip(sequential.val())
                .all(|(a, b)| a.to_bits() == b.to_bits()));
        }
    }

    #[test]
    fn banded_jacobian_wider_than_the_system_is_the_full_dense_jacobian() {
        // Fewer columns than colors: every column gets its own color and
        // every band covers the whole column.
        let scenario = Scenario::single(
            Box::new(TridiagonalDiffusion { diffusion: 0.7 }),
            make_mesh(4),
        );
        assert_banded_jacobian_equals_dense(&scenario, 6);
    }

    #[test]
    fn banded_jacobian_equals_dense_jacobian_on_a_vector_field() {
        let scenario = Scenario::single(
            Box::new(super::tests::coupled_reaction_diffusion()),
            make_mesh(6),
        );
        // d·(s + 1) − 1 with d = 2 components and a stencil of half-width 1.
        assert_banded_jacobian_equals_dense(&scenario, 3);
    }

    #[test]
    fn adaptive_step_with_no_bandwidth_matches_theta_method_step_exactly() {
        let n = 6;
        let scenario = Scenario::single(
            Box::new(TridiagonalDiffusion { diffusion: 0.5 }),
            make_mesh(n),
        );
        let domain = scenario.single_domain().unwrap();
        let requirements = scenario.context_requirements();
        let chain = build_calculator_chain(&requirements, &[]).unwrap();

        let mut state_a = domain.model.initial_state(domain.mesh.as_ref());
        let mut state_b = state_a.clone();

        let via_adaptive = theta_method_step_adaptive(
            domain,
            &chain,
            &mut state_a,
            0.0,
            0.05,
            1.0,
            &NalgebraDenseSolver,
            &PanicIfCalled,
            100,
            None, // no bandwidth declared -> dense path, unconditionally
            &JacobianDispatch::default(),
        )
        .unwrap();

        let via_plain = theta_method_step(
            domain,
            &chain,
            &mut state_b,
            0.0,
            0.05,
            1.0,
            &NalgebraDenseSolver,
            &JacobianDispatch::default(),
        )
        .unwrap();

        let a = via_adaptive.as_scalar_field().unwrap();
        let b = via_plain.as_scalar_field().unwrap();
        for i in 0..n {
            assert!(
                (a[i] - b[i]).abs() < 1e-12,
                "mismatch at {i}: {} vs {}",
                a[i],
                b[i]
            );
        }
    }

    #[test]
    fn small_system_stays_dense_even_with_bandwidth_declared() {
        let n = 6; // below sparse_threshold below
        let scenario = Scenario::single(
            Box::new(TridiagonalDiffusion { diffusion: 0.5 }),
            make_mesh(n),
        );
        let domain = scenario.single_domain().unwrap();
        let requirements = scenario.context_requirements();
        let chain = build_calculator_chain(&requirements, &[]).unwrap();
        let mut state = domain.model.initial_state(domain.mesh.as_ref());

        // PanicIfCalled proves the sparse path was never entered: bandwidth
        // is declared (Some(1)) but n (6) does not exceed sparse_threshold
        // (100), so the dense path must be taken regardless.
        let result = theta_method_step_adaptive(
            domain,
            &chain,
            &mut state,
            0.0,
            0.05,
            1.0,
            &NalgebraDenseSolver,
            &PanicIfCalled,
            100,
            Some(1),
            &JacobianDispatch::default(),
        )
        .unwrap();

        assert!(result
            .as_scalar_field()
            .unwrap()
            .iter()
            .all(|v| v.is_finite()));
    }

    #[test]
    fn adaptive_step_with_bandwidth_matches_dense_result_above_threshold() {
        let n = 40; // above the small sparse_threshold used below
        let diffusion = 0.3;
        let dt = 0.02;

        let scenario_a =
            Scenario::single(Box::new(TridiagonalDiffusion { diffusion }), make_mesh(n));
        let domain_a = scenario_a.single_domain().unwrap();
        let requirements_a = scenario_a.context_requirements();
        let chain_a = build_calculator_chain(&requirements_a, &[]).unwrap();
        let mut state_a = domain_a.model.initial_state(domain_a.mesh.as_ref());

        let scenario_b =
            Scenario::single(Box::new(TridiagonalDiffusion { diffusion }), make_mesh(n));
        let domain_b = scenario_b.single_domain().unwrap();
        let requirements_b = scenario_b.context_requirements();
        let chain_b = build_calculator_chain(&requirements_b, &[]).unwrap();
        let mut state_b = domain_b.model.initial_state(domain_b.mesh.as_ref());

        let via_sparse = theta_method_step_adaptive(
            domain_a,
            &chain_a,
            &mut state_a,
            0.0,
            dt,
            1.0,
            &NalgebraDenseSolver,
            &FaerSparseSolver,
            10, // sparse_threshold well below n=40
            Some(1),
            &JacobianDispatch::default(),
        )
        .unwrap();

        let via_dense = theta_method_step(
            domain_b,
            &chain_b,
            &mut state_b,
            0.0,
            dt,
            1.0,
            &NalgebraDenseSolver,
            &JacobianDispatch::default(),
        )
        .unwrap();

        let a = via_sparse.as_scalar_field().unwrap();
        let b = via_dense.as_scalar_field().unwrap();
        for i in 0..n {
            assert!(
                (a[i] - b[i]).abs() < 1e-6,
                "sparse/dense mismatch at {i}: {} vs {}",
                a[i],
                b[i]
            );
        }
    }
}
