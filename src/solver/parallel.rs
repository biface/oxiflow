//! # Module `solver::parallel`
//!
//! Shared Rayon-dispatch-threshold mechanism (DD-014, DD-048) — a small,
//! reusable building block for every site in the crate that decides, per
//! element, whether to run a computation sequentially or through Rayon.
//! Extracted after the same field/builder/guard triplet was written
//! twice, verbatim, on `FDGradientCalculator` and `FDLaplacianCalculator`
//! (DD-014, #51) — see DD-048 for the full rationale (architecture
//! principles P1-P5) behind this module's shape.
//!
//! ## What this is not
//!
//! Not a trait imposed on `ContextCalculator`/`DiscreteOperator`/
//! `FluxDivergenceOperator` — DD-048 explicitly rejects that shape,
//! mirroring the precedent already set twice (DD-012 amendment 2, DD-039
//! option A): forcing an orthogonal capability onto every implementer of
//! a shared trait, when only some of them benefit from it. Each site that
//! wants Rayon dispatch holds a [`ParallelThreshold`] field and reads it
//! itself — a plain data-holding type, not a behavioural contract.
//!
//! Entirely gated behind the `parallel` feature, at the module
//! declaration itself (`solver::mod`) — mirrors `solver::sparse`/
//! `solver::gpu`'s own gating, not merely the items inside it.

use std::sync::atomic::{AtomicUsize, Ordering};

/// Per-instance Rayon dispatch threshold.
///
/// Stores `0` internally to mean "no override" — `0` is otherwise
/// forbidden by [`Self::set`], so it is free to reuse as a sentinel
/// without an `Option` (which would block mutation through `&self`
/// without extra indirection). [`Self::get`] resolves the sentinel to a
/// caller-supplied default rather than materializing that default at
/// construction — one authoritative default, consulted on demand, rather
/// than copied into every instance (DD-048; closer to chrom-rs's single
/// process-wide default than to a per-instance constant).
///
/// Deliberately per-instance, not a shared `static` (chrom-rs's own
/// design) — `MultiDomainOrchestrator` (DD-031) can run several domains,
/// each with its own solver, in the same process; a single global would
/// force them to share one setting, with a "last writer wins" race
/// between domains adjusting it independently. Reconfigurable at runtime
/// through `&self` via [`Self::set`] — no rebuild required, and no
/// cross-instance leakage to guard against (unlike chrom-rs's
/// `ThresholdGuard`, needed only because its threshold is global).
#[derive(Debug, Default)]
pub(crate) struct ParallelThreshold(AtomicUsize);

impl ParallelThreshold {
    /// Creates a threshold with no override — [`Self::get`] reports the
    /// caller-supplied default until [`Self::set`] is called.
    pub(crate) fn unset() -> Self {
        Self(AtomicUsize::new(0))
    }

    /// Overrides the threshold.
    ///
    /// # Panics
    ///
    /// Panics when `threshold == 0` — reserved as the "unset" sentinel
    /// (see the type's own docs), and not a value any real workload
    /// should want anyway: it would force parallel dispatch on every
    /// non-empty input, never the intended behaviour (mirrors chrom-rs's
    /// own `set_parallel_threshold` guard). No other value is validated
    /// — DD-048 deliberately leaves "how low is too low" to the caller's
    /// own measurement (see [`calibrate_parallel_threshold`]), consistent
    /// with `sparse_threshold`/`jacobian_bandwidth` (DD-043), which
    /// validate nothing at all: there is no threshold floor that is
    /// universally sensible across workloads (chrom-rs's 999 is nearly a
    /// total loss for the FD stencils DD-014 calibrated 49999 against).
    pub(crate) fn set(&self, threshold: usize) {
        assert!(threshold > 0, "parallel threshold must be at least 1");
        self.0.store(threshold, Ordering::Relaxed);
    }

    /// Returns the configured override, or `default` if none was set.
    pub(crate) fn get(&self, default: usize) -> usize {
        match self.0.load(Ordering::Relaxed) {
            0 => default,
            configured => configured,
        }
    }
}

/// Empirically measures the sequential/parallel crossover point for a
/// given per-element workload, on the machine actually running the code
/// — rather than assuming a number calibrated for a different workload
/// or a different machine (DD-048).
///
/// Tries each of `candidate_sizes`, in order, timing a sequential loop
/// and a Rayon `into_par_iter` dispatch of `work` over `0..n` elements;
/// returns the first `n` where the parallel run was faster. Returns the
/// largest candidate if none crossed over (a conservative "parallel
/// never won at any size tried" outcome, not a panic), or `0` if
/// `candidate_sizes` is empty.
///
/// `work` should be representative of the real per-element cost it will
/// be used to threshold for — measuring a cheap placeholder closure and
/// applying the result to a heavier one (or vice versa) defeats the
/// purpose. Feeds #53 (the Criterion benchmark suite) rather than
/// duplicating its own crossover measurement.
///
/// # Examples
///
/// ```rust
/// use oxiflow::solver::parallel::calibrate_parallel_threshold;
///
/// let threshold = calibrate_parallel_threshold(
///     |i| {
///         let _ = (i as f64).sqrt();
///     },
///     &[100, 1_000, 10_000, 100_000],
/// );
/// assert!(threshold > 0);
/// ```
pub fn calibrate_parallel_threshold<F>(work: F, candidate_sizes: &[usize]) -> usize
where
    F: Fn(usize) + Sync,
{
    use rayon::prelude::*;
    use std::time::Instant;

    for &n in candidate_sizes {
        let seq_start = Instant::now();
        for i in 0..n {
            work(i);
        }
        let seq_elapsed = seq_start.elapsed();

        let par_start = Instant::now();
        (0..n).into_par_iter().for_each(&work);
        let par_elapsed = par_start.elapsed();

        if par_elapsed < seq_elapsed {
            return n;
        }
    }

    candidate_sizes.iter().copied().max().unwrap_or(0)
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── ParallelThreshold ──────────────────────────────────────────────────────

    #[test]
    fn unset_reports_the_supplied_default() {
        let t = ParallelThreshold::unset();
        assert_eq!(t.get(12345), 12345);
    }

    #[test]
    fn set_overrides_the_default() {
        let t = ParallelThreshold::unset();
        t.set(42);
        assert_eq!(t.get(12345), 42);
    }

    #[test]
    #[should_panic(expected = "parallel threshold must be at least 1")]
    fn set_rejects_zero() {
        ParallelThreshold::unset().set(0);
    }

    #[test]
    fn set_is_visible_through_shared_reference() {
        let t = ParallelThreshold::unset();
        let t_ref = &t;
        t_ref.set(7);
        assert_eq!(t.get(0), 7);
    }

    // ── calibrate_parallel_threshold ───────────────────────────────────────────
    //
    // No assertion depends on *which* candidate wins — that's a timing
    // outcome, and asserting a specific crossover would make this test
    // flaky under CI load. Only the contract (returns a candidate, or 0
    // on empty input) is checked.

    #[test]
    fn calibrate_returns_one_of_the_candidates() {
        let candidates = [10, 100, 1_000];
        let threshold = calibrate_parallel_threshold(
            |i| {
                std::hint::black_box(i);
            },
            &candidates,
        );
        assert!(candidates.contains(&threshold));
    }

    #[test]
    fn calibrate_on_empty_candidates_returns_zero() {
        let threshold = calibrate_parallel_threshold(
            |i: usize| {
                std::hint::black_box(i);
            },
            &[],
        );
        assert_eq!(threshold, 0);
    }
}
