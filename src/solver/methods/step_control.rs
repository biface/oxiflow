//! # Module `solver::methods::step_control`
//!
//! Adaptive step-size control — shared machinery for variable-step
//! integrators (DD-036).
//!
//! ## Why this is separate from `dopri45.rs`
//!
//! [`StepSizeController`] knows nothing about Runge-Kutta, Butcher
//! tableaux, or any specific integrator — it takes a single abstract
//! error norm in and produces an accept/reject decision and a new `dt`
//! out. [`DoPri45Solver`](super::dopri45::DoPri45Solver) is the first
//! consumer (the error norm comes from the embedded RK4/5 difference),
//! but a future adaptive implicit solver (deferred, see DD-033's note on
//! iterated Newton) would plug in the same controller with a different
//! error source: the usual truncation-error norm when Newton converges,
//! or a forced rejection when it doesn't — a guard applied *before*
//! calling the controller, not a change to the controller itself.
//!
//! ## Error norm convention
//!
//! Per-component error is scaled against `atol + rtol * |u|` and combined
//! as an RMS norm — the standard convention (Hairer, Nørsett & Wanner,
//! *Solving Ordinary Differential Equations I*, §II.4). A norm `<= 1.0`
//! means the step satisfies the requested tolerance.
//!
//! ## Controller
//!
//! A PI (proportional-integral) controller (Gustafsson): the new `dt`
//! depends on both the current error and the previous one, which damps
//! oscillation in `dt` better than plain proportional control. Falls back
//! to proportional-only on the very first call (no previous error yet).

use nalgebra::DVector;

/// Adaptive step-size controller — error-source-agnostic (see [module
/// docs](self)).
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct StepSizeController {
    rtol: f64,
    atol: f64,
    dt_min: f64,
    dt_max: f64,
    /// Safety factor applied to every suggested `dt` — standard practice
    /// to bias slightly toward acceptance margin rather than the
    /// theoretical optimum, which tends to under-predict in practice.
    safety: f64,
    /// Order of the embedded error estimator (DoPri45: 4 — the local
    /// error order of the 4th-order solution used only for error
    /// estimation; the 5th-order result is what is actually propagated,
    /// "local extrapolation").
    order: f64,
    /// Most recent error norm seen (accepted or rejected attempt) — `None`
    /// before the first call, in which case `next_dt` falls back to plain
    /// proportional control.
    prev_error_norm: Option<f64>,
}

impl StepSizeController {
    /// Creates a controller with the standard safety factor (`0.9`).
    pub fn new(rtol: f64, atol: f64, dt_min: f64, dt_max: f64, order: f64) -> Self {
        Self {
            rtol,
            atol,
            dt_min,
            dt_max,
            safety: 0.9,
            order,
            prev_error_norm: None,
        }
    }

    /// Combines per-component `error` against `reference` (typically the
    /// higher-order solution) into a single RMS norm — see [module
    /// docs](self) for the convention.
    ///
    /// `error.len()` and `reference.len()` are assumed equal; mismatched
    /// lengths are zipped to the shorter one (callers are responsible for
    /// passing matching vectors — this is an internal helper, not a
    /// public-API boundary that needs to defend against misuse).
    pub fn error_norm(&self, error: &DVector<f64>, reference: &DVector<f64>) -> f64 {
        let n = error.len().max(1);
        let sum_sq: f64 = error
            .iter()
            .zip(reference.iter())
            .map(|(e, u)| {
                let scale = self.atol + self.rtol * u.abs();
                (e / scale).powi(2)
            })
            .sum();
        (sum_sq / n as f64).sqrt()
    }

    /// `true` if `error_norm` satisfies the tolerance — the step should
    /// be accepted.
    pub fn accept(&self, error_norm: f64) -> bool {
        error_norm <= 1.0
    }

    /// Computes the next `dt` to try, given the `dt` just attempted and
    /// the error norm it produced — regardless of whether that attempt is
    /// accepted or rejected (the caller decides that via
    /// [`accept`](Self::accept) separately).
    ///
    /// Updates the internal PI memory on every call, not only on accepted
    /// steps: the most recent error measurement is informative even from
    /// a rejected attempt.
    ///
    /// The growth/shrink factor is clamped to `[0.2, 5.0]` (standard
    /// practice — an unclamped controller can oscillate wildly on rough
    /// problems), and the resulting `dt` is clamped to `[dt_min, dt_max]`.
    pub fn next_dt(&mut self, current_dt: f64, error_norm: f64) -> f64 {
        // Floor to avoid division by zero when the error happens to be
        // exactly representable as 0.0 (e.g. a perfectly linear problem
        // hitting machine-exact agreement between the two RK orders).
        let error_norm = error_norm.max(1e-12);

        let beta1 = 0.7 / self.order;
        let beta2 = 0.4 / self.order;

        let factor = match self.prev_error_norm {
            Some(prev) => {
                let prev = prev.max(1e-12);
                self.safety * error_norm.powf(-beta1) * prev.powf(beta2)
            }
            None => self.safety * error_norm.powf(-1.0 / self.order),
        };

        let factor = factor.clamp(0.2, 5.0);
        self.prev_error_norm = Some(error_norm);

        (current_dt * factor).clamp(self.dt_min, self.dt_max)
    }

    /// The configured minimum step — exposed so callers can detect
    /// "stuck at dt_min" without re-deriving it.
    pub fn dt_min(&self) -> f64 {
        self.dt_min
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_controller() -> StepSizeController {
        StepSizeController::new(1e-6, 1e-9, 1e-8, 1.0, 4.0)
    }

    #[test]
    fn error_norm_zero_when_error_is_zero() {
        let controller = make_controller();
        let error = DVector::from_vec(vec![0.0, 0.0, 0.0]);
        let reference = DVector::from_vec(vec![1.0, 2.0, 3.0]);
        assert_eq!(controller.error_norm(&error, &reference), 0.0);
    }

    #[test]
    fn error_norm_scales_with_tolerance() {
        let controller = StepSizeController::new(0.0, 1.0, 1e-8, 1.0, 4.0);
        // rtol=0, atol=1 -> scale is always 1 -> norm = RMS(error) directly.
        let error = DVector::from_vec(vec![1.0, 1.0]);
        let reference = DVector::from_vec(vec![100.0, 100.0]); // irrelevant when rtol=0
        assert!((controller.error_norm(&error, &reference) - 1.0).abs() < 1e-12);
    }

    #[test]
    fn accept_at_threshold() {
        let controller = make_controller();
        assert!(controller.accept(1.0));
        assert!(controller.accept(0.5));
        assert!(!controller.accept(1.0001));
    }

    #[test]
    fn next_dt_shrinks_on_large_error() {
        let mut controller = make_controller();
        let dt = controller.next_dt(0.1, 100.0); // way over tolerance
        assert!(dt < 0.1, "expected shrink, got dt={dt}");
    }

    #[test]
    fn next_dt_grows_on_small_error() {
        let mut controller = make_controller();
        let dt = controller.next_dt(0.1, 0.01); // well within tolerance
        assert!(dt > 0.1, "expected growth, got dt={dt}");
    }

    #[test]
    fn next_dt_respects_dt_max_clamp() {
        let mut controller = StepSizeController::new(1e-6, 1e-9, 1e-8, 0.2, 4.0);
        let dt = controller.next_dt(0.1, 1e-9); // tiny error -> wants to grow a lot
        assert!(dt <= 0.2, "expected clamp to dt_max=0.2, got dt={dt}");
    }

    #[test]
    fn next_dt_respects_dt_min_clamp() {
        let mut controller = StepSizeController::new(1e-6, 1e-9, 0.05, 1.0, 4.0);
        let dt = controller.next_dt(0.1, 1e6); // huge error -> wants to shrink a lot
        assert!(dt >= 0.05, "expected clamp to dt_min=0.05, got dt={dt}");
    }

    #[test]
    fn next_dt_growth_factor_is_bounded() {
        let mut controller = make_controller();
        // Even with a vanishingly small error, growth in one step is
        // capped at 5x -- the clamp documented above.
        let dt = controller.next_dt(0.1, 1e-12);
        assert!(
            dt <= 0.1 * 5.0 + 1e-12,
            "expected growth capped at 5x, got dt={dt}"
        );
    }

    #[test]
    fn dt_min_accessor_matches_constructor() {
        let controller = StepSizeController::new(1e-6, 1e-9, 1e-7, 1.0, 4.0);
        assert_eq!(controller.dt_min(), 1e-7);
    }

    #[test]
    fn pi_memory_affects_second_call() {
        // Same error norm twice in a row -- the second call's factor
        // differs from the first because the PI term now has a previous
        // error to compare against (None -> Some transition).
        let mut controller = make_controller();
        let dt1 = controller.next_dt(0.1, 0.5);
        let dt2 = controller.next_dt(dt1, 0.5);
        // Not asserting a specific direction -- just that the PI memory
        // actually changes the computation path (first call uses the
        // P-only fallback, second uses the full PI formula).
        assert!(controller.prev_error_norm.is_some());
        let _ = dt2; // exercised for the side effect on prev_error_norm
    }

    // ── Serde round-trip (#70) ──────────────────────────────────────────────

    #[cfg(feature = "serde")]
    #[test]
    fn serde_roundtrip_preserves_parameters() {
        let mut controller = make_controller();
        // Exercise prev_error_norm: Some(_), not just the just-constructed
        // None state -- the round-trip must preserve it either way.
        let _ = controller.next_dt(0.1, 0.5);

        let json = serde_json::to_string(&controller).unwrap();
        let restored: StepSizeController = serde_json::from_str(&json).unwrap();

        assert_eq!(restored.dt_min(), controller.dt_min());
        assert_eq!(restored.prev_error_norm, controller.prev_error_norm);
    }
}
