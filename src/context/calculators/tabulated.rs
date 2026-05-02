//! # Module `context::calculators::tabulated`
//!
//! Tabulated time-dependent data calculator with interpolation.

use crate::context::calculator::ContextCalculator;
use crate::context::compute::ComputeContext;
use crate::context::error::OxiflowError;
use crate::context::value::ContextValue;
use crate::context::variable::ContextVariable;
use crate::model::traits::RequiresContext;

// ── Interpolation ─────────────────────────────────────────────────────────────

/// Interpolation strategy for [`ExternalTabulated`].
///
/// # Variants
///
/// - `Linear` — piecewise linear (1st-order accurate). Exact for linear f(t).
///
/// # Reserved
///
/// `PiecewiseCubic` (natural cubic spline, 4th-order accurate) is planned for
/// J5 (v0.7.0) and requires the `spline` feature flag.
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Interpolation {
    /// Piecewise linear interpolation: `f(t) = f_i + (f_{i+1} − f_i) · (t − t_i) / (t_{i+1} − t_i)`.
    Linear,
    // PiecewiseCubic — RESERVED J5 (v0.7.0), requires feature `spline`.
}

// ── ExternalTabulated ─────────────────────────────────────────────────────────

/// Provides a time-dependent scalar from tabulated (t, value) data.
///
/// Data points must be sorted by ascending `t`. The calculator interpolates at
/// `ctx.time()` using the chosen [`Interpolation`] strategy.
///
/// Outside the data range the calculator clamps to the nearest endpoint value
/// rather than extrapolating, and emits an `ExternalData` error if `data` is
/// empty or has only one point (interpolation is undefined).
///
/// # Examples
///
/// ```rust
/// use std::borrow::Cow;
/// use oxiflow::context::calculator::ContextCalculator;
/// use oxiflow::context::calculators::{ExternalTabulated, Interpolation};
/// use oxiflow::context::compute::ComputeContext;
/// use oxiflow::context::value::ContextValue;
/// use oxiflow::context::variable::ContextVariable;
///
/// let var = ContextVariable::External { name: Cow::Borrowed("feed_conc") };
/// let data = vec![(0.0, 1.0), (1.0, 2.0), (2.0, 1.5)];
/// let calc = ExternalTabulated::new(var, data, Interpolation::Linear).unwrap();
///
/// // Interpolate at t = 0.5  →  1.0 + (2.0 − 1.0) × 0.5 = 1.5
/// let ctx = ComputeContext::new(0.5, 0.01);
/// let val = calc.compute(&ContextValue::Scalar(0.0), &ctx).unwrap();
/// assert!((val.as_scalar().unwrap() - 1.5).abs() < 1e-10);
/// ```
#[derive(Debug)]
pub struct ExternalTabulated {
    variable: ContextVariable,
    /// (t, value) pairs, sorted by t ascending.
    data: Vec<(f64, f64)>,
    interpolation: Interpolation,
}

impl ExternalTabulated {
    /// Creates a new tabulated external calculator.
    ///
    /// # Arguments
    ///
    /// - `variable` — the `ContextVariable` this calculator provides.
    /// - `data` — `(t, value)` pairs; must be sorted by ascending `t` and
    ///   contain at least 2 points.
    /// - `interpolation` — interpolation strategy.
    ///
    /// # Errors
    ///
    /// Returns `Err(OxiflowError::ExternalData)` if `data` has fewer than 2 points
    /// or is not sorted by ascending `t`.
    pub fn new(
        variable: ContextVariable,
        data: Vec<(f64, f64)>,
        interpolation: Interpolation,
    ) -> Result<Self, OxiflowError> {
        if data.len() < 2 {
            return Err(OxiflowError::ExternalData(format!(
                "ExternalTabulated requires at least 2 data points, got {}",
                data.len()
            )));
        }

        // Verify ascending t order.
        for w in data.windows(2) {
            if w[0].0 >= w[1].0 {
                return Err(OxiflowError::ExternalData(format!(
                    "ExternalTabulated data must be sorted by ascending t: \
                     t[i]={} >= t[i+1]={}",
                    w[0].0, w[1].0
                )));
            }
        }

        Ok(Self {
            variable,
            data,
            interpolation,
        })
    }

    /// Interpolates the tabulated data at time `t`.
    ///
    /// Clamps to endpoint values outside the data range.
    fn interpolate(&self, t: f64) -> f64 {
        let (t_min, v_min) = self.data[0];
        let (t_max, v_max) = *self.data.last().unwrap();

        // Clamp outside range.
        if t <= t_min {
            return v_min;
        }
        if t >= t_max {
            return v_max;
        }

        // Binary search for the bracketing interval.
        let idx = self
            .data
            .partition_point(|(ti, _)| *ti <= t)
            .saturating_sub(1);

        let (t0, v0) = self.data[idx];
        let (t1, v1) = self.data[idx + 1];

        match self.interpolation {
            Interpolation::Linear => v0 + (v1 - v0) * (t - t0) / (t1 - t0),
            // J5+: PiecewiseCubic will be added here.
            #[allow(unreachable_patterns)]
            _ => v0, // unreachable at J2
        }
    }
}

impl RequiresContext for ExternalTabulated {
    fn required_variables(&self) -> Vec<ContextVariable> {
        vec![]
    }

    // External data runs before derived quantities (priority 100) but after
    // time built-ins (priority 0).
    fn priority(&self) -> u32 {
        50
    }
}

impl ContextCalculator for ExternalTabulated {
    fn provides(&self) -> ContextVariable {
        self.variable.clone()
    }

    fn compute(
        &self,
        _state: &ContextValue,
        ctx: &ComputeContext,
    ) -> Result<ContextValue, OxiflowError> {
        let value = self.interpolate(ctx.time());
        Ok(ContextValue::Scalar(value))
    }

    fn name(&self) -> &str {
        "external_tabulated (built-in)"
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use std::borrow::Cow;

    use super::*;

    fn var() -> ContextVariable {
        ContextVariable::External {
            name: Cow::Borrowed("feed"),
        }
    }

    fn linear_data() -> Vec<(f64, f64)> {
        vec![(0.0, 0.0), (1.0, 1.0), (2.0, 2.0)]
    }

    fn calc(data: Vec<(f64, f64)>) -> ExternalTabulated {
        ExternalTabulated::new(var(), data, Interpolation::Linear).unwrap()
    }

    fn ctx(t: f64) -> ComputeContext {
        ComputeContext::new(t, 0.01)
    }

    // ── constructor ───────────────────────────────────────────────────────────

    #[test]
    fn new_succeeds_with_valid_data() {
        assert!(ExternalTabulated::new(var(), linear_data(), Interpolation::Linear).is_ok());
    }

    #[test]
    fn new_fails_with_single_point() {
        let result = ExternalTabulated::new(var(), vec![(0.0, 1.0)], Interpolation::Linear);
        assert!(matches!(result, Err(OxiflowError::ExternalData(_))));
    }

    #[test]
    fn new_fails_with_empty_data() {
        let result = ExternalTabulated::new(var(), vec![], Interpolation::Linear);
        assert!(matches!(result, Err(OxiflowError::ExternalData(_))));
    }

    #[test]
    fn new_fails_when_not_sorted() {
        let result =
            ExternalTabulated::new(var(), vec![(1.0, 1.0), (0.0, 0.0)], Interpolation::Linear);
        assert!(matches!(result, Err(OxiflowError::ExternalData(_))));
    }

    #[test]
    fn new_fails_on_duplicate_t() {
        let result = ExternalTabulated::new(
            var(),
            vec![(0.0, 0.0), (0.0, 1.0), (1.0, 2.0)],
            Interpolation::Linear,
        );
        assert!(matches!(result, Err(OxiflowError::ExternalData(_))));
    }

    // ── provides / priority ───────────────────────────────────────────────────

    #[test]
    fn provides_configured_variable() {
        let v = var();
        let c = calc(linear_data());
        assert_eq!(c.provides(), v);
    }

    #[test]
    fn priority_is_fifty() {
        assert_eq!(calc(linear_data()).priority(), 50);
    }

    // ── linear interpolation ──────────────────────────────────────────────────

    #[test]
    fn interpolates_at_midpoint() {
        // data: (0,0), (1,1), (2,2)  →  at t=0.5: value = 0.5
        let c = calc(linear_data());
        let val = c.compute(&ContextValue::Scalar(0.0), &ctx(0.5)).unwrap();
        assert!((val.as_scalar().unwrap() - 0.5).abs() < 1e-10);
    }

    #[test]
    fn interpolates_exactly_at_knot() {
        let c = calc(linear_data());
        let val = c.compute(&ContextValue::Scalar(0.0), &ctx(1.0)).unwrap();
        assert!((val.as_scalar().unwrap() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn non_linear_interpolation_between_knots() {
        // data: (0, 1.0), (1, 2.0), (2, 1.5)
        let data = vec![(0.0, 1.0), (1.0, 2.0), (2.0, 1.5)];
        let c = calc(data);
        // t = 1.5 → between (1, 2.0) and (2, 1.5) → 2.0 + (1.5 - 2.0) * 0.5 = 1.75
        let val = c.compute(&ContextValue::Scalar(0.0), &ctx(1.5)).unwrap();
        assert!((val.as_scalar().unwrap() - 1.75).abs() < 1e-10);
    }

    // ── clamping ──────────────────────────────────────────────────────────────

    #[test]
    fn clamps_to_first_value_before_range() {
        let c = calc(linear_data());
        let val = c.compute(&ContextValue::Scalar(0.0), &ctx(-1.0)).unwrap();
        assert_eq!(val.as_scalar().unwrap(), 0.0);
    }

    #[test]
    fn clamps_to_last_value_after_range() {
        let c = calc(linear_data());
        let val = c.compute(&ContextValue::Scalar(0.0), &ctx(5.0)).unwrap();
        assert_eq!(val.as_scalar().unwrap(), 2.0);
    }

    #[test]
    fn clamps_exactly_at_lower_bound() {
        let c = calc(linear_data());
        let val = c.compute(&ContextValue::Scalar(0.0), &ctx(0.0)).unwrap();
        assert_eq!(val.as_scalar().unwrap(), 0.0);
    }

    #[test]
    fn clamps_exactly_at_upper_bound() {
        let c = calc(linear_data());
        let val = c.compute(&ContextValue::Scalar(0.0), &ctx(2.0)).unwrap();
        assert_eq!(val.as_scalar().unwrap(), 2.0);
    }

    // ── object safety ─────────────────────────────────────────────────────────

    #[test]
    fn is_object_safe() {
        let c: Box<dyn ContextCalculator> = Box::new(calc(linear_data()));
        assert_eq!(c.provides(), var());
    }
}
