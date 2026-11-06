//! Conductivity graded linearly along a bar, the coefficient profile of
//! case (b) of the heat-transfer reference example.
//!
//! Shared by the heat-transfer scenario `rsc/models/heat_transfer_1d.rs`
//! (the example `examples/heat_transfer_1d.rs`, which documents it, and
//! its plot) and by `benches/heat_transfer_1d.rs`, each of which includes
//! this file as a module with `#[path]`. Kept apart from the scenario
//! because the benchmark times the coefficient field alone, without the
//! model.

/// Conductivity graded linearly along a bar of length `length`:
/// `lambda(x) = lambda0 * (1 + beta * x / length)`.
#[derive(Debug, Clone, Copy)]
pub(crate) struct GradedConductivity {
    /// Conductivity at x = 0, in W/(m*K).
    pub(crate) lambda0: f64,
    /// Relative increase over the bar: lambda(length) = lambda0 * (1 + beta).
    pub(crate) beta: f64,
    /// Bar length, in m.
    pub(crate) length: f64,
}

impl GradedConductivity {
    /// lambda(x), in W/(m*K).
    pub(crate) fn value(&self, x: f64) -> f64 {
        self.lambda0 * (1.0 + self.beta * x / self.length)
    }

    /// lambda'(x), constant along the bar, in W/(m^2*K).
    // Not used by benches/heat_transfer_1d.rs, which only times lambda(x).
    #[allow(dead_code)]
    pub(crate) fn slope(&self) -> f64 {
        self.lambda0 * self.beta / self.length
    }
}
