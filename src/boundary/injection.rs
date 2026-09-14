//! # Module `boundary::injection`
//!
//! Time-varying injection profiles for a domain inlet, and the
//! [`BoundaryCondition`] that exposes one as a ghost value —
//! [`InjectionInlet`]. Motivated by #138 (competitive-adsorption
//! chromatography): the inlet concentration is a genuine function of time
//! (a pulse, a ramp, a sustained feed), not a fixed Robin/Danckwerts-style
//! formula — see [`ghost_value`](crate::boundary::BoundaryCondition::ghost_value)'s
//! own docs for why that needed a signature change (DD-042's two original
//! motivating cases were both steady-state).
//!
//! ## Dirac is deliberately not one of these profiles
//!
//! chrom-rs models a Dirac injection as a boundary *event* at a single
//! instant — a flux impulse, numerically awkward (infinite rate over zero
//! time). This crate takes a different, simpler stance: a Dirac injection
//! is an **initial condition**, not a boundary profile — the injected mass
//! is already inside the domain at `t = 0` (see [`dirac_initial_state`]),
//! and the inlet stays closed afterward. It therefore has no place in the
//! [`TemporalInjection`] enum, which is exclusively for profiles evaluated
//! *at the boundary as time advances* — a Dirac has nothing to evaluate
//! past `t = 0`.

use crate::boundary::{BoundaryCondition, BoundaryType};
use crate::context::compute::ComputeContext;
use crate::context::error::OxiflowError;
use crate::context::variable::ContextVariable;
use crate::mesh::Mesh;
use crate::model::traits::RequiresContext;
use nalgebra::DVector;

// ── TemporalInjection ────────────────────────────────────────────────────────

/// A concentration profile evaluated at a domain inlet, as a function of
/// simulation time.
///
/// `#[non_exhaustive]`: new profile shapes may be added without a breaking
/// change (mirrors `FluxBoundary`'s own convention in `operators::mod`).
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq)]
pub enum TemporalInjection {
    /// Constant `concentration` on `[start, end)`, zero outside — a single
    /// rectangular pulse.
    Rectangle {
        start: f64,
        end: f64,
        concentration: f64,
    },
    /// Gaussian pulse: `peak_concentration * exp(-(t-center)^2 / (2*width^2))`.
    /// Never exactly zero, but negligible a few `width`s away from `center`.
    Gaussian {
        center: f64,
        width: f64,
        peak_concentration: f64,
    },
    /// Linear ramp up over `[ramp_up_start, plateau_start)`, constant
    /// `concentration` over `[plateau_start, plateau_end)`, linear ramp
    /// down over `[plateau_end, ramp_down_end)`, zero outside
    /// `[ramp_up_start, ramp_down_end)`.
    Trapezoid {
        ramp_up_start: f64,
        plateau_start: f64,
        plateau_end: f64,
        ramp_down_end: f64,
        concentration: f64,
    },
    /// Constant `concentration` from `start` onward, indefinitely — a
    /// continuous feed rather than a pulse (steady-state operation, not
    /// modeled by any of the pulse-shaped variants above).
    SustainedStep { start: f64, concentration: f64 },
    /// Exponential decay from `concentration` at `start`, time constant
    /// `tau` (`concentration * exp(-(t-start)/tau)` for `t >= start`, zero
    /// before) — a bolus followed by a wash-out, common after a Rectangle
    /// or Gaussian pulse in practice, modeled here as its own standalone
    /// profile rather than composed with one.
    ExponentialDecay {
        start: f64,
        concentration: f64,
        tau: f64,
    },
}

impl TemporalInjection {
    /// Evaluates the profile at time `t`.
    pub fn evaluate(&self, t: f64) -> f64 {
        match self {
            Self::Rectangle {
                start,
                end,
                concentration,
            } => {
                if t >= *start && t < *end {
                    *concentration
                } else {
                    0.0
                }
            }
            Self::Gaussian {
                center,
                width,
                peak_concentration,
            } => peak_concentration * (-(t - center).powi(2) / (2.0 * width.powi(2))).exp(),
            Self::Trapezoid {
                ramp_up_start,
                plateau_start,
                plateau_end,
                ramp_down_end,
                concentration,
            } => {
                if t < *ramp_up_start || t >= *ramp_down_end {
                    0.0
                } else if t < *plateau_start {
                    concentration * (t - ramp_up_start) / (plateau_start - ramp_up_start)
                } else if t < *plateau_end {
                    *concentration
                } else {
                    concentration * (ramp_down_end - t) / (ramp_down_end - plateau_end)
                }
            }
            Self::SustainedStep {
                start,
                concentration,
            } => {
                if t >= *start {
                    *concentration
                } else {
                    0.0
                }
            }
            Self::ExponentialDecay {
                start,
                concentration,
                tau,
            } => {
                if t < *start {
                    0.0
                } else {
                    concentration * (-(t - start) / tau).exp()
                }
            }
        }
    }
}

// ── InjectionInlet ────────────────────────────────────────────────────────────

/// A [`BoundaryCondition`] whose ghost value follows a [`TemporalInjection`]
/// profile, evaluated at the current simulation time via `ctx.time()`.
///
/// Applies no direct constraint to `state` (`apply()` is a no-op): the
/// profile only supplies the ghost value `FluxBoundary::GhostCell` needs to
/// close its face-flux/gradient stencil at the boundary. Interior evolution
/// (including the boundary node itself) proceeds through the normal
/// transport equation, same posture as `DanckwertsInlet`'s `apply()`
/// leaving interior dynamics to `compute_physics`.
#[derive(Debug, Clone)]
pub struct InjectionInlet {
    profile: TemporalInjection,
}

impl InjectionInlet {
    /// Creates an inlet boundary condition from a [`TemporalInjection`]
    /// profile.
    pub fn new(profile: TemporalInjection) -> Self {
        Self { profile }
    }
}

impl RequiresContext for InjectionInlet {
    fn required_variables(&self) -> Vec<ContextVariable> {
        // The profile is a pure function of Time, which every solver
        // populates unconditionally -- nothing to declare.
        vec![]
    }
}

impl BoundaryCondition for InjectionInlet {
    fn boundary_type(&self) -> BoundaryType {
        // Prescribing a concentration value at the boundary -- Dirichlet,
        // even though it arrives through the ghost-value mechanism rather
        // than a direct `apply()` constraint.
        BoundaryType::Dirichlet
    }

    fn apply(
        &self,
        _state: &mut DVector<f64>,
        _ctx: &ComputeContext,
        _mesh: &dyn Mesh,
    ) -> Result<(), OxiflowError> {
        Ok(())
    }

    fn ghost_value(
        &self,
        _depth: usize,
        _interior_at_depth: f64,
        _dx: f64,
        ctx: &ComputeContext,
    ) -> Option<f64> {
        Some(self.profile.evaluate(ctx.time()))
    }
}

// ── Dirac — initial condition, not a boundary profile ──────────────────────────

/// Builds a Dirac-like initial condition: all `amount` concentrated at the
/// first node, zero elsewhere — the injected mass is already inside the
/// domain at `t = 0`, rather than delivered as a boundary event over time.
/// See the module documentation for why this crate treats Dirac
/// differently from chrom-rs (a boundary-flux impulse there).
///
/// Returns a zero vector unchanged if `n_dof == 0`.
pub fn dirac_initial_state(n_dof: usize, amount: f64) -> DVector<f64> {
    let mut v = DVector::zeros(n_dof);
    if n_dof > 0 {
        v[0] = amount;
    }
    v
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── Rectangle ────────────────────────────────────────────────────────────

    #[test]
    fn rectangle_is_zero_outside_the_window() {
        let p = TemporalInjection::Rectangle {
            start: 1.0,
            end: 3.0,
            concentration: 0.5,
        };
        assert_eq!(p.evaluate(0.0), 0.0);
        assert_eq!(p.evaluate(3.0), 0.0); // half-open: end is excluded
        assert_eq!(p.evaluate(10.0), 0.0);
    }

    #[test]
    fn rectangle_is_constant_inside_the_window() {
        let p = TemporalInjection::Rectangle {
            start: 1.0,
            end: 3.0,
            concentration: 0.5,
        };
        assert_eq!(p.evaluate(1.0), 0.5); // start is included
        assert_eq!(p.evaluate(2.0), 0.5);
        assert!((p.evaluate(2.999) - 0.5).abs() < 1e-12);
    }

    // ── Gaussian ─────────────────────────────────────────────────────────────

    #[test]
    fn gaussian_peaks_exactly_at_center() {
        let p = TemporalInjection::Gaussian {
            center: 5.0,
            width: 2.0,
            peak_concentration: 3.0,
        };
        assert!((p.evaluate(5.0) - 3.0).abs() < 1e-12);
    }

    #[test]
    fn gaussian_is_symmetric_around_center() {
        let p = TemporalInjection::Gaussian {
            center: 5.0,
            width: 2.0,
            peak_concentration: 3.0,
        };
        assert!((p.evaluate(5.0 - 1.3) - p.evaluate(5.0 + 1.3)).abs() < 1e-12);
    }

    #[test]
    fn gaussian_at_one_width_from_center_matches_the_closed_form() {
        let p = TemporalInjection::Gaussian {
            center: 5.0,
            width: 2.0,
            peak_concentration: 3.0,
        };
        // exp(-w^2/(2*w^2)) = exp(-0.5) at exactly one width away.
        let expected = 3.0 * (-0.5f64).exp();
        assert!((p.evaluate(5.0 + 2.0) - expected).abs() < 1e-12);
    }

    // ── Trapezoid ────────────────────────────────────────────────────────────

    fn sample_trapezoid() -> TemporalInjection {
        TemporalInjection::Trapezoid {
            ramp_up_start: 0.0,
            plateau_start: 2.0,
            plateau_end: 5.0,
            ramp_down_end: 7.0,
            concentration: 10.0,
        }
    }

    #[test]
    fn trapezoid_is_zero_before_and_after_the_window() {
        let p = sample_trapezoid();
        assert_eq!(p.evaluate(-1.0), 0.0);
        assert_eq!(p.evaluate(7.0), 0.0); // half-open at the far end
        assert_eq!(p.evaluate(100.0), 0.0);
    }

    #[test]
    fn trapezoid_ramps_linearly_up_and_down() {
        let p = sample_trapezoid();
        // Midpoint of the up-ramp [0,2) -> half of concentration.
        assert!((p.evaluate(1.0) - 5.0).abs() < 1e-12);
        // Midpoint of the down-ramp [5,7) -> half of concentration.
        assert!((p.evaluate(6.0) - 5.0).abs() < 1e-12);
    }

    #[test]
    fn trapezoid_is_constant_on_the_plateau() {
        let p = sample_trapezoid();
        assert!((p.evaluate(2.0) - 10.0).abs() < 1e-12);
        assert!((p.evaluate(3.5) - 10.0).abs() < 1e-12);
        assert!((p.evaluate(4.999) - 10.0).abs() < 1e-12);
    }

    // ── SustainedStep ────────────────────────────────────────────────────────

    #[test]
    fn sustained_step_stays_on_indefinitely() {
        let p = TemporalInjection::SustainedStep {
            start: 2.0,
            concentration: 7.0,
        };
        assert_eq!(p.evaluate(0.0), 0.0);
        assert_eq!(p.evaluate(2.0), 7.0);
        assert_eq!(p.evaluate(1_000_000.0), 7.0);
    }

    // ── ExponentialDecay ─────────────────────────────────────────────────────

    #[test]
    fn exponential_decay_matches_the_closed_form() {
        let p = TemporalInjection::ExponentialDecay {
            start: 1.0,
            concentration: 4.0,
            tau: 2.0,
        };
        assert_eq!(p.evaluate(0.0), 0.0); // before start
        assert!((p.evaluate(1.0) - 4.0).abs() < 1e-12); // at start
                                                        // One time constant later: concentration / e.
        let expected = 4.0 * (-1.0f64).exp();
        assert!((p.evaluate(3.0) - expected).abs() < 1e-12);
    }

    // ── InjectionInlet ───────────────────────────────────────────────────────

    #[test]
    fn injection_inlet_ghost_value_follows_the_profile_at_ctx_time() {
        let profile = TemporalInjection::Rectangle {
            start: 0.0,
            end: 1.0,
            concentration: 0.2,
        };
        let inlet = InjectionInlet::new(profile);

        let ctx_inside = ComputeContext::new(0.5, 0.01);
        assert_eq!(inlet.ghost_value(1, 0.0, 0.01, &ctx_inside), Some(0.2));

        let ctx_outside = ComputeContext::new(5.0, 0.01);
        assert_eq!(inlet.ghost_value(1, 0.0, 0.01, &ctx_outside), Some(0.0));
    }

    #[test]
    fn injection_inlet_apply_does_not_modify_state() {
        let inlet = InjectionInlet::new(TemporalInjection::SustainedStep {
            start: 0.0,
            concentration: 1.0,
        });
        let mut state = DVector::from_vec(vec![1.0, 2.0, 3.0]);
        let original = state.clone();
        let ctx = ComputeContext::new(10.0, 0.01);

        struct DummyMesh;
        impl crate::mesh::Mesh for DummyMesh {
            fn n_dof(&self) -> usize {
                3
            }
            fn coordinates(&self, i: usize) -> &[f64] {
                const COORDS: [[f64; 1]; 3] = [[0.0], [1.0], [2.0]];
                &COORDS[i]
            }
            fn spatial_dimension(&self) -> usize {
                1
            }
            fn characteristic_length(&self) -> f64 {
                1.0
            }
        }

        inlet.apply(&mut state, &ctx, &DummyMesh).unwrap();
        assert_eq!(state, original);
    }

    #[test]
    fn injection_inlet_boundary_type_is_dirichlet() {
        let inlet = InjectionInlet::new(TemporalInjection::SustainedStep {
            start: 0.0,
            concentration: 1.0,
        });
        assert_eq!(inlet.boundary_type(), BoundaryType::Dirichlet);
    }

    // ── dirac_initial_state ──────────────────────────────────────────────────

    #[test]
    fn dirac_initial_state_concentrates_amount_at_the_first_node() {
        let v = dirac_initial_state(5, 3.0);
        assert_eq!(v[0], 3.0);
        assert!(v.iter().skip(1).all(|&x| x == 0.0));
    }

    #[test]
    fn dirac_initial_state_handles_empty_mesh() {
        let v = dirac_initial_state(0, 3.0);
        assert_eq!(v.len(), 0);
    }
}
