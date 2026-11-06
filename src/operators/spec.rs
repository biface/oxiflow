//! # Module `operators::spec`
//!
//! External, serialisable description of a flux scheme — the scheme part of
//! the WHAT-axis configuration ([DT-027](https://github.com/biface/oxiflow/issues/75)
//! amendment 1, [#108](https://github.com/biface/oxiflow/issues/108)).
//!
//! [`FluxSchemeSpec`] names one of the [`FluxDivergenceOperator`](super::FluxDivergenceOperator)
//! schemes of this module and its constructor parameters; a variant drops
//! the role suffix of its type (`FVUpwindFlux` is `FVUpwind`), as
//! `IntegratorSpec` does for solvers (`sparse` feature). It is
//! consumed by [`ModelSpec`](crate::model::ModelSpec), which wraps the scheme
//! in a [`DiscretizedModel`](crate::model::DiscretizedModel).
//!
//! [`FluxBoundarySpec`] covers the stencil closures that hold no data.
//! [`FluxBoundary::GhostCell`] is left out: it holds the boundary conditions
//! of the domain by position, a form that DD-051
//! ([#152](https://github.com/biface/oxiflow/issues/152)) is to revisit.

use super::limiters::Limiter;
use super::FluxBoundary;

/// Closure of the stencil at the two ends of the domain; see
/// [`FluxBoundary`] for each convention.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Deserialize))]
pub enum FluxBoundarySpec {
    /// [`FluxBoundary::Periodic`].
    Periodic,
    /// [`FluxBoundary::Truncation`].
    Truncation,
}

impl FluxBoundarySpec {
    /// The closure this spec describes.
    pub(crate) fn into_flux_boundary(self) -> FluxBoundary {
        match self {
            Self::Periodic => FluxBoundary::Periodic,
            Self::Truncation => FluxBoundary::Truncation,
        }
    }
}

/// Which flux scheme to build, and with what parameters. Each variant takes
/// the parameters of the scheme's constructor.
///
/// With the `serde` feature, a scheme is written as its variant name around
/// its parameters:
///
/// ```text
/// scheme:
///   Limited: { velocity: 1.0, diffusion: 1.0e-3, limiter: VanLeer, boundary: Truncation }
/// ```
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Deserialize))]
pub enum FluxSchemeSpec {
    /// [`FVCenteredFlux`](super::fv::FVCenteredFlux).
    FVCentered {
        /// Advection velocity.
        velocity: f64,
        /// Diffusion coefficient.
        diffusion: f64,
        /// Closure of the stencil.
        boundary: FluxBoundarySpec,
    },
    /// [`FVUpwindFlux`](super::fv::FVUpwindFlux).
    FVUpwind {
        /// Advection velocity.
        velocity: f64,
        /// Diffusion coefficient.
        diffusion: f64,
        /// Closure of the stencil.
        boundary: FluxBoundarySpec,
    },
    /// [`WENO3`](super::weno::WENO3).
    WENO3 {
        /// Advection velocity.
        velocity: f64,
        /// Diffusion coefficient.
        diffusion: f64,
        /// Closure of the stencil.
        boundary: FluxBoundarySpec,
    },
    /// [`WENO5`](super::weno::WENO5).
    WENO5 {
        /// Advection velocity.
        velocity: f64,
        /// Diffusion coefficient.
        diffusion: f64,
        /// Closure of the stencil.
        boundary: FluxBoundarySpec,
    },
    /// [`LimitedFlux`](super::limiters::LimitedFlux).
    Limited {
        /// Advection velocity.
        velocity: f64,
        /// Diffusion coefficient.
        diffusion: f64,
        /// Flux limiter.
        limiter: Limiter,
        /// Closure of the stencil.
        boundary: FluxBoundarySpec,
    },
    /// [`AdaptiveFlux`](super::adaptive::AdaptiveFlux).
    Adaptive {
        /// Advection velocity.
        velocity: f64,
        /// Diffusion coefficient.
        diffusion: f64,
        /// Flux limiter of the blended limited flux.
        limiter: Limiter,
        /// Mesh Péclet number above which the limited flux is blended in.
        peclet_threshold: f64,
        /// Closure of the stencil.
        boundary: FluxBoundarySpec,
    },
}

impl FluxSchemeSpec {
    /// Short name of the scheme, the default name of a model built on it.
    pub(crate) fn label(&self) -> &'static str {
        match self {
            Self::FVCentered { .. } => "fv_centered",
            Self::FVUpwind { .. } => "fv_upwind",
            Self::WENO3 { .. } => "weno3",
            Self::WENO5 { .. } => "weno5",
            Self::Limited { .. } => "limited",
            Self::Adaptive { .. } => "adaptive",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(feature = "serde")]
    fn read(json: &str) -> FluxSchemeSpec {
        serde_json::from_str(json).unwrap()
    }

    #[cfg(feature = "serde")]
    #[test]
    fn every_scheme_reads_from_its_parameters() {
        let common = r#""velocity": 1.0, "diffusion": 0.5, "boundary": "Truncation""#;
        let cases = [
            ("FVCentered", "fv_centered"),
            ("FVUpwind", "fv_upwind"),
            ("WENO3", "weno3"),
            ("WENO5", "weno5"),
        ];
        for (variant, label) in cases {
            let spec = read(&format!(r#"{{ "{variant}": {{ {common} }} }}"#));
            assert_eq!(spec.label(), label);
        }
        let limited = read(&format!(
            r#"{{ "Limited": {{ {common}, "limiter": "VanLeer" }} }}"#
        ));
        assert_eq!(
            limited,
            FluxSchemeSpec::Limited {
                velocity: 1.0,
                diffusion: 0.5,
                limiter: Limiter::VanLeer,
                boundary: FluxBoundarySpec::Truncation
            }
        );
        let adaptive = read(&format!(
            r#"{{ "Adaptive": {{ {common}, "limiter": "MinMod", "peclet_threshold": 2.0 }} }}"#
        ));
        assert_eq!(adaptive.label(), "adaptive");
    }

    #[cfg(feature = "serde")]
    #[test]
    fn a_closure_holding_boundary_conditions_is_refused() {
        let json =
            r#"{ "FVUpwind": { "velocity": 1.0, "diffusion": 0.0, "boundary": "GhostCell" } }"#;
        assert!(serde_json::from_str::<FluxSchemeSpec>(json).is_err());
    }

    #[test]
    fn closures_map_to_their_flux_boundary() {
        assert!(matches!(
            FluxBoundarySpec::Periodic.into_flux_boundary(),
            FluxBoundary::Periodic
        ));
        assert!(matches!(
            FluxBoundarySpec::Truncation.into_flux_boundary(),
            FluxBoundary::Truncation
        ));
    }
}
