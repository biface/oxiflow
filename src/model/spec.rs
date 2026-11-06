//! # Module `model::spec`
//!
//! External, serialisable description of a physical model and of its initial
//! state — the model part of the WHAT-axis configuration
//! ([DT-027](https://github.com/biface/oxiflow/issues/75) amendment 1,
//! [#108](https://github.com/biface/oxiflow/issues/108)).
//!
//! [`ModelSpec`] describes the model of a domain; [`InitialStateSpec`] the
//! state it starts from, when it is not the model's own. Both are consumed by
//! [`DomainSpec`](crate::solver::DomainSpec), which holds the mesh they are
//! built on.
//!
//! Only the models built from public components are covered: a
//! [`DiscretizedModel`] over one of the flux schemes of
//! [`operators`](crate::operators). [`CompositeModel`](super::CompositeModel)
//! is left out of this first increment, since no public source term exists
//! to compose with a flux; physical models defined outside the crate cannot
//! be described ([DD-072](https://github.com/biface/oxiflow/issues/215)).

use std::sync::Arc;

use nalgebra::DVector;

use super::{DiscretizedModel, PhysicalModel};
use crate::boundary::dirac_initial_state;
use crate::context::value::ContextValue;
use crate::mesh::UniformGrid1D;
use crate::operators::adaptive::AdaptiveFlux;
use crate::operators::fv::{FVCenteredFlux, FVUpwindFlux};
use crate::operators::limiters::LimitedFlux;
use crate::operators::spec::FluxSchemeSpec;
use crate::operators::weno::{WENO3, WENO5};

/// Which physical model to build, and with what parameters.
///
/// With the `serde` feature, a model is written as its variant name around
/// its parameters:
///
/// ```text
/// model:
///   Discretized:
///     scheme:
///       FVUpwind: { velocity: 1.0e-3, diffusion: 1.0e-6, boundary: Truncation }
///     name: advection   # optional
/// ```
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Deserialize))]
pub enum ModelSpec {
    /// A [`DiscretizedModel`] over the flux scheme `scheme`, on the mesh of
    /// the domain.
    Discretized {
        /// The flux scheme.
        scheme: FluxSchemeSpec,
        /// Name of the model, used in diagnostics only; the short name of
        /// the scheme (`fv_upwind`, `weno5`…) when absent.
        #[cfg_attr(feature = "serde", serde(default))]
        name: Option<String>,
    },
}

impl ModelSpec {
    /// The model this spec describes, on `grid`.
    pub(crate) fn into_model(self, grid: Arc<UniformGrid1D>) -> Box<dyn PhysicalModel> {
        match self {
            Self::Discretized { scheme, name } => {
                let name = name.unwrap_or_else(|| scheme.label().to_owned());
                discretized(scheme, grid, name)
            }
        }
    }
}

/// A [`DiscretizedModel`] over the scheme `scheme` describes.
fn discretized(
    scheme: FluxSchemeSpec,
    grid: Arc<UniformGrid1D>,
    name: String,
) -> Box<dyn PhysicalModel> {
    match scheme {
        FluxSchemeSpec::FVCentered {
            velocity,
            diffusion,
            boundary,
        } => Box::new(DiscretizedModel::new(
            Arc::new(FVCenteredFlux::new(
                velocity,
                diffusion,
                boundary.into_flux_boundary(),
            )),
            grid,
            name,
        )),
        FluxSchemeSpec::FVUpwind {
            velocity,
            diffusion,
            boundary,
        } => Box::new(DiscretizedModel::new(
            Arc::new(FVUpwindFlux::new(
                velocity,
                diffusion,
                boundary.into_flux_boundary(),
            )),
            grid,
            name,
        )),
        FluxSchemeSpec::WENO3 {
            velocity,
            diffusion,
            boundary,
        } => Box::new(DiscretizedModel::new(
            Arc::new(WENO3::new(
                velocity,
                diffusion,
                boundary.into_flux_boundary(),
            )),
            grid,
            name,
        )),
        FluxSchemeSpec::WENO5 {
            velocity,
            diffusion,
            boundary,
        } => Box::new(DiscretizedModel::new(
            Arc::new(WENO5::new(
                velocity,
                diffusion,
                boundary.into_flux_boundary(),
            )),
            grid,
            name,
        )),
        FluxSchemeSpec::Limited {
            velocity,
            diffusion,
            limiter,
            boundary,
        } => Box::new(DiscretizedModel::new(
            Arc::new(LimitedFlux::new(
                velocity,
                diffusion,
                limiter,
                boundary.into_flux_boundary(),
            )),
            grid,
            name,
        )),
        FluxSchemeSpec::Adaptive {
            velocity,
            diffusion,
            limiter,
            peclet_threshold,
            boundary,
        } => Box::new(DiscretizedModel::new(
            Arc::new(AdaptiveFlux::new(
                velocity,
                diffusion,
                limiter,
                peclet_threshold,
                boundary.into_flux_boundary(),
            )),
            grid,
            name,
        )),
    }
}

/// The state a domain starts from, one value per node of its mesh.
///
/// With the `serde` feature:
///
/// ```text
/// initial_state: { Uniform: 0.0 }
/// initial_state: { Values: [0.0, 0.5, 1.0] }
/// initial_state: { Dirac: { amount: 1.0 } }
/// ```
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Deserialize))]
pub enum InitialStateSpec {
    /// The same value on every node.
    Uniform(f64),
    /// One value per node, in the order of the mesh. A list whose length
    /// differs from the number of nodes is refused when the domain is built.
    Values(Vec<f64>),
    /// `amount` on the first node, zero elsewhere; see
    /// [`dirac_initial_state`].
    Dirac {
        /// Value on the first node.
        amount: f64,
    },
}

impl InitialStateSpec {
    /// The state this spec describes, on a mesh of `n_dof` nodes. The length
    /// of [`Self::Values`] is not checked here: the domain checks the state
    /// against its mesh once built.
    pub(crate) fn into_state(self, n_dof: usize) -> ContextValue {
        let field = match self {
            Self::Uniform(value) => DVector::from_element(n_dof, value),
            Self::Values(values) => DVector::from_vec(values),
            Self::Dirac { amount } => dirac_initial_state(n_dof, amount),
        };
        ContextValue::ScalarField(field)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::operators::spec::FluxBoundarySpec;

    fn upwind() -> FluxSchemeSpec {
        FluxSchemeSpec::FVUpwind {
            velocity: 1.0,
            diffusion: 0.0,
            boundary: FluxBoundarySpec::Periodic,
        }
    }

    fn grid() -> Arc<UniformGrid1D> {
        Arc::new(UniformGrid1D::new(5, 0.0, 1.0).unwrap())
    }

    #[test]
    fn a_model_takes_the_name_of_its_scheme_by_default() {
        let spec = ModelSpec::Discretized {
            scheme: upwind(),
            name: None,
        };
        assert_eq!(spec.into_model(grid()).name(), "fv_upwind");
    }

    #[test]
    fn a_given_name_replaces_the_default() {
        let spec = ModelSpec::Discretized {
            scheme: upwind(),
            name: Some("advection".to_owned()),
        };
        assert_eq!(spec.into_model(grid()).name(), "advection");
    }

    #[test]
    fn every_scheme_builds_a_model_named_after_it() {
        let boundary = FluxBoundarySpec::Truncation;
        let schemes = [
            FluxSchemeSpec::FVCentered {
                velocity: 1.0,
                diffusion: 0.1,
                boundary,
            },
            upwind(),
            FluxSchemeSpec::WENO3 {
                velocity: 1.0,
                diffusion: 0.1,
                boundary,
            },
            FluxSchemeSpec::WENO5 {
                velocity: 1.0,
                diffusion: 0.1,
                boundary,
            },
            FluxSchemeSpec::Limited {
                velocity: 1.0,
                diffusion: 0.1,
                limiter: crate::operators::limiters::Limiter::MinMod,
                boundary,
            },
            FluxSchemeSpec::Adaptive {
                velocity: 1.0,
                diffusion: 0.1,
                limiter: crate::operators::limiters::Limiter::Superbee,
                peclet_threshold: 2.0,
                boundary,
            },
        ];
        for scheme in schemes {
            let label = scheme.label();
            let model = ModelSpec::Discretized { scheme, name: None }.into_model(grid());
            assert_eq!(model.name(), label);
        }
    }

    #[test]
    fn initial_states_have_one_value_per_node() {
        let field = |spec: InitialStateSpec| spec.into_state(4).as_scalar_field().unwrap().clone();
        assert_eq!(
            field(InitialStateSpec::Uniform(1.5)),
            DVector::from_element(4, 1.5)
        );
        assert_eq!(
            field(InitialStateSpec::Values(vec![0.0, 1.0, 2.0, 3.0])),
            DVector::from_vec(vec![0.0, 1.0, 2.0, 3.0])
        );
        assert_eq!(
            field(InitialStateSpec::Dirac { amount: 2.0 }),
            dirac_initial_state(4, 2.0)
        );
    }

    #[cfg(feature = "serde")]
    #[test]
    fn a_model_reads_with_or_without_its_name() {
        let scheme = r#""scheme": { "FVUpwind": { "velocity": 1.0, "diffusion": 0.0, "boundary": "Periodic" } }"#;
        let unnamed: ModelSpec =
            serde_json::from_str(&format!(r#"{{ "Discretized": {{ {scheme} }} }}"#)).unwrap();
        assert_eq!(
            unnamed,
            ModelSpec::Discretized {
                scheme: upwind(),
                name: None
            }
        );
        let named: ModelSpec = serde_json::from_str(&format!(
            r#"{{ "Discretized": {{ {scheme}, "name": "advection" }} }}"#
        ))
        .unwrap();
        assert_eq!(
            named,
            ModelSpec::Discretized {
                scheme: upwind(),
                name: Some("advection".to_owned())
            }
        );
    }

    #[cfg(feature = "serde")]
    #[test]
    fn initial_states_read_from_their_values() {
        let read = |json: &str| serde_json::from_str::<InitialStateSpec>(json).unwrap();
        assert_eq!(
            read(r#"{ "Uniform": 0.5 }"#),
            InitialStateSpec::Uniform(0.5)
        );
        assert_eq!(
            read(r#"{ "Values": [0.0, 1.0] }"#),
            InitialStateSpec::Values(vec![0.0, 1.0])
        );
        assert_eq!(
            read(r#"{ "Dirac": { "amount": 3.0 } }"#),
            InitialStateSpec::Dirac { amount: 3.0 }
        );
    }
}
