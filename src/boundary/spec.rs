//! # Module `boundary::spec`
//!
//! External, serialisable description of boundary conditions — the boundary
//! part of the WHAT-axis configuration
//! ([DT-027](https://github.com/biface/oxiflow/issues/75) amendment 1,
//! [#108](https://github.com/biface/oxiflow/issues/108)).
//!
//! [`BoundaryConditionSpec`] names one of the conditions of this module and
//! its parameters; [`TimeFunctionSpec`] describes a value that depends on
//! time. Both are consumed by [`DomainSpec`](crate::solver::DomainSpec).
//!
//! A function of time held as a Rust closure, as
//! [`Dirichlet::of_t`](super::Dirichlet::of_t) takes, has no serialisable
//! form: its type gives its signature, not the function. [`TimeFunctionSpec`]
//! describes the function instead (a constant, a parametric profile, or a
//! tabulated series), and the closure is rebuilt from that description when
//! the domain is built. Closed-form expressions of time are planned by
//! [DD-072](https://github.com/biface/oxiflow/issues/215).

use super::danckwerts::{DanckwertsInlet, DanckwertsOutlet};
use super::dirichlet::Dirichlet;
use super::injection::{InjectionInlet, TemporalInjection};
use super::BoundaryCondition;
use crate::context::calculators::tabulated::{Interpolation, TabulatedSeries};
use crate::context::error::OxiflowError;
use crate::mesh::BoundaryRegion;

/// A value that depends on time.
///
/// With the `serde` feature:
///
/// ```text
/// value: { Constant: 1.0 }
/// value: { Profile: { Gaussian: { center: 5.0, width: 1.0, peak_concentration: 2.0 } } }
/// value: { Tabulated: { points: [[0.0, 0.0], [10.0, 1.0]], interpolation: Linear } }
/// ```
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Deserialize))]
pub enum TimeFunctionSpec {
    /// The same value at every time.
    Constant(f64),
    /// A parametric profile of time; see [`TemporalInjection`].
    Profile(TemporalInjection),
    /// `(t, value)` pairs, interpolated between them and held at the end
    /// values outside their range, as
    /// [`ExternalTabulated`](crate::context::calculators::ExternalTabulated)
    /// does. At least two points, `t` strictly ascending, checked when the
    /// domain is built.
    Tabulated {
        /// `(t, value)` pairs, `t` strictly ascending.
        points: Vec<(f64, f64)>,
        /// Interpolation between the points.
        interpolation: Interpolation,
    },
}

/// Which boundary condition to build, and with what parameters.
///
/// With the `serde` feature, a condition is written as its variant name
/// around its parameters:
///
/// ```text
/// boundary_conditions:
///   - Dirichlet: { region: { Face: { axis: 0, side: Lower } }, value: { Constant: 1.0 } }
///   - DanckwertsOutlet: {}
/// ```
#[non_exhaustive]
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Deserialize))]
pub enum BoundaryConditionSpec {
    /// A [`Dirichlet`] condition holding `value` on `region`.
    Dirichlet {
        /// Boundary region the condition acts on.
        region: BoundaryRegion,
        /// Value held on the region.
        value: TimeFunctionSpec,
    },
    /// A [`DanckwertsInlet`], given by its own fields; `region` defaults to
    /// the lower face of axis 0.
    DanckwertsInlet(DanckwertsInlet),
    /// A [`DanckwertsOutlet`], given by its own fields; `region` defaults to
    /// the upper face of axis 0.
    DanckwertsOutlet(DanckwertsOutlet),
    /// An [`InjectionInlet`] following `profile`.
    InjectionInlet {
        /// Injection profile in time.
        profile: TemporalInjection,
    },
}

impl BoundaryConditionSpec {
    /// The condition this spec describes.
    ///
    /// # Errors
    ///
    /// `OxiflowError::ExternalData` if a tabulated value has fewer than two
    /// points or its `t` are not strictly ascending.
    pub(crate) fn into_condition(self) -> Result<Box<dyn BoundaryCondition>, OxiflowError> {
        Ok(match self {
            Self::Dirichlet { region, value } => Box::new(dirichlet(region, value)?),
            Self::DanckwertsInlet(inlet) => Box::new(inlet),
            Self::DanckwertsOutlet(outlet) => Box::new(outlet),
            Self::InjectionInlet { profile } => Box::new(InjectionInlet::new(profile)),
        })
    }
}

/// A [`Dirichlet`] condition holding `value` on `region`: a constant keeps
/// the constant form, any other value becomes a function of time.
fn dirichlet(region: BoundaryRegion, value: TimeFunctionSpec) -> Result<Dirichlet, OxiflowError> {
    Ok(match value {
        TimeFunctionSpec::Constant(value) => Dirichlet::constant(region, value),
        TimeFunctionSpec::Profile(profile) => Dirichlet::of_t(region, move |t| profile.evaluate(t)),
        TimeFunctionSpec::Tabulated {
            points,
            interpolation,
        } => {
            let series =
                TabulatedSeries::new("TimeFunctionSpec::Tabulated", points, interpolation)?;
            Dirichlet::of_t(region, move |t| series.value_at(t))
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mesh::FaceSide;

    const LOWER: BoundaryRegion = BoundaryRegion::Face {
        axis: 0,
        side: FaceSide::Lower,
    };

    #[test]
    fn a_tabulated_value_needs_ordered_points() {
        let spec = BoundaryConditionSpec::Dirichlet {
            region: LOWER,
            value: TimeFunctionSpec::Tabulated {
                points: vec![(1.0, 0.0), (0.0, 1.0)],
                interpolation: Interpolation::Linear,
            },
        };
        let Err(OxiflowError::ExternalData(message)) = spec.into_condition() else {
            panic!("an unordered series must be refused");
        };
        assert!(
            message.starts_with("TimeFunctionSpec::Tabulated"),
            "{message}"
        );
    }

    #[test]
    fn every_condition_builds() {
        let specs = [
            BoundaryConditionSpec::Dirichlet {
                region: LOWER,
                value: TimeFunctionSpec::Constant(1.0),
            },
            BoundaryConditionSpec::Dirichlet {
                region: LOWER,
                value: TimeFunctionSpec::Profile(TemporalInjection::SustainedStep {
                    start: 0.0,
                    concentration: 1.0,
                }),
            },
            BoundaryConditionSpec::DanckwertsInlet(DanckwertsInlet::new(
                1e-3,
                0.1,
                crate::context::variable::ContextVariable::Time,
            )),
            BoundaryConditionSpec::DanckwertsOutlet(DanckwertsOutlet::new()),
            BoundaryConditionSpec::InjectionInlet {
                profile: TemporalInjection::SustainedStep {
                    start: 0.0,
                    concentration: 1.0,
                },
            },
        ];
        for spec in specs {
            assert!(spec.into_condition().is_ok());
        }
    }

    #[cfg(feature = "serde")]
    #[test]
    fn time_functions_read_from_their_description() {
        let read = |json: &str| serde_json::from_str::<TimeFunctionSpec>(json).unwrap();
        assert_eq!(
            read(r#"{ "Constant": 2.0 }"#),
            TimeFunctionSpec::Constant(2.0)
        );
        assert_eq!(
            read(
                r#"{ "Profile": { "Gaussian": { "center": 5.0, "width": 1.0, "peak_concentration": 2.0 } } }"#
            ),
            TimeFunctionSpec::Profile(TemporalInjection::Gaussian {
                center: 5.0,
                width: 1.0,
                peak_concentration: 2.0
            })
        );
        assert_eq!(
            read(
                r#"{ "Tabulated": { "points": [[0.0, 0.0], [10.0, 1.0]], "interpolation": "Linear" } }"#
            ),
            TimeFunctionSpec::Tabulated {
                points: vec![(0.0, 0.0), (10.0, 1.0)],
                interpolation: Interpolation::Linear
            }
        );
    }

    #[cfg(feature = "serde")]
    #[test]
    fn danckwerts_conditions_read_their_own_fields_and_default_regions() {
        let read = |json: &str| serde_json::from_str::<BoundaryConditionSpec>(json).unwrap();
        let BoundaryConditionSpec::DanckwertsInlet(inlet) = read(
            r#"{ "DanckwertsInlet": { "dispersion": 1e-6, "velocity": 1e-3, "feed_variable": { "External": { "name": "feed" } } } }"#,
        ) else {
            panic!("expected an inlet");
        };
        assert_eq!(inlet.region, LOWER);
        assert_eq!(inlet.velocity, 1e-3);
        let BoundaryConditionSpec::DanckwertsOutlet(outlet) = read(r#"{ "DanckwertsOutlet": {} }"#)
        else {
            panic!("expected an outlet");
        };
        assert_eq!(
            outlet.region,
            BoundaryRegion::Face {
                axis: 0,
                side: FaceSide::Upper
            }
        );
    }

    #[cfg(feature = "serde")]
    #[test]
    fn a_dirichlet_condition_reads_its_region_and_value() {
        let json = r#"{ "Dirichlet": { "region": { "Face": { "axis": 0, "side": "Lower" } }, "value": { "Constant": 1.0 } } }"#;
        let BoundaryConditionSpec::Dirichlet { region, value } =
            serde_json::from_str::<BoundaryConditionSpec>(json).unwrap()
        else {
            panic!("expected a Dirichlet condition");
        };
        assert_eq!(region, LOWER);
        assert_eq!(value, TimeFunctionSpec::Constant(1.0));
    }
}
