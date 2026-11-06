//! # Integration test: domains built from their spec ([#108](https://github.com/biface/oxiflow/issues/108))
//!
//! A [`DomainSpec`] is the external form of a [`Domain`]
//! ([DT-027](https://github.com/biface/oxiflow/issues/75) amendment 1). These
//! tests check, through the public API only, that a domain built from its
//! spec runs exactly as the same domain built in code, that the conversion
//! reports the errors of its parts, and, with the `serde` feature, that a
//! domain reads from a configuration file.

use std::sync::Arc;

use nalgebra::DVector;
use oxiflow::{
    boundary::{BoundaryConditionSpec, Dirichlet, TimeFunctionSpec},
    context::{calculators::Interpolation, error::OxiflowError, value::ContextValue},
    mesh::{BoundaryRegion, FaceSide, MeshSpec, UniformGrid1D},
    model::{DiscretizedModel, InitialStateSpec, ModelSpec, PrescribedInitialState},
    operators::{fv::FVUpwindFlux, FluxBoundary, FluxBoundarySpec, FluxSchemeSpec},
    solver::{
        config::{IntegratorKind, StepControl, TimeConfiguration},
        methods::ForwardEulerSolver,
        Domain, DomainSpec, Scenario, Solver, SolverConfiguration,
    },
};

const N_POINTS: usize = 51;
const VELOCITY: f64 = 1.0;
const DIFFUSION: f64 = 1e-3;
/// Courant number 0.5 on the grid below, Fourier number 0.025.
const DT: f64 = 0.01;
const T_END: f64 = 0.5;

const INLET: BoundaryRegion = BoundaryRegion::Face {
    axis: 0,
    side: FaceSide::Lower,
};
const OUTLET: BoundaryRegion = BoundaryRegion::Face {
    axis: 0,
    side: FaceSide::Upper,
};

fn mesh_spec(n_points: usize) -> MeshSpec {
    MeshSpec::UniformGrid1D {
        n_points,
        x_start: 0.0,
        x_end: 1.0,
    }
}

fn upwind_spec() -> ModelSpec {
    ModelSpec::Discretized {
        scheme: FluxSchemeSpec::FVUpwind {
            velocity: VELOCITY,
            diffusion: DIFFUSION,
            boundary: FluxBoundarySpec::Truncation,
        },
        name: None,
    }
}

/// An advected ramp: the inlet value rises linearly from 0 to 1 over the
/// first unit of time, given as a tabulated series; the outlet is held at 0.
fn ramp_spec() -> DomainSpec {
    DomainSpec::new("column", mesh_spec(N_POINTS), upwind_spec())
        .with_boundary_conditions(vec![
            BoundaryConditionSpec::Dirichlet {
                region: INLET,
                value: TimeFunctionSpec::Tabulated {
                    points: vec![(0.0, 0.0), (1.0, 1.0)],
                    interpolation: Interpolation::Linear,
                },
            },
            BoundaryConditionSpec::Dirichlet {
                region: OUTLET,
                value: TimeFunctionSpec::Constant(0.0),
            },
        ])
        .with_initial_state(InitialStateSpec::Uniform(0.0))
}

/// The domain of [`ramp_spec`], built in code. Between the points (0, 0)
/// and (1, 1), linear interpolation computes `0 + (1 − 0)·(t − 0)/(1 − 0)`,
/// which is `t` exactly in floating point.
fn ramp_in_code() -> Result<Domain, OxiflowError> {
    let grid = Arc::new(UniformGrid1D::new(N_POINTS, 0.0, 1.0)?);
    let scheme = Arc::new(FVUpwindFlux::new(
        VELOCITY,
        DIFFUSION,
        FluxBoundary::Truncation,
    ));
    let model = PrescribedInitialState::new(
        Box::new(DiscretizedModel::new(
            scheme,
            Arc::clone(&grid),
            "fv_upwind",
        )),
        ContextValue::ScalarField(DVector::zeros(N_POINTS)),
    );
    Ok(
        Domain::new("column", Box::new(model), Box::new(grid)).with_boundary_conditions(vec![
            Box::new(Dirichlet::of_t(INLET, |t: f64| t.clamp(0.0, 1.0))),
            Box::new(Dirichlet::constant(OUTLET, 0.0)),
        ]),
    )
}

fn run(domain: Domain) -> Result<Vec<ContextValue>, OxiflowError> {
    let scenario = Scenario::multi(vec![domain])?;
    let config = SolverConfiguration::new(
        TimeConfiguration::new(T_END, StepControl::Fixed { dt: DT }),
        IntegratorKind::Euler,
    );
    Ok(ForwardEulerSolver.solve(&scenario, &config)?.states)
}

#[test]
fn a_domain_from_its_spec_runs_as_the_domain_built_in_code() -> Result<(), OxiflowError> {
    let from_spec = Domain::try_from(ramp_spec())?;
    assert_eq!(from_spec.model.name(), "fv_upwind");
    assert_eq!(from_spec.boundary_conditions.len(), 2);

    let expected = run(ramp_in_code()?)?;
    let states = run(from_spec)?;
    assert!(expected.len() > 1);
    assert_eq!(states, expected);
    // The ramp has entered the column: the comparison is not between zeros.
    let last = states.last().unwrap().as_scalar_field()?;
    assert!(last[1] > 0.0);
    Ok(())
}

#[test]
fn an_invalid_mesh_is_reported_by_the_mesh() {
    let spec = DomainSpec::new("column", mesh_spec(1), upwind_spec());
    assert!(matches!(
        Domain::try_from(spec),
        Err(OxiflowError::PreconditionFailed {
            context: "UniformGrid1D",
            ..
        })
    ));
}

#[test]
fn an_initial_state_of_the_wrong_length_is_refused() {
    let spec = DomainSpec::new("column", mesh_spec(5), upwind_spec())
        .with_initial_state(InitialStateSpec::Values(vec![0.0, 1.0, 2.0]));
    let Err(OxiflowError::InvalidDomain(message)) = Domain::try_from(spec) else {
        panic!("a mis-sized initial state must be refused");
    };
    assert!(message.contains("'column'"), "{message}");
    assert!(message.contains("3 rows"), "{message}");
}

#[test]
fn an_unordered_tabulated_value_is_refused() {
    let spec =
        DomainSpec::new("column", mesh_spec(5), upwind_spec()).with_boundary_conditions(vec![
            BoundaryConditionSpec::Dirichlet {
                region: INLET,
                value: TimeFunctionSpec::Tabulated {
                    points: vec![(0.0, 0.0), (0.0, 1.0)],
                    interpolation: Interpolation::Linear,
                },
            },
        ]);
    assert!(matches!(
        Domain::try_from(spec),
        Err(OxiflowError::ExternalData(_))
    ));
}

#[cfg(feature = "serde")]
#[test]
fn a_domain_reads_from_a_configuration_file() -> Result<(), OxiflowError> {
    let json = r#"{
        "id": "column",
        "mesh": { "UniformGrid1D": { "n_points": 200, "x_start": 0.0, "x_end": 0.25 } },
        "model": { "Discretized": { "scheme": {
            "FVUpwind": { "velocity": 1.0e-3, "diffusion": 1.0e-6, "boundary": "Truncation" }
        } } },
        "boundary_conditions": [
            { "DanckwertsInlet": { "dispersion": 1.0e-6, "velocity": 1.0e-3,
                                   "feed_variable": { "External": { "name": "feed" } } } },
            { "DanckwertsOutlet": {} }
        ],
        "initial_state": { "Uniform": 0.0 }
    }"#;
    let spec: DomainSpec = serde_json::from_str(json).expect("valid configuration");
    let domain = Domain::try_from(spec)?;
    assert_eq!(domain.id.to_string(), "column");
    assert_eq!(domain.mesh.n_dof(), 200);
    assert_eq!(domain.model.name(), "fv_upwind");
    assert_eq!(domain.boundary_conditions.len(), 2);
    Ok(())
}

#[cfg(feature = "serde")]
#[test]
fn optional_parts_of_a_domain_may_be_left_out() {
    let json = r#"{
        "id": "column",
        "mesh": { "UniformGrid1D": { "n_points": 5, "x_start": 0.0, "x_end": 1.0 } },
        "model": { "Discretized": { "scheme": {
            "WENO3": { "velocity": 1.0, "diffusion": 0.0, "boundary": "Periodic" }
        }, "name": "advection" } }
    }"#;
    let spec: DomainSpec = serde_json::from_str(json).expect("valid configuration");
    assert!(spec.boundary_conditions.is_empty());
    assert!(spec.initial_state.is_none());
    let domain = Domain::try_from(spec).unwrap();
    assert_eq!(domain.model.name(), "advection");
}
