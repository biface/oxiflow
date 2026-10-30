//! Calibration of the Phase 1 dispatch of `MultiDomainOrchestrator`
//! ([#141](https://github.com/biface/oxiflow/issues/141), [DA-048](https://github.com/biface/oxiflow/issues/140)): the domains of a
//! synchronised step computed one after the other, or through Rayon.
//!
//! The orchestrator compares the aggregate mesh volume of the scenario,
//! `Σ mesh.n_dof()` (nodes, whatever the number of fields per node and the
//! cost of a node), with its `parallel_threshold`. Every case is timed
//! twice, with the threshold forced to `usize::MAX` (`sequential`) and to
//! `1` (`parallel`), and the parameter of every case is that same sum, so
//! that a crossover reads directly against the threshold. Scenarios with
//! the same sum split into a different number of domains (for instance two
//! domains of 4,000 nodes and eight of 1,000) test the assumption behind an
//! aggregate threshold, which ignores the number of domains.
//!
//! Three scenarios, each from the shared models in `rsc/models/`:
//!
//! - [`langmuir_multi_domains`] -- K independent copies of the Langmuir
//!   column, started at `t = 0`, with explicit Euler and with Crank–Nicolson
//!   on the banded path (`sparse` feature), at Newton budgets of one and two
//!   iterations;
//! - [`lahar_multifield_domains`] -- K independent copies of the lahar
//!   validation scenario, started from its Chezy-drag-dominated state at
//!   `t = 10 s`, with explicit Euler;
//! - [`lahar_lake_wave_domains`] -- the coupled channel and lake, started
//!   once the bore runs in the lake (`t = 500 s`), with explicit Euler. The
//!   coupling exchange stays sequential: the gain read here is net of it.
//!
//! Every case runs [`MEASURED_STEPS`] steps per iteration, at a time step
//! scaled with the grid spacing so that the Courant number of explicit Euler
//! stays that of the reference run. The dispatch sites inside a domain
//! (model loops, calculators, implicit Jacobian) are either forced
//! sequential (`inner_seq`) or left at their defaults (`inner_default`),
//! which nests their dispatch inside the orchestrator's.
//!
//! A case is named `orchestrator/<scenario>/<integrator>/<inner>`, then
//! `K<k>/<sequential|parallel>`, with the aggregate mesh volume as
//! parameter: one report per scenario, integrator and inner dispatch, with
//! one curve per number of domains and orchestrator threshold, so that the
//! weight of the number of domains reads on the same chart as the
//! crossover. `<integrator>` is `euler` or `cn_banded_newton<N>`
//! (Crank–Nicolson on the banded path, Newton budget of N iterations);
//! `<inner>` is `inner_seq` or `inner_default`. Criterion cuts each
//! component of a name to 64 bytes for its directory under
//! `target/criterion/`, and tells the names it cut apart with suffixes
//! `_2`, `_3`…; the names are checked against that limit when they are
//! built.
//!
//! The orchestrator holds one calculator chain for all its domains, built
//! on one grid ([#210](https://github.com/biface/oxiflow/issues/210)): the K domains of a scenario share the same
//! grid, and the two grids of the lake scenario the same spacing.
//!
//! The measurements depend on the host: the number of Rayon threads is
//! printed before the first case that runs. Command:
//!
//! ```text
//! cargo bench --bench oxiflow_parallel --all-features -- orchestrator/
//! ```
//!
//! Each scenario is built the first time one of its cases runs, so a filter
//! that selects other benchmarks builds nothing. The starting states of the
//! two lahar scenarios are computed once per grid, then shared by every
//! case on that grid; the lake at its finest spacing takes most of that
//! preparation. Only exists when the crate is built with the `parallel`
//! feature.

use std::{cell::OnceCell, hint::black_box, time::Duration};

use criterion::{measurement::WallTime, BenchmarkGroup, BenchmarkId, Criterion};
use nalgebra::DMatrix;
use oxiflow::{
    context::{error::OxiflowError, quantity::PhysicalQuantity, value::ContextValue},
    mesh::Mesh,
    model::PrescribedInitialState,
    solver::{
        methods::{ForwardEulerSolver, SteppableSolver},
        orchestrator::{MultiDomainOrchestrator, OrchestratorConfig},
        Domain, DomainId, IntegratorKind, Scenario, Solver, SolverConfiguration, StepControl,
        TimeConfiguration,
    },
};

use crate::{
    lahar_outfall, lahar_source::ValidationRegime, lahar_validation, langmuir_multi_model,
};

// ── Settings shared by the three scenarios ───────────────────────────────

/// Steps timed per iteration.
pub const MEASURED_STEPS: usize = 10;

/// Numbers of independent domains of the Langmuir and lahar scenarios.
const DOMAIN_COUNTS: [usize; 3] = [2, 4, 8];

/// Dispatch sites inside a domain: forced sequential, or at their defaults.
const INTERNAL_MODES: [(&str, Option<usize>); 2] =
    [("inner_seq", Some(usize::MAX)), ("inner_default", None)];

/// Thresholds of the orchestrator: every step sequential, every step
/// through Rayon.
const ORCHESTRATOR_MODES: [(&str, usize); 2] = [("sequential", usize::MAX), ("parallel", 1)];

/// First component of every group name, and the filter of the command.
const GROUP_PREFIX: &str = "orchestrator";

/// Length in bytes past which Criterion cuts a group or function name for
/// its directory under `target/criterion/` (`MAX_DIRECTORY_NAME_LEN` in
/// its `report` module).
const CRITERION_NAME_MAX: usize = 64;

// ── Langmuir ──────────────────────────────────────────────────────────────

/// Nodes per domain of the Langmuir scenario.
const LANGMUIR_NODES: [usize; 3] = [250, 1_000, 4_000];

/// Integrator of the Langmuir domains.
#[derive(Clone, Copy)]
enum LangmuirIntegrator {
    Euler,
    /// Crank–Nicolson on the banded path, with this Newton budget.
    #[cfg(feature = "sparse")]
    CrankNicolsonBanded(usize),
}

impl LangmuirIntegrator {
    fn label(self) -> String {
        match self {
            Self::Euler => "euler".to_owned(),
            #[cfg(feature = "sparse")]
            Self::CrankNicolsonBanded(budget) => format!("cn_banded_newton{budget}"),
        }
    }

    /// One solver of this integrator, its Jacobian dispatch forced
    /// sequential when `internal` is `Some`.
    #[cfg_attr(not(feature = "sparse"), allow(unused_variables))]
    fn solver(self, n_species: usize, internal: Option<usize>) -> Box<dyn SteppableSolver> {
        match self {
            Self::Euler => Box::new(ForwardEulerSolver),
            #[cfg(feature = "sparse")]
            Self::CrankNicolsonBanded(budget) => {
                use oxiflow::solver::{
                    methods::crank_nicolson::CrankNicolsonSolver, sparse::FaerSparseSolver,
                };

                let solver = CrankNicolsonSolver::new()
                    .with_max_newton_iterations(budget)
                    .with_sparse_solver(Box::new(FaerSparseSolver))
                    .with_jacobian_bandwidth(2 * n_species - 1)
                    .with_sparse_threshold(0);
                match internal {
                    Some(threshold) => Box::new(solver.with_parallel_threshold(threshold)),
                    None => Box::new(solver),
                }
            }
        }
    }
}

/// K independent Langmuir columns, explicit Euler and Crank–Nicolson.
///
/// The columns start empty at `t = 0`, inside the injection window. The
/// cost of a step does not depend on the concentrations: the model solves
/// a linear system of fixed size per node, and Crank–Nicolson converges in
/// one Newton iteration on every case, measured from `t = 0` and from
/// states at `t = 100`, `300` and `500 s`, both at the scaled time step and
/// at the reference one. The two budgets therefore take the same path; the
/// cases at two iterations check that the budget changes nothing, and a
/// difference would show a second iteration.
pub fn langmuir_multi_domains(c: &mut Criterion) {
    let integrators = vec![
        LangmuirIntegrator::Euler,
        #[cfg(feature = "sparse")]
        LangmuirIntegrator::CrankNicolsonBanded(1),
        #[cfg(feature = "sparse")]
        LangmuirIntegrator::CrankNicolsonBanded(2),
    ];

    for integrator in integrators {
        for (internal_label, internal) in INTERNAL_MODES {
            let mut group =
                configured_group(c, "langmuir_multi", &integrator.label(), internal_label);
            for k in DOMAIN_COUNTS {
                for n_points in LANGMUIR_NODES {
                    let n_species = langmuir_multi_model::make_model(
                        langmuir_multi_model::column_spacing(n_points),
                    )
                    .n_species();
                    let total_nodes = k * langmuir_multi_model::column_grid(n_points)
                        .expect("Langmuir grid")
                        .n_dof();
                    time_both_dispatches(
                        &mut group,
                        k,
                        total_nodes,
                        || langmuir_scenario(n_points, k, internal).expect("Langmuir scenario"),
                        |threshold| {
                            independent_domains(
                                k,
                                threshold,
                                PhysicalQuantity::concentration(),
                                || integrator.solver(n_species, internal),
                            )
                        },
                    );
                }
            }
            group.finish();
        }
    }
}

/// `k` Langmuir columns of `n_points` nodes, and the configuration of
/// [`MEASURED_STEPS`] steps from `t = 0`.
///
/// The time step is the reference one, scaled with the node spacing: the
/// model is pure advection (backward gradients), so the Courant number
/// stays that of the reference run.
fn langmuir_scenario(
    n_points: usize,
    k: usize,
    internal: Option<usize>,
) -> Result<(Scenario, OrchestratorConfig), OxiflowError> {
    use langmuir_multi_model::{
        column_grid, column_spacing, gradient_calculators, make_model, N_POINTS, N_STEPS,
        TOTAL_TIME,
    };

    let dt = TOTAL_TIME / N_STEPS as f64 * (N_POINTS - 1) as f64 / (n_points - 1) as f64;
    let mut n_species = 0;
    let mut domains = Vec::with_capacity(k);
    for i in 0..k {
        let mut model = make_model(column_spacing(n_points));
        n_species = model.n_species();
        if let Some(threshold) = internal {
            model.parallel_threshold = threshold;
        }
        domains.push(Domain::new(
            domain_id(i),
            Box::new(model),
            Box::new(column_grid(n_points)?),
        ));
    }

    let mut config = OrchestratorConfig::new(measured_time(0.0, dt));
    for calculator in gradient_calculators(n_points, n_species, internal)? {
        config = config.with_calculator(calculator);
    }
    Ok((Scenario::multi(domains)?, config))
}

// ── Lahar, multi-field ────────────────────────────────────────────────────

/// Nodes per domain of the lahar scenario.
const LAHAR_NODES: [usize; 3] = [500, 1_000, 4_000];

/// Half-width of the lahar validation grid, as in its reference run.
const LAHAR_HALF_WIDTH: f64 = 50.0;

/// Time step of the lahar validation reference run, on
/// [`LAHAR_REFERENCE_NODES`] nodes.
const LAHAR_REFERENCE_DT: f64 = 5.0e-4;
const LAHAR_REFERENCE_NODES: usize = 1_001;

/// Start of the measured steps: the end of the validation run, in the
/// Chezy-drag-dominated regime.
const LAHAR_START: f64 = 10.0;

/// K independent copies of the lahar validation scenario, explicit Euler.
///
/// The friction law of the source term branches on the local regime, so
/// the domains start from the state of the validation run at
/// [`LAHAR_START`], computed once per grid with the internal dispatch at
/// its defaults and prescribed through `PrescribedInitialState`.
pub fn lahar_multifield_domains(c: &mut Criterion) {
    let starts: Vec<OnceCell<ContextValue>> = LAHAR_NODES.iter().map(|_| OnceCell::new()).collect();

    for (internal_label, internal) in INTERNAL_MODES {
        let mut group = configured_group(c, "lahar_multifield", "euler", internal_label);
        for k in DOMAIN_COUNTS {
            for (&n_points, start) in LAHAR_NODES.iter().zip(&starts) {
                let total_nodes = k * lahar_validation::grid(n_points, LAHAR_HALF_WIDTH)
                    .expect("lahar grid")
                    .n_dof();
                time_both_dispatches(
                    &mut group,
                    k,
                    total_nodes,
                    || {
                        let start = start.get_or_init(|| {
                            lahar_start_state(n_points).expect("lahar starting state")
                        });
                        lahar_scenario(n_points, k, start, internal).expect("lahar scenario")
                    },
                    |threshold| {
                        independent_domains(
                            k,
                            threshold,
                            PhysicalQuantity::custom("shallow_layer"),
                            || Box::new(ForwardEulerSolver),
                        )
                    },
                );
            }
        }
        group.finish();
    }
}

/// Time step of the lahar scenario on `n_points` nodes: the reference one,
/// scaled with the node spacing.
fn lahar_dt(n_points: usize) -> f64 {
    LAHAR_REFERENCE_DT * (LAHAR_REFERENCE_NODES - 1) as f64 / (n_points - 1) as f64
}

/// State of the validation run on `n_points` nodes at [`LAHAR_START`].
fn lahar_start_state(n_points: usize) -> Result<ContextValue, OxiflowError> {
    let dt = lahar_dt(n_points);
    let scenario = lahar_validation::scenario(
        ValidationRegime::ChezyDragDominated,
        n_points,
        LAHAR_HALF_WIDTH,
        None,
    )?;
    let mut time = TimeConfiguration::new(LAHAR_START, StepControl::Fixed { dt });
    time.save_every = Some((LAHAR_START / dt).round() as usize);
    let config = SolverConfiguration::new(time, IntegratorKind::Euler).with_calculator(Box::new(
        lahar_validation::bed_gradient(n_points, LAHAR_HALF_WIDTH, None)?,
    ));
    let result = ForwardEulerSolver.solve(&scenario, &config)?;
    final_state(result.states)
}

/// `k` lahar domains of `n_points` nodes started from `start`, and the
/// configuration of [`MEASURED_STEPS`] steps from [`LAHAR_START`].
fn lahar_scenario(
    n_points: usize,
    k: usize,
    start: &ContextValue,
    internal: Option<usize>,
) -> Result<(Scenario, OrchestratorConfig), OxiflowError> {
    let mut domains = Vec::with_capacity(k);
    for i in 0..k {
        let model = lahar_validation::model(
            ValidationRegime::ChezyDragDominated,
            n_points,
            LAHAR_HALF_WIDTH,
            internal,
        )?;
        domains.push(Domain::new(
            domain_id(i),
            Box::new(PrescribedInitialState::new(Box::new(model), start.clone())),
            Box::new(lahar_validation::grid(n_points, LAHAR_HALF_WIDTH)?),
        ));
    }

    let config = OrchestratorConfig::new(measured_time(LAHAR_START, lahar_dt(n_points)))
        .with_calculator(Box::new(lahar_validation::bed_gradient(
            n_points,
            LAHAR_HALF_WIDTH,
            internal,
        )?));
    Ok((Scenario::multi(domains)?.with_t_start(LAHAR_START), config))
}

// ── Lahar into a lake ─────────────────────────────────────────────────────

/// Grid spacings of the lake scenario [m].
const LAKE_SPACINGS: [f64; 5] = [10.0, 5.0, 2.5, 1.25, 0.625];

/// Time step per metre of spacing, that of the reference example
/// (`dt = 0.1 s` at `dx = 5 m`) [s/m].
const LAKE_DT_PER_DX: f64 = 0.02;

/// Start of the measured steps: the time from which the reference example
/// measures the bore running in the lake.
const LAKE_START: f64 = 500.0;

/// The coupled channel and lake, explicit Euler on both domains.
///
/// The domains start from the state of the reference run at [`LAKE_START`],
/// computed once per spacing with the internal dispatch at its defaults.
/// The parameter of each case is the node count of both grids; each spacing
/// is printed with its node count when its starting states are computed.
pub fn lahar_lake_wave_domains(c: &mut Criterion) {
    let starts: Vec<OnceCell<(DMatrix<f64>, DMatrix<f64>)>> =
        LAKE_SPACINGS.iter().map(|_| OnceCell::new()).collect();
    let total_nodes: Vec<usize> = LAKE_SPACINGS
        .iter()
        .map(|&dx| {
            lahar_outfall::channel_grid(dx)
                .expect("channel grid")
                .n_dof()
                + lahar_outfall::lake_grid(dx).expect("lake grid").n_dof()
        })
        .collect();

    for (internal_label, internal) in INTERNAL_MODES {
        let mut group = configured_group(c, "lahar_lake_wave", "euler", internal_label);
        for ((&dx, start), &nodes) in LAKE_SPACINGS.iter().zip(&starts).zip(&total_nodes) {
            time_both_dispatches(
                &mut group,
                2,
                nodes,
                || {
                    let (channel, lake) = start.get_or_init(|| {
                        eprintln!("{GROUP_PREFIX}/lahar_lake_wave: dx = {dx} m, {nodes} nodes");
                        lake_start_states(dx).expect("lake starting states")
                    });
                    lake_scenario(dx, channel, lake, internal).expect("lake scenario")
                },
                |threshold| lahar_outfall::orchestrator().with_parallel_threshold(threshold),
            );
        }
        group.finish();
    }
}

/// States of the channel and the lake of the reference run at spacing
/// `dx`, at [`LAKE_START`].
fn lake_start_states(dx: f64) -> Result<(DMatrix<f64>, DMatrix<f64>), OxiflowError> {
    let dt = LAKE_DT_PER_DX * dx;
    let (channel0, lake0) = lahar_outfall::initial_states(dx)?;
    let scenario = lahar_outfall::scenario(dx, &channel0, &lake0, None)?;
    let mut time = TimeConfiguration::new(LAKE_START, StepControl::Fixed { dt });
    time.save_every = Some((LAKE_START / dt).round() as usize);
    let config = OrchestratorConfig::new(time)
        .with_calculator(Box::new(lahar_outfall::bed_gradient(dx, None)?));
    let result = lahar_outfall::orchestrator().run(&scenario, &config)?;
    let last = result
        .states
        .last()
        .ok_or_else(|| missing_state("lahar_lake_wave"))?;
    Ok((
        lahar_outfall::field(last, &lahar_outfall::channel_id())?.clone(),
        lahar_outfall::field(last, &lahar_outfall::lake_id())?.clone(),
    ))
}

/// The coupled scenario at spacing `dx` started from `channel` and `lake`,
/// and the configuration of [`MEASURED_STEPS`] steps from [`LAKE_START`].
fn lake_scenario(
    dx: f64,
    channel: &DMatrix<f64>,
    lake: &DMatrix<f64>,
    internal: Option<usize>,
) -> Result<(Scenario, OrchestratorConfig), OxiflowError> {
    let scenario = lahar_outfall::scenario(dx, channel, lake, internal)?.with_t_start(LAKE_START);
    let config = OrchestratorConfig::new(measured_time(LAKE_START, LAKE_DT_PER_DX * dx))
        .with_calculator(Box::new(lahar_outfall::bed_gradient(dx, internal)?));
    Ok((scenario, config))
}

// ── Helpers ───────────────────────────────────────────────────────────────

/// The group of `scenario`, `integrator` and `inner` dispatch, sized for
/// runs of several seconds per iteration on the largest cases.
fn configured_group<'a>(
    c: &'a mut Criterion,
    scenario: &str,
    integrator: &str,
    inner: &str,
) -> BenchmarkGroup<'a, WallTime> {
    let name = checked_name(format!("{GROUP_PREFIX}/{scenario}/{integrator}/{inner}"));
    let mut group = c.benchmark_group(name);
    group
        .sample_size(10)
        .measurement_time(Duration::from_secs(3))
        .warm_up_time(Duration::from_secs(1));
    group
}

/// Times [`MEASURED_STEPS`] steps of the scenario of `k` domains built by
/// `prepare`, once per orchestrator threshold, with `total_nodes`, the
/// aggregate mesh volume of the scenario, as parameter. `prepare` runs the
/// first time one of the two cases runs, and the scenario is shared by
/// both; `orchestrator` builds the orchestrator for a given threshold.
fn time_both_dispatches(
    group: &mut BenchmarkGroup<'_, WallTime>,
    k: usize,
    total_nodes: usize,
    prepare: impl Fn() -> (Scenario, OrchestratorConfig),
    orchestrator: impl Fn(usize) -> MultiDomainOrchestrator,
) {
    let prepared = OnceCell::new();
    for (label, threshold) in ORCHESTRATOR_MODES {
        let orchestrator = orchestrator(threshold);
        group.bench_with_input(
            BenchmarkId::new(checked_name(format!("K{k}/{label}")), total_nodes),
            &total_nodes,
            |b, _| {
                let (scenario, config) = prepared.get_or_init(|| {
                    print_rayon_threads();
                    prepare()
                });
                assert_eq!(
                    scenario
                        .domains()
                        .iter()
                        .map(|d| d.mesh.n_dof())
                        .sum::<usize>(),
                    total_nodes,
                    "parameter and scenario disagree on the aggregate mesh volume"
                );
                b.iter(|| black_box(orchestrator.run(scenario, config).unwrap()))
            },
        );
    }
}

/// `name`, checked to fit [`CRITERION_NAME_MAX`]: a longer name would be
/// cut, and its reports filed under a suffixed directory.
fn checked_name(name: String) -> String {
    assert!(
        name.len() <= CRITERION_NAME_MAX,
        "benchmark name `{name}` is {} bytes long; Criterion cuts it to {CRITERION_NAME_MAX}",
        name.len()
    );
    name
}

/// Orchestrator of `k` independent domains ([`domain_id`]), each with a
/// solver from `solver`, at `threshold`.
fn independent_domains(
    k: usize,
    threshold: usize,
    quantity: PhysicalQuantity,
    solver: impl Fn() -> Box<dyn SteppableSolver>,
) -> MultiDomainOrchestrator {
    (0..k).fold(
        MultiDomainOrchestrator::new().with_parallel_threshold(threshold),
        |orchestrator, i| orchestrator.with_domain(domain_id(i), solver(), quantity.clone()),
    )
}

/// [`MEASURED_STEPS`] steps of `dt` from `t_start`, the last state saved.
fn measured_time(t_start: f64, dt: f64) -> TimeConfiguration {
    let mut time = TimeConfiguration::new(
        t_start + MEASURED_STEPS as f64 * dt,
        StepControl::Fixed { dt },
    );
    time.save_every = Some(MEASURED_STEPS);
    time
}

/// Identifier of the `i`-th independent domain.
fn domain_id(i: usize) -> DomainId {
    DomainId::new(format!("domain_{i}"))
}

/// Last state of a run that saved its final state.
fn final_state(mut states: Vec<ContextValue>) -> Result<ContextValue, OxiflowError> {
    states
        .pop()
        .ok_or_else(|| missing_state("lahar_multifield"))
}

fn missing_state(context: &'static str) -> OxiflowError {
    OxiflowError::PreconditionFailed {
        context,
        message: "the run saved no state".to_owned(),
    }
}

/// Number of Rayon threads of the host, printed before the first case
/// that runs.
fn print_rayon_threads() {
    static PRINTED: std::sync::Once = std::sync::Once::new();
    PRINTED.call_once(|| {
        eprintln!(
            "{GROUP_PREFIX}: Rayon threads = {}",
            rayon::current_num_threads()
        )
    });
}
