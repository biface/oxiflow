//! Shared measurement for the Jacobian-assembly dispatch diagnostics of the
//! implicit solvers (#199, DT-069), used by the `implicit_jacobian_dispatch`
//! benchmark of each model module.
//!
//! Declared only with the `parallel` feature: there is nothing to compare
//! against without it.

use std::hint::black_box;

use criterion::measurement::WallTime;
use criterion::{BenchmarkGroup, BenchmarkId};
use oxiflow::solver::{scenario::Scenario, Solver, SolverConfiguration};

/// Times one `solve` of `scenario` under `config` with the solver built by
/// `solver_with_threshold`, its Jacobian assembly forced sequential
/// (threshold `usize::MAX`), then parallel (threshold `1`). Reported as
/// `<path>/sequential` and `<path>/parallel` for the given `size`.
pub fn sequential_vs_parallel<S: Solver>(
    group: &mut BenchmarkGroup<'_, WallTime>,
    path: &str,
    size: usize,
    scenario: &Scenario,
    config: &SolverConfiguration,
    solver_with_threshold: impl Fn(usize) -> S,
) {
    for (mode, threshold) in [("sequential", usize::MAX), ("parallel", 1)] {
        let solver = solver_with_threshold(threshold);
        group.bench_with_input(
            BenchmarkId::new(format!("{path}/{mode}"), size),
            &size,
            |b, _| b.iter(|| black_box(solver.solve(scenario, config).unwrap())),
        );
    }
}
