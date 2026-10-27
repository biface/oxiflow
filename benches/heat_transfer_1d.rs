//! Criterion benchmark for the coefficient fields of the heat-transfer
//! reference example (#175). The graded conductivity comes from
//! `rsc/models/graded_conductivity.rs`, shared with the example.
//!
//! - [`coefficient_dispatch`] — dispatch of `CoefficientField`'s node
//!   evaluation, sequential vs. parallel forced through the public
//!   `with_parallel_threshold` API (`usize::MAX` / `1`), across problem
//!   sizes, for two conductivity profiles of different cost. Only exists
//!   when the crate is built with the `parallel` feature.

use std::hint::black_box;
use std::sync::Arc;

use criterion::{BenchmarkId, Criterion};
use nalgebra::DVector;
use oxiflow::{
    context::{
        calculators::CoefficientField, compute::ComputeContext, value::ContextValue,
        ContextCalculator,
    },
    mesh::{Mesh, UniformGrid1D},
};

#[path = "../rsc/models/graded_conductivity.rs"]
mod graded_conductivity;

use graded_conductivity::GradedConductivity;

/// Bar length of the example, m.
const LENGTH: f64 = 0.1;

/// The graded bar of the example: lambda(x) = 50 (1 + x / L) W/(m·K).
const GRADED: GradedConductivity = GradedConductivity {
    lambda0: 50.0,
    beta: 1.0,
    length: LENGTH,
};

/// A conductivity profile lambda(x), in W/(m·K).
type Profile = fn(f64) -> f64;

/// Two layers bonded at mid-length, steel (50 W/(m·K)) then copper
/// (400 W/(m·K)), with a smooth transition of width `L / 100`. One `tanh`
/// per node, where the graded profile costs one multiply-add: the two
/// profiles bracket the cost of a closed-form conductivity.
fn layered(x: f64) -> f64 {
    const STEEL: f64 = 50.0;
    const COPPER: f64 = 400.0;
    let s = ((x - 0.5 * LENGTH) / (0.01 * LENGTH)).tanh();
    STEEL + (COPPER - STEEL) * 0.5 * (1.0 + s)
}

/// Parallel-threshold crossover for `CoefficientField`'s node evaluation.
///
/// What is timed is the call the solver makes at every derivative
/// evaluation, `compute_shared`, on a c(x, t) field
/// (`CoefficientField::of_x_t`), which is where the threshold applies:
/// c(x) is evaluated once, at construction. The profiles do not depend on
/// t, but every iteration runs at a new time, so the field's cache never
/// answers and each call evaluates every node: state check, cache lookup,
/// evaluation, hand-over through an `Arc`.
///
/// The graded profile is the one of the example and the cheapest case, so
/// its crossover is an upper bound for a closed-form coefficient; the
/// layered profile shows how the crossover moves when the function costs
/// more. Both feed the dispatch charts of DA-048; the default threshold
/// (10 000 nodes) stays indicative.
pub fn coefficient_dispatch(c: &mut Criterion) {
    let mut group = c.benchmark_group("heat_transfer_1d_coefficient_dispatch");
    let profiles: [(&str, Profile); 2] = [("graded", |x| GRADED.value(x)), ("layered", layered)];

    for &n in &[
        1_000usize, 10_000, 100_000, 1_000_000, 3_000_000, 10_000_000,
    ] {
        let mesh: Arc<dyn Mesh> = Arc::new(UniformGrid1D::new(n, 0.0, LENGTH).unwrap());
        let state = ContextValue::ScalarField(DVector::zeros(n));

        for (profile, lambda) in profiles {
            for (dispatch, threshold) in [("sequential", usize::MAX), ("parallel", 1)] {
                let field =
                    CoefficientField::of_x_t(Arc::clone(&mesh), "conductivity", move |x, _| {
                        lambda(x[0])
                    })
                    .with_parallel_threshold(threshold);
                let mut time = 0.0;
                group.bench_with_input(
                    BenchmarkId::new(format!("{profile}/{dispatch}"), n),
                    &n,
                    |b, _| {
                        b.iter(|| {
                            time += 1.0;
                            let ctx = ComputeContext::new(time, 1.0);
                            black_box(field.compute_shared(&state, &ctx).unwrap())
                        })
                    },
                );
            }
        }
    }

    group.finish();
}
