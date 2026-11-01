# oxiflow

[![CI](https://github.com/biface/oxiflow/actions/workflows/ci.yml/badge.svg)](https://github.com/biface/oxiflow/actions/workflows/ci.yml)
[![Coverage](https://codecov.io/gh/biface/oxiflow/graph/badge.svg?token=M6807F5BDQ)](https://codecov.io/gh/biface/oxiflow)
[![Crates.io](https://img.shields.io/crates/v/oxiflow.svg)](https://crates.io/crates/oxiflow)
[![Docs.rs](https://docs.rs/oxiflow/badge.svg)](https://docs.rs/oxiflow)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](#license)

**Generic numerical engine for transport–reaction–diffusion problems.**

oxiflow provides the architectural building blocks to construct rigorous, maintainable and
performant PDE solvers in Rust — from simple 1D chromatography models to complex multi-physics
coupled systems. The architecture will support:

- structured grids — 1D today (`UniformGrid1D`), nD from v0.11.5
- unstructured finite element meshes from v0.11.0, the finite element and structured-grid
  paths being developed side by side up to v0.15.x ([DQ-047](https://github.com/biface/oxiflow/issues/132))
- a family of niche frameworks (v0.30.0)

---

## Architecture

oxiflow is structured as an **engine + niche frameworks**:

```
oxiflow              (engine — fields, fluxes, meshes, couplings, solvers)
├── oxiflow-chrom    (chromatography framework)
├── oxiflow-geo      (surface geophysics framework)
├── oxiflow-thermo   (heat transfer framework)
├── oxiflow-em       (diffusive electromagnetism framework)
└── ...              (third-party frameworks on crates.io)
```

Each framework is an independent crate that will adapt the core engine to the domain-specific
physical models, boundary conditions, nomenclature and declarative configuration of its field.
This work aims to show third parties how to use the core `oxiflow` library, so they can then
publish their own `oxiflow-*` frameworks on crates.io using the same plugin API — the engine
exposes stable, object-safe extension points for exactly this purpose.

---

## Engine Features

- **Generic context system** — `ContextValue` supports scalars, vectors, matrices and 2D
  fields; no more `f64`-only bottlenecks
- **Type-safe dependency declaration** — models declare what they need via `RequiresContext`;
  missing calculators are caught before the solver runs
- **Multi-component systems** — indexed `PhysicalQuantity` handles N-solute problems,
  competitive adsorption, thermal coupling and more
- **Multi-domain coupling** — `CouplingOperator` connects distinct physical domains across
  moving interfaces (gravitational flows, fluid–solid interaction, ...)
- **Abstract spatial operators** — `DiscreteOperator` decouples solvers from discretisation
  schemes; FD, FV and FEM (landing progressively from v0.11.0 to v0.15.0) plug in without
  rewriting integrators
- **Abstract mesh** — `Mesh` trait keeps `PhysicalState` free of grid assumptions;
  `UniformGrid1D` today, nD structured grids in v0.11.5, unstructured 2D/3D meshes in v0.11.0
- **Rich integrator library** — Forward Euler, RK4, Dormand–Prince 4/5, Backward Euler,
  Crank–Nicolson, BDF2, IMEX (Strang splitting) — with iterated Newton correction ([DD-044](https://github.com/biface/oxiflow/issues/109),
  v0.7.0) for the implicit methods, configurable per solver (convergence criterion,
  Jacobian refresh strategy, iteration budget); higher-order BDF formulas (BDF3 to BDF6) are
  considered if a model needs them
- **Spatial schemes** — upwind/centred FD, WENO3/5, conservative FV, Lax–Wendroff,
  flux limiters (MinMod, Van Leer, Superbee)
- **Plugin-safe API** — all public traits are object-safe, enabling third-party niche
  frameworks to extend the engine without touching its internals (v0.30.0 — INV-4)
- **Computation optimisation** — opt-in Rayon parallelism (`parallel` feature) and
  GPU-readiness via `wgpu` (`gpu` feature, since v0.8.0)

---

## Quick Start

```toml
[dependencies]
oxiflow = "0.10"
# oxiflow-chrom          # chromatography framework (planned for v0.30.0)
```

A minimal diffusion model using the engine directly: the model (WHAT) declares the
Laplacian it needs, the solver configuration (HOW) supplies the calculator that computes it.

```rust
use std::sync::Arc;

use nalgebra::DVector;
use oxiflow::{
    boundary::{BoundaryCondition, Dirichlet},
    context::{
        calculators::FDLaplacianCalculator, compute::ComputeContext, error::OxiflowError,
        value::ContextValue, variable::ContextVariable,
    },
    mesh::{BoundaryRegion, FaceSide, Mesh, UniformGrid1D},
    model::traits::{PhysicalModel, RequiresContext},
    solver::{
        config::{IntegratorKind, StepControl, TimeConfiguration},
        methods::CrankNicolsonSolver,
        scenario::{Domain, Scenario},
        Solver, SolverConfiguration,
    },
};

/// du/dt = D d²u/dx², the Laplacian being supplied by a context calculator.
struct Diffusion {
    diffusivity: f64,
}

fn laplacian() -> ContextVariable {
    ContextVariable::External {
        name: "laplacian".into(),
    }
}

impl RequiresContext for Diffusion {
    fn required_variables(&self) -> Vec<ContextVariable> {
        vec![laplacian()]
    }
}

impl PhysicalModel for Diffusion {
    fn compute_physics(
        &self,
        _state: &ContextValue,
        ctx: &ComputeContext,
    ) -> Result<ContextValue, OxiflowError> {
        let lap_u = ctx.external(laplacian())?.as_scalar_field()?;
        Ok(ContextValue::ScalarField(lap_u * self.diffusivity))
    }

    fn initial_state(&self, mesh: &dyn Mesh) -> ContextValue {
        ContextValue::ScalarField(DVector::zeros(mesh.n_dof()))
    }

    fn name(&self) -> &str {
        "diffusion"
    }
}

fn main() -> Result<(), OxiflowError> {
    // WHAT: a rod of unit length, held at u = 1 on the left and u = 0 on the right.
    let ends: Vec<Box<dyn BoundaryCondition>> = vec![
        Box::new(Dirichlet::constant(
            BoundaryRegion::Face {
                axis: 0,
                side: FaceSide::Lower,
            },
            1.0,
        )),
        Box::new(Dirichlet::constant(
            BoundaryRegion::Face {
                axis: 0,
                side: FaceSide::Upper,
            },
            0.0,
        )),
    ];
    let rod = Domain::new(
        "rod",
        Box::new(Diffusion { diffusivity: 1e-3 }),
        Box::new(UniformGrid1D::new(101, 0.0, 1.0)?),
    )
    .with_boundary_conditions(ends);
    let scenario = Scenario::multi(vec![rod])?;

    // HOW: Crank–Nicolson, fixed step, Laplacian by finite differences.
    let mesh: Arc<dyn Mesh> = Arc::new(UniformGrid1D::new(101, 0.0, 1.0)?);
    let config = SolverConfiguration::new(
        TimeConfiguration::new(10.0, StepControl::Fixed { dt: 0.01 }),
        IntegratorKind::CrankNicolson,
    )
    .with_calculator(Box::new(FDLaplacianCalculator::new(mesh, laplacian())));

    let result = CrankNicolsonSolver::new().solve(&scenario, &config)?;
    if let Some(last) = result.states.last() {
        let u = last.as_scalar_field()?;
        println!("u(x = 0.5, t = 10) = {:.3e}", u[50]);
    }
    Ok(())
}
```

It prints `u(x = 0.5, t = 10) = 4.200e-4`; the solution for a semi-infinite rod,
`erfc(x / (2 sqrt(D t)))`, gives 4.07e-4 at that point.

---

## Covered Domains

The engine is domain-agnostic. Any problem of the canonical form
`∂u/∂t + ∇·F(u, ∇u) = S(u, x, t)` is a candidate.
See the [oxiflow wiki](https://github.com/biface/oxiflow/wiki) for introductory notes on each domain.

| Domain             | Example                                               | Target framework |
|--------------------|-------------------------------------------------------|------------------|
| Chromatography     | Binary frontal analysis with competitive adsorption   | `oxiflow-chrom`  |
| Heat transfer      | 1D transient conduction in a bar (uniform and graded) | `oxiflow-thermo` |
| Reaction–diffusion | Gray–Scott in 1D: Turing instability, pulse splitting | engine direct    |
| Fluid mechanics    | 1D viscous boundary layer at a wall (Burgers)         | engine direct    |
| Geomechanics       | 1D consolidation of a clay layer (Terzaghi)           | engine direct    |
| Electromagnetism   | 1D magnetic diffusion into a conductor (proto)        | `oxiflow-em`     |
| Surface geophysics | Lahar–lake wave generation                            | `oxiflow-geo`    |

---

## Design Invariants

Four constraints guarantee that finite element support (from v0.11.0) and third-party
frameworks (v0.30.0) arrive without breaking existing code, and that those frameworks
remain compatible across engine versions:

| Invariant | Description                                                               | Introduced |
|-----------|---------------------------------------------------------------------------|------------|
| **INV-1** | `Mesh` is abstract — no public API exposes `dx`, `nx` or raw indices      | v0.1       |
| **INV-2** | `DiscreteOperator` is abstract — integrators are generic over the scheme  | v0.5       |
| **INV-3** | Inter-domain coupling only through `CouplingOperator`, never in a solver  | v0.3       |
| **INV-4** | All public traits are object-safe — third-party crates can implement them | v0.30      |

---

## Development Status

| Milestone                                              | Version                              | Status        | Date                    | Theme                                                                                                                                                                                                                                                                                  |
|--------------------------------------------------------|--------------------------------------|---------------|-------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| J0 — Foundations                                       | v0.0.1 / v0.0.5                      | ✅ Published   | 2026-03-20              | Placeholder · CI · project structure                                                                                                                                                                                                                                                   |
| J1 — Core Architecture                                 | v0.1.0                               | ✅ Published   | 2026-04-12              | ContextValue · OxiflowError · Mesh (INV-1)                                                                                                                                                                                                                                             |
| J2 — Complete Context                                  | v0.2.0                               | ✅ Published   | 2026-05-02              | Requiring BCs · topological ordering · calculators                                                                                                                                                                                                                                     |
| J3 — Multi-Component                                   | v0.3.0                               | ✅ Published   | 2026-06-03              | PhysicalQuantity · MultiDomainState · CouplingOperator (INV-3)                                                                                                                                                                                                                         |
| J4 — Integrators                                       | v0.4.0                               | ✅ Published   | 2026-06-25              | Euler, RK4, DoPri45, Backward Euler, Crank–Nicolson, BDF2, IMEX                                                                                                                                                                                                                        |
| J5 — Discretisation                                    | v0.5.0                               | ✅ Published   | 2026-07-17              | DiscreteOperator (INV-2) · FD/FV · WENO3/5 · flux limiters + adaptive selection                                                                                                                                                                                                        |
| J6 — Sparse Linear Algebra & Persistence               | v0.6.0                               | ✅ Published   | 2026-07-21              | faer sparse solver · SimulationSnapshot (checkpoint/on_divergence) · HDF5 loading · VTK export · IntegratorSpec                                                                                                                                                                        |
| J7 — Nonlinear Time Integration                        | v0.7.0                               | ✅ Published   | 2026-07-30              | Iterated Newton for implicit integrators ([DD-044](https://github.com/biface/oxiflow/issues/109)) · `IntegratorSpec` extended to BE/CN/BDF2                                                                                                                                            |
| J8 — Computational Optimisation                        | v0.8.0                               | ✅ Published   | 2026-08-03              | GPU-readiness ([DA-026](https://github.com/biface/oxiflow/issues/73), [DT-045](https://github.com/biface/oxiflow/issues/128)) · `gpu` feature (`wgpu`) · MSRV 1.84                                                                                                                     |
| J9 — Parallelism & Benchmarking                        | v0.9.0                               | ✅ Published   | 2026-09-16              | Rayon dispatch ([DT-014](https://github.com/biface/oxiflow/issues/14)/[DA-048](https://github.com/biface/oxiflow/issues/140)) · Criterion benchmark suite (`oxiflow_parallel`) · lahar multi-field model ([DQ-032](https://github.com/biface/oxiflow/issues/85)) · SVG reference plots |
| J10 — Stable Ecosystem                                 | v0.10.0                              | 🚧 In progress | 2026-11-30              | Seven reference examples · `(x, t)` coefficients (`CoefficientField`, [DD-066](https://github.com/biface/oxiflow/issues/181)) · FEM invariant audit · strict SemVer, `cargo-semver-checks` ([DQ-016](https://github.com/biface/oxiflow/issues/16))                                     |
| J11 — nD Mesh Foundations                              | v0.11.0 (FEM) / v0.11.5 (structured) | ⏳ Planned     | 2027-01-14 / 2027-02-28 | `FiniteElementMesh`, unstructured 2D/3D meshes ([DA-049](https://github.com/biface/oxiflow/issues/150)) · `UniformGrid1D` generalised to nD ([DD-051](https://github.com/biface/oxiflow/issues/152))                                                                                   |
| J12 — Velocity-Pressure Discretisation                 | v0.12.0 / v0.12.5                    | ⏳ Planned     | 2027-04-14 / 2027-05-29 | FEM (Gmsh import, [DT-028](https://github.com/biface/oxiflow/issues/76)) or MAC/collocated + Rhie–Chow (structured)                                                                                                                                                                    |
| J13 — SaddlePointSolver                                | v0.13.0                              | ⏳ Planned     | 2027-07-13              | `SaddlePointSolver`, shared by both paths                                                                                                                                                                                                                                              |
| J14 — Differential-Algebraic Equations & Navier–Stokes | v0.14.0                              | ⏳ Planned     | 2027-08-27              | DAE time integration · Navier–Stokes as a `PhysicalModel`, shared                                                                                                                                                                                                                      |
| J15 — End-to-End NS Wiring                             | v0.15.0 / v0.15.5                    | ⏳ Planned     | 2027-10-11 / 2027-11-25 | Taylor–Hood/MINI (FEM) or full structured pipeline                                                                                                                                                                                                                                     |
| J16 — Preconditioned Sparse Solvers                    | v0.16.0                              | ⏳ Planned     | 2028-01-09              | ILU/AMG preconditioners, ahead of the comparison that needs them at fine resolution ([DQ-047](https://github.com/biface/oxiflow/issues/132))                                                                                                                                           |
| J17 — FEM vs Structured Comparison                     | v0.17.0                              | ⏳ Planned     | 2028-02-23              | Lid-driven cavity, Poiseuille, Ghia et al. 1982                                                                                                                                                                                                                                        |
| J18 — Arbitrary Lagrangian–Eulerian                    | v0.18.0                              | ⏳ Planned     | 2028-04-08              | ALE formulation                                                                                                                                                                                                                                                                        |
| J19 — Spectral Methods                                 | v0.19.0                              | ⏳ Planned     | 2028-05-23              | `Discretization` common ancestor, `Mesh`/`SpectralBasis` siblings ([DA-024](https://github.com/biface/oxiflow/issues/63))                                                                                                                                                              |
| J20 — Stabilisation                                    | v0.20.0                              | 🔭 Horizon     | 2028-07-07              | Checkpoint closing the first cycle ([DQ-063](https://github.com/biface/oxiflow/issues/165)) · 1.0.0 is tied to no milestone ([DQ-016](https://github.com/biface/oxiflow/issues/16))                                                                                                    |
| J30 — Niche Frameworks                                 | v0.30.0                              | 🔭 Horizon     | 2029-12-31              | oxiflow-chrom · oxiflow-geo · oxiflow-thermo · oxiflow-em · CLI `oxiflow run` · third-party plugins ([DA-017](https://github.com/biface/oxiflow/issues/17))                                                                                                                            |
| Future Phase                                           | v0.40.0                              | 🔭 Horizon     | 2030-12-31              | Scope not yet defined                                                                                                                                                                                                                                                                  |

Date: publication date of a published milestone, due date of the others (GitHub milestones).

The INV-GPU-1 to INV-GPU-5 sub-invariants ([DA-026](https://github.com/biface/oxiflow/issues/73)) have governed GPU-readiness of numerical
data structures since v0.3.0; the `gpu` feature itself (backend selection, adapter
acquisition, CPU↔GPU boundary conversion) landed in v0.8.0 ([DT-045](https://github.com/biface/oxiflow/issues/128)) with no breaking API
changes to any type designed against those invariants.

See [DEVELOPMENT.md](DEVELOPMENT.md) for the full architectural specification.

---

## Feature Flags

| Flag        | Description                                          | Available from |
|-------------|------------------------------------------------------|----------------|
| *(default)* | Engine core, serial execution                        | v0.2           |
| `serde`     | Serialisation of states and scenarios                | v0.6           |
| `hdf5`      | HDF5 import/export for tabulated external data       | v0.6           |
| `vtk`       | VTK (`.vtu`) export of `SimulationResult`            | v0.6           |
| `sparse`    | `faer` sparse linear solver for implicit integrators | v0.6           |
| `parallel`  | Rayon parallelism for independent calculators        | v0.9           |
| `gpu`       | GPU acceleration via `wgpu` (Vulkan/Metal/DX12)      | v0.8           |

---

## Contributing

Contributions are welcome at every milestone — bug fixes, new features (after discussion),
documentation, benchmarks, tests, and **niche framework development**.

**Building a framework on oxiflow?** Open a GitHub Discussion before publishing so it can
be listed in the official ecosystem documentation. Third-party `oxiflow-*` crates are
explicitly encouraged — the engine plugin API (INV-4) is designed for exactly this.

Read [CONTRIBUTING.md](CONTRIBUTING.md) before submitting a pull request.
Coverage is tracked on [Codecov](https://codecov.io/gh/[USER]/oxiflow):
target ≥ 85% overall, ≥ 90% on INV components.

---

## License

Copyright 2026 [biface](https://github.com/biface)

Licensed under the [Apache License, Version 2.0](LICENSE.txt).

You may use, distribute and modify this software freely, including for commercial purposes,
provided you retain the copyright notice and the `NOTICE` file in any redistribution.
The license includes a patent retaliation clause protecting the author — see the full
text for details.

The [oxiflow wiki](https://github.com/biface/oxiflow/wiki) is not covered by this license:
its text and figures are licensed under
[CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/). Any reuse must credit the
source, and commercial use is not permitted.
