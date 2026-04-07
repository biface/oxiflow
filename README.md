# oxiflow

[![CI](https://github.com/biface/oxiflow/actions/workflows/ci.yml/badge.svg)](https://github.com/biface/oxiflow/actions/workflows/ci.yml)
[![Coverage](https://codecov.io/gh/[USER]/oxiflow/branch/main/graph/badge.svg)](https://codecov.io/gh/biface/oxiflow)
[![Crates.io](https://img.shields.io/crates/v/oxiflow.svg)](https://crates.io/crates/oxiflow)
[![Docs.rs](https://docs.rs/oxiflow/badge.svg)](https://docs.rs/oxiflow)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](#license)

**Generic numerical engine for transport–reaction–diffusion problems.**

oxiflow provides the architectural building blocks to construct rigorous, maintainable and
performant PDE solvers in Rust — from simple 1D chromatography models to complex multi-physics
coupled systems. It is designed from the ground up to support structured grids (v1.0),
unstructured finite element meshes (v2.0), and a family of domain-specific frameworks (v3.0).

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

Each framework is an independent crate that depends on the engine and provides
domain-specific models, boundary conditions, nomenclature and declarative configuration.
Third parties can publish their own `oxiflow-*` frameworks on crates.io using the same
plugin API — the engine exposes stable, object-safe extension points for exactly this purpose.

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
  schemes; FD, FV and (in v2.0) FEM plug in without rewriting integrators
- **Abstract mesh** — `Mesh` trait keeps `PhysicalState` free of grid assumptions;
  `UniformGrid1D` in v1.0, unstructured triangular meshes in v2.0
- **Rich integrator library** — Forward Euler, RK4, Dormand–Prince 4/5, Backward Euler,
  Crank–Nicolson, BDF2/3, IMEX (Strang splitting)
- **Spatial schemes** — upwind/centred FD, WENO3/5, conservative FV, Lax–Wendroff,
  flux limiters (MinMod, Van Leer, Superbee)
- **Plugin-safe API** — all public traits are object-safe, enabling third-party niche
  frameworks to extend the engine without touching its internals (v2.0 — INV-4)
- **Optional parallelism** — Rayon-based, opt-in via `parallel` feature flag

---

## Quick Start

```toml
[dependencies]
oxiflow = "0.2"
# oxiflow-chrom = "3.0"    # chromatography framework (available from v3.0)
```

A minimal transport–diffusion model using the engine directly:

```rust
use oxiflow::prelude::*;

struct DiffusionModel { diffusivity: f64 }

impl PhysicalModel for DiffusionModel {
    fn compute(&self, state: &PhysicalState, ctx: &ComputeContext)
        -> Result<PhysicalState, OxiflowError>
    {
        let u     = ctx.vector(ContextVariable::Concentration)?;
        let lap_u = ctx.vector(ContextVariable::Laplacian)?;
        Ok(state.update(u + self.diffusivity * lap_u * ctx.time_step()?))
    }
}

impl RequiresContext for DiffusionModel {
    fn required_variables(&self) -> Vec<ContextVariable> {
        vec![
            ContextVariable::Concentration,
            ContextVariable::Laplacian,
            ContextVariable::TimeStep,
        ]
    }
}

fn main() -> Result<(), OxiflowError> {
    let result = Scenario::builder()
        .mesh(UniformGrid1D::new(100, 0.0, 1.0))
        .model(DiffusionModel { diffusivity: 1e-3 })
        .integrator(CrankNicolson::default())
        .time_span(0.0, 10.0)
        .dt(0.01)
        .build()?
        .solve()?;

    println!("{:?}", result.state());
    Ok(())
}
```

> **Note:** The API above reflects the v0.2 target — see [Development Status](#development-status).

---

## Covered Domains

The engine is domain-agnostic. Any problem of the canonical form
`∂u/∂t + ∇·F(u, ∇u) = S(u, x, t)` is a candidate:

| Domain             | Example                       | Target framework |
|--------------------|-------------------------------|------------------|
| Chromatography     | Multi-solute gradient elution | `oxiflow-chrom`  |
| Heat transfer      | 1D transient bar cooling      | `oxiflow-thermo` |
| Reaction–diffusion | Gray–Scott Turing patterns    | engine direct    |
| Fluid mechanics    | Burgers boundary layer        | engine direct    |
| Geomechanics       | Terzaghi consolidation        | engine direct    |
| Surface geophysics | Lahar–lake wave generation    | `oxiflow-geo`    |

---

## Design Invariants

Four constraints guarantee a non-breaking evolution from v1.0 to v3.0, and ensure that
third-party frameworks remain compatible across engine versions:

| Invariant | Description                                                               | Introduced |
|-----------|---------------------------------------------------------------------------|------------|
| **INV-1** | `Mesh` is abstract — `PhysicalState` carries no grid assumptions          | v0.2       |
| **INV-2** | `DiscreteOperator` is abstract — integrators are generic over the scheme  | v0.6       |
| **INV-3** | `CouplingOperator` supports distinct domains with moving interfaces       | v0.4       |
| **INV-4** | All public traits are object-safe — third-party crates can implement them | v2.0       |

---

## Development Status

| Milestone              | Version  | Status         | Theme                                           |
|------------------------|----------|----------------|-------------------------------------------------|
| J0 — Foundations       | v0.1     | ✅ Published    | Placeholder · CI · project structure            |
| J1 — Core Architecture | v0.2     | 🔄 In progress | ContextValue · OxiflowError · Mesh (INV-1)      |
| J2 — Complete Context  | v0.3     | ⏳ Planned      | Requiring BCs · topological ordering            |
| J3 — Multi-Component   | v0.4     | ⏳ Planned      | PhysicalQuantity · CouplingOperator (INV-3)     |
| J4 — Solvers           | v0.5–0.6 | ⏳ Planned      | Integrators · DiscreteOperator (INV-2)          |
| J5 — Performance       | v0.7     | ⏳ Planned      | Rayon · cache · benchmarks                      |
| J6 — Ecosystem v1.0    | v1.0     | ⏳ Planned      | 7 examples · FEM audit · stable API             |
| J7 — FEM               | v2.0     | 🔭 Horizon     | Unstructured meshes · ALE · INV-4 plugin-safe   |
| J8 — Frameworks        | v3.0     | 🔭 Horizon     | oxiflow-chrom · oxiflow-geo · CLI `oxiflow run` |

See [DEVELOPMENT.md](DEVELOPMENT.md) for the full architectural specification.

---

## Feature Flags

| Flag        | Description                                    | Available from |
|-------------|------------------------------------------------|----------------|
| *(default)* | Engine core, serial execution                  | v0.2           |
| `parallel`  | Rayon parallelism for independent calculators  | v0.7           |
| `serde`     | Serialisation of states and scenarios          | v0.7           |
| `hdf5`      | HDF5 import/export for tabulated external data | v0.7           |

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

Copyright 2026 [ton nom]

Licensed under the [Apache License, Version 2.0](LICENSE.txt).

You may use, distribute and modify this software freely, including for commercial purposes,
provided you retain the copyright notice and the `NOTICE` file in any redistribution.
The license includes a patent retaliation clause protecting the author — see the full
text for details.
