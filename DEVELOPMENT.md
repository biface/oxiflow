# oxiflow — Development Program

This document is the architectural reference for oxiflow. It covers the design principles,
milestone specifications, design invariants, ecosystem strategy, and decision log that guide
all implementation work from v0.1 to v3.0.

> **Current version:** v0.0.1 (crates.io name reservation)
> **Active development:** v0.1 placeholder in preparation
> **Document version:** 2.0 — March 2026

---

## Table of Contents

1. [Vision & Principles](#1-vision--principles)
2. [Milestone Overview](#2-milestone-overview)
3. [J1 — Core Architecture (v0.2)](#3-j1--core-architecture-v02)
4. [J2 — Complete Context (v0.3)](#4-j2--complete-context-v03)
5. [J3 — Multi-Component (v0.4)](#5-j3--multi-component-v04)
6. [J4 — Solvers & Discretisation (v0.5–0.6)](#6-j4--solvers--discretisation-v05-06)
7. [J5 — Performance (v0.7)](#7-j5--performance-v07)
8. [J6 — Ecosystem v1.0](#8-j6--ecosystem-v10)
9. [FEM Compatibility — v2.0 Trajectory](#9-fem-compatibility--v20-trajectory)
10. [J8 — Niche Frameworks — v3.0](#10-j8--niche-frameworks--v30)
11. [Known Ecosystem Frameworks](#11-known-ecosystem-frameworks)
12. [Architectural Decision Log](#12-architectural-decision-log)
13. [Risk Register](#13-risk-register)
14. [Timeline](#14-timeline)

---

## 1. Vision & Principles

oxiflow is a generic Rust engine for numerical modelling of fields and fluxes — any problem
governed by conservation laws or field equations of the canonical form:

```
∂u/∂t + ∇·F(u, ∇u) = S(u, x, t)
```

where `u` is a field (concentration, velocity, temperature, pressure, magnetic field...),
`F` is a flux (advective + diffusive + dispersive), and `S` is a source or reaction term.

The engine serves as the foundation for a family of domain-specific **niche frameworks**
that add physical vocabulary, pre-implemented models, and declarative configuration for
specific scientific communities — chromatography, surface geophysics, heat transfer,
diffusive electromagnetism, and any domain a third party wishes to address.

### Non-negotiable design principles

- **Declarative before implicit** — model requirements are expressed in types, not runtime
  conventions
- **Generic ContextValue** — context variables cover scalars, vectors, matrices and fields,
  not just `f64`
- **Compile-time type safety** — configuration errors cause compile errors or immediate
  pre-solve failures, never silent panics
- **Zero overhead for simple cases** — a scalar model pays no cost from generics
- **Open extensibility** — adding a context type, solver, or domain requires no modification
  of the engine core
- **Strict separation of concerns** — model declares, calculator executes, solver
  orchestrates, Scenario validates
- **Anticipated FEM compatibility** — v1.0 abstractions are designed to not presuppose
  structured grids (INV-1/2/3)
- **Plugin-safe API** — all public traits are object-safe so third-party framework crates
  can implement them without accessing engine internals (INV-4, from v2.0)

### Positioning

oxiflow is not a full CFD framework (like OpenFOAM) and not a Python wrapper around LAPACK.
It is a numerical composition engine providing building blocks for rigorous, maintainable,
performant PDE solvers in any domain of continuous physics — and the foundation for a family
of niche frameworks that bring that power to specific scientific communities with minimal
boilerplate.

---

## 2. Milestone Overview

| Milestone | Version | Target | Theme |
|---|---|---|---|
| J0 — Foundations | v0.1 | Acquired | crates.io placeholder · CI · project structure |
| J1 — Core Architecture | v0.2 | M+2 | ContextValue · OxiflowError · Mesh (INV-1) |
| J2 — Complete Context | v0.3 | M+4 | Requiring BCs · topological ordering |
| J3 — Multi-Component | v0.4 | M+6 | PhysicalQuantity · CouplingOperator (INV-3) |
| J4a — Integrators | v0.5 | M+8 | Temporal integrators |
| J4b — Discretisation | v0.6 | M+10 | DiscreteOperator (INV-2) · FD/FV/WENO |
| J5 — Performance | v0.7 | M+13 | Rayon · cache · benchmarks |
| J6 — Ecosystem v1.0 | v1.0 | M+16 | 7 examples · FEM audit · stable API |
| J7 — FEM | v2.0 | M+24 | Unstructured meshes · ALE · INV-4 plugin-safe |
| J8 — Frameworks | v3.0 | M+32 | oxiflow-chrom · oxiflow-geo · CLI · third-party |

Each milestone is independently deliverable. J1 alone (v0.2) is a usable library for
chromatography modelling. Third-party framework development can begin as soon as v2.0
is published and INV-4 is in place.

---

## 3. J1 — Core Architecture (v0.2)

### 3.1 ContextValue

```rust
pub enum ContextValue {
    Scalar(f64),
    Vector(DVector<f64>),
    Matrix(DMatrix<f64>),
    Field2D(DMatrix<f64>),
    Boolean(bool),
}
```

### 3.2 OxiflowError

```rust
#[derive(Debug, thiserror::Error)]
pub enum OxiflowError {
    #[error("Missing calculator for variable: {0:?}")]
    MissingCalculator(ContextVariable),
    #[error("Computation failed for {variable:?}: {source}")]
    ComputationFailed { variable: ContextVariable, source: Box<dyn std::error::Error> },
    #[error("Circular dependency detected involving: {0:?}")]
    CircularDependency(ContextVariable),
    #[error("Type mismatch: expected {expected:?}, got {actual:?}")]
    TypeMismatch { expected: &'static str, actual: &'static str },
    #[error("Invalid domain configuration: {0}")]
    InvalidDomain(String),
    #[error("External data error: {0}")]
    ExternalData(String),
    #[error("Solver diverged at t={time:.4e}: {reason}")]
    SolverDivergence { time: f64, reason: String },
}
```

### 3.3 RequiresContext

```rust
pub trait RequiresContext {
    fn required_variables(&self) -> Vec<ContextVariable>;
    fn optional_variables(&self) -> Vec<ContextVariable> { vec![] }
    fn depends_on(&self) -> Vec<ContextVariable> { vec![] }
    fn priority(&self) -> u32 { 100 }
}
```

### 3.4 Mesh trait — INV-1

```rust
pub trait Mesh: Send + Sync {
    fn n_dof(&self) -> usize;
    fn coordinates(&self, i: usize) -> Vec<f64>;
    fn spatial_dimension(&self) -> usize;
    fn characteristic_length(&self) -> f64;
}
```

**Exit criterion:** a simple chromatography model works end-to-end with `ComputeContext`.
`UniformGrid1D` implements `Mesh`.

---

## 4. J2 — Complete Context (v0.3)

Requiring `BoundaryCondition` — closes the gap from the original architecture.
Topological ordering via Kahn's algorithm. Enriched built-in calculators (gradient, Laplacian,
quadrature, external tabulated data, HDF5 file reader).

Chromatography BC mappings:

| Chromatography BC | Mathematical type | Context needed |
|---|---|---|
| Simplified BC | Dirichlet | injection concentration profile |
| Danckwerts inlet | Robin | time + gradient |
| Danckwerts outlet | Neumann | gradient only |

---

## 5. J3 — Multi-Component (v0.4)

Indexed `PhysicalQuantity`. `MultiDomainState`. `CouplingOperator` inter-domain (INV-3).
Proto lahar–lake example on regular grids — the regression base for v2.0 FEM.

---

## 6. J4 — Solvers & Discretisation (v0.5–0.6)

Temporal integrators: Forward Euler, RK4, DoPri45, Backward Euler, Crank–Nicolson, BDF2/3,
IMEX (Strang splitting).

Abstract `DiscreteOperator` (INV-2) — integrators are generic over the scheme:

```rust
pub trait DiscreteOperator: Send + Sync {
    type MeshType: Mesh;
    fn apply(&self, field: &ContextValue, mesh: &Self::MeshType)
        -> Result<ContextValue, OxiflowError>;
}
```

Spatial schemes: FD (upwind/centred, 1st/2nd order), WENO3/5, conservative FV,
Lax–Wendroff, flux limiters (MinMod, Van Leer, Superbee), adaptive Péclet-based selection.

Linear algebra delegated to `nalgebra` (dense) and `faer` (sparse).

---

## 7. J5 — Performance (v0.7)

Rayon parallelism (opt-in `parallel` feature). Dirty-flag cache. Criterion benchmarks.
Feature flags: `parallel`, `serde`, `hdf5`.

**Exit criterion:** reference benchmark (1D diffusion, 1000 points, 10k steps) < 100 ms.

---

## 8. J6 — Ecosystem v1.0

Seven multi-domain examples: competitive chromatography, transient heat transfer,
Gray–Scott Turing patterns, Burgers boundary layer, Terzaghi consolidation,
magnetic diffusion (proto), lahar–lake coupled grids.

FEM invariant audit before publication (INV-1/2/3 verified across the full codebase).

API stability: SemVer strict, `cargo-semver-checks` in release pipeline, MSRV documented.

---

## 9. FEM Compatibility — v2.0 Trajectory

### 9.1 Motivating case

A rapid gravitational movement (lahar, landslide) entering a body of water and generating
a submersion wave. Requires unstructured meshing for irregular geometry and adaptive
refinement for the wave front — both impossible with finite differences.

| Component | Model | Challenge |
|---|---|---|
| Granular domain | Bingham + extended Saint-Venant | moving boundary · adaptive mesh |
| Fluid domain | Shallow Water equations | irregular bathymetry |
| Moving interface | ALE formulation | mass/momentum transfer |

### 9.2 INV-4 — Plugin-safe API

**Introduced at v2.0.** All public traits must be object-safe and fully accessible from
an external crate without depending on engine internals.

Verification: a dedicated integration test crate `oxiflow-test-plugin` (external, not
part of the workspace) implements all four public traits and is compiled in CI.

```rust
// This must compile from an external crate — never from pub(crate) types
use oxiflow::{PhysicalModel, BoundaryCondition, CouplingOperator, DiscreteOperator, Mesh};

struct ExternalModel;
impl PhysicalModel for ExternalModel { /* ... */ }
impl RequiresContext for ExternalModel { /* ... */ }
```

INV-4 is the prerequisite for v3.0. No niche framework can be developed before it is in
place and verified.

### 9.3 v2.0 scope

Unstructured mesh (Gmsh/Triangle readers, triangles 2D, tetrahedra 3D, h-adaptive
refinement). Function spaces (P1, P2 Lagrange, Raviart–Thomas, DG0). FEM assembler
(stiffness and mass matrices, Gauss quadrature, face integration). Sparse linear solvers
(`faer-sparse`, ILU/AMG preconditioners). ALE formulation for the lahar–lake example.

---

## 10. J8 — Niche Frameworks — v3.0

### 10.1 Architecture

The engine exposes a `PluginRegistry` that frameworks use to register their components:

```rust
// Engine (oxiflow)
pub struct PluginRegistry {
    models:      HashMap<&'static str, Box<dyn ModelFactory>>,
    calculators: HashMap<&'static str, Box<dyn CalculatorFactory>>,
    boundaries:  HashMap<&'static str, Box<dyn BCFactory>>,
}

// Framework (e.g. oxiflow-chrom)
pub fn register(registry: &mut PluginRegistry) {
    registry.register_model("langmuir",      LangmuirFactory);
    registry.register_model("thomas",         ThomasFactory);
    registry.register_model("sma",            SMAFactory);
    registry.register_bc("danckwerts",        DanckwertsFactory);
    registry.register_bc("simplified",        SimplifiedBCFactory);
    registry.register_calculator("dispersion",AxialDispersionFactory);
}
```

The engine has no knowledge of any framework. Frameworks depend on the engine.
This is a strict one-direction dependency.

### 10.2 Declarative configuration

The engine provides the generic TOML infrastructure. Each framework extends it with
domain-specific sections:

```toml
# Resolved by the engine
[solver]
integrator = "crank_nicolson"
dt = 0.01
t_end = 600.0

[mesh.column]
type = "uniform_1d"
length = 0.25
n_points = 500

# Resolved by oxiflow-chrom
[chromatography.column]
mode = "gradient_elution"

[[chromatography.solute]]
name = "protein_A"
isotherm = "langmuir"
H = 2.5
b = 0.08

[chromatography.boundary]
inlet  = "danckwerts"
outlet = "danckwerts"
```

### 10.3 CLI

```bash
oxiflow run problem.toml          # solve
oxiflow check problem.toml        # validate before solving
oxiflow list frameworks           # oxiflow-chrom, oxiflow-geo, ...
oxiflow list models --framework chrom
```

### 10.4 Planned first-party frameworks

| Crate | Domain | Key models |
|---|---|---|
| `oxiflow-chrom` | Chromatography | Langmuir, SMA, Thomas, gradient elution, Danckwerts BC |
| `oxiflow-geo` | Surface geophysics | Bingham Saint-Venant, Shallow Water, ALE interface |
| `oxiflow-thermo` | Heat transfer | Fourier flux, Robin BC, phase change |
| `oxiflow-em` | Diffusive electromagnetism | magnetic diffusion, eddy currents |

### 10.5 Third-party frameworks

Third parties are explicitly encouraged to publish `oxiflow-*` crates on crates.io.
Requirements for a third-party framework:

- Depends on `oxiflow = "2"` (or higher).
- Preserves the `NOTICE` file from the engine in any redistribution (Apache 2.0 requirement).
- Uses a compatible license (Apache 2.0 recommended; any OSI-approved license accepted).
- Uses the `oxiflow-` prefix on crates.io for discoverability.
- Opens a PR against the engine repository to be added to the
  [Known Ecosystem Frameworks](#11-known-ecosystem-frameworks) list below.

---

## 11. Known Ecosystem Frameworks

| Crate | Domain | Maintainer | Status |
|---|---|---|---|
| `oxiflow-chrom` | Chromatography | oxiflow core team | Planned v3.0 |
| `oxiflow-geo` | Surface geophysics | oxiflow core team | Planned v3.0 |
| `oxiflow-thermo` | Heat transfer | oxiflow core team | Planned v3.0 |
| `oxiflow-em` | Diffusive EM | oxiflow core team | Planned v3.0 |

*To add a framework to this list, open a PR modifying this table.*

---

## 12. Architectural Decision Log

| Decision | Choice | Rejected alternative | Milestone | Invariant |
|---|---|---|---|---|
| Calculator return type | `ContextValue` enum | `f64` scalar only | J1 | |
| Error type | `OxiflowError` enum | `String` | J1 | |
| Context access API | `ComputeContext` type-safe from v0.2 | Progressive migration | J1 | |
| Needs declaration | Separate `RequiresContext` trait | Method on `PhysicalModel` | J1 | |
| Spatial support | Abstract `Mesh` trait | `dx`/`nx` in `PhysicalState` | J1 | INV-1 |
| BC requirements | `RequiresContext` on `BoundaryCondition` | Manual aggregation | J2 | |
| Ordering | Hybrid topology + priority | Pure DAG or priority only | J2 | |
| Multi-component | Indexed `PhysicalQuantity` | Flat enum with breaking changes | J3 | |
| Multi-domain coupling | `CouplingOperator` with `DomainId` + `Interface` | Ad-hoc method | J3 | INV-3 |
| Spatial operators | Abstract `DiscreteOperator` parameterised by `Mesh` | FD hardcoded | J4 | INV-2 |
| Linear solvers | `faer`/`nalgebra` delegation | Custom implementation | J4 | |
| Parallelism | Rayon, opt-in feature flag | Mandatory or absent | J5 | |
| Caching | Dirty flag + temporal invalidation | Systematic recomputation | J5 | |
| API stability | SemVer + `cargo-semver-checks` + FEM audit | Informal convention | J6 | |
| Plugin architecture | Object-safe traits + `PluginRegistry` | Monolithic crate | J7 | INV-4 |
| Framework config | TOML + runtime registry | proc-macro DSL | J8 | |
| License | Apache 2.0 only | MIT or dual MIT/Apache | J0 | |

---

## 13. Risk Register

| ID | Risk | Probability | Mitigation |
|---|---|---|---|
| R1 | `ContextValue` generics too complex for users | Medium | Ergonomic helpers (`.as_scalar()?`); user testing at v0.2 |
| R2 | Silent dependency ordering bugs | Low | Exhaustive cycle detection tests; debug logging |
| R3 | `PhysicalQuantity` indexing too verbose | Medium | Idiomatic constructors (`::solute(k)`); UX feedback before v1.0 |
| R4 | Implicit solvers require heavy linear algebra | High | Delegate to `faer`/`nalgebra`; document limits |
| R5 | Rayon + potential `unsafe` | Low | Opt-in flag; ThreadSanitizer in CI; explicit `unsafe` review |
| R6 | Scope too ambitious — no milestone delivered | Medium | Each milestone independently deliverable |
| R7 | Breaking change forced before v1.0 | Low | Accepted pre-1.0 but documented |
| R8 | INV-1/2/3 silently violated during J1–J6 | Medium | Formal audit at J6; dedicated integration tests |
| R9 | ALE incompatible with CouplingOperator design | Low | Proto lahar–lake at J3 is the test bench |
| **R10** | **INV-4 violated — third-party frameworks break on engine update** | **Medium** | **`oxiflow-test-plugin` external crate in CI from v2.0; `cargo-semver-checks` in release pipeline** |
| **R11** | **Fragmentation — incompatible third-party frameworks** | **Low** | **INV-4 + stable public API is the only compatibility contract; framework authors are responsible for their own SemVer** |

---

## 14. Timeline

| Month | Milestone | Key objectives |
|---|---|---|
| M0 | v0.1 — Foundations | crates.io placeholder · CI · README · NOTICE |
| M+1–2 | v0.2 — J1 | ContextValue · OxiflowError · Mesh (INV-1) |
| M+3–4 | v0.3 — J2 | Requiring BCs · topology · built-in calculators |
| M+5–6 | v0.4 — J3 | PhysicalQuantity · CouplingOperator (INV-3) · proto lahar–lake |
| M+7–8 | v0.5 — J4a | Temporal integrators |
| M+9–10 | v0.6 — J4b | DiscreteOperator (INV-2) · FD/FV · WENO |
| M+11–13 | v0.7 — J5 | Rayon · cache · benchmarks |
| M+14–15 | v0.9 — RC | 7 examples · API freeze · FEM audit |
| M+16 | v1.0 | Stable release · official publication |
| M+17–24 | v2.0 — J7 | Unstructured mesh · FEM assembler · ALE · INV-4 |
| M+25–32 | v3.0 — J8 | oxiflow-chrom · oxiflow-geo · oxiflow-thermo · CLI |
| M+32+ | Third-party | Community frameworks on crates.io |

---

*oxiflow Development Program v2.0 · March 2026 · Living document — updated at each milestone*
