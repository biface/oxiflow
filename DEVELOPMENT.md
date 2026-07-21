# oxiflow — Development Program

This document is the architectural reference for oxiflow. It covers the design principles,
milestone specifications, design invariants, ecosystem strategy, and decision log that guide
all implementation work from v0.1 to v3.0.

> **Current version:** v0.6.0 — Sparse Algebra & Persistence (closed)
> **Active development:** v0.7.0 — Nonlinear Time Integration (J7), not yet started
> **Document version:** 2.3 — July 2026

---

## Table of Contents

1. [Vision & Principles](#1-vision--principles)
2. [Milestone Overview](#2-milestone-overview)
3. [J1 — Core Architecture (v0.1)](#3-j1--core-architecture-v01)
4. [J2 — Complete Context (v0.2)](#4-j2--complete-context-v02)
5. [J3 — Multi-Component (v0.3)](#5-j3--multi-component-v03)
6. [J4a — Integrators (v0.4)](#6-j4a--integrators-v04)
7. [J5 — Discretisation (v0.5)](#7-j5--discretisation-v05)
8. [J6 — Sparse Algebra & Persistence (v0.6)](#8-j6--sparse-algebra--persistence-v06)
9. [J7 — Nonlinear Time Integration (v0.7)](#9-j7--nonlinear-time-integration-v07)
10. [J8 — Computational Optimisation (v0.8)](#10-j8--computational-optimisation-v08)
11. [J9 — Parallelism & Benchmarking (v0.9)](#11-j9--parallelism--benchmarking-v09)
12. [J10 — Stable Ecosystem (v1.0)](#12-j10--stable-ecosystem-v10)
13. [FEM Compatibility — v2.0 Trajectory (J20)](#13-fem-compatibility--v20-trajectory-j20)
14. [J30 — Niche Frameworks (v3.0)](#14-j30--niche-frameworks-v30)
15. [Known Ecosystem Frameworks](#15-known-ecosystem-frameworks)
16. [Architectural Decision Log](#16-architectural-decision-log)
17. [Risk Register](#17-risk-register)
18. [Timeline](#18-timeline)

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

| Milestone | Version | State | Theme |
|---|---|---|---|
| J0 — Foundations | v0.0.1–v0.0.5 | ✅ Achieved | crates.io placeholder · CI · project structure |
| J1 — Core Architecture | v0.1.0 | ✅ Achieved | ContextValue · OxiflowError · Mesh (INV-1) |
| J2 — Complete Context | v0.2.0 | ✅ Achieved | Requiring BCs · topological ordering · built-in calculators |
| J3 — Multi-Component | v0.3.0 | ✅ Achieved | PhysicalQuantity · MultiDomainState · CouplingOperator (INV-3) |
| J4a — Integrators | v0.4.0 | ✅ Achieved | Euler, RK4, DoPri45, Backward Euler, Crank-Nicolson, BDF2, IMEX |
| J5 — Discretisation | v0.5.0 | ✅ Achieved | DiscreteOperator (INV-2) · FD/FV · WENO3/5 |
| J6 — Sparse Algebra & Persistence | v0.6.0 | ✅ Achieved | faer-sparse · VTK/HDF5 export · SimulationSnapshot |
| J7 — Nonlinear Time Integration | v0.7.0 | ⏳ Planned | Newton and related methods for implicit integrators |
| J8 — Computational Optimisation | v0.8.0 | ⏳ Planned | Profiling, algorithmic/memory optimisation, GPU (`wgpu`) |
| J9 — Parallelism & Benchmarking | v0.9.0 | ⏳ Planned | Rayon · dirty-flag cache · Criterion benchmarks |
| J10 — Stable Ecosystem | v1.0.0 | ⏳ Planned | 7 examples · FEM audit INV-1/2/3 · stable API |
| J20 — FEM | v2.0.0 | ⏳ Planned | Unstructured meshes · ALE · INV-4 plugin-safe |
| J30 — Niche Frameworks | v3.0.0 | ⏳ Planned | oxiflow-chrom · oxiflow-geo · oxiflow-thermo · oxiflow-em · CLI · third-party |

Each milestone is independently deliverable. J1 alone (v0.1) is a usable library for
chromatography modelling. Third-party framework development can begin as soon as v2.0
is published and INV-4 is in place.

---

## 3. J1 — Core Architecture (v0.1)

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

## 4. J2 — Complete Context (v0.2)

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

## 5. J3 — Multi-Component (v0.3)

Indexed `PhysicalQuantity`. `MultiDomainState`. `CouplingOperator` inter-domain (INV-3).
Proto lahar–lake example on regular grids — the regression base for v2.0 FEM.

---

## 6. J4a — Integrators (v0.4)

| Integrator | Type | Status | Issue / DD |
|---|---|---|---|
| Forward Euler | Explicit, 1st order | ✅ Closed | #33, #41 |
| RK4 | Explicit, 4th order | ✅ Closed | #41 |
| Backward Euler | Implicit, 1st order | ✅ Closed | #43, DD-013, DD-033 |
| Crank-Nicolson | Semi-implicit, 2nd order | ✅ Closed | #43, DD-013, DD-033 |
| BDF2 | Implicit multi-step, 2nd order | ✅ Closed | #44, DD-034 |
| DoPri45 | Adaptive explicit, order 5 | ✅ Closed | #42, DD-036 |
| IMEX (Strang splitting) | Transport-reaction | ✅ Closed | #45, DD-037 |

J4a is fully delivered: all seven integrators are closed, including IMEX (#45) and the
serde `cfg_attr` annotations on J4 types (#70).

Architecture established along the way, reusable beyond J4a:

- **`SteppableSolver`** (DD-031, DD-034) — per-step primitive (`step()`), bounded history via
  `history_depth()` for multi-step methods (BDF2). Default `solve_fixed_step()` body (DD-035)
  shared by every fixed-step integrator — each `Solver::solve()` above is a single call to it.
- **`MultiDomainOrchestrator`** (DD-031) — drives several coupled domains, each with its own
  `SteppableSolver`; `dt` synchronised across domains (multirate explicitly deferred).
- **`LinearSolver`** (DD-013, `solver::linear`) — backend-agnostic `Ax=b`; `nalgebra` dense
  delivered at J4a, `faer` sparse planned at J6 (v0.6.0, #50) behind the same trait.
- **`StepSizeController`** (DD-036, `solver::methods::step_control`) — error-source-agnostic
  adaptive step control (PI controller); DoPri45 is the first consumer, a future adaptive
  implicit solver (iterated Newton, DD-033, J7) is the anticipated second.
- **`CompositeModel`** (DD-037, `model::composite`) — sum of several `PhysicalModel` on the
  same state; serves as a testable monolithic reference for `OperatorSplittingSolver`
  (`solver::methods::imex`).
- DoPri45 implements `Solver` only, not `SteppableSolver` — choosing its own `dt` across calls is
  in direct tension with the orchestrator's synchronised-`dt` v1 scope, not an orthogonal gap.

---

## 7. J5 — Discretisation (v0.5)

**Active development.** Two sibling traits, each carrying only what its family actually
needs (DD-012, DD-039):

```rust
/// Raw differential quantity — never needs context (dx comes from the mesh).
pub trait DiscreteOperator: Send + Sync {
    type MeshType: Mesh;
    fn apply(&self, field: &ContextValue, mesh: &Self::MeshType)
        -> Result<ContextValue, OxiflowError>;
}

/// COMPLETE -∇·F(u, ∇u) contribution — needs ComputeContext for F's physics
/// (velocity, diffusion, potentially variable). Does NOT extend DiscreteOperator:
/// the apply() signatures diverge (ctx present vs absent).
pub trait FluxDivergenceOperator: RequiresContext + Send + Sync {
    type MeshType: Mesh;
    fn apply(&self, field: &ContextValue, mesh: &Self::MeshType, ctx: &ComputeContext)
        -> Result<ContextValue, OxiflowError>;
}
```

Spatial schemes: upwind/centred FD (#47, `DiscreteOperator`), conservative FV (#48),
WENO3/5 with flux limiters (MinMod, Van Leer, Superbee) and adaptive Péclet-based selection
(#49, both `FluxDivergenceOperator`).

Two distinct insertion points, each consistent with its family's trait:
- **FD** — consumed via the existing `ContextCalculator` pipeline; `FDGradientCalculator`/
  `FDLaplacianCalculator` delegate their stencil to `operators::fd` with no change to the
  public API (dedicated refactor issue, depends on #47).
- **FV/WENO** — consumed directly in `compute_physics()` via the new
  `DiscretizedModel<Op: FluxDivergenceOperator>` composite (DD-038 amended, DD-039), which
  wraps the spatial scheme as an ordinary `PhysicalModel` returning only `-∇·F`. The type
  bound guarantees at compile time that an FD operator cannot be wired in by mistake. The
  source term S, when it exists, is composed via `CompositeModel` (DD-037, already
  delivered) — no new `SourceTerm` trait is introduced this sprint. F's physical parameters
  (velocity, diffusion) are either constant construction fields (simple case,
  `required_variables() -> vec![]`) or referenced by `ContextVariable` and supplied by an
  ordinary `ContextCalculator` already scheduled by `chain.rs` (DD-009) — no new machinery.
  The internal calculator (`FluxDivergenceCalculator`) stays private this sprint — reserved
  `instance_id` field for a future publication in `ComputeContext` when VTK/HDF5 export
  lands (J6, DD-027).

Linear algebra delegated to `nalgebra` (dense, delivered J4a) and `faer` (sparse, planned J6) —
`faer` integration extends the `LinearSolver` trait already established at J4a (DD-013), not a
new abstraction.

**Exit criterion:** #46–#49 closed, DD-012, DD-038, and DD-039 closed, FD delegation refactor
delivered.

---

## 8. J6 — Sparse Algebra & Persistence (v0.6)

Sparse `faer-sparse` linear solver for implicit systems (#50, DD-013 phase 2). Results export:
VTK (`vtkio`) as the interop pivot for `SimulationResult` (#78, DD-027), HDF5 (`hdf5-metno`,
dependency migration #79) for bulk datasets, including HDF5-backed loading for
`ExternalTabulated` (#105, split from #75). `SimulationSnapshot` generalised beyond crash
recovery to normal checkpoint/resume (DD-029, extends #71, #77) — no `Solver::checkpoint()`
method; callers construct the snapshot directly, see `snapshot.rs` module docs.

DD-027 amendment 1 adds the first config-driven HOW-axis DTO, `IntegratorSpec` (#104) —
`TryFrom<IntegratorSpec> for Box<dyn Solver>`, `BackwardEuler` variant only for this
increment (the sparse dispatch path, #50). Remaining integrators and the WHAT axis
(`ModelConfig`/`MeshConfig`/`BoundaryConditionConfig`) are follow-up, not part of J6.

This is also the milestone at which the private `FluxDivergenceCalculator` (DD-038, J5) becomes
a candidate for registration in the `ContextCalculator` pipeline, if generic flux-field export
turns out to be needed — a decision to revisit with the DD-027 detail in hand, not fixed at J5.

Feature flags: `sparse`, `hdf5`, `vtk`.

---

## 9. J7 — Nonlinear Time Integration (v0.7)

DD-033 freezes the theta-method implicit integrators (Backward Euler, Crank-Nicolson, BDF2)
to a single non-iterated Newton-style correction for J4a — exact for problems affine in `u`,
a first-order approximation otherwise. J7 lifts this limitation: a nonlinear solver (Newton
iterated to convergence, or a related method) plugged in behind the same extension point
DD-033 already anticipated, with no rewrite of the J4a solvers.

**Exit criterion:** a problem nonlinear in `u` converges at the expected order under
Backward Euler/Crank-Nicolson/BDF2 with the nonlinear solver, where the frozen J4a
correction only gave a first-order approximation.

---

## 10. J8 — Computational Optimisation (v0.8)

Profiling of the computational core (calculator chain, J5 spatial operators, J6 linear
solvers) and algorithmic/memory optimisation targeted at measured hotspots. Formalisation
of GPU-readiness (DD-026, five structural invariants INV-GPU-1 through INV-GPU-5: contiguous
memory, explicit dimensions, no trait objects in the numerical core). `gpu` feature flag
(`wgpu`) introduced behind these invariants — not before they are verified against the
existing codebase. ROCm explicitly out of scope.

---

## 11. J9 — Parallelism & Benchmarking (v0.9)

Rayon parallelism (opt-in `parallel` feature, configurable threshold via
`set_parallel_threshold()`, DD-014). Dirty-flag cache with temporal invalidation on
`ComputeContext` (DD-015). Criterion benchmarks.

**Exit criterion:** reference benchmark (1D diffusion, 1000 points, 10k steps) < 100 ms on a
modern CPU, with and without `parallel`.

---

## 12. J10 — Stable Ecosystem (v1.0)

Seven multi-domain examples: competitive chromatography, transient heat transfer,
Gray–Scott Turing patterns, Burgers boundary layer, Terzaghi consolidation,
magnetic diffusion (proto), lahar–lake coupled grids.

FEM invariant audit before publication (INV-1/2/3 verified across the full codebase).

API stability: SemVer strict, `cargo-semver-checks` in the release pipeline, MSRV documented.

`oxiflow-prelude` (DD-023): ergonomic entry crate — re-exports + built-in calculators +
`quick_config()`/`run()` for simple cases — strictly one-directional dependency onto the
engine.

---

## 13. FEM Compatibility — v2.0 Trajectory (J20)

### 13.1 Motivating case

A rapid gravitational movement (lahar, landslide) entering a body of water and generating
a submersion wave. Requires unstructured meshing for irregular geometry and adaptive
refinement for the wave front — both impossible with finite differences.

| Component | Model | Numerical challenge |
|---|---|---|
| Granular domain | Bingham + extended Saint-Venant | moving boundary · adaptive mesh |
| Fluid domain | Shallow Water equations | irregular bathymetry |
| Moving interface | ALE formulation | mass/momentum transfer |

### 13.2 INV-4 — Plugin-safe API

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

### 13.3 v2.0 scope

Unstructured mesh (minimal internal Gmsh `.msh` parser — nodes, connectivity, physical
groups → `BoundaryLocation`, DD-028; triangles 2D, tetrahedra 3D, h-adaptive refinement).
Function spaces (P1, P2 Lagrange, Raviart–Thomas, DG0). FEM assembler
(stiffness and mass matrices, Gauss quadrature, face integration). Sparse linear solvers
(`faer-sparse`, ILU/AMG preconditioners). ALE formulation for the lahar–lake example.
Spectral methods (DD-024) remain an open question deferred past v1.0 — the FEM experience
with `Mesh::coordinates()` at J20 will inform its viability.

---

## 14. J30 — Niche Frameworks (v3.0)

### 14.1 Architecture

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

### 14.2 Declarative configuration

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

### 14.3 CLI

```bash
oxiflow run problem.toml          # solve
oxiflow check problem.toml        # validate before solving
oxiflow list frameworks           # oxiflow-chrom, oxiflow-geo, ...
oxiflow list models --framework chrom
```

### 14.4 Planned first-party frameworks

| Crate | Domain | Key models |
|---|---|---|
| `oxiflow-chrom` | Chromatography | Langmuir, SMA, Thomas, gradient elution, Danckwerts BC |
| `oxiflow-geo` | Surface geophysics | Bingham Saint-Venant, Shallow Water, ALE interface |
| `oxiflow-thermo` | Heat transfer | Fourier flux, Robin BC, phase change |
| `oxiflow-em` | Diffusive electromagnetism | magnetic diffusion, eddy currents |

### 14.5 Third-party frameworks

Third parties are explicitly encouraged to publish `oxiflow-*` crates on crates.io.
Requirements for a third-party framework:

- Depends on `oxiflow = "2"` (or higher).
- Preserves the `NOTICE` file from the engine in any redistribution (Apache 2.0 requirement).
- Uses a compatible license (Apache 2.0 recommended; any OSI-approved license accepted).
- Uses the `oxiflow-` prefix on crates.io for discoverability.
- Opens a PR against the engine repository to be added to the
  [Known Ecosystem Frameworks](#15-known-ecosystem-frameworks) list below.

---

## 15. Known Ecosystem Frameworks

| Crate | Domain | Maintainer | Status |
|---|---|---|---|
| `oxiflow-chrom` | Chromatography | oxiflow core team | Planned v3.0 |
| `oxiflow-geo` | Surface geophysics | oxiflow core team | Planned v3.0 |
| `oxiflow-thermo` | Heat transfer | oxiflow core team | Planned v3.0 |
| `oxiflow-em` | Diffusive EM | oxiflow core team | Planned v3.0 |

*To add a framework to this list, open a PR modifying this table.*

---

## 16. Architectural Decision Log

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
| Linear solvers (dense) | `nalgebra` delegation | Custom implementation | J4a | |
| Temporal composition | `SplitOperator`/`OperatorSplittingSolver` (Strang) | Fixed explicit/implicit pair | J4a | |
| Spatial operators | Abstract `DiscreteOperator` (associated type `MeshType`) | FD hardcoded | J5 | INV-2 |
| Spatial F/S composition | `DiscretizedModel<Op>` (F) + existing `CompositeModel` (F+S) | New `SourceTerm` trait; flux exposed via `ContextVariable` | J5 | INV-2 |
| Operator context access | `FluxDivergenceOperator`: sibling trait to `DiscreteOperator`, carries `ctx`/`RequiresContext` | Adding `ctx` directly to `DiscreteOperator`; subtrait of `DiscreteOperator` | J5 | INV-2 |
| Linear solvers (sparse) | `faer-sparse` delegation | Custom implementation | J6 | |
| Results export | VTK interop pivot + HDF5 for bulk data | Custom format | J6 | |
| Nonlinear integration | Iterated Newton, DD-033 extension point | Rewrite of J4a solvers | J7 | |
| GPU readiness | Structural invariants formalised before `gpu` feature | `wgpu` without prior constraints | J8 | |
| Parallelism | Rayon, opt-in feature flag | Mandatory or absent | J9 | |
| Caching | Dirty flag + temporal invalidation | Systematic recomputation | J9 | |
| API stability | SemVer + `cargo-semver-checks` + FEM audit | Informal convention | J10 | |
| Ergonomics | `oxiflow-prelude`, separate crate | Builder integrated into the engine | J10 | |
| Plugin architecture | Object-safe traits + `PluginRegistry` | Monolithic crate | J20 | INV-4 |
| Framework config | TOML + runtime registry | proc-macro DSL | J30 | |
| License | Apache 2.0 only | MIT or dual MIT/Apache | J0 | |

---

## 17. Risk Register

| ID | Risk | Probability | Mitigation |
|---|---|---|---|
| R1 | `ContextValue` generics too complex for users | Medium | Ergonomic helpers; user testing at v0.2 |
| R2 | Silent dependency ordering bugs | Low | Exhaustive cycle detection tests; debug logging |
| R3 | `PhysicalQuantity` indexing too verbose | Medium | Idiomatic constructors; UX feedback before v1.0 |
| R4 | Implicit solvers require heavy linear algebra | High | Delegate to `faer`/`nalgebra`; document limits |
| R5 | Rayon + potential `unsafe` | Low | Opt-in flag; ThreadSanitizer in CI |
| R6 | Scope too ambitious | Medium | Each milestone independently deliverable |
| R7 | Breaking change forced before v1.0 | Low | Accepted pre-1.0 but documented |
| R8 | INV-1/2/3 silently violated | Medium | Formal audit at J10; dedicated integration tests |
| R9 | ALE incompatible with CouplingOperator design | Low | Proto lahar–lake at J3 is the test bench |
| R10 | INV-4 violated — third-party frameworks break on engine update | Medium | `oxiflow-test-plugin` external crate in CI from v2.0; `cargo-semver-checks` in release pipeline |
| R11 | Fragmentation — incompatible third-party frameworks | Low | INV-4 + stable public API is the only compatibility contract; framework authors are responsible for their own SemVer |
| R12 | FV/WENO insertion point (DD-038) chosen wrong, costly to undo | Low | Internal calculator kept private (reserved `instance_id` field) — additive extension toward J6, no restructuring |

---

## 18. Timeline

GitHub milestone due dates (`oxiflow-milestones.yml`), not a relative month estimate —
replaces the old M+N scheme, which had drifted from the real due dates.

| Milestone | Version | Due date | Key objectives |
|---|---|---|---|
| J0 | v0.0.1–v0.0.5 | Closed (2026-03) | crates.io placeholder · CI · README · NOTICE |
| J1 | v0.1.0 | Closed (2026-04) | ContextValue · OxiflowError · Mesh (INV-1) |
| J2 | v0.2.0 | Closed (2026-05) | Requiring BCs · topology · built-in calculators |
| J3 | v0.3.0 | Closed (2026-06) | PhysicalQuantity · CouplingOperator (INV-3) · proto lahar–lake |
| J4a | v0.4.0 | Closed (2026-06) | Temporal integrators, IMEX included |
| J5 | v0.5.0 | Closed (2026-07-17) | DiscreteOperator (INV-2) · FD/FV · WENO3/5 |
| J6 | v0.6.0 | Closed (2026-07-21) | faer-sparse · VTK/HDF5 export · SimulationSnapshot · IntegratorSpec |
| J7 | v0.7.0 | 2026-10-07 | Nonlinear time integration (Newton) |
| J8 | v0.8.0 | 2026-11-18 | Computational optimisation, GPU readiness |
| J9 | v0.9.0 | 2026-12-30 | Rayon · dirty-flag cache · Criterion benchmarks |
| J10 | v1.0.0 | 2027-03-04 | 7 examples · API freeze · FEM audit · stable release |
| J20 | v2.0.0 | 2027-09-02 | Unstructured mesh · FEM assembler · ALE · INV-4 |
| J30 | v3.0.0 | 2028-03-02 | oxiflow-chrom · oxiflow-geo · oxiflow-thermo · oxiflow-em · CLI |
| — | Third-party | ongoing | Community frameworks on crates.io |

---

*oxiflow Development Program v2.2 · July 2026 · Living document — updated at each milestone*
