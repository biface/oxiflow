# oxiflow — Development Program

This document is the architectural reference for oxiflow. It covers the design principles,
milestone specifications, design invariants, ecosystem strategy, and decision log that guide
all implementation work from v0.1.0 to v0.30.0 and beyond.

> **Current version:** v0.9.0 — Parallelism & Benchmarking (published)
> **Active development:** v0.10.0 — Stable Ecosystem (J10), in progress
> **Document version:** 3.0 — October 2026

---

## Table of Contents

1. [Vision & Principles](#1-vision--principles)
2. [Milestone Overview](#2-milestone-overview)
3. [J1 — Core Architecture (v0.1)](#3-j1--core-architecture-v01)
4. [J2 — Complete Context (v0.2)](#4-j2--complete-context-v02)
5. [J3 — Multi-Component (v0.3)](#5-j3--multi-component-v03)
6. [J4 — Integrators (v0.4)](#6-j4--integrators-v04)
7. [J5 — Discretisation (v0.5)](#7-j5--discretisation-v05)
8. [J6 — Sparse Linear Algebra & Persistence (v0.6)](#8-j6--sparse-linear-algebra--persistence-v06)
9. [J7 — Nonlinear Time Integration (v0.7)](#9-j7--nonlinear-time-integration-v07)
10. [J8 — Computational Optimisation (v0.8)](#10-j8--computational-optimisation-v08)
11. [J9 — Parallelism & Benchmarking (v0.9)](#11-j9--parallelism--benchmarking-v09)
12. [J10 — Stable Ecosystem (v0.10)](#12-j10--stable-ecosystem-v010)
13. [FEM Compatibility — Trajectory from J11 to J20](#13-fem-compatibility--trajectory-from-j11-to-j20)
14. [J30 — Niche Frameworks (v0.30)](#14-j30--niche-frameworks-v030)
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
- **Anticipated FEM compatibility** — the J1–J10 abstractions are designed to not presuppose
  structured grids (INV-1/2/3)
- **Plugin-safe API** — all public traits are object-safe so third-party framework crates
  can implement them without accessing engine internals (INV-4, from v0.30.0)

### Positioning

oxiflow is not a full CFD framework (like OpenFOAM) and not a Python wrapper around LAPACK.
It is a numerical composition engine providing building blocks for rigorous, maintainable,
performant PDE solvers in any domain of continuous physics — and the foundation for a family
of niche frameworks that bring that power to specific scientific communities with minimal
boilerplate.

---

## 2. Milestone Overview

| Milestone | Version | Status | Date | Theme |
|---|---|---|---|---|
| J0 — Foundations | v0.0.1 / v0.0.5 | ✅ Published | 2026-03-20 | Placeholder · CI · project structure |
| J1 — Core Architecture | v0.1.0 | ✅ Published | 2026-04-12 | ContextValue · OxiflowError · Mesh (INV-1) |
| J2 — Complete Context | v0.2.0 | ✅ Published | 2026-05-02 | Requiring BCs · topological ordering · calculators |
| J3 — Multi-Component | v0.3.0 | ✅ Published | 2026-06-03 | PhysicalQuantity · MultiDomainState · CouplingOperator (INV-3) |
| J4 — Integrators | v0.4.0 | ✅ Published | 2026-06-25 | Euler, RK4, DoPri45, Backward Euler, Crank–Nicolson, BDF2, IMEX |
| J5 — Discretisation | v0.5.0 | ✅ Published | 2026-07-17 | DiscreteOperator (INV-2) · FD/FV · WENO3/5 · flux limiters + adaptive selection |
| J6 — Sparse Linear Algebra & Persistence | v0.6.0 | ✅ Published | 2026-07-21 | faer sparse solver · SimulationSnapshot (checkpoint/on_divergence) · HDF5 loading · VTK export · IntegratorSpec |
| J7 — Nonlinear Time Integration | v0.7.0 | ✅ Published | 2026-07-30 | Iterated Newton for implicit integrators ([DD-044](https://github.com/biface/oxiflow/issues/109)) · `IntegratorSpec` extended to BE/CN/BDF2 |
| J8 — Computational Optimisation | v0.8.0 | ✅ Published | 2026-08-03 | GPU-readiness ([DA-026](https://github.com/biface/oxiflow/issues/73), [DT-045](https://github.com/biface/oxiflow/issues/128)) · `gpu` feature (`wgpu`) · MSRV 1.84 |
| J9 — Parallelism & Benchmarking | v0.9.0 | ✅ Published | 2026-09-16 | Rayon dispatch ([DT-014](https://github.com/biface/oxiflow/issues/14)/[DA-048](https://github.com/biface/oxiflow/issues/140)) · Criterion benchmark suite (`oxiflow_parallel`) · lahar multi-field model ([DQ-032](https://github.com/biface/oxiflow/issues/85)) · SVG reference plots |
| J10 — Stable Ecosystem | v0.10.0 | 🚧 In progress | 2026-11-30 | Seven reference examples · `(x, t)` coefficients (`CoefficientField`, [DD-066](https://github.com/biface/oxiflow/issues/181)) · FEM invariant audit · strict SemVer, `cargo-semver-checks` ([DQ-016](https://github.com/biface/oxiflow/issues/16)) |
| J11 — nD Mesh Foundations | v0.11.0 (FEM) / v0.11.5 (structured) | ⏳ Planned | 2027-01-14 / 2027-02-28 | `FiniteElementMesh`, unstructured 2D/3D meshes ([DA-049](https://github.com/biface/oxiflow/issues/150)) · `UniformGrid1D` generalised to nD ([DD-051](https://github.com/biface/oxiflow/issues/152)) |
| J12 — Velocity-Pressure Discretisation | v0.12.0 / v0.12.5 | ⏳ Planned | 2027-04-14 / 2027-05-29 | FEM (Gmsh import, [DT-028](https://github.com/biface/oxiflow/issues/76)) or MAC/collocated + Rhie–Chow (structured) |
| J13 — SaddlePointSolver | v0.13.0 | ⏳ Planned | 2027-07-13 | `SaddlePointSolver`, shared by both paths |
| J14 — Differential-Algebraic Equations & Navier–Stokes | v0.14.0 | ⏳ Planned | 2027-08-27 | DAE time integration · Navier–Stokes as a `PhysicalModel`, shared |
| J15 — End-to-End NS Wiring | v0.15.0 / v0.15.5 | ⏳ Planned | 2027-10-11 / 2027-11-25 | Taylor–Hood/MINI (FEM) or full structured pipeline |
| J16 — Preconditioned Sparse Solvers | v0.16.0 | ⏳ Planned | 2028-01-09 | ILU/AMG preconditioners, ahead of the comparison that needs them at fine resolution ([DQ-047](https://github.com/biface/oxiflow/issues/132)) |
| J17 — FEM vs Structured Comparison | v0.17.0 | ⏳ Planned | 2028-02-23 | Lid-driven cavity, Poiseuille, Ghia et al. 1982 |
| J18 — Arbitrary Lagrangian–Eulerian | v0.18.0 | ⏳ Planned | 2028-04-08 | ALE formulation |
| J19 — Spectral Methods | v0.19.0 | ⏳ Planned | 2028-05-23 | `Discretization` common ancestor, `Mesh`/`SpectralBasis` siblings ([DA-024](https://github.com/biface/oxiflow/issues/63)) |
| J20 — Stabilisation | v0.20.0 | 🔭 Horizon | 2028-07-07 | Checkpoint closing the first cycle ([DQ-063](https://github.com/biface/oxiflow/issues/165)) · 1.0.0 is tied to no milestone ([DQ-016](https://github.com/biface/oxiflow/issues/16)) |
| J30 — Niche Frameworks | v0.30.0 | 🔭 Horizon | 2029-12-31 | oxiflow-chrom · oxiflow-geo · oxiflow-thermo · oxiflow-em · CLI `oxiflow run` · third-party plugins ([DA-017](https://github.com/biface/oxiflow/issues/17)) |
| Future Phase | v0.40.0 | 🔭 Horizon | 2030-12-31 | Scope not yet defined |

Date: publication date of a published milestone, due date of the others (GitHub milestones).

Each milestone is independently deliverable. J1 alone (v0.1) is a usable library for
chromatography modelling. Third-party framework development can begin as soon as INV-4 is
in place (v0.30.0, §13.2).

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
| Danckwerts inlet | Robin | feed concentration (function of time) |
| Danckwerts outlet | Neumann | none (neighbouring node) |

---

## 5. J3 — Multi-Component (v0.3)

Indexed `PhysicalQuantity`. `MultiDomainState`. `CouplingOperator` inter-domain (INV-3).
Proto lahar–lake example on regular grids — the regression base for the FEM path (J11 onward).

---

## 6. J4 — Integrators (v0.4)

| Integrator | Type | Status | Issue / DD |
|---|---|---|---|
| Forward Euler | Explicit, 1st order | ✅ Closed | [#33](https://github.com/biface/oxiflow/issues/33), [#41](https://github.com/biface/oxiflow/issues/41) |
| RK4 | Explicit, 4th order | ✅ Closed | [#41](https://github.com/biface/oxiflow/issues/41) |
| Backward Euler | Implicit, 1st order | ✅ Closed | [#43](https://github.com/biface/oxiflow/issues/43), [DT-013](https://github.com/biface/oxiflow/issues/13), [DD-033](https://github.com/biface/oxiflow/issues/86) |
| Crank-Nicolson | Semi-implicit, 2nd order | ✅ Closed | [#43](https://github.com/biface/oxiflow/issues/43), [DT-013](https://github.com/biface/oxiflow/issues/13), [DD-033](https://github.com/biface/oxiflow/issues/86) |
| BDF2 | Implicit multi-step, 2nd order | ✅ Closed | [#44](https://github.com/biface/oxiflow/issues/44), [DD-034](https://github.com/biface/oxiflow/issues/87) |
| DoPri45 | Adaptive explicit, order 5 | ✅ Closed | [#42](https://github.com/biface/oxiflow/issues/42), [DD-036](https://github.com/biface/oxiflow/issues/89) |
| IMEX (Strang splitting) | Transport-reaction | ✅ Closed | [#45](https://github.com/biface/oxiflow/issues/45), [DA-037](https://github.com/biface/oxiflow/issues/90) |

J4 is fully delivered: all seven integrators are closed, including IMEX ([#45](https://github.com/biface/oxiflow/issues/45)) and the
serde `cfg_attr` annotations on J4 types ([#70](https://github.com/biface/oxiflow/issues/70)).

Architecture established along the way, reusable beyond J4:

- **`SteppableSolver`** ([DA-031](https://github.com/biface/oxiflow/issues/82), [DD-034](https://github.com/biface/oxiflow/issues/87)) — per-step primitive (`step()`), bounded history via
  `history_depth()` for multi-step methods (BDF2). Default `solve_fixed_step()` body ([DD-035](https://github.com/biface/oxiflow/issues/88))
  shared by every fixed-step integrator — each `Solver::solve()` above is a single call to it.
- **`MultiDomainOrchestrator`** ([DA-031](https://github.com/biface/oxiflow/issues/82)) — drives several coupled domains, each with its own
  `SteppableSolver`; `dt` synchronised across domains (multirate explicitly deferred).
- **`LinearSolver`** ([DT-013](https://github.com/biface/oxiflow/issues/13), `solver::linear`) — backend-agnostic `Ax=b`; `nalgebra` dense
  delivered at J4, `faer` sparse planned at J6 (v0.6.0, [#50](https://github.com/biface/oxiflow/issues/50)) behind the same trait.
- **`StepSizeController`** ([DD-036](https://github.com/biface/oxiflow/issues/89), `solver::methods::step_control`) — error-source-agnostic
  adaptive step control (PI controller); DoPri45 is the first consumer, a future adaptive
  implicit solver (iterated Newton, [DD-033](https://github.com/biface/oxiflow/issues/86), J7) is the anticipated second.
- **`CompositeModel`** ([DA-037](https://github.com/biface/oxiflow/issues/90), `model::composite`) — sum of several `PhysicalModel` on the
  same state; serves as a testable monolithic reference for `OperatorSplittingSolver`
  (`solver::methods::imex`).
- DoPri45 implements `Solver` only, not `SteppableSolver` — choosing its own `dt` across calls is
  in direct tension with the orchestrator's synchronised-`dt` v1 scope, not an orthogonal gap.

---

## 7. J5 — Discretisation (v0.5)

Two sibling traits, each carrying only what its family actually
needs ([DA-012](https://github.com/biface/oxiflow/issues/12), [DA-039](https://github.com/biface/oxiflow/issues/95)):

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

Spatial schemes: upwind/centred FD ([#47](https://github.com/biface/oxiflow/issues/47), `DiscreteOperator`), conservative FV ([#48](https://github.com/biface/oxiflow/issues/48)),
WENO3/5 with flux limiters (MinMod, Van Leer, Superbee) and adaptive Péclet-based selection
([#49](https://github.com/biface/oxiflow/issues/49), both `FluxDivergenceOperator`).

Two distinct insertion points, each consistent with its family's trait:
- **FD** — consumed via the existing `ContextCalculator` pipeline; `FDGradientCalculator`/
  `FDLaplacianCalculator` delegate their stencil to `operators::fd` with no change to the
  public API (dedicated refactor issue, depends on [#47](https://github.com/biface/oxiflow/issues/47)).
- **FV/WENO** — consumed directly in `compute_physics()` via the new
  `DiscretizedModel<Op: FluxDivergenceOperator>` composite ([DA-038](https://github.com/biface/oxiflow/issues/92) amended, [DA-039](https://github.com/biface/oxiflow/issues/95)), which
  wraps the spatial scheme as an ordinary `PhysicalModel` returning only `-∇·F`. The type
  bound guarantees at compile time that an FD operator cannot be wired in by mistake. The
  source term S, when it exists, is composed via `CompositeModel` ([DA-037](https://github.com/biface/oxiflow/issues/90), already
  delivered) — no new `SourceTerm` trait is introduced this sprint. F's physical parameters
  (velocity, diffusion) are either constant construction fields (simple case,
  `required_variables() -> vec![]`) or referenced by `ContextVariable` and supplied by an
  ordinary `ContextCalculator` already scheduled by `chain.rs` ([DD-009](https://github.com/biface/oxiflow/issues/9)) — no new machinery.
  The internal calculator (`FluxDivergenceCalculator`) stays private this sprint — reserved
  `instance_id` field for a future publication in `ComputeContext` when VTK/HDF5 export
  lands (J6, [DT-027](https://github.com/biface/oxiflow/issues/75)).

Linear algebra delegated to `nalgebra` (dense, delivered J4) and `faer` (sparse, planned J6) —
`faer` integration extends the `LinearSolver` trait already established at J4 ([DT-013](https://github.com/biface/oxiflow/issues/13)), not a
new abstraction.

**Exit criterion:** [#46](https://github.com/biface/oxiflow/issues/46)–[#49](https://github.com/biface/oxiflow/issues/49) closed, [DA-012](https://github.com/biface/oxiflow/issues/12), [DA-038](https://github.com/biface/oxiflow/issues/92), and [DA-039](https://github.com/biface/oxiflow/issues/95) closed, FD delegation refactor
delivered.

---

## 8. J6 — Sparse Linear Algebra & Persistence (v0.6)

Sparse `faer-sparse` linear solver for implicit systems ([#50](https://github.com/biface/oxiflow/issues/50), [DT-013](https://github.com/biface/oxiflow/issues/13) phase 2). Results export:
VTK (`vtkio` at the time, replaced by an in-crate `.vtu` writer in v0.10.0, [#191](https://github.com/biface/oxiflow/issues/191)) as the interop pivot for `SimulationResult` ([#78](https://github.com/biface/oxiflow/issues/78), [DT-027](https://github.com/biface/oxiflow/issues/75)), HDF5 (`hdf5-metno`,
dependency migration [#79](https://github.com/biface/oxiflow/issues/79)) for bulk datasets, including HDF5-backed loading for
`ExternalTabulated` ([#105](https://github.com/biface/oxiflow/issues/105), split from [#75](https://github.com/biface/oxiflow/issues/75)). `SimulationSnapshot` generalised beyond crash
recovery to normal checkpoint/resume ([DD-029](https://github.com/biface/oxiflow/issues/77), extends [#71](https://github.com/biface/oxiflow/issues/71), [#77](https://github.com/biface/oxiflow/issues/77)) — no `Solver::checkpoint()`
method; callers construct the snapshot directly, see `snapshot.rs` module docs.

[DT-027](https://github.com/biface/oxiflow/issues/75) amendment 1 adds the first config-driven HOW-axis DTO, `IntegratorSpec` ([#104](https://github.com/biface/oxiflow/issues/104)) —
`TryFrom<IntegratorSpec> for Box<dyn Solver>`, `BackwardEuler` variant only for this
increment (the sparse dispatch path, [#50](https://github.com/biface/oxiflow/issues/50)). Remaining integrators and the WHAT axis
(`ModelConfig`/`MeshConfig`/`BoundaryConditionConfig`) are follow-up, not part of J6.

This is also the milestone at which the private `FluxDivergenceCalculator` ([DA-038](https://github.com/biface/oxiflow/issues/92), J5) becomes
a candidate for registration in the `ContextCalculator` pipeline, if generic flux-field export
turns out to be needed — a decision to revisit with the [DT-027](https://github.com/biface/oxiflow/issues/75) detail in hand, not fixed at J5.

Feature flags: `sparse`, `hdf5`, `vtk`.

---

## 9. J7 — Nonlinear Time Integration (v0.7)

[DD-033](https://github.com/biface/oxiflow/issues/86) freezes the theta-method implicit integrators (Backward Euler, Crank-Nicolson, BDF2)
to a single non-iterated Newton-style correction for J4 — exact for problems affine in `u`,
a first-order approximation otherwise. J7 lifts this limitation: a nonlinear solver (Newton
iterated to convergence, or a related method) plugged in behind the same extension point
[DD-033](https://github.com/biface/oxiflow/issues/86) already anticipated, with no rewrite of the J4 solvers.

### Why this is needed: the discretised residual is not always linear

The canonical form (§1) is `∂u/∂t + ∇·F(u, ∇u) = S(u, x, t)`. Write the right-hand side as a
single function of the state, `f(u) = -∇·F(u, ∇u) + S(u, x, t)`. Discretising with the
theta-method (Backward Euler: θ=1, Crank-Nicolson: θ=0.5) turns the PDE into, at every time
step, a root-finding problem in the unknown `u^{n+1}`:

```
g(u^{n+1}) = u^{n+1} - u^n - Δt · [(1-θ)·f(u^n) + θ·f(u^{n+1})] = 0
```

(BDF2 has its own residual, `g(u) = (3/2)u - 2u^n + (1/2)u^{n-1} - Δt·f(u)`, but the same
argument applies.) `g` is **linear** in `u^{n+1}` exactly when `f` is affine in `u` — i.e. when
both `F` and `S` are affine. That is the case J4's single frozen-Jacobian correction ([DD-033](https://github.com/biface/oxiflow/issues/86),
Option B) already solves exactly: for affine `f`, the Jacobian `∂g/∂u` is constant, independent
of `u^{n+1}`, so a single Newton step from `u^n` lands exactly on the root — no iteration
needed. This is not an approximation choice, it's a property of linear algebra.

Any **physically nonlinear** `F` or `S` breaks this: `g` becomes a nonlinear algebraic equation,
and J4's single correction is only a first-order approximation to its root — good enough for
mildly nonlinear problems and small `Δt`, not exact in general. Concrete cases oxiflow needs to
handle correctly, drawn from the project's own target domains:

- **Nonlinear flux `F(u)`** — e.g. Burgers-type nonlinear advection (`F(u) = u²/2`), or a
  diffusion coefficient that depends on the field itself (`F = -D(u)∇u`, degenerate/nonlinear
  diffusion).
- **Nonlinear source `S(u)`** — e.g. Langmuir-type adsorption isotherms in chromatography
  (`q(c) = q_max·K·c / (1 + K·c)`, oxiflow's own originating domain via chrom-rs), or
  Arrhenius-type reaction-rate kinetics (`S ∝ exp(-E_a / R·T)`).

Newton solves `g(u) = 0` by linearising it around the current iterate `u^k` and correcting:

```
J^k · Δu^k = -g(u^k),    J^k = ∂g/∂u |_{u=u^k} = I - Δt·θ · ∂f/∂u |_{u=u^k}
u^{k+1} = u^k + Δu^k
```

`∂f/∂u` is exactly what `finite_difference_jacobian` ([DT-013](https://github.com/biface/oxiflow/issues/13), already in place since J4)
approximates by finite differences — Newton reuses it unchanged. What [DD-044](https://github.com/biface/oxiflow/issues/109) changes is *how
many times* this Jacobian is (re-)evaluated per time step and *when the loop stops* — see
below — not the residual, the Jacobian assembly, or the canonical form itself.

### [DD-044](https://github.com/biface/oxiflow/issues/109) — activating this correctly

[DD-044](https://github.com/biface/oxiflow/issues/109) formalises this activation: `NewtonConvergence` (residual-based, default) and
`JacobianStrategy` (modified/frozen, default) as configurable HOW-axis enums;
`OxiflowError::NewtonNotConverged` routed through `Solver::on_divergence()`; a single generic
Newton loop shared by theta-method (Backward Euler/Crank-Nicolson) and BDF2, rather than the
per-family duplication J4 used when Newton had only one consumer; `IntegratorSpec` extended
to `BackwardEuler` (Newton fields added) plus new `CrankNicolson` and `BDF2` variants (the
latter without sparse fields — `BDF2Solver` had no sparse dispatch, a [DA-043](https://github.com/biface/oxiflow/issues/103) gap left open
here, not a J7 concern; closed in v0.10.0 by [#133](https://github.com/biface/oxiflow/issues/133)).

Operational breakdown ([DD-044](https://github.com/biface/oxiflow/issues/109), `oxiflow-issues-v0.7.0.yml`):

1. Newton core loop (`NewtonConvergence`, `JacobianStrategy`, `NewtonNotConverged`,
   `newton_residual`/`newton_linear_correction`, generic loop) — prerequisite for 2–3.
2. Wire into Backward Euler/Crank-Nicolson.
3. Wire into BDF2.
4. `IntegratorSpec::BackwardEuler` extended, `CrankNicolson` and `BDF2` variants added —
   depend on 2/3 respectively.
5. Regression test for the [DD-033](https://github.com/biface/oxiflow/issues/86) known-untested limitation (perturbed-Jacobian ×
   Dirichlet BC interaction) — independent of 1–4.

**Exit criterion:** a problem nonlinear in `u` converges at the expected order under
Backward Euler/Crank-Nicolson/BDF2 with the nonlinear solver, where the frozen J4
correction only gave a first-order approximation.

---

## 10. J8 — Computational Optimisation (v0.8)

Profiling of the computational core (calculator chain, J5 spatial operators, J6 linear
solvers) and algorithmic/memory optimisation targeted at measured hotspots. Formalisation
of GPU-readiness ([DA-026](https://github.com/biface/oxiflow/issues/73), five structural invariants INV-GPU-1 through INV-GPU-5: contiguous
memory, explicit dimensions, no trait objects in the numerical core). `gpu` feature flag
(`wgpu`) introduced behind these invariants — not before they are verified against the
existing codebase. ROCm explicitly out of scope.

---

## 11. J9 — Parallelism & Benchmarking (v0.9)

Rayon parallelism (opt-in `parallel` feature, configurable threshold via
`set_parallel_threshold()`, [DT-014](https://github.com/biface/oxiflow/issues/14)). Dirty-flag cache with temporal invalidation on
`ComputeContext` ([DD-015](https://github.com/biface/oxiflow/issues/15)). Criterion benchmarks.

**Exit criterion:** reference benchmark (1D diffusion, 1000 points, 10k steps) < 100 ms on a
modern CPU, with and without `parallel`.

---

## 12. J10 — Stable Ecosystem (v0.10)

Seven multi-domain reference examples, one per domain of the README table: binary frontal
analysis with competitive adsorption; 1D transient conduction in a bar (uniform and graded);
Gray–Scott in 1D: Turing instability, pulse splitting; 1D viscous boundary layer at a wall
(Burgers); 1D consolidation of a clay layer (Terzaghi); 1D magnetic diffusion into a conductor
(proto); a lahar front running into a lake. Each one checks itself against its validation
reference and also runs in CI.

Coefficients that depend on position and time, `(x, t)`, through `CoefficientField` ([DD-066](https://github.com/biface/oxiflow/issues/181),
[#181](https://github.com/biface/oxiflow/issues/181)).

FEM invariant audit before publication (INV-1/2/3 verified across the full codebase).

API stability: SemVer strict, `cargo-semver-checks` in the release pipeline, MSRV documented.

`oxiflow-prelude` ([DA-023](https://github.com/biface/oxiflow/issues/62)), an ergonomic entry crate — re-exports + built-in calculators +
`quick_config()`/`run()` for simple cases — with a strictly one-directional dependency onto the
engine, is not part of J10: it is deferred to J20 (v0.20.0) and on hold.

### 12.1 Examples, tests, benchmarks and shared models

Each reference model is defined once and used by every target that needs it ([#194](https://github.com/biface/oxiflow/issues/194)):

| Directory | Role | Runs |
|---|---|---|
| `examples/` | Complete, validated examples showing how to instantiate oxiflow's tools | By hand; the reference examples of [#173](https://github.com/biface/oxiflow/issues/173) also run in CI, the others are compile-checked by `clippy --all-targets` |
| `tests/` | Integration tests on real model chains; the epsilon-level comparisons with analytical and literature references (`*_analytical.rs`, `*_reference.rs`) live here | CI (`cargo test`) |
| `benches/` | Benchmarking against the literature and dispatch charts: conditions of use of the solvers, implementation choices, reproducibility | By hand, locally (`7. Bench` run configuration), not in CI |
| `rsc/models/` | Shared model definitions, included by the three targets above with `#[path]` | Through the targets that include them |

Rules for `rsc/models/`:

- The main example of a model carries its documentation (physics, exact solution, validation);
  the shared file points to it. Scenario constants, `main`, plots and assertions stay in each
  target, unless a benchmark must run the scenario of an example: the scenario (physical
  constants, grids, initial states, couplings) then moves to the shared file, parametrised by
  what the benchmark varies, and only the run settings stay in each target (`lahar_outfall`,
  `lahar_validation`, for [#141](https://github.com/biface/oxiflow/issues/141)).
- A file is included as a module named after it, with a `_model` suffix when the file holds a
  whole model (`#[path = "../rsc/models/diffusion_1d.rs"] mod diffusion_1d_model;`), so that
  it clashes neither with a module of the crate (`oxiflow::model`) nor with a bench module of
  the same name. The lahar files (`lahar_physics`, `lahar_flux`, `lahar_source`,
  `lahar_outfall`, `lahar_validation`) keep their own names; `lahar_flux` and `lahar_source`
  import from a sibling module `lahar_physics`, `lahar_outfall` and `lahar_validation` from all
  three.
- Shared items are `pub(crate)`: they belong to the target that includes them, never to the
  crate's public API.
- An item that one target does not use carries a local `#[allow(dead_code)]` with a comment
  naming that target; never a blanket `allow` on a shared file.
- `rsc/` is not discovered by Cargo, so `Cargo.toml` does not list it. A future `include`
  list for `cargo publish` must contain it, otherwise examples, tests and benchmarks would
  not build from the published package.

---

## 13. FEM Compatibility — Trajectory from J11 to J20

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

**Introduced at v0.30.0 [J30]** ([DA-017](https://github.com/biface/oxiflow/issues/17)). All public traits must be object-safe and fully accessible from
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

INV-4 is the prerequisite for the niche frameworks of v0.30.0. No niche framework can be developed before it is in
place and verified.

### 13.3 Scope of the FEM path

Spread over J11 to J18 (see §2). Unstructured mesh (minimal internal Gmsh `.msh` parser — nodes,
connectivity, physical groups → `BoundaryLocation`, [DT-028](https://github.com/biface/oxiflow/issues/76); triangles 2D, tetrahedra 3D,
h-adaptive refinement). Function spaces (P1, P2 Lagrange, Raviart–Thomas, DG0). FEM assembler
(stiffness and mass matrices, Gauss quadrature, face integration). Sparse linear solvers
(`faer-sparse`, ILU/AMG preconditioners). ALE formulation for the lahar–lake example. Spectral
methods ([DA-024](https://github.com/biface/oxiflow/issues/63)) come after this path, at J19: the FEM experience with `Mesh::coordinates()`
will inform their design.

---

## 14. J30 — Niche Frameworks (v0.30)

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

- Depends on `oxiflow = "0.30"` (or higher), the first version with INV-4.
- Preserves the `NOTICE` file from the engine in any redistribution (Apache 2.0 requirement).
- Uses a compatible license (Apache 2.0 recommended; any OSI-approved license accepted).
- Uses the `oxiflow-` prefix on crates.io for discoverability.
- Opens a PR against the engine repository to be added to the
  [Known Ecosystem Frameworks](#15-known-ecosystem-frameworks) list below.

---

## 15. Known Ecosystem Frameworks

| Crate | Domain | Maintainer | Status |
|---|---|---|---|
| `oxiflow-chrom` | Chromatography | oxiflow core team | Planned v0.30.0 |
| `oxiflow-geo` | Surface geophysics | oxiflow core team | Planned v0.30.0 |
| `oxiflow-thermo` | Heat transfer | oxiflow core team | Planned v0.30.0 |
| `oxiflow-em` | Diffusive EM | oxiflow core team | Planned v0.30.0 |

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
| Linear solvers (dense) | `nalgebra` delegation | Custom implementation | J4 | |
| Temporal composition | `SplitOperator`/`OperatorSplittingSolver` (Strang) | Fixed explicit/implicit pair | J4 | |
| Spatial operators | Abstract `DiscreteOperator` (associated type `MeshType`) | FD hardcoded | J5 | INV-2 |
| Spatial F/S composition | `DiscretizedModel<Op>` (F) + existing `CompositeModel` (F+S) | New `SourceTerm` trait; flux exposed via `ContextVariable` | J5 | INV-2 |
| Operator context access | `FluxDivergenceOperator`: sibling trait to `DiscreteOperator`, carries `ctx`/`RequiresContext` | Adding `ctx` directly to `DiscreteOperator`; subtrait of `DiscreteOperator` | J5 | INV-2 |
| Linear solvers (sparse) | `faer-sparse` delegation | Custom implementation | J6 | |
| Results export | VTK interop pivot + HDF5 for bulk data | Custom format | J6 | |
| Nonlinear integration | Iterated Newton, [DD-033](https://github.com/biface/oxiflow/issues/86) extension point activated by [DD-044](https://github.com/biface/oxiflow/issues/109) (generic loop, no per-family duplication) | Rewrite of J4 solvers; per-family duplicated loop | J7 | |
| GPU readiness | Structural invariants formalised before `gpu` feature | `wgpu` without prior constraints | J8 | |
| Parallelism | Rayon, opt-in feature flag | Mandatory or absent | J9 | |
| Caching | Dirty flag + temporal invalidation | Systematic recomputation | J9 | |
| API stability | SemVer + `cargo-semver-checks` + FEM audit | Informal convention | J10 | |
| Ergonomics | `oxiflow-prelude`, separate crate | Builder integrated into the engine | J20 (on hold) | |
| Plugin architecture | Object-safe traits + `PluginRegistry` | Monolithic crate | J30 | INV-4 |
| Framework config | TOML + runtime registry | proc-macro DSL | J30 | |
| License | Apache 2.0 only | MIT or dual MIT/Apache | J0 | |

---

## 17. Risk Register

| ID | Risk | Probability | Mitigation |
|---|---|---|---|
| R1 | `ContextValue` generics too complex for users | Medium | Ergonomic helpers; user testing at v0.2 |
| R2 | Silent dependency ordering bugs | Low | Exhaustive cycle detection tests; debug logging |
| R3 | `PhysicalQuantity` indexing too verbose | Medium | Idiomatic constructors; UX feedback before 1.0.0 |
| R4 | Implicit solvers require heavy linear algebra | High | Delegate to `faer`/`nalgebra`; document limits |
| R5 | Rayon + potential `unsafe` | Low | Opt-in flag; ThreadSanitizer in CI |
| R6 | Scope too ambitious | Medium | Each milestone independently deliverable |
| R7 | Breaking change forced before 1.0.0 | Low | Accepted pre-1.0 but documented |
| R8 | INV-1/2/3 silently violated | Medium | Formal audit at J10; dedicated integration tests |
| R9 | ALE incompatible with CouplingOperator design | Low | Proto lahar–lake at J3 is the test bench |
| R10 | INV-4 violated — third-party frameworks break on engine update | Medium | `oxiflow-test-plugin` external crate in CI from v0.30.0; `cargo-semver-checks` in release pipeline |
| R11 | Fragmentation — incompatible third-party frameworks | Low | INV-4 + stable public API is the only compatibility contract; framework authors are responsible for their own SemVer |
| R12 | FV/WENO insertion point ([DA-038](https://github.com/biface/oxiflow/issues/92)) chosen wrong, costly to undo | Low | Internal calculator kept private (reserved `instance_id` field) — additive extension toward J6, no restructuring |

---

## 18. Timeline

Dates live in the §2 table, taken from the GitHub milestones: publication date of a published
milestone, due date of the others. They are not repeated here, so that the two tables cannot
drift apart. Third-party frameworks follow J30 (v0.30.0) on crates.io, with no fixed date.

---

*oxiflow Development Program · document version 3.0 · October 2026 · Living document — updated at each milestone*
