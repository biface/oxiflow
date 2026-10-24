# Changelog

All notable changes to oxiflow are documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
oxiflow adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

### Added

- `SimulationResult::write_vtk` exports a `VectorField` final state (n nodes
  × d components) as well as a `ScalarField`: one scalar `Float64` point-data
  array per component, named `field_0` … `field_{d-1}`, the first one being
  the active scalars. Each component gets its own colour map in ParaView,
  values stay bit-for-bit (NaN and infinities included), and the mesh is
  still 1D. `ScalarField` output is byte-for-byte unchanged (#193).
- With the `parallel` feature, `BackwardEulerSolver`, `CrankNicolsonSolver`
  and `BDF2Solver` assemble their finite-difference Jacobians through Rayon:
  one task per column on the dense path, one per colour on the banded path.
  Each solver has its own dispatch threshold, in unknowns (nodes ×
  components), configured with `with_parallel_threshold` or
  `set_parallel_threshold` and read with `parallel_threshold`. The default,
  256, is indicative: measure the crossover on your workload. Sequential and
  parallel assembly give bit-for-bit identical Jacobians, and results are
  unchanged (#199).
- `benches/oxiflow_parallel`: an `implicit_jacobian_dispatch` diagnostic for
  `diffusion_1d` (`ScalarField`) and `langmuir_multi` (`VectorField`). It
  times one Backward Euler step with the Jacobian assembly forced sequential
  or parallel, on the dense path and, with the `sparse` feature, on the
  banded path, so that the solvers' dispatch threshold can be measured
  locally (#199).
- `ContextCalculator::compute_shared`, a provided method returning
  `Arc<ContextValue>`. The solver now calls it at every derivative
  evaluation. By default it wraps `compute` in an `Arc`, so existing
  calculators are unaffected. A calculator that keeps its value between
  evaluations overrides it to return a clone of the `Arc` it keeps, and every
  evaluation then gets the same allocation instead of a copy. An override
  must return the same value as `compute` (#196).
- `CoefficientField` (`context::calculators`), a coefficient c(x) or c(x, t)
  evaluated on every mesh node from `Mesh::coordinates()`, in any dimension,
  and published as a `ScalarField` under the new variant
  `ContextVariable::Coefficient { name }` (DD-066). `CoefficientField::of_x`
  evaluates c(x) once at construction; `CoefficientField::of_x_t` evaluates
  c(x, t) at the context's time and reuses the field for further evaluations
  at the same time. Both hand the solver the field they keep, through
  `compute_shared`, instead of a copy. A non-finite value is reported as
  `ExternalData`, naming the first node, its coordinates and the time. With
  the `parallel` feature, nodes are evaluated through Rayon above a
  threshold set with `with_parallel_threshold` or `set_parallel_threshold`;
  the default, 10 000 nodes, is indicative. The function must be pure
  (#182).
- `BoundaryCondition::constrained_nodes`, a provided method returning the
  nodes `apply` sets (empty by default). The solver sets the derivative to
  zero at these nodes, so the integrators and the Newton iterations leave
  them to the condition; their rows of the implicit system become identity
  rows. `DanckwertsOutlet` declares its last node (DD-070, #201).

### Changed

- Boundary conditions are applied before the calculators that read the
  state: at each derivative evaluation, the calculators providing the
  variables the conditions require run first, then the conditions, then the
  other calculators and `compute_physics`. A gradient or Laplacian no longer
  sees a boundary value that drifted since the last evaluation; with
  boundary conditions, results change accordingly (DD-070, #201).
- The solver drivers (fixed-step loop, DoPri45, operator splitting,
  `MultiDomainOrchestrator`) apply the boundary conditions to the initial
  state and to the state at the end of each step, before checking and saving
  it: every saved state satisfies them, including a value that depends on
  time. `SteppableSolver::step` itself does not (DD-070, #201).
- `DanckwertsInlet` sets the inlet node from its neighbour,
  `u0 = (u_feed + κ u1) / (1 + κ)` with `κ = D / (v (x1 − x0))`, the
  one-sided discretisation of the Robin condition solved for `u0`. It now
  requires the feed variable only, no longer the spatial gradient, declares
  node 0, and refuses a mesh with fewer than two nodes or whose node 1 is not
  downstream of node 0. Its values change accordingly (DD-070, #201).
- `ComputeContext` stores its values behind `Arc`. `insert` and the typed
  accessors keep their signatures and still return references, so models
  read the context as before (#196).
- The finite-difference Jacobians of the implicit methods (dense and
  banded) allocate one work state per Jacobian and perturb it in place,
  instead of a new copy per column or colour plus, for a `VectorField`
  state, two transposing copies (one to build the perturbed state, one to
  read the derivative). The Jacobians are bit-for-bit unchanged (#197).
- The banded finite-difference Jacobian (`sparse` feature) is assembled
  directly in compressed-column form: the band structure is built from the
  system size and the bandwidth, and each column is written once in its
  final place. The θ-method folds `I − θΔt·J` into it in place. The
  intermediate list of entries, its conversion to faer triplets and the
  sort done by `SparseColMat::try_new_from_triplets` are gone. Matrices and
  steps are bit-for-bit unchanged (#199).
- `Cargo.toml` sets `autobenches = false`: `oxiflow_parallel` is the only
  bench target, as intended. Cargo no longer also builds each module of
  `benches/` as an empty standalone target (#199).
- **Breaking:** `UniformGrid1D::new` returns `Result<Self, OxiflowError>`
  (`OxiflowError::PreconditionFailed`, context `"UniformGrid1D"`) instead of
  `Result<Self, String>`. `?` now works directly in functions returning
  `OxiflowError`; the `.map_err(OxiflowError::InvalidDomain)` workaround is
  gone from `examples/lahar_multifield.rs` and its test.
- Each reference model of the examples, tests and benchmarks (diffusion,
  viscous Burgers, multi-compound Langmuir, multi-field lahar, lahar–lake)
  is defined once under `rsc/models/` and included by every target that
  uses it, instead of up to four copies. Example outputs and test
  tolerances are unchanged. The `langmuir_multi_plot` example gets back the
  parallel branch its copy had lost. The dispatch diagnostics of the
  Langmuir and lahar benchmarks now time the models' own `compute_physics`
  and `apply` with the threshold forced, instead of partial copies of their
  loops; their default thresholds stay as measured at v0.9.0. The crate
  itself and its public API are unchanged (#194).
- Documentation: `ContextValue::VectorField` holds `d` values per node,
  either vector components or `d` unknowns unrelated to the mesh dimension;
  the default dispatch threshold of the finite-difference operators
  (49 999) is documented as indicative (#140, #194).

### Fixed

- `DanckwertsInlet` diverged as soon as `κ = D / (v Δx) > 1`, i.e.
  `(n − 1) / Pe > 1` on a uniform grid with `Pe = vL/D`: it set `u0` from
  the gradient at node 0, computed from the previous `u0`, which iterated
  `u0` with a factor `−κ` from one evaluation to the next. The inlet value
  no longer depends on its previous value (#201).
- `UniformGrid1D::new` rejects non-finite bounds: a NaN bound used to pass
  the `x_end > x_start` check and yield a grid of NaN nodes.
- `FDGradientCalculator` and `FDLaplacianCalculator` return
  `OxiflowError::PreconditionFailed` from `compute()` on a mesh their
  stencils cannot handle, instead of a silently wrong field: a mesh that is
  not 1D, a 1D mesh whose node spacing is not uniform or differs from
  `characteristic_length()` (relative tolerance `1e-6`), and, for the
  gradient, a `dimension` beyond the mesh dimension. The check runs once, at
  construction; `new()` keeps its signature and results on `UniformGrid1D`
  are unchanged (#186).
- `DanckwertsInlet::apply` and `DanckwertsOutlet::apply` return
  `OxiflowError::PreconditionFailed` on a mesh that is not 1D: both locate
  the boundary by node index (`0` and `n − 1`), which only an ordered 1D grid
  makes meaningful (#187).
- Divergence detection covers every numeric state variant. A NaN or an
  infinity in a `VectorField` state used to go unnoticed: the run ended with
  `Ok` and non-finite states, and `Solver::on_divergence` was never called.
  It now raises `OxiflowError::SolverDivergence` with a snapshot, as for a
  `ScalarField` (#192).
- `OperatorSplittingSolver::solve` hands a `SimulationSnapshot` to
  `Solver::on_divergence` before returning `SolverDivergence`, like every
  other solver. The snapshot is built by one shared helper in all solver
  loops (#134).
- Time integrators advance `VectorField` states (n nodes × d components)
  as they do `ScalarField` states, instead of failing with `TypeMismatch`:
  `RK4Solver`, `DoPri45Solver` (error norm over every component),
  `BackwardEulerSolver` and `CrankNicolsonSolver`, dense and banded paths,
  `BDF2Solver`, and `OperatorSplittingSolver` through its sub-solvers.
  The implicit solvers order the unknowns of a `VectorField` by node, so
  `with_jacobian_bandwidth` takes `d·(s + 1) − 1` for `d` components and a
  stencil of half-width `s`, and `with_sparse_threshold` counts unknowns
  (nodes × components). `ScalarField` results are bit-identical. Boundary
  conditions applied through `BoundaryCondition::apply` still take a single
  field: a `VectorField` state with conditions attached is refused with
  `TypeMismatch`, as documented; its boundaries go through `ghost_value()`
  (#192).

### Security

- The `vtk` feature no longer depends on `vtkio`. `SimulationResult::write_vtk`
  now writes the `.vtu` file itself, with an unchanged signature and the same
  inline `binary` (base64) data arrays as before, so NaN and infinities in a
  diverged state are still exported bit for bit. This removes
  RUSTSEC-2026-0041 (`lz4_flex 0.7.5`), RUSTSEC-2026-0194 and
  RUSTSEC-2026-0195 (`quick-xml 0.22.0`) from the dependency tree, along with
  the future-incompatibility warning `nom 3.2.1` raised on builds with `vtk`
  (#191).

---

## [0.9.0] - 2026-09-16

### Added

- `parallel` feature flag (Rayon, optional) — per-site dispatch calibration
  rather than a single crate-wide threshold ([DD-048](https://github.com/biface/oxiflow/issues/140)): `fd.rs`'s
  `default_parallel_threshold()` (`49_999`, [DD-014](https://github.com/biface/oxiflow/issues/14), bare-stencil profile),
  `langmuir_multi`'s measured `1_000` (#138), `lahar_multifield`'s two
  independent sites — `default_source_parallel_threshold()` (`1_000`,
  friction/erosion loop) and `default_flux_parallel_threshold()`
  (`10_000`, flux divergence loop, set above an unstable crossover region
  rather than at its lower bound) ([DD-048](https://github.com/biface/oxiflow/issues/140), #139)
- `ParallelThreshold`/`calibrate_parallel_threshold` (`src/solver/parallel.rs`)
  — shared `AtomicUsize`-backed mechanism for future `src/`-internal
  dispatch sites; `pub(crate)`, not usable from `examples/`/`tests/`/
  `benches/` ([DD-048](https://github.com/biface/oxiflow/issues/140))
- `benches/oxiflow_parallel` — Criterion suite (#53) covering four
  benchmark models: `diffusion_1d` (#136), `viscous_burgers` (#137),
  `langmuir_multi` (#138), `lahar_multifield` (#139); full write-up in
  `BENCHMARKS.md`
- `examples/lahar_multifield.rs` — multi-field lahar model ([DD-032](https://github.com/biface/oxiflow/issues/85), #139):
  Woodhouse/Hogg/Phillips shallow-layer equations reduced to 1D, four
  coupled conservative quantities `(m, p, s, b)` including bed elevation,
  Rusanov flux with MinMod-limited MUSCL reconstruction
  (`ShallowLayerFlux`), full Pouliquen-Forterre + Chezy blended friction,
  Shields-threshold erosion, Soulsby/Richardson-Zaki hindered-settling
  deposition (`LaharSource`); validated against three independent
  regimes (early inertial, Chezy-drag-dominated, constant-Coulomb
  slump/stall pattern) in `tests/lahar_multifield_analytical.rs`
- `examples/{diffusion_1d,viscous_burgers,langmuir_multi,lahar_multifield}_plot.rs`
  — SVG reference plots for all four benchmark models, matching the
  literature figures / chrom-rs's own chromatogram output each model is
  validated against (#145)
- `plotters` dev-dependency for the above

### Changed

- Criterion bench target renamed `oxiflow_benchmarks` → `oxiflow_parallel`
  — named for its actual object (parallel-dispatch calibration), since
  other `[[bench]]` targets for other purposes are expected once 2D/3D
  work starts

### Not implemented

- Dirty-flag caching for repeated `ContextCalculator` evaluations
  ([DD-015](https://github.com/biface/oxiflow/issues/15), #52) — closed without implementation this sprint; scope
  parked, not delivered

---

## [0.8.0] - 2026-08-03

### Added

- `gpu` feature flag (`wgpu 26.0.1`, `pollster`, `bytemuck`, all optional) —
  `solver::gpu::GpuContext::new()` acquires a GPU adapter/device/queue,
  blocking via `pollster`; `to_gpu_bytes()` reinterprets `ContextValue`'s
  contiguous `f64` payloads as raw bytes via `bytemuck::cast_slice`, without
  requiring `ContextValue` itself to implement `bytemuck::Pod` (the enum
  discriminant remains the blocker for that, deferred) (DD-026, DD-045,
  #74)
- `OxiflowError::GpuUnavailable` — returned when no compatible GPU adapter
  or device is found; never a silent CPU fallback, the caller chooses its
  own CPU solver (DD-045)
- CI: `clippy --features gpu` and `test --features gpu` jobs added to
  `ci.yml` (#122)

### Changed

- **MSRV raised `1.80` → `1.84`** — no stable `wgpu` release declares
  compatibility with `1.80`; `1.84` is the lowest MSRV declared by `wgpu`
  26.0.1, the version selected (DD-045)
- `ContextValue`'s `GPU-READY` annotation (DD-026 INV-GPU-5) relocated onto
  `ContextValue` itself (previously misapplied to `MultiDomainState`, which
  can never be `bytemuck::Pod` — a `HashMap` has no fixed, Pod-compatible
  layout); column-major layout (INV-GPU-4) documented directly on
  `ContextValue::Matrix`/`ContextValue::VectorField` (#127)
- `MultiDomainState` module docs corrected: requalified as an
  indexed/orchestration container, out of INV-GPU-1's scope, rather than
  claiming to satisfy it (#127)
- Rustdoc intra-doc links and DD citation style normalised across the
  crate; `cargo doc --no-deps --all-features` now emits **zero warnings**
  (validated by a real run) (#122)

### Known gaps

- `derive(Pod, Zeroable)` on `ContextValue` remains blocked by its enum
  discriminant — resolution deferred past this release, not part of the
  `gpu` feature's initial scope (DD-045)
- No real-GPU integration test runs in CI (no GPU hardware available);
  the one that exists is `#[ignore]`d (#74)
- `BDF2Solver` still has no sparse builders (`sparse_solver`,
  `sparse_threshold`, `jacobian_bandwidth` absent) — a DD-043 gap,
  unaddressed by this release, carried forward to v0.9.0
- `OperatorSplittingSolver::solve` (IMEX) does not call
  `self.on_divergence()` / build a `SimulationSnapshot` on divergence,
  unlike every other solver in the crate — discovered this release,
  carried forward to v0.9.0

## [0.7.0] - 2026-07-30

### Added

- Dedicated test for the perturbed-Jacobian × Dirichlet boundary-condition
  interaction, closing a known-untested gap documented since DD-033
  (v0.4.0): the constrained node's column in `finite_difference_jacobian`
  is exactly zero, while its row keeps genuine physics-based coupling to
  unconstrained neighbours (#113)
- `IntegratorSpec::BDF2` variant — Newton fields only, deliberately no
  sparse fields (`BDF2Solver` has no sparse builders to map, a DD-043 gap
  distinct from this DTO increment) (DD-044, #116)
- `IntegratorSpec::CrankNicolson` variant, symmetric to `BackwardEuler` in
  every field (sparse dispatch + Newton configuration); `theta = 0.5`
  stays implicit to the variant, matching `BackwardEuler`'s own
  `theta = 1.0` (DD-044, #115)
- `IntegratorSpec::BackwardEuler` gains `newton_convergence`,
  `jacobian_strategy`, `max_iterations` fields (`serde` feature), mapped to
  `BackwardEulerSolver`'s three Newton builders via `TryFrom`; defaults
  match the solver's own zero-config values (DD-044, #114)
- `NewtonConvergence`/`JacobianStrategy` gain `serde::Deserialize` under
  the `serde` feature (#114)
- `BDF2Solver` gains `with_newton_convergence()`, `with_jacobian_strategy()`,
  `with_max_newton_iterations()` builders; `bdf2_step_newton()`
  (`solver::methods::bdf2`) routes through the generic Newton loop with
  caller-supplied configuration — defaults reproduce the pre-DD-044
  single-correction behaviour exactly on affine problems. The
  startup/bootstrap step is unaffected: it always uses
  `theta_method_step()`'s non-configurable path (DD-044, #109, #112)
- `BackwardEulerSolver`/`CrankNicolsonSolver` gain `with_newton_convergence()`,
  `with_jacobian_strategy()`, `with_max_newton_iterations()` builders;
  `theta_method_step_newton()` (`solver::methods::implicit`) routes through
  the generic Newton loop with caller-supplied configuration — defaults
  reproduce the pre-DD-044 single-correction behaviour exactly on affine
  problems (DD-044, #109, #111)
- Iterated Newton solver core (`solver::methods::newton`): `NewtonConvergence`
  (`ResidualOnly`/`CombinedOr`/`CombinedAnd`), `JacobianStrategy`
  (`ModifiedFrozen`/`FullNewton`/`ModifiedWithStagnationCheck`), and the
  generic convergence loop shared by theta-method and BDF2 (DD-044, #109,
  #110)
- `OxiflowError::NewtonNotConverged { iterations, residual }` — routed
  through `Solver::on_divergence()` once wired (#110)

### Changed

- Introduced `stiff_jacobian_convergence()` (`solver::methods::implicit`,
  `tol_abs = 1e-5`), used by `theta_method_step()`/`bdf2_step()` and by
  `BackwardEulerSolver`/`CrankNicolsonSolver`/`BDF2Solver`'s zero-config
  constructors — distinct from `NewtonConvergence::default()` (unchanged,
  `tol_abs = 1e-8`). The existing very-stiff (`λ = 1e4`) affine regression
  tests' single Newton correction is exact, but its finite-difference
  Jacobian carries catastrophic-cancellation error of that order — a
  precision floor, not a nonlinearity signal; zero-config solver
  construction needs the looser value to keep reproducing history exactly,
  without loosening what `NewtonConvergence::default()` means for callers
  configuring the loop themselves (#111, #112)

### Known gaps

- `theta_method_step_adaptive` (sparse/banded dispatch, DD-043) is not yet
  Newton-aware: configuring both a sparse backend and a non-default Newton
  budget on `BackwardEulerSolver`/`CrankNicolsonSolver` silently keeps the
  sparse path single-shot (#111)

## [0.6.0] — 2026-07-21

### Highlights

Sparse linear algebra and persistence: a `faer`-backed sparse solver path for
implicit integrators (`BackwardEulerSolver`, `CrankNicolsonSolver`), a
crash/checkpoint snapshot mechanism, HDF5-backed loading of experimental
tabulated data, XML VTK export for ParaView/VisIt, and the first
config-driven integrator selection DTO (`IntegratorSpec`).

### Added

- `faer`-backed sparse linear solver path (`SparseLinearSolver`,
  `FaerSparseSolver`) for `BackwardEulerSolver`/`CrankNicolsonSolver`,
  feature-gated `sparse` (DD-013, DD-043, #50)
- `SimulationSnapshot`, `SnapshotFormat`, `write_snapshot()`/`read_snapshot()`
  (JSON/bincode) — `src/solver/snapshot.rs` (DD-025 Option B / DD-029, #71)
- `Solver::on_divergence()` — default no-op hook, called automatically before
  `OxiflowError::SolverDivergence` is returned (#71)
- `OxiflowError::Persistence` — snapshot/HDF5/VTK I/O and (de)serialisation
  failures
- `ExternalTabulated::from_hdf5()` — loads tabulated data from an HDF5 file,
  feature-gated `hdf5` (DD-027, #105)
- `SimulationResult::write_vtk()` — XML `.vtu` export for ParaView/VisIt,
  feature-gated `vtk` (DD-027, #78)
- `IntegratorSpec` — external config DTO for `BackwardEulerSolver` +
  `TryFrom<IntegratorSpec> for Box<dyn Solver>`, feature-gated `sparse`
  (DD-027 amendment 1, #104)

### Changed

- `SimulationResult` now derives `Clone`
- `hdf5` feature migrated from the unmaintained `hdf5` crate (0.8, aldanor)
  to the actively-maintained `hdf5-metno` fork (chore, #79)

### Design Decisions

- DD-027 — Results export: VTK as interop pivot, HDF5 for bulk datasets (#75)
- DD-027 amendment 1 — config-driven WHAT/HOW construction from JSON/YAML
- DD-029 — Generalised `SimulationSnapshot`: normal checkpoint/resume (#77)
- DD-043 — `SparseLinearSolver`: sibling trait to `DiscreteOperator` (faer)

## [0.5.0] — 2026-07-17

### Added

- `DiscreteOperator` trait — `operators::fd::{UpwindGradient, CenteredGradient,
  CenteredLaplacian}` (#46, #47, DD-012)
- `FluxDivergenceOperator` trait, sibling to `DiscreteOperator` — computes the full
  `∇·F(u, ∇u)` term, needs physical parameters `DiscreteOperator` deliberately does not
  provide (#97, DD-039)
- `DiscretizedModel<Op: FluxDivergenceOperator>` composite — plugs any
  `FluxDivergenceOperator` into `PhysicalModel` (#93, DD-038)
- `operators::fv::{FVCenteredFlux, FVUpwindFlux}` — finite-volume advection/diffusion,
  cell-average semantics (#48, DD-040)
- `operators::weno::{WENO3, WENO5}` — Jiang-Shu substencils, WENO-Z nonlinear weighting
  (Borges et al. 2008) for critical-point accuracy (#49, DD-041)
- `FluxBoundary` enum — explicit boundary-treatment choice per operator:
  - `Periodic` — exact conservation via telescoping (#48, #49)
  - `Truncation` — one-sided decentered fallback at boundary cells, breaks exact
    conservation (documented trade-off) (DD-042)
  - `GhostCell` — real `BoundaryCondition` physics via `ghost_value(depth,
    interior_at_depth, dx)`, method-of-images convention, exact at every depth for a
    Robin condition (DD-042, amendment 1)
- `BoundaryCondition::ghost_value(depth, interior_at_depth, dx) -> Option<f64>` — default
  method, `None` by default, non-breaking (DD-042)
- `operators::limiters::{Limiter, LimitedFlux}` — `MinMod`/`VanLeer`/`Superbee`, MUSCL
  reconstruction (#99)
- `operators::adaptive::AdaptiveFlux` — blends `WENO3`/`LimitedFlux` per face, gated by a
  global mesh Péclet number, driven by `WENO3`'s own smoothness indicator (#99)
- `operators::check_cfl` — shared explicit-advection CFL check (FV, WENO, limiters,
  adaptive)

### Changed

- `FDGradientCalculator`/`FDLaplacianCalculator` delegate their stencil math to
  `operators::fd` instead of hand-rolled loops (#94, DD-012)
- `FluxBoundary` drops `Copy`/`PartialEq`/`Eq` (`Arc<dyn BoundaryCondition>` has neither) —
  `FVCenteredFlux`/`FVUpwindFlux`/`WENO3`/`WENO5` drop `Copy` accordingly
- Wide-stencil divergence helpers (`periodic_wide_divergence`, `truncated_wide_divergence`,
  `ghost_padded_field`, `ghost_cell_wide_divergence`, `wrap`) relocated from
  `operators::weno` to `operators` (`pub(crate)`) — shared by `operators::limiters` and
  `operators::adaptive`

### Fixed

- No user-facing bug in this release; two internal design errors were caught and
  corrected before merge (see `DESIGN_DECISIONS.md` for the full account, not silently
  fixed): WENO's initial nodal-semantics claim (DD-041), and `FluxBoundary::GhostCell`'s
  initial single-point `ghost_value` signature being insufficient for WENO's wider
  stencils (DD-042, amendment 1)

## [0.4.0] — 2026-06-25

### Added

- `SteppableSolver` trait in `src/solver/methods/mod.rs` — single-step `step()` primitive,
  extracted from existing `ForwardEulerSolver`/`RK4Solver` logic with no behaviour change
  (Ref #82, DD-031)
- `RK4Solver` — explicit, 4-stage Runge-Kutta, order 4 (#41)
- `MultiDomainOrchestrator` — advances each `Domain` with its own `SteppableSolver` and
  invokes the `Scenario`'s registered `CouplingOperator`s between steps; lets a coupled
  scenario mix integrators per domain (partitioned coupling) (Ref #82, DD-031)
- Integration test for the lahar–lake prototype — first real time loop over coupled domains
  via `MultiDomainOrchestrator`; validates orchestration only, not lahar physics
  (#40, Ref #82, DD-031)
- `LinearSolver` trait + `NalgebraDenseSolver` (dense LU) default implementation (#13, DD-013)
- `finite_difference_jacobian`, `theta_method_step` in `src/solver/methods/implicit.rs` —
  shared by `BackwardEulerSolver` (θ=1) and `CrankNicolsonSolver` (θ=0.5)
  (#43, Refs #13, #86, DD-013, DD-033)
- `BackwardEulerSolver` — implicit, 1st order (#43, Ref #86, DD-033)
- `CrankNicolsonSolver` — semi-implicit, 2nd order (#43, Ref #86, DD-033)
- `SteppableSolver::history_depth()` and a `history: &[ContextValue]` parameter on `step()` —
  supports multi-step integrators; defaults to `0`, no change for one-step solvers
  (Ref #87, DD-034)
- `BDF2Solver` — implicit multi-step, 2nd order; starts with one Backward Euler step when
  history is empty (#44, Ref #87, DD-034)
- `SteppableSolver::solve_fixed_step()` default method — factors out the fixed-step loop
  previously duplicated near-verbatim across `ForwardEulerSolver`, `RK4Solver`,
  `BackwardEulerSolver`, `CrankNicolsonSolver`, `BDF2Solver` (Ref #88, DD-035)
- `StepControl::Adaptive` activated (`dt_init`, `dt_min`, `dt_max`, `rtol`, `atol`) — reserved
  since DD-021, v0.1.0, activated alongside DoPri45 (Ref #89, DD-036)
- `StepSizeController` — PI step-size controller with a source-agnostic error norm, shared
  across adaptive integrators (Ref #89, DD-036)
- `DoPri45Solver` — Dormand-Prince, 7-stage FSAL, adaptive step, order 5; `Solver` only, not
  `SteppableSolver` (#42, Ref #89, DD-036)
- `OperatorSplittingSolver`, `SplitOperator`, `SplittingScheme` in
  `src/solver/methods/imex.rs` — n ≥ 2 operator splitting integrator (Strang scheme); `Solver`
  only, not `SteppableSolver` (#45, DD-037)
- `CompositeModel` in `src/model/composite.rs` — sums multiple `PhysicalModel` contributions
  on the same state; used as the outer `Domain`'s bookkeeping model for IMEX scenarios and as
  a monolithic reference solution (#45, DD-037)
- `IntegratorKind::Imex` (#45, DD-037)

### Changed

- `SteppableSolver::step()` gains a `history: &[ContextValue]` parameter on every
  implementor — no behaviour change for one-step solvers, which ignore it (Ref #87, DD-034)
- `Solver::solve()` on `ForwardEulerSolver`, `RK4Solver`, `BackwardEulerSolver`,
  `CrankNicolsonSolver`, `BDF2Solver` reduced to a single call to `solve_fixed_step()` (Ref #88, DD-035)
- `BoundaryCondition` now requires `Send + Sync` supertraits — additive; existing
  implementations (`DanckwertsInlet`, `DanckwertsOutlet`) already satisfy it trivially.
  Required because `SplitOperator` is the first place a `Domain` is owned rather than
  borrowed by a `Send + Sync`-bound type (`Solver: Send + Sync`) (#45, DD-037)

### Fixed

- `ForwardEulerSolver` — boundary-condition handling correction
- `RK4Solver` — boundary conditions now applied at every Runge-Kutta stage, not once per
  outer step; floating-point accumulation error in the multi-stage state combination

## [0.3.0] — 2026-06-03

### Added

- `PhysicalQuantity` indexed enum in `src/context/quantity.rs` — variants
  `Concentration { component }`, `Temperature`, `Pressure`, `Velocity { component }`,
  `Custom { name: Cow<'static, str>, component }` with idiomatic constructors
  (`concentration()`, `temperature()`, `pressure()`, `velocity()`, `custom(name)`) (DD-010, #10)
- `MultiDomainState` in `src/context/state.rs` — `HashMap<(DomainId, PhysicalQuantity), ContextValue>`
  composite key; explicit entry serialisation for external tool readability (DD-010, #38)
- `Interface` struct in `src/coupling/mod.rs` — source domain, target domain, optional label (DD-011, #11)
- `CouplingOperator` trait in `src/coupling/mod.rs` — `RequiresContext + Send + Sync` supertraits,
  object-safe, INV-3 active from v0.3.0 (DD-011, #11)
- `Scenario::with_coupling()` builder — registers a `CouplingOperator` and its `Interface` (#73)
- `Scenario::couplings()`, `Scenario::interfaces()`, `Scenario::n_couplings()` accessors (#73)
- `Scenario::context_requirements()` now aggregates variables from coupling operators (J3, #73)
- `serde_json` in `[dev-dependencies]` for serde feature tests (#69)
- Integration test `tests/coupling_proto.rs` — proto lahar–lake validates INV-3 structural
  contract (J3 exit criterion, #73)

### Changed

- `src/context/mod.rs` — `quantity` and `state` modules declared and re-exported

### Feature flags

- `serde` — `PhysicalQuantity` and `MultiDomainState` annotated with
  `#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]` (#69)

## [0.2.0] — 2026-05-02

### Added

- `BoundaryCondition` trait requires `RequiresContext + Debug` supertraits (DD-008)
- `BoundaryLocation` enum with nine variants including `PhaseInterface`, `CouplingInterface`,
  `Custom(Cow<'static, str>)` (DD-008)
- Calculator chain with hybrid two-path ordering: priority fast path + Kahn topological sort (DD-009)
- Built-in calculator suite in `src/context/calculators/`: `TimeCalculator`, `TimeStepCalculator`,
  `FDGradientCalculator`, `FDLaplacianCalculator`, `TrapezoidalIntegral`, `ExternalTabulated` (DD-009)
- `FDScheme` and `Interpolation` enums
- `ContextVariable::External::name` migrated from `&'static str` to `Cow<'static, str>`
- `OxiflowError::PreconditionFailed { context, message }` variant
- `SimulationSnapshot` two-phase restore contract (DD-025)
- Danckwerts inlet feed concentration read from `ComputeContext` via `ContextVariable`
- Integration test `tests/solver_pipeline_chromatography.rs`

## [0.1.0] — 2026-04-12

### Added

- `ContextValue` generic enum: `Scalar`, `Vector`, `Matrix`, `Field2D`, `Boolean`
- `OxiflowError` typed error enum (DD-004)
- `RequiresContext` trait — declarative variable dependency system
- `ComputeContext` — type-safe context access (DD-005, DD-006)
- `Mesh` abstract trait — INV-1 active (DD-003)
- `UniformGrid1D` — structured 1D mesh implementation
- `PhysicalModel` trait — WHAT/HOW separation
- `Scenario` / `SolverConfiguration` / `Solver` architecture
- `DomainId` typed domain identifier
- `ContextVariable` typed key enum

## [0.0.5] — 2026-04-15

### Added

- `Cargo.toml`: full project structure — dependencies (`nalgebra`, `thiserror`),
  features skeleton (`parallel`, `serde`, `hdf5`), crates.io metadata, MSRV 1.80 (#20)
- `src/`: module skeleton — `context`, `mesh`, `model`, `boundary`,
  `solver`, `operators`, `coupling` (#21)
- CI pipeline `ci.yml`: `cargo fmt --check`, `cargo clippy -D warnings`,
  `cargo test`, `cargo doc` → GitHub Pages (#23)
- `.codecov.yml`: 80% coverage threshold, `default` flag active (#23)
- `CHANGELOG.md`: initial Keep a Changelog entry (#23)

## [0.0.1] — 2026-02-15

### Added

- Reserved `oxiflow` crate name on crates.io
- Apache 2.0 license (`LICENSE`, `NOTICE`)
- Initial `README.md`
- CI infrastructure: GitHub → GitLab mirror (`mirror.yml`, `mirror-patch.yml`),
  coverage (`coverage.yml`)

---

[Unreleased]: https://github.com/biface/oxiflow/compare/v0.5.0...HEAD
[0.5.0]: https://github.com/biface/oxiflow/compare/v0.4.0...v0.5.0
[0.4.0]: https://github.com/biface/oxiflow/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/biface/oxiflow/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/biface/oxiflow/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/biface/oxiflow/compare/v0.0.5...v0.1.0
[0.0.5]: https://github.com/biface/oxiflow/compare/v0.0.1...v0.0.5
[0.0.1]: https://github.com/biface/oxiflow/releases/tag/v0.0.1
