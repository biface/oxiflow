# Changelog

All notable changes to oxiflow are documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
oxiflow adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

### Added

- `PhysicalModel::max_wave_speed` returns the largest advective wave speed
  of a model on a state, `None` by default for a model without an advective
  term ([DA-073](https://github.com/biface/oxiflow/issues/229),
  [#228](https://github.com/biface/oxiflow/issues/228)). `DiscretizedModel`
  returns the speed of its scheme, `CompositeModel` the sum of the speeds
  its contributions declare, and `PrescribedInitialState` the speed of the
  model it wraps. A model that wraps another one must forward the method.
- Domains described by data ([#108](https://github.com/biface/oxiflow/issues/108),
  amendment 1 of [DT-027](https://github.com/biface/oxiflow/issues/75)):
  `DomainSpec` holds a mesh, a model, boundary conditions and an optional
  initial state, each given by its own spec, and `Domain::try_from` builds
  the domain, with its mesh built once and shared. The specs are
  `MeshSpec` (`UniformGrid1D`); `ModelSpec` (`Discretized` over a
  `FluxSchemeSpec`: `FVCentered`, `FVUpwind`, `WENO3`, `WENO5`, `Limited`,
  `Adaptive`, closed by a `FluxBoundarySpec`, `Periodic` or `Truncation`);
  `BoundaryConditionSpec` (`Dirichlet`, whose value is a `TimeFunctionSpec`:
  a constant, a `TemporalInjection` profile or a tabulated series;
  `DanckwertsInlet`, `DanckwertsOutlet`, `InjectionInlet`); and
  `InitialStateSpec` (`Uniform`, `Values`, `Dirac`). All are
  `#[non_exhaustive]`. With the `serde` feature they derive `Deserialize`, as
  do `Interpolation`, `Limiter` and `TemporalInjection`, so that a domain
  reads from a configuration file. A domain built from its spec runs
  bit-for-bit as the same domain built in code. Not covered yet:
  `CompositeModel`, `FluxBoundary::GhostCell` and values given as closures;
  functions given as closed-form expressions are planned by
  [DD-072](https://github.com/biface/oxiflow/issues/215).
- `Mesh` is implemented for `Arc<M>`, for any `M: Mesh` including
  `dyn Mesh`. Every method forwards to the shared mesh, those with a default
  implementation included, so a mesh that overrides them keeps its own
  answers. One grid can then serve a flux scheme, which holds an `Arc`, and
  a `Domain`, which owns a `Box<dyn Mesh>`, without being built twice
  ([#108](https://github.com/biface/oxiflow/issues/108)).
- `SimulationResult::write_vtk` exports a `VectorField` final state (n nodes
  × d components) as well as a `ScalarField`: one scalar `Float64` point-data
  array per component, named `field_0` … `field_{d-1}`, the first one being
  the active scalars. Each component gets its own colour map in ParaView,
  values stay bit-for-bit (NaN and infinities included), and the mesh is
  still 1D. `ScalarField` output is byte-for-byte unchanged ([#193](https://github.com/biface/oxiflow/issues/193)).
- With the `parallel` feature, `BackwardEulerSolver`, `CrankNicolsonSolver`
  and `BDF2Solver` assemble their finite-difference Jacobians through Rayon:
  one task per column on the dense path, one per colour on the banded path.
  Each solver has its own dispatch threshold, in unknowns (nodes ×
  components), configured with `with_parallel_threshold` or
  `set_parallel_threshold` and read with `parallel_threshold`. The default,
  256, is indicative: measure the crossover on your workload. Sequential and
  parallel assembly give bit-for-bit identical Jacobians, and results are
  unchanged ([#199](https://github.com/biface/oxiflow/issues/199)).
- `benches/oxiflow_parallel`: an `implicit_jacobian_dispatch` diagnostic for
  `diffusion_1d` (`ScalarField`) and `langmuir_multi` (`VectorField`). It
  times one Backward Euler step with the Jacobian assembly forced sequential
  or parallel, on the dense path and, with the `sparse` feature, on the
  banded path, so that the solvers' dispatch threshold can be measured
  locally ([#199](https://github.com/biface/oxiflow/issues/199)).
- `benches/oxiflow_parallel`: an `orchestrator_phase1` calibration of the
  `MultiDomainOrchestrator` dispatch. It times ten synchronised steps with
  the orchestrator's threshold forced sequential or parallel, and the
  dispatch inside each domain forced sequential, left at its defaults or
  forced parallel, on K independent Langmuir columns (explicit Euler, and
  Crank–Nicolson on the banded path with the `sparse` feature), K
  independent lahar domains and the coupled lahar–lake scenario. Each case is reported against the
  aggregate mesh volume the threshold compares, so that the threshold can
  be calibrated on the host ([#141](https://github.com/biface/oxiflow/issues/141)).
- `ContextCalculator::compute_shared`, a provided method returning
  `Arc<ContextValue>`. The solver now calls it at every derivative
  evaluation. By default it wraps `compute` in an `Arc`, so existing
  calculators are unaffected. A calculator that keeps its value between
  evaluations overrides it to return a clone of the `Arc` it keeps, and every
  evaluation then gets the same allocation instead of a copy. An override
  must return the same value as `compute` ([#196](https://github.com/biface/oxiflow/issues/196)).
- `CoefficientField` (`context::calculators`), a coefficient c(x) or c(x, t)
  evaluated on every mesh node from `Mesh::coordinates()`, in any dimension,
  and published as a `ScalarField` under the new variant
  `ContextVariable::Coefficient { name }` ([DD-066](https://github.com/biface/oxiflow/issues/181)). `CoefficientField::of_x`
  evaluates c(x) once at construction; `CoefficientField::of_x_t` evaluates
  c(x, t) at the context's time and reuses the field for further evaluations
  at the same time. Both hand the solver the field they keep, through
  `compute_shared`, instead of a copy. A non-finite value is reported as
  `ExternalData`, naming the first node, its coordinates and the time. With
  the `parallel` feature, nodes are evaluated through Rayon above a
  threshold set with `with_parallel_threshold` or `set_parallel_threshold`;
  the default, 100 000 nodes, is indicative. The function must be pure
  ([#182](https://github.com/biface/oxiflow/issues/182)).
- `BoundaryCondition::constrained_nodes`, a provided method returning the
  nodes `apply` sets (empty by default). The solver sets the derivative to
  zero at these nodes, so the integrators and the Newton iterations leave
  them to the condition; their rows of the implicit system become identity
  rows. `DanckwertsOutlet` declares its last node ([DD-070](https://github.com/biface/oxiflow/issues/200), [#201](https://github.com/biface/oxiflow/issues/201)).
- Boundary regions in `mesh` ([DA-071](https://github.com/biface/oxiflow/issues/203), [#204](https://github.com/biface/oxiflow/issues/204)): `BoundaryRegion` (one variant
  today, `Face { axis, side }`), `FaceSide` and `BoundaryNode`, with two
  provided methods on `Mesh`. `boundary_nodes` lists the nodes of a region;
  `boundary_neighbours` pairs each with its first neighbour along the inward
  normal and their distance. Their default covers the two faces of a 1D mesh
  whose nodes are ordered along the axis; any other region is refused with
  `PreconditionFailed`.
- `boundary::Dirichlet`, a Dirichlet condition holding every node of a
  boundary region at a prescribed value, constant (`Dirichlet::constant`) or
  a function of time (`Dirichlet::of_t`, evaluated at the context's time).
  It declares the nodes of its region as constrained nodes, so every saved
  state holds the value of its own time, for every integrator. A region the
  mesh does not define is refused with `PreconditionFailed`, a non-finite
  value with `ExternalData`, naming the region and the time. The condition
  is `Clone` (a clone shares the function) and is not serialisable ([#202](https://github.com/biface/oxiflow/issues/202)).
- `benches/oxiflow_parallel`: a `boundary_condition_overhead` measurement
  for `diffusion_1d`. It times full solves without boundary conditions, with
  the two Danckwerts conditions and with two `Dirichlet` conditions, under
  RK4 and Backward Euler on two mesh sizes each, so that the cost of the
  conditions can be measured locally. It does not need the `parallel`
  feature ([#202](https://github.com/biface/oxiflow/issues/202)).
- `examples/heat_transfer_1d`, the heat-transfer reference example: a steel
  bar in physical units, with both ends held by `Dirichlet` conditions and
  the conduction term written as λT″ + λ′T′ from the Laplacian, the centred
  gradient and two `CoefficientField`s. It cools a homogeneous bar and
  checks Crank–Nicolson and BDF2 against the Fourier-series solution, then
  heats a bar whose conductivity doubles along its length and checks
  Backward Euler against the exact steady profile. It exits with an error
  beyond its tolerances and runs in CI as the first reference example
  ([#175](https://github.com/biface/oxiflow/issues/175)).
- `benches/oxiflow_parallel`: a `coefficient_dispatch` diagnostic for
  `CoefficientField`, with the `parallel` feature. It times the evaluation
  of a c(x, t) field forced sequential or parallel, at a new time on every
  call, for the graded conductivity of the heat-transfer example and for a
  costlier two-layer profile, so that the field's dispatch threshold can be
  measured locally ([#175](https://github.com/biface/oxiflow/issues/175)).
- `examples/magnetic_diffusion_1d`, the electromagnetism reference example
  (prototype of the diffusive regime): a field step of 0.1 T at the surface
  of a copper plate and of an aluminium plate, in physical units, with both
  faces held by `Dirichlet` conditions and BDF2 at the solver's default
  Newton budget. It checks the field against the half-space `erfc` solution
  and the half-field penetration depth against its sqrt(ηt) scaling, in both
  metals and at three times, exits with an error beyond its tolerances and
  runs in CI with the reference examples ([#179](https://github.com/biface/oxiflow/issues/179)).
- `examples/consolidation_1d`, the geomechanics reference example: Terzaghi
  consolidation of a clay layer under a 100 kPa load, in physical units,
  with both faces drained by `Dirichlet` conditions and BDF2 at the solver's
  default Newton budget. By symmetry the run also stands for a layer half as
  thick on an impermeable base. It checks the excess pore-pressure profiles
  and the average degree of consolidation U(Tv) against Terzaghi's series,
  truncated with an explicit bound on the neglected terms, exits with an
  error beyond its tolerances and runs in CI with the reference examples
  ([#178](https://github.com/biface/oxiflow/issues/178)).
- `examples/burgers_boundary_layer`, the fluid-mechanics reference example:
  a viscous Burgers shock, in dimensionless form, runs into a wall held at
  zero velocity and relaxes to a steady boundary layer, for Re = 50, 100 and
  200 on 401 nodes with RK4. It reuses the shared `viscous_burgers` model
  unchanged, checks the steady profile u = A tanh(A (1 − x) / (2ν)) and the
  99 % thickness (2ν / A) atanh(0.99), exits with an error beyond its
  tolerances and runs in CI with the reference examples ([#177](https://github.com/biface/oxiflow/issues/177)).
- `examples/competitive_frontal_analysis`, the chromatography reference
  example: a clean column fed with 0.1 mol/L of two solutes competing for
  the same sites (competitive Langmuir isotherm, ideal model), on 401 nodes
  with RK4. It reuses the shared `langmuir_multi` model unchanged, checks the
  outlet times of the two shocks, the roll-up plateau of the less retained
  solute and the mass balance against the equilibrium theory (Nicoud,
  *Chromatographic Processes*, 2015), exits with an error beyond its
  tolerances and runs in CI with the reference examples ([#174](https://github.com/biface/oxiflow/issues/174)).
- `examples/gray_scott_1d`, the reaction–diffusion reference example: the
  Gray–Scott model in dimensionless form on a periodic 1D ring, closed by
  the model itself across its two end nodes. Single modes laid on a
  Turing-unstable homogeneous state grow or decay at the rates of the
  linearised discrete equations (four modes, relative error checked to
  1e-7); a local perturbation of the trivial state grows into pulses that
  split until the ring fills, checked by the pulse count at three times and
  by the mirror symmetry of the state. The example exits with an error
  beyond its tolerances and runs in CI with the reference examples ([#176](https://github.com/biface/oxiflow/issues/176)).
- `examples/lahar_lake_wave`, the surface-geophysics reference example: a
  lahar (the [DQ-032](https://github.com/biface/oxiflow/issues/85) multi-field model, 20 m²/s with a solids fraction of
  0.2) runs down a 4 km channel and pours over a lip into a lake 15 m deep.
  Channel and lake are two domains on their own grids, advanced by
  `MultiDomainOrchestrator` and coupled by two `CouplingOperator`s that
  carry the outflow across the lip. The shared shallow-layer flux
  (`rsc/models/lahar_flux.rs`) takes one treatment per edge (truncation,
  wall, or a face left to a coupling), its default unchanged. The example
  checks that the lake stays bitwise at rest until the lahar reaches it,
  that mass and solids cross the lip without loss to rounding, and that the
  bore raised in the lake obeys the Rankine–Hugoniot conditions; it exits
  with an error beyond its tolerances and runs in CI with the reference
  examples ([#180](https://github.com/biface/oxiflow/issues/180)).
- `examples/lahar_lake_wave_simul` replays the run of `lahar_lake_wave` as
  an animated SVG, one frame per saved state: the depth of the lahar along
  the channel and the free surface of the lake, with a time bar. The
  animation uses SVG animation elements only, with no script, so it plays
  in a web browser and on the wiki; a viewer that does not animate shows
  the first frame. Both examples take the settings of the run from
  `rsc/models/lahar_outfall.rs` ([#149](https://github.com/biface/oxiflow/issues/149)).
- `examples/heat_transfer_1d_plot` draws the two runs of
  `heat_transfer_1d` against their exact solutions:
  `heat_transfer_1d_cooling.svg`, the cooling bar at four times against the
  Fourier series, and `heat_transfer_1d_graded.svg`, the approach of the
  graded bar to its steady profile, with the linear profile of a
  homogeneous bar for comparison. The model, the runs and the exact
  solutions move to `rsc/models/heat_transfer_1d.rs`, shared by both
  examples; the output of `heat_transfer_1d` is unchanged
  ([#149](https://github.com/biface/oxiflow/issues/149)).
- `examples/consolidation_1d_plot` draws the run of `consolidation_1d`
  against Terzaghi's series: `consolidation_1d_isochrones.svg`, the excess
  pore pressure across the layer at five time factors, and
  `consolidation_1d_degree.svg`, the average degree of consolidation
  against the time factor on a logarithmic axis. The model, the run and the
  series move to `rsc/models/consolidation_1d.rs`, shared by both examples;
  the output of `consolidation_1d` is unchanged
  ([#149](https://github.com/biface/oxiflow/issues/149)).
- `examples/magnetic_diffusion_1d_plot` draws the runs of
  `magnetic_diffusion_1d` in copper and aluminium against the exact
  half-space solution: `magnetic_diffusion_1d_profiles.svg`, the field at
  three times, one panel per metal; `magnetic_diffusion_1d_depth.svg`, the
  depth where the field reaches half its surface value against
  `sqrt(eta t)`, on which both metals fall on the same line; and
  `magnetic_diffusion_1d_current.svg`, the induced current density against
  its exact Gaussian. The model, the runs and the exact solution move to
  `rsc/models/magnetic_diffusion_1d.rs`, shared by both examples; the
  output of `magnetic_diffusion_1d` is unchanged
  ([#149](https://github.com/biface/oxiflow/issues/149)).
- `examples/magnetic_diffusion_1d_simul` replays the runs of
  `magnetic_diffusion_1d` as an animated SVG, one frame per step: the field
  entering copper and aluminium on the same axes, with a time bar. The
  drawing pieces of the animations (panels, animated curves, legend, time
  bar) move from `lahar_lake_wave_simul` to `rsc/render/svg_animation.rs`,
  shared by both; the animation of `lahar_lake_wave_simul` is unchanged
  ([#149](https://github.com/biface/oxiflow/issues/149)).
- `BDF2Solver` takes the banded path of `BackwardEulerSolver` and
  `CrankNicolsonSolver` (`sparse` feature), configured by the same
  builders: `with_sparse_solver`, `with_sparse_threshold` (default 100) and
  `with_jacobian_bandwidth`, counted in flat unknowns (`d·(s + 1) − 1` for
  `d` interleaved components). The Backward Euler bootstrap step and the
  BDF2 update both take it, through the iterated Newton loop and with the
  solver's Newton parameters. Without these builders the solver stays on
  the dense path, and its results are unchanged bit for bit ([#133](https://github.com/biface/oxiflow/issues/133)).
- `model::PrescribedInitialState` wraps any `PhysicalModel` and replaces its
  initial condition with a given `ContextValue`; every other method,
  `name()` included, is delegated. It completes the restore contract of
  [DT-025](https://github.com/biface/oxiflow/issues/66): to resume from a `SimulationSnapshot`, wrap the rebuilt model with
  `snapshot.state` and start the scenario at `snapshot.t` with
  `Scenario::with_t_start`. A resume reproduces the uninterrupted run for
  one-step integrators only, the snapshot carrying no history (BDF2
  restarts on its bootstrap step), and its times agree to rounding, the
  fixed-step loop computing `t_start + k·dt`. The `gray_scott_1d` and
  `lahar_lake_wave` examples use it instead of their own wrappers, with
  unchanged results ([#212](https://github.com/biface/oxiflow/issues/212)).

### Changed

- Boundary conditions are applied before the calculators that read the
  state: at each derivative evaluation, the calculators providing the
  variables the conditions require run first, then the conditions, then the
  other calculators and `compute_physics`. A gradient or Laplacian no longer
  sees a boundary value that drifted since the last evaluation; with
  boundary conditions, results change accordingly ([DD-070](https://github.com/biface/oxiflow/issues/200), [#201](https://github.com/biface/oxiflow/issues/201)).
- The solver drivers (fixed-step loop, DoPri45, operator splitting,
  `MultiDomainOrchestrator`) apply the boundary conditions to the initial
  state and to the state at the end of each step, before checking and saving
  it: every saved state satisfies them, including a value that depends on
  time. `SteppableSolver::step` itself does not ([DD-070](https://github.com/biface/oxiflow/issues/200), [#201](https://github.com/biface/oxiflow/issues/201)).
- `DanckwertsInlet` sets the inlet node from its neighbour,
  `u0 = (u_feed + κ u1) / (1 + κ)` with `κ = D / (v (x1 − x0))`, the
  one-sided discretisation of the Robin condition solved for `u0`. It now
  requires the feed variable only, no longer the spatial gradient, declares
  node 0, and refuses a mesh with fewer than two nodes or whose node 1 is not
  downstream of node 0. Its values change accordingly ([DD-070](https://github.com/biface/oxiflow/issues/200), [#201](https://github.com/biface/oxiflow/issues/201)).
- `DanckwertsInlet` and `DanckwertsOutlet` act on a boundary region of the
  mesh, resolved through `Mesh::boundary_neighbours`, instead of node indices:
  every node of the region is set from its interior neighbour. `new` keeps its
  signature and its region (the lower face of axis 0 for the inlet, the upper
  one for the outlet), so results on 1D meshes are unchanged bit for bit; the
  new `on` method chooses another face, e.g. for a flow towards decreasing
  `x`. Both conditions gain a public `region` field. The 1D-only check moves
  to the mesh, which now also refuses a boundary pair not ordered along the
  axis. The outlet, formerly a unit value, serialises as a map holding its
  region ([DA-071](https://github.com/biface/oxiflow/issues/203), [#204](https://github.com/biface/oxiflow/issues/204)).
- `ComputeContext` stores its values behind `Arc`. `insert` and the typed
  accessors keep their signatures and still return references, so models
  read the context as before ([#196](https://github.com/biface/oxiflow/issues/196)).
- The finite-difference Jacobians of the implicit methods (dense and
  banded) allocate one work state per Jacobian and perturb it in place,
  instead of a new copy per column or colour plus, for a `VectorField`
  state, two transposing copies (one to build the perturbed state, one to
  read the derivative). The Jacobians are bit-for-bit unchanged ([#197](https://github.com/biface/oxiflow/issues/197)).
- The banded finite-difference Jacobian (`sparse` feature) is assembled
  directly in compressed-column form: the band structure is built from the
  system size and the bandwidth, and each column is written once in its
  final place. The θ-method folds `I − θΔt·J` into it in place. The
  intermediate list of entries, its conversion to faer triplets and the
  sort done by `SparseColMat::try_new_from_triplets` are gone. Matrices and
  steps are bit-for-bit unchanged ([#199](https://github.com/biface/oxiflow/issues/199)).
- `benches/oxiflow_parallel` draws the summary plots of its parameterised
  groups on logarithmic axes. The groups sweep their size over several
  decades, and on linear axes every point but the largest sat against the
  origin, where the sequential/parallel crossover lies. Measurements and
  saved data are unchanged; existing results can be redrawn without new
  sampling through Criterion's `--load-baseline`
  ([#214](https://github.com/biface/oxiflow/issues/214)).
- The published package keeps what Cargo and docs.rs use: the library,
  the examples, tests and benchmarks with the models of `rsc/` they
  include, the KaTeX header of the documentation, the licence and notice,
  the README and this changelog. Repository configuration (CI workflows,
  hooks, IDE run configurations, scripts) and the contribution guides stay
  on GitHub; the package goes from 159 to 119 files. `Cargo.toml` lists the
  contents with `include`, so a new repository file stays out of the
  package unless added there
  ([#214](https://github.com/biface/oxiflow/issues/214)).
- The documentation of every Rayon dispatch threshold cites the reference
  run of the `oxiflow_parallel` benchmarks for v0.10.0 instead of earlier
  measurements: the Jacobian assembly of the implicit solvers, the FD
  operators, `MultiDomainOrchestrator`, `CoefficientField` and the
  reference models of `rsc/`. The documentation of `solver::parallel`
  explains how to size a threshold, as the sequential time of one call of
  the dispatched loop, and links the dispatch abaques of the wiki
  ([#214](https://github.com/biface/oxiflow/issues/214)).
- `Cargo.toml` sets `autobenches = false`: `oxiflow_parallel` is the only
  bench target, as intended. Cargo no longer also builds each module of
  `benches/` as an empty standalone target ([#199](https://github.com/biface/oxiflow/issues/199)).
- **Breaking:** `IntegratorSpec::BDF2` gains `sparse_threshold` (default
  100) and `jacobian_bandwidth` (default `None`), as `BackwardEuler` and
  `CrankNicolson` have: a declared bandwidth builds the solver with
  `FaerSparseSolver`. Code that builds the variant with a struct literal
  must add the two fields; deserialised specifications are unaffected,
  both fields having defaults ([#133](https://github.com/biface/oxiflow/issues/133)).
- **Breaking:** `UniformGrid1D::new` returns `Result<Self, OxiflowError>`
  (`OxiflowError::PreconditionFailed`, context `"UniformGrid1D"`) instead of
  `Result<Self, String>`. `?` now works directly in functions returning
  `OxiflowError`; the `.map_err(OxiflowError::InvalidDomain)` workaround is
  gone from `examples/lahar_multifield.rs` and its test.
- **Breaking:** `Scenario::validate`, called by `Solver::solve` and the
  orchestrator before the first step, rejects with
  `OxiflowError::InvalidDomain` a domain whose initial `ScalarField` or
  `VectorField` does not have one row per mesh node, and names the domain.
  Such a state used to be integrated without any error when no calculator
  or boundary condition reads the mesh: a run on a 10-node mesh from a
  7-value state returned 7-value states ([#212](https://github.com/biface/oxiflow/issues/212)).
- **Breaking:** `FluxDivergenceOperator` gains a required method,
  `max_wave_speed`, returning the largest absolute wave speed of the scheme
  on a field, from the field and the context. Like `stencil_radius`, it has
  no default: a scheme implemented outside the crate must state its speed,
  `0.0` for a flux that transports nothing. The schemes of the crate
  (`FVCenteredFlux`, `FVUpwindFlux`, `WENO3`, `WENO5`, `LimitedFlux`,
  `AdaptiveFlux`) return the absolute value of their advection velocity
  ([DA-073](https://github.com/biface/oxiflow/issues/229),
  [#228](https://github.com/biface/oxiflow/issues/228)).
- **Breaking:** `CouplingOperator::apply` takes a `&CouplingContext`
  instead of a `&ComputeContext`. `CouplingContext`, in `coupling`, holds one
  `ComputeContext` per side of the interface, read through `source()` and
  `target()`; `time()` and `time_step()` read the time and step both share.
  `MultiDomainOrchestrator` fills each side with the variables the operator
  requires, and the calculators these depend on, computed on that domain's
  current state, after the operators declared before it. An implementation
  outside the crate changes the type of its `ctx` parameter, and reads
  `ctx.time_step()` as before; a unit test of a coupling builds its context
  with `CouplingContext::new(t, dt)` ([DA-011](https://github.com/biface/oxiflow/issues/11)
  amendment 1, [#231](https://github.com/biface/oxiflow/issues/231)).
- The explicit integrators own the advective stability check
  ([DA-073](https://github.com/biface/oxiflow/issues/229),
  [#228](https://github.com/biface/oxiflow/issues/228)). At the start of each
  step they evaluate the wave speed `s` the model declares, with the first
  stage, and derive `dt_max = dx / s` from the characteristic length of the
  mesh: a Courant limit of 1, a necessary condition only. `ForwardEulerSolver`
  and `RK4Solver` refuse a larger fixed step with
  `OxiflowError::PreconditionFailed`, whose context now names the solver
  (`"ForwardEulerSolver::step"`, `"RK4Solver::step"`) instead of the scheme,
  and whose message gives the Courant number and `dt_max`; the check is made
  once per step instead of once per stage. `DoPri45Solver` shortens a trial
  step above `dt_max` instead of aborting the run, and reports the number of
  shortened steps under the metadata key `solver.cfl_limited_steps`. A model
  that declares no speed is not checked, as before for models outside
  `DiscretizedModel`. The shallow-layer flux of the lahar models
  (`rsc/models/lahar_flux.rs`) drops its own copy of the check and declares
  its state-dependent speed, `|u| + √(g·h)` over the cells, instead.
- Each reference model of the examples, tests and benchmarks (diffusion,
  viscous Burgers, multi-compound Langmuir, multi-field lahar, lahar–lake)
  is defined once under `rsc/models/` and included by every target that
  uses it, instead of up to four copies. Example outputs and test
  tolerances are unchanged. The `langmuir_multi_plot` example gets back the
  parallel branch its copy had lost. The dispatch diagnostics of the
  Langmuir and lahar benchmarks now time the models' own `compute_physics`
  and `apply` with the threshold forced, instead of partial copies of their
  loops; their default thresholds stay as measured at v0.9.0. The crate
  itself and its public API are unchanged ([#194](https://github.com/biface/oxiflow/issues/194)).
- The lahar–lake scenario of `examples/lahar_lake_wave` (constants, grids,
  initial states, domain models, outfall coupling) moves to
  `rsc/models/lahar_outfall.rs`, with the grid spacing as a parameter and,
  with the `parallel` feature, a dispatch threshold for the sites inside a
  domain, so that a benchmark can run the same scenario at other sizes. The
  example keeps its run settings, diagnostics and checks; its saved states
  are bit-for-bit unchanged ([#141](https://github.com/biface/oxiflow/issues/141)).
- The column of the multi-compound Langmuir model (length, grid and spacing
  for a node count, one backward-gradient calculator per species) and the
  reference scenario of `examples/langmuir_multi` (100 nodes, 4000 steps
  over 800 s) are defined once in `rsc/models/langmuir_multi.rs`, instead of
  being copied by the two Langmuir examples, `competitive_frontal_analysis`,
  the reference test and the benchmark. With the `parallel` feature, the
  calculators take a dispatch threshold. Saved states, outputs and the plot
  are bit-for-bit unchanged ([#141](https://github.com/biface/oxiflow/issues/141)).
- The validation scenario of the multi-field lahar model (paper Sec. 4.1:
  source term and flux on a flat, non-erodible bed, bed-slope calculator,
  front position) is defined once in `rsc/models/lahar_validation.rs`,
  instead of being copied by the two lahar examples, the analytical test and
  the benchmark. The analytical front laws of the three drag regimes join
  `ValidationRegime` in `rsc/models/lahar_source.rs`, next to the friction
  constants they depend on; `build_source` takes the dispatch threshold of
  the source term. With the `parallel` feature, the builders take a
  dispatch threshold for the source term, the flux and the bed-slope
  calculator. Saved states, outputs and the plots are bit-for-bit unchanged
  ([#141](https://github.com/biface/oxiflow/issues/141)).
- Documentation: `ContextValue::VectorField` holds `d` values per node,
  either vector components or `d` unknowns unrelated to the mesh dimension;
  the default dispatch threshold of the finite-difference operators
  (49 999) is documented as indicative ([#140](https://github.com/biface/oxiflow/issues/140), [#194](https://github.com/biface/oxiflow/issues/194)).
- Documentation: the READMEs, the contributing guides and the development
  guides follow the 0.xx milestone numbering. Their milestone tables give
  the titles, versions and dates of the GitHub milestones, from J0 to J30
  and v0.40.0, and the old v1.0, v2.0 and v3.0 targets are gone. The quick
  start of the READMEs is rewritten: the previous one used an API that
  oxiflow never had (`oxiflow::prelude`, `Scenario::builder`); the new one
  compiles against the crate and runs a diffusion case. Decision
  identifiers carry the prefix of their issue ([#169](https://github.com/biface/oxiflow/issues/169)).

### Fixed

- A coupling operator receives the values of the variables it requires.
  The orchestrator checked that a calculator provides each of them, then
  passed the operator a context holding only the time and the step: a
  coupling reading a gradient failed at the first step. None of the
  couplings of the crate, its examples or its tests declares a requirement,
  so no result changes (see *Changed*,
  [#231](https://github.com/biface/oxiflow/issues/231)).
- `Scenario` refuses two domains with the same identifier and a coupling
  interface that names a domain it does not hold. `Scenario::multi`
  accepted both; the orchestrator keys states, histories and solvers by
  identifier, so the second of two domains named alike overwrote the first
  at every step, with no error. `Scenario::multi` and `Scenario::validate`
  now return `OxiflowError::InvalidDomain`, naming the identifier or the
  interface ([#230](https://github.com/biface/oxiflow/issues/230)).
- The implicit integrators (Backward Euler, Crank–Nicolson, BDF2) run a
  flux scheme at any Courant number. Every flux scheme checked the explicit
  advective condition `|v|·dt/dx ≤ 1` in its own `apply()` with the step of
  the context, which the solvers fill with the integrator's step for
  implicit integrators too: a periodic upwind advection with Backward Euler
  failed at a Courant number of 2. The schemes no longer check it, and
  `operators::check_cfl` is removed; the explicit integrators check instead
  (see *Changed*) ([DA-073](https://github.com/biface/oxiflow/issues/229),
  [#228](https://github.com/biface/oxiflow/issues/228)).
- The advective check no longer lets a NaN through: `check_cfl` compared
  `|v|·dt/dx > 1`, false for a NaN speed. The explicit integrators refuse a
  declared wave speed that is NaN, infinite or negative with
  `OxiflowError::PreconditionFailed`, naming the model
  ([#228](https://github.com/biface/oxiflow/issues/228)).
- The `oxiflow_parallel` benchmark ran to its end only without the `sparse`
  feature: with `parallel,sparse`, the Jacobian dispatch diagnostic of the
  multi-compound Langmuir model stopped on `NewtonNotConverged` at 10,000
  nodes on the banded path, and the cases after it, the lahar groups
  included, never ran. The diagnostic now gives Backward Euler a budget of
  two Newton iterations, the measured need of its two largest banded cases,
  on both paths. The documentation of `OxiflowError::NewtonNotConverged`
  and of the three implicit solvers says which settings to adjust (budget,
  Jacobian strategy, convergence criterion) and when (a nonlinear model, a
  fine grid on the banded path); the default budget, unchanged at one
  iteration, is defined once for the solvers and for `IntegratorSpec`
  ([#213](https://github.com/biface/oxiflow/issues/213)).
- The lahar source term shared by the lahar examples, test and benchmark
  (`rsc/models/lahar_source.rs`) injected the mass of water, `rho_f * Q`,
  whatever the solids fraction `psi` of the inflow, so injected material
  lost a share `(rho_s − rho_f) / rho_f * psi` of its volume; it now injects
  the bulk mass `rho_bar(psi) * Q`. Deposition is also capped so that one
  step deposits at most the solids present: in a thin layer, an explicit
  step could take more and drive the depth negative. Pure-water runs, the
  only ones so far, are unchanged bit for bit ([#209](https://github.com/biface/oxiflow/issues/209)).
- The banded path of `BackwardEulerSolver` and `CrankNicolsonSolver`
  (`sparse` feature: sparse solver configured, bandwidth declared, system
  above the sparse threshold) ran a single linearised correction without
  any convergence check, so a nonlinear model silently got a first-order
  result whatever Newton budget was configured. It now runs the same
  iterated Newton loop as the dense path, with the solver's convergence
  criterion, Jacobian strategy and iteration budget, and reports
  `NewtonNotConverged` when the budget runs out. The dense fallback taken
  below the sparse threshold also honours that configuration instead of a
  fixed single correction. Both paths now treat the explicit and implicit
  terms the same way, so a banded step matches a dense one up to rounding
  ([#206](https://github.com/biface/oxiflow/issues/206)).
- `BDF2Solver` runs its Backward Euler bootstrap step with its own Newton
  configuration (`with_newton_convergence`, `with_jacobian_strategy`,
  `with_max_newton_iterations`), as it does the BDF2 update. The bootstrap
  step used a fixed single frozen-Jacobian correction, so a nonlinear model
  could fail at the first step with `NewtonNotConverged { iterations: 1 }`
  whatever budget was configured. Results with the default configuration
  are unchanged ([#205](https://github.com/biface/oxiflow/issues/205)).
- `DanckwertsInlet` diverged as soon as `κ = D / (v Δx) > 1`, i.e.
  `(n − 1) / Pe > 1` on a uniform grid with `Pe = vL/D`: it set `u0` from
  the gradient at node 0, computed from the previous `u0`, which iterated
  `u0` with a factor `−κ` from one evaluation to the next. The inlet value
  no longer depends on its previous value ([#201](https://github.com/biface/oxiflow/issues/201)).
- The documentation of `boundary` stated that the conditions of a domain
  are applied in ascending `priority()` order. The solver applies them in
  the order they are declared, as it always has; the documentation now says
  so, and `priority()` is documented as reserved for boundary conditions
  ([#202](https://github.com/biface/oxiflow/issues/202)).
- `UniformGrid1D::new` rejects non-finite bounds: a NaN bound used to pass
  the `x_end > x_start` check and yield a grid of NaN nodes.
- `FDGradientCalculator` and `FDLaplacianCalculator` return
  `OxiflowError::PreconditionFailed` from `compute()` on a mesh their
  stencils cannot handle, instead of a silently wrong field: a mesh that is
  not 1D, a 1D mesh whose node spacing is not uniform or differs from
  `characteristic_length()` (relative tolerance `1e-6`), and, for the
  gradient, a `dimension` beyond the mesh dimension. The check runs once, at
  construction; `new()` keeps its signature and results on `UniformGrid1D`
  are unchanged ([#186](https://github.com/biface/oxiflow/issues/186)).
- `DanckwertsInlet::apply` and `DanckwertsOutlet::apply` return
  `OxiflowError::PreconditionFailed` on a mesh that is not 1D: both locate
  the boundary by node index (`0` and `n − 1`), which only an ordered 1D grid
  makes meaningful ([#187](https://github.com/biface/oxiflow/issues/187)).
- Divergence detection covers every numeric state variant. A NaN or an
  infinity in a `VectorField` state used to go unnoticed: the run ended with
  `Ok` and non-finite states, and `Solver::on_divergence` was never called.
  It now raises `OxiflowError::SolverDivergence` with a snapshot, as for a
  `ScalarField` ([#192](https://github.com/biface/oxiflow/issues/192)).
- `OperatorSplittingSolver::solve` hands a `SimulationSnapshot` to
  `Solver::on_divergence` before returning `SolverDivergence`, like every
  other solver. The snapshot is built by one shared helper in all solver
  loops ([#134](https://github.com/biface/oxiflow/issues/134)).
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
  ([#192](https://github.com/biface/oxiflow/issues/192)).

### Security

- The `vtk` feature no longer depends on `vtkio`. `SimulationResult::write_vtk`
  now writes the `.vtu` file itself, with an unchanged signature and the same
  inline `binary` (base64) data arrays as before, so NaN and infinities in a
  diverged state are still exported bit for bit. This removes
  RUSTSEC-2026-0041 (`lz4_flex 0.7.5`), RUSTSEC-2026-0194 and
  RUSTSEC-2026-0195 (`quick-xml 0.22.0`) from the dependency tree, along with
  the future-incompatibility warning `nom 3.2.1` raised on builds with `vtk`
  ([#191](https://github.com/biface/oxiflow/issues/191)).

---

## [0.9.0] - 2026-09-16

### Added

- `parallel` feature flag (Rayon, optional) — per-site dispatch calibration
  rather than a single crate-wide threshold ([DA-048](https://github.com/biface/oxiflow/issues/140)): `fd.rs`'s
  `default_parallel_threshold()` (`49_999`, [DT-014](https://github.com/biface/oxiflow/issues/14), bare-stencil profile),
  `langmuir_multi`'s measured `1_000` ([#138](https://github.com/biface/oxiflow/issues/138)), `lahar_multifield`'s two
  independent sites — `default_source_parallel_threshold()` (`1_000`,
  friction/erosion loop) and `default_flux_parallel_threshold()`
  (`10_000`, flux divergence loop, set above an unstable crossover region
  rather than at its lower bound) ([DA-048](https://github.com/biface/oxiflow/issues/140), [#139](https://github.com/biface/oxiflow/issues/139))
- `ParallelThreshold`/`calibrate_parallel_threshold` (`src/solver/parallel.rs`)
  — shared `AtomicUsize`-backed mechanism for future `src/`-internal
  dispatch sites; `pub(crate)`, not usable from `examples/`/`tests/`/
  `benches/` ([DA-048](https://github.com/biface/oxiflow/issues/140))
- `benches/oxiflow_parallel` — Criterion suite ([#53](https://github.com/biface/oxiflow/issues/53)) covering four
  benchmark models: `diffusion_1d` ([#136](https://github.com/biface/oxiflow/issues/136)), `viscous_burgers` ([#137](https://github.com/biface/oxiflow/issues/137)),
  `langmuir_multi` ([#138](https://github.com/biface/oxiflow/issues/138)), `lahar_multifield` ([#139](https://github.com/biface/oxiflow/issues/139)); full write-up in
  `BENCHMARKS.md`
- `examples/lahar_multifield.rs` — multi-field lahar model ([DQ-032](https://github.com/biface/oxiflow/issues/85), [#139](https://github.com/biface/oxiflow/issues/139)):
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
  validated against ([#145](https://github.com/biface/oxiflow/issues/145))
- `plotters` dev-dependency for the above

### Changed

- Criterion bench target renamed `oxiflow_benchmarks` → `oxiflow_parallel`
  — named for its actual object (parallel-dispatch calibration), since
  other `[[bench]]` targets for other purposes are expected once 2D/3D
  work starts

### Not implemented

- Dirty-flag caching for repeated `ContextCalculator` evaluations
  ([DD-015](https://github.com/biface/oxiflow/issues/15), [#52](https://github.com/biface/oxiflow/issues/52)) — closed without implementation this sprint; scope
  parked, not delivered

---

## [0.8.0] - 2026-08-03

### Added

- `gpu` feature flag (`wgpu 26.0.1`, `pollster`, `bytemuck`, all optional) —
  `solver::gpu::GpuContext::new()` acquires a GPU adapter/device/queue,
  blocking via `pollster`; `to_gpu_bytes()` reinterprets `ContextValue`'s
  contiguous `f64` payloads as raw bytes via `bytemuck::cast_slice`, without
  requiring `ContextValue` itself to implement `bytemuck::Pod` (the enum
  discriminant remains the blocker for that, deferred) ([DA-026](https://github.com/biface/oxiflow/issues/73), [DT-045](https://github.com/biface/oxiflow/issues/128),
  [#74](https://github.com/biface/oxiflow/issues/74))
- `OxiflowError::GpuUnavailable` — returned when no compatible GPU adapter
  or device is found; never a silent CPU fallback, the caller chooses its
  own CPU solver ([DT-045](https://github.com/biface/oxiflow/issues/128))
- CI: `clippy --features gpu` and `test --features gpu` jobs added to
  `ci.yml` ([#122](https://github.com/biface/oxiflow/issues/122))

### Changed

- **MSRV raised `1.80` → `1.84`** — no stable `wgpu` release declares
  compatibility with `1.80`; `1.84` is the lowest MSRV declared by `wgpu`
  26.0.1, the version selected ([DT-045](https://github.com/biface/oxiflow/issues/128))
- `ContextValue`'s `GPU-READY` annotation ([DA-026](https://github.com/biface/oxiflow/issues/73) INV-GPU-5) relocated onto
  `ContextValue` itself (previously misapplied to `MultiDomainState`, which
  can never be `bytemuck::Pod` — a `HashMap` has no fixed, Pod-compatible
  layout); column-major layout (INV-GPU-4) documented directly on
  `ContextValue::Matrix`/`ContextValue::VectorField` ([#127](https://github.com/biface/oxiflow/issues/127))
- `MultiDomainState` module docs corrected: requalified as an
  indexed/orchestration container, out of INV-GPU-1's scope, rather than
  claiming to satisfy it ([#127](https://github.com/biface/oxiflow/issues/127))
- Rustdoc intra-doc links and DD citation style normalised across the
  crate; `cargo doc --no-deps --all-features` now emits **zero warnings**
  (validated by a real run) ([#122](https://github.com/biface/oxiflow/issues/122))

### Known gaps

- `derive(Pod, Zeroable)` on `ContextValue` remains blocked by its enum
  discriminant — resolution deferred past this release, not part of the
  `gpu` feature's initial scope ([DT-045](https://github.com/biface/oxiflow/issues/128))
- No real-GPU integration test runs in CI (no GPU hardware available);
  the one that exists is `#[ignore]`d ([#74](https://github.com/biface/oxiflow/issues/74))
- `BDF2Solver` still has no sparse builders (`sparse_solver`,
  `sparse_threshold`, `jacobian_bandwidth` absent) — a [DA-043](https://github.com/biface/oxiflow/issues/103) gap,
  unaddressed by this release, carried forward to v0.9.0
- `OperatorSplittingSolver::solve` (IMEX) does not call
  `self.on_divergence()` / build a `SimulationSnapshot` on divergence,
  unlike every other solver in the crate — discovered this release,
  carried forward to v0.9.0

## [0.7.0] - 2026-07-30

### Added

- Dedicated test for the perturbed-Jacobian × Dirichlet boundary-condition
  interaction, closing a known-untested gap documented since [DD-033](https://github.com/biface/oxiflow/issues/86)
  (v0.4.0): the constrained node's column in `finite_difference_jacobian`
  is exactly zero, while its row keeps genuine physics-based coupling to
  unconstrained neighbours ([#113](https://github.com/biface/oxiflow/issues/113))
- `IntegratorSpec::BDF2` variant — Newton fields only, deliberately no
  sparse fields (`BDF2Solver` has no sparse builders to map, a [DA-043](https://github.com/biface/oxiflow/issues/103) gap
  distinct from this DTO increment) ([DD-044](https://github.com/biface/oxiflow/issues/109), [#116](https://github.com/biface/oxiflow/issues/116))
- `IntegratorSpec::CrankNicolson` variant, symmetric to `BackwardEuler` in
  every field (sparse dispatch + Newton configuration); `theta = 0.5`
  stays implicit to the variant, matching `BackwardEuler`'s own
  `theta = 1.0` ([DD-044](https://github.com/biface/oxiflow/issues/109), [#115](https://github.com/biface/oxiflow/issues/115))
- `IntegratorSpec::BackwardEuler` gains `newton_convergence`,
  `jacobian_strategy`, `max_iterations` fields (`serde` feature), mapped to
  `BackwardEulerSolver`'s three Newton builders via `TryFrom`; defaults
  match the solver's own zero-config values ([DD-044](https://github.com/biface/oxiflow/issues/109), [#114](https://github.com/biface/oxiflow/issues/114))
- `NewtonConvergence`/`JacobianStrategy` gain `serde::Deserialize` under
  the `serde` feature ([#114](https://github.com/biface/oxiflow/issues/114))
- `BDF2Solver` gains `with_newton_convergence()`, `with_jacobian_strategy()`,
  `with_max_newton_iterations()` builders; `bdf2_step_newton()`
  (`solver::methods::bdf2`) routes through the generic Newton loop with
  caller-supplied configuration — defaults reproduce the pre-[DD-044](https://github.com/biface/oxiflow/issues/109)
  single-correction behaviour exactly on affine problems. The
  startup/bootstrap step is unaffected: it always uses
  `theta_method_step()`'s non-configurable path ([DD-044](https://github.com/biface/oxiflow/issues/109), [#112](https://github.com/biface/oxiflow/issues/112))
- `BackwardEulerSolver`/`CrankNicolsonSolver` gain `with_newton_convergence()`,
  `with_jacobian_strategy()`, `with_max_newton_iterations()` builders;
  `theta_method_step_newton()` (`solver::methods::implicit`) routes through
  the generic Newton loop with caller-supplied configuration — defaults
  reproduce the pre-[DD-044](https://github.com/biface/oxiflow/issues/109) single-correction behaviour exactly on affine
  problems ([DD-044](https://github.com/biface/oxiflow/issues/109), [#111](https://github.com/biface/oxiflow/issues/111))
- Iterated Newton solver core (`solver::methods::newton`): `NewtonConvergence`
  (`ResidualOnly`/`CombinedOr`/`CombinedAnd`), `JacobianStrategy`
  (`ModifiedFrozen`/`FullNewton`/`ModifiedWithStagnationCheck`), and the
  generic convergence loop shared by theta-method and BDF2 ([DD-044](https://github.com/biface/oxiflow/issues/109),
  [#110](https://github.com/biface/oxiflow/issues/110))
- `OxiflowError::NewtonNotConverged { iterations, residual }` — routed
  through `Solver::on_divergence()` once wired ([#110](https://github.com/biface/oxiflow/issues/110))

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
  configuring the loop themselves ([#111](https://github.com/biface/oxiflow/issues/111), [#112](https://github.com/biface/oxiflow/issues/112))

### Known gaps

- `theta_method_step_adaptive` (sparse/banded dispatch, [DA-043](https://github.com/biface/oxiflow/issues/103)) is not yet
  Newton-aware: configuring both a sparse backend and a non-default Newton
  budget on `BackwardEulerSolver`/`CrankNicolsonSolver` silently keeps the
  sparse path single-shot ([#111](https://github.com/biface/oxiflow/issues/111))

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
  feature-gated `sparse` ([DT-013](https://github.com/biface/oxiflow/issues/13), [DA-043](https://github.com/biface/oxiflow/issues/103), [#50](https://github.com/biface/oxiflow/issues/50))
- `SimulationSnapshot`, `SnapshotFormat`, `write_snapshot()`/`read_snapshot()`
  (JSON/bincode) — `src/solver/snapshot.rs` ([DT-025](https://github.com/biface/oxiflow/issues/66) Option B / [DD-029](https://github.com/biface/oxiflow/issues/77), [#71](https://github.com/biface/oxiflow/issues/71))
- `Solver::on_divergence()` — default no-op hook, called automatically before
  `OxiflowError::SolverDivergence` is returned ([#71](https://github.com/biface/oxiflow/issues/71))
- `OxiflowError::Persistence` — snapshot/HDF5/VTK I/O and (de)serialisation
  failures
- `ExternalTabulated::from_hdf5()` — loads tabulated data from an HDF5 file,
  feature-gated `hdf5` ([DT-027](https://github.com/biface/oxiflow/issues/75), [#105](https://github.com/biface/oxiflow/issues/105))
- `SimulationResult::write_vtk()` — XML `.vtu` export for ParaView/VisIt,
  feature-gated `vtk` ([DT-027](https://github.com/biface/oxiflow/issues/75), [#78](https://github.com/biface/oxiflow/issues/78))
- `IntegratorSpec` — external config DTO for `BackwardEulerSolver` +
  `TryFrom<IntegratorSpec> for Box<dyn Solver>`, feature-gated `sparse`
  ([DT-027](https://github.com/biface/oxiflow/issues/75) amendment 1, [#104](https://github.com/biface/oxiflow/issues/104))

### Changed

- `SimulationResult` now derives `Clone`
- `hdf5` feature migrated from the unmaintained `hdf5` crate (0.8, aldanor)
  to the actively-maintained `hdf5-metno` fork (chore, [#79](https://github.com/biface/oxiflow/issues/79))

### Design Decisions

- [DT-027](https://github.com/biface/oxiflow/issues/75) — Results export: VTK as interop pivot, HDF5 for bulk datasets ([#75](https://github.com/biface/oxiflow/issues/75))
- [DT-027](https://github.com/biface/oxiflow/issues/75) amendment 1 — config-driven WHAT/HOW construction from JSON/YAML
- [DD-029](https://github.com/biface/oxiflow/issues/77) — Generalised `SimulationSnapshot`: normal checkpoint/resume ([#77](https://github.com/biface/oxiflow/issues/77))
- [DA-043](https://github.com/biface/oxiflow/issues/103) — `SparseLinearSolver`: sibling trait to `DiscreteOperator` (faer)

## [0.5.0] — 2026-07-17

### Added

- `DiscreteOperator` trait — `operators::fd::{UpwindGradient, CenteredGradient,
  CenteredLaplacian}` ([#46](https://github.com/biface/oxiflow/issues/46), [#47](https://github.com/biface/oxiflow/issues/47), [DA-012](https://github.com/biface/oxiflow/issues/12))
- `FluxDivergenceOperator` trait, sibling to `DiscreteOperator` — computes the full
  `∇·F(u, ∇u)` term, needs physical parameters `DiscreteOperator` deliberately does not
  provide ([#97](https://github.com/biface/oxiflow/issues/97), [DA-039](https://github.com/biface/oxiflow/issues/95))
- `DiscretizedModel<Op: FluxDivergenceOperator>` composite — plugs any
  `FluxDivergenceOperator` into `PhysicalModel` ([#93](https://github.com/biface/oxiflow/issues/93), [DA-038](https://github.com/biface/oxiflow/issues/92))
- `operators::fv::{FVCenteredFlux, FVUpwindFlux}` — finite-volume advection/diffusion,
  cell-average semantics ([#48](https://github.com/biface/oxiflow/issues/48), [DD-040](https://github.com/biface/oxiflow/issues/96))
- `operators::weno::{WENO3, WENO5}` — Jiang-Shu substencils, WENO-Z nonlinear weighting
  (Borges et al. 2008) for critical-point accuracy ([#49](https://github.com/biface/oxiflow/issues/49), [DD-041](https://github.com/biface/oxiflow/issues/98))
- `FluxBoundary` enum — explicit boundary-treatment choice per operator:
  - `Periodic` — exact conservation via telescoping ([#48](https://github.com/biface/oxiflow/issues/48), [#49](https://github.com/biface/oxiflow/issues/49))
  - `Truncation` — one-sided decentered fallback at boundary cells, breaks exact
    conservation (documented trade-off) ([DD-042](https://github.com/biface/oxiflow/issues/100))
  - `GhostCell` — real `BoundaryCondition` physics via `ghost_value(depth,
    interior_at_depth, dx)`, method-of-images convention, exact at every depth for a
    Robin condition ([DD-042](https://github.com/biface/oxiflow/issues/100), amendment 1)
- `BoundaryCondition::ghost_value(depth, interior_at_depth, dx) -> Option<f64>` — default
  method, `None` by default, non-breaking ([DD-042](https://github.com/biface/oxiflow/issues/100))
- `operators::limiters::{Limiter, LimitedFlux}` — `MinMod`/`VanLeer`/`Superbee`, MUSCL
  reconstruction ([#99](https://github.com/biface/oxiflow/issues/99))
- `operators::adaptive::AdaptiveFlux` — blends `WENO3`/`LimitedFlux` per face, gated by a
  global mesh Péclet number, driven by `WENO3`'s own smoothness indicator ([#99](https://github.com/biface/oxiflow/issues/99))
- `operators::check_cfl` — shared explicit-advection CFL check (FV, WENO, limiters,
  adaptive)

### Changed

- `FDGradientCalculator`/`FDLaplacianCalculator` delegate their stencil math to
  `operators::fd` instead of hand-rolled loops ([#94](https://github.com/biface/oxiflow/issues/94), [DA-012](https://github.com/biface/oxiflow/issues/12))
- `FluxBoundary` drops `Copy`/`PartialEq`/`Eq` (`Arc<dyn BoundaryCondition>` has neither) —
  `FVCenteredFlux`/`FVUpwindFlux`/`WENO3`/`WENO5` drop `Copy` accordingly
- Wide-stencil divergence helpers (`periodic_wide_divergence`, `truncated_wide_divergence`,
  `ghost_padded_field`, `ghost_cell_wide_divergence`, `wrap`) relocated from
  `operators::weno` to `operators` (`pub(crate)`) — shared by `operators::limiters` and
  `operators::adaptive`

### Fixed

- No user-facing bug in this release; two internal design errors were caught and
  corrected before merge (see `DESIGN_DECISIONS.md` for the full account, not silently
  fixed): WENO's initial nodal-semantics claim ([DD-041](https://github.com/biface/oxiflow/issues/98)), and `FluxBoundary::GhostCell`'s
  initial single-point `ghost_value` signature being insufficient for WENO's wider
  stencils ([DD-042](https://github.com/biface/oxiflow/issues/100), amendment 1)

## [0.4.0] — 2026-06-25

### Added

- `SteppableSolver` trait in `src/solver/methods/mod.rs` — single-step `step()` primitive,
  extracted from existing `ForwardEulerSolver`/`RK4Solver` logic with no behaviour change
  (Ref [#82](https://github.com/biface/oxiflow/issues/82), [DA-031](https://github.com/biface/oxiflow/issues/82))
- `RK4Solver` — explicit, 4-stage Runge-Kutta, order 4 ([#41](https://github.com/biface/oxiflow/issues/41))
- `MultiDomainOrchestrator` — advances each `Domain` with its own `SteppableSolver` and
  invokes the `Scenario`'s registered `CouplingOperator`s between steps; lets a coupled
  scenario mix integrators per domain (partitioned coupling) (Ref [#82](https://github.com/biface/oxiflow/issues/82), [DA-031](https://github.com/biface/oxiflow/issues/82))
- Integration test for the lahar–lake prototype — first real time loop over coupled domains
  via `MultiDomainOrchestrator`; validates orchestration only, not lahar physics
  ([#40](https://github.com/biface/oxiflow/issues/40), Ref [#82](https://github.com/biface/oxiflow/issues/82), [DA-031](https://github.com/biface/oxiflow/issues/82))
- `LinearSolver` trait + `NalgebraDenseSolver` (dense LU) default implementation ([#13](https://github.com/biface/oxiflow/issues/13), [DT-013](https://github.com/biface/oxiflow/issues/13))
- `finite_difference_jacobian`, `theta_method_step` in `src/solver/methods/implicit.rs` —
  shared by `BackwardEulerSolver` (θ=1) and `CrankNicolsonSolver` (θ=0.5)
  ([#43](https://github.com/biface/oxiflow/issues/43), Refs [#13](https://github.com/biface/oxiflow/issues/13), [#86](https://github.com/biface/oxiflow/issues/86), [DT-013](https://github.com/biface/oxiflow/issues/13), [DD-033](https://github.com/biface/oxiflow/issues/86))
- `BackwardEulerSolver` — implicit, 1st order ([#43](https://github.com/biface/oxiflow/issues/43), Ref [#86](https://github.com/biface/oxiflow/issues/86), [DD-033](https://github.com/biface/oxiflow/issues/86))
- `CrankNicolsonSolver` — semi-implicit, 2nd order ([#43](https://github.com/biface/oxiflow/issues/43), Ref [#86](https://github.com/biface/oxiflow/issues/86), [DD-033](https://github.com/biface/oxiflow/issues/86))
- `SteppableSolver::history_depth()` and a `history: &[ContextValue]` parameter on `step()` —
  supports multi-step integrators; defaults to `0`, no change for one-step solvers
  (Ref [#87](https://github.com/biface/oxiflow/issues/87), [DD-034](https://github.com/biface/oxiflow/issues/87))
- `BDF2Solver` — implicit multi-step, 2nd order; starts with one Backward Euler step when
  history is empty ([#44](https://github.com/biface/oxiflow/issues/44), Ref [#87](https://github.com/biface/oxiflow/issues/87), [DD-034](https://github.com/biface/oxiflow/issues/87))
- `SteppableSolver::solve_fixed_step()` default method — factors out the fixed-step loop
  previously duplicated near-verbatim across `ForwardEulerSolver`, `RK4Solver`,
  `BackwardEulerSolver`, `CrankNicolsonSolver`, `BDF2Solver` (Ref [#88](https://github.com/biface/oxiflow/issues/88), [DD-035](https://github.com/biface/oxiflow/issues/88))
- `StepControl::Adaptive` activated (`dt_init`, `dt_min`, `dt_max`, `rtol`, `atol`) — reserved
  since [DA-021](https://github.com/biface/oxiflow/issues/58), v0.1.0, activated alongside DoPri45 (Ref [#89](https://github.com/biface/oxiflow/issues/89), [DD-036](https://github.com/biface/oxiflow/issues/89))
- `StepSizeController` — PI step-size controller with a source-agnostic error norm, shared
  across adaptive integrators (Ref [#89](https://github.com/biface/oxiflow/issues/89), [DD-036](https://github.com/biface/oxiflow/issues/89))
- `DoPri45Solver` — Dormand-Prince, 7-stage FSAL, adaptive step, order 5; `Solver` only, not
  `SteppableSolver` ([#42](https://github.com/biface/oxiflow/issues/42), Ref [#89](https://github.com/biface/oxiflow/issues/89), [DD-036](https://github.com/biface/oxiflow/issues/89))
- `OperatorSplittingSolver`, `SplitOperator`, `SplittingScheme` in
  `src/solver/methods/imex.rs` — n ≥ 2 operator splitting integrator (Strang scheme); `Solver`
  only, not `SteppableSolver` ([#45](https://github.com/biface/oxiflow/issues/45), [DA-037](https://github.com/biface/oxiflow/issues/90))
- `CompositeModel` in `src/model/composite.rs` — sums multiple `PhysicalModel` contributions
  on the same state; used as the outer `Domain`'s bookkeeping model for IMEX scenarios and as
  a monolithic reference solution ([#45](https://github.com/biface/oxiflow/issues/45), [DA-037](https://github.com/biface/oxiflow/issues/90))
- `IntegratorKind::Imex` ([#45](https://github.com/biface/oxiflow/issues/45), [DA-037](https://github.com/biface/oxiflow/issues/90))

### Changed

- `SteppableSolver::step()` gains a `history: &[ContextValue]` parameter on every
  implementor — no behaviour change for one-step solvers, which ignore it (Ref [#87](https://github.com/biface/oxiflow/issues/87), [DD-034](https://github.com/biface/oxiflow/issues/87))
- `Solver::solve()` on `ForwardEulerSolver`, `RK4Solver`, `BackwardEulerSolver`,
  `CrankNicolsonSolver`, `BDF2Solver` reduced to a single call to `solve_fixed_step()` (Ref [#88](https://github.com/biface/oxiflow/issues/88), [DD-035](https://github.com/biface/oxiflow/issues/88))
- `BoundaryCondition` now requires `Send + Sync` supertraits — additive; existing
  implementations (`DanckwertsInlet`, `DanckwertsOutlet`) already satisfy it trivially.
  Required because `SplitOperator` is the first place a `Domain` is owned rather than
  borrowed by a `Send + Sync`-bound type (`Solver: Send + Sync`) ([#45](https://github.com/biface/oxiflow/issues/45), [DA-037](https://github.com/biface/oxiflow/issues/90))

### Fixed

- `ForwardEulerSolver` — boundary-condition handling correction
- `RK4Solver` — boundary conditions now applied at every Runge-Kutta stage, not once per
  outer step; floating-point accumulation error in the multi-stage state combination

## [0.3.0] — 2026-06-03

### Added

- `PhysicalQuantity` indexed enum in `src/context/quantity.rs` — variants
  `Concentration { component }`, `Temperature`, `Pressure`, `Velocity { component }`,
  `Custom { name: Cow<'static, str>, component }` with idiomatic constructors
  (`concentration()`, `temperature()`, `pressure()`, `velocity()`, `custom(name)`) ([DD-010](https://github.com/biface/oxiflow/issues/10))
- `MultiDomainState` in `src/context/state.rs` — `HashMap<(DomainId, PhysicalQuantity), ContextValue>`
  composite key; explicit entry serialisation for external tool readability ([DD-010](https://github.com/biface/oxiflow/issues/10), [#38](https://github.com/biface/oxiflow/issues/38))
- `Interface` struct in `src/coupling/mod.rs` — source domain, target domain, optional label ([DA-011](https://github.com/biface/oxiflow/issues/11))
- `CouplingOperator` trait in `src/coupling/mod.rs` — `RequiresContext + Send + Sync` supertraits,
  object-safe, INV-3 active from v0.3.0 ([DA-011](https://github.com/biface/oxiflow/issues/11))
- `Scenario::with_coupling()` builder — registers a `CouplingOperator` and its `Interface` ([#73](https://github.com/biface/oxiflow/issues/73))
- `Scenario::couplings()`, `Scenario::interfaces()`, `Scenario::n_couplings()` accessors ([#73](https://github.com/biface/oxiflow/issues/73))
- `Scenario::context_requirements()` now aggregates variables from coupling operators (J3, [#73](https://github.com/biface/oxiflow/issues/73))
- `serde_json` in `[dev-dependencies]` for serde feature tests ([#69](https://github.com/biface/oxiflow/issues/69))
- Integration test `tests/coupling_proto.rs` — proto lahar–lake validates INV-3 structural
  contract (J3 exit criterion, [#73](https://github.com/biface/oxiflow/issues/73))

### Changed

- `src/context/mod.rs` — `quantity` and `state` modules declared and re-exported

### Feature flags

- `serde` — `PhysicalQuantity` and `MultiDomainState` annotated with
  `#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]` ([#69](https://github.com/biface/oxiflow/issues/69))

## [0.2.0] — 2026-05-02

### Added

- `BoundaryCondition` trait requires `RequiresContext + Debug` supertraits ([DA-008](https://github.com/biface/oxiflow/issues/8))
- `BoundaryLocation` enum with nine variants including `PhaseInterface`, `CouplingInterface`,
  `Custom(Cow<'static, str>)` ([DA-008](https://github.com/biface/oxiflow/issues/8))
- Calculator chain with hybrid two-path ordering: priority fast path + Kahn topological sort ([DD-009](https://github.com/biface/oxiflow/issues/9))
- Built-in calculator suite in `src/context/calculators/`: `TimeCalculator`, `TimeStepCalculator`,
  `FDGradientCalculator`, `FDLaplacianCalculator`, `TrapezoidalIntegral`, `ExternalTabulated` ([DD-009](https://github.com/biface/oxiflow/issues/9))
- `FDScheme` and `Interpolation` enums
- `ContextVariable::External::name` migrated from `&'static str` to `Cow<'static, str>`
- `OxiflowError::PreconditionFailed { context, message }` variant
- `SimulationSnapshot` two-phase restore contract ([DT-025](https://github.com/biface/oxiflow/issues/66))
- Danckwerts inlet feed concentration read from `ComputeContext` via `ContextVariable`
- Integration test `tests/solver_pipeline_chromatography.rs`

## [0.1.0] — 2026-04-12

### Added

- `ContextValue` generic enum: `Scalar`, `Vector`, `Matrix`, `Field2D`, `Boolean`
- `OxiflowError` typed error enum ([DD-004](https://github.com/biface/oxiflow/issues/4))
- `RequiresContext` trait — declarative variable dependency system
- `ComputeContext` — type-safe context access ([DA-005](https://github.com/biface/oxiflow/issues/5), [DA-006](https://github.com/biface/oxiflow/issues/6))
- `Mesh` abstract trait — INV-1 active ([DD-003](https://github.com/biface/oxiflow/issues/3))
- `UniformGrid1D` — structured 1D mesh implementation
- `PhysicalModel` trait — WHAT/HOW separation
- `Scenario` / `SolverConfiguration` / `Solver` architecture
- `DomainId` typed domain identifier
- `ContextVariable` typed key enum

## [0.0.5] — 2026-04-15

### Added

- `Cargo.toml`: full project structure — dependencies (`nalgebra`, `thiserror`),
  features skeleton (`parallel`, `serde`, `hdf5`), crates.io metadata, MSRV 1.80 ([#20](https://github.com/biface/oxiflow/issues/20))
- `src/`: module skeleton — `context`, `mesh`, `model`, `boundary`,
  `solver`, `operators`, `coupling` ([#21](https://github.com/biface/oxiflow/issues/21))
- CI pipeline `ci.yml`: `cargo fmt --check`, `cargo clippy -D warnings`,
  `cargo test`, `cargo doc` → GitHub Pages ([#23](https://github.com/biface/oxiflow/issues/23))
- `.codecov.yml`: 80% coverage threshold, `default` flag active ([#23](https://github.com/biface/oxiflow/issues/23))
- `CHANGELOG.md`: initial Keep a Changelog entry ([#23](https://github.com/biface/oxiflow/issues/23))

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
