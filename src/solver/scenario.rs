//! # Module `solver::scenario`
//!
//! Problem declaration — `Scenario` (WHAT pole, [DD-020](https://github.com/biface/oxiflow/issues/57), issue [#32](https://github.com/biface/oxiflow/issues/32)).
//!
//! ## Design
//!
//! `Scenario` is unified: it handles both single-domain (J1) and multi-domain
//! (J3) problems. The single-domain case is the degenerate case with one element
//! in `domains` and empty `couplings`. Ergonomic helpers cover the single-domain
//! case without verbosity; additional domains and coupling operators are added
//! without any API change ([DD-020](https://github.com/biface/oxiflow/issues/57)).

use std::collections::HashSet;
use std::sync::Arc;

use crate::boundary::{BoundaryCondition, BoundaryConditionSpec};
use crate::context::error::OxiflowError;
use crate::context::value::ContextValue;
use crate::context::variable::ContextVariable;
use crate::coupling::{CouplingOperator, Interface};
use crate::mesh::{Mesh, MeshSpec, UniformGrid1D};
use crate::model::traits::PhysicalModel;
use crate::model::{InitialStateSpec, ModelSpec, PrescribedInitialState};

// ── DomainId ──────────────────────────────────────────────────────────────────

/// Typed identifier for a domain within a multi-domain scenario.
///
/// A single-domain scenario uses `DomainId::default()` ("default"). In a
/// multi-domain scenario, each domain gets a meaningful identifier for
/// coupling resolution.
///
/// # Examples
///
/// ```rust
/// use oxiflow::solver::scenario::DomainId;
///
/// let id = DomainId::new("column");
/// assert_eq!(id.as_str(), "column");
///
/// let default_id = DomainId::default();
/// assert_eq!(default_id.as_str(), "default");
/// ```
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct DomainId(String);

impl DomainId {
    /// Creates a new domain identifier.
    pub fn new(id: impl Into<String>) -> Self {
        Self(id.into())
    }

    /// Returns the identifier as a string slice.
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl Default for DomainId {
    fn default() -> Self {
        Self("default".to_string())
    }
}

impl std::fmt::Display for DomainId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl From<&str> for DomainId {
    fn from(s: &str) -> Self {
        Self(s.to_string())
    }
}

// ── Domain ────────────────────────────────────────────────────────────────────

/// Single physical domain: model + mesh + boundary conditions.
///
/// `boundary_conditions` is empty unless conditions are added with the
/// `Domain::with_boundary_conditions` builder.
///
/// # Serialisation
///
/// `Domain` does not implement `serde::Serialize` / `serde::Deserialize`.
/// It holds `Box<dyn PhysicalModel>`, `Box<dyn Mesh>`, and
/// `Box<dyn BoundaryCondition>` (trait objects), which cannot be serialised
/// directly. See `SimulationSnapshot` ([DT-025](https://github.com/biface/oxiflow/issues/66) Option B).
#[non_exhaustive]
pub struct Domain {
    /// Unique identifier for this domain.
    pub id: DomainId,
    /// Physical model — declares and computes field equations.
    pub model: Box<dyn PhysicalModel>,
    /// Spatial mesh — INV-1.
    pub mesh: Box<dyn Mesh>,
    /// Boundary conditions applied after context calculation, before physics.
    pub boundary_conditions: Vec<Box<dyn BoundaryCondition>>,
}

impl Domain {
    /// Creates a new domain with the given id, model, and mesh.
    ///
    /// `boundary_conditions` is initialised to an empty list. Use
    /// `with_boundary_conditions` to attach BCs.
    pub fn new(
        id: impl Into<DomainId>,
        model: Box<dyn PhysicalModel>,
        mesh: Box<dyn Mesh>,
    ) -> Self {
        Self {
            id: id.into(),
            model,
            mesh,
            boundary_conditions: vec![],
        }
    }

    /// Attaches boundary conditions to this domain.
    ///
    /// Replaces any previously set boundary conditions.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use oxiflow::boundary::{DanckwertsInlet, DanckwertsOutlet};
    /// use oxiflow::mesh::UniformGrid1D;
    /// use oxiflow::solver::Domain;
    /// # use nalgebra::DVector;
    /// # use oxiflow::context::{ComputeContext, ContextValue, ContextVariable, OxiflowError};
    /// # use oxiflow::mesh::Mesh;
    /// # use oxiflow::model::{PhysicalModel, RequiresContext};
    /// # struct Decay;
    /// # impl RequiresContext for Decay {
    /// #     fn required_variables(&self) -> Vec<ContextVariable> { vec![] }
    /// # }
    /// # impl PhysicalModel for Decay {
    /// #     fn compute_physics(
    /// #         &self,
    /// #         state: &ContextValue,
    /// #         _ctx: &ComputeContext,
    /// #     ) -> Result<ContextValue, OxiflowError> {
    /// #         Ok(ContextValue::ScalarField(-state.as_scalar_field()?.clone()))
    /// #     }
    /// #     fn initial_state(&self, mesh: &dyn Mesh) -> ContextValue {
    /// #         ContextValue::ScalarField(DVector::from_element(mesh.n_dof(), 1.0))
    /// #     }
    /// #     fn name(&self) -> &str { "decay" }
    /// # }
    /// # let model = Box::new(Decay);
    /// let mesh = Box::new(UniformGrid1D::new(50, 0.0, 1.0)?);
    /// let inlet_bc = DanckwertsInlet::new(
    ///     1e-3, // dispersion
    ///     0.1,  // velocity
    ///     ContextVariable::External { name: "feed".into() },
    /// );
    /// let outlet_bc = DanckwertsOutlet::new();
    /// let domain = Domain::new("column", model, mesh)
    ///     .with_boundary_conditions(vec![Box::new(inlet_bc), Box::new(outlet_bc)]);
    /// # Ok::<(), OxiflowError>(())
    /// ```
    pub fn with_boundary_conditions(mut self, bcs: Vec<Box<dyn BoundaryCondition>>) -> Self {
        self.boundary_conditions = bcs;
        self
    }
}

// ── DomainSpec ────────────────────────────────────────────────────────────────

/// External, serialisable description of a [`Domain`]: its mesh, model,
/// boundary conditions and initial state, each given by its own spec
/// ([DT-027](https://github.com/biface/oxiflow/issues/75) amendment 1,
/// [#108](https://github.com/biface/oxiflow/issues/108)).
///
/// A `DomainSpec` is consumed once, by `TryFrom<DomainSpec> for Domain`:
/// the mesh is built once and shared between the domain and the model that
/// needs it (a flux scheme), the model is wrapped in a
/// [`PrescribedInitialState`] when `initial_state` is given, and the domain
/// is checked against its mesh as [`Scenario::validate`] would. `Domain`
/// itself stays non-serialisable ([DT-025](https://github.com/biface/oxiflow/issues/66)):
/// the spec is the external form, never a serialised domain.
///
/// With the `serde` feature, a domain reads from a configuration file:
///
/// ```text
/// id: column
/// mesh:
///   UniformGrid1D: { n_points: 200, x_start: 0.0, x_end: 0.25 }
/// model:
///   Discretized:
///     scheme:
///       FVUpwind: { velocity: 1.0e-3, diffusion: 1.0e-6, boundary: Truncation }
/// boundary_conditions:
///   - DanckwertsInlet: { dispersion: 1.0e-6, velocity: 1.0e-3, feed_variable: { External: { name: feed } } }
///   - DanckwertsOutlet: {}
/// initial_state: { Uniform: 0.0 }   # optional, the model's own state otherwise
/// ```
///
/// # Examples
///
/// ```rust
/// use oxiflow::boundary::{BoundaryConditionSpec, TimeFunctionSpec};
/// use oxiflow::mesh::{BoundaryRegion, FaceSide, MeshSpec};
/// use oxiflow::model::{InitialStateSpec, ModelSpec};
/// use oxiflow::operators::{FluxBoundarySpec, FluxSchemeSpec};
/// use oxiflow::solver::{Domain, DomainSpec};
///
/// let spec = DomainSpec::new(
///     "column",
///     MeshSpec::UniformGrid1D { n_points: 51, x_start: 0.0, x_end: 1.0 },
///     ModelSpec::Discretized {
///         scheme: FluxSchemeSpec::FVUpwind {
///             velocity: 1.0,
///             diffusion: 1e-3,
///             boundary: FluxBoundarySpec::Truncation,
///         },
///         name: None,
///     },
/// )
/// .with_boundary_conditions(vec![BoundaryConditionSpec::Dirichlet {
///     region: BoundaryRegion::Face { axis: 0, side: FaceSide::Lower },
///     value: TimeFunctionSpec::Constant(1.0),
/// }])
/// .with_initial_state(InitialStateSpec::Uniform(0.0));
///
/// let domain = Domain::try_from(spec)?;
/// assert_eq!(domain.mesh.n_dof(), 51);
/// assert_eq!(domain.model.name(), "fv_upwind");
/// # Ok::<(), oxiflow::context::OxiflowError>(())
/// ```
#[non_exhaustive]
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Deserialize))]
pub struct DomainSpec {
    /// Identifier of the domain.
    pub id: DomainId,
    /// The mesh.
    pub mesh: MeshSpec,
    /// The physical model.
    pub model: ModelSpec,
    /// Boundary conditions, none when absent.
    #[cfg_attr(feature = "serde", serde(default))]
    pub boundary_conditions: Vec<BoundaryConditionSpec>,
    /// Initial state; the model's own when absent.
    #[cfg_attr(feature = "serde", serde(default))]
    pub initial_state: Option<InitialStateSpec>,
}

impl DomainSpec {
    /// A domain `id` with `mesh` and `model`, no boundary condition, and the
    /// model's own initial state.
    pub fn new(id: impl Into<DomainId>, mesh: MeshSpec, model: ModelSpec) -> Self {
        Self {
            id: id.into(),
            mesh,
            model,
            boundary_conditions: vec![],
            initial_state: None,
        }
    }

    /// Sets the boundary conditions, replacing any set before.
    pub fn with_boundary_conditions(mut self, conditions: Vec<BoundaryConditionSpec>) -> Self {
        self.boundary_conditions = conditions;
        self
    }

    /// Sets the initial state.
    pub fn with_initial_state(mut self, state: InitialStateSpec) -> Self {
        self.initial_state = Some(state);
        self
    }
}

impl TryFrom<DomainSpec> for Domain {
    type Error = OxiflowError;

    /// Builds the domain `spec` describes.
    ///
    /// # Errors
    ///
    /// - `OxiflowError::PreconditionFailed` if the mesh parameters are
    ///   invalid (see [`UniformGrid1D::new`]);
    /// - `OxiflowError::ExternalData` if a tabulated value of a boundary
    ///   condition has fewer than two points or unordered times;
    /// - `OxiflowError::InvalidDomain` if the initial state does not have one
    ///   value per node of the mesh.
    fn try_from(spec: DomainSpec) -> Result<Self, OxiflowError> {
        let grid = match spec.mesh {
            MeshSpec::UniformGrid1D {
                n_points,
                x_start,
                x_end,
            } => Arc::new(UniformGrid1D::new(n_points, x_start, x_end)?),
        };
        let mut model = spec.model.into_model(Arc::clone(&grid));
        if let Some(initial_state) = spec.initial_state {
            let state = initial_state.into_state(grid.n_dof());
            model = Box::new(PrescribedInitialState::new(model, state));
        }
        let boundary_conditions = spec
            .boundary_conditions
            .into_iter()
            .map(BoundaryConditionSpec::into_condition)
            .collect::<Result<Vec<_>, _>>()?;
        let domain = Domain::new(spec.id, model, Box::new(grid))
            .with_boundary_conditions(boundary_conditions);
        check_domain(&domain)?;
        Ok(domain)
    }
}

// ── Scenario ──────────────────────────────────────────────────────────────────

/// Complete problem declaration — WHAT pole.
///
/// Holds one or more `Domain`s, coupling operators between them (J3), and
/// the simulation start time. Single-domain problems use the ergonomic helper
/// `Scenario::single()`. Multi-domain problems add domains and couplings without
/// any breaking change ([DD-020](https://github.com/biface/oxiflow/issues/57)).
///
/// `Scenario` is declarative — it contains no solving logic. The `Solver`
/// receives it and validates it via `context_requirements()` before solving.
///
/// # Serialisation
///
/// `Scenario` does not implement `serde::Serialize` / `serde::Deserialize`.
/// It contains `Vec<Domain>`, which holds trait objects (`Box<dyn PhysicalModel>`,
/// `Box<dyn Mesh>`). Restoring a simulation from a snapshot requires user code
/// to reconstruct `Scenario` from its own configuration and inject the physical
/// state from `SimulationSnapshot` ([DT-025](https://github.com/biface/oxiflow/issues/66) Option B, v0.6.0).
///
/// # Examples
///
/// ```rust
/// use oxiflow::solver::scenario::Scenario;
/// use oxiflow::model::traits::{PhysicalModel, RequiresContext};
/// use oxiflow::context::variable::ContextVariable;
/// use oxiflow::context::value::ContextValue;
/// use oxiflow::context::compute::ComputeContext;
/// use oxiflow::context::error::OxiflowError;
/// use oxiflow::mesh::{Mesh, UniformGrid1D};
/// use nalgebra::DVector;
///
/// struct ConstantModel;
/// impl RequiresContext for ConstantModel {
///     fn required_variables(&self) -> Vec<ContextVariable> { vec![] }
/// }
/// impl PhysicalModel for ConstantModel {
///     fn compute_physics(&self, s: &ContextValue, _: &ComputeContext)
///         -> Result<ContextValue, OxiflowError> { Ok(s.clone()) }
///     fn initial_state(&self, mesh: &dyn Mesh) -> ContextValue {
///         ContextValue::ScalarField(DVector::from_element(mesh.n_dof(), 0.0))
///     }
///     fn name(&self) -> &str { "constant" }
/// }
///
/// let mesh = Box::new(UniformGrid1D::new(10, 0.0, 1.0).unwrap());
/// let scenario = Scenario::single(Box::new(ConstantModel), mesh);
/// assert_eq!(scenario.n_domains(), 1);
/// ```
pub struct Scenario {
    /// Physical domains — at least one required.
    domains: Vec<Domain>,
    /// Coupling operators between domains (J3, [DA-011](https://github.com/biface/oxiflow/issues/11), INV-3).
    couplings: Vec<Box<dyn CouplingOperator>>,
    /// Interfaces shared between domains (J3, [DA-011](https://github.com/biface/oxiflow/issues/11)).
    interfaces: Vec<Interface>,
    /// Simulation start time.
    pub t_start: f64,
}

impl Scenario {
    // ── Constructors ──────────────────────────────────────────────────────────

    /// Creates a single-domain scenario (J1 default).
    ///
    /// Uses `DomainId::default()` as the domain identifier.
    /// `t_start` defaults to `0.0`.
    pub fn single(model: Box<dyn PhysicalModel>, mesh: Box<dyn Mesh>) -> Self {
        Self {
            domains: vec![Domain::new(DomainId::default(), model, mesh)],
            couplings: vec![],
            interfaces: vec![],
            t_start: 0.0,
        }
    }

    /// Creates a single-domain scenario with a custom start time.
    pub fn single_from(model: Box<dyn PhysicalModel>, mesh: Box<dyn Mesh>, t_start: f64) -> Self {
        Self {
            domains: vec![Domain::new(DomainId::default(), model, mesh)],
            couplings: vec![],
            interfaces: vec![],
            t_start,
        }
    }

    /// Creates a multi-domain scenario from a list of domains (J3 path).
    ///
    /// Domains are provided as pre-built `Domain` instances, each under its
    /// own identifier: the solvers and the orchestrator key states, histories
    /// and couplings by [`DomainId`].
    ///
    /// # Errors
    ///
    /// `OxiflowError::InvalidDomain` if `domains` is empty or if two domains
    /// share an identifier ([#230](https://github.com/biface/oxiflow/issues/230)).
    pub fn multi(domains: Vec<Domain>) -> Result<Self, OxiflowError> {
        if domains.is_empty() {
            return Err(OxiflowError::InvalidDomain(
                "Scenario requires at least one domain".into(),
            ));
        }
        check_unique_ids(&domains)?;
        Ok(Self {
            domains,
            couplings: vec![],
            interfaces: vec![],
            t_start: 0.0,
        })
    }

    /// Sets the simulation start time.
    pub fn with_t_start(mut self, t_start: f64) -> Self {
        self.t_start = t_start;
        self
    }

    /// Adds a coupling operator and its associated interface (J3, INV-3).
    ///
    /// The interface is extracted from the operator via `CouplingOperator::interface()`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use oxiflow::context::MultiDomainState;
    /// # use oxiflow::coupling::{CouplingContext, CouplingOperator, Interface};
    /// use oxiflow::mesh::UniformGrid1D;
    /// use oxiflow::solver::{Domain, DomainId, Scenario};
    /// # use nalgebra::DVector;
    /// # use oxiflow::context::{ComputeContext, ContextValue, ContextVariable, OxiflowError};
    /// # use oxiflow::mesh::Mesh;
    /// # use oxiflow::model::{PhysicalModel, RequiresContext};
    /// # struct Decay;
    /// # impl RequiresContext for Decay {
    /// #     fn required_variables(&self) -> Vec<ContextVariable> { vec![] }
    /// # }
    /// # impl PhysicalModel for Decay {
    /// #     fn compute_physics(
    /// #         &self,
    /// #         state: &ContextValue,
    /// #         _ctx: &ComputeContext,
    /// #     ) -> Result<ContextValue, OxiflowError> {
    /// #         Ok(ContextValue::ScalarField(-state.as_scalar_field()?.clone()))
    /// #     }
    /// #     fn initial_state(&self, mesh: &dyn Mesh) -> ContextValue {
    /// #         ContextValue::ScalarField(DVector::from_element(mesh.n_dof(), 1.0))
    /// #     }
    /// #     fn name(&self) -> &str { "decay" }
    /// # }
    /// # struct Transfer { interface: Interface }
    /// # impl RequiresContext for Transfer {
    /// #     fn required_variables(&self) -> Vec<ContextVariable> { vec![] }
    /// # }
    /// # impl CouplingOperator for Transfer {
    /// #     fn apply(
    /// #         &self,
    /// #         states: &MultiDomainState,
    /// #         _ctx: &CouplingContext,
    /// #         _interface: &Interface,
    /// #     ) -> Result<MultiDomainState, OxiflowError> {
    /// #         Ok(states.clone())
    /// #     }
    /// #     fn interface(&self) -> &Interface { &self.interface }
    /// # }
    /// let domains = vec![
    ///     Domain::new("lahar", Box::new(Decay), Box::new(UniformGrid1D::new(20, 0.0, 1.0)?)),
    ///     Domain::new("lake", Box::new(Decay), Box::new(UniformGrid1D::new(10, 0.0, 1.0)?)),
    /// ];
    /// # let my_coupling_op = Transfer {
    /// #     interface: Interface::new(DomainId::new("lahar"), DomainId::new("lake")),
    /// # };
    /// let scenario = Scenario::multi(domains)?
    ///     .with_coupling(Box::new(my_coupling_op));
    /// # Ok::<(), OxiflowError>(())
    /// ```
    pub fn with_coupling(mut self, coupling: Box<dyn CouplingOperator>) -> Self {
        self.interfaces.push(coupling.interface().clone());
        self.couplings.push(coupling);
        self
    }

    // ── Accessors ─────────────────────────────────────────────────────────────

    /// Returns the number of domains.
    pub fn n_domains(&self) -> usize {
        self.domains.len()
    }

    /// Returns the number of coupling operators.
    pub fn n_couplings(&self) -> usize {
        self.couplings.len()
    }

    /// Returns a reference to all coupling operators.
    pub fn couplings(&self) -> &[Box<dyn CouplingOperator>] {
        &self.couplings
    }

    /// Returns a reference to all interfaces.
    pub fn interfaces(&self) -> &[Interface] {
        &self.interfaces
    }

    /// Returns a reference to all domains.
    pub fn domains(&self) -> &[Domain] {
        &self.domains
    }

    /// Returns the single domain — convenience for J1 single-domain scenarios.
    ///
    /// # Errors
    ///
    /// Returns `OxiflowError::InvalidDomain` if the scenario has more than one domain.
    pub fn single_domain(&self) -> Result<&Domain, OxiflowError> {
        if self.domains.len() != 1 {
            return Err(OxiflowError::InvalidDomain(format!(
                "expected 1 domain, found {}",
                self.domains.len()
            )));
        }
        Ok(&self.domains[0])
    }

    // ── Context aggregation ───────────────────────────────────────────────────

    /// Aggregates context variable requirements from all domains.
    ///
    /// Collects the requirements of the models, of the boundary conditions
    /// and of the coupling operators, sorted and deduplicated.
    pub fn context_requirements(&self) -> Vec<ContextVariable> {
        let mut requirements: Vec<ContextVariable> = self
            .domains
            .iter()
            .flat_map(|d| d.model.required_variables())
            .collect();

        // J2: extend with BC requirements
        self.domains.iter().for_each(|d| {
            requirements.extend(
                d.boundary_conditions
                    .iter()
                    .flat_map(|bc| bc.required_variables()),
            );
        });

        // J3: extend with coupling operator requirements
        self.couplings.iter().for_each(|c| {
            requirements.extend(c.required_variables());
        });

        requirements.sort_by(|a, b| format!("{a:?}").cmp(&format!("{b:?}")));
        requirements.dedup();
        requirements
    }

    // ── Validation ────────────────────────────────────────────────────────────

    /// Validates scenario consistency before solving.
    ///
    /// Verifies that at least one domain is present, that no two domains
    /// share an identifier, that every coupling interface names two domains
    /// of the scenario ([#230](https://github.com/biface/oxiflow/issues/230)),
    /// that no domain has an empty mesh, and that each domain's initial state
    /// has one row per mesh node: the length of a `ScalarField`, the row
    /// count of a `VectorField`, against `mesh.n_dof()`. Other `ContextValue`
    /// variants are not sized by the mesh and are left unchecked. The initial
    /// state is checked here because [`PhysicalModel::initial_state`] cannot
    /// report an error, and a state prescribed from outside the model
    /// ([`PrescribedInitialState`]) can be of any size.
    ///
    /// Context requirements are not checked here: a requirement that no
    /// calculator provides is reported when the calculator chain is built
    /// ([`build_calculator_chain`](crate::solver::chain::build_calculator_chain)).
    ///
    /// [`Solver::solve`](crate::solver::Solver::solve) and the orchestrator
    /// call it before the first step; a caller that drives
    /// [`SteppableSolver::step`](crate::solver::methods::SteppableSolver::step)
    /// in its own loop calls it itself.
    ///
    /// # Errors
    ///
    /// `OxiflowError::InvalidDomain` if the scenario has no domain, if two
    /// domains share an identifier, if an interface names a domain the
    /// scenario does not hold, if a domain's mesh is empty, or if a domain's
    /// initial state does not match its mesh.
    pub fn validate(&self) -> Result<(), OxiflowError> {
        if self.domains.is_empty() {
            return Err(OxiflowError::InvalidDomain(
                "Scenario has no domains".into(),
            ));
        }
        check_unique_ids(&self.domains)?;
        self.check_interfaces()?;
        self.domains.iter().try_for_each(check_domain)
    }

    /// Checks that every coupling interface names two domains of the
    /// scenario: the orchestrator reads and writes the coupled states under
    /// these identifiers.
    fn check_interfaces(&self) -> Result<(), OxiflowError> {
        for interface in &self.interfaces {
            for id in [interface.source(), interface.target()] {
                if !self.domains.iter().any(|domain| &domain.id == id) {
                    return Err(OxiflowError::InvalidDomain(format!(
                        "{interface} names domain '{id}', which the scenario does not hold"
                    )));
                }
            }
        }
        Ok(())
    }
}

/// Checks that no two of `domains` share an identifier.
fn check_unique_ids(domains: &[Domain]) -> Result<(), OxiflowError> {
    let mut seen = HashSet::with_capacity(domains.len());
    for domain in domains {
        if !seen.insert(&domain.id) {
            return Err(OxiflowError::InvalidDomain(format!(
                "two domains share the identifier '{}'",
                domain.id
            )));
        }
    }
    Ok(())
}

/// Checks that `domain` has a non-empty mesh and an initial state sized by
/// it: the per-domain part of [`Scenario::validate`], also run on a domain
/// built from a [`DomainSpec`].
fn check_domain(domain: &Domain) -> Result<(), OxiflowError> {
    let n_dof = domain.mesh.n_dof();
    if n_dof == 0 {
        return Err(OxiflowError::InvalidDomain(format!(
            "domain '{}' has empty mesh",
            domain.id
        )));
    }
    let rows = match domain.model.initial_state(domain.mesh.as_ref()) {
        ContextValue::ScalarField(field) => Some(field.len()),
        ContextValue::VectorField(field) => Some(field.nrows()),
        _ => None,
    };
    if let Some(rows) = rows.filter(|&rows| rows != n_dof) {
        return Err(OxiflowError::InvalidDomain(format!(
            "domain '{}': initial state has {rows} rows, its mesh has {n_dof} nodes",
            domain.id
        )));
    }
    Ok(())
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::context::compute::ComputeContext;
    use crate::context::error::OxiflowError;
    use crate::context::value::ContextValue;
    use crate::context::variable::ContextVariable;
    use crate::mesh::structured::UniformGrid1D;
    use crate::model::traits::{PhysicalModel, RequiresContext};
    use nalgebra::DVector;

    // ── Fixtures ──────────────────────────────────────────────────────────────

    struct NeedsTime;
    impl RequiresContext for NeedsTime {
        fn required_variables(&self) -> Vec<ContextVariable> {
            vec![ContextVariable::Time]
        }
    }
    impl PhysicalModel for NeedsTime {
        fn compute_physics(
            &self,
            s: &ContextValue,
            _: &ComputeContext,
        ) -> Result<ContextValue, OxiflowError> {
            Ok(s.clone())
        }
        fn initial_state(&self, mesh: &dyn Mesh) -> ContextValue {
            ContextValue::ScalarField(DVector::from_element(mesh.n_dof(), 0.0))
        }
        fn name(&self) -> &str {
            "needs_time"
        }
    }

    struct NeedsGradient;
    impl RequiresContext for NeedsGradient {
        fn required_variables(&self) -> Vec<ContextVariable> {
            vec![ContextVariable::SpatialGradient {
                dimension: 0,
                component: None,
            }]
        }
    }
    impl PhysicalModel for NeedsGradient {
        fn compute_physics(
            &self,
            s: &ContextValue,
            _: &ComputeContext,
        ) -> Result<ContextValue, OxiflowError> {
            Ok(s.clone())
        }
        fn initial_state(&self, mesh: &dyn Mesh) -> ContextValue {
            ContextValue::ScalarField(DVector::from_element(mesh.n_dof(), 0.0))
        }
        fn name(&self) -> &str {
            "needs_gradient"
        }
    }

    fn make_mesh() -> Box<dyn Mesh> {
        Box::new(UniformGrid1D::new(10, 0.0, 1.0).unwrap())
    }

    // ── DomainId ──────────────────────────────────────────────────────────────

    #[test]
    fn domain_id_new() {
        let id = DomainId::new("col_a");
        assert_eq!(id.as_str(), "col_a");
    }

    #[test]
    fn domain_id_default_is_default() {
        assert_eq!(DomainId::default().as_str(), "default");
    }

    #[test]
    fn domain_id_from_str() {
        let id: DomainId = "lake".into();
        assert_eq!(id.as_str(), "lake");
    }

    #[test]
    fn domain_id_display() {
        assert_eq!(format!("{}", DomainId::new("river")), "river");
    }

    #[test]
    fn domain_id_equality() {
        assert_eq!(DomainId::new("a"), DomainId::new("a"));
        assert_ne!(DomainId::new("a"), DomainId::new("b"));
    }

    // ── Scenario::single ──────────────────────────────────────────────────────

    #[test]
    fn single_creates_one_domain() {
        let s = Scenario::single(Box::new(NeedsTime), make_mesh());
        assert_eq!(s.n_domains(), 1);
    }

    #[test]
    fn single_t_start_defaults_to_zero() {
        let s = Scenario::single(Box::new(NeedsTime), make_mesh());
        assert_eq!(s.t_start, 0.0);
    }

    #[test]
    fn single_from_sets_t_start() {
        let s = Scenario::single_from(Box::new(NeedsTime), make_mesh(), 5.0);
        assert_eq!(s.t_start, 5.0);
    }

    #[test]
    fn with_t_start_builder() {
        let s = Scenario::single(Box::new(NeedsTime), make_mesh()).with_t_start(2.5);
        assert_eq!(s.t_start, 2.5);
    }

    // ── Scenario::multi ───────────────────────────────────────────────────────

    #[test]
    fn multi_with_two_domains() {
        let d1 = Domain::new("a", Box::new(NeedsTime), make_mesh());
        let d2 = Domain::new("b", Box::new(NeedsGradient), make_mesh());
        let s = Scenario::multi(vec![d1, d2]).unwrap();
        assert_eq!(s.n_domains(), 2);
    }

    #[test]
    fn multi_empty_returns_error() {
        assert!(Scenario::multi(vec![]).is_err());
    }

    #[test]
    fn multi_rejects_two_domains_with_the_same_identifier() {
        let d1 = Domain::new("lake", Box::new(NeedsTime), make_mesh());
        let d2 = Domain::new("lake", Box::new(NeedsGradient), make_mesh());
        let err = Scenario::multi(vec![d1, d2]).err().unwrap();
        assert!(matches!(err, OxiflowError::InvalidDomain(ref m) if m.contains("'lake'")));
    }

    // ── single_domain ─────────────────────────────────────────────────────────

    #[test]
    fn single_domain_ok_for_one_domain() {
        let s = Scenario::single(Box::new(NeedsTime), make_mesh());
        assert!(s.single_domain().is_ok());
        assert_eq!(s.single_domain().unwrap().id, DomainId::default());
    }

    #[test]
    fn single_domain_err_for_multi() {
        let d1 = Domain::new("a", Box::new(NeedsTime), make_mesh());
        let d2 = Domain::new("b", Box::new(NeedsGradient), make_mesh());
        let s = Scenario::multi(vec![d1, d2]).unwrap();
        assert!(s.single_domain().is_err());
    }

    // ── context_requirements ─────────────────────────────────────────────────

    #[test]
    fn requirements_from_single_domain() {
        let s = Scenario::single(Box::new(NeedsTime), make_mesh());
        let reqs = s.context_requirements();
        assert!(reqs.contains(&ContextVariable::Time));
    }

    #[test]
    fn requirements_aggregated_and_deduped_across_domains() {
        let d1 = Domain::new("a", Box::new(NeedsTime), make_mesh());
        let d2 = Domain::new("b", Box::new(NeedsTime), make_mesh());
        let s = Scenario::multi(vec![d1, d2]).unwrap();
        // Time appears in both domains — dedup → only once
        let reqs = s.context_requirements();
        assert_eq!(
            reqs.iter().filter(|v| **v == ContextVariable::Time).count(),
            1
        );
    }

    #[test]
    fn requirements_union_of_all_domains() {
        let d1 = Domain::new("a", Box::new(NeedsTime), make_mesh());
        let d2 = Domain::new("b", Box::new(NeedsGradient), make_mesh());
        let s = Scenario::multi(vec![d1, d2]).unwrap();
        let reqs = s.context_requirements();
        assert!(reqs.contains(&ContextVariable::Time));
        assert!(reqs.contains(&ContextVariable::SpatialGradient {
            dimension: 0,
            component: None
        }));
    }

    // ── validate ──────────────────────────────────────────────────────────────

    #[test]
    fn validate_ok_for_valid_scenario() {
        let s = Scenario::single(Box::new(NeedsTime), make_mesh());
        assert!(s.validate().is_ok());
    }

    /// `ZeroDerivative` started from `state`, on the 10-node `make_mesh()`.
    fn prescribed_scenario(state: ContextValue) -> Scenario {
        use crate::model::{PrescribedInitialState, ZeroDerivative};
        let model = PrescribedInitialState::new(Box::new(ZeroDerivative { initial: 0.0 }), state);
        Scenario::single(Box::new(model), make_mesh())
    }

    #[test]
    fn validate_accepts_initial_fields_sized_by_the_mesh() {
        use nalgebra::DMatrix;
        let scalar = ContextValue::ScalarField(DVector::zeros(10));
        let vector = ContextValue::VectorField(DMatrix::zeros(10, 3));
        assert!(prescribed_scenario(scalar).validate().is_ok());
        assert!(prescribed_scenario(vector).validate().is_ok());
    }

    #[test]
    fn validate_rejects_a_scalar_initial_field_of_the_wrong_length() {
        let state = ContextValue::ScalarField(DVector::zeros(9));
        let err = prescribed_scenario(state).validate().unwrap_err();
        assert!(matches!(err, OxiflowError::InvalidDomain(ref m)
            if m.contains("9 rows") && m.contains("10 nodes")));
    }

    #[test]
    fn validate_rejects_a_vector_initial_field_with_the_wrong_row_count() {
        use nalgebra::DMatrix;
        let state = ContextValue::VectorField(DMatrix::zeros(11, 2));
        let err = prescribed_scenario(state).validate().unwrap_err();
        assert!(matches!(err, OxiflowError::InvalidDomain(ref m)
            if m.contains("11 rows") && m.contains("10 nodes")));
    }

    #[test]
    fn validate_leaves_initial_values_not_sized_by_the_mesh_unchecked() {
        assert!(prescribed_scenario(ContextValue::Scalar(1.0))
            .validate()
            .is_ok());
    }

    #[test]
    fn validate_names_the_domain_whose_initial_state_is_mis_sized() {
        use crate::model::{PrescribedInitialState, ZeroDerivative};
        let wrong = PrescribedInitialState::new(
            Box::new(ZeroDerivative { initial: 0.0 }),
            ContextValue::ScalarField(DVector::zeros(3)),
        );
        let s = Scenario::multi(vec![
            Domain::new("first", Box::new(NeedsTime), make_mesh()),
            Domain::new("second", Box::new(wrong), make_mesh()),
        ])
        .unwrap();
        let err = s.validate().unwrap_err();
        assert!(matches!(err, OxiflowError::InvalidDomain(ref m) if m.contains("'second'")));
    }

    /// Two domains, `a` and `b`, coupled across `interface`.
    fn coupled_scenario(interface: crate::coupling::Interface) -> Scenario {
        use crate::coupling::PassthroughCoupling;
        Scenario::multi(vec![
            Domain::new("a", Box::new(NeedsTime), make_mesh()),
            Domain::new("b", Box::new(NeedsTime), make_mesh()),
        ])
        .unwrap()
        .with_coupling(Box::new(PassthroughCoupling { interface }))
    }

    #[test]
    fn validate_accepts_an_interface_between_two_domains_of_the_scenario() {
        use crate::coupling::Interface;
        let s = coupled_scenario(Interface::new(DomainId::new("a"), DomainId::new("b")));
        assert!(s.validate().is_ok());
    }

    #[test]
    fn validate_rejects_an_interface_on_an_unknown_source() {
        use crate::coupling::Interface;
        let s = coupled_scenario(Interface::new(DomainId::new("river"), DomainId::new("b")));
        let err = s.validate().unwrap_err();
        assert!(matches!(err, OxiflowError::InvalidDomain(ref m)
            if m.contains("Interface(river → b)") && m.contains("'river'")));
    }

    #[test]
    fn validate_rejects_an_interface_on_an_unknown_target() {
        use crate::coupling::Interface;
        let s = coupled_scenario(Interface::new(DomainId::new("a"), DomainId::new("lake")));
        let err = s.validate().unwrap_err();
        assert!(matches!(err, OxiflowError::InvalidDomain(ref m) if m.contains("'lake'")));
    }

    #[test]
    fn validate_rejects_two_domains_with_the_same_identifier() {
        // `multi` refuses them; `validate` guards any other way to build the
        // list, so the check is exercised on the list directly.
        let domains = vec![
            Domain::new("lake", Box::new(NeedsTime), make_mesh()),
            Domain::new("lake", Box::new(NeedsTime), make_mesh()),
        ];
        let s = Scenario {
            domains,
            couplings: vec![],
            interfaces: vec![],
            t_start: 0.0,
        };
        let err = s.validate().unwrap_err();
        assert!(matches!(err, OxiflowError::InvalidDomain(ref m) if m.contains("'lake'")));
    }

    // ── BoundaryCondition integration ─────────────────────────────────────────

    use crate::boundary::BoundaryCondition;
    use crate::context::compute::ComputeContext as Ctx;

    #[derive(Debug)]
    struct TimeDependentBC;

    impl RequiresContext for TimeDependentBC {
        fn required_variables(&self) -> Vec<ContextVariable> {
            vec![ContextVariable::Time]
        }
    }

    impl BoundaryCondition for TimeDependentBC {
        fn boundary_type(&self) -> crate::boundary::BoundaryType {
            crate::boundary::BoundaryType::Dirichlet
        }
        fn apply(
            &self,
            _state: &mut DVector<f64>,
            _ctx: &Ctx,
            _mesh: &dyn Mesh,
        ) -> Result<(), OxiflowError> {
            Ok(())
        }
    }

    #[test]
    fn domain_with_boundary_conditions_stores_bcs() {
        let bc: Box<dyn BoundaryCondition> = Box::new(TimeDependentBC);
        let domain =
            Domain::new("col", Box::new(NeedsTime), make_mesh()).with_boundary_conditions(vec![bc]);
        assert_eq!(domain.boundary_conditions.len(), 1);
    }

    #[test]
    fn domain_new_has_empty_bcs() {
        let domain = Domain::new("col", Box::new(NeedsTime), make_mesh());
        assert!(domain.boundary_conditions.is_empty());
    }

    #[test]
    fn context_requirements_includes_bc_variables() {
        let bc: Box<dyn BoundaryCondition> = Box::new(TimeDependentBC);
        let domain = Domain::new("col", Box::new(NeedsGradient), make_mesh())
            .with_boundary_conditions(vec![bc]);
        let scenario = Scenario::multi(vec![domain]).unwrap();
        let reqs = scenario.context_requirements();
        // NeedsGradient requires SpatialGradient; TimeDependentBC requires Time.
        assert!(reqs.contains(&ContextVariable::Time));
        assert!(reqs.contains(&ContextVariable::SpatialGradient {
            dimension: 0,
            component: None
        }));
    }

    #[test]
    fn context_requirements_deduplicates_bc_and_model_variables() {
        // Both model and BC require Time — must appear only once.
        let bc: Box<dyn BoundaryCondition> = Box::new(TimeDependentBC);
        let domain =
            Domain::new("col", Box::new(NeedsTime), make_mesh()).with_boundary_conditions(vec![bc]);
        let scenario = Scenario::multi(vec![domain]).unwrap();
        let reqs = scenario.context_requirements();
        assert_eq!(
            reqs.iter().filter(|v| **v == ContextVariable::Time).count(),
            1
        );
    }
}
