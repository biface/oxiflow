//! # Module `solver::scenario`
//!
//! Problem declaration — `Scenario` (WHAT pole, DD-020, issue #32).
//!
//! ## Design
//!
//! `Scenario` is unified: it handles both single-domain (J1) and multi-domain
//! (J3) problems. The single-domain case is the degenerate case with one element
//! in `domains` and empty `couplings`. Ergonomic helpers cover the J1 use case
//! without verbosity. At J3, additional domains and coupling operators are added
//! without any API change (DD-020).

use crate::boundary::BoundaryCondition;
use crate::context::error::OxiflowError;
use crate::context::variable::ContextVariable;
use crate::coupling::{CouplingOperator, Interface};
use crate::mesh::Mesh;
use crate::model::traits::PhysicalModel;

// ── DomainId ──────────────────────────────────────────────────────────────────

/// Typed identifier for a domain within a multi-domain scenario.
///
/// At J1, a single-domain scenario uses `DomainId::default()` ("default").
/// At J3, each domain gets a meaningful identifier for coupling resolution.
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
/// `boundary_conditions` is empty at J1. At J2, BCs are added via the
/// `Domain::with_boundary_conditions` builder.
///
/// # Serialisation
///
/// `Domain` does not implement `serde::Serialize` / `serde::Deserialize`.
/// It holds `Box<dyn PhysicalModel>`, `Box<dyn Mesh>`, and
/// `Box<dyn BoundaryCondition>` (trait objects), which cannot be serialised
/// directly. See `SimulationSnapshot` (DD-025 Option B).
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
    /// ```rust,ignore
    /// let domain = Domain::new("column", model, mesh)
    ///     .with_boundary_conditions(vec![Box::new(inlet_bc), Box::new(outlet_bc)]);
    /// ```
    pub fn with_boundary_conditions(mut self, bcs: Vec<Box<dyn BoundaryCondition>>) -> Self {
        self.boundary_conditions = bcs;
        self
    }
}

// ── Scenario ──────────────────────────────────────────────────────────────────

/// Complete problem declaration — WHAT pole.
///
/// Holds one or more `Domain`s, coupling operators between them (J3), and
/// the simulation start time. Single-domain problems use the ergonomic helper
/// `Scenario::single()`. Multi-domain problems add domains and couplings without
/// any breaking change (DD-020).
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
/// state from `SimulationSnapshot` (DD-025 Option B, v0.6.0).
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
    /// Coupling operators between domains (J3, DD-011, INV-3).
    couplings: Vec<Box<dyn CouplingOperator>>,
    /// Interfaces shared between domains (J3, DD-011).
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
    /// Domains are provided as pre-built `Domain` instances.
    pub fn multi(domains: Vec<Domain>) -> Result<Self, OxiflowError> {
        if domains.is_empty() {
            return Err(OxiflowError::InvalidDomain(
                "Scenario requires at least one domain".into(),
            ));
        }
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
    /// ```rust,ignore
    /// let scenario = Scenario::multi(domains)?
    ///     .with_coupling(Box::new(my_coupling_op));
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
    /// At J1: model requirements only.
    /// At J2: model + boundary condition requirements (deduplicated).
    /// At J3: + coupling operator requirements.
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
    /// At J1: verifies at least one domain is present.
    /// At J3: will verify interface consistency and coupling coverage.
    pub fn validate(&self) -> Result<(), OxiflowError> {
        if self.domains.is_empty() {
            return Err(OxiflowError::InvalidDomain(
                "Scenario has no domains".into(),
            ));
        }
        for domain in &self.domains {
            if domain.mesh.n_dof() == 0 {
                return Err(OxiflowError::InvalidDomain(format!(
                    "domain '{}' has empty mesh",
                    domain.id
                )));
            }
        }
        Ok(())
    }
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
