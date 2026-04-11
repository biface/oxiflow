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

use crate::context::error::OxiflowError;
use crate::context::variable::ContextVariable;
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
/// At J1, `boundary_conditions` is always empty. At J2, BCs are added via
/// `Scenario::with_bcs()` without touching this struct definition.
#[non_exhaustive]
pub struct Domain {
    /// Unique identifier for this domain.
    pub id: DomainId,
    /// Physical model — declares and computes field equations.
    pub model: Box<dyn PhysicalModel>,
    /// Spatial mesh — INV-1.
    pub mesh: Box<dyn Mesh>,
    // boundary_conditions: Vec<Box<dyn BoundaryCondition>>  — RESERVED J2 (DD-008)
}

impl Domain {
    /// Creates a new domain with the given id, model, and mesh.
    pub fn new(
        id: impl Into<DomainId>,
        model: Box<dyn PhysicalModel>,
        mesh: Box<dyn Mesh>,
    ) -> Self {
        Self {
            id: id.into(),
            model,
            mesh,
        }
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
    // couplings: Vec<Box<dyn CouplingOperator>>  — RESERVED J3 (DD-011, INV-3)
    // interfaces: Vec<Interface>                 — RESERVED J3
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
            t_start: 0.0,
        }
    }

    /// Creates a single-domain scenario with a custom start time.
    pub fn single_from(model: Box<dyn PhysicalModel>, mesh: Box<dyn Mesh>, t_start: f64) -> Self {
        Self {
            domains: vec![Domain::new(DomainId::default(), model, mesh)],
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
            t_start: 0.0,
        })
    }

    /// Sets the simulation start time.
    pub fn with_t_start(mut self, t_start: f64) -> Self {
        self.t_start = t_start;
        self
    }

    // ── Accessors ─────────────────────────────────────────────────────────────

    /// Returns the number of domains.
    pub fn n_domains(&self) -> usize {
        self.domains.len()
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
        // self.domains.iter().for_each(|d| {
        //     d.boundary_conditions.iter().for_each(|bc| {
        //         requirements.extend(bc.required_variables());
        //     });
        // });

        // J3: extend with coupling operator requirements
        // self.couplings.iter().for_each(|c| {
        //     requirements.extend(c.required_variables());
        // });

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
}
