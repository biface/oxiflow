//! # Module `solver::chain`
//!
//! Calculator chain — validation and ordered execution (issues #33, #36, DD-009).
//!
//! ## Responsibility
//!
//! `build_calculator_chain` verifies that every context variable required by the
//! scenario has a corresponding calculator in `SolverConfiguration`, then returns
//! the calculators in a correct execution order.
//!
//! ## Ordering strategy (DD-009)
//!
//! A hybrid two-path algorithm is selected at chain-build time:
//!
//! **Priority path (fast path)** — no calculator declares a non-empty `depends_on()`.
//! Calculators are sorted by ascending `priority()` (stable sort, O(n log n)).
//! A total order by priority cannot contain a cycle. Priority ranges:
//! - **0–49**: system variables (`Time`, `TimeStep`) — injected directly by the solver
//! - **50–99**: external data providers
//! - **100+**: derived quantities (default)
//!
//! **Topological path (Kahn)** — at least one calculator declares a non-empty
//! `depends_on()`. A directed acyclic graph is built from the declared dependencies:
//! if calculator C declares `depends_on: [X]`, an edge is drawn from every calculator
//! providing X to C. Within each topological tier, calculators are ordered by
//! ascending `priority()`. A cycle returns
//! [`OxiflowError::CircularDependency`](crate::context::error::OxiflowError::CircularDependency).
//!
//! ## Built-in variables
//!
//! `Time` and `TimeStep` are always available in `ComputeContext` — the solver
//! injects them directly via `ComputeContext::new(t, dt)` before running the
//! chain. Built-in variables declared in `depends_on()` are ignored when building
//! the dependency graph.

use crate::context::calculator::ContextCalculator;
use crate::context::error::OxiflowError;
use crate::context::variable::ContextVariable;

/// Built-in variables provided directly by the solver — no calculator needed.
///
/// `Time` and `TimeStep` are injected via `ComputeContext::new(t, dt)` before
/// the calculator chain runs. Declaring them as requirements is valid; checking
/// them against the calculator list would be a false negative. Declaring them in
/// `depends_on()` is valid and ignored when building the dependency graph.
const BUILTIN_VARIABLES: &[ContextVariable] = &[ContextVariable::Time, ContextVariable::TimeStep];

/// Validates requirements against provided calculators and returns an
/// execution-ordered chain.
///
/// # Validation
///
/// Every variable in `requirements` must be either:
/// - a built-in variable (`Time`, `TimeStep`), or
/// - covered by at least one calculator via `provides()`.
///
/// A variable covered by multiple calculators is accepted — the last one
/// to execute wins (consistent with `ComputeContext::insert` overwrite).
///
/// # Ordering (DD-009)
///
/// If no calculator declares a non-empty `depends_on()`, calculators are sorted
/// by ascending `priority()` (stable sort — registration order preserved within
/// equal priorities).
///
/// If at least one calculator declares a non-empty `depends_on()`, Kahn's
/// algorithm is applied on the full dependency graph. Within each topological
/// tier, calculators are sorted by ascending `priority()`. Built-in variables
/// in `depends_on()` declarations are ignored when building the graph.
///
/// # Errors
///
/// - [`OxiflowError::MissingCalculator`] — first uncovered non-builtin requirement.
/// - [`OxiflowError::CircularDependency`] — cycle detected in the dependency graph
///   (Kahn path only).
///
/// # Examples
///
/// ```rust
/// use oxiflow::solver::chain::build_calculator_chain;
/// use oxiflow::context::variable::ContextVariable;
/// use oxiflow::context::value::ContextValue;
/// use oxiflow::context::compute::ComputeContext;
/// use oxiflow::context::error::OxiflowError;
/// use oxiflow::context::calculator::ContextCalculator;
/// use oxiflow::model::traits::RequiresContext;
///
/// #[derive(Debug)]
/// struct TimeCalc;
/// impl RequiresContext for TimeCalc {
///     fn required_variables(&self) -> Vec<ContextVariable> { vec![] }
///     fn priority(&self) -> u32 { 0 }
/// }
/// impl ContextCalculator for TimeCalc {
///     fn provides(&self) -> ContextVariable { ContextVariable::Time }
///     fn compute(&self, _: &ContextValue, ctx: &ComputeContext)
///         -> Result<ContextValue, OxiflowError>
///     { Ok(ContextValue::Scalar(ctx.time())) }
/// }
///
/// let requirements = vec![ContextVariable::Time];
/// let calculators: Vec<Box<dyn ContextCalculator>> = vec![Box::new(TimeCalc)];
/// let chain = build_calculator_chain(&requirements, &calculators).unwrap();
/// assert_eq!(chain.len(), 1);
/// ```
pub fn build_calculator_chain<'a>(
    requirements: &[ContextVariable],
    calculators: &'a [Box<dyn ContextCalculator>],
) -> Result<Vec<&'a dyn ContextCalculator>, OxiflowError> {
    // Validate: every non-builtin requirement must have a provider.
    for req in requirements {
        if is_builtin(req) {
            continue;
        }
        let covered = calculators.iter().any(|c| &c.provides() == req);
        if !covered {
            return Err(OxiflowError::MissingCalculator(req.clone()));
        }
    }

    // Select ordering path based on whether any calculator declares dependencies.
    let has_deps = calculators.iter().any(|c| !c.depends_on().is_empty());
    if has_deps {
        build_kahn_chain(calculators)
    } else {
        build_priority_chain(calculators)
    }
}

/// Fast path — no `depends_on()` declared. Stable sort by ascending priority.
fn build_priority_chain(
    calculators: &[Box<dyn ContextCalculator>],
) -> Result<Vec<&dyn ContextCalculator>, OxiflowError> {
    let mut chain: Vec<&dyn ContextCalculator> = calculators.iter().map(|c| c.as_ref()).collect();
    chain.sort_by_key(|c| c.priority());
    Ok(chain)
}

/// Kahn path — topological sort with priority as tiebreaker within each tier.
///
/// Edges: for each calculator C declaring `depends_on: [X]`, draw an edge from
/// every calculator providing X to C. Built-in variables are excluded from the
/// graph. A cycle returns `CircularDependency` naming the first `depends_on`
/// variable of the first blocked calculator found.
fn build_kahn_chain(
    calculators: &[Box<dyn ContextCalculator>],
) -> Result<Vec<&dyn ContextCalculator>, OxiflowError> {
    let n = calculators.len();

    // Build adjacency list and in-degree table.
    // successors[j] = indices of calculators that must run after j.
    let mut successors: Vec<Vec<usize>> = vec![vec![]; n];
    let mut in_degree: Vec<usize> = vec![0; n];

    for (i, calc) in calculators.iter().enumerate() {
        for dep_var in calc.depends_on() {
            if is_builtin(&dep_var) {
                continue;
            }
            for (j, provider) in calculators.iter().enumerate() {
                if provider.provides() == dep_var {
                    successors[j].push(i);
                    in_degree[i] += 1;
                }
            }
        }
    }

    // Initial queue: calculators with no unresolved predecessors, by priority.
    let mut queue: Vec<usize> = (0..n).filter(|&i| in_degree[i] == 0).collect();
    queue.sort_by_key(|&i| calculators[i].priority());

    let mut result: Vec<&dyn ContextCalculator> = Vec::with_capacity(n);

    while !queue.is_empty() {
        // Emit the first element (lowest priority in current tier).
        let i = queue.remove(0);
        result.push(calculators[i].as_ref());

        // Decrement in-degree of successors; collect newly unblocked nodes.
        let mut newly_free: Vec<usize> = Vec::new();
        for &j in &successors[i] {
            in_degree[j] -= 1;
            if in_degree[j] == 0 {
                newly_free.push(j);
            }
        }

        // Merge newly unblocked nodes into the queue, maintaining priority order.
        queue.extend(newly_free);
        queue.sort_by_key(|&i| calculators[i].priority());
    }

    // If not all calculators were emitted, a cycle exists.
    if result.len() < n {
        let blocked = (0..n).find(|&i| in_degree[i] > 0).unwrap();
        let var = calculators[blocked]
            .depends_on()
            .into_iter()
            .find(|v| !is_builtin(v))
            .unwrap_or_else(|| calculators[blocked].provides());
        return Err(OxiflowError::CircularDependency(var));
    }

    Ok(result)
}

fn is_builtin(var: &ContextVariable) -> bool {
    BUILTIN_VARIABLES.contains(var)
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::context::compute::ComputeContext;
    use crate::context::value::ContextValue;
    use crate::model::traits::RequiresContext;

    // ── Fixtures ──────────────────────────────────────────────────────────────

    /// Calculator with configurable provides, priority, and depends_on.
    #[derive(Debug)]
    struct NamedCalc {
        provides: ContextVariable,
        priority: u32,
        depends_on: Vec<ContextVariable>,
    }

    impl RequiresContext for NamedCalc {
        fn required_variables(&self) -> Vec<ContextVariable> {
            vec![]
        }
        fn priority(&self) -> u32 {
            self.priority
        }
        fn depends_on(&self) -> Vec<ContextVariable> {
            self.depends_on.clone()
        }
    }

    impl ContextCalculator for NamedCalc {
        fn provides(&self) -> ContextVariable {
            self.provides.clone()
        }
        fn compute(
            &self,
            _state: &ContextValue,
            ctx: &ComputeContext,
        ) -> Result<ContextValue, OxiflowError> {
            Ok(ContextValue::Scalar(ctx.time()))
        }
    }

    fn make_calc(provides: ContextVariable, priority: u32) -> Box<dyn ContextCalculator> {
        Box::new(NamedCalc {
            provides,
            priority,
            depends_on: vec![],
        })
    }

    fn make_deps_calc(
        provides: ContextVariable,
        priority: u32,
        depends_on: Vec<ContextVariable>,
    ) -> Box<dyn ContextCalculator> {
        Box::new(NamedCalc {
            provides,
            priority,
            depends_on,
        })
    }

    fn var(name: &'static str) -> ContextVariable {
        ContextVariable::External { name: name.into() }
    }

    // ── Validation ────────────────────────────────────────────────────────────

    #[test]
    fn empty_requirements_with_no_calculators_succeeds() {
        let chain = build_calculator_chain(&[], &[]).unwrap();
        assert!(chain.is_empty());
    }

    #[test]
    fn builtin_time_requires_no_calculator() {
        let requirements = vec![ContextVariable::Time, ContextVariable::TimeStep];
        let chain = build_calculator_chain(&requirements, &[]).unwrap();
        assert!(chain.is_empty());
    }

    #[test]
    fn satisfied_requirement_succeeds() {
        let requirements = vec![var("D_ax")];
        let calcs = vec![make_calc(var("D_ax"), 100)];
        let chain = build_calculator_chain(&requirements, &calcs).unwrap();
        assert_eq!(chain.len(), 1);
    }

    #[test]
    fn missing_calculator_returns_error() {
        let requirements = vec![var("missing")];
        let err = build_calculator_chain(&requirements, &[]).unwrap_err();
        assert!(matches!(err, OxiflowError::MissingCalculator(_)));
    }

    #[test]
    fn missing_calculator_error_names_the_variable() {
        let v = ContextVariable::SpatialGradient {
            dimension: 0,
            component: None,
        };
        let requirements = vec![v.clone()];
        let err = build_calculator_chain(&requirements, &[]).unwrap_err();
        assert!(matches!(err, OxiflowError::MissingCalculator(x) if x == v));
    }

    #[test]
    fn duplicate_calculator_for_same_variable_is_accepted() {
        let requirements = vec![var("v")];
        let calcs = vec![make_calc(var("v"), 100), make_calc(var("v"), 50)];
        assert!(build_calculator_chain(&requirements, &calcs).is_ok());
    }

    #[test]
    fn extra_calculators_beyond_requirements_are_included() {
        let requirements = vec![var("a")];
        let calcs = vec![make_calc(var("a"), 100), make_calc(var("b"), 100)];
        let chain = build_calculator_chain(&requirements, &calcs).unwrap();
        assert_eq!(chain.len(), 2);
    }

    // ── Priority path (fast path) ─────────────────────────────────────────────

    #[test]
    fn chain_sorted_by_ascending_priority() {
        let calcs = vec![
            make_calc(var("c"), 200),
            make_calc(var("a"), 50),
            make_calc(var("b"), 100),
        ];
        let chain = build_calculator_chain(&[], &calcs).unwrap();
        assert_eq!(chain[0].priority(), 50);
        assert_eq!(chain[1].priority(), 100);
        assert_eq!(chain[2].priority(), 200);
    }

    #[test]
    fn stable_sort_preserves_registration_order_within_same_priority() {
        let calcs = vec![make_calc(var("first"), 100), make_calc(var("second"), 100)];
        let chain = build_calculator_chain(&[], &calcs).unwrap();
        assert_eq!(chain[0].provides(), var("first"));
        assert_eq!(chain[1].provides(), var("second"));
    }

    #[test]
    fn mixed_builtin_and_user_requirements() {
        let requirements = vec![ContextVariable::Time, var("D_ax")];
        let calcs = vec![make_calc(var("D_ax"), 100)];
        let chain = build_calculator_chain(&requirements, &calcs).unwrap();
        assert_eq!(chain.len(), 1);
    }

    // ── Kahn path ─────────────────────────────────────────────────────────────

    #[test]
    fn kahn_simple_chain() {
        // A provides X, B depends on X and provides Y, C depends on Y.
        // Expected order: A, B, C.
        let calcs = vec![
            make_deps_calc(var("Y"), 100, vec![var("X")]), // B — registered first
            make_deps_calc(var("Z"), 100, vec![var("Y")]), // C
            make_calc(var("X"), 100),                      // A — no deps
        ];
        let chain = build_calculator_chain(&[], &calcs).unwrap();
        assert_eq!(chain[0].provides(), var("X"));
        assert_eq!(chain[1].provides(), var("Y"));
        assert_eq!(chain[2].provides(), var("Z"));
    }

    #[test]
    fn kahn_diamond() {
        // A provides X; B depends on X, provides Y; C depends on X, provides Z;
        // D depends on Y and Z, provides W.
        // Expected: A first, D last, B and C in between.
        let calcs = vec![
            make_deps_calc(var("W"), 100, vec![var("Y"), var("Z")]), // D
            make_deps_calc(var("Y"), 100, vec![var("X")]),           // B
            make_deps_calc(var("Z"), 100, vec![var("X")]),           // C
            make_calc(var("X"), 100),                                // A
        ];
        let chain = build_calculator_chain(&[], &calcs).unwrap();
        assert_eq!(chain[0].provides(), var("X"));
        assert_eq!(chain[3].provides(), var("W"));
        // B and C are in the middle (order between them is by priority — equal here).
        let middle: Vec<ContextVariable> = chain[1..3].iter().map(|c| c.provides()).collect();
        assert!(middle.contains(&var("Y")));
        assert!(middle.contains(&var("Z")));
    }

    #[test]
    fn kahn_priority_tiebreaker_within_tier() {
        // B and C are independent (both depend on X). priority(C) < priority(B).
        // Expected within their tier: C before B.
        let calcs = vec![
            make_deps_calc(var("B_out"), 200, vec![var("X")]),
            make_deps_calc(var("C_out"), 50, vec![var("X")]),
            make_calc(var("X"), 10),
        ];
        let chain = build_calculator_chain(&[], &calcs).unwrap();
        assert_eq!(chain[0].provides(), var("X"));
        assert_eq!(chain[1].provides(), var("C_out"));
        assert_eq!(chain[2].provides(), var("B_out"));
    }

    #[test]
    fn kahn_multiple_providers_all_precede_dependent() {
        // A and B both provide X; C depends on X.
        // Both A and B must appear before C.
        let calcs = vec![
            make_deps_calc(var("Z"), 100, vec![var("X")]), // C
            make_calc(var("X"), 60),                       // B
            make_calc(var("X"), 50),                       // A (lower priority)
        ];
        let chain = build_calculator_chain(&[], &calcs).unwrap();
        let c_pos = chain.iter().position(|c| c.provides() == var("Z")).unwrap();
        let a_pos = chain
            .iter()
            .position(|c| c.priority() == 50 && c.provides() == var("X"))
            .unwrap();
        let b_pos = chain
            .iter()
            .position(|c| c.priority() == 60 && c.provides() == var("X"))
            .unwrap();
        assert!(a_pos < c_pos);
        assert!(b_pos < c_pos);
    }

    #[test]
    fn kahn_builtin_in_depends_on_is_ignored() {
        // Declaring Time in depends_on must not create a graph edge or a cycle.
        let calcs = vec![make_deps_calc(
            var("flux"),
            100,
            vec![ContextVariable::Time],
        )];
        let chain = build_calculator_chain(&[], &calcs).unwrap();
        assert_eq!(chain.len(), 1);
        assert_eq!(chain[0].provides(), var("flux"));
    }

    #[test]
    fn kahn_cycle_two_nodes_returns_error() {
        // A depends on B's output, B depends on A's output.
        let calcs = vec![
            make_deps_calc(var("A_out"), 100, vec![var("B_out")]),
            make_deps_calc(var("B_out"), 100, vec![var("A_out")]),
        ];
        let err = build_calculator_chain(&[], &calcs).unwrap_err();
        assert!(matches!(err, OxiflowError::CircularDependency(_)));
    }

    #[test]
    fn kahn_cycle_three_nodes_returns_error() {
        // A → B → C → A
        let calcs = vec![
            make_deps_calc(var("A_out"), 100, vec![var("C_out")]),
            make_deps_calc(var("B_out"), 100, vec![var("A_out")]),
            make_deps_calc(var("C_out"), 100, vec![var("B_out")]),
        ];
        let err = build_calculator_chain(&[], &calcs).unwrap_err();
        assert!(matches!(err, OxiflowError::CircularDependency(_)));
    }

    #[test]
    fn kahn_mixed_with_and_without_deps() {
        // Calculators without depends_on coexist with one that has deps.
        // Independent ones (no deps) run by priority; the dependent one after its provider.
        let calcs = vec![
            make_calc(var("alpha"), 200),
            make_deps_calc(var("beta"), 100, vec![var("alpha")]),
            make_calc(var("gamma"), 50),
        ];
        let chain = build_calculator_chain(&[], &calcs).unwrap();
        // gamma (priority 50) and alpha (priority 200) have no deps on each other.
        // beta depends on alpha → alpha must precede beta.
        let alpha_pos = chain
            .iter()
            .position(|c| c.provides() == var("alpha"))
            .unwrap();
        let beta_pos = chain
            .iter()
            .position(|c| c.provides() == var("beta"))
            .unwrap();
        assert!(alpha_pos < beta_pos);
        // gamma has no deps → should appear first (priority 50 < 200).
        assert_eq!(chain[0].provides(), var("gamma"));
    }
}
