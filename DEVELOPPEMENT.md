# oxiflow — Programme de Développement

Ce document est la référence architecturale d'oxiflow. Il couvre les principes de conception,
les spécifications de jalons, les invariants de conception, la stratégie d'écosystème et le
journal de décisions qui guident l'ensemble du travail d'implémentation, de v0.1.0 à v0.30.0
et au-delà.

> **Version actuelle :** v0.9.0 — Parallélisme & benchmarking (publiée)
> **Développement actif :** v0.10.0 — Écosystème stable (J10), en cours
> **Version du document :** 3.0 — Octobre 2026

---

## Table des matières

1. [Vision & Principes](#1-vision--principes)
2. [Vue d'ensemble des jalons](#2-vue-densemble-des-jalons)
3. [J1 — Architecture cœur (v0.1)](#3-j1--architecture-cœur-v01)
4. [J2 — Contexte complet (v0.2)](#4-j2--contexte-complet-v02)
5. [J3 — Multi-composants (v0.3)](#5-j3--multi-composants-v03)
6. [J4 — Intégrateurs (v0.4)](#6-j4--intégrateurs-v04)
7. [J5 — Discrétisation (v0.5)](#7-j5--discrétisation-v05)
8. [J6 — Algèbre linéaire creuse & persistance (v0.6)](#8-j6--algèbre-linéaire-creuse--persistance-v06)
9. [J7 — Intégration temporelle non linéaire (v0.7)](#9-j7--intégration-temporelle-non-linéaire-v07)
10. [J8 — Optimisation des calculs (v0.8)](#10-j8--optimisation-des-calculs-v08)
11. [J9 — Parallélisme & benchmarking (v0.9)](#11-j9--parallélisme--benchmarking-v09)
12. [J10 — Écosystème stable (v0.10)](#12-j10--écosystème-stable-v010)
13. [Compatibilité FEM — Trajectoire de J11 à J20](#13-compatibilité-fem--trajectoire-de-j11-à-j20)
14. [J30 — Frameworks de niche (v0.30)](#14-j30--frameworks-de-niche-v030)
15. [Frameworks de l'écosystème connus](#15-frameworks-de-lécosystème-connus)
16. [Journal des décisions architecturales](#16-journal-des-décisions-architecturales)
17. [Registre des risques](#17-registre-des-risques)
18. [Chronologie](#18-chronologie)

---

## 1. Vision & Principes

oxiflow est un moteur Rust générique pour la modélisation numérique de champs et de flux —
tout problème gouverné par des lois de conservation ou des équations de champ de la forme
canonique :

```
∂u/∂t + ∇·F(u, ∇u) = S(u, x, t)
```

où `u` est un champ (concentration, vitesse, température, pression, champ magnétique...),
`F` est un flux (advectif + diffusif + dispersif), et `S` est un terme source ou de réaction.

Le moteur sert de fondation à une famille de **frameworks de niche** qui ajoutent le
vocabulaire physique, les modèles pré-implémentés et la configuration déclarative propres à
des communautés scientifiques spécifiques — chromatographie, géophysique de surface,
transfert thermique, électromagnétisme diffusif, et tout domaine qu'un tiers souhaite adresser.

### Principes non-négociables

- **Déclaratif avant implicite** — les besoins d'un modèle sont exprimés dans les types
- **ContextValue générique** — les variables de contexte couvrent scalaires, vecteurs,
  matrices et champs, pas seulement `f64`
- **Type-safety à la compilation** — toute erreur de configuration provoque une erreur de
  compilation ou un échec immédiat avant la résolution
- **Zéro overhead pour les cas simples** — un modèle scalaire ne paie aucun coût lié
  à la généricité
- **Extensibilité ouverte** — ajouter un type de variable, un solveur ou un domaine
  ne nécessite pas de modifier le cœur du moteur
- **Séparation stricte des responsabilités** — le modèle déclare, le calculateur exécute,
  le solveur orchestre, le Scenario valide
- **Compatibilité FEM anticipée** — les abstractions de J1 à J10 ne présupposent pas de grille
  structurée (INV-1/2/3)
- **API plugin-safe** — tous les traits publics sont object-safe afin que des crates de
  frameworks tiers puissent les implémenter sans accéder aux internals du moteur (INV-4,
  à partir de v0.30.0)

### Positionnement

oxiflow n'est pas un framework CFD complet (comme OpenFOAM) ni un wrapper Python autour
de LAPACK. C'est un moteur de composition numérique fournissant les blocs architecturaux
pour construire des solveurs d'EDPs rigoureux, maintenables et performants — et la fondation
d'une famille de frameworks de niche qui mettent cette puissance à la portée de communautés
scientifiques spécifiques avec un minimum de code de plomberie.

---

## 2. Vue d'ensemble des jalons

| Jalon | Version | Statut | Date | Thème |
|---|---|---|---|---|
| J0 — Fondations | v0.0.1 / v0.0.5 | ✅ Publié | 2026-03-20 | Placeholder · CI · structure du projet |
| J1 — Architecture cœur | v0.1.0 | ✅ Publié | 2026-04-12 | ContextValue · OxiflowError · Mesh (INV-1) |
| J2 — Contexte complet | v0.2.0 | ✅ Publié | 2026-05-02 | Conditions aux limites requérantes · tri topologique · calculateurs |
| J3 — Multi-composants | v0.3.0 | ✅ Publié | 2026-06-03 | PhysicalQuantity · MultiDomainState · CouplingOperator (INV-3) |
| J4 — Intégrateurs | v0.4.0 | ✅ Publié | 2026-06-25 | Euler, RK4, DoPri45, Euler implicite, Crank–Nicolson, BDF2, IMEX |
| J5 — Discrétisation | v0.5.0 | ✅ Publié | 2026-07-17 | DiscreteOperator (INV-2) · FD/FV · WENO3/5 · limiteurs de flux + sélection adaptative |
| J6 — Algèbre linéaire creuse & persistance | v0.6.0 | ✅ Publié | 2026-07-21 | Solveur creux faer · SimulationSnapshot (checkpoint/on_divergence) · chargement HDF5 · export VTK · IntegratorSpec |
| J7 — Intégration temporelle non linéaire | v0.7.0 | ✅ Publié | 2026-07-30 | Itération de Newton pour les intégrateurs implicites ([DD-044](https://github.com/biface/oxiflow/issues/109)) · `IntegratorSpec` étendu à BE/CN/BDF2 |
| J8 — Optimisation des calculs | v0.8.0 | ✅ Publié | 2026-08-03 | GPU-readiness ([DA-026](https://github.com/biface/oxiflow/issues/73), [DT-045](https://github.com/biface/oxiflow/issues/128)) · feature `gpu` (`wgpu`) · MSRV 1.84 |
| J9 — Parallélisme & benchmarking | v0.9.0 | ✅ Publié | 2026-09-16 | Dispatch Rayon ([DT-014](https://github.com/biface/oxiflow/issues/14)/[DA-048](https://github.com/biface/oxiflow/issues/140)) · suite de benchmarks Criterion (`oxiflow_parallel`) · modèle lahar multi-champs ([DQ-032](https://github.com/biface/oxiflow/issues/85)) · graphiques SVG de référence |
| J10 — Écosystème stable | v0.10.0 | 🚧 En cours | 2026-11-30 | Sept exemples de référence · coefficients `(x, t)` (`CoefficientField`, [DD-066](https://github.com/biface/oxiflow/issues/181)) · audit des invariants FEM · SemVer strict, `cargo-semver-checks` ([DQ-016](https://github.com/biface/oxiflow/issues/16)) |
| J11 — Fondations de maillage nD | v0.11.0 (FEM) / v0.11.5 (structuré) | ⏳ Planifié | 2027-01-14 / 2027-02-28 | `FiniteElementMesh`, maillages non structurés 2D/3D ([DA-049](https://github.com/biface/oxiflow/issues/150)) · `UniformGrid1D` généralisée en nD ([DD-051](https://github.com/biface/oxiflow/issues/152)) |
| J12 — Discrétisation vitesse-pression | v0.12.0 / v0.12.5 | ⏳ Planifié | 2027-04-14 / 2027-05-29 | FEM (import Gmsh, [DT-028](https://github.com/biface/oxiflow/issues/76)) ou MAC/colocalisé + Rhie–Chow (structuré) |
| J13 — SaddlePointSolver | v0.13.0 | ⏳ Planifié | 2027-07-13 | `SaddlePointSolver`, commun aux deux voies |
| J14 — Systèmes différentiels-algébriques & Navier–Stokes | v0.14.0 | ⏳ Planifié | 2027-08-27 | Intégration temporelle DAE · Navier–Stokes en `PhysicalModel`, commun |
| J15 — Branchement de bout en bout de NS | v0.15.0 / v0.15.5 | ⏳ Planifié | 2027-10-11 / 2027-11-25 | Taylor–Hood/MINI (FEM) ou chaîne structurée complète |
| J16 — Solveurs creux préconditionnés | v0.16.0 | ⏳ Planifié | 2028-01-09 | Préconditionneurs ILU/AMG, avant la comparaison qui en a besoin à résolution fine ([DQ-047](https://github.com/biface/oxiflow/issues/132)) |
| J17 — Comparaison FEM / structuré | v0.17.0 | ⏳ Planifié | 2028-02-23 | Cavité entraînée, Poiseuille, Ghia et al. 1982 |
| J18 — Arbitrary Lagrangian–Eulerian | v0.18.0 | ⏳ Planifié | 2028-04-08 | Formulation ALE |
| J19 — Méthodes spectrales | v0.19.0 | ⏳ Planifié | 2028-05-23 | `Discretization` ancêtre commun, `Mesh`/`SpectralBasis` frères ([DA-024](https://github.com/biface/oxiflow/issues/63)) |
| J20 — Stabilisation | v0.20.0 | 🔭 Horizon | 2028-07-07 | Point d'étape qui clôt le premier cycle ([DQ-063](https://github.com/biface/oxiflow/issues/165)) · 1.0.0 n'est lié à aucun jalon ([DQ-016](https://github.com/biface/oxiflow/issues/16)) |
| J30 — Frameworks de niche | v0.30.0 | 🔭 Horizon | 2029-12-31 | oxiflow-chrom · oxiflow-geo · oxiflow-thermo · oxiflow-em · CLI `oxiflow run` · plugins tiers ([DA-017](https://github.com/biface/oxiflow/issues/17)) |
| Phase future | v0.40.0 | 🔭 Horizon | 2030-12-31 | Périmètre à définir |

Date : date de publication d'un jalon publié, échéance des autres (jalons GitHub).

Chaque jalon est livrable indépendamment. J1 seul (v0.1) est une bibliothèque utilisable
pour la modélisation en chromatographie. Le développement de frameworks tiers peut démarrer
dès la mise en place d'INV-4 (v0.30.0, §13.2).

---

## 3. J1 — Architecture cœur (v0.1)

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
    #[error("Calculateur manquant pour la variable : {0:?}")]
    MissingCalculator(ContextVariable),
    #[error("Échec de calcul pour {variable:?} : {source}")]
    ComputationFailed { variable: ContextVariable, source: Box<dyn std::error::Error> },
    #[error("Dépendance circulaire détectée impliquant : {0:?}")]
    CircularDependency(ContextVariable),
    #[error("Incompatibilité de type : attendu {expected:?}, obtenu {actual:?}")]
    TypeMismatch { expected: &'static str, actual: &'static str },
    #[error("Configuration de domaine invalide : {0}")]
    InvalidDomain(String),
    #[error("Erreur de données externes : {0}")]
    ExternalData(String),
    #[error("Divergence du solveur à t={time:.4e} : {reason}")]
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

### 3.4 Trait Mesh — INV-1

```rust
pub trait Mesh: Send + Sync {
    fn n_dof(&self) -> usize;
    fn coordinates(&self, i: usize) -> Vec<f64>;
    fn spatial_dimension(&self) -> usize;
    fn characteristic_length(&self) -> f64;
}
```

**Critère de sortie :** un modèle de chromatographie simple fonctionne de bout en bout
avec `ComputeContext`. `UniformGrid1D` implémente `Mesh`.

---

## 4. J2 — Contexte complet (v0.2)

BoundaryConditions requirantes — ferme la lacune de l'architecture d'origine.
Ordonnancement topologique (algorithme de Kahn). Calculateurs built-in enrichis :
gradient FD, Laplacien, quadrature, interpolation tabulée externe, lecteur HDF5.

Correspondances BC chromatographiques :

| BC chromatographique | Type mathématique | Contexte nécessaire |
|---|---|---|
| BC simplifiée | Dirichlet | profil de concentration d'injection |
| BC de Danckwerts (entrée) | Robin | concentration d'alimentation (fonction du temps) |
| BC de Danckwerts (sortie) | Neumann | aucun (nœud voisin) |

---

## 5. J3 — Multi-composants (v0.3)

`PhysicalQuantity` indexé. `MultiDomainState`. `CouplingOperator` inter-domaines (INV-3).
Proto lahar–lac sur grilles régulières — base de régression pour la voie FEM (à partir de J11).

---

## 6. J4 — Intégrateurs (v0.4)

| Intégrateur | Type | Statut | Issue / DD |
|---|---|---|---|
| Forward Euler | Explicite, 1er ordre | ✅ Clos | [#33](https://github.com/biface/oxiflow/issues/33), [#41](https://github.com/biface/oxiflow/issues/41) |
| RK4 | Explicite, 4e ordre | ✅ Clos | [#41](https://github.com/biface/oxiflow/issues/41) |
| Backward Euler | Implicite, 1er ordre | ✅ Clos | [#43](https://github.com/biface/oxiflow/issues/43), [DT-013](https://github.com/biface/oxiflow/issues/13), [DD-033](https://github.com/biface/oxiflow/issues/86) |
| Crank-Nicolson | Semi-implicite, 2e ordre | ✅ Clos | [#43](https://github.com/biface/oxiflow/issues/43), [DT-013](https://github.com/biface/oxiflow/issues/13), [DD-033](https://github.com/biface/oxiflow/issues/86) |
| BDF2 | Implicite multi-pas, 2e ordre | ✅ Clos | [#44](https://github.com/biface/oxiflow/issues/44), [DD-034](https://github.com/biface/oxiflow/issues/87) |
| DoPri45 | Explicite adaptatif, ordre 5 | ✅ Clos | [#42](https://github.com/biface/oxiflow/issues/42), [DD-036](https://github.com/biface/oxiflow/issues/89) |
| IMEX (splitting de Strang) | Transport-réaction | ✅ Clos | [#45](https://github.com/biface/oxiflow/issues/45), [DA-037](https://github.com/biface/oxiflow/issues/90) |

J4 est intégralement livré : les sept intégrateurs sont clos, y compris IMEX ([#45](https://github.com/biface/oxiflow/issues/45)) et les
annotations serde `cfg_attr` sur les types J4 ([#70](https://github.com/biface/oxiflow/issues/70)).

Architecture posée en chemin, réutilisable au-delà de J4 :

- **`SteppableSolver`** ([DA-031](https://github.com/biface/oxiflow/issues/82), [DD-034](https://github.com/biface/oxiflow/issues/87)) — primitive de pas (`step()`), historique borné via
  `history_depth()` pour les méthodes multi-pas (BDF2). Corps `solve_fixed_step()` par défaut
  ([DD-035](https://github.com/biface/oxiflow/issues/88)) partagé par tous les intégrateurs à pas fixe — chaque `Solver::solve()` ci-dessus est
  un appel unique à cette méthode.
- **`MultiDomainOrchestrator`** ([DA-031](https://github.com/biface/oxiflow/issues/82)) — pilote plusieurs domaines couplés, chacun avec son
  propre `SteppableSolver` ; `dt` synchronisé entre domaines (multi-rate explicitement reporté).
- **`LinearSolver`** ([DT-013](https://github.com/biface/oxiflow/issues/13), `solver::linear`) — `Ax=b` indépendant du backend ; `nalgebra` dense
  livré à J4, `faer` creux prévu à J6 (v0.6.0, [#50](https://github.com/biface/oxiflow/issues/50)) derrière le même trait.
- **`StepSizeController`** ([DD-036](https://github.com/biface/oxiflow/issues/89), `solver::methods::step_control`) — contrôle de pas adaptatif
  indépendant de la source d'erreur (contrôleur PI) ; DoPri45 en est le premier consommateur, un
  futur solveur implicite adaptatif (Newton itéré, [DD-033](https://github.com/biface/oxiflow/issues/86), J7) en est le second anticipé.
- **`CompositeModel`** ([DA-037](https://github.com/biface/oxiflow/issues/90), `model::composite`) — somme de plusieurs `PhysicalModel` sur un
  même état ; sert de référence monolithique testable pour `OperatorSplittingSolver`
  (`solver::methods::imex`).
- DoPri45 implémente `Solver` seul, pas `SteppableSolver` — choisir son propre `dt` d'un appel à
  l'autre entre directement en tension avec le périmètre v1 à `dt` synchronisé de l'orchestrateur,
  ce n'est pas un trou orthogonal.

---

## 7. J5 — Discrétisation (v0.5)

Deux traits frères, chacun ne portant que ce dont sa famille a
réellement besoin ([DA-012](https://github.com/biface/oxiflow/issues/12), [DA-039](https://github.com/biface/oxiflow/issues/95)) :

```rust
/// Quantité différentielle brute — jamais besoin du contexte (dx vient du maillage).
pub trait DiscreteOperator: Send + Sync {
    type MeshType: Mesh;
    fn apply(&self, field: &ContextValue, mesh: &Self::MeshType)
        -> Result<ContextValue, OxiflowError>;
}

/// Contribution COMPLÈTE -∇·F(u, ∇u) — a besoin de ComputeContext pour la physique de F
/// (vitesse, diffusion, potentiellement variables). N'étend PAS DiscreteOperator : les
/// signatures d'apply() divergent (ctx présent vs absent).
pub trait FluxDivergenceOperator: RequiresContext + Send + Sync {
    type MeshType: Mesh;
    fn apply(&self, field: &ContextValue, mesh: &Self::MeshType, ctx: &ComputeContext)
        -> Result<ContextValue, OxiflowError>;
}
```

Schémas spatiaux : FD décentrées/centrées ([#47](https://github.com/biface/oxiflow/issues/47), `DiscreteOperator`), FV conservatifs ([#48](https://github.com/biface/oxiflow/issues/48)),
WENO3/5 + limiteurs de flux MinMod/Van Leer/Superbee avec sélection adaptative selon le
Péclet local ([#49](https://github.com/biface/oxiflow/issues/49), tous deux `FluxDivergenceOperator`).

Deux points de branchement distincts, chacun cohérent avec le trait de sa famille :
- **FD** — consommé via le pipeline de `ContextCalculator` existant ; `FDGradientCalculator`/
  `FDLaplacianCalculator` délèguent leur stencil à `operators::fd` sans changement d'API
  publique (issue de refactorisation dédiée, dépend de [#47](https://github.com/biface/oxiflow/issues/47)).
- **FV/WENO** — consommés directement dans `compute_physics()` via le nouveau composite
  `DiscretizedModel<Op: FluxDivergenceOperator>` ([DA-038](https://github.com/biface/oxiflow/issues/92) amendée, [DA-039](https://github.com/biface/oxiflow/issues/95)), qui enveloppe le
  schéma spatial comme un `PhysicalModel` ordinaire ne renvoyant que `-∇·F`. La borne de type
  garantit à la compilation qu'un opérateur FD ne peut pas y être branché par erreur. Le
  terme source S, quand il existe, se compose via `CompositeModel` ([DA-037](https://github.com/biface/oxiflow/issues/90), déjà livré) —
  aucun nouveau trait `SourceTerm` n'est introduit ce sprint. Les paramètres physiques de F
  (vitesse, diffusion) sont soit des champs constants à la construction (cas simple,
  `required_variables() -> vec![]`), soit référencés par `ContextVariable` et fournis par un
  `ContextCalculator` ordinaire déjà ordonnancé par `chain.rs` ([DD-009](https://github.com/biface/oxiflow/issues/9)) — aucun mécanisme
  nouveau. Le calculateur interne (`FluxDivergenceCalculator`) reste privé ce sprint — champ
  `instance_id` réservé pour une publication dans `ComputeContext` au moment de l'export
  VTK/HDF5 (J6, [DT-027](https://github.com/biface/oxiflow/issues/75)).

Algèbre linéaire déléguée à `nalgebra` (dense, livré J4) et `faer` (creux, prévu J6) —
l'intégration de `faer` prolonge le trait `LinearSolver` déjà posé à J4 ([DT-013](https://github.com/biface/oxiflow/issues/13)), pas une
nouvelle abstraction.

**Critère de sortie :** [#46](https://github.com/biface/oxiflow/issues/46)–[#49](https://github.com/biface/oxiflow/issues/49) clos, [DA-012](https://github.com/biface/oxiflow/issues/12), [DA-038](https://github.com/biface/oxiflow/issues/92) et [DA-039](https://github.com/biface/oxiflow/issues/95) fermées, refactorisation FD
livrée.

---

## 8. J6 — Algèbre linéaire creuse & persistance (v0.6)

Solveur linéaire creux `faer-sparse` pour les systèmes implicites ([#50](https://github.com/biface/oxiflow/issues/50), [DT-013](https://github.com/biface/oxiflow/issues/13) phase 2).
Export des résultats : VTK (`vtkio` à l'époque, remplacé par une écriture `.vtu` interne en v0.10.0, [#191](https://github.com/biface/oxiflow/issues/191)) comme pivot interop pour `SimulationResult` ([#78](https://github.com/biface/oxiflow/issues/78),
[DT-027](https://github.com/biface/oxiflow/issues/75)), HDF5 (`hdf5-metno`, migration de dépendance [#79](https://github.com/biface/oxiflow/issues/79)) pour les données volumineuses,
incluant le chargement HDF5 pour `ExternalTabulated` ([#105](https://github.com/biface/oxiflow/issues/105), scindée de [#75](https://github.com/biface/oxiflow/issues/75)).
`SimulationSnapshot` généralisé au-delà de la reprise sur crash vers un checkpoint/reprise
normal ([DD-029](https://github.com/biface/oxiflow/issues/77), étend [#71](https://github.com/biface/oxiflow/issues/71), [#77](https://github.com/biface/oxiflow/issues/77)) — pas de méthode `Solver::checkpoint()` : l'appelant
construit le snapshot directement, voir la documentation de module de `snapshot.rs`.

L'amendement 1 de [DT-027](https://github.com/biface/oxiflow/issues/75) ajoute le premier DTO de configuration côté HOW,
`IntegratorSpec` ([#104](https://github.com/biface/oxiflow/issues/104)) — `TryFrom<IntegratorSpec> for Box<dyn Solver>`, seule la
variante `BackwardEuler` pour cet incrément (le chemin de dispatch creux, [#50](https://github.com/biface/oxiflow/issues/50)). Les
autres intégrateurs et l'axe WHAT (`ModelConfig`/`MeshConfig`/`BoundaryConditionConfig`)
restent à faire, hors périmètre de J6.

C'est également au moment de cette étape que le calculateur privé `FluxDivergenceCalculator`
([DA-038](https://github.com/biface/oxiflow/issues/92), J5) est candidat à un branchement dans le pipeline de `ContextCalculator`, si l'export
générique de champs de flux se révèle nécessaire — décision à reprendre avec le détail de [DT-027](https://github.com/biface/oxiflow/issues/75),
pas figée à J5.

Feature flags : `sparse`, `hdf5`, `vtk`.

---

## 9. J7 — Intégration temporelle non linéaire (v0.7)

[DD-033](https://github.com/biface/oxiflow/issues/86) gèle la méthode theta des intégrateurs implicites (Backward Euler, Crank-Nicolson,
BDF2) à une correction de type Newton non itérée pour J4 — exact pour les problèmes affines
en `u`, approximation de premier ordre sinon. J7 lève cette limitation : solveur non linéaire
(Newton itéré à convergence, ou méthode apparentée) branché derrière le même point d'extension
que [DD-033](https://github.com/biface/oxiflow/issues/86) anticipait déjà, sans réécriture des solveurs J4.

### Pourquoi c'est nécessaire : le résidu discrétisé n'est pas toujours linéaire

La forme canonique (§1) est `∂u/∂t + ∇·F(u, ∇u) = S(u, x, t)`. En notant
`f(u) = -∇·F(u, ∇u) + S(u, x, t)` le second membre, la discrétisation par méthode theta
(Backward Euler : θ=1, Crank-Nicolson : θ=0,5) transforme l'EDP en, à chaque pas de temps, un
problème de recherche de racine sur l'inconnue `u^{n+1}` :

```
g(u^{n+1}) = u^{n+1} - u^n - Δt · [(1-θ)·f(u^n) + θ·f(u^{n+1})] = 0
```

(BDF2 a son propre résidu, `g(u) = (3/2)u - 2u^n + (1/2)u^{n-1} - Δt·f(u)`, mais le
raisonnement est identique.) `g` est **linéaire** en `u^{n+1}` exactement lorsque `f` est
affine en `u` — c'est-à-dire lorsque `F` et `S` sont tous deux affines. C'est précisément le
cas que la correction unique à Jacobien gelé de J4 ([DD-033](https://github.com/biface/oxiflow/issues/86), Option B) résout déjà de façon
exacte : pour `f` affine, le Jacobien `∂g/∂u` est constant, indépendant de `u^{n+1}` — un
unique pas de Newton depuis `u^n` tombe donc exactement sur la racine, sans itération
nécessaire. Ce n'est pas un choix d'approximation, c'est une propriété d'algèbre linéaire.

Toute non-linéarité **physique** de `F` ou `S` brise cette propriété : `g` devient une
équation algébrique non linéaire, et la correction unique de J4 n'en donne alors qu'une
approximation de premier ordre de la racine — suffisant pour des problèmes faiblement non
linéaires et un `Δt` petit, pas exact en général. Cas physiques concrets qu'oxiflow doit
traiter correctement, issus des domaines cibles du projet lui-même :

- **Flux non linéaire `F(u)`** — ex. advection non linéaire de type Burgers
  (`F(u) = u²/2`), ou un coefficient de diffusion dépendant du champ lui-même
  (`F = -D(u)∇u`, diffusion non linéaire/dégénérée).
- **Terme source non linéaire `S(u)`** — ex. isothermes d'adsorption de type Langmuir en
  chromatographie (`q(c) = q_max·K·c / (1 + K·c)`, le domaine d'origine d'oxiflow via
  chrom-rs), ou cinétique de réaction de type Arrhenius (`S ∝ exp(-E_a / R·T)`).

Newton résout `g(u) = 0` en linéarisant autour de l'itéré courant `u^k` et en corrigeant :

```
J^k · Δu^k = -g(u^k),    J^k = ∂g/∂u |_{u=u^k} = I - Δt·θ · ∂f/∂u |_{u=u^k}
u^{k+1} = u^k + Δu^k
```

`∂f/∂u` est exactement ce que `finite_difference_jacobian` ([DT-013](https://github.com/biface/oxiflow/issues/13), déjà en place depuis J4)
approxime par différences finies — Newton la réutilise inchangée. Ce que [DD-044](https://github.com/biface/oxiflow/issues/109) change, c'est
*combien de fois* ce Jacobien est réévalué par pas de temps et *quand la boucle s'arrête* —
voir ci-dessous — pas le résidu, l'assemblage du Jacobien, ni la forme canonique elle-même.

### [DD-044](https://github.com/biface/oxiflow/issues/109) — activer cela correctement

[DD-044](https://github.com/biface/oxiflow/issues/109) formalise cette activation : `NewtonConvergence` (basé résidu, par défaut) et
`JacobianStrategy` (Newton modifié/gelé, par défaut) comme énumérations HOW configurables ;
`OxiflowError::NewtonNotConverged` routée via `Solver::on_divergence()` élargi ; une boucle
Newton générique unique partagée par la méthode theta (Backward Euler/Crank-Nicolson) et BDF2,
au lieu de la duplication par famille que J4 utilisait quand Newton n'avait qu'un seul
consommateur ; `IntegratorSpec` étendu à `BackwardEuler` (champs Newton ajoutés) plus nouveaux
variants `CrankNicolson` et `BDF2` (ce dernier sans champs sparse — `BDF2Solver` n'avait pas de
dispatch sparse, un écart [DA-043](https://github.com/biface/oxiflow/issues/103) laissé ouvert ici, pas une préoccupation de J7 ; comblé en
v0.10.0 par [#133](https://github.com/biface/oxiflow/issues/133)).

Découpage opérationnel ([DD-044](https://github.com/biface/oxiflow/issues/109), `oxiflow-issues-v0.7.0.yml`) :

1. Cœur de la boucle Newton (`NewtonConvergence`, `JacobianStrategy`, `NewtonNotConverged`,
   `newton_residual`/`newton_linear_correction`, boucle générique) — prérequis pour 2–3.
2. Branchement Backward Euler/Crank-Nicolson.
3. Branchement BDF2.
4. `IntegratorSpec::BackwardEuler` étendu, variants `CrankNicolson` et `BDF2` ajoutés —
   dépendent respectivement de 2/3.
5. Test de régression pour la limitation connue et non testée de [DD-033](https://github.com/biface/oxiflow/issues/86) (interaction
   Jacobien perturbé × condition Dirichlet) — indépendant de 1–4.

**Critère de sortie :** un problème non affine en `u` converge à l'ordre attendu sous
Backward Euler/Crank-Nicolson/BDF2 avec le solveur non linéaire, là où la correction gelée
de J4 ne donnait qu'une approximation de premier ordre.

---

## 10. J8 — Optimisation des calculs (v0.8)

Profilage du cœur de calcul (chaîne de calculateurs, opérateurs spatiaux J5, solveurs
linéaires J6) et optimisation algorithmique/mémoire ciblée sur les points chauds mesurés.
Formalisation du GPU-readiness ([DA-026](https://github.com/biface/oxiflow/issues/73), cinq invariants structurels INV-GPU-1 à INV-GPU-5 :
mémoire contigüe, dimensions explicites, absence de trait objects dans le cœur numérique).
Feature flag `gpu` (`wgpu`) introduite derrière ces invariants — pas avant qu'ils soient
vérifiés sur la codebase existante. ROCm explicitement hors périmètre.

---

## 11. J9 — Parallélisme & benchmarking (v0.9)

Parallélisme Rayon (feature `parallel` opt-in, seuil configurable via
`set_parallel_threshold()`, [DT-014](https://github.com/biface/oxiflow/issues/14)). Cache dirty-flag + invalidation temporelle sur
`ComputeContext` ([DD-015](https://github.com/biface/oxiflow/issues/15)). Benchmarks Criterion.

**Critère de sortie :** benchmark de référence (diffusion 1D, 1000 points, 10 000 pas)
< 100 ms sur un CPU moderne, avec et sans `parallel`.

---

## 12. J10 — Écosystème stable (v0.10)

Sept exemples de référence multi-domaines, un par domaine du tableau du README : analyse
frontale binaire en adsorption compétitive ; conduction transitoire 1D dans une barre (homogène
et graduée) ; Gray–Scott en 1D : instabilité de Turing, réplication ; couche limite visqueuse
1D contre une paroi (Burgers) ; consolidation 1D d'une couche d'argile (Terzaghi) ; diffusion
magnétique 1D dans un conducteur (proto) ; génération d'une vague par un lahar dans un lac.
Chacun se contrôle contre sa référence de validation et tourne aussi en CI.

Coefficients dépendant de la position et du temps, `(x, t)`, via `CoefficientField` ([DD-066](https://github.com/biface/oxiflow/issues/181),
[#181](https://github.com/biface/oxiflow/issues/181)).

Audit des invariants FEM avant publication (INV-1/2/3 vérifiés sur l'ensemble de la codebase).

Stabilité API : SemVer strict, `cargo-semver-checks` dans le pipeline de release, MSRV documenté.

`oxiflow-prelude` ([DA-023](https://github.com/biface/oxiflow/issues/62)), crate d'entrée ergonomique (réexports + calculateurs built-in +
`quick_config()`/`run()` pour les cas simples) à dépendance strictement unidirectionnelle vers
le moteur, ne fait pas partie de J10 : elle est reportée à J20 (v0.20.0) et en attente.

### 12.1 Exemples, tests, benchmarks et modèles partagés

Chaque modèle de référence est défini une seule fois et utilisé par toutes les cibles qui en
ont besoin ([#194](https://github.com/biface/oxiflow/issues/194)) :

| Répertoire | Rôle | Exécution |
|---|---|---|
| `examples/` | Exemples complets et validés, qui montrent comment instancier les outils d'oxiflow | À la main ; les exemples de référence de [#173](https://github.com/biface/oxiflow/issues/173) tournent aussi en CI, les autres sont seulement compilés par `clippy --all-targets` |
| `tests/` | Tests d'intégration sur de vrais enchaînements de modèles ; les comparaisons à l'epsilon aux solutions analytiques et à la littérature (`*_analytical.rs`, `*_reference.rs`) y vivent | CI (`cargo test`) |
| `benches/` | Parangonnage, comparaison à la littérature et abaques de dispatch : conditions d'emploi des solveurs, choix d'implémentation, reproductibilité | À la main, en local (configuration `7. Bench`), hors CI |
| `rsc/models/` | Définitions de modèles partagées, incluses par les trois cibles ci-dessus avec `#[path]` | Par les cibles qui les incluent |

Règles pour `rsc/models/` :

- L'exemple principal d'un modèle porte sa documentation (physique, solution exacte,
  validation) ; le fichier partagé y renvoie. Constantes de scénario, `main`, tracés et
  assertions restent propres à chaque cible.
- Un fichier est inclus comme module de son nom, suffixé `_model` quand il contient un modèle
  entier (`#[path = "../rsc/models/diffusion_1d.rs"] mod diffusion_1d_model;`), pour ne
  heurter ni un module du crate (`oxiflow::model`) ni un module de bench du même nom. Les trois
  fichiers du lahar multi-champ (`lahar_physics`, `lahar_flux`, `lahar_source`) gardent leur
  nom ; les deux derniers importent depuis un module frère `lahar_physics`.
- Les éléments partagés sont `pub(crate)` : ils appartiennent à la cible qui les inclut, jamais
  à l'API publique du crate.
- Un élément qu'une cible n'utilise pas porte un `#[allow(dead_code)]` local, avec un
  commentaire qui nomme cette cible ; jamais d'`allow` global sur un fichier partagé.
- Cargo ne découvre pas `rsc/`, donc `Cargo.toml` ne le mentionne pas. Une future liste
  `include` pour `cargo publish` devra le contenir, faute de quoi exemples, tests et
  benchmarks ne compileraient plus depuis le paquet publié.

---

## 13. Compatibilité FEM — Trajectoire de J11 à J20

### 13.1 Cas moteur

Un mouvement gravitaire rapide (lahar, glissement de terrain) entrant dans un plan d'eau
et générant une vague de submersion. Nécessite un maillage non structuré pour la géométrie
irrégulière et un raffinement adaptatif pour le front d'onde — impossible avec les
différences finies.

| Composante | Modèle | Défi numérique |
|---|---|---|
| Domaine granulaire | Bingham + Saint-Venant étendu | frontière mobile · maillage adaptatif |
| Domaine fluide | Équations de Shallow Water | bathymétrie irrégulière |
| Interface mobile | Formulation ALE | transfert masse/quantité de mouvement |

### 13.2 INV-4 — API plugin-safe

**Introduit en v0.30.0 [J30]** ([DA-017](https://github.com/biface/oxiflow/issues/17)). Tous les traits publics doivent être object-safe et entièrement
accessibles depuis une crate externe sans dépendre des internals du moteur.

Vérification : une crate d'intégration dédiée `oxiflow-test-plugin` (externe, hors
workspace) implémente les quatre traits publics et est compilée en CI.

```rust
// Ce code doit compiler depuis une crate externe — jamais depuis des types pub(crate)
use oxiflow::{PhysicalModel, BoundaryCondition, CouplingOperator, DiscreteOperator, Mesh};

struct ModeleExterne;
impl PhysicalModel for ModeleExterne { /* ... */ }
impl RequiresContext for ModeleExterne { /* ... */ }
```

INV-4 est le prérequis des frameworks de niche de v0.30.0. Aucun framework de niche ne peut être développé avant
qu'il soit en place et vérifié.

### 13.3 Périmètre de la voie FEM

Réparti de J11 à J18 (voir §2). Maillage non structuré (parseur interne minimal pour `.msh` Gmsh
— nœuds, connectivité, groupes physiques → `BoundaryLocation`, [DT-028](https://github.com/biface/oxiflow/issues/76) ; triangles 2D, tétraèdres
3D, raffinement h-adaptatif). Espaces fonctionnels (P1, P2 Lagrange, Raviart–Thomas, DG0).
Assembleur FEM (matrices de rigidité et de masse, quadratures de Gauss, intégration sur faces).
Solveurs linéaires creux (`faer-sparse`, préconditionneurs ILU/AMG). Formulation ALE pour
l'exemple lahar–lac. Les méthodes spectrales ([DA-024](https://github.com/biface/oxiflow/issues/63)) viennent après cette voie, à J19 :
l'expérience FEM sur `Mesh::coordinates()` en guidera la conception.

---

## 14. J30 — Frameworks de niche (v0.30)

### 14.1 Architecture

Le moteur expose un `PluginRegistry` que les frameworks utilisent pour enregistrer
leurs composants :

```rust
// Moteur (oxiflow)
pub struct PluginRegistry {
    models:      HashMap<&'static str, Box<dyn ModelFactory>>,
    calculators: HashMap<&'static str, Box<dyn CalculatorFactory>>,
    boundaries:  HashMap<&'static str, Box<dyn BCFactory>>,
}

// Framework (ex. oxiflow-chrom)
pub fn register(registry: &mut PluginRegistry) {
    registry.register_model("langmuir",       LangmuirFactory);
    registry.register_model("thomas",          ThomasFactory);
    registry.register_model("sma",             SMAFactory);
    registry.register_bc("danckwerts",         DanckwertsFactory);
    registry.register_bc("simplified",         SimplifiedBCFactory);
    registry.register_calculator("dispersion", AxialDispersionFactory);
}
```

Le moteur n'a aucune connaissance des frameworks. Les frameworks dépendent du moteur.
La dépendance est strictement unidirectionnelle.

### 14.2 Configuration déclarative

Le moteur fournit l'infrastructure TOML générique. Chaque framework l'étend avec ses
sections spécifiques :

```toml
# Résolu par le moteur
[solver]
integrator = "crank_nicolson"
dt = 0.01
t_end = 600.0

[mesh.colonne]
type = "uniform_1d"
length = 0.25
n_points = 500

# Résolu par oxiflow-chrom
[chromatography.column]
mode = "gradient_elution"

[[chromatography.solute]]
name = "proteine_A"
isotherm = "langmuir"
H = 2.5
b = 0.08

[chromatography.boundary]
inlet  = "danckwerts"
outlet = "danckwerts"
```

### 14.3 CLI

```bash
oxiflow run probleme.toml         # résoudre
oxiflow check probleme.toml       # valider avant de résoudre
oxiflow list frameworks           # oxiflow-chrom, oxiflow-geo, ...
oxiflow list models --framework chrom
```

### 14.4 Frameworks first-party prévus

| Crate | Domaine | Modèles clés |
|---|---|---|
| `oxiflow-chrom` | Chromatographie | Langmuir, SMA, Thomas, élution gradient, BC de Danckwerts |
| `oxiflow-geo` | Géophysique de surface | Bingham Saint-Venant, Shallow Water, interface ALE |
| `oxiflow-thermo` | Transfert thermique | flux de Fourier, BC de Robin, changement de phase |
| `oxiflow-em` | Électromagnétisme diffusif | diffusion magnétique, courants de Foucault |

### 14.5 Frameworks tiers

Les tiers sont explicitement encouragés à publier des crates `oxiflow-*` sur crates.io.
Conditions pour un framework tiers :

- Dépend de `oxiflow = "0.30"` (ou supérieur), première version dotée d'INV-4.
- Conserve le fichier `NOTICE` du moteur dans toute redistribution (exigence Apache 2.0).
- Utilise une licence compatible (Apache 2.0 recommandé ; toute licence OSI acceptée).
- Utilise le préfixe `oxiflow-` sur crates.io pour la découvrabilité.
- Ouvre une PR sur le dépôt du moteur pour être ajouté à la liste
  [Frameworks de l'écosystème connus](#15-frameworks-de-lécosystème-connus) ci-dessous.

---

## 15. Frameworks de l'écosystème connus

| Crate | Domaine | Mainteneur | Statut |
|---|---|---|---|
| `oxiflow-chrom` | Chromatographie | équipe core oxiflow | Planifié v0.30.0 |
| `oxiflow-geo` | Géophysique de surface | équipe core oxiflow | Planifié v0.30.0 |
| `oxiflow-thermo` | Transfert thermique | équipe core oxiflow | Planifié v0.30.0 |
| `oxiflow-em` | Électromagnétisme diffusif | équipe core oxiflow | Planifié v0.30.0 |

*Pour ajouter un framework à cette liste, ouvrir une PR modifiant ce tableau.*

---

## 16. Journal des décisions architecturales

| Décision | Choix retenu | Alternative rejetée | Jalon | Invariant |
|---|---|---|---|---|
| Type de retour calculateur | `ContextValue` enum | `f64` scalaire | J1 | |
| Type d'erreur | `OxiflowError` enum | `String` | J1 | |
| API d'accès au contexte | `ComputeContext` type-safe dès v0.2 | Migration progressive | J1 | |
| Déclaration des besoins | Trait `RequiresContext` séparé | Méthode sur `PhysicalModel` | J1 | |
| Support spatial | Trait abstrait `Mesh` | `dx`/`nx` dans `PhysicalState` | J1 | INV-1 |
| BCs requirantes | `RequiresContext` sur `BoundaryCondition` | Agrégation manuelle | J2 | |
| Ordonnancement | Topologie + priorité hybride | DAG pur ou priorité seule | J2 | |
| Multi-composants | `PhysicalQuantity` indexé | Enum plat avec breaking changes | J3 | |
| Couplage multi-physique | `CouplingOperator` avec `DomainId` + `Interface` | Méthode ad-hoc | J3 | INV-3 |
| Solveurs linéaires (dense) | Délégation `nalgebra` | Implémentation maison | J4 | |
| Composition temporelle | `SplitOperator`/`OperatorSplittingSolver` (Strang) | Paire figée explicite/implicite | J4 | |
| Opérateurs spatiaux | `DiscreteOperator` abstrait (type associé `MeshType`) | FD codé en dur | J5 | INV-2 |
| Composition spatiale F/S | `DiscretizedModel<Op>` (F) + `CompositeModel` existant (F+S) | Nouveau trait `SourceTerm` ; flux exposé via `ContextVariable` | J5 | INV-2 |
| Accès contexte des opérateurs | `FluxDivergenceOperator` : trait frère de `DiscreteOperator`, porte `ctx`/`RequiresContext` | Ajout de `ctx` directement sur `DiscreteOperator` ; sous-trait de `DiscreteOperator` | J5 | INV-2 |
| Solveurs linéaires (creux) | Délégation `faer-sparse` | Implémentation maison | J6 | |
| Export de résultats | VTK pivot interop + HDF5 données volumineuses | Format maison | J6 | |
| Intégration non linéaire | Newton itéré, point d'extension [DD-033](https://github.com/biface/oxiflow/issues/86) activé par [DD-044](https://github.com/biface/oxiflow/issues/109) (boucle générique, pas de duplication par famille) | Réécriture des solveurs J4 ; boucle dupliquée par famille | J7 | |
| GPU-readiness | Invariants structurels formalisés avant feature `gpu` | `wgpu` sans contrainte préalable | J8 | |
| Parallélisme | Rayon, opt-in feature flag | Obligatoire ou absent | J9 | |
| Cache | Dirty flag + invalidation temporelle | Recalcul systématique | J9 | |
| Stabilité API | SemVer + `cargo-semver-checks` + audit FEM | Convention informelle | J10 | |
| Ergonomie | `oxiflow-prelude`, crate séparée | Builder intégré au moteur | J20 (en attente) | |
| Architecture plugin | Traits object-safe + `PluginRegistry` | Crate monolithique | J30 | INV-4 |
| Config framework | TOML + registre runtime | DSL proc-macro | J30 | |
| Licence | Apache 2.0 seule | MIT ou double MIT/Apache | J0 | |

---

## 17. Registre des risques

| ID | Risque | Probabilité | Mitigation |
|---|---|---|---|
| R1 | Généricité `ContextValue` trop complexe | Moyenne | Helpers ergonomiques ; tests utilisateurs dès v0.2 |
| R2 | Bugs silencieux d'ordonnancement | Faible | Tests exhaustifs de détection de cycles ; logging debug |
| R3 | `PhysicalQuantity` indexé trop verbeux | Moyenne | Constructeurs idiomatiques ; feedback UX avant 1.0.0 |
| R4 | Solveurs implicites requièrent algèbre linéaire lourde | Haute | Déléguer à `faer`/`nalgebra` ; documenter les limites |
| R5 | Rayon + `unsafe` potentiel | Faible | Feature flag opt-in ; ThreadSanitizer en CI |
| R6 | Périmètre trop ambitieux | Moyenne | Chaque jalon livrable indépendamment |
| R7 | Breaking change forcé avant 1.0.0 | Faible | Accepté pre-1.0 mais documenté |
| R8 | INV-1/2/3 silencieusement violés | Moyenne | Audit formel à J10 ; tests d'intégration dédiés |
| R9 | ALE incompatible avec CouplingOperator | Faible | Proto lahar–lac à J3 est le banc d'essai |
| R10 | INV-4 violé — frameworks tiers cassés lors d'une mise à jour du moteur | Moyenne | Crate `oxiflow-test-plugin` externe en CI dès v0.30.0 ; `cargo-semver-checks` dans le pipeline |
| R11 | Fragmentation — frameworks tiers incompatibles entre eux | Faible | INV-4 + API publique stable est le seul contrat de compatibilité ; les auteurs de frameworks sont responsables de leur propre SemVer |
| R12 | Point de branchement FV/WENO ([DA-038](https://github.com/biface/oxiflow/issues/92)) mal choisi et coûteux à défaire | Faible | Calculateur interne gardé privé (champ `instance_id` réservé) — extension additive vers J6 sans restructuration |

---

## 18. Chronologie

Les dates figurent dans le tableau du §2, tirées des jalons GitHub : date de publication d'un
jalon publié, échéance des autres. Elles ne sont pas répétées ici, pour que les deux tableaux
ne puissent pas diverger. Les frameworks tiers suivent J30 (v0.30.0) sur crates.io, sans date
fixée.

---

*Programme de développement oxiflow · version du document 3.0 · Octobre 2026 · Document vivant — mis à jour à chaque jalon*
