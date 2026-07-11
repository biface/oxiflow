# oxiflow — Programme de Développement

Ce document est la référence architecturale d'oxiflow. Il couvre les principes de conception,
les spécifications de jalons, les invariants de conception, la stratégie d'écosystème et le
journal de décisions qui guident l'ensemble du travail d'implémentation de v0.1 à v3.0.

> **Version actuelle :** v0.4.0 — Integrators (clos)
> **Développement actif :** v0.5.0 — Discretisation (J5) — DiscreteOperator (DD-012/#46), FD (#47),
> FV (#48), WENO/limiteurs (#49), terme source & DiscretizedModel (DD-038)
> **Version du document :** 2.2 — Juillet 2026

---

## Table des matières

1. [Vision & Principes](#1-vision--principes)
2. [Vue d'ensemble des jalons](#2-vue-densemble-des-jalons)
3. [J1 — Architecture cœur (v0.1)](#3-j1--architecture-cœur-v01)
4. [J2 — Contexte complet (v0.2)](#4-j2--contexte-complet-v02)
5. [J3 — Multi-composants (v0.3)](#5-j3--multi-composants-v03)
6. [J4a — Intégrateurs (v0.4)](#6-j4a--intégrateurs-v04)
7. [J5 — Discrétisation (v0.5)](#7-j5--discrétisation-v05)
8. [J6 — Algèbre creuse & persistance (v0.6)](#8-j6--algèbre-creuse--persistance-v06)
9. [J7 — Intégration temporelle non linéaire (v0.7)](#9-j7--intégration-temporelle-non-linéaire-v07)
10. [J8 — Optimisation computationnelle (v0.8)](#10-j8--optimisation-computationnelle-v08)
11. [J9 — Parallélisme & benchmarking (v0.9)](#11-j9--parallélisme--benchmarking-v09)
12. [J10 — Écosystème stable (v1.0)](#12-j10--écosystème-stable-v10)
13. [Compatibilité FEM — Trajectoire v2.0 (J20)](#13-compatibilité-fem--trajectoire-v20-j20)
14. [J30 — Frameworks de niche (v3.0)](#14-j30--frameworks-de-niche-v30)
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
- **Compatibilité FEM anticipée** — les abstractions v1.0 ne présupposent pas de grille
  structurée (INV-1/2/3)
- **API plugin-safe** — tous les traits publics sont object-safe afin que des crates de
  frameworks tiers puissent les implémenter sans accéder aux internals du moteur (INV-4,
  à partir de v2.0)

### Positionnement

oxiflow n'est pas un framework CFD complet (comme OpenFOAM) ni un wrapper Python autour
de LAPACK. C'est un moteur de composition numérique fournissant les blocs architecturaux
pour construire des solveurs d'EDPs rigoureux, maintenables et performants — et la fondation
d'une famille de frameworks de niche qui mettent cette puissance à la portée de communautés
scientifiques spécifiques avec un minimum de code de plomberie.

---

## 2. Vue d'ensemble des jalons

| Jalon | Version | État | Thème |
|---|---|---|---|
| J0 — Fondations | v0.0.1–v0.0.5 | ✅ Acquis | placeholder crates.io · CI · structure projet |
| J1 — Architecture cœur | v0.1.0 | ✅ Acquis | ContextValue · OxiflowError · Mesh (INV-1) |
| J2 — Contexte complet | v0.2.0 | ✅ Acquis | BCs requirantes · ordonnancement topologique · calculateurs built-in |
| J3 — Multi-composants | v0.3.0 | ✅ Acquis | PhysicalQuantity · MultiDomainState · CouplingOperator (INV-3) |
| J4a — Intégrateurs | v0.4.0 | ✅ Acquis | Euler, RK4, DoPri45, Euler implicite, Crank-Nicolson, BDF2, IMEX |
| J5 — Discrétisation | v0.5.0 | 🔄 en cours | DiscreteOperator (INV-2) · FD/FV · WENO3/5 |
| J6 — Algèbre creuse & persistance | v0.6.0 | ⏳ Planifié | faer-sparse · export VTK/HDF5 · SimulationSnapshot |
| J7 — Intégration temporelle non linéaire | v0.7.0 | ⏳ Planifié | Newton et méthodes apparentées pour intégrateurs implicites |
| J8 — Optimisation computationnelle | v0.8.0 | ⏳ Planifié | Profilage, optimisation algorithmique/mémoire, GPU (`wgpu`) |
| J9 — Parallélisme & benchmarking | v0.9.0 | ⏳ Planifié | Rayon · cache dirty-flag · benchmarks Criterion |
| J10 — Écosystème stable | v1.0.0 | ⏳ Planifié | 7 exemples · audit FEM INV-1/2/3 · API stable |
| J20 — FEM | v2.0.0 | ⏳ Planifié | Maillages non structurés · ALE · INV-4 plugin-safe |
| J30 — Frameworks de niche | v3.0.0 | ⏳ Planifié | oxiflow-chrom · oxiflow-geo · oxiflow-thermo · oxiflow-em · CLI · tiers |

Chaque jalon est livrable indépendamment. J1 seul (v0.1) est une bibliothèque utilisable
pour la modélisation en chromatographie. Le développement de frameworks tiers peut démarrer
dès la publication de v2.0 et la mise en place d'INV-4.

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
| BC de Danckwerts (entrée) | Robin | temps + gradient |
| BC de Danckwerts (sortie) | Neumann | gradient uniquement |

---

## 5. J3 — Multi-composants (v0.3)

`PhysicalQuantity` indexé. `MultiDomainState`. `CouplingOperator` inter-domaines (INV-3).
Proto lahar–lac sur grilles régulières — base de régression pour la FEM v2.0.

---

## 6. J4a — Intégrateurs (v0.4)

| Intégrateur | Type | Statut | Issue / DD |
|---|---|---|---|
| Forward Euler | Explicite, 1er ordre | ✅ Clos | #33, #41 |
| RK4 | Explicite, 4e ordre | ✅ Clos | #41 |
| Backward Euler | Implicite, 1er ordre | ✅ Clos | #43, DD-013, DD-033 |
| Crank-Nicolson | Semi-implicite, 2e ordre | ✅ Clos | #43, DD-013, DD-033 |
| BDF2 | Implicite multi-pas, 2e ordre | ✅ Clos | #44, DD-034 |
| DoPri45 | Explicite adaptatif, ordre 5 | ✅ Clos | #42, DD-036 |
| IMEX (splitting de Strang) | Transport-réaction | ✅ Clos | #45, DD-037 |

J4a est intégralement livré : les sept intégrateurs sont clos, y compris IMEX (#45) et les
annotations serde `cfg_attr` sur les types J4 (#70).

Architecture posée en chemin, réutilisable au-delà de J4a :

- **`SteppableSolver`** (DD-031, DD-034) — primitive de pas (`step()`), historique borné via
  `history_depth()` pour les méthodes multi-pas (BDF2). Corps `solve_fixed_step()` par défaut
  (DD-035) partagé par tous les intégrateurs à pas fixe — chaque `Solver::solve()` ci-dessus est
  un appel unique à cette méthode.
- **`MultiDomainOrchestrator`** (DD-031) — pilote plusieurs domaines couplés, chacun avec son
  propre `SteppableSolver` ; `dt` synchronisé entre domaines (multi-rate explicitement reporté).
- **`LinearSolver`** (DD-013, `solver::linear`) — `Ax=b` indépendant du backend ; `nalgebra` dense
  livré à J4a, `faer` creux prévu à J6 (v0.6.0, #50) derrière le même trait.
- **`StepSizeController`** (DD-036, `solver::methods::step_control`) — contrôle de pas adaptatif
  indépendant de la source d'erreur (contrôleur PI) ; DoPri45 en est le premier consommateur, un
  futur solveur implicite adaptatif (Newton itéré, DD-033, J7) en est le second anticipé.
- **`CompositeModel`** (DD-037, `model::composite`) — somme de plusieurs `PhysicalModel` sur un
  même état ; sert de référence monolithique testable pour `OperatorSplittingSolver`
  (`solver::methods::imex`).
- DoPri45 implémente `Solver` seul, pas `SteppableSolver` — choisir son propre `dt` d'un appel à
  l'autre entre directement en tension avec le périmètre v1 à `dt` synchronisé de l'orchestrateur,
  ce n'est pas un trou orthogonal.

---

## 7. J5 — Discrétisation (v0.5)

**Développement actif.** `DiscreteOperator` abstrait (INV-2, DD-012) — type associé, pas
paramètre générique :

```rust
pub trait DiscreteOperator: Send + Sync {
    type MeshType: Mesh;
    fn apply(&self, field: &ContextValue, mesh: &Self::MeshType)
        -> Result<ContextValue, OxiflowError>;
}
```

Schémas spatiaux : FD décentrées/centrées (#47), FV conservatifs (#48), WENO3/5 + limiteurs
de flux MinMod/Van Leer/Superbee avec sélection adaptative selon le Péclet local (#49).

Deux points de branchement distincts (DD-012 amendée, DD-038) :
- **FD** — consommé via le pipeline de `ContextCalculator` existant ; `FDGradientCalculator`/
  `FDLaplacianCalculator` délèguent leur stencil à `operators::fd` sans changement d'API publique
  (issue de refactorisation dédiée, dépend de #47).
- **FV/WENO** — consommés directement dans `compute_physics()` via le nouveau composite
  `DiscretizedModel<Op>` (DD-038, Option C), qui lie le schéma spatial (terme F) à un nouveau
  trait `SourceTerm` (terme S, anticipé depuis DD-005). Le calculateur interne
  (`FluxDivergenceCalculator`) reste privé ce sprint — champ `instance_id` réservé pour une
  publication dans `ComputeContext` au moment de l'export VTK/HDF5 (J6, DD-027).

Algèbre linéaire déléguée à `nalgebra` (dense, livré J4a) et `faer` (creux, prévu J6) —
l'intégration de `faer` prolonge le trait `LinearSolver` déjà posé à J4a (DD-013), pas une
nouvelle abstraction.

**Critère de sortie :** #46–#49 clos, DD-012 et DD-038 fermées, refactorisation FD livrée.

---

## 8. J6 — Algèbre creuse & persistance (v0.6)

Solveur linéaire creux `faer-sparse` pour les systèmes implicites (#50, DD-013 phase 2).
Export des résultats : VTK (`vtkio`) comme pivot interop pour `SimulationResult` (#78,
DD-027), HDF5 (`hdf5-metno`, migration de dépendance #79) pour les données volumineuses.
`SimulationSnapshot` généralisé au-delà de la reprise sur crash vers un checkpoint/reprise
normal (DD-029, étend #71, #77).

C'est également au moment de cette étape que le calculateur privé `FluxDivergenceCalculator`
(DD-038, J5) est candidat à un branchement dans le pipeline de `ContextCalculator`, si l'export
générique de champs de flux se révèle nécessaire — décision à reprendre avec le détail de DD-027,
pas figée à J5.

Feature flags : `sparse`, `hdf5`, `vtk`.

---

## 9. J7 — Intégration temporelle non linéaire (v0.7)

DD-033 gèle la méthode theta des intégrateurs implicites (Backward Euler, Crank-Nicolson,
BDF2) à une correction de type Newton non itérée pour J4a — exact pour les problèmes affines
en `u`, approximation de premier ordre sinon. J7 lève cette limitation : solveur non linéaire
(Newton itéré à convergence, ou méthode apparentée) branché derrière le même point d'extension
que DD-033 anticipait déjà, sans réécriture des solveurs J4a.

**Critère de sortie :** un problème non affine en `u` converge à l'ordre attendu sous
Backward Euler/Crank-Nicolson/BDF2 avec le solveur non linéaire, là où la correction gelée
de J4a ne donnait qu'une approximation de premier ordre.

---

## 10. J8 — Optimisation computationnelle (v0.8)

Profilage du cœur de calcul (chaîne de calculateurs, opérateurs spatiaux J5, solveurs
linéaires J6) et optimisation algorithmique/mémoire ciblée sur les points chauds mesurés.
Formalisation du GPU-readiness (DD-026, cinq invariants structurels INV-GPU-1 à INV-GPU-5 :
mémoire contigüe, dimensions explicites, absence de trait objects dans le cœur numérique).
Feature flag `gpu` (`wgpu`) introduite derrière ces invariants — pas avant qu'ils soient
vérifiés sur la codebase existante. ROCm explicitement hors périmètre.

---

## 11. J9 — Parallélisme & benchmarking (v0.9)

Parallélisme Rayon (feature `parallel` opt-in, seuil configurable via
`set_parallel_threshold()`, DD-014). Cache dirty-flag + invalidation temporelle sur
`ComputeContext` (DD-015). Benchmarks Criterion.

**Critère de sortie :** benchmark de référence (diffusion 1D, 1000 points, 10 000 pas)
< 100 ms sur un CPU moderne, avec et sans `parallel`.

---

## 12. J10 — Écosystème stable (v1.0)

Sept exemples multi-domaines : chromatographie compétitive, transfert thermique transitoire,
motifs de Turing (Gray–Scott), couche limite de Burgers, consolidation de Terzaghi,
diffusion magnétique (proto), lahar–lac sur grilles couplées.

Audit des invariants FEM avant publication (INV-1/2/3 vérifiés sur l'ensemble de la codebase).

Stabilité API : SemVer strict, `cargo-semver-checks` dans le pipeline de release, MSRV documenté.

`oxiflow-prelude` (DD-023) : crate d'entrée ergonomique, réexports + calculateurs built-in +
`quick_config()`/`run()` pour les cas simples — dépendance strictement unidirectionnelle vers
le moteur.

---

## 13. Compatibilité FEM — Trajectoire v2.0 (J20)

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

**Introduit en v2.0.** Tous les traits publics doivent être object-safe et entièrement
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

INV-4 est le prérequis de v3.0. Aucun framework de niche ne peut être développé avant
qu'il soit en place et vérifié.

### 13.3 Périmètre v2.0

Maillage non structuré (parseur interne minimal pour `.msh` Gmsh — nœuds, connectivité,
groupes physiques → `BoundaryLocation`, DD-028 ; triangles 2D, tétraèdres 3D, raffinement
h-adaptatif). Espaces fonctionnels (P1, P2 Lagrange, Raviart–Thomas, DG0). Assembleur FEM
(matrices de rigidité et de masse, quadratures de Gauss, intégration sur faces). Solveurs
linéaires creux (`faer-sparse`, préconditionneurs ILU/AMG). Formulation ALE pour
l'exemple lahar–lac. Méthodes spectrales (DD-024) restent une question ouverte différée
post-v1.0 — l'expérience FEM sur `Mesh::coordinates()` à J20 en éclairera la viabilité.

---

## 14. J30 — Frameworks de niche (v3.0)

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

- Dépend de `oxiflow = "2"` (ou supérieur).
- Conserve le fichier `NOTICE` du moteur dans toute redistribution (exigence Apache 2.0).
- Utilise une licence compatible (Apache 2.0 recommandé ; toute licence OSI acceptée).
- Utilise le préfixe `oxiflow-` sur crates.io pour la découvrabilité.
- Ouvre une PR sur le dépôt du moteur pour être ajouté à la liste
  [Frameworks de l'écosystème connus](#15-frameworks-de-lécosystème-connus) ci-dessous.

---

## 15. Frameworks de l'écosystème connus

| Crate | Domaine | Mainteneur | Statut |
|---|---|---|---|
| `oxiflow-chrom` | Chromatographie | équipe core oxiflow | Planifié v3.0 |
| `oxiflow-geo` | Géophysique de surface | équipe core oxiflow | Planifié v3.0 |
| `oxiflow-thermo` | Transfert thermique | équipe core oxiflow | Planifié v3.0 |
| `oxiflow-em` | Électromagnétisme diffusif | équipe core oxiflow | Planifié v3.0 |

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
| Solveurs linéaires (dense) | Délégation `nalgebra` | Implémentation maison | J4a | |
| Composition temporelle | `SplitOperator`/`OperatorSplittingSolver` (Strang) | Paire figée explicite/implicite | J4a | |
| Opérateurs spatiaux | `DiscreteOperator` abstrait (type associé `MeshType`) | FD codé en dur | J5 | INV-2 |
| Composition spatiale F/S | `DiscretizedModel<Op>` + trait `SourceTerm` | Flux exposé via `ContextVariable` | J5 | INV-2 |
| Solveurs linéaires (creux) | Délégation `faer-sparse` | Implémentation maison | J6 | |
| Export de résultats | VTK pivot interop + HDF5 données volumineuses | Format maison | J6 | |
| Intégration non linéaire | Newton itéré, point d'extension DD-033 | Réécriture des solveurs J4a | J7 | |
| GPU-readiness | Invariants structurels formalisés avant feature `gpu` | `wgpu` sans contrainte préalable | J8 | |
| Parallélisme | Rayon, opt-in feature flag | Obligatoire ou absent | J9 | |
| Cache | Dirty flag + invalidation temporelle | Recalcul systématique | J9 | |
| Stabilité API | SemVer + `cargo-semver-checks` + audit FEM | Convention informelle | J10 | |
| Ergonomie | `oxiflow-prelude`, crate séparée | Builder intégré au moteur | J10 | |
| Architecture plugin | Traits object-safe + `PluginRegistry` | Crate monolithique | J20 | INV-4 |
| Config framework | TOML + registre runtime | DSL proc-macro | J30 | |
| Licence | Apache 2.0 seule | MIT ou double MIT/Apache | J0 | |

---

## 17. Registre des risques

| ID | Risque | Probabilité | Mitigation |
|---|---|---|---|
| R1 | Généricité `ContextValue` trop complexe | Moyenne | Helpers ergonomiques ; tests utilisateurs dès v0.2 |
| R2 | Bugs silencieux d'ordonnancement | Faible | Tests exhaustifs de détection de cycles ; logging debug |
| R3 | `PhysicalQuantity` indexé trop verbeux | Moyenne | Constructeurs idiomatiques ; feedback UX avant v1.0 |
| R4 | Solveurs implicites requièrent algèbre linéaire lourde | Haute | Déléguer à `faer`/`nalgebra` ; documenter les limites |
| R5 | Rayon + `unsafe` potentiel | Faible | Feature flag opt-in ; ThreadSanitizer en CI |
| R6 | Périmètre trop ambitieux | Moyenne | Chaque jalon livrable indépendamment |
| R7 | Breaking change forcé avant v1.0 | Faible | Accepté pre-1.0 mais documenté |
| R8 | INV-1/2/3 silencieusement violés | Moyenne | Audit formel à J10 ; tests d'intégration dédiés |
| R9 | ALE incompatible avec CouplingOperator | Faible | Proto lahar–lac à J3 est le banc d'essai |
| R10 | INV-4 violé — frameworks tiers cassés lors d'une mise à jour du moteur | Moyenne | Crate `oxiflow-test-plugin` externe en CI dès v2.0 ; `cargo-semver-checks` dans le pipeline |
| R11 | Fragmentation — frameworks tiers incompatibles entre eux | Faible | INV-4 + API publique stable est le seul contrat de compatibilité ; les auteurs de frameworks sont responsables de leur propre SemVer |
| R12 | Point de branchement FV/WENO (DD-038) mal choisi et coûteux à défaire | Faible | Calculateur interne gardé privé (champ `instance_id` réservé) — extension additive vers J6 sans restructuration |

---

## 18. Chronologie

Dates d'échéance des jalons GitHub (`oxiflow-milestones.yml`), pas une estimation en mois
relatifs — remplace l'ancien schéma M+N, devenu incohérent avec les échéances réelles.

| Jalon | Version | Échéance | Objectifs clés |
|---|---|---|---|
| J0 | v0.0.1–v0.0.5 | Clos (2026-03) | placeholder crates.io · CI · README · NOTICE |
| J1 | v0.1.0 | Clos (2026-04) | ContextValue · OxiflowError · Mesh (INV-1) |
| J2 | v0.2.0 | Clos (2026-05) | BCs requirantes · topologie · calculateurs built-in |
| J3 | v0.3.0 | Clos (2026-06) | PhysicalQuantity · CouplingOperator (INV-3) · proto lahar–lac |
| J4a | v0.4.0 | Clos (2026-06) | Intégrateurs temporels, IMEX inclus |
| J5 | v0.5.0 | 2026-08-06 | DiscreteOperator (INV-2) · FD/FV · WENO3/5 |
| J6 | v0.6.0 | 2026-09-16 | faer-sparse · export VTK/HDF5 · SimulationSnapshot |
| J7 | v0.7.0 | 2026-10-07 | Intégration temporelle non linéaire (Newton) |
| J8 | v0.8.0 | 2026-11-18 | Optimisation computationnelle, GPU-readiness |
| J9 | v0.9.0 | 2026-12-30 | Rayon · cache dirty-flag · benchmarks Criterion |
| J10 | v1.0.0 | 2027-03-04 | 7 exemples · gel API · audit FEM · publication stable |
| J20 | v2.0.0 | 2027-09-02 | Maillage non structuré · assembleur FEM · ALE · INV-4 |
| J30 | v3.0.0 | 2028-03-02 | oxiflow-chrom · oxiflow-geo · oxiflow-thermo · oxiflow-em · CLI |
| — | Tiers | continu | Frameworks communautaires sur crates.io |

---

*Programme de développement oxiflow v2.2 · Juillet 2026 · Document vivant — mis à jour à chaque jalon*
