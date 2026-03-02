# oxiflow — Programme de Développement

Ce document est la référence architecturale d'oxiflow. Il couvre les principes de conception,
les spécifications de jalons, les invariants de conception, la stratégie d'écosystème et le
journal de décisions qui guident l'ensemble du travail d'implémentation de v0.1 à v3.0.

> **Version actuelle :** v0.0.1 (réservation du nom sur crates.io)
> **Développement actif :** v0.1 placeholder en préparation
> **Version du document :** 2.0 — Mars 2026

---

## Table des matières

1. [Vision & Principes](#1-vision--principes)
2. [Vue d'ensemble des jalons](#2-vue-densemble-des-jalons)
3. [J1 — Architecture cœur (v0.2)](#3-j1--architecture-cœur-v02)
4. [J2 — Contexte complet (v0.3)](#4-j2--contexte-complet-v03)
5. [J3 — Multi-composants (v0.4)](#5-j3--multi-composants-v04)
6. [J4 — Solveurs & Discrétisation (v0.5–0.6)](#6-j4--solveurs--discrétisation-v05-06)
7. [J5 — Performance (v0.7)](#7-j5--performance-v07)
8. [J6 — Écosystème v1.0](#8-j6--écosystème-v10)
9. [Compatibilité FEM — Trajectoire v2.0](#9-compatibilité-fem--trajectoire-v20)
10. [J8 — Frameworks de niche — v3.0](#10-j8--frameworks-de-niche--v30)
11. [Frameworks de l'écosystème connus](#11-frameworks-de-lécosystème-connus)
12. [Journal des décisions architecturales](#12-journal-des-décisions-architecturales)
13. [Registre des risques](#13-registre-des-risques)
14. [Chronologie](#14-chronologie)

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

| Jalon | Version | Échéance | Thème |
|---|---|---|---|
| J0 — Fondations | v0.1 | Acquis | placeholder crates.io · CI · structure projet |
| J1 — Architecture cœur | v0.2 | M+2 | ContextValue · OxiflowError · Mesh (INV-1) |
| J2 — Contexte complet | v0.3 | M+4 | BCs requirantes · ordonnancement topologique |
| J3 — Multi-composants | v0.4 | M+6 | PhysicalQuantity · CouplingOperator (INV-3) |
| J4a — Intégrateurs | v0.5 | M+8 | Intégrateurs temporels |
| J4b — Discrétisation | v0.6 | M+10 | DiscreteOperator (INV-2) · FD/FV/WENO |
| J5 — Performance | v0.7 | M+13 | Rayon · cache · benchmarks |
| J6 — Écosystème v1.0 | v1.0 | M+16 | 7 exemples · audit FEM · API stable |
| J7 — FEM | v2.0 | M+24 | Maillages non structurés · ALE · INV-4 plugin-safe |
| J8 — Frameworks | v3.0 | M+32 | oxiflow-chrom · oxiflow-geo · CLI · tiers |

Chaque jalon est livrable indépendamment. J1 seul (v0.2) est une bibliothèque utilisable
pour la modélisation en chromatographie. Le développement de frameworks tiers peut démarrer
dès la publication de v2.0 et la mise en place d'INV-4.

---

## 3. J1 — Architecture cœur (v0.2)

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

## 4. J2 — Contexte complet (v0.3)

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

## 5. J3 — Multi-composants (v0.4)

`PhysicalQuantity` indexé. `MultiDomainState`. `CouplingOperator` inter-domaines (INV-3).
Proto lahar–lac sur grilles régulières — base de régression pour la FEM v2.0.

---

## 6. J4 — Solveurs & Discrétisation (v0.5–0.6)

Intégrateurs temporels : Euler explicite, RK4, DoPri45, Euler implicite, Crank–Nicolson,
BDF2/3, IMEX (splitting de Strang).

`DiscreteOperator` abstrait (INV-2) — les intégrateurs sont génériques sur le schéma :

```rust
pub trait DiscreteOperator: Send + Sync {
    type MeshType: Mesh;
    fn apply(&self, field: &ContextValue, mesh: &Self::MeshType)
        -> Result<ContextValue, OxiflowError>;
}
```

Schémas spatiaux : FD décentrées/centrées, WENO3/5, FV conservatifs, Lax–Wendroff,
limiteurs de flux (MinMod, Van Leer, Superbee), sélection adaptative selon le Péclet local.

Algèbre linéaire déléguée à `nalgebra` (dense) et `faer` (creux).

---

## 7. J5 — Performance (v0.7)

Parallélisme Rayon (feature `parallel` opt-in). Cache dirty-flag. Benchmarks Criterion.
Feature flags : `parallel`, `serde`, `hdf5`.

**Critère de sortie :** benchmark de référence (diffusion 1D, 1000 points, 10 000 pas)
< 100 ms sur un CPU moderne.

---

## 8. J6 — Écosystème v1.0

Sept exemples multi-domaines : chromatographie compétitive, transfert thermique transitoire,
motifs de Turing (Gray–Scott), couche limite de Burgers, consolidation de Terzaghi,
diffusion magnétique (proto), lahar–lac sur grilles couplées.

Audit des invariants FEM avant publication (INV-1/2/3 vérifiés sur l'ensemble de la codebase).

Stabilité API : SemVer strict, `cargo-semver-checks` dans le pipeline de release, MSRV documenté.

---

## 9. Compatibilité FEM — Trajectoire v2.0

### 9.1 Cas moteur

Un mouvement gravitaire rapide (lahar, glissement de terrain) entrant dans un plan d'eau
et générant une vague de submersion. Nécessite un maillage non structuré pour la géométrie
irrégulière et un raffinement adaptatif pour le front d'onde — impossible avec les
différences finies.

| Composante | Modèle | Défi numérique |
|---|---|---|
| Domaine granulaire | Bingham + Saint-Venant étendu | frontière mobile · maillage adaptatif |
| Domaine fluide | Équations de Shallow Water | bathymétrie irrégulière |
| Interface mobile | Formulation ALE | transfert masse/quantité de mouvement |

### 9.2 INV-4 — API plugin-safe

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

### 9.3 Périmètre v2.0

Maillage non structuré (lecteurs Gmsh/Triangle, triangles 2D, tétraèdres 3D, raffinement
h-adaptatif). Espaces fonctionnels (P1, P2 Lagrange, Raviart–Thomas, DG0). Assembleur FEM
(matrices de rigidité et de masse, quadratures de Gauss, intégration sur faces). Solveurs
linéaires creux (`faer-sparse`, préconditionneurs ILU/AMG). Formulation ALE pour
l'exemple lahar–lac.

---

## 10. J8 — Frameworks de niche — v3.0

### 10.1 Architecture

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

### 10.2 Configuration déclarative

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

### 10.3 CLI

```bash
oxiflow run probleme.toml         # résoudre
oxiflow check probleme.toml       # valider avant de résoudre
oxiflow list frameworks           # oxiflow-chrom, oxiflow-geo, ...
oxiflow list models --framework chrom
```

### 10.4 Frameworks first-party prévus

| Crate | Domaine | Modèles clés |
|---|---|---|
| `oxiflow-chrom` | Chromatographie | Langmuir, SMA, Thomas, élution gradient, BC de Danckwerts |
| `oxiflow-geo` | Géophysique de surface | Bingham Saint-Venant, Shallow Water, interface ALE |
| `oxiflow-thermo` | Transfert thermique | flux de Fourier, BC de Robin, changement de phase |
| `oxiflow-em` | Électromagnétisme diffusif | diffusion magnétique, courants de Foucault |

### 10.5 Frameworks tiers

Les tiers sont explicitement encouragés à publier des crates `oxiflow-*` sur crates.io.
Conditions pour un framework tiers :

- Dépend de `oxiflow = "2"` (ou supérieur).
- Conserve le fichier `NOTICE` du moteur dans toute redistribution (exigence Apache 2.0).
- Utilise une licence compatible (Apache 2.0 recommandé ; toute licence OSI acceptée).
- Utilise le préfixe `oxiflow-` sur crates.io pour la découvrabilité.
- Ouvre une PR sur le dépôt du moteur pour être ajouté à la liste
  [Frameworks de l'écosystème connus](#11-frameworks-de-lécosystème-connus) ci-dessous.

---

## 11. Frameworks de l'écosystème connus

| Crate | Domaine | Mainteneur | Statut |
|---|---|---|---|
| `oxiflow-chrom` | Chromatographie | équipe core oxiflow | Planifié v3.0 |
| `oxiflow-geo` | Géophysique de surface | équipe core oxiflow | Planifié v3.0 |
| `oxiflow-thermo` | Transfert thermique | équipe core oxiflow | Planifié v3.0 |
| `oxiflow-em` | Électromagnétisme diffusif | équipe core oxiflow | Planifié v3.0 |

*Pour ajouter un framework à cette liste, ouvrir une PR modifiant ce tableau.*

---

## 12. Journal des décisions architecturales

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
| Opérateurs spatiaux | `DiscreteOperator` abstrait paramétré par `Mesh` | FD codé en dur | J4 | INV-2 |
| Solveurs linéaires | Délégation `faer`/`nalgebra` | Implémentation maison | J4 | |
| Parallélisme | Rayon, opt-in feature flag | Obligatoire ou absent | J5 | |
| Cache | Dirty flag + invalidation temporelle | Recalcul systématique | J5 | |
| Stabilité API | SemVer + `cargo-semver-checks` + audit FEM | Convention informelle | J6 | |
| Architecture plugin | Traits object-safe + `PluginRegistry` | Crate monolithique | J7 | INV-4 |
| Config framework | TOML + registre runtime | DSL proc-macro | J8 | |
| Licence | Apache 2.0 seule | MIT ou double MIT/Apache | J0 | |

---

## 13. Registre des risques

| ID | Risque | Probabilité | Mitigation |
|---|---|---|---|
| R1 | Généricité `ContextValue` trop complexe | Moyenne | Helpers ergonomiques ; tests utilisateurs dès v0.2 |
| R2 | Bugs silencieux d'ordonnancement | Faible | Tests exhaustifs de détection de cycles ; logging debug |
| R3 | `PhysicalQuantity` indexé trop verbeux | Moyenne | Constructeurs idiomatiques ; feedback UX avant v1.0 |
| R4 | Solveurs implicites requièrent algèbre linéaire lourde | Haute | Déléguer à `faer`/`nalgebra` ; documenter les limites |
| R5 | Rayon + `unsafe` potentiel | Faible | Feature flag opt-in ; ThreadSanitizer en CI |
| R6 | Périmètre trop ambitieux | Moyenne | Chaque jalon livrable indépendamment |
| R7 | Breaking change forcé avant v1.0 | Faible | Accepté pre-1.0 mais documenté |
| R8 | INV-1/2/3 silencieusement violés | Moyenne | Audit formel à J6 ; tests d'intégration dédiés |
| R9 | ALE incompatible avec CouplingOperator | Faible | Proto lahar–lac à J3 est le banc d'essai |
| **R10** | **INV-4 violé — frameworks tiers cassés lors d'une mise à jour du moteur** | **Moyenne** | **Crate `oxiflow-test-plugin` externe en CI dès v2.0 ; `cargo-semver-checks` dans le pipeline** |
| **R11** | **Fragmentation — frameworks tiers incompatibles entre eux** | **Faible** | **INV-4 + API publique stable est le seul contrat de compatibilité ; les auteurs de frameworks sont responsables de leur propre SemVer** |

---

## 14. Chronologie

| Mois | Jalon | Objectifs clés |
|---|---|---|
| M0 | v0.1 — Fondations | placeholder crates.io · CI · README · NOTICE |
| M+1–2 | v0.2 — J1 | ContextValue · OxiflowError · Mesh (INV-1) |
| M+3–4 | v0.3 — J2 | BCs requirantes · topologie · calculateurs built-in |
| M+5–6 | v0.4 — J3 | PhysicalQuantity · CouplingOperator (INV-3) · proto lahar–lac |
| M+7–8 | v0.5 — J4a | Intégrateurs temporels |
| M+9–10 | v0.6 — J4b | DiscreteOperator (INV-2) · FD/FV · WENO |
| M+11–13 | v0.7 — J5 | Rayon · cache · benchmarks Criterion |
| M+14–15 | v0.9 — RC | 7 exemples · gel API · audit FEM |
| M+16 | v1.0 | Publication stable officielle |
| M+17–24 | v2.0 — J7 | Maillage non structuré · assembleur FEM · ALE · INV-4 |
| M+25–32 | v3.0 — J8 | oxiflow-chrom · oxiflow-geo · oxiflow-thermo · CLI |
| M+32+ | Tiers | Frameworks communautaires sur crates.io |

---

*Programme de développement oxiflow v2.0 · Mars 2026 · Document vivant — mis à jour à chaque jalon*
