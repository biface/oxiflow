# oxiflow

[![CI](https://github.com/biface/oxiflow/actions/workflows/ci.yml/badge.svg)](https://github.com/biface/oxiflow/actions/workflows/ci.yml)
[![Couverture](https://codecov.io/gh/biface/oxiflow/graph/badge.svg?token=M6807F5BDQ)](https://codecov.io/gh/biface/oxiflow)
[![Crates.io](https://img.shields.io/crates/v/oxiflow.svg)](https://crates.io/crates/oxiflow)
[![Docs.rs](https://docs.rs/oxiflow/badge.svg)](https://docs.rs/oxiflow)
[![Licence](https://img.shields.io/badge/licence-Apache--2.0-blue.svg)](#licence)

**Moteur numérique générique pour les problèmes de transport–réaction–diffusion.**

oxiflow fournit les blocs architecturaux permettant de construire des solveurs d'équations aux
dérivées partielles rigoureux, maintenables et performants en Rust — des modèles de
chromatographie 1D aux systèmes multi-physiques couplés les plus complexes. L'architecture
supportera :

- les grilles structurées (v1.0)
- les maillages non structurés pour la méthode des éléments finis — un ensemble complet ciblé
  pour v2.0 — et un jeu complémentaire de composants pour les grilles structurées, déjà amorcé
  en v1.0
- des familles de frameworks de niche (v3.0)

---

## Architecture

oxiflow est structuré comme un **moteur + frameworks de niche** :

```
oxiflow              (moteur — champs, flux, maillages, couplages, solveurs)
├── oxiflow-chrom    (framework chromatographie)
├── oxiflow-geo      (framework géophysique de surface)
├── oxiflow-thermo   (framework transfert thermique)
├── oxiflow-em       (framework électromagnétisme diffusif)
└── ...              (frameworks tiers sur crates.io)
```

Chaque framework est une crate indépendante qui adaptera le moteur central aux problématiques
spécifiques des modèles physiques, conditions aux limites, nomenclature et configuration
déclarative propres à son domaine. Ce travail aura pour objectif de montrer comment utiliser
la bibliothèque centrale `oxiflow` à des tiers qui pourront alors publier leurs propres
frameworks `oxiflow-*` sur crates.io en utilisant la même API de plugin — le moteur expose des
points d'extension stables et object-safe précisément dans ce but.

---

## Fonctionnalités du moteur

- **Système de contexte générique** — `ContextValue` supporte scalaires, vecteurs, matrices
  et champs 2D ; plus de goulot d'étranglement `f64`
- **Déclaration de dépendances type-safe** — les modèles déclarent leurs besoins via
  `RequiresContext` ; les calculateurs manquants sont détectés avant la résolution
- **Systèmes multi-composants** — `PhysicalQuantity` indexé gère les problèmes à N solutés,
  l'adsorption compétitive, le couplage thermique, etc.
- **Couplage multi-domaines** — `CouplingOperator` connecte des domaines physiques distincts
  à travers des interfaces mobiles (mouvements gravitaires, interaction fluide–solide, ...)
- **Opérateurs spatiaux abstraits** — `DiscreteOperator` découple les solveurs des schémas
  de discrétisation ; FD, FV et FEM (arrivant progressivement de v1.1 à v1.9, [DD-047](https://github.com/biface/oxiflow/issues/132))
  s'enfichent sans réécrire les intégrateurs
- **Maillage abstrait** — le trait `Mesh` libère `PhysicalState` de toute hypothèse de grille
- **Bibliothèque d'intégrateurs** — Euler, RK4, DoPri45, Euler implicite, Crank–Nicolson,
  BDF2/3, IMEX (splitting de Strang) — avec correction de Newton itérée ([DD-044](https://github.com/biface/oxiflow/issues/109), v0.7.0)
  pour les méthodes implicites, configurable par solveur (critère de convergence, stratégie
  de rafraîchissement du jacobien, budget d'itérations)
- **Schémas spatiaux** — FD décentrées/centrées, WENO3/5, FV conservatifs, Lax–Wendroff,
  limiteurs de flux (MinMod, Van Leer, Superbee)
- **API plugin-safe** — tous les traits publics sont object-safe, permettant à des crates
  tierces d'implémenter des frameworks de niche (v2.0 — INV-4)
- **Optimisation des calculs** — parallélisme Rayon opt-in (feature `parallel`) et
  accélération GPU via `wgpu` (post-v0.5.0, feature `gpu`)

---

## Démarrage rapide

```toml
[dependencies]
oxiflow = "0.9"
# oxiflow-chrom = "3.0"    # framework chromatographie (disponible dès v3.0)
```

Un modèle de transport–diffusion minimal avec le moteur directement :

```rust
use oxiflow::prelude::*;

struct DiffusionModel { diffusivite: f64 }

impl PhysicalModel for DiffusionModel {
    fn compute(&self, state: &PhysicalState, ctx: &ComputeContext)
        -> Result<PhysicalState, OxiflowError>
    {
        let u     = ctx.vector(ContextVariable::Concentration)?;
        let lap_u = ctx.vector(ContextVariable::Laplacian)?;
        Ok(state.update(u + self.diffusivite * lap_u * ctx.time_step()?))
    }
}

impl RequiresContext for DiffusionModel {
    fn required_variables(&self) -> Vec<ContextVariable> {
        vec![
            ContextVariable::Concentration,
            ContextVariable::Laplacian,
            ContextVariable::TimeStep,
        ]
    }
}

fn main() -> Result<(), OxiflowError> {
    let resultat = Scenario::builder()
        .mesh(UniformGrid1D::new(100, 0.0, 1.0))
        .model(DiffusionModel { diffusivite: 1e-3 })
        .integrator(CrankNicolson::default())
        .time_span(0.0, 10.0)
        .dt(0.01)
        .build()?
        .solve()?;

    println!("{:?}", resultat.state());
    Ok(())
}
```

> **Note :** L'API ci-dessus correspond à l'objectif v0.2 — voir
> [État de développement](#état-de-développement).

---

## Domaines couverts

Le moteur est agnostique au domaine physique. Tout problème de la forme canonique
`∂u/∂t + ∇·F(u, ∇u) = S(u, x, t)` est un candidat.
Consultez le [wiki oxiflow](https://github.com/biface/oxiflow/wiki) pour les notes d'introduction sur chaque domaine.

| Domaine                | Exemple                        | Framework cible  |
|------------------------|--------------------------------|------------------|
| Chromatographie        | Élution gradient multi-solutés | `oxiflow-chrom`  |
| Transfert thermique    | Refroidissement transitoire 1D | `oxiflow-thermo` |
| Réaction–diffusion     | Motifs de Turing (Gray–Scott)  | moteur direct    |
| Mécanique des fluides  | Couche limite de Burgers       | moteur direct    |
| Géomécanique           | Consolidation de Terzaghi      | moteur direct    |
| Géophysique de surface | Interaction lahar–lac          | `oxiflow-geo`    |

---

## Invariants de conception

Quatre contraintes garantissent une évolution sans breaking change de v1.0 à v3.0 et
assurent la compatibilité des frameworks tiers entre les versions du moteur :

| Invariant | Description                                                                           | Introduit en |
|-----------|---------------------------------------------------------------------------------------|--------------|
| **INV-1** | `Mesh` est abstrait — `PhysicalState` ne présuppose aucune structure de grille        | v0.2         |
| **INV-2** | `DiscreteOperator` est abstrait — les intégrateurs sont génériques sur le schéma      | v0.5         |
| **INV-3** | `CouplingOperator` supporte des domaines distincts avec interfaces mobiles            | v0.3         |
| **INV-4** | Tous les traits publics sont object-safe — des crates tierces peuvent les implémenter | v2.0         |

---

## État de Développement

| Jalon                                    | Version                           | Statut      | Thème                                                                                                                                                                                                                                                                                                           |
|------------------------------------------|-----------------------------------|-------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| J0 — Fondations                          | v0.0.5                            | ✅ Publié   | Placeholder · CI · structure projet                                                                                                                                                                                                                                                                             |
| J1 — Architecture cœur                   | v0.1.0                            | ✅ Publié   | ContextValue · OxiflowError · Mesh (INV-1)                                                                                                                                                                                                                                                                      |
| J2 — Contexte complet                    | v0.2.0                            | ✅ Publié   | BCs requirantes · ordonnancement topologique · calculateurs                                                                                                                                                                                                                                                     |
| J3 — Multi-composants                    | v0.3.0                            | ✅ Publié   | PhysicalQuantity · MultiDomainState · CouplingOperator (INV-3)                                                                                                                                                                                                                                                  |
| J4a — Intégrateurs                       | v0.4.0                            | ✅ Publié   | Euler, RK4, DoPri45, Euler implicite, Crank–Nicolson, BDF2, IMEX                                                                                                                                                                                                                                                |
| J5 — Discrétisation                      | v0.5.0                            | ✅ Publié   | DiscreteOperator (INV-2) · FD/FV · WENO3/5 · limiteurs de flux + sélection adaptative                                                                                                                                                                                                                           |
| J6 — Algèbre creuse & persistance        | v0.6.0                            | ✅ Publié   | solveur creux faer · SimulationSnapshot (checkpoint/on_divergence) · chargement HDF5 · export VTK · IntegratorSpec                                                                                                                                                                                              |
| J7 — Intégration temporelle non linéaire | v0.7.0                            | ✅ Publié   | Itération de Newton pour les intégrateurs implicites ([DD-044](https://github.com/biface/oxiflow/issues/109)) · `IntegratorSpec` étendu à BE/CN/BDF2                                                                                                                                                            |
| J8 — Optimisation des calculs            | v0.8.0                            | ✅ Publié   | GPU-readiness ([DD-026](https://github.com/biface/oxiflow/issues/73), [DD-045](https://github.com/biface/oxiflow/issues/128)) · feature `gpu` (`wgpu`) · MSRV 1.84                                                                                                                                              |
| J9 — Parallélisme & Benchmarking         | v0.9.0                            | ✅ Publié   | Dispatch Rayon ([DD-014](https://github.com/biface/oxiflow/issues/14)/[DD-048](https://github.com/biface/oxiflow/issues/140)) · suite de benchmarks Criterion (`oxiflow_parallel`, 4 modèles) · modèle lahar multi-champs ([DD-032](https://github.com/biface/oxiflow/issues/85)) · graphiques SVG de référence |
| J10 — Écosystème v1.0                    | v1.0.0                            | ⏳ Planifié | 7 exemples multi-domaines · coefficients `(x,t)` (`ExternalTabulated`) · API stable ([DD-016](https://github.com/biface/oxiflow/issues/16))                                                                                                                                                                     |
| J11 — Fondations de maillage nD          | v1.1.0 (FEM) / v1.1.5 (structuré) | ⏳ Planifié | Fondations de maillage pour la scission FEM/structurée ([DD-047](https://github.com/biface/oxiflow/issues/132))                                                                                                                                                                                                 |
| J12 — Discrétisation vitesse-pression    | v1.2.0 / v1.2.5                   | ⏳ Planifié | Import Gmsh (FEM, [DD-028](https://github.com/biface/oxiflow/issues/76)) ou MAC/collocated+Rhie-Chow (structuré)                                                                                                                                                                                                |
| J13 — Solveur point-selle                | v1.3.0                            | ⏳ Planifié | `SaddlePointSolver`, partagé entre les deux voies                                                                                                                                                                                                                                                               |
| J14 — DAE & Navier-Stokes                | v1.4.0                            | ⏳ Planifié | Intégration temporelle DAE + Navier-Stokes en `PhysicalModel`, partagé                                                                                                                                                                                                                                          |
| J15 — Branchement bout-en-bout           | v1.5.0 / v1.5.5                   | ⏳ Planifié | Taylor-Hood/MINI (FEM) ou pipeline structuré complet                                                                                                                                                                                                                                                            |
| J16 — Protocole de comparaison           | v1.6.0                            | ⏳ Planifié | Validation FEM vs structuré (cavité entraînée, Poiseuille, Ghia et al. 1982)                                                                                                                                                                                                                                    |
| J17 — ALE                                | v1.7.0                            | ⏳ Planifié | Formulation Arbitrary Lagrangian-Eulerian                                                                                                                                                                                                                                                                       |
| J18 — Solveurs creux préconditionnés     | v1.8.0                            | ⏳ Planifié | Préconditionneurs ILU/AMG                                                                                                                                                                                                                                                                                       |
| J19 — Méthodes spectrales                | v1.9.0                            | ⏳ Planifié | `Discretization` ancêtre commun, `Mesh`/`SpectralBasis` frères ([DD-024](https://github.com/biface/oxiflow/issues/63))                                                                                                                                                                                          |
| J20 — Stabilisation                      | v2.0.0                            | 🔭 Horizon  | Gel SemVer sur l'ensemble du cœur numérique — aucune fonctionnalité neuve                                                                                                                                                                                                                                       |
| J30 — Frameworks                         | v3.0.0                            | 🔭 Horizon  | oxiflow-chrom · oxiflow-geo · CLI `oxiflow run`                                                                                                                                                                                                                                                                 |

Les sous-invariants INV-GPU-1 à INV-GPU-5 ([DD-026](https://github.com/biface/oxiflow/issues/73)) gouvernent la GPU-readiness des
structures de données numériques depuis v0.3.0 ; la feature `gpu` elle-même (sélection de
backend, acquisition d'adaptateur, conversion à la frontière CPU↔GPU) est arrivée en
v0.8.0 ([DD-045](https://github.com/biface/oxiflow/issues/128)), sans aucun breaking change sur les types conçus contre ces invariants.

Voir [DEVELOPPEMENT.md](DEVELOPPEMENT.md) pour la spécification architecturale complète.

---

## Feature Flags

| Flag       | Description                                                | Disponible avec  |
|------------|------------------------------------------------------------|------------------|
| *(défaut)* | Moteur cœur, exécution séquentielle                        | v0.2             |
| `serde`    | Sérialisation des états et scénarios                       | v0.6             |
| `hdf5`     | Import/export HDF5 pour données tabulées externes          | v0.6             |
| `vtk`      | Export VTK (`.vtu`) de `SimulationResult`                  | v0.6             |
| `sparse`   | Solveur linéaire creux `faer` pour intégrateurs implicites | v0.6             |
| `parallel` | Parallélisme Rayon pour les calculateurs indépendants      | v0.9             |
| `gpu`      | Accélération GPU via `wgpu` (Vulkan/Metal/DX12)            | v0.8             |

---

## Contribuer

Les contributions sont bienvenues à chaque jalon — corrections de bugs, nouvelles
fonctionnalités (après discussion), documentation, benchmarks, tests, et
**développement de frameworks de niche**.

**Vous construisez un framework sur oxiflow ?** Ouvrez une Discussion GitHub avant de
publier afin que la crate soit référencée dans la documentation officielle de l'écosystème.
Les crates `oxiflow-*` tierces sont explicitement encouragées — l'API plugin (INV-4) est
conçue précisément pour ça.

Consultez [CONTRIBUER.md](CONTRIBUER.md) avant de soumettre une pull request.
La couverture est suivie sur [Codecov](https://codecov.io/gh/[USER]/oxiflow) :
objectif ≥ 85% global, ≥ 90% sur les composants INV.

---

## Licence

Copyright 2026 [biface](https://github.com/biface)

Distribué sous la [Licence Apache, Version 2.0](LICENSE.txt).

Ce logiciel peut être utilisé, distribué et modifié librement, y compris à des fins
commerciales, à condition de conserver la notice de copyright et le fichier `NOTICE`
dans toute redistribution. La licence inclut une clause de représailles brevets
protégeant l'auteur — voir le texte complet pour les détails.
