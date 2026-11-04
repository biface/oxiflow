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

- les grilles structurées — 1D aujourd'hui (`UniformGrid1D`), nD à partir de v0.11.5
- les maillages non structurés de la méthode des éléments finis à partir de v0.11.0, les voies
  éléments finis et grilles structurées avançant côte à côte jusqu'à v0.15.x ([DQ-047](https://github.com/biface/oxiflow/issues/132))
- une famille de frameworks de niche (v0.30.0)

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
  de discrétisation ; FD, FV et FEM (arrivant progressivement de v0.11.0 à v0.15.0)
  s'enfichent sans réécrire les intégrateurs
- **Maillage abstrait** — le trait `Mesh` libère `PhysicalState` de toute hypothèse de grille ;
  `UniformGrid1D` aujourd'hui, grilles structurées nD en v0.11.5, maillages non structurés 2D/3D
  en v0.11.0
- **Bibliothèque d'intégrateurs** — Euler, RK4, DoPri45, Euler implicite, Crank–Nicolson,
  BDF2, IMEX (splitting de Strang) — avec correction de Newton itérée ([DD-044](https://github.com/biface/oxiflow/issues/109), v0.7.0)
  pour les méthodes implicites, configurable par solveur (critère de convergence, stratégie
  de rafraîchissement du jacobien, budget d'itérations) ; les formules BDF d'ordre supérieur
  (BDF3 à BDF6) sont envisagées si un modèle en a besoin
- **Schémas spatiaux** — FD décentrées/centrées, WENO3/5, FV conservatifs, Lax–Wendroff,
  limiteurs de flux (MinMod, Van Leer, Superbee)
- **API plugin-safe** — tous les traits publics sont object-safe, permettant à des crates
  tierces d'implémenter des frameworks de niche (v0.30.0 — INV-4)
- **Optimisation des calculs** — parallélisme Rayon opt-in (feature `parallel`) et
  GPU-readiness via `wgpu` (feature `gpu`, depuis v0.8.0)

---

## Démarrage rapide

```toml
[dependencies]
oxiflow = "0.10"
# oxiflow-chrom          # framework chromatographie (prévu pour v0.30.0)
```

Un modèle de diffusion minimal avec le moteur directement : le modèle (QUOI) déclare le
laplacien dont il a besoin, la configuration du solveur (COMMENT) fournit le calculateur
qui le calcule.

```rust
use std::sync::Arc;

use nalgebra::DVector;
use oxiflow::{
    boundary::{BoundaryCondition, Dirichlet},
    context::{
        calculators::FDLaplacianCalculator, compute::ComputeContext, error::OxiflowError,
        value::ContextValue, variable::ContextVariable,
    },
    mesh::{BoundaryRegion, FaceSide, Mesh, UniformGrid1D},
    model::traits::{PhysicalModel, RequiresContext},
    solver::{
        config::{IntegratorKind, StepControl, TimeConfiguration},
        methods::CrankNicolsonSolver,
        scenario::{Domain, Scenario},
        Solver, SolverConfiguration,
    },
};

/// du/dt = D d²u/dx², le laplacien étant fourni par un calculateur de contexte.
struct Diffusion {
    diffusivity: f64,
}

fn laplacian() -> ContextVariable {
    ContextVariable::External {
        name: "laplacian".into(),
    }
}

impl RequiresContext for Diffusion {
    fn required_variables(&self) -> Vec<ContextVariable> {
        vec![laplacian()]
    }
}

impl PhysicalModel for Diffusion {
    fn compute_physics(
        &self,
        _state: &ContextValue,
        ctx: &ComputeContext,
    ) -> Result<ContextValue, OxiflowError> {
        let lap_u = ctx.external(laplacian())?.as_scalar_field()?;
        Ok(ContextValue::ScalarField(lap_u * self.diffusivity))
    }

    fn initial_state(&self, mesh: &dyn Mesh) -> ContextValue {
        ContextValue::ScalarField(DVector::zeros(mesh.n_dof()))
    }

    fn name(&self) -> &str {
        "diffusion"
    }
}

fn main() -> Result<(), OxiflowError> {
    // QUOI : une tige de longueur unité, maintenue à u = 1 à gauche et u = 0 à droite.
    let ends: Vec<Box<dyn BoundaryCondition>> = vec![
        Box::new(Dirichlet::constant(
            BoundaryRegion::Face {
                axis: 0,
                side: FaceSide::Lower,
            },
            1.0,
        )),
        Box::new(Dirichlet::constant(
            BoundaryRegion::Face {
                axis: 0,
                side: FaceSide::Upper,
            },
            0.0,
        )),
    ];
    let rod = Domain::new(
        "rod",
        Box::new(Diffusion { diffusivity: 1e-3 }),
        Box::new(UniformGrid1D::new(101, 0.0, 1.0)?),
    )
    .with_boundary_conditions(ends);
    let scenario = Scenario::multi(vec![rod])?;

    // COMMENT : Crank–Nicolson, pas fixe, laplacien par différences finies.
    let mesh: Arc<dyn Mesh> = Arc::new(UniformGrid1D::new(101, 0.0, 1.0)?);
    let config = SolverConfiguration::new(
        TimeConfiguration::new(10.0, StepControl::Fixed { dt: 0.01 }),
        IntegratorKind::CrankNicolson,
    )
    .with_calculator(Box::new(FDLaplacianCalculator::new(mesh, laplacian())));

    let result = CrankNicolsonSolver::new().solve(&scenario, &config)?;
    if let Some(last) = result.states.last() {
        let u = last.as_scalar_field()?;
        println!("u(x = 0.5, t = 10) = {:.3e}", u[50]);
    }
    Ok(())
}
```

Il affiche `u(x = 0.5, t = 10) = 4.200e-4` ; la solution pour une tige semi-infinie,
`erfc(x / (2 sqrt(D t)))`, donne 4,07e-4 en ce point.

---

## Domaines couverts

Le moteur est agnostique au domaine physique. Tout problème de la forme canonique
`∂u/∂t + ∇·F(u, ∇u) = S(u, x, t)` est un candidat.
Consultez le [wiki oxiflow](https://github.com/biface/oxiflow/wiki) pour les notes d'introduction sur chaque domaine.

| Domaine                | Exemple                                                        | Framework cible  |
|------------------------|----------------------------------------------------------------|------------------|
| Chromatographie        | Analyse frontale binaire en adsorption compétitive             | `oxiflow-chrom`  |
| Transfert thermique    | Conduction transitoire 1D dans une barre (homogène et graduée) | `oxiflow-thermo` |
| Réaction–diffusion     | Gray–Scott en 1D : instabilité de Turing, réplication          | moteur direct    |
| Mécanique des fluides  | Couche limite visqueuse 1D contre une paroi (Burgers)          | moteur direct    |
| Géomécanique           | Consolidation 1D d'une couche d'argile (Terzaghi)              | moteur direct    |
| Électromagnétisme      | Diffusion magnétique 1D dans un conducteur (proto)             | `oxiflow-em`     |
| Géophysique de surface | Front d'un lahar entrant dans un lac                           | `oxiflow-geo`    |

---

## Invariants de conception

Quatre contraintes garantissent que la méthode des éléments finis (à partir de v0.11.0) et
les frameworks tiers (v0.30.0) arrivent sans breaking change pour le code existant, et que
ces frameworks restent compatibles d'une version du moteur à l'autre :

| Invariant | Description                                                                           | Introduit en |
|-----------|---------------------------------------------------------------------------------------|--------------|
| **INV-1** | `Mesh` est abstrait — aucune API publique n'expose `dx`, `nx` ou des indices bruts    | v0.1         |
| **INV-2** | `DiscreteOperator` est abstrait — les intégrateurs sont génériques sur le schéma      | v0.5         |
| **INV-3** | Couplage inter-domaines uniquement par `CouplingOperator`, jamais dans un solveur     | v0.3         |
| **INV-4** | Tous les traits publics sont object-safe — des crates tierces peuvent les implémenter | v0.30        |

---

## État de Développement

| Jalon                                                    | Version                             | Statut     | Date                    | Thème                                                                                                                                                                                                                                                                                                |
|----------------------------------------------------------|-------------------------------------|------------|-------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| J0 — Fondations                                          | v0.0.1 / v0.0.5                     | ✅ Publié   | 2026-03-20              | Placeholder · CI · structure du projet                                                                                                                                                                                                                                                               |
| J1 — Architecture cœur                                   | v0.1.0                              | ✅ Publié   | 2026-04-12              | ContextValue · OxiflowError · Mesh (INV-1)                                                                                                                                                                                                                                                           |
| J2 — Contexte complet                                    | v0.2.0                              | ✅ Publié   | 2026-05-02              | Conditions aux limites requérantes · tri topologique · calculateurs                                                                                                                                                                                                                                  |
| J3 — Multi-composants                                    | v0.3.0                              | ✅ Publié   | 2026-06-03              | PhysicalQuantity · MultiDomainState · CouplingOperator (INV-3)                                                                                                                                                                                                                                       |
| J4 — Intégrateurs                                        | v0.4.0                              | ✅ Publié   | 2026-06-25              | Euler, RK4, DoPri45, Euler implicite, Crank–Nicolson, BDF2, IMEX                                                                                                                                                                                                                                     |
| J5 — Discrétisation                                      | v0.5.0                              | ✅ Publié   | 2026-07-17              | DiscreteOperator (INV-2) · FD/FV · WENO3/5 · limiteurs de flux + sélection adaptative                                                                                                                                                                                                                |
| J6 — Algèbre linéaire creuse & persistance               | v0.6.0                              | ✅ Publié   | 2026-07-21              | Solveur creux faer · SimulationSnapshot (checkpoint/on_divergence) · chargement HDF5 · export VTK · IntegratorSpec                                                                                                                                                                                   |
| J7 — Intégration temporelle non linéaire                 | v0.7.0                              | ✅ Publié   | 2026-07-30              | Itération de Newton pour les intégrateurs implicites ([DD-044](https://github.com/biface/oxiflow/issues/109)) · `IntegratorSpec` étendu à BE/CN/BDF2                                                                                                                                                 |
| J8 — Optimisation des calculs                            | v0.8.0                              | ✅ Publié   | 2026-08-03              | GPU-readiness ([DA-026](https://github.com/biface/oxiflow/issues/73), [DT-045](https://github.com/biface/oxiflow/issues/128)) · feature `gpu` (`wgpu`) · MSRV 1.84                                                                                                                                   |
| J9 — Parallélisme & benchmarking                         | v0.9.0                              | ✅ Publié   | 2026-09-16              | Dispatch Rayon ([DT-014](https://github.com/biface/oxiflow/issues/14)/[DA-048](https://github.com/biface/oxiflow/issues/140)) · suite de benchmarks Criterion (`oxiflow_parallel`) · modèle lahar multi-champs ([DQ-032](https://github.com/biface/oxiflow/issues/85)) · graphiques SVG de référence |
| J10 — Écosystème stable                                  | v0.10.0                             | 🚧 En cours | 2026-11-30              | Sept exemples de référence · coefficients `(x, t)` (`CoefficientField`, [DD-066](https://github.com/biface/oxiflow/issues/181)) · audit des invariants FEM · SemVer strict, `cargo-semver-checks` ([DQ-016](https://github.com/biface/oxiflow/issues/16))                                            |
| J11 — Fondations de maillage nD                          | v0.11.0 (FEM) / v0.11.5 (structuré) | ⏳ Planifié | 2027-01-14 / 2027-02-28 | `FiniteElementMesh`, maillages non structurés 2D/3D ([DA-049](https://github.com/biface/oxiflow/issues/150)) · `UniformGrid1D` généralisée en nD ([DD-051](https://github.com/biface/oxiflow/issues/152))                                                                                            |
| J12 — Discrétisation vitesse-pression                    | v0.12.0 / v0.12.5                   | ⏳ Planifié | 2027-04-14 / 2027-05-29 | FEM (import Gmsh, [DT-028](https://github.com/biface/oxiflow/issues/76)) ou MAC/colocalisé + Rhie–Chow (structuré)                                                                                                                                                                                   |
| J13 — SaddlePointSolver                                  | v0.13.0                             | ⏳ Planifié | 2027-07-13              | `SaddlePointSolver`, commun aux deux voies                                                                                                                                                                                                                                                           |
| J14 — Systèmes différentiels-algébriques & Navier–Stokes | v0.14.0                             | ⏳ Planifié | 2027-08-27              | Intégration temporelle DAE · Navier–Stokes en `PhysicalModel`, commun                                                                                                                                                                                                                                |
| J15 — Branchement de bout en bout de NS                  | v0.15.0 / v0.15.5                   | ⏳ Planifié | 2027-10-11 / 2027-11-25 | Taylor–Hood/MINI (FEM) ou chaîne structurée complète                                                                                                                                                                                                                                                 |
| J16 — Solveurs creux préconditionnés                     | v0.16.0                             | ⏳ Planifié | 2028-01-09              | Préconditionneurs ILU/AMG, avant la comparaison qui en a besoin à résolution fine ([DQ-047](https://github.com/biface/oxiflow/issues/132))                                                                                                                                                           |
| J17 — Comparaison FEM / structuré                        | v0.17.0                             | ⏳ Planifié | 2028-02-23              | Cavité entraînée, Poiseuille, Ghia et al. 1982                                                                                                                                                                                                                                                       |
| J18 — Arbitrary Lagrangian–Eulerian                      | v0.18.0                             | ⏳ Planifié | 2028-04-08              | Formulation ALE                                                                                                                                                                                                                                                                                      |
| J19 — Méthodes spectrales                                | v0.19.0                             | ⏳ Planifié | 2028-05-23              | `Discretization` ancêtre commun, `Mesh`/`SpectralBasis` frères ([DA-024](https://github.com/biface/oxiflow/issues/63))                                                                                                                                                                               |
| J20 — Stabilisation                                      | v0.20.0                             | 🔭 Horizon  | 2028-07-07              | Point d'étape qui clôt le premier cycle ([DQ-063](https://github.com/biface/oxiflow/issues/165)) · 1.0.0 n'est lié à aucun jalon ([DQ-016](https://github.com/biface/oxiflow/issues/16))                                                                                                             |
| J30 — Frameworks de niche                                | v0.30.0                             | 🔭 Horizon  | 2029-12-31              | oxiflow-chrom · oxiflow-geo · oxiflow-thermo · oxiflow-em · CLI `oxiflow run` · plugins tiers ([DA-017](https://github.com/biface/oxiflow/issues/17))                                                                                                                                                |
| Phase future                                             | v0.40.0                             | 🔭 Horizon  | 2030-12-31              | Périmètre à définir                                                                                                                                                                                                                                                                                  |

Date : date de publication d'un jalon publié, échéance des autres (jalons GitHub).

Les sous-invariants INV-GPU-1 à INV-GPU-5 ([DA-026](https://github.com/biface/oxiflow/issues/73)) gouvernent la GPU-readiness des
structures de données numériques depuis v0.3.0 ; la feature `gpu` elle-même (sélection de
backend, acquisition d'adaptateur, conversion à la frontière CPU↔GPU) est arrivée en
v0.8.0 ([DT-045](https://github.com/biface/oxiflow/issues/128)), sans aucun breaking change sur les types conçus contre ces invariants.

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

Le [wiki oxiflow](https://github.com/biface/oxiflow/wiki) n'est pas couvert par cette licence :
ses textes et ses figures sont distribués sous licence
[CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/deed.fr). Toute réutilisation
doit citer la source, et l'usage commercial n'est pas autorisé.
