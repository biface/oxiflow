# oxiflow

[![CI](https://github.com/[USER]/oxiflow/actions/workflows/ci.yml/badge.svg)](https://github.com/biface/oxiflow/actions/workflows/ci.yml)
[![Couverture](https://codecov.io/gh/[USER]/oxiflow/branch/main/graph/badge.svg)](https://codecov.io/gh/biface/oxiflow)
[![Crates.io](https://img.shields.io/crates/v/oxiflow.svg)](https://crates.io/crates/oxiflow)
[![Docs.rs](https://docs.rs/oxiflow/badge.svg)](https://docs.rs/oxiflow)
[![Licence](https://img.shields.io/badge/licence-Apache--2.0-blue.svg)](#licence)

**Moteur numérique générique pour les problèmes de transport–réaction–diffusion.**

oxiflow fournit les blocs architecturaux permettant de construire des solveurs d'équations aux
dérivées partielles rigoureux, maintenables et performants en Rust — des modèles de
chromatographie 1D aux systèmes multi-physiques couplés les plus complexes. L'architecture
supporte les grilles structurées (v1.0), les maillages éléments finis non structurés (v2.0),
et une famille de frameworks de niche (v3.0).

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

Chaque framework est une crate indépendante qui dépend du moteur et apporte des modèles
physiques, conditions aux limites, nomenclature et configuration déclarative propres à
son domaine. Des tiers peuvent publier leurs propres frameworks `oxiflow-*` sur crates.io
en utilisant la même API de plugin — le moteur expose des points d'extension stables et
object-safe précisément dans ce but.

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
  de discrétisation ; FD, FV et FEM (v2.0) s'enfichent sans réécrire les intégrateurs
- **Maillage abstrait** — le trait `Mesh` libère `PhysicalState` de toute hypothèse de grille
- **Bibliothèque d'intégrateurs** — Euler, RK4, DoPri45, Euler implicite, Crank–Nicolson,
  BDF2/3, IMEX (splitting de Strang)
- **Schémas spatiaux** — FD décentrées/centrées, WENO3/5, FV conservatifs, Lax–Wendroff,
  limiteurs de flux (MinMod, Van Leer, Superbee)
- **API plugin-safe** — tous les traits publics sont object-safe, permettant à des crates
  tierces d'implémenter des frameworks de niche (v2.0 — INV-4)
- **Parallélisme optionnel** — Rayon, activé via le feature flag `parallel`

---

## Démarrage rapide

```toml
[dependencies]
oxiflow = "0.2"
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
`∂u/∂t + ∇·F(u, ∇u) = S(u, x, t)` est un candidat :

| Domaine | Exemple | Framework cible |
|---|---|---|
| Chromatographie | Élution gradient multi-solutés | `oxiflow-chrom` |
| Transfert thermique | Refroidissement transitoire 1D | `oxiflow-thermo` |
| Réaction–diffusion | Motifs de Turing (Gray–Scott) | moteur direct |
| Mécanique des fluides | Couche limite de Burgers | moteur direct |
| Géomécanique | Consolidation de Terzaghi | moteur direct |
| Géophysique de surface | Interaction lahar–lac | `oxiflow-geo` |

---

## Invariants de conception

Quatre contraintes garantissent une évolution sans breaking change de v1.0 à v3.0 et
assurent la compatibilité des frameworks tiers entre les versions du moteur :

| Invariant | Description | Introduit en |
|---|---|---|
| **INV-1** | `Mesh` est abstrait — `PhysicalState` ne présuppose aucune structure de grille | v0.2 |
| **INV-2** | `DiscreteOperator` est abstrait — les intégrateurs sont génériques sur le schéma | v0.6 |
| **INV-3** | `CouplingOperator` supporte des domaines distincts avec interfaces mobiles | v0.4 |
| **INV-4** | Tous les traits publics sont object-safe — des crates tierces peuvent les implémenter | v2.0 |

---

## État de Développement

| Jalon | Version | Statut | Thème |
|---|---|---|---|
| J0 — Fondations | v0.1 | ✅ Publié | Placeholder · CI · structure projet |
| J1 — Architecture cœur | v0.2 | 🔄 En cours | ContextValue · OxiflowError · Mesh (INV-1) |
| J2 — Contexte complet | v0.3 | ⏳ Planifié | BCs requirantes · ordonnancement |
| J3 — Multi-composants | v0.4 | ⏳ Planifié | PhysicalQuantity · CouplingOperator (INV-3) |
| J4 — Solveurs | v0.5–0.6 | ⏳ Planifié | Intégrateurs · DiscreteOperator (INV-2) |
| J5 — Performance | v0.7 | ⏳ Planifié | Rayon · cache · benchmarks |
| J6 — Écosystème v1.0 | v1.0 | ⏳ Planifié | 7 exemples · audit FEM · API stable |
| J7 — FEM | v2.0 | 🔭 Horizon | Maillages non structurés · ALE · INV-4 plugin-safe |
| J8 — Frameworks | v3.0 | 🔭 Horizon | oxiflow-chrom · oxiflow-geo · CLI `oxiflow run` |

Voir [DEVELOPPEMENT.md](DEVELOPPEMENT.md) pour la spécification architecturale complète.

---

## Feature Flags

| Flag | Description | Disponible dès |
|---|---|---|
| *(défaut)* | Moteur cœur, exécution séquentielle | v0.2 |
| `parallel` | Parallélisme Rayon pour les calculateurs indépendants | v0.7 |
| `serde` | Sérialisation des états et scénarios | v0.7 |
| `hdf5` | Import/export HDF5 pour données tabulées externes | v0.7 |

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

Copyright 2026 [ton nom]

Distribué sous la [Licence Apache, Version 2.0](LICENSE-2.0.txt).

Ce logiciel peut être utilisé, distribué et modifié librement, y compris à des fins
commerciales, à condition de conserver la notice de copyright et le fichier `NOTICE`
dans toute redistribution. La licence inclut une clause de représailles brevets
protégeant l'auteur — voir le texte complet pour les détails.
