# Contribuer à oxiflow

## Types de contributions

- **Corrections de bugs** — ouvrir une issue d'abord pour confirmer le bug, puis soumettre une PR.
- **Nouvelles fonctionnalités** — ouvrir une issue et attendre l'accord du mainteneur avant
  d'écrire du code. Les fonctionnalités qui violent les invariants de conception (voir
  [DEVELOPPEMENT.md](DEVELOPPEMENT.md)) ne seront pas acceptées.
- **Documentation et exemples** — toujours bienvenus, sans discussion préalable nécessaire.
- **Benchmarks et tests** — toujours bienvenus, particulièrement pour les composants INV
  (`mesh`, `coupling`, `solver_spatial`).
- **Frameworks de niche** — développer une crate `oxiflow-*` spécialisée est explicitement
  encouragé. Voir la section dédiée ci-dessous.

## Workflow

```
1. Forker le dépôt sur GitHub
2. Créer une branche depuis develop — pas depuis main
       git checkout develop
       git pull upstream develop
       git checkout -b fix/mon-bug        # ou feat/, docs/, bench/, chore/
3. Effectuer les modifications
4. Ouvrir une pull request vers develop
```

`main` ne reçoit des merges depuis `develop` qu'à chaque release.
Les PRs directes vers `main` seront redirigées vers `develop`.

## Exigences techniques

Toute pull request doit satisfaire l'ensemble des points suivants avant d'être fusionnée.

**Tests**
- `cargo test --all-features` passe.
- Toute nouvelle fonction publique a au moins un test.
- `cargo test --test fem_invariants` passe.

**Qualité du code**
- `cargo fmt --all` appliqué, aucune modification de formatage non commitée.
- `cargo clippy --all-targets --all-features -- -D warnings` propre.

**Couverture**
- Couverture globale ≥ 85% (suivie sur Codecov).
- Composants INV (`mesh`, `coupling`, `solver_spatial`) ≥ 90%.
- Si la modification fait baisser la couverture, ajouter les tests manquants avant
  de demander une revue.

**CHANGELOG**
- Ajouter une entrée sous `[Unreleased]` dans `CHANGELOG.md`.
- Utiliser `### Added`, `### Changed`, `### Fixed`, ou `### Breaking`.
- Une ligne par changement logique.

## Invariants de conception

Toute contribution touchant `src/mesh/`, `src/coupling/`, `src/solver/spatial/`, ou
tout trait public doit préserver les quatre invariants de conception :

- **INV-1** — `Mesh` reste abstrait ; `PhysicalState` ne doit pas acquérir d'hypothèses
  de grille.
- **INV-2** — `DiscreteOperator` reste abstrait ; les intégrateurs restent génériques
  sur le schéma.
- **INV-3** — `CouplingOperator` supporte des domaines distincts avec interfaces mobiles.
- **INV-4** — Tous les traits publics restent object-safe ; aucun breaking change pour
  les crates externes.

INV-4 est critique à partir de v2.0 : les frameworks de niche publiés par des tiers sur
crates.io dépendent de la stabilité et de l'object-safety de l'API publique du moteur.
La violation de cet invariant bloque le merge indépendamment des résultats des tests.

## Développer un framework de niche

Si vous développez un framework spécialisé (`oxiflow-chrom`, `oxiflow-geo`, ou tout
autre domaine), voici la démarche recommandée.

**Avant de commencer**, ouvrez une Discussion GitHub décrivant le domaine, les modèles
envisagés et le public cible. Cela évite les redondances avec des travaux en cours et
permet au framework d'être référencé dans la documentation officielle de l'écosystème.

**Structure d'une crate framework de niche :**

```
oxiflow-votredomaine/
├── Cargo.toml          # dépend de oxiflow = "2", versioning indépendant
├── NOTICE              # requis par Apache 2.0 — doit mentionner le copyright oxiflow
├── LICENSE             # votre propre licence Apache 2.0 (ou compatible)
├── src/
│   ├── lib.rs
│   ├── models/         # implémentations de PhysicalModel
│   ├── boundary/       # implémentations de BoundaryCondition
│   ├── calculators/    # implémentations de ContextCalculator
│   └── config.rs       # désérialisation TOML (optionnel, pour intégration CLI)
├── examples/
└── tests/
```

**Pattern d'enregistrement** — implémenter la fonction d'enregistrement pour que
votre framework s'intègre à la CLI `oxiflow` (v3.0) :

```rust
// Dans oxiflow-votredomaine/src/lib.rs
pub fn register(registry: &mut oxiflow::PluginRegistry) {
    registry.register_model("nom-du-modele", VotreModelFactory);
    registry.register_bc("nom-de-la-bc",    VotreBCFactory);
}
```

**Convention de nommage** — utiliser le préfixe `oxiflow-` sur crates.io. Ce n'est pas
imposé mais facilite la découverte et signale la compatibilité avec le moteur.

**Couverture** — viser ≥ 80% dans votre crate. Utiliser le même setup
`cargo-llvm-cov` + Codecov que le moteur si vous souhaitez un badge de couverture.

**Une fois publié**, ouvrir une PR sur le dépôt du moteur pour ajouter votre crate à la
liste de l'écosystème dans `DEVELOPPEMENT.md`. C'est la seule contribution au dépôt du
moteur requise pour un framework tiers.

## Checklist de pull request

- [ ] `cargo test --all-features` passe
- [ ] `cargo test --test fem_invariants` passe
- [ ] `cargo fmt --all` appliqué
- [ ] `cargo clippy --all-targets --all-features -- -D warnings` propre
- [ ] Codecov ≥ 85% global et ≥ 90% sur les composants INV
- [ ] `CHANGELOG.md` mis à jour sous `[Unreleased]`
- [ ] La PR cible `develop`, pas `main`

## Messages de commit

```
type(portée): description courte à l'impératif

Corps optionnel expliquant le pourquoi, pas le quoi.
```

Types : `fix`, `feat`, `docs`, `bench`, `test`, `chore`, `refactor`.

## Questions

Ouvrir une issue avec le label `question` ou démarrer une Discussion GitHub.

## Licence

En contribuant à oxiflow, vous acceptez que vos contributions soient distribuées sous
la Licence Apache, Version 2.0, la même licence que le projet. Le fichier `NOTICE` doit
être conservé dans toute redistribution.
