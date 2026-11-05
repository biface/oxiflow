# Contribuer à oxiflow

## Types de contributions

- **Corrections de bugs** — ouvrir une issue d'abord pour confirmer le bug, puis soumettre une PR.
- **Nouvelles fonctionnalités** — ouvrir une issue et attendre l'accord du mainteneur avant
  d'écrire du code. Les fonctionnalités qui violent les invariants de conception (voir
  [DEVELOPPEMENT.md](DEVELOPPEMENT.md)) ne seront pas acceptées.
- **Documentation et exemples** — toujours bienvenus, sans discussion préalable nécessaire.
- **Benchmarks et tests** — toujours bienvenus, particulièrement pour les modules soumis
  aux invariants FEM (`mesh`, `operators`, `coupling`).
- **Frameworks de niche** — développer une crate `oxiflow-*` spécialisée est explicitement
  encouragé. Voir la section dédiée ci-dessous.

## Workflow

Le développement s'organise en sprints, un par jalon (milestone). Toute pull
request cible la branche `update/x.y.z` d'un sprint ouvert. Les sprints suivent
les jalons : tant que le jalon d'une issue n'a pas de sprint ouvert, le travail
attend.

### Branches et versions

```
update/x.y.z  --PR-->  staging/x.y.z  --PR-->  main  --tag vx.y.z-->  crates.io
```

| Branche | Rôle | Version dans `Cargo.toml` |
|---|---|---|
| `update/x.y.z` | branche de développement du sprint | `x.y.z-rc.N` |
| `staging/x.y.z` | copie de `main` qui reçoit le sprint, pour vérifier que rien ne casse | `x.y.z` |
| `main` | versions stables uniquement | une version de release, sans suffixe de pré-release |

1. Les modifications se font sur `update/x.y.z`, en version `x.y.z-rc.N`.
2. Une pull request de `update/x.y.z` vers `staging/x.y.z` lance les suites de
   tests complètes. Une fois fusionnée, la version passe à `x.y.z` et le code
   est gelé : seuls `README.*`, `CHANGELOG.md` et les fichiers du même ordre
   peuvent encore changer.
3. Une pull request de `staging/x.y.z` vers `main` ajoute une répétition de la
   publication (packaging).
4. Le tag `vx.y.z`, sur `main`, publie la crate sur crates.io. Les branches du
   sprint sont ensuite supprimées.

`scripts/check-version.sh` contrôle la colonne des versions, en CI et dans le
hook pre-push. Une pull request vers `staging/x.y.z` doit porter une version
`x.y.z-rc.N` ; une pull request de `staging/x.y.z` vers `main`, exactement
`x.y.z`.

### Contribuer depuis un fork

```
1. Forker le dépôt sur GitHub
2. Créer une branche depuis la branche update du sprint
       git fetch upstream
       git checkout -b fix/mon-bug upstream/update/x.y.z
3. Activer les hooks git (voir plus bas) et effectuer les modifications
4. Ouvrir une pull request vers update/x.y.z
```

Le nommage des branches est une recommandation : le préfixe reprend le type de
commit de la modification (`feat/`, `fix/`, `docs/`, `bench/`, `chore/`…), et
une branche `feature/` est tout aussi bienvenue. Les branches des contributeurs
sont consolidées dans `update/x.y.z` avant que le sprint passe sur
`staging/x.y.z`. Les pull requests de contributeurs vers `staging/*` ou
`main` seront redirigées.

## Hooks git locaux

Des hooks versionnés se trouvent dans `.githooks/`. Les activer une fois par
clone :

```
git config core.hooksPath .githooks
```

| Hook | Ce qu'il exécute |
|---|---|
| `pre-commit` | refuse les chemins indexés correspondant à `oxiflow.privatePath` (ci-dessous), puis `cargo fmt --check` |
| `commit-msg` | vérifie l'en-tête Conventional Commits (voir [Messages de commit](#messages-de-commit)) |
| `pre-push` | pour chaque branche poussée, la version de `Cargo.toml` selon la branche ; pour un tag `vX.Y.Z`, le tag selon la version et le commit étiqueté selon `<remote>/main` ; puis, si une branche est poussée, `cargo clippy --locked --all-features --all-targets -- -D warnings`, `cargo test --locked --lib` et `cargo test --locked --doc` |

Le hook pre-push est le pendant local des contrôles que la CI lance à chaque
push. Sa compilation en `--all-features` demande les fichiers de développement
HDF5 (`libhdf5-dev` sous Debian et Ubuntu). Tout hook peut être contourné avec
`--no-verify` (`git commit --no-verify`, `git push --no-verify`) ; la CI reste
l'autorité.

**Chemins privés.** Les fichiers de travail qui ne doivent jamais entrer dans
le dépôt se déclarent localement, un motif par entrée :

```
git config --add oxiflow.privatePath '*PROJECT_STATUS*'
```

Un motif est un motif `case` de `sh`, comparé au chemin complet depuis la
racine du dépôt ; contrairement au globbing du shell, `*` couvre aussi `/`. La
suppression d'un tel fichier reste permise, pour pouvoir retirer un fichier
commité par erreur. Sans motif configuré, rien n'est vérifié.

**Limites.**
- `cargo fmt --check` et les commandes cargo du hook pre-push portent sur
  l'arbre de travail, et non sur l'index ou sur les commits poussés. Le hook
  pre-push affiche une note quand des fichiers suivis ont des modifications
  non commitées.
- Un tag est vérifié par rapport à `<remote>/main` tel que connu localement :
  cette référence avance au `git fetch` et lors d'un push vers `main`. Faire
  un `git fetch` d'abord quand `main` a été fusionnée sur GitHub, et pousser
  les tags par un remote nommé.
- Les IDE exécutent aussi les hooks ; dans RustRover, garder *Run Git hooks*
  coché dans les fenêtres de commit et de push.

## Ce que lance la CI, et quand

| Événement | Workflows et contrôles |
|---|---|
| Push sur toute branche | `Format`, `Version check`, `Clippy` (features par défaut, chaque feature seule, toutes les features), `Unit and doc tests`, `Security audit` |
| Pull request vers `staging/**` ou `main` | les précédents, plus `Test` (un job par jeu de features), `MSRV`, `SemVer checks`, `Documentation`, `Reference examples`, `Coverage` |
| Pull request vers `main` | en plus, `Publish dry run` |
| Push sur `main` | `Coverage` (référence Codecov), Release Drafter ; la documentation est déployée sur GitHub Pages une fois la CI réussie |
| Tag `vX.Y.Z` | `publish.yml` : `Verify release`, puis `Publish to crates.io` |
| Chaque lundi, 06:00 UTC | `Security audit` |

Les pull requests vers `update/x.y.z` ne lancent aucun de ces contrôles dans ce
dépôt : les contrôles de push tournent dans votre fork si GitHub Actions y est
activé, et les suites complètes tournent quand le sprint passe sur
`staging/x.y.z`.

## Exigences techniques

Toute pull request doit satisfaire l'ensemble des points suivants avant d'être fusionnée.

Ce sont les contrôles que le CI exécute sur une pull request vers `staging/x.y.z` ; les
lancer en local avant de demander une revue évite un aller-retour. `--all-features` demande
les fichiers de développement HDF5 (`libhdf5-dev` sous Debian et Ubuntu). Le seul test qui
demande un adaptateur graphique est ignoré ; le lancer avec
`cargo test --features gpu -- --ignored` sur une machine qui en a un.

**Tests**
- `cargo test --all-features` passe.
- Toute nouvelle fonction publique a au moins un test.
- `cargo test --test fem_invariants` passe.
- `scripts/run-examples.sh` exécute tous les exemples en release, exemples de
  référence compris, et rassemble les figures qu'ils tracent ; il se termine
  en erreur si l'un d'eux échoue.

**Qualité du code**
- `cargo fmt --all -- --check` ne signale rien.
- `cargo clippy --all-targets --all-features -- -D warnings` propre ; le CI lance aussi
  clippy avec chaque feature seule (`sparse`, `hdf5`, `gpu`, `parallel`, `serde`, `vtk`) et
  avec les features par défaut.
- `RUSTDOCFLAGS="-D warnings" cargo doc --no-deps --all-features` sans avertissement.

**Couverture** (Codecov, `.codecov.yml`)
- Couverture globale ≥ 85 %, avec une tolérance de 2 % d'un commit à l'autre.
- Lignes nouvelles (couverture du patch) ≥ 80 %.
- `context`, `model` et `mesh` ≥ 90 %, et les modules soumis aux invariants FEM (`mesh`,
  `operators`, `coupling`) ≥ 90 % pris ensemble.
- Si la modification fait baisser la couverture, ajouter les tests manquants avant
  de demander une revue.

**Stabilité de l'API** ([DQ-016](https://github.com/biface/oxiflow/issues/16))
- `cargo semver-checks` tourne sur chaque pull request vers `staging/x.y.z` et `main`.
  Tant que la crate est en 0.x, une version mineure peut casser l'API publique, mais la
  rupture doit être déclarée : `!` dans l'en-tête du commit et une entrée `**Breaking:**`
  dans le CHANGELOG.

**CHANGELOG**
- Ajouter une entrée sous `[Unreleased]` dans `CHANGELOG.md`.
- Utiliser `### Added`, `### Changed`, `### Fixed` ou `### Security` ; un changement
  cassant est une entrée qui commence par `**Breaking:**`, en général sous `### Changed`.
- Une ligne par changement logique.

**Documentation (locale, pour contributeurs)**
- La documentation publique déployée sur GitHub Pages (générée par le CI,
  `cargo doc --no-deps --all-features`) ne couvre volontairement que la surface
  d'API publique — les éléments internes `pub(crate)` (mécanique Newton, dispatch
  bande/creux, etc.) sont abondamment référencés entre eux en rustdoc, mais ne
  font pas partie de ce site public.
- Pour parcourir localement le graphe complet des références croisées internes
  (recommandé avant de relire une PR touchant `solver::methods` ou `operators`) :
  ```
  cargo doc --no-deps --all-features --document-private-items --open
  ```
- Cette commande n'est jamais exécutée en CI et son résultat n'est jamais publié —
  usage local uniquement.

## Invariants de conception

Toute contribution touchant `src/mesh/`, `src/operators/`, `src/coupling/`, ou
tout trait public doit préserver les quatre invariants de conception :

- **INV-1** — `Mesh` reste abstrait ; aucune API publique n'expose `dx`, `nx` ou des
  indices bruts.
- **INV-2** — `DiscreteOperator` reste abstrait ; les intégrateurs restent génériques
  sur le schéma.
- **INV-3** — Le couplage inter-domaines passe uniquement par `CouplingOperator`, jamais
  par un solveur.
- **INV-4** — Tous les traits publics restent object-safe et implémentables depuis une
  crate externe, sans accès aux internes du moteur.

INV-1 à INV-3 sont actifs (depuis v0.1.0, v0.5.0 et v0.3.0) ; `tests/fem_invariants.rs`
vérifie INV-1 et INV-3 depuis l'extérieur de la crate, comme le ferait une crate aval.
INV-4 devient contraignant à v0.30.0 [J30], avec l'architecture de plugins de
[DA-017](https://github.com/biface/oxiflow/issues/17) : les frameworks de niche publiés
par des tiers sur crates.io dépendent de l'object-safety des traits publics du moteur.
D'ici là, une modification qui rend un trait public non object-safe est un changement
cassant, déclaré comme tel.

## Développer un framework de niche

Si vous développez un framework spécialisé (`oxiflow-chrom`, `oxiflow-geo`, ou tout
autre domaine), voici la démarche recommandée.

**Avant de commencer**, ouvrez une [Discussion
GitHub](https://github.com/biface/oxiflow/discussions) décrivant le domaine, les modèles
envisagés et le public cible. Cela évite les redondances avec des travaux en cours et
permet au framework d'être référencé dans la documentation officielle de l'écosystème.

**Structure d'une crate framework de niche :**

```
oxiflow-votredomaine/
├── Cargo.toml          # dépend de oxiflow = "0.10", versioning indépendant
├── NOTICE              # requis par Apache 2.0 — doit mentionner le copyright oxiflow
├── LICENSE             # votre propre licence Apache 2.0 (ou compatible)
├── src/
│   ├── lib.rs
│   ├── models/         # implémentations de PhysicalModel
│   ├── boundary/       # implémentations de BoundaryCondition
│   ├── calculators/    # implémentations de ContextCalculator
│   └── config.rs       # configuration TOML (optionnel, pour la CLI)
├── examples/
└── tests/
```

Tant qu'oxiflow est en 0.x, chaque version mineure peut casser l'API publique
([DQ-016](https://github.com/biface/oxiflow/issues/16)) : dépendre d'une seule mineure
(`"0.10"` n'admet que 0.10.x) et passer à la suivante en connaissance de cause.

**Pattern d'enregistrement (prévu)** — la CLI `oxiflow` et son registre de plugins
arrivent à v0.30.0 [J30] ([DA-017](https://github.com/biface/oxiflow/issues/17),
[DT-018](https://github.com/biface/oxiflow/issues/18)) ; `PluginRegistry` n'existe pas
encore dans la crate. La fonction d'enregistrement envisagée a cette forme :

```rust,ignore
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
liste de l'écosystème dans `DEVELOPPEMENT.md` (§15, Frameworks de l'écosystème connus).
C'est la seule contribution au dépôt du moteur requise pour un framework tiers.

## Checklist de pull request

- [ ] `cargo test --all-features` passe
- [ ] `cargo test --test fem_invariants` passe
- [ ] `cargo fmt --all -- --check` ne signale rien
- [ ] `cargo clippy --all-targets --all-features -- -D warnings` propre
- [ ] `RUSTDOCFLAGS="-D warnings" cargo doc --no-deps --all-features` propre
- [ ] Couverture : 85 % global, 80 % sur les lignes nouvelles, 90 % sur `context`,
      `model`, `mesh` et les modules soumis aux invariants FEM
- [ ] Changement cassant, s'il y en a un, déclaré (`!` dans l'en-tête du commit,
      entrée `**Breaking:**`)
- [ ] `CHANGELOG.md` mis à jour sous `[Unreleased]`
- [ ] La PR cible la branche `update/x.y.z` du sprint ouvert

## Messages de commit

```
type(portée): description courte à l'impératif

Corps optionnel expliquant le pourquoi, pas le quoi.
```

Types : `feat`, `fix`, `docs`, `style`, `refactor`, `perf`, `test`, `build`,
`ci`, `chore`, `revert`, `bench`.

La portée est optionnelle (minuscules, chiffres, `.`, `_`, `/`, `-`). Un `!`
avant les deux-points signale un changement cassant (breaking change), comme
dans `fix(mesh)!: return OxiflowError from UniformGrid1D::new`. Les en-têtes
générés par git (`Merge …`, `Revert "…"`, `fixup! …`, `squash! …`,
`amend! …`) sont acceptés. Le hook `commit-msg` vérifie l'en-tête.

## Questions

Démarrer une [Discussion GitHub](https://github.com/biface/oxiflow/discussions) pour les
questions d'usage et les propositions informelles ; ouvrir une issue pour un bug ou une
demande de fonctionnalité.

## Licence

En contribuant à oxiflow, vous acceptez que vos contributions soient distribuées sous
la Licence Apache, Version 2.0, la même licence que le projet. Le fichier `NOTICE` doit
être conservé dans toute redistribution.
