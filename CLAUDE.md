# Polipy — Instructions pour Claude

## Description du projet

Polipy est un script Python générant une visualisation interactive (~163 MB) des IRIS (zones de recensement INSEE) de Saint-Étienne sur des axes sociologiques 2D, avec superposition de données électorales françaises (législatives, présidentielles, européennes, municipales 2017–2024).

## Comment lancer

```bash
conda run -n vadim_env python rebuild_vizu_iris.py
```

Génère `saint_graphique_iris.html` (à ouvrir dans un navigateur).

**Important** : toujours utiliser `conda run -n vadim_env python` pour toute commande Python dans ce projet. L'environnement `vadim_env` contient toutes les dépendances (pandas, plotly, pyarrow, etc.).

## Architecture du fichier principal

`rebuild_vizu_iris.py` (~2730 lignes) = script Python pur qui :
1. Charge les données géographiques/socio (parquet/CSV)
2. Charge les données électorales par IRIS (CSV dans `iris/elections/`)
3. Génère une figure Plotly statique
4. Embed **tout le HTML + CSS + JS** dans un f-string Python → fichier HTML autonome

Le JS est inline dans le f-string Python. Les accolades JS `{}` doivent être doublées `{{}}` dans le f-string.

## Structures de données JS clés

```js
IRIS_ELECTION_DATA[electionId] = {
  colors: Array<string|null>,      // couleur hex parti dominant par IRIS
  partis: Array<string|null>,      // nom parti dominant par IRIS
  scores: Array<Object|null>,      // {RN: 25.6, LFI: 18.2, ...} par IRIS (% des exprimés)
  abst:   Array<number|null>,      // taux abstention % par IRIS
  inscrits: Array<number|null>,    // nb inscrits par IRIS
  exprimes: Array<number|null>,    // nb exprimés par IRIS (base correcte pour calculer les voix absolues)
  blancs:   Array<number|null>,    // nb votes blancs par IRIS
  nuls:     Array<number|null>,    // nb votes nuls par IRIS
}

IRIS_POPS[i]           // population totale IRIS i (proxy si inscrits absent)
IRIS_X[varName][i]     // valeur variable socio X pour IRIS i
baryMeans              // {parti: {varName: valeur}} — barycentres courants (global mutable)
currentBarySizeMap     // {parti: taille_px} — tailles barycentres (global mutable)
currentElectionId      // élection affichée (string)
```

## Fonctions JS importantes

- `applyElection(electionId)` — change l'élection affichée, recalcule tout
- `updateAxis(preset, xVar, yVar, xInvert)` — change les axes, recalcule positions barycentres
- `computeWeightedBaryMeans(electionId)` — barycentres pondérés score×pop sur tous les IRIS
- `computeBarySizes(electionId)` — tailles px barycentres (15–40) proportionnelles aux votes absolus
- `computeAbstBary(electionId)` — barycentre de l'abstention (pondéré par pop×abst)
- `updateAbstPanel(electionId)` — met à jour le panneau stats abstention sous le graphique

## Trace Plotly 0 = barycentres

La trace index 0 contient les barycentres partis + abstention. Mise à jour via `Plotly.restyle(..., [0])`.

## Conventions

- Éviter l'over-engineering. Pas de helpers inutiles.
- Le JS est directement dans le f-string Python — pas de fichiers séparés.
- `IRIS_POPS` est la population totale (pas les inscrits électoraux) — utiliser `inscrits` quand disponible pour les calculs électoraux.
- Les `scores` dans `IRIS_ELECTION_DATA` sont en **% des exprimés** — pour calculer des voix absolues, multiplier par `exprimes`, pas par `inscrits`.
- Elections disponibles : `ELECTIONS_AVAILABLE` dict Python, exporté en JS `ELECTIONS_META`.

## Fichiers locaux non trackés par git (NE PAS SUPPRIMER)

Les dossiers suivants sont dans `.gitignore` et n'existent que localement. Ne jamais les supprimer, déplacer ou écraser :

- `communes/` — données INSEE Filosofi commune (sources pour `build_iris_final.py`) :
  - `communes/FILO2021_DISP_COM.csv`
  - `communes/FILO2021_DISP_PAUVRES_COM.csv`
- `iris/` — fichiers sources INSEE IRIS lourds (base-ic-*.CSV, contours, etc.)
- `data/` — JSON générés par les scripts rebuild (ne pas supprimer manuellement)

Ces fichiers ne sont pas récupérables via git. Si supprimés par erreur, les fichiers `communes/` viennent de **INSEE Filosofi 2021** (téléchargement manuel depuis insee.fr).

## deploy.sh — règles de sécurité

Le script `deploy.sh` utilise `git worktree` pour déployer sur `gh-pages` sans quitter `main`. Il ne doit jamais :
- Faire de `git checkout`, `git reset`, ou `git clean` sur la branche `main`
- Toucher aux fichiers sources (uniquement copier les fichiers générés vers le worktree)
- Être modifié pour inclure des globes (`**`) ou des `rm -rf` sur des dossiers sources

## Mises à jour du journal de bord

Après chaque modification, mettre à jour `TASKS.md` :
- Marquer les tâches terminées dans le tableau de l'update en cours
- Ajouter une entrée dans la section `## Historique` avec la date et le détail des changements effectués
