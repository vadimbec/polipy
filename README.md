# Polipy

Visualisation interactive des IRIS sur des axes sociologiques 2D, avec superposition de données électorales françaises (2012–2026).

**Demo** : [vadimbec.github.io/polipy/](https://vadimbec.github.io/polipy/)

---

## Principe

Chaque point est un IRIS (zone de recensement INSEE, ~2 000 habitants). Les axes X/Y sont des scores sociologiques composites calculés depuis les données INSEE Filosofi et RP 2021. La couleur de chaque point correspond au parti dominant lors de l'élection sélectionnée. Les barycentres (gros points) montrent le centre de gravité pondéré de chaque parti sur les axes choisis.

---

## Architecture des scripts

```
shared_config.py            ← config centralisée (partis, scores, élections, fonctions)
build_iris_final.py         ← pipeline de données : INSEE brut → iris_final_socio_politique.csv
rebuild_vizu_iris.py        ← génère saint_graphique_iris.html (desktop, ~163 MB)
rebuild_vizu_iris_mobile.py ← génère saint_graphique_iris_mobile.html (mobile, ~114 MB)
```

### `shared_config.py`

Configuration et fonctions partagées entre desktop et mobile :

- **Partis** : `COULEURS`, `LABELS`, `SHORT`, `OPACITY`, `ORDER`, `ALL_ORDER` (~40 partis/candidats)
- **Scores** : `SCORES_CONFIG_GROUPED_PCA` — architecture en groupes pondérés (zscore → rang centile → moyenne des rangs)
- **Axes** : `AXIS_PRESETS` — presets nommés (Saint-Graphique, Bourdieu, Territoire×Précarité, t-SNE, UMAP, PCA, etc.)
- **Variables** : `VARS_BY_CAT`, `VAR_LABELS`, `ALL_VARS` — ~200 variables sélectionnables dans les dropdowns
- **Élections** : `ELECTIONS_AVAILABLE` — 22 scrutins (législatives, présidentielles, européennes, municipales 2012–2026)
- **Fonctions** : `load_iris_base`, `load_election_data`, `compute_parti_dominant`, `compute_marker_sizes`, `compute_jitter_vars`, `compute_barycentres`

### `build_iris_final.py`

Pipeline de reconstruction de `iris/iris_final_socio_politique.csv` depuis les sources INSEE brutes.

**Étapes** :
1. **Hybride** : fusion Census (diplômes, activité, population RP 2021) + FILO IRIS 2021 + fallback commune si données IRIS censurées
2. **BPE** : équipements de proximité normalisés pour 1 000 habitants (santé, enseignement, sports, commerces...)
3. **Logement** : taux propriétaires, HLM, suroccupation, taille logements, modes de chauffage
4. **Démographie** : CSP, diplômes, âge, immigration, mobilité domicile-travail
5. **Scores** : calcul des scores composites via `make_score_pca_grouped` (ACP two-stage pondérée par population)
6. **ML** : imputation `HistGradientBoostingRegressor` pour les variables FILO manquantes ; colonnes `ml_imputed_*` pour traçabilité
7. **Embeddings** (optionnel, `--with-embeddings`) : t-SNE et UMAP sur les scores PCA ; versions `_strict` excluant les IRIS imputés ML

```bash
python build_iris_final.py
python build_iris_final.py --with-embeddings
```

### `rebuild_vizu_iris.py` / `rebuild_vizu_iris_mobile.py`

Génèrent les fichiers HTML autonomes. Structure identique :

1. Chargement des données (`load_iris_base`, `load_election_data`, etc.)
2. Calcul jitter, barycentres pré-calculés par élection
3. Construction des traces Plotly (3 traces : densité population, barycentres, IRIS)
4. Embed de toutes les données JSON dans le HTML (inline)
5. Génération du HTML final avec CSS + JS inline

```bash
python rebuild_vizu_iris.py
python rebuild_vizu_iris_mobile.py

# Réutiliser les JSON déjà générés (évite le recalcul)
python rebuild_vizu_iris.py --skipbuild
```

---

## Scores sociologiques

Les scores sont calculés en 3 étapes :

1. **Zscore pondéré** par population pour chaque variable
2. **Agrégation pondérée** par poids théoriques → indice composite de groupe
3. **Rang centile pondéré** → moyenne des rangs de groupes → score final ∈ [-50, +50]

Scores disponibles (extrait) :

| Score | Description |
|-------|-------------|
| `score_domination` | Position dans la hiérarchie sociale |
| `score_exploitation` | Revenus patrimoniaux/bénéfices vs salaires (axe x Saint-Graphique) |
| `score_cap_eco` | Capital économique (revenus, patrimoine, propriété) |
| `score_cap_cult` | Capital culturel (diplômes, CSP intermédiaires+) |
| `score_precarite` | Précarité sociale (chômage, minima sociaux, taux pauvreté) |
| `score_urbanite` | Degré d'urbanité (densité, transports, appartements vs maisons/voitures) |
| `score_confort_residentiel` | Confort du logement (surface, propriété, grands logements) |
| `score_dependance_carbone` | Dépendance aux énergies fossiles (mobilité + chauffage) |
| `score_pca_1..8` | ACP pondérée sur ~80 variables (composantes principales) |

Les variantes `_strict` excluent les IRIS dont les variables FILO ont été imputées par ML.

---

## Données électorales

22 scrutins couverts :

- **Législatives** : 2012, 2017, 2022, 2024 (T1 + T2)
- **Présidentielles** : 2012, 2017, 2022 (T1 + T2)
- **Européennes** : 2014, 2019, 2024
- **Municipales** : 2014, 2020, 2026 (T1 + T2)


---

## Sources

- **INSEE RP 2021** : `base-ic-diplomes-formation`, `base-ic-activite-residents`, `base-ic-evol-struct-pop`
- **INSEE Filosofi 2021** : `BASE_TD_FILO_IRIS_2021_DISP` (revenus, pauvreté, patrimoine)
- **INSEE BPE 2022** : équipements et services de proximité
- **Ministère de l'Intérieur** : résultats électoraux par bureau de vote
- **COG INSEE 2026** : table des communes

---
