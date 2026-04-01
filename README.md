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
rebuild_vizu_iris.py        ← génère .html (desktop, ~163 MB)
rebuild_vizu_iris_mobile.py ← génère .html (mobile, ~114 MB)
```

### `shared_config.py`

Configuration et fonctions partagées entre desktop et mobile :

- **Partis** : `COULEURS`, `LABELS`, `SHORT`, `OPACITY`, `ORDER`, `ALL_ORDER` (~40 partis/candidats)
- **Scores** : `SCORES_CONFIG_GROUPED_PCA` — architecture en groupes pondérés (zscore → rang centile → moyenne des rangs)
- **Axes** : `AXIS_PRESETS` — presets nommés (Domination x Eploitation, Bourdieu, Territoire×Précarité, t-SNE, UMAP, PCA, etc.)
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

Les scores sont calculés par ACP pondérée (`make_score_pca_grouped`) :

1. **Z-score pondéré** par population pour chaque variable 
2. **ACP pondérée** par population sur les z-scores → PC1
3. Si le score est défini par **plusieurs groupes** de variables : répétition de l'étape 2 par groupe, z-score des PC1 obtenus, puis ACP finale sur ces PC1
4. **Rang centile pondéré** par population → score final ∈ [-50, +50]

Scores disponibles (extrait) :

| Score | Description |
|-------|-------------|
| `score_domination` | Position dans la hiérarchie sociale, pouvoir de décision (CSP, diplômes, stabilité contractuelle) |
| `score_exploitation` | Revenus patrimoniaux/bénéfices vs salaires, rapport aux moyens de productions |
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

### Pipeline élections BV → IRIS

Les résultats électoraux sont fournis par bureau de vote (BV). Le rattachement à un IRIS se fait via une **table de passage** construite en interne :

```
build_contours_iris.py   ← télécharge + fusionne les contours IRIS IGN 2025 (métropole + DOM)
                           → iris/contours_iris_2025.gpkg

build_passage_bv_iris.py ← spatial join BV→IRIS (within + nearest fallback)
                           → table_passage_BV_IRIS.csv

merge_bv_iris.py         ← agrège les voix par BV au niveau IRIS via la table de passage
                           → iris/elections/<id_election>.csv (un fichier par scrutin+tour)
```

#### `build_contours_iris.py`

Télécharge les Contours IRIS IGN 2025 pour la métropole et les DOM depuis [data.geopf.fr](https://data.geopf.fr), les extrait (7-Zip) et les fusionne en un seul GeoPackage WGS84 (`iris/contours_iris_2025.gpkg`).

#### `build_passage_bv_iris.py`

Construit `table_passage_BV_IRIS.csv` (colonnes : `ID_BUREAU_VOTE`, `CODE_IRIS`) par spatial join entre la couche de bureaux de vote ([data.gouv.fr — reconstruction géométrique BV](https://www.data.gouv.fr/datasets/reconstruction-automatique-de-la-geometrie-des-bureaux-de-vote-depuis-insee-reu-et-openstreetmap)) et les contours IRIS :

1. **Within** : le point représentatif du BV tombe dans un IRIS → match direct
2. **Nearest (fallback)** : BV hors contour → IRIS le plus proche
3. **Passe inverse** : IRIS sans aucun BV → BV le plus proche assigné, afin qu'aucun IRIS ne reste vide

#### `merge_bv_iris.py`

Agrège `resultats_elections/candidats_results.parquet` et `resultats_elections/general_results.parquet` au niveau IRIS via `table_passage_BV_IRIS.csv` :

- Mapping nuances/noms de candidats → partis (`NUANCE_TO_PARTI_BASE`, surcharges par élection, mapping présidentielles par nom)
- Calcul des scores (% des exprimés) et du taux d'abstention par IRIS
- Un CSV de sortie par scrutin+tour dans `iris/elections/`

```bash
~/anaconda3/envs/vadim_env/python.exe merge_bv_iris.py
~/anaconda3/envs/vadim_env/python.exe merge_bv_iris.py --elections 2024_legi_t1 2024_legi_t2
~/anaconda3/envs/vadim_env/python.exe merge_bv_iris.py --list   # liste les élections disponibles
```

---

## Sources

- **INSEE RP 2021** : `base-ic-diplomes-formation`, `base-ic-activite-residents`, `base-ic-evol-struct-pop`
- **INSEE Filosofi 2021** : `BASE_TD_FILO_IRIS_2021_DISP` (revenus, pauvreté, patrimoine)
- **INSEE BPE 2022** : équipements et services de proximité
- **Ministère de l'Intérieur** : résultats électoraux par bureau de vote
- **COG INSEE 2026** : table des communes

---