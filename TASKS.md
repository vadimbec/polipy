# Polipy — Journal de bord des tâches


## Historique

### 2026-03-26 — OSM lieux de culte : exploration + notebook corrélations politique

- **`export.geojson`** (Overpass, 170 MB) : 60 993 lieux de culte France entière avec `religion` + `denomination`. Bien plus riche que RNA/BPE.
- **`osm_cultes/`** : nouveau dossier contenant 5 scripts d'analyse :
  - `01_osm_explore.py` — extraction centroïdes + stats tags → `osm_lieux_culte_centroides.csv`
  - `02_osm_join_iris.py` — jointure spatiale (geopandas) → `osm_iris_counts.csv` (catégories : catholic, islam_total, jewish, evangelical, protestant, buddhist, etc.)
  - `03_osm_normalise.py` — normalisation pour 1000 hab → `osm_iris_pour1000.csv`
  - `04_osm_vs_rna_compare.py` — comparaison OSM vs RNA : corrélations faibles/négatives → OSM et RNA mesurent des choses différentes (bâtiments vs associations)
  - `05_osm_eda.py` — distributions + top IRIS
- **`eda_osm_cultes_politique.ipynb`** : nouveau notebook focalisé corrélations partielles OSM × vote :
  - §1 : heatmaps par culte (10 nuances × 13 élections), une figure par culte
  - §2 : synthèse T1 seulement (tous cultes × toutes nuances, 10 subplots)
  - §3 : évolution temporelle par culte (courbes ρ par nuance)
  - §4 : table des effets les plus forts (|ρ| décroissant)

### 2026-03-25 — Variables électorales comme axes + réduction tailles points
- **Nouvelles variables dans les sélecteurs d'axes** : ajout d'une catégorie `Élections` (~85 variables) dans `VARS_BY_CAT` des deux scripts (desktop + mobile). Inclut : `pop_iris`, `pop_commune` (dans `Démographie`), `pct_abst_*` pour 12 élections (legi/pres/euro 2017–2024), et `pct_PARTI_*` pour tous les partis/candidats disponibles par élection (RN, LFI, NUPES, NFP, ENS, LR, PS, PS_PP, EELV, PCF, EXD, UXD, REC, LE_PEN, MACRON, MÉLENCHON, FILLON, PÉCRESSE, HAMON, ZEMMOUR, JADOT, ROUSSEL, HIDALGO). Legi 2017 T2 : tous les partis sauf AUTRE. Legi 2024 T2 : RN, NFP, UXD, LR, ENS.
- **Fonction `_build_electoral_axis_vars(df)`** : ajoutée dans les deux scripts après la section 3b. Charge les CSV électoraux depuis `iris/elections/`, merge sur CODE_IRIS, et assigne les colonnes au dataframe principal. Cache les CSV par ID d'élection pour éviter les relectures.
- **Tailles de points desktop** : barycentres réduits de 8–45 px → 7–38 px (fallback 22→18) ; petits points IRIS 2.5–6 px → 2–5 px.
- **Tailles de points mobile** : barycentres réduits de 8–45 px → 6–32 px (fallback 22→15) ; petits points IRIS 2.5–4 px → 1.5–3 px.
- `data/iris_x_desktop.json` et `iris_y_desktop.json` passent à ~47 MB (doublement, attendu : ~85 nouvelles variables × N_IRIS).

### 2026-03-24 — T15/T16 : Fallback T1 + code IRIS dans fiche
- **T15** : Quand un IRIS n'a pas de données T2 (élu·e au 1er tour avec +50%), les fonctions `applyElection`, `restyleIRIS`, `buildMapGeoJSON` et les fiches info utilisent désormais les données T1 comme fallback. Les IRIS concernés apparaissent avec leur couleur T1 mais délavée (mélange 45% couleur + 55% blanc) pour signaler visuellement qu'il s'agit de données T1. La fiche info affiche les scores T1 avec la mention *"Élu·e au T1 — résultats du T1 affichés"*. Les données T1 sont chargées en parallèle avec les données T2 dans `applyElection` (`Promise.all`). Appliqué dans les deux versions desktop et mobile.
- **T16** : La fiche info affiche désormais le code IRIS (`cd[0]`, ex. `420011101`) en gris clair après le nom de commune. Évite la confusion avec les stats de la commune entière.

### 2026-03-19 — T12/T13/T14 : Remplacement jolivet + audit + nouvelles variables
- **T12** : Cellule finale `eda_iris.ipynb` (id `54ad0c7f`) réécrite. Suppression de la dépendance à `iris-stats.geojson` (jolivet). Les variables démographiques (`pop_totale`, `pct_0_19/20_64/65_plus`, `pct_etrangers`, `pct_immigres`, `age_moyen`, `pct_csp_*`) sont maintenant calculées directement depuis `iris/base-ic-evol-struct-pop-2020.CSV` avec les mêmes formules que `template_merge_iris_bv/utils.py`. La fusion est une LEFT join au lieu d'une INNER join → passage de 45 650 à ~49 280 IRIS conservés (+3 630). `nom_commune` récupéré via `cog_ensemble_2026_csv/v_commune_2026.csv`. Correction du bug `utils.py` : `pct_0_19/20_64/65_plus` utilisaient `P20_F*` (femmes seulement) au lieu de `P20_POP*` — valeurs ~2× plus correctes. Toutes les colonnes brutes `P20_*/C20_*` incluses dans `iris_final_socio_politique.csv`.
- **T13** : Nouvelle cellule d'audit FILOSOFI insérée après la cellule ML. Compare les NaN à 3 étapes : (A) IRIS FILO seul, (B) + fallback commune, (C) + imputation ML. Quantifie si le fallback commune est utile (hypothèse : Gain B-A faible car communes petites aussi censurées).
- **T14** : `rebuild_vizu_iris.py` mis à jour : ajout de 80+ entrées dans `VAR_LABELS` pour toutes les colonnes `P20_*/C20_*` de `base-ic-evol-struct-pop-2020.CSV`. Nouvelles variables calculées (`pct_femmes`, `taille_menage_moy`, `pct_hors_menage`, `ecart_csp_plus_hf`) ajoutées à `VARS_BY_CAT['Démographie']` et à `VAR_LABELS`. Variables `pct_0_19`, `pct_20_64`, `pct_65_plus`, `pct_immigres` également ajoutées à `VARS_BY_CAT['Démographie']`.

### 2026-03-19 — T4 : Feature parity mobile ↔ desktop
- **Architecture** : refactorisation complète de `build_mobile_html()`. Remplacement de `groupDataX/Y` + ~10 traces Plotly par la même architecture que le desktop : `IRIS_X/Y`, `IRIS_ELECTION_DATA`, `IRIS_INFO`, `IRIS_POPS`, `MARKER_SIZES`, `ELECTIONS_META`, `GROUP_INDICES` + 2 traces Plotly (barycentres + IRIS unique).
- **Election-bar** : ajout d'une barre de sélection Type/Année/Tour (boutons T1/T2) sous l'axis-bar. CSS compact adapté mobile (font 9px, padding réduit, `-webkit-tap-highlight-color: transparent`).
- **Changement d'élection** : `applyElection()` copié du desktop — recalcule couleurs IRIS, barycentres, barycentre abstention, stats abstention, label élection. Smart state preservation (wasAllOn/wasAllOff) des partis filtrés au changement d'élection.
- **Barycentres dynamiques** : `restyleBarycentres()` avec barycentre abstention (point gris "Abst.") — identique au desktop.
- **Panneau stats abstention** : version compacte mobile sous le graphe (1 ligne : % abst + top 4 partis).
- **Fiche IRIS dynamique** : `showCard(irisGlobalIdx)` lit depuis `IRIS_ELECTION_DATA[currentElectionId]` — label élection dynamique (`#cardElecLabel`), scores votes en grille 3 colonnes avec les vrais scores de l'élection courante.
- **Touch gestures** : conservés sans modification (pinch-zoom, pan, tap → findNearest → showCard par index global).
- **Suppression** : `_build_js_data()` et `_build_trace_data()` toujours présentes (non supprimées par sécurité, non appelées).
- Outputs : `saint_graphique_iris_mobile.html` maintenant généré par défaut (~114 MB).

### 2026-03-18 — T3 Phase A : Optimisation taille et performance (231→114 MB, -51%)
- **A1** : Suppression `groupDataX`/`groupDataY` redondants (-13 MB). Desktop utilise `IRIS_X`/`IRIS_Y` + `GROUP_INDICES`.
- **A2** : Traces initiales vides (`x:[], y:[]`) — `applyElection()` remplit les données après `Plotly.newPlot()` (-11 MB).
- **A3** : `ALL_CUSTOMDATA` réduit à 13 champs → `IRIS_INFO`. Scores partis lus depuis `IRIS_ELECTION_DATA.scores` (-5 MB).
- **A4** : Précision floats réduite (3 déc. scores composites, 1 déc. %, 0 déc. entiers). Plus gros gain (-88 MB).
- **A5** : 35 traces `scattergl` fusionnées en 1 trace unique. `restyleIRIS()` et `restyleBarycentres()` remplacent les boucles par-trace. `setGroupVisible()` filtre via `activeGroups` Set. Click utilise `customdata` = index global IRIS.
- **A6** : Barycentres pré-calculés en Python (`baryMeans`, `barySizes`, `abstBary`, `buttonPcts` par élection). JS fait un simple lookup au lieu d'itérer 45650 IRIS × partis × variables. Suppression de `computeWeightedBaryMeans`, `computeBarySizes`, `computeBaryMeans`, `ALL_VARS_LIST`.
- Perf estimée : changement d'élection 8s→<2s, changement d'axes 5s→<1s (à tester en navigateur).

### 2026-03-18 — Suppression ellipses σ et bouton associé
- Les ellipses 1σ étaient trop larges et se chevauchaient → illisibles. Les barycentres seuls suffisent.
- Supprimé : bouton "Ellipses σ", variables `densityVisible`/`densityTraceIndices`, fonctions `computeWeightedStdDevs`, `hexToRgb`, `buildEllipseTraces`, `showDensityTraces`, `hideDensityTraces`, `toggleDensity`.
- Conservé : barycentres avec échelle sqrt 8-45px et filtre >2% des exprimés.

### 2026-03-18 — Ellipses σ + ajustement barycentres + filtre 2%
- Remplacement des density contours (`histogram2dcontour`, illisibles) par des **ellipses 1σ** (`scattergl` fill) pour chaque parti > 2% + abstention. Légères et lisibles.
- Nouvelle fonction `computeWeightedStdDevs(electionId, xVar, yVar, xInvert)` : calcule σx/σy pondérés (w = score × pop) pour chaque parti et l'abstention.
- `buildDensityTraces` → `buildEllipseTraces` : génère des traces `scattergl` mode `lines` + `fill: 'toself'` (51 points par ellipse, remplissage 8% opacité, contour 35%).
- `computeBarySizes` : échelle sqrt 8-45px au lieu de linéaire 15-40px → plus de contraste entre petits et gros partis.
- Filtre barycentres : seuls les partis > 2% des exprimés sont affichés (au lieu de top 15 par nb IRIS dominés). Appliqué dans `applyElection()` et `applyAxes()`.
- Suppression de `computeTop4Parties` (plus utilisée).
- Bouton renommé "Densités" → "Ellipses σ".

### 2026-03-18 — Correctif Plotly.addTraces indices pour density
- Bug : `Plotly.addTraces(chartDiv, traces, 1)` passait un entier unique comme 3e argument pour 5 traces → erreur `traces.length must equal indices.length`.
- Correction : passe `traces.map((_, i) => 1 + i)` comme tableau d'indices.

### 2026-03-18 — Correctif density traces (IRIS_X→IRIS_Y + top 3→4 partis)
- Bug : `buildDensityTraces()` lisait `IRIS_X[yVar]` pour l'axe Y au lieu de `IRIS_Y[yVar]` → coordonnées Y toujours `undefined` → aucun point tracé → contours vides.
- Correction : `IRIS_Y[yVar]` aux deux endroits (boucle partis + boucle abstention).
- `computeTop3Parties` renommé `computeTop4Parties`, `.slice(0, 3)` → `.slice(0, 4)` : affiche les 4 partis dominants au lieu de 3.

### 2026-03-18 — T7+T10 : Density contours (toggle, top 3 partis + abstention)
- Nouveau bouton "Densités" dans la barre d'axes (off par défaut).
- `computeTop3Parties(electionId)` : calcule les 3 partis avec le plus de voix absolues pour l'élection courante.
- `buildDensityTraces(electionId, xVar, yVar, xInvert)` : construit 3 traces `Histogram2dContour` (top 3 partis + abstention, opacité 0.22, 35 bins, sans lignes de contour).
- `showDensityTraces()` / `hideDensityTraces()` / `toggleDensity()` : gestion du toggle, traces insérées à l'index 1 (derrière les IRIS, devant les barycentres).
- `hexToRgb(hex)` : helper inline hex → [r,g,b].
- Mise à jour automatique dans `applyElection()` et `applyAxes()` si toggle actif.
- T7 : barycentre abstention vérifié correct (poids = pop × abst%), density contour abstention ajouté.
- T10 : density contours top 3 partis, lisibilité préservée par sélection restreinte et opacité faible.

### 2026-03-18 — Correctif résidu applyElection écrasant b.pct
- Suppression du bloc de calcul `b.pct` résiduel dans `applyElection()` qui utilisait `inscrits` comme base et écrasait le résultat de `computeButtonPcts()` → boutons haut maintenant identiques au panneau bas (% des exprimés).

### 2026-03-18 — Cohérence % partis haut/bas + ajout blancs/nuls
- `computeButtonPcts(electionId)` : remplace `initPct()` — calcule les % des boutons filtres en % des exprimés pondérés par IRIS (`exprimes[i] × score[i] / 100 / total_exprimes`). Appelé aussi dans `applyElection()` pour recalcul à chaque changement d'élection.
- `updateAbstPanel()` : partis affichés avec double % (exp. et ins.) + blancs et nuls ajoutés. Python exporte `blancs` et `nuls` par IRIS dans `IRIS_ELECTION_DATA`.

### 2026-03-18 — Correctif voix absolues partis (base de calcul exprimes)
- `updateAbstPanel()` JS : les scores sont en % des exprimés, mais le calcul des voix absolues utilisait `inscrits` comme base → voix ~2× trop élevées (ex: NUPES 11.27M au lieu de ~5.8M). Fix : ajout de `exprimes` par IRIS dans `IRIS_ELECTION_DATA` (Python + JS export), et utilisation de `exprimes` comme base dans `updateAbstPanel`.

### 2026-03-17 — Correctifs colorimétriques
- Corrigé couleur parti REG : `#89712FFF` → `#89712F` (format hex 8 chiffres non supporté par Plotly)

### 2026-03-17 — Correctif dtype pandas (bug critique inscrits)
- `_load_election_iris_data()` : ajout `dtype={'CODE_IRIS': str}` dans `pd.read_csv` — sans ce fix, pandas lisait les codes IRIS avec un type mixte (str/int selon le chunk), ce qui faisait échouer le merge silencieusement : seulement ~15k IRIS matchés au lieu de ~44k, donnant 13M inscrits au lieu de ~43M.

### 2026-03-17 — Correctifs cohérence scores
- Scores boutons filtres (haut) : maintenant pondérés par inscrits par IRIS, même base que le panneau stats bas (% des inscrits). Avant : moyenne non pondérée des % par IRIS.
- Barycentre abstention : taille calculée dynamiquement via `computeBarySizes()` (clé `__ABST__`) au lieu de hardcodé 18px. L'abstention ~50% est maintenant proportionnellement grande.
- `has_inscrits` et `has_exprimes` déplacés hors de la boucle Python dans `_load_election_iris_data()`.

### 2026-03-17 — Update 3 en cours
- T11 : panneau stats sous le graphique — abstention % + absolu (inscrits×abst/100) + top 7 partis en % inscrits et absolu. Fonction JS `updateAbstPanel(electionId)` appelée dans `applyElection()`. Export Python `inscrits` par IRIS ajouté dans `IRIS_ELECTION_DATA`.

### 2026-03-17 — Update 2 terminée
- T8 : nouvelle fonction JS `computeBarySizes(electionId)` — taille barycentre = `15 + (votes_parti - min) / (max - min) × 25` px, avec `votes_parti = sum(pop_IRIS × score_parti/100)`. Variable globale `currentBarySizeMap` partagée entre `applyElection()` et `updateAxis()`.

### 2026-03-17 — Update 1 terminée
- T1 : nouvelle fonction JS `computeWeightedBaryMeans(electionId)` — pondère par `score_parti × population` sur tous les IRIS (remplace l'ancienne pondération pop seule sur IRIS dominants uniquement). Appelée dans `applyElection()`.
- T2 : ajout `.then(() => applyElection(DEFAULT_ELECTION_ID))` après `Plotly.newPlot(...)` — force le recalcul complet des barycentres dès le premier chargement.
- T9 : suppression du bloc `legend-area` (CSS + div HTML + fonction JS `rebuildLegend` + ses appels) — les boutons du haut suffisent.
