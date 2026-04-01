#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Analyse statistique approfondie des IRIS — Socio-démographie × Élections 2022
==============================================================================

Script autonome produisant :
- Part 1 : Analyse agnostique (qualité variables, corrélations, PCA, scores, t-SNE)
- Part 2 : Analyse conditionnée par candidat (profils, PCA cond., LDA, importance, scores)

Usage : conda run -n vadim_env python analyse_iris.py
"""

import os
import sys
import warnings

# Force UTF-8 output on Windows
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if sys.stderr.encoding != 'utf-8':
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from scipy import stats
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from scipy.spatial.distance import squareform
from sklearn.manifold import TSNE
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.inspection import permutation_importance, PartialDependenceDisplay
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.feature_selection import mutual_info_regression

import statsmodels.api as sm
import networkx as nx

try:
    import umap
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False
    print("[!] umap-learn non disponible — section UMAP sera ignorée")

warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

# ═══════════════════════════════════════════════════════════════════════════════
#  CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

OUTPUT_DIR = "analyse_output"
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# Paths
SOCIO_CSV = "iris/iris_final_socio_politique_bis.csv"
ELECTION_CSV = "iris/elections/2022_pres_t1.csv"

CANDIDATES = ['MACRON', 'LE_PEN', 'MELENCHON', 'ZEMMOUR', 'PECRESSE',
              'JADOT', 'ROUSSEL', 'HIDALGO', 'DUPONT_AIGNAN', 'AUTRE']

CANDIDATE_COLORS = {
    'MACRON': '#F97316', 'LE_PEN': '#374151', 'MELENCHON': '#DC2626',
    'ZEMMOUR': '#0F172A', 'PECRESSE': '#3B82F6', 'JADOT': '#16A34A',
    'ROUSSEL': '#9B1C1C', 'HIDALGO': '#DB2777', 'DUPONT_AIGNAN': '#6B7280',
    'AUTRE': '#9CA3AF',
}

# ── Variable inventory ───────────────────────────────────────────────────────
# (var_name, var_type, label)
# Types: 'pct', 'euro', 'ratio', 'rate', 'years', 'count', 'share'

VAR_INVENTORY = {
    'Démographie': [
        ('pct_etrangers', 'pct', '% étrangers'),
        ('pct_immigres', 'pct', '% immigrés'),
        ('age_moyen', 'years', 'Âge moyen'),
        ('pct_femmes', 'pct', '% femmes'),
        ('taille_menage_moy', 'count', 'Taille ménage moy.'),
        ('pct_hors_menage', 'pct', '% hors ménage'),
        ('ecart_csp_plus_hf', 'pct', 'Écart CSP+ H/F'),
        ('pct_0_19', 'pct', '% 0-19 ans'),
        ('pct_20_64', 'pct', '% 20-64 ans'),
        ('pct_65_plus', 'pct', '% 65+ ans'),
    ],
    'CSP': [
        ('pct_csp_agriculteur', 'pct', '% agriculteurs'),
        ('pct_csp_independant', 'pct', '% indépendants'),
        ('pct_csp_plus', 'pct', '% cadres sup.'),
        ('pct_csp_intermediaire', 'pct', '% prof. intermédiaires'),
        ('pct_csp_employe', 'pct', '% employés'),
        ('pct_csp_ouvrier', 'pct', '% ouvriers'),
        ('pct_csp_retraite', 'pct', '% retraités'),
        ('pct_csp_sans_emploi', 'pct', '% sans emploi'),
    ],
    'Revenus': [
        ('DISP_MED21', 'euro', 'Revenu médian €/UC'),
        ('DISP_TP6021', 'ratio', 'Taux pauvreté %'),
        ('DISP_GI21', 'ratio', 'Indice Gini'),
        ('DISP_RD21', 'ratio', 'Rapport interdécile D9/D1'),
        ('DISP_S80S2021', 'ratio', 'Ratio S80/S20'),
    ],
    'Composition revenus': [
        ('DISP_PTSA21', 'share', '% salaires'),
        ('DISP_PPAT21', 'share', '% patrimoine'),
        ('DISP_PPEN21', 'share', '% pensions'),
        ('DISP_PPSOC21', 'share', '% prestations sociales'),
        ('DISP_PCHO21', 'share', '% allocations chômage'),
        ('DISP_PPFAM21', 'share', '% prestations familiales'),
        ('DISP_PPLOGT21', 'share', '% aides logement'),
        ('DISP_PPMINI21', 'share', '% minimas sociaux'),
        ('DISP_PBEN21', 'share', '% bénéfices indépendants'),
        ('DISP_PIMPOT21', 'share', '% impôts (négatif)'),
        ('DISP_PACT21', 'share', '% revenus activité'),
    ],
    'Diplômes / Emploi': [
        ('pct_sup5', 'pct', '% BAC+5'),
        ('pct_sans_diplome', 'pct', '% sans diplôme'),
        ('pct_capbep', 'pct', '% CAP/BEP'),
        ('pct_bac_plus', 'pct', '% BAC+'),
        ('pct_chomage', 'pct', '% chômage'),
        ('pct_cdi', 'pct', '% CDI'),
        ('pct_cdd', 'pct', '% CDD'),
        ('pct_interim', 'pct', '% intérim'),
        ('pct_temps_partiel', 'pct', '% temps partiel'),
        ('pct_inactif', 'pct', '% inactifs'),
        ('pct_etudiants', 'pct', '% étudiants'),
    ],
    'Transport': [
        ('pct_actifs_voiture', 'pct', '% voiture'),
        ('pct_actifs_transports', 'pct', '% transports en commun'),
        ('pct_actifs_velo', 'pct', '% vélo'),
        ('pct_actifs_2roues', 'pct', '% 2 roues'),
        ('pct_actifs_marche', 'pct', '% marche'),
    ],
    'Logement': [
        ('pct_proprietaires', 'pct', '% propriétaires'),
        ('pct_locataires', 'pct', '% locataires'),
        ('pct_hlm', 'pct', '% HLM'),
        ('pct_logvac', 'pct', '% logements vacants'),
        ('pct_maison', 'pct', '% maisons'),
        ('pct_appart', 'pct', '% appartements'),
        ('pct_petits_logements', 'pct', '% petits logements'),
        ('pct_grands_logements', 'pct', '% grands logements'),
        ('pct_logements_anciens', 'pct', '% logements anciens'),
        ('pct_logements_recents', 'pct', '% logements récents'),
        ('pct_voiture_0', 'pct', '% 0 voiture'),
        ('pct_voiture_2plus', 'pct', '% 2+ voitures'),
        ('surface_moyenne', 'count', 'Surface moy. m²'),
        ('pct_suroccupation', 'pct', '% suroccupation'),
    ],
    'Chauffage': [
        ('pct_chauffage_elec', 'pct', '% chauffage élec.'),
        ('pct_chauffage_fioul', 'pct', '% chauffage fioul'),
        ('pct_chauffage_gaz_ville', 'pct', '% chauffage gaz ville'),
        ('pct_chauffage_gaz_bouteille', 'pct', '% chauffage gaz bout.'),
        ('pct_chauffage_autre', 'pct', '% chauffage autre'),
    ],
    'Confort logement': [
        ('pct_garage', 'pct', '% garage'),
        ('nb_pieces_moyen', 'count', 'Nb pièces moy.'),
        ('pct_studios', 'pct', '% studios'),
        ('pct_logements_5p_plus', 'pct', '% 5+ pièces'),
    ],
    'Équipements BPE': [
        ('bpe_total_pour1000', 'rate', 'Équip. totaux /1000'),
        ('bpe_A_services_pour1000', 'rate', 'Services /1000'),
        ('bpe_B_commerces_pour1000', 'rate', 'Commerces /1000'),
        ('bpe_C_enseignement_pour1000', 'rate', 'Enseignement /1000'),
        ('bpe_D_sante_pour1000', 'rate', 'Santé /1000'),
        ('bpe_E_transports_pour1000', 'rate', 'Transports /1000'),
        ('bpe_F_sports_culture_pour1000', 'rate', 'Sports/culture /1000'),
        ('bpe_G_tourisme_pour1000', 'rate', 'Tourisme /1000'),
        ('bpe_educ_prioritaire_pour1000', 'rate', 'Éduc. prioritaire /1000'),
        ('bpe_ecole_privee_pour1000', 'rate', 'Écoles privées /1000'),
        ('bpe_sport_indoor_pour1000', 'rate', 'Sport indoor /1000'),
        ('pct_sport_accessible', 'pct', '% sport accessible'),
    ],
}

# Compositional groups
COMPOSITIONAL_GROUPS = {
    'CSP': ['pct_csp_agriculteur', 'pct_csp_independant', 'pct_csp_plus',
            'pct_csp_intermediaire', 'pct_csp_employe', 'pct_csp_ouvrier',
            'pct_csp_retraite', 'pct_csp_sans_emploi'],
    'Transport': ['pct_actifs_voiture', 'pct_actifs_transports',
                  'pct_actifs_velo', 'pct_actifs_2roues', 'pct_actifs_marche'],
    'Chauffage': ['pct_chauffage_elec', 'pct_chauffage_fioul',
                  'pct_chauffage_gaz_ville', 'pct_chauffage_gaz_bouteille',
                  'pct_chauffage_autre'],
}

# Existing scores config (from rebuild_vizu_iris.py)
EXISTING_SCORES_CONFIG = {
    'score_exploitation': {
        'pos_vars': ['DISP_PPAT21', 'P21_NSAL15P_EMPLOY', 'pct_csp_plus',
                     'pct_csp_retraite', 'DISP_MED21', 'DISP_PPEN21', 'DISP_PBEN21'],
        'neg_vars': ['DISP_TP6021', 'DISP_PTSA21', 'P21_NSAL15P_AIDFAM', 'DISP_PPLOGT21'],
    },
    'score_domination': {
        'pos_vars': ['pct_csp_plus', 'pct_csp_intermediaire', 'pct_sup5',
                     'pct_cdi', 'P21_NSAL15P_EMPLOY'],
        'neg_vars': ['pct_csp_ouvrier', 'pct_csp_sans_emploi', 'pct_csp_employe',
                     'pct_csp_independant', 'pct_sans_diplome', 'pct_capbep', 'pct_cdd',
                     'pct_interim', 'pct_temps_partiel', 'pct_chomage', 'DISP_TP6021',
                     'DISP_PPSOC21', 'DISP_PPMINI21'],
    },
    'score_cap_cult': {
        'pos_vars': ['pct_csp_plus', 'pct_csp_intermediaire', 'pct_sup5', 'pct_actifs_velo'],
        'neg_vars': ['pct_csp_ouvrier', 'pct_sans_diplome', 'pct_csp_sans_emploi',
                     'pct_capbep', 'pct_interim', 'pct_temps_partiel', 'pct_chomage'],
    },
    'score_cap_eco': {
        'pos_vars': ['DISP_PPAT21', 'P21_NSAL15P_EMPLOY', 'pct_csp_plus',
                     'pct_csp_retraite', 'DISP_MED21', 'DISP_PPEN21', 'DISP_PBEN21'],
        'neg_vars': ['DISP_TP6021', 'DISP_PTSA21', 'P21_NSAL15P_AIDFAM',
                     'DISP_PPLOGT21', 'DISP_PCHO21'],
    },
    'score_precarite': {
        'pos_vars': ['DISP_TP6021', 'pct_csp_sans_emploi', 'DISP_PPSOC21',
                     'DISP_PPMINI21', 'pct_chomage'],
        'neg_vars': ['DISP_MED21', 'DISP_PPAT21'],
    },
    'score_rentier': {
        'pos_vars': ['DISP_PPAT21', 'DISP_PPEN21', 'pct_csp_retraite'],
        'neg_vars': ['DISP_PACT21', 'DISP_PPSOC21', 'pct_csp_employe'],
    },
    'score_ruralite': {
        'pos_vars': ['pct_csp_agriculteur', 'pct_sans_diplome', 'pct_actifs_voiture',
                     'P21_ACTOCC15P_ILT3'],
        'neg_vars': ['pct_immigres', 'pct_actifs_velo', 'pct_actifs_transports',
                     'pct_actifs_marche', 'pct_etudiants', 'P21_ACTOCC15P_ILT1'],
    },
    'score_urbanite': {
        'pos_vars': ['pct_appart', 'pct_locataires', 'pct_petits_logements',
                     'pct_voiture_0', 'pct_chauffage_gaz_ville',
                     'bpe_E_transports_pour1000', 'bpe_total_pour1000'],
        'neg_vars': ['pct_maison', 'pct_voiture_2plus', 'pct_chauffage_fioul',
                     'pct_grands_logements', 'surface_moyenne', 'pct_garage'],
    },
    'score_confort_residentiel': {
        'pos_vars': ['pct_proprietaires', 'pct_grands_logements', 'surface_moyenne',
                     'pct_garage', 'nb_pieces_moyen', 'pct_logements_5p_plus'],
        'neg_vars': ['pct_suroccupation', 'pct_petits_logements', 'pct_hlm',
                     'pct_logvac', 'pct_studios'],
    },
    'score_equipement_public': {
        'pos_vars': ['bpe_total_pour1000', 'bpe_D_sante_pour1000',
                     'bpe_C_enseignement_pour1000', 'bpe_F_sports_culture_pour1000',
                     'bpe_A_services_pour1000', 'bpe_B_commerces_pour1000'],
        'neg_vars': [],
    },
}


# ═══════════════════════════════════════════════════════════════════════════════
#  UTILITY FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def get_all_var_names():
    """Flat list of all variable names from inventory."""
    return [v[0] for cat in VAR_INVENTORY.values() for v in cat]

def get_var_labels():
    """Dict var_name -> label."""
    return {v[0]: v[2] for cat in VAR_INVENTORY.values() for v in cat}

def get_var_types():
    """Dict var_name -> type."""
    return {v[0]: v[1] for cat in VAR_INVENTORY.values() for v in cat}

def get_var_category():
    """Dict var_name -> category."""
    return {v[0]: cat_name for cat_name, vars_list in VAR_INVENTORY.items() for v in vars_list}

def weighted_mean(x, w):
    """Population-weighted mean, ignoring NaN."""
    mask = np.isfinite(x) & np.isfinite(w) & (w > 0)
    if mask.sum() == 0:
        return np.nan
    return np.average(x[mask], weights=w[mask])

def weighted_std(x, w):
    """Population-weighted standard deviation."""
    mask = np.isfinite(x) & np.isfinite(w) & (w > 0)
    if mask.sum() < 2:
        return np.nan
    wm = np.average(x[mask], weights=w[mask])
    return np.sqrt(np.average((x[mask] - wm) ** 2, weights=w[mask]))

def weighted_corr(X, w):
    """Population-weighted Pearson correlation matrix.
    X: (n, p) array, w: (n,) weights."""
    mask = np.all(np.isfinite(X), axis=1) & np.isfinite(w) & (w > 0)
    X_m = X[mask]
    w_m = w[mask]
    w_norm = w_m / w_m.sum()
    means = (X_m * w_norm[:, None]).sum(axis=0)
    X_c = X_m - means
    cov = (X_c * w_norm[:, None]).T @ X_c
    std = np.sqrt(np.diag(cov))
    std[std == 0] = 1e-10
    corr = cov / np.outer(std, std)
    np.fill_diagonal(corr, 1.0)
    return corr

def weighted_spearman(X, w):
    """Population-weighted Spearman correlation (rank then weighted Pearson)."""
    n, p = X.shape
    X_ranked = np.empty_like(X)
    for j in range(p):
        col = X[:, j]
        valid = np.isfinite(col)
        ranks = np.full(n, np.nan)
        ranks[valid] = stats.rankdata(col[valid])
        X_ranked[:, j] = ranks
    return weighted_corr(X_ranked, w)

def _rang_pondere(series, pop):
    """Centile pondéré par population, centré à 0 (range ≈ -50 à +50).
    Reproduced from rebuild_vizu_iris.py."""
    s = series.copy().astype(float)
    p = pop.copy().astype(float)
    valid = s.notna() & p.notna() & (p > 0)
    if valid.sum() < 10:
        return pd.Series(0.0, index=s.index)
    s_v = s[valid]
    p_v = p[valid]
    order = s_v.argsort()
    p_sorted = p_v.iloc[order]
    cumsum = p_sorted.cumsum()
    total = p_sorted.sum()
    centile = (cumsum - p_sorted / 2) / total * 100
    result = pd.Series(np.nan, index=s.index)
    orig_positions = np.where(valid.values)[0][order.values]
    result.iloc[orig_positions] = centile.values
    return result.fillna(50.0) - 50.0

def make_score(df, pop, pos_vars, neg_vars):
    """Score composite par rang centile pondéré par population."""
    parts = []
    for v in pos_vars:
        if v not in df.columns:
            continue
        parts.append(_rang_pondere(df[v], pop))
    for v in neg_vars:
        if v not in df.columns:
            continue
        parts.append(-_rang_pondere(df[v], pop))
    if not parts:
        return pd.Series(0.0, index=df.index)
    return pd.concat(parts, axis=1).mean(axis=1)


# ═══════════════════════════════════════════════════════════════════════════════
#  DATA LOADING & PREPARATION
# ═══════════════════════════════════════════════════════════════════════════════

def load_and_prepare():
    """Load socio + election data, compute derived variables, return clean dataset."""
    print("\n── Chargement des données ──")
    df = pd.read_csv(SOCIO_CSV, dtype={'IRIS': str, 'COM': str}, low_memory=False)
    print(f"  Socio : {df.shape[0]} IRIS × {df.shape[1]} colonnes")

    # Population
    df['_pop'] = df['pop_totale'].fillna(df['pop_totale'].median())

    # ── Derived variables (replicated from rebuild_vizu_iris.py lines 810-829) ──
    _nscol = df['P21_NSCOL15P'].replace(0, np.nan)
    df['pct_sup5'] = df['P21_NSCOL15P_SUP5'] / _nscol * 100
    df['pct_sans_diplome'] = df['P21_NSCOL15P_DIPLMIN'] / _nscol * 100
    df['pct_bac_plus'] = (df['P21_NSCOL15P_SUP2'] + df['P21_NSCOL15P_SUP34'] + df['P21_NSCOL15P_SUP5']) / _nscol * 100
    df['pct_capbep'] = df['P21_NSCOL15P_CAPBEP'] / _nscol * 100
    df['pct_chomage'] = df['P21_CHOM1564'] / df['P21_ACT1564'].replace(0, np.nan) * 100
    df['pct_inactif'] = df['P21_INACT1564'] / df['P21_POP1564'].replace(0, np.nan) * 100
    df['pct_etudiants'] = df['P21_ETUD1564'] / df['P21_POP1564'].replace(0, np.nan) * 100

    _sal = df['P21_SAL15P'].replace(0, np.nan)
    df['pct_cdi'] = df['P21_SAL15P_CDI'] / _sal * 100
    df['pct_cdd'] = df['P21_SAL15P_CDD'] / _sal * 100
    df['pct_interim'] = df['P21_SAL15P_INTERIM'] / _sal * 100
    df['pct_temps_partiel'] = df['P21_SAL15P_TP'] / _sal * 100

    _actocc = df['P21_ACTOCC15P'].replace(0, np.nan)
    df['pct_actifs_voiture'] = df['C21_ACTOCC15P_VOIT'] / _actocc * 100
    df['pct_actifs_transports'] = df['C21_ACTOCC15P_TCOM'] / _actocc * 100
    df['pct_actifs_velo'] = df['C21_ACTOCC15P_VELO'] / _actocc * 100
    df['pct_actifs_2roues'] = df['C21_ACTOCC15P_2ROUESMOT'] / _actocc * 100
    df['pct_actifs_marche'] = df['C21_ACTOCC15P_MAR'] / _actocc * 100

    # ── Existing composite scores ──
    pop = df['_pop']
    for score_name, cfg in EXISTING_SCORES_CONFIG.items():
        df[score_name] = make_score(df, pop, cfg['pos_vars'], cfg['neg_vars'])

    # ── Election data ──
    elec = pd.read_csv(ELECTION_CSV, dtype={'CODE_IRIS': str})
    print(f"  Élection 2022 T1 : {elec.shape[0]} IRIS × {elec.shape[1]} colonnes")

    score_cols = [f'score_{c}' for c in CANDIDATES if f'score_{c}' in elec.columns]
    extra_cols = [c for c in ['pct_abstention', 'inscrits', 'exprimes', 'blancs', 'nuls']
                  if c in elec.columns]

    elec_merged = df[['IRIS']].merge(
        elec.set_index('CODE_IRIS')[score_cols + extra_cols],
        left_on='IRIS', right_index=True, how='left'
    )
    for sc in score_cols:
        df[sc] = elec_merged[sc].values
    for ec in extra_cols:
        df[ec] = elec_merged[ec].values

    # Dominant candidate
    score_matrix = df[[f'score_{c}' for c in CANDIDATES if f'score_{c}' in df.columns]].fillna(0)
    cand_names = [c.replace('score_', '') for c in score_matrix.columns]
    df['dominant_candidate'] = score_matrix.values.argmax(axis=1)
    df['dominant_candidate'] = df['dominant_candidate'].map(lambda i: cand_names[i])
    df.loc[score_matrix.sum(axis=1) == 0, 'dominant_candidate'] = None

    # ── Filter: keep IRIS with pop >= 50 and election data ──
    mask = (df['_pop'] >= 50) & (df['exprimes'].notna()) & (df['exprimes'] > 0)
    df_clean = df[mask].copy().reset_index(drop=True)
    print(f"  Après filtrage (pop≥50 + élection) : {df_clean.shape[0]} IRIS")

    # ── Extract analysis variables ──
    all_vars = get_all_var_names()
    available = [v for v in all_vars if v in df_clean.columns]
    missing = [v for v in all_vars if v not in df_clean.columns]
    if missing:
        print(f"  [!]  Variables manquantes : {missing}")

    # NaN report per variable
    nan_counts = df_clean[available].isna().sum()
    high_nan = nan_counts[nan_counts > len(df_clean) * 0.3]
    if len(high_nan) > 0:
        print(f"  [!]  Variables > 30% NaN (exclues de la PCA) : {list(high_nan.index)}")
    vars_for_pca = [v for v in available if nan_counts[v] <= len(df_clean) * 0.3]

    print(f"  Variables retenues pour l'analyse : {len(vars_for_pca)}")

    return df_clean, vars_for_pca, available


# ═══════════════════════════════════════════════════════════════════════════════
#  PART 1: PARTY-AGNOSTIC ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════

# ── 1.1 Variable Quality ─────────────────────────────────────────────────────

def variable_quality_assessment(df, var_names):
    """Compute descriptive stats, flag outliers and low-variance variables."""
    print("\n" + "─" * 70)
    print("  1.1  QUALITÉ DES VARIABLES")
    print("─" * 70)

    pop = df['_pop'].values
    labels = get_var_labels()
    vtypes = get_var_types()
    cats = get_var_category()

    rows = []
    for v in var_names:
        x = df[v].values.astype(float)
        valid = np.isfinite(x)
        n_valid = valid.sum()
        n_nan = (~valid).sum()
        pct_nan = n_nan / len(x) * 100

        if n_valid < 10:
            rows.append({'variable': v, 'label': labels.get(v, v), 'type': vtypes.get(v, '?'),
                         'category': cats.get(v, '?'), 'n_valid': n_valid, 'pct_nan': pct_nan,
                         'wmean': np.nan, 'wstd': np.nan, 'min': np.nan, 'max': np.nan,
                         'p1': np.nan, 'p5': np.nan, 'p25': np.nan, 'p50': np.nan,
                         'p75': np.nan, 'p95': np.nan, 'p99': np.nan,
                         'skew': np.nan, 'kurtosis': np.nan, 'wcv': np.nan,
                         'flag_outlier': False, 'flag_low_var': False})
            continue

        xv = x[valid]
        wm = weighted_mean(x, pop)
        ws = weighted_std(x, pop)
        wcv = ws / abs(wm) if abs(wm) > 1e-10 else np.nan
        sk = stats.skew(xv)
        ku = stats.kurtosis(xv)

        rows.append({
            'variable': v, 'label': labels.get(v, v), 'type': vtypes.get(v, '?'),
            'category': cats.get(v, '?'),
            'n_valid': n_valid, 'pct_nan': round(pct_nan, 1),
            'wmean': round(wm, 3), 'wstd': round(ws, 3),
            'min': round(np.nanmin(xv), 3), 'max': round(np.nanmax(xv), 3),
            'p1': round(np.percentile(xv, 1), 3), 'p5': round(np.percentile(xv, 5), 3),
            'p25': round(np.percentile(xv, 25), 3), 'p50': round(np.percentile(xv, 50), 3),
            'p75': round(np.percentile(xv, 75), 3), 'p95': round(np.percentile(xv, 95), 3),
            'p99': round(np.percentile(xv, 99), 3),
            'skew': round(sk, 2), 'kurtosis': round(ku, 2),
            'wcv': round(wcv, 3) if wcv is not np.nan else np.nan,
            'flag_outlier': abs(sk) > 3 or ku > 20,
            'flag_low_var': wcv < 0.05 if np.isfinite(wcv) else False,
        })

    quality = pd.DataFrame(rows)
    quality.to_csv(os.path.join(OUTPUT_DIR, '01_variable_quality.csv'), index=False)

    # Console summary
    flagged_out = quality[quality['flag_outlier']]
    flagged_low = quality[quality['flag_low_var']]
    print(f"\n  Variables totales : {len(quality)}")
    print(f"  Variables avec outliers extrêmes (|skew|>3 ou kurtosis>20) : {len(flagged_out)}")
    if len(flagged_out) > 0:
        for _, r in flagged_out.iterrows():
            print(f"    {r['variable']:35s}  skew={r['skew']:7.2f}  kurt={r['kurtosis']:7.2f}")
    print(f"  Variables à faible variance (CV<0.05) : {len(flagged_low)}")
    if len(flagged_low) > 0:
        for _, r in flagged_low.iterrows():
            print(f"    {r['variable']:35s}  CV={r['wcv']:.4f}  mean={r['wmean']:.2f}")

    # ── Distribution plots (with KDE overlay, colored by category) ──
    cat_colors = {
        'Démographie': '#e41a1c', 'CSP': '#377eb8', 'Revenus': '#4daf4a',
        'Composition revenus': '#984ea3', 'Diplômes / Emploi': '#ff7f00',
        'Transport': '#a65628', 'Logement': '#f781bf', 'Chauffage': '#999999',
        'Confort logement': '#66c2a5', 'Équipements BPE': '#fc8d62',
    }
    n_vars = len(var_names)
    ncols = 6
    nrows = (n_vars + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(24, 3 * nrows))
    axes = axes.flatten()
    for i, v in enumerate(var_names):
        ax = axes[i]
        xv = df[v].dropna().values
        cat = cats.get(v, 'autre')
        color = cat_colors.get(cat, 'steelblue')
        ax.hist(xv, bins=50, color=color, alpha=0.5, density=True)
        # KDE overlay
        if len(xv) > 10 and np.std(xv) > 1e-10:
            try:
                from scipy.stats import gaussian_kde
                kde = gaussian_kde(xv)
                x_grid = np.linspace(xv.min(), xv.max(), 200)
                ax.plot(x_grid, kde(x_grid), color=color, lw=1.5)
            except Exception:
                pass
        ax.set_title(f'{v}\n({cat})', fontsize=6, fontweight='bold')
        ax.tick_params(labelsize=5)
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)
    fig.suptitle('Distributions de toutes les variables (KDE + histogramme, colorées par catégorie)', fontsize=14, y=1.0)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, '01_distributions.png'), dpi=120, bbox_inches='tight')
    plt.close(fig)

    # ── Skewness vs Kurtosis scatter ──
    fig, ax = plt.subplots(figsize=(10, 7))
    q = quality[quality['skew'].notna()]
    ax.scatter(q['skew'], q['kurtosis'], c='steelblue', alpha=0.7, s=40)
    for _, r in q.iterrows():
        if abs(r['skew']) > 2 or r['kurtosis'] > 10:
            ax.annotate(r['variable'], (r['skew'], r['kurtosis']),
                        fontsize=6, alpha=0.8)
    ax.axvline(0, color='gray', ls='--', lw=0.5)
    ax.axhline(3, color='red', ls='--', lw=0.5, label='kurtosis=3 (Gauss)')
    ax.set_xlabel('Skewness')
    ax.set_ylabel('Kurtosis')
    ax.set_title('Skewness vs Kurtosis des variables IRIS')
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, '01_skew_kurtosis.png'), dpi=120)
    plt.close(fig)

    print(f"  → Fichiers : 01_variable_quality.csv, 01_distributions.png, 01_skew_kurtosis.png")
    return quality


# ── 1.2 Correlation Analysis ─────────────────────────────────────────────────

def correlation_analysis(df, var_names):
    """Weighted Pearson & Spearman, redundant pairs, dendrogram."""
    print("\n" + "─" * 70)
    print("  1.2  CORRÉLATIONS")
    print("─" * 70)

    pop = df['_pop'].values
    X = df[var_names].values.astype(float)

    # Impute NaN with weighted median for correlation
    for j in range(X.shape[1]):
        col = X[:, j]
        nans = ~np.isfinite(col)
        if nans.any():
            med = np.nanmedian(col)
            col[nans] = med

    # Weighted Pearson
    corr_p = weighted_corr(X, pop)
    df_corr_p = pd.DataFrame(corr_p, index=var_names, columns=var_names)
    df_corr_p.to_csv(os.path.join(OUTPUT_DIR, '02_corr_pearson.csv'))

    # Weighted Spearman
    corr_s = weighted_spearman(X, pop)
    df_corr_s = pd.DataFrame(corr_s, index=var_names, columns=var_names)
    df_corr_s.to_csv(os.path.join(OUTPUT_DIR, '02_corr_spearman.csv'))

    # ── Redundant pairs (|r| > 0.85) ──
    redundant = []
    n = len(var_names)
    for i in range(n):
        for j in range(i + 1, n):
            r = corr_p[i, j]
            if abs(r) > 0.85:
                redundant.append({
                    'var1': var_names[i], 'var2': var_names[j],
                    'pearson_r': round(r, 3),
                    'spearman_r': round(corr_s[i, j], 3),
                })
    df_red = pd.DataFrame(redundant).sort_values('pearson_r', key=abs, ascending=False)
    df_red.to_csv(os.path.join(OUTPUT_DIR, '02_redundant_pairs.csv'), index=False)

    print(f"\n  Paires redondantes (|r| > 0.85) : {len(df_red)}")
    for _, r in df_red.head(15).iterrows():
        print(f"    {r['var1']:35s} × {r['var2']:35s}  r={r['pearson_r']:+.3f}")

    # ── Non-linear pairs (|Spearman - Pearson| > 0.15) ──
    nonlin = []
    for i in range(n):
        for j in range(i + 1, n):
            diff = abs(corr_s[i, j] - corr_p[i, j])
            if diff > 0.15:
                nonlin.append({
                    'var1': var_names[i], 'var2': var_names[j],
                    'pearson_r': round(corr_p[i, j], 3),
                    'spearman_r': round(corr_s[i, j], 3),
                    'diff': round(diff, 3),
                })
    df_nl = pd.DataFrame(nonlin).sort_values('diff', ascending=False)
    df_nl.to_csv(os.path.join(OUTPUT_DIR, '02_nonlinear_pairs.csv'), index=False)
    print(f"  Paires non-linéaires (|ρ_s - r_p| > 0.15) : {len(df_nl)}")
    for _, r in df_nl.head(10).iterrows():
        print(f"    {r['var1']:35s} × {r['var2']:35s}  Δ={r['diff']:+.3f} (P={r['pearson_r']:+.3f} S={r['spearman_r']:+.3f})")

    # ── Dendrogram (compute linkage first, needed for heatmap ordering) ──
    dist = 1 - np.abs(corr_p)
    np.fill_diagonal(dist, 0)
    dist = np.clip(dist, 0, None)
    dist_condensed = squareform(dist, checks=False)
    Z = linkage(dist_condensed, method='ward')

    fig, ax = plt.subplots(figsize=(20, 10))
    dendrogram(Z, labels=var_names, ax=ax, leaf_rotation=90, leaf_font_size=6,
               color_threshold=0.7)
    ax.set_title('Clustering hiérarchique des variables (1 - |r|, Ward)', fontsize=12)
    ax.set_ylabel('Distance')
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, '02_dendrogram.png'), dpi=120, bbox_inches='tight')
    plt.close(fig)

    # ── Heatmap (reordered by hierarchical clustering) ──
    fig, ax = plt.subplots(figsize=(20, 18))
    from scipy.cluster.hierarchy import leaves_list
    leaf_order = leaves_list(Z)
    ordered_vars = [var_names[i] for i in leaf_order]
    corr_ordered = df_corr_p.loc[ordered_vars, ordered_vars]
    n_ov = len(ordered_vars)
    mask_ordered = np.triu(np.ones((n_ov, n_ov), dtype=bool), k=1)
    sns.heatmap(corr_ordered, mask=mask_ordered, cmap='RdBu_r', center=0, vmin=-1, vmax=1,
                ax=ax, square=True, linewidths=0.1,
                xticklabels=True, yticklabels=True,
                cbar_kws={'shrink': 0.6})
    ax.tick_params(labelsize=5)
    ax.set_title('Matrice de corrélation Pearson pondérée — ordonnée par clustering hiérarchique', fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, '02_corr_heatmap.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)

    print(f"  → Fichiers : 02_corr_pearson.csv, 02_corr_spearman.csv, 02_redundant_pairs.csv,")
    print(f"               02_nonlinear_pairs.csv, 02_corr_heatmap.png, 02_dendrogram.png")

    return df_corr_p, df_red, Z, corr_p


# ── 1.3 Weighted PCA ─────────────────────────────────────────────────────────

def weighted_pca_analysis(df, var_names):
    """Population-weighted PCA via eigendecomposition."""
    print("\n" + "─" * 70)
    print("  1.3  PCA PONDÉRÉE PAR POPULATION")
    print("─" * 70)

    pop = df['_pop'].values.astype(float)
    X = df[var_names].values.astype(float)

    # Impute NaN with weighted median
    for j in range(X.shape[1]):
        col = X[:, j]
        nans = ~np.isfinite(col)
        if nans.any():
            valid_mask = np.isfinite(col) & (pop > 0)
            if valid_mask.sum() > 0:
                med = np.median(col[valid_mask])
            else:
                med = 0.0
            col[nans] = med

    # Weighted standardization
    w_norm = pop / pop.sum()
    w_means = (X * w_norm[:, None]).sum(axis=0)
    X_c = X - w_means
    w_stds = np.sqrt((X_c ** 2 * w_norm[:, None]).sum(axis=0))
    w_stds[w_stds < 1e-10] = 1e-10
    X_std = X_c / w_stds

    # Weighted covariance (= correlation since standardized)
    C = (X_std * w_norm[:, None]).T @ X_std

    # Eigendecomposition
    eigenvalues, eigenvectors = np.linalg.eigh(C)
    # Sort descending
    idx = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    # Clip small negative eigenvalues (numerical)
    eigenvalues = np.maximum(eigenvalues, 0)

    total_var = eigenvalues.sum()
    explained_ratio = eigenvalues / total_var
    cumulative = np.cumsum(explained_ratio)

    # Kaiser criterion
    n_kaiser = (eigenvalues > 1).sum()

    # Broken stick
    p = len(var_names)
    broken_stick = np.array([sum(1.0 / k for k in range(j, p + 1)) / p for j in range(1, p + 1)])
    n_broken = (explained_ratio > broken_stick).sum()

    n_retain = max(n_kaiser, n_broken, 2)
    n_retain = min(n_retain, 15)  # Cap

    print(f"\n  Dimensions : {p}")
    print(f"  Kaiser (eigenvalue > 1) : {n_kaiser} composantes")
    print(f"  Broken stick : {n_broken} composantes")
    print(f"  Retenues : {n_retain}")
    print(f"\n  {'PC':>5s}  {'Eigenval':>10s}  {'%Var':>7s}  {'Cumul%':>7s}  {'B.Stick':>7s}")
    print(f"  {'─'*5}  {'─'*10}  {'─'*7}  {'─'*7}  {'─'*7}")
    for i in range(min(n_retain + 3, p)):
        marker = " <--" if i < n_retain else ""
        print(f"  PC{i+1:>2d}  {eigenvalues[i]:10.3f}  {explained_ratio[i]*100:6.2f}%  {cumulative[i]*100:6.2f}%  {broken_stick[i]*100:6.2f}%{marker}")

    # Project data
    scores = X_std @ eigenvectors

    # Loadings (eigenvectors)
    loadings = pd.DataFrame(eigenvectors[:, :n_retain],
                            index=var_names,
                            columns=[f'PC{i+1}' for i in range(n_retain)])

    # ── Print top variables per PC ──
    print(f"\n  Top loadings par composante (top 10 par |loading|) :")
    for i in range(min(n_retain, 8)):
        pc = f'PC{i+1}'
        L = loadings[pc].abs().sort_values(ascending=False)
        top = L.head(10)
        print(f"\n  {pc} ({explained_ratio[i]*100:.1f}% variance) :")
        for v in top.index:
            val = loadings.loc[v, pc]
            sign = "+" if val > 0 else "-"
            print(f"    {sign} {v:35s}  {val:+.3f}")

    loadings.to_csv(os.path.join(OUTPUT_DIR, '03_pca_loadings.csv'))

    # Scores CSV (first n_retain PCs)
    df_scores = pd.DataFrame(scores[:, :n_retain],
                             columns=[f'PC{i+1}' for i in range(n_retain)])
    df_scores.to_csv(os.path.join(OUTPUT_DIR, '03_pca_scores.csv'), index=False)

    # ── Scree plot ──
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Scree
    x_range = range(1, min(25, p) + 1)
    ax1.bar(x_range, explained_ratio[:len(x_range)] * 100, color='steelblue', alpha=0.7)
    ax1.plot(x_range, broken_stick[:len(x_range)] * 100, 'r--', label='Broken stick', lw=1.5)
    ax1.axhline(100 / p, color='green', ls=':', label='1/p (Kaiser ≈ eigenval=1)', lw=1)
    ax1.axvline(n_retain + 0.5, color='orange', ls='--', label=f'Retenues: {n_retain}', lw=1.5)
    ax1.set_xlabel('Composante')
    ax1.set_ylabel('% Variance expliquée')
    ax1.set_title('Scree plot')
    ax1.legend(fontsize=8)

    # Cumulative
    ax2.plot(range(1, p + 1), cumulative * 100, 'o-', markersize=2, color='steelblue')
    ax2.axhline(80, color='red', ls='--', label='80%', lw=1)
    ax2.axhline(90, color='orange', ls='--', label='90%', lw=1)
    ax2.axvline(n_retain, color='green', ls='--', label=f'{n_retain} PCs: {cumulative[n_retain-1]*100:.1f}%')
    ax2.set_xlabel('Nombre de composantes')
    ax2.set_ylabel('Variance cumulée %')
    ax2.set_title('Variance cumulée')
    ax2.legend(fontsize=8)

    fig.suptitle('PCA pondérée par population — Scree plot', fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, '03_pca_scree.png'), dpi=120)
    plt.close(fig)

    # ── Biplot PC1 vs PC2 (colored by dominant candidate) ──
    fig, ax = plt.subplots(figsize=(14, 12))

    # Subsample for scatter
    n_plot = min(5000, len(df))
    rng = np.random.RandomState(RANDOM_SEED)
    idx_plot = rng.choice(len(df), n_plot, replace=False)

    # Color by dominant candidate
    cands_plot = df['dominant_candidate'].iloc[idx_plot].values
    for cand in CANDIDATES:
        mask_c = cands_plot == cand
        if mask_c.sum() > 0:
            ax.scatter(scores[idx_plot[mask_c], 0], scores[idx_plot[mask_c], 1],
                       c=CANDIDATE_COLORS.get(cand, '#ccc'), s=3, alpha=0.4,
                       label=cand, rasterized=True)
    ax.legend(fontsize=7, markerscale=3, loc='upper right')

    # Loading arrows
    scale = max(abs(scores[:, 0]).max(), abs(scores[:, 1]).max()) * 0.8
    for j, v in enumerate(var_names):
        lx, ly = eigenvectors[j, 0], eigenvectors[j, 1]
        magnitude = np.sqrt(lx**2 + ly**2)
        if magnitude > 0.2:  # Only plot significant loadings
            ax.annotate('', xy=(lx * scale, ly * scale), xytext=(0, 0),
                        arrowprops=dict(arrowstyle='->', color='red', lw=1.2))
            ax.text(lx * scale * 1.05, ly * scale * 1.05, v,
                    fontsize=6, color='red', ha='center', va='center')

    ax.axhline(0, color='gray', lw=0.3)
    ax.axvline(0, color='gray', lw=0.3)
    ax.set_xlabel(f'PC1 ({explained_ratio[0]*100:.1f}%)')
    ax.set_ylabel(f'PC2 ({explained_ratio[1]*100:.1f}%)')
    ax.set_title('Biplot PCA pondérée — PC1 × PC2 (coloré par candidat dominant)')
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, '03_pca_biplot.png'), dpi=120)
    plt.close(fig)

    print(f"\n  → Fichiers : 03_pca_scree.png, 03_pca_loadings.csv, 03_pca_scores.csv, 03_pca_biplot.png")

    return {
        'eigenvalues': eigenvalues,
        'eigenvectors': eigenvectors,
        'explained_ratio': explained_ratio,
        'loadings': loadings,
        'scores': scores,
        'n_retain': n_retain,
        'X_std': X_std,
        'w_means': w_means,
        'w_stds': w_stds,
    }


# ── 1.4 Composite Score Construction ─────────────────────────────────────────

def construct_composite_scores(pca_results, df, var_names):
    """Use PCA loadings to propose data-driven composite scores."""
    print("\n" + "─" * 70)
    print("  1.4  CONSTRUCTION DE SCORES COMPOSITES")
    print("─" * 70)

    loadings = pca_results['loadings']
    n_retain = pca_results['n_retain']
    pop = df['_pop']
    cats = get_var_category()

    # For each retained PC, extract high-loading variables and group by sign
    proposed = {}
    for i in range(min(n_retain, 8)):
        pc = f'PC{i+1}'
        L = loadings[pc]
        # Use top variables by absolute loading (top 30% quantile or at least 0.15)
        threshold = max(L.abs().quantile(0.70), 0.15)
        sig = L[L.abs() > threshold].sort_values()

        pos_vars = list(sig[sig > 0].index)
        neg_vars = list(sig[sig < 0].index)

        if len(pos_vars) + len(neg_vars) < 3:
            continue

        # Name based on dominant categories
        all_vars_pc = pos_vars + neg_vars
        cat_counts = pd.Series([cats.get(v, 'autre') for v in all_vars_pc]).value_counts()
        dominant_cats = list(cat_counts.head(2).index)
        name = f"score_pca_{pc.lower()}_{'_'.join(c.split()[0].lower() for c in dominant_cats)}"

        proposed[name] = {
            'pos_vars': pos_vars,
            'neg_vars': neg_vars,
            'pc_origin': pc,
            'explained_var': float(pca_results['explained_ratio'][i]),
        }

    # Compute each proposed score
    scores_df = pd.DataFrame(index=df.index)
    for name, cfg in proposed.items():
        scores_df[name] = make_score(df, pop, cfg['pos_vars'], cfg['neg_vars'])

    # Also compute existing scores for comparison
    existing_names = list(EXISTING_SCORES_CONFIG.keys())
    for sn in existing_names:
        if sn in df.columns:
            scores_df[sn] = df[sn]

    # Correlation matrix of all scores
    all_score_names = list(proposed.keys()) + existing_names
    available_scores = [s for s in all_score_names if s in scores_df.columns]
    corr_scores = scores_df[available_scores].corr()
    corr_scores.to_csv(os.path.join(OUTPUT_DIR, '04_score_correlations.csv'))

    # Evaluate each proposed score
    eval_rows = []
    for name in proposed:
        if name not in scores_df.columns:
            continue
        s = scores_df[name]
        # Max |r| with existing scores
        max_r_existing = 0
        max_r_name = ''
        for en in existing_names:
            if en in scores_df.columns:
                r_val = abs(scores_df[name].corr(scores_df[en]))
                if r_val > max_r_existing:
                    max_r_existing = r_val
                    max_r_name = en

        w_var = weighted_std(s.values, pop.values) ** 2

        eval_rows.append({
            'score': name,
            'pc_origin': proposed[name]['pc_origin'],
            'pct_var_explained': round(proposed[name]['explained_var'] * 100, 2),
            'n_pos_vars': len(proposed[name]['pos_vars']),
            'n_neg_vars': len(proposed[name]['neg_vars']),
            'weighted_variance': round(w_var, 2),
            'score_range': f"[{s.min():.1f}, {s.max():.1f}]",
            'max_r_with_existing': round(max_r_existing, 3),
            'most_correlated_with': max_r_name,
            'keep': max_r_existing < 0.5,
        })

    df_eval = pd.DataFrame(eval_rows)
    df_eval.to_csv(os.path.join(OUTPUT_DIR, '04_score_evaluation.csv'), index=False)

    # Print results
    print(f"\n  Scores proposés basés sur PCA :")
    print(f"  {'Score':40s}  {'PC':>5s}  {'%Var':>6s}  {'max|r|':>7s}  {'vs':20s}  {'Garder':>7s}")
    print(f"  {'─'*40}  {'─'*5}  {'─'*6}  {'─'*7}  {'─'*20}  {'─'*7}")
    for _, r in df_eval.iterrows():
        keep = "OUI" if r['keep'] else "NON"
        print(f"  {r['score']:40s}  {r['pc_origin']:>5s}  {r['pct_var_explained']:5.1f}%  {r['max_r_with_existing']:+.3f}  {r['most_correlated_with']:20s}  {keep:>7s}")

    kept = df_eval[df_eval['keep']]
    print(f"\n  Scores retenus (max|r| < 0.5 avec existants) : {len(kept)}")
    for _, r in kept.iterrows():
        name = r['score']
        cfg = proposed[name]
        print(f"\n    {name}:")
        print(f"      pos: {cfg['pos_vars']}")
        print(f"      neg: {cfg['neg_vars']}")

    # Save proposed scores definition
    proposed_data = []
    for name, cfg in proposed.items():
        proposed_data.append({
            'score': name,
            'pos_vars': '|'.join(cfg['pos_vars']),
            'neg_vars': '|'.join(cfg['neg_vars']),
            'pc_origin': cfg['pc_origin'],
        })
    pd.DataFrame(proposed_data).to_csv(os.path.join(OUTPUT_DIR, '04_proposed_scores.csv'), index=False)

    print(f"\n  → Fichiers : 04_proposed_scores.csv, 04_score_correlations.csv, 04_score_evaluation.csv")

    return proposed, scores_df


# ── 1.5 Non-linear Exploration ────────────────────────────────────────────────

def nonlinear_exploration(df, var_names, pca_results):
    """t-SNE and non-linear relationship analysis."""
    print("\n" + "─" * 70)
    print("  1.5  EXPLORATION NON-LINÉAIRE")
    print("─" * 70)

    pop = df['_pop'].values
    X_std = pca_results['X_std']

    # ── t-SNE on subsample ──
    n_sample = min(15000, len(df))
    rng = np.random.RandomState(RANDOM_SEED)

    # Stratified by population decile
    pop_deciles = pd.qcut(pop, 10, labels=False, duplicates='drop')
    idx_sample = []
    for d in range(10):
        d_idx = np.where(pop_deciles == d)[0]
        n_take = min(len(d_idx), n_sample // 10)
        idx_sample.extend(rng.choice(d_idx, n_take, replace=False))
    idx_sample = np.array(idx_sample)

    print(f"\n  t-SNE sur {len(idx_sample)} IRIS (sous-échantillon stratifié)...")

    # PCA pre-reduction to 20 dims
    X_sub = X_std[idx_sample]
    n_pca_dims = min(20, X_sub.shape[1])
    scores_sub = X_sub @ pca_results['eigenvectors'][:, :n_pca_dims]

    tsne = TSNE(n_components=2, perplexity=30, max_iter=1000,
                random_state=RANDOM_SEED, init='pca', learning_rate='auto')
    embed = tsne.fit_transform(scores_sub)
    print(f"  t-SNE terminé (KL divergence: {tsne.kl_divergence_:.2f})")

    # ── Plot by dominant candidate ──
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))

    # By party
    ax = axes[0]
    cands_sub = df['dominant_candidate'].iloc[idx_sample].values
    for cand in CANDIDATES:
        mask = cands_sub == cand
        if mask.sum() > 0:
            ax.scatter(embed[mask, 0], embed[mask, 1], c=CANDIDATE_COLORS.get(cand, '#ccc'),
                       s=3, alpha=0.4, label=cand, rasterized=True)
    ax.set_title('t-SNE coloré par candidat dominant (Prés. 2022 T1)')
    ax.legend(fontsize=7, markerscale=3)
    ax.set_xticks([])
    ax.set_yticks([])

    # By score_cap_cult (most interpretable axis usually)
    ax = axes[1]
    if 'score_cap_cult' in df.columns:
        color_var = df['score_cap_cult'].iloc[idx_sample].values
    else:
        color_var = df['DISP_MED21'].iloc[idx_sample].values
    sc = ax.scatter(embed[:, 0], embed[:, 1], c=color_var, cmap='RdBu_r',
                    s=3, alpha=0.4, rasterized=True)
    plt.colorbar(sc, ax=ax, shrink=0.7)
    ax.set_title('t-SNE coloré par score capital culturel')
    ax.set_xticks([])
    ax.set_yticks([])

    fig.suptitle(f't-SNE ({len(idx_sample)} IRIS, perplexity=30)', fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, '05_tsne_by_party.png'), dpi=120)
    plt.close(fig)

    # ── t-SNE colored by multiple scores ──
    score_vars = [s for s in EXISTING_SCORES_CONFIG if s in df.columns]
    n_scores = len(score_vars)
    ncols = 4
    nrows = (n_scores + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(20, 5 * nrows))
    axes = axes.flatten()

    for i, sv in enumerate(score_vars):
        ax = axes[i]
        vals = df[sv].iloc[idx_sample].values
        sc = ax.scatter(embed[:, 0], embed[:, 1], c=vals, cmap='RdBu_r',
                        s=2, alpha=0.4, rasterized=True)
        plt.colorbar(sc, ax=ax, shrink=0.6)
        ax.set_title(sv, fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle('t-SNE coloré par scores composites existants', fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, '05_tsne_by_score.png'), dpi=120)
    plt.close(fig)

    print(f"  → Fichiers : 05_tsne_by_party.png, 05_tsne_by_score.png")


# ── 1.6 Normalization Analysis ────────────────────────────────────────────────

def normalization_analysis(df, var_names):
    """Analyze and recommend normalization per variable type."""
    print("\n" + "─" * 70)
    print("  1.6  ANALYSE DE NORMALISATION")
    print("─" * 70)

    vtypes = get_var_types()
    rows = []

    for v in var_names:
        x = df[v].dropna().values
        if len(x) < 100:
            continue

        vt = vtypes.get(v, 'pct')
        raw_skew = stats.skew(x)

        # Test transformations
        transforms = {'raw': (x, raw_skew)}

        if vt in ('euro', 'rate', 'count') and np.all(x >= 0):
            x_log = np.log1p(x)
            transforms['log1p'] = (x_log, stats.skew(x_log))

        if vt == 'pct' and abs(raw_skew) > 1:
            # For highly skewed percentages, try sqrt
            x_sqrt = np.sqrt(np.clip(x, 0, None))
            transforms['sqrt'] = (x_sqrt, stats.skew(x_sqrt))

        # Find best transform
        best = min(transforms, key=lambda k: abs(transforms[k][1]))
        best_skew = transforms[best][1]

        rows.append({
            'variable': v,
            'type': vt,
            'raw_skew': round(raw_skew, 2),
            'best_transform': best,
            'post_transform_skew': round(best_skew, 2),
            'improvement': round(abs(raw_skew) - abs(best_skew), 2),
            'recommend': best if abs(raw_skew) > 1 and abs(best_skew) < abs(raw_skew) * 0.7 else 'none',
        })

    df_norm = pd.DataFrame(rows)
    df_norm.to_csv(os.path.join(OUTPUT_DIR, '06_normalization_recommendations.csv'), index=False)

    # ── Compositional analysis ──
    print(f"\n  Groupes compositionnels détectés :")
    for grp_name, grp_vars in COMPOSITIONAL_GROUPS.items():
        available = [v for v in grp_vars if v in df.columns]
        if not available:
            continue
        sums = df[available].sum(axis=1)
        print(f"    {grp_name}: {len(available)} vars, somme médiane = {sums.median():.1f}%  "
              f"(min={sums.min():.1f}, max={sums.max():.1f})")

    # Print recommendations
    recommended = df_norm[df_norm['recommend'] != 'none']
    print(f"\n  Transformations recommandées ({len(recommended)} variables) :")
    for _, r in recommended.iterrows():
        print(f"    {r['variable']:35s}  {r['type']:6s}  skew {r['raw_skew']:+.2f} → "
              f"{r['recommend']:5s} → skew {r['post_transform_skew']:+.2f}")

    not_recommended = df_norm[df_norm['recommend'] == 'none']
    n_ok = len(not_recommended[not_recommended['raw_skew'].abs() <= 1])
    print(f"\n  Variables déjà bien distribuées (|skew| ≤ 1) : {n_ok}/{len(df_norm)}")

    print(f"\n  → Fichier : 06_normalization_recommendations.csv")
    return df_norm


# ═══════════════════════════════════════════════════════════════════════════════
#  PART 2: PARTY-CONDITIONED ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════

# ── 2.1 Candidate Profiles ───────────────────────────────────────────────────

def candidate_profiles(df, var_names):
    """Vote-share-weighted profiles for each candidate."""
    print("\n" + "─" * 70)
    print("  2.1  PROFILS CANDIDATS (pondérés par score × population)")
    print("─" * 70)

    pop = df['_pop'].values
    labels = get_var_labels()

    # National weighted mean
    nat_means = {}
    for v in var_names:
        nat_means[v] = weighted_mean(df[v].values, pop)

    # Per-candidate weighted mean
    profiles = {}
    for cand in CANDIDATES:
        sc = f'score_{cand}'
        if sc not in df.columns:
            continue
        w = df[sc].fillna(0).values * pop
        if w.sum() <= 0:
            continue
        cand_means = {}
        for v in var_names:
            cand_means[v] = weighted_mean(df[v].values, w)
        profiles[cand] = cand_means

    # Also abstention profile
    if 'pct_abstention' in df.columns:
        w_abst = df['pct_abstention'].fillna(0).values * pop
        if w_abst.sum() > 0:
            abst_means = {}
            for v in var_names:
                abst_means[v] = weighted_mean(df[v].values, w_abst)
            profiles['ABSTENTION'] = abst_means

    # Build DataFrame
    df_prof = pd.DataFrame(profiles, index=var_names)
    df_prof.insert(0, 'national', pd.Series(nat_means))
    df_prof.to_csv(os.path.join(OUTPUT_DIR, '07_candidate_profiles.csv'))

    # Z-score deviations from national mean
    nat_stds = {}
    for v in var_names:
        nat_stds[v] = weighted_std(df[v].values, pop)

    df_z = pd.DataFrame(index=var_names)
    for cand in list(profiles.keys()):
        z_vals = {}
        for v in var_names:
            s = nat_stds.get(v, 1)
            if s and s > 1e-10:
                z_vals[v] = (profiles[cand][v] - nat_means[v]) / s
            else:
                z_vals[v] = 0
        df_z[cand] = pd.Series(z_vals)

    # Print top deviations per candidate
    print(f"\n  Déviations les plus marquées par candidat (z-scores) :")
    for cand in CANDIDATES:
        if cand not in df_z.columns:
            continue
        z = df_z[cand].abs().sort_values(ascending=False)
        top5 = z.head(5)
        print(f"\n  {cand}:")
        for v in top5.index:
            zval = df_z[cand][v]
            label = labels.get(v, v)
            raw_val = profiles[cand][v]
            nat_val = nat_means[v]
            print(f"    {label:35s}  z={zval:+.2f}  ({raw_val:.1f} vs national {nat_val:.1f})")

    # ── Heatmap ──
    cand_order = [c for c in CANDIDATES if c in df_z.columns]
    if 'ABSTENTION' in df_z.columns:
        cand_order.append('ABSTENTION')

    # Select top discriminating variables (highest max |z| across candidates)
    max_z = df_z[cand_order].abs().max(axis=1).sort_values(ascending=False)
    top_vars = max_z.head(40).index.tolist()

    # Reorder variables by hierarchical clustering for better readability
    z_data = df_z.loc[top_vars, cand_order].values
    if len(top_vars) > 3:
        from scipy.cluster.hierarchy import linkage as _linkage, leaves_list as _leaves_list
        from scipy.spatial.distance import pdist
        dist_vars = pdist(z_data, metric='euclidean')
        Z_vars = _linkage(dist_vars, method='ward')
        var_order = _leaves_list(Z_vars)
        top_vars_ordered = [top_vars[i] for i in var_order]
    else:
        top_vars_ordered = top_vars

    fig, ax = plt.subplots(figsize=(14, 16))
    z_plot = df_z.loc[top_vars_ordered, cand_order]
    sns.heatmap(z_plot, cmap='RdBu_r', center=0, vmin=-3, vmax=3,
                ax=ax, linewidths=0.5, annot=True, fmt='.1f', annot_kws={'size': 6})
    ax.set_title('Profils candidats — Z-scores vs moyenne nationale\n(top 40 variables, ordonnées par clustering)',
                 fontsize=11)
    ax.tick_params(labelsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, '07_candidate_profiles_heatmap.png'), dpi=120, bbox_inches='tight')
    plt.close(fig)

    print(f"\n  → Fichiers : 07_candidate_profiles.csv, 07_candidate_profiles_heatmap.png")
    return df_prof, df_z


# ── 2.2 Conditional PCA ──────────────────────────────────────────────────────

def conditional_pca(df, var_names, pca_results):
    """PCA weighted by each candidate's vote share × population."""
    print("\n" + "─" * 70)
    print("  2.2  PCA CONDITIONNELLE PAR CANDIDAT")
    print("─" * 70)

    pop = df['_pop'].values
    X = df[var_names].values.astype(float)

    # Impute NaN
    for j in range(X.shape[1]):
        col = X[:, j]
        nans = ~np.isfinite(col)
        if nans.any():
            col[nans] = np.nanmedian(col)

    # Top 5 candidates by total votes
    total_votes = {}
    for cand in CANDIDATES:
        sc = f'score_{cand}'
        if sc in df.columns:
            total_votes[cand] = (df[sc].fillna(0) * df['exprimes'].fillna(0) / 100).sum()
    top5 = sorted(total_votes, key=total_votes.get, reverse=True)[:5]
    print(f"\n  Top 5 candidats : {top5}")

    all_loadings = {}
    fig, axes = plt.subplots(1, len(top5), figsize=(6 * len(top5), 8))

    for idx, cand in enumerate(top5):
        sc = f'score_{cand}'
        w = df[sc].fillna(0).values * pop
        w = np.maximum(w, 0)
        w_sum = w.sum()
        if w_sum <= 0:
            continue
        w_norm = w / w_sum

        # Weighted standardization
        means = (X * w_norm[:, None]).sum(axis=0)
        X_c = X - means
        stds = np.sqrt((X_c ** 2 * w_norm[:, None]).sum(axis=0))
        stds[stds < 1e-10] = 1e-10
        X_std = X_c / stds

        # Weighted covariance
        C = (X_std * w_norm[:, None]).T @ X_std
        eigenvalues, eigenvectors = np.linalg.eigh(C)
        idx_sort = eigenvalues.argsort()[::-1]
        eigenvalues = np.maximum(eigenvalues[idx_sort], 0)
        eigenvectors = eigenvectors[:, idx_sort]

        # Store loadings (PC1, PC2)
        pc1_loadings = pd.Series(eigenvectors[:, 0], index=var_names, name=f'{cand}_PC1')
        pc2_loadings = pd.Series(eigenvectors[:, 1], index=var_names, name=f'{cand}_PC2')
        all_loadings[f'{cand}_PC1'] = pc1_loadings
        all_loadings[f'{cand}_PC2'] = pc2_loadings

        ev_ratio = eigenvalues / eigenvalues.sum()

        # Print top loadings
        print(f"\n  {cand} — PC1 ({ev_ratio[0]*100:.1f}%) + PC2 ({ev_ratio[1]*100:.1f}%)")
        top_pc1 = pc1_loadings.abs().sort_values(ascending=False).head(5)
        for v in top_pc1.index:
            sign = "+" if pc1_loadings[v] > 0 else "-"
            print(f"    PC1: {sign} {v:30s}  {pc1_loadings[v]:+.3f}")

        # Biplot
        ax = axes[idx]
        scores = X_std @ eigenvectors
        n_plot = min(3000, len(df))
        rng = np.random.RandomState(RANDOM_SEED)
        plot_idx = rng.choice(len(df), n_plot, replace=False, p=w_norm)
        ax.scatter(scores[plot_idx, 0], scores[plot_idx, 1],
                   c=CANDIDATE_COLORS.get(cand, '#999'), s=2, alpha=0.3, rasterized=True)

        # Arrows
        scale = np.percentile(np.abs(scores[:, :2]), 95)
        for j, v in enumerate(var_names):
            lx, ly = eigenvectors[j, 0], eigenvectors[j, 1]
            if np.sqrt(lx**2 + ly**2) > 0.25:
                ax.annotate('', xy=(lx * scale, ly * scale), xytext=(0, 0),
                            arrowprops=dict(arrowstyle='->', color='red', lw=0.8))
                ax.text(lx * scale * 1.08, ly * scale * 1.08, v, fontsize=4, color='red')

        ax.set_title(f'{cand}\nPC1={ev_ratio[0]*100:.1f}% PC2={ev_ratio[1]*100:.1f}%', fontsize=9)
        ax.set_xlabel('PC1')
        ax.set_ylabel('PC2')

    fig.suptitle('PCA conditionnelle par candidat (pondérée par score × pop)', fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, '08_conditional_pca_biplots.png'), dpi=120, bbox_inches='tight')
    plt.close(fig)

    # Save loadings
    df_loadings = pd.DataFrame(all_loadings)
    df_loadings.to_csv(os.path.join(OUTPUT_DIR, '08_conditional_pca_loadings.csv'))

    print(f"\n  → Fichiers : 08_conditional_pca_biplots.png, 08_conditional_pca_loadings.csv")
    return all_loadings


# ── 2.3 Discriminant Analysis (LDA) ──────────────────────────────────────────

def discriminant_analysis(df, var_names):
    """LDA on dominant candidate labels."""
    print("\n" + "─" * 70)
    print("  2.3  ANALYSE DISCRIMINANTE (LDA)")
    print("─" * 70)

    pop = df['_pop'].values
    X = df[var_names].values.astype(float)

    # Impute NaN
    for j in range(X.shape[1]):
        col = X[:, j]
        nans = ~np.isfinite(col)
        if nans.any():
            col[nans] = np.nanmedian(col)

    # Label = dominant candidate (with >= 25% threshold for stronger dominance)
    score_cols = [f'score_{c}' for c in CANDIDATES if f'score_{c}' in df.columns]
    cand_names = [c.replace('score_', '') for c in score_cols]
    scores = df[score_cols].fillna(0).values

    dominant_idx = scores.argmax(axis=1)
    dominant_score = scores.max(axis=1)
    labels = np.array([cand_names[i] for i in dominant_idx])
    labels[dominant_score < 25] = 'MIXED'

    # Filter: remove MIXED and very rare classes
    class_counts = pd.Series(labels).value_counts()
    valid_classes = class_counts[class_counts >= 100].index.tolist()
    if 'MIXED' in valid_classes:
        valid_classes.remove('MIXED')

    mask = np.isin(labels, valid_classes)
    X_lda = X[mask]
    y_lda = labels[mask]
    pop_lda = pop[mask]

    print(f"\n  Classes retenues (≥100 IRIS, score dominant ≥25%) :")
    for cls in valid_classes:
        n = (y_lda == cls).sum()
        pop_total = pop_lda[y_lda == cls].sum()
        print(f"    {cls:20s}  n={n:6d}  pop={pop_total/1e6:.2f}M")

    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_lda)

    # LDA with population-weighted priors
    class_pop = {cls: pop_lda[y_lda == cls].sum() for cls in valid_classes}
    total_pop = sum(class_pop.values())
    priors = np.array([class_pop[cls] / total_pop for cls in valid_classes])

    lda = LinearDiscriminantAnalysis(priors=priors)
    lda.fit(X_scaled, y_lda)

    # Transform
    X_lda_proj = lda.transform(X_scaled)
    n_components = X_lda_proj.shape[1]

    # Coefficients (loadings)
    coefs = pd.DataFrame(lda.scalings_[:, :n_components],
                         index=var_names,
                         columns=[f'LD{i+1}' for i in range(n_components)])
    coefs.to_csv(os.path.join(OUTPUT_DIR, '09_lda_loadings.csv'))

    # Print top loadings per LD
    print(f"\n  {n_components} axes discriminants :")
    ev_ratio = lda.explained_variance_ratio_ if hasattr(lda, 'explained_variance_ratio_') else None
    for i in range(min(n_components, 4)):
        ld = f'LD{i+1}'
        pct = f" ({ev_ratio[i]*100:.1f}%)" if ev_ratio is not None else ""
        print(f"\n  {ld}{pct} — top loadings :")
        top = coefs[ld].abs().sort_values(ascending=False).head(8)
        for v in top.index:
            sign = "+" if coefs.loc[v, ld] > 0 else "-"
            print(f"    {sign} {v:35s}  {coefs.loc[v, ld]:+.3f}")

    # Cross-validated accuracy
    cv_scores = cross_val_score(
        LinearDiscriminantAnalysis(priors=priors),
        X_scaled, y_lda, cv=5, scoring='accuracy'
    )
    print(f"\n  Précision cross-validée (5-fold) : {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

    # ── Scatter LD1 × LD2 (with 95% confidence ellipses) ──
    fig, ax = plt.subplots(figsize=(12, 10))
    from matplotlib.patches import Ellipse
    for cls in valid_classes:
        mask_cls = y_lda == cls
        pts = X_lda_proj[mask_cls, :2]
        ax.scatter(pts[:, 0], pts[:, 1],
                   c=CANDIDATE_COLORS.get(cls, '#999'), s=3, alpha=0.3,
                   label=cls, rasterized=True)
        # 95% confidence ellipse
        if pts.shape[0] > 5:
            mean_x, mean_y = pts[:, 0].mean(), pts[:, 1].mean()
            cov = np.cov(pts[:, 0], pts[:, 1])
            eigenvals, eigenvecs = np.linalg.eigh(cov)
            order = eigenvals.argsort()[::-1]
            eigenvals = eigenvals[order]
            eigenvecs = eigenvecs[:, order]
            angle = np.degrees(np.arctan2(eigenvecs[1, 0], eigenvecs[0, 0]))
            # 95% CI: chi2(2) = 5.991
            width, height = 2 * np.sqrt(eigenvals * 5.991)
            ell = Ellipse(xy=(mean_x, mean_y), width=width, height=height,
                          angle=angle, facecolor='none',
                          edgecolor=CANDIDATE_COLORS.get(cls, '#999'),
                          lw=1.5, ls='--', alpha=0.8)
            ax.add_patch(ell)
    ax.legend(fontsize=9, markerscale=4)
    ld1_pct = f" ({ev_ratio[0]*100:.1f}%)" if ev_ratio is not None else ""
    ld2_pct = f" ({ev_ratio[1]*100:.1f}%)" if ev_ratio is not None and len(ev_ratio) > 1 else ""
    ax.set_xlabel(f'LD1{ld1_pct}')
    ax.set_ylabel(f'LD2{ld2_pct}')
    ax.set_title(f'LDA — Présidentielles 2022 T1 (ellipses IC 95%)\n(accuracy CV: {cv_scores.mean():.1%})')
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, '09_lda_scatter.png'), dpi=120)
    plt.close(fig)

    # ── Confusion-like: class means on LD axes ──
    class_means = pd.DataFrame(index=valid_classes)
    for i in range(min(n_components, 4)):
        ld = f'LD{i+1}'
        for cls in valid_classes:
            class_means.loc[cls, ld] = X_lda_proj[y_lda == cls, i].mean()

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(class_means, cmap='RdBu_r', center=0, annot=True, fmt='.2f',
                ax=ax, linewidths=0.5)
    ax.set_title('Moyennes des classes sur les axes discriminants')
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, '09_lda_class_means.png'), dpi=120)
    plt.close(fig)

    print(f"\n  → Fichiers : 09_lda_loadings.csv, 09_lda_scatter.png, 09_lda_class_means.png")

    return {
        'lda': lda, 'coefs': coefs, 'scaler': scaler,
        'valid_classes': valid_classes, 'cv_accuracy': cv_scores.mean(),
    }


# ── 2.4 Variable Importance per Candidate ─────────────────────────────────────

def variable_importance(df, var_names):
    """Feature importance via GradientBoosting + correlations per candidate."""
    print("\n" + "─" * 70)
    print("  2.4  IMPORTANCE DES VARIABLES PAR CANDIDAT")
    print("─" * 70)

    pop = df['_pop'].values
    X = df[var_names].values.astype(float)

    # Impute NaN
    for j in range(X.shape[1]):
        col = X[:, j]
        nans = ~np.isfinite(col)
        if nans.any():
            col[nans] = np.nanmedian(col)

    # ── Correlations ──
    corr_results = {}
    for cand in CANDIDATES:
        sc = f'score_{cand}'
        if sc not in df.columns:
            continue
        y = df[sc].fillna(0).values
        corrs = []
        for j, v in enumerate(var_names):
            r, _ = stats.pearsonr(X[:, j], y)
            corrs.append(r)
        corr_results[cand] = corrs

    df_corr = pd.DataFrame(corr_results, index=var_names)
    df_corr.to_csv(os.path.join(OUTPUT_DIR, '10_correlations_by_candidate.csv'))

    # ── GradientBoosting per candidate ──
    importance_results = {}
    top5_cands = sorted(CANDIDATES, key=lambda c: df.get(f'score_{c}', pd.Series(0)).sum(),
                        reverse=True)[:6]

    for cand in top5_cands:
        sc = f'score_{cand}'
        if sc not in df.columns:
            continue
        print(f"\n  {cand} — entraînement GBM...")

        # Binary: dominant IRIS vs rest
        y_bin = (df['dominant_candidate'] == cand).astype(int).values

        if y_bin.sum() < 50:
            print(f"    Trop peu d'IRIS dominants ({y_bin.sum()}), skip")
            continue

        # Subsample for speed (max 20k)
        n = len(X)
        if n > 20000:
            rng = np.random.RandomState(RANDOM_SEED)
            idx = rng.choice(n, 20000, replace=False)
            X_sub, y_sub, w_sub = X[idx], y_bin[idx], pop[idx]
        else:
            X_sub, y_sub, w_sub = X, y_bin, pop

        gb = GradientBoostingClassifier(
            n_estimators=150, max_depth=5, learning_rate=0.1,
            subsample=0.8, random_state=RANDOM_SEED
        )
        gb.fit(X_sub, y_sub, sample_weight=w_sub)

        # Feature importance (impurity-based)
        imp = gb.feature_importances_
        importance_results[cand] = imp

        # Print top 10
        top_idx = imp.argsort()[::-1][:10]
        print(f"    Top 10 variables (importance) :")
        for k in top_idx:
            corr_val = corr_results.get(cand, [0] * len(var_names))[k]
            print(f"      {var_names[k]:35s}  imp={imp[k]:.4f}  r={corr_val:+.3f}")

    # ── Importance DataFrame ──
    df_imp = pd.DataFrame(importance_results, index=var_names)
    df_imp.to_csv(os.path.join(OUTPUT_DIR, '10_variable_importance.csv'))

    # ── Heatmap: correlations ──
    fig, ax = plt.subplots(figsize=(14, 18))

    # Select top 40 most discriminating vars
    max_abs_corr = df_corr.abs().max(axis=1).sort_values(ascending=False)
    top_vars = max_abs_corr.head(40).index.tolist()

    cand_order = [c for c in CANDIDATES if c in df_corr.columns]
    sns.heatmap(df_corr.loc[top_vars, cand_order], cmap='RdBu_r', center=0,
                vmin=-0.6, vmax=0.6, ax=ax, linewidths=0.5,
                annot=True, fmt='.2f', annot_kws={'size': 6})
    ax.set_title('Corrélations variables × candidats (Prés. 2022 T1)\n(top 40 variables)', fontsize=11)
    ax.tick_params(labelsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, '10_importance_heatmap.png'), dpi=120, bbox_inches='tight')
    plt.close(fig)

    print(f"\n  → Fichiers : 10_correlations_by_candidate.csv, 10_variable_importance.csv, 10_importance_heatmap.png")
    return df_imp, df_corr


# ── 2.5 Party-Informed Composite Scores ──────────────────────────────────────

def party_informed_scores(df, var_names, lda_results, corr_by_candidate):
    """Construct composite scores informed by electoral patterns."""
    print("\n" + "─" * 70)
    print("  2.5  SCORES COMPOSITES INFORMÉS PAR LES PARTIS")
    print("─" * 70)

    pop = df['_pop']
    coefs = lda_results['coefs']

    proposed = {}

    # ── Score from LDA axis 1: Gauche-Droite socioéconomique ──
    if 'LD1' in coefs.columns:
        ld1 = coefs['LD1']
        sig = ld1[ld1.abs() > ld1.abs().quantile(0.7)]
        pos_vars = list(sig[sig > 0].sort_values(ascending=False).head(8).index)
        neg_vars = list(sig[sig < 0].sort_values().head(8).index)
        proposed['score_lda_axe1'] = {'pos_vars': pos_vars, 'neg_vars': neg_vars,
                                       'description': 'Axe LDA 1 — Discrimination électorale principale'}

    # ── Score from LDA axis 2 ──
    if 'LD2' in coefs.columns:
        ld2 = coefs['LD2']
        sig = ld2[ld2.abs() > ld2.abs().quantile(0.7)]
        pos_vars = list(sig[sig > 0].sort_values(ascending=False).head(8).index)
        neg_vars = list(sig[sig < 0].sort_values().head(8).index)
        proposed['score_lda_axe2'] = {'pos_vars': pos_vars, 'neg_vars': neg_vars,
                                       'description': 'Axe LDA 2 — Discrimination électorale secondaire'}

    # ── Score Périphérie-Métropole (Le Pen/Zemmour vs Macron/Jadot/Hidalgo) ──
    # Variables positively correlated with Le Pen/Zemmour AND negatively with Macron/Jadot
    peripherie_pos = []
    peripherie_neg = []
    if corr_by_candidate is not None:
        for v in var_names:
            r_lepen = corr_by_candidate.loc[v, 'LE_PEN'] if 'LE_PEN' in corr_by_candidate.columns else 0
            r_zemmour = corr_by_candidate.loc[v, 'ZEMMOUR'] if 'ZEMMOUR' in corr_by_candidate.columns else 0
            r_macron = corr_by_candidate.loc[v, 'MACRON'] if 'MACRON' in corr_by_candidate.columns else 0
            r_jadot = corr_by_candidate.loc[v, 'JADOT'] if 'JADOT' in corr_by_candidate.columns else 0

            # Strong Le Pen/Zemmour signal AND anti-Macron/Jadot
            avg_right = (r_lepen + r_zemmour) / 2
            avg_metro = (r_macron + r_jadot) / 2

            if avg_right > 0.15 and avg_metro < -0.1:
                peripherie_pos.append((v, avg_right))
            elif avg_metro > 0.15 and avg_right < -0.1:
                peripherie_neg.append((v, avg_metro))

        peripherie_pos.sort(key=lambda x: x[1], reverse=True)
        peripherie_neg.sort(key=lambda x: x[1], reverse=True)

        if len(peripherie_pos) >= 3 and len(peripherie_neg) >= 3:
            proposed['score_peripherie_metropole'] = {
                'pos_vars': [v for v, _ in peripherie_pos[:10]],
                'neg_vars': [v for v, _ in peripherie_neg[:10]],
                'description': 'Périphérie (Le Pen/Zemmour) vs Métropole (Macron/Jadot)',
            }

    # ── Score Gauche-Populaire vs Droite-Bourgeoise ──
    # Variables correlated with Mélenchon vs Pécresse/Macron
    gauche_pos = []
    gauche_neg = []
    if corr_by_candidate is not None:
        for v in var_names:
            r_mel = corr_by_candidate.loc[v, 'MELENCHON'] if 'MELENCHON' in corr_by_candidate.columns else 0
            r_pec = corr_by_candidate.loc[v, 'PECRESSE'] if 'PECRESSE' in corr_by_candidate.columns else 0
            r_mac = corr_by_candidate.loc[v, 'MACRON'] if 'MACRON' in corr_by_candidate.columns else 0

            avg_mel = r_mel
            avg_droite = (r_pec + r_mac) / 2

            if avg_mel > 0.15 and avg_droite < -0.05:
                gauche_pos.append((v, avg_mel))
            elif avg_droite > 0.15 and avg_mel < -0.05:
                gauche_neg.append((v, avg_droite))

        gauche_pos.sort(key=lambda x: x[1], reverse=True)
        gauche_neg.sort(key=lambda x: x[1], reverse=True)

        if len(gauche_pos) >= 3 and len(gauche_neg) >= 3:
            proposed['score_gauche_pop_droite_bourg'] = {
                'pos_vars': [v for v, _ in gauche_pos[:10]],
                'neg_vars': [v for v, _ in gauche_neg[:10]],
                'description': 'Gauche populaire (Mélenchon) vs Droite bourgeoise (Pécresse/Macron)',
            }

    # ── Compute and evaluate ──
    scores_df = pd.DataFrame(index=df.index)
    for name, cfg in proposed.items():
        scores_df[name] = make_score(df, pop, cfg['pos_vars'], cfg['neg_vars'])

    # Add existing scores for correlation comparison
    existing_names = list(EXISTING_SCORES_CONFIG.keys())
    for sn in existing_names:
        if sn in df.columns:
            scores_df[sn] = df[sn]

    # Evaluate
    eval_rows = []
    for name in proposed:
        s = scores_df[name]
        max_r = 0
        max_r_name = ''
        for en in existing_names:
            if en in scores_df.columns:
                r_val = abs(s.corr(scores_df[en]))
                if r_val > max_r:
                    max_r = r_val
                    max_r_name = en

        eval_rows.append({
            'score': name,
            'description': proposed[name]['description'],
            'n_pos': len(proposed[name]['pos_vars']),
            'n_neg': len(proposed[name]['neg_vars']),
            'range': f"[{s.min():.1f}, {s.max():.1f}]",
            'weighted_std': round(weighted_std(s.values, pop.values), 2),
            'max_r_with_existing': round(max_r, 3),
            'most_similar_to': max_r_name,
            'novel': max_r < 0.5,
        })

    df_eval = pd.DataFrame(eval_rows)
    df_eval.to_csv(os.path.join(OUTPUT_DIR, '11_party_scores_evaluation.csv'), index=False)

    # Print
    print(f"\n  Scores proposés :")
    for _, r in df_eval.iterrows():
        novel = "NOUVEAU" if r['novel'] else "REDONDANT"
        print(f"\n  {r['score']}:")
        print(f"    {r['description']}")
        print(f"    pos ({r['n_pos']}): {proposed[r['score']]['pos_vars']}")
        print(f"    neg ({r['n_neg']}): {proposed[r['score']]['neg_vars']}")
        print(f"    range={r['range']}  σ_pop={r['weighted_std']:.1f}  max|r|={r['max_r_with_existing']:.3f} vs {r['most_similar_to']}  → {novel}")

    # Correlation matrix of all proposed + existing
    all_names = list(proposed.keys()) + existing_names
    available = [s for s in all_names if s in scores_df.columns]
    corr_all = scores_df[available].corr()
    corr_all.to_csv(os.path.join(OUTPUT_DIR, '11_party_scores_correlation.csv'))

    # Heatmap
    fig, ax = plt.subplots(figsize=(14, 12))
    sns.heatmap(corr_all, cmap='RdBu_r', center=0, vmin=-1, vmax=1,
                ax=ax, annot=True, fmt='.2f', annot_kws={'size': 7},
                linewidths=0.5, square=True)
    ax.set_title('Corrélations entre tous les scores\n(proposés par l\'analyse + existants)', fontsize=11)
    ax.tick_params(labelsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, '11_party_scores_heatmap.png'), dpi=120, bbox_inches='tight')
    plt.close(fig)

    # Save score definitions
    defs = []
    for name, cfg in proposed.items():
        defs.append({
            'score': name, 'description': cfg['description'],
            'pos_vars': '|'.join(cfg['pos_vars']),
            'neg_vars': '|'.join(cfg['neg_vars']),
        })
    pd.DataFrame(defs).to_csv(os.path.join(OUTPUT_DIR, '11_party_informed_scores.csv'), index=False)

    print(f"\n  → Fichiers : 11_party_informed_scores.csv, 11_party_scores_evaluation.csv,")
    print(f"               11_party_scores_correlation.csv, 11_party_scores_heatmap.png")

    return proposed, scores_df


# ═══════════════════════════════════════════════════════════════════════════════
#  PART 1 (cont.) — EXTENSIONS AGNOSTIQUES
# ═══════════════════════════════════════════════════════════════════════════════

# ── 1.7 UMAP ────────────────────────────────────────────────────────────────

def umap_exploration(df, var_names, pca_results):
    """UMAP dimensionality reduction — complements t-SNE with better global structure."""
    print("\n" + "─" * 70)
    print("  1.7  UMAP (alternative au t-SNE)")
    print("─" * 70)

    if not HAS_UMAP:
        print("  [!] umap-learn non installé — section ignorée")
        return

    X_std = pca_results['X_std']
    pop = df['_pop'].values

    # Same stratified subsample as t-SNE
    n_sample = min(15000, len(df))
    rng = np.random.RandomState(RANDOM_SEED)
    pop_deciles = pd.qcut(pop, 10, labels=False, duplicates='drop')
    idx_sample = []
    for d in range(10):
        d_idx = np.where(pop_deciles == d)[0]
        n_take = min(len(d_idx), n_sample // 10)
        idx_sample.extend(rng.choice(d_idx, n_take, replace=False))
    idx_sample = np.array(idx_sample)

    # PCA pre-reduction
    n_pca_dims = min(20, X_std.shape[1])
    scores_sub = X_std[idx_sample] @ pca_results['eigenvectors'][:, :n_pca_dims]

    print(f"\n  UMAP sur {len(idx_sample)} IRIS (n_neighbors=15, min_dist=0.1)...")
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2,
                         random_state=RANDOM_SEED, metric='euclidean')
    embed = reducer.fit_transform(scores_sub)

    print("  INTERPRÉTATION SOCIO-POLITIQUE :")
    print("    UMAP préserve mieux la structure globale que t-SNE : les distances entre")
    print("    clusters ont un sens (clusters éloignés = profils sociologiques très différents).")
    print("    C'est particulièrement utile pour identifier les 'ponts' sociologiques entre")
    print("    électorats (ex: zones péri-urbaines entre bloc Le Pen et bloc Macron).")

    # Plot by dominant candidate
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    ax = axes[0]
    cands_sub = df['dominant_candidate'].iloc[idx_sample].values
    for cand in CANDIDATES:
        mask = cands_sub == cand
        if mask.sum() > 0:
            ax.scatter(embed[mask, 0], embed[mask, 1], c=CANDIDATE_COLORS.get(cand, '#ccc'),
                       s=3, alpha=0.4, label=cand, rasterized=True)
    ax.set_title('UMAP coloré par candidat dominant (Prés. 2022 T1)')
    ax.legend(fontsize=7, markerscale=3)
    ax.set_xticks([])
    ax.set_yticks([])

    ax = axes[1]
    if 'score_cap_cult' in df.columns:
        color_var = df['score_cap_cult'].iloc[idx_sample].values
    else:
        color_var = df['DISP_MED21'].iloc[idx_sample].values
    sc = ax.scatter(embed[:, 0], embed[:, 1], c=color_var, cmap='RdBu_r',
                    s=3, alpha=0.4, rasterized=True)
    plt.colorbar(sc, ax=ax, shrink=0.7)
    ax.set_title('UMAP coloré par score capital culturel')
    ax.set_xticks([])
    ax.set_yticks([])

    fig.suptitle(f'UMAP ({len(idx_sample)} IRIS, n_neighbors=15)', fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, '05b_umap_by_party.png'), dpi=120)
    plt.close(fig)

    # UMAP colored by multiple scores
    score_vars = [s for s in EXISTING_SCORES_CONFIG if s in df.columns]
    n_scores = len(score_vars)
    ncols = 4
    nrows = (n_scores + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(20, 5 * nrows))
    axes = axes.flatten()
    for i, sv in enumerate(score_vars):
        ax = axes[i]
        vals = df[sv].iloc[idx_sample].values
        sc = ax.scatter(embed[:, 0], embed[:, 1], c=vals, cmap='RdBu_r',
                        s=2, alpha=0.4, rasterized=True)
        plt.colorbar(sc, ax=ax, shrink=0.6)
        ax.set_title(sv, fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)
    fig.suptitle('UMAP coloré par scores composites existants', fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, '05b_umap_by_score.png'), dpi=120)
    plt.close(fig)

    print(f"  → Fichiers : 05b_umap_by_party.png, 05b_umap_by_score.png")


# ── 1.8 Clustering K-Means ──────────────────────────────────────────────────

def clustering_analysis(df, var_names, pca_results):
    """K-Means clustering on PCA scores with profiling."""
    print("\n" + "─" * 70)
    print("  1.8  ANALYSE DE CLUSTERING (K-Means)")
    print("─" * 70)

    pop = df['_pop'].values
    n_retain = pca_results['n_retain']
    scores = pca_results['scores'][:, :n_retain]

    # Test k=4 to k=12
    k_range = range(4, 13)
    inertias = []
    silhouettes = []

    # Subsample for silhouette (too slow on full dataset)
    n_sil = min(20000, len(df))
    rng = np.random.RandomState(RANDOM_SEED)
    sil_idx = rng.choice(len(df), n_sil, replace=False)

    print(f"\n  Test de k=4 à k=12 sur {n_retain} PCs...")
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=RANDOM_SEED, n_init=10, max_iter=300)
        km.fit(scores)
        inertias.append(km.inertia_)
        sil = silhouette_score(scores[sil_idx], km.labels_[sil_idx], sample_size=n_sil)
        silhouettes.append(sil)
        print(f"    k={k:2d}  inertia={km.inertia_:12.0f}  silhouette={sil:.4f}")

    # Best k by silhouette
    best_k = list(k_range)[np.argmax(silhouettes)]
    print(f"\n  Meilleur k par silhouette : k={best_k} (sil={max(silhouettes):.4f})")

    # Elbow + silhouette plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    ax1.plot(list(k_range), inertias, 'bo-')
    ax1.set_xlabel('k')
    ax1.set_ylabel('Inertia')
    ax1.set_title('Méthode du coude')
    ax1.axvline(best_k, color='red', ls='--', label=f'best k={best_k}')
    ax1.legend()

    ax2.plot(list(k_range), silhouettes, 'go-')
    ax2.set_xlabel('k')
    ax2.set_ylabel('Silhouette score')
    ax2.set_title('Score silhouette')
    ax2.axvline(best_k, color='red', ls='--', label=f'best k={best_k}')
    ax2.legend()

    fig.suptitle('K-Means : choix du nombre de clusters', fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, '12_cluster_silhouette.png'), dpi=120)
    plt.close(fig)

    # Final clustering with best k
    km_final = KMeans(n_clusters=best_k, random_state=RANDOM_SEED, n_init=10)
    df['cluster'] = km_final.fit_predict(scores)

    # Cluster profiles (weighted means of original variables)
    profile_rows = []
    for c in range(best_k):
        mask = df['cluster'] == c
        n_iris = mask.sum()
        pop_total = pop[mask].sum()
        row = {'cluster': c, 'n_iris': n_iris, 'pop_total': int(pop_total)}
        for v in var_names:
            row[v] = weighted_mean(df.loc[mask, v].values, pop[mask])
        profile_rows.append(row)
    df_profiles = pd.DataFrame(profile_rows)
    df_profiles.to_csv(os.path.join(OUTPUT_DIR, '12_cluster_profiles.csv'), index=False)

    # Cluster naming heuristics
    cluster_names = {}
    for c in range(best_k):
        row = df_profiles.iloc[c]
        traits = []
        if row.get('pct_hlm', 0) > 20: traits.append('HLM')
        if row.get('pct_csp_plus', 0) > 15: traits.append('cadres')
        if row.get('pct_csp_ouvrier', 0) > 15: traits.append('ouvrier')
        if row.get('pct_65_plus', 0) > 25: traits.append('âgé')
        if row.get('pct_etudiants', 0) > 10: traits.append('étudiant')
        if row.get('pct_immigres', 0) > 15: traits.append('immigré')
        if row.get('pct_maison', 0) > 60: traits.append('pavillonnaire')
        if row.get('pct_appart', 0) > 80: traits.append('urbain-dense')
        if row.get('DISP_MED21', 0) > 25000: traits.append('aisé')
        if row.get('DISP_TP6021', 0) > 20: traits.append('précaire')
        name = '/'.join(traits[:3]) if traits else f'cluster_{c}'
        cluster_names[c] = name
        print(f"    Cluster {c} ({name}): n={row['n_iris']}, pop={row['pop_total']/1e6:.2f}M")

    # Profiles heatmap (z-scored)
    var_cols = [v for v in var_names if v in df_profiles.columns]
    z_profiles = df_profiles[var_cols].copy()
    for v in var_cols:
        mu = weighted_mean(df[v].values, pop)
        sigma = weighted_std(df[v].values, pop)
        if sigma > 1e-10:
            z_profiles[v] = (z_profiles[v] - mu) / sigma
        else:
            z_profiles[v] = 0
    z_profiles.index = [f"C{c}: {cluster_names[c]}" for c in range(best_k)]

    # Select top 40 most variable across clusters
    var_range = z_profiles.abs().max(axis=0).sort_values(ascending=False)
    top40 = var_range.head(40).index.tolist()

    fig, ax = plt.subplots(figsize=(16, 10))
    sns.heatmap(z_profiles[top40].T, cmap='RdBu_r', center=0, vmin=-3, vmax=3,
                ax=ax, linewidths=0.5, annot=True, fmt='.1f', annot_kws={'size': 6})
    ax.set_title(f'Profils des {best_k} clusters (z-scores, top 40 variables)', fontsize=11)
    ax.tick_params(labelsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, '12_cluster_profiles_heatmap.png'), dpi=120, bbox_inches='tight')
    plt.close(fig)

    # Cross-tabulation clusters × dominant candidate
    ct = pd.crosstab(df['cluster'].map(lambda c: f"C{c}: {cluster_names[c]}"),
                     df['dominant_candidate'], margins=True)
    ct.to_csv(os.path.join(OUTPUT_DIR, '12_cluster_party_crosstab.csv'))

    # Chi² test
    ct_no_margins = ct.iloc[:-1, :-1]
    chi2, p_chi2, dof, expected = stats.chi2_contingency(ct_no_margins)
    print(f"\n  Chi² clusters × candidat dominant : χ²={chi2:.1f}, p={p_chi2:.2e}, dof={dof}")

    # Mosaic-like stacked bar plot
    ct_pct = ct_no_margins.div(ct_no_margins.sum(axis=1), axis=0) * 100
    fig, ax = plt.subplots(figsize=(14, 8))
    ct_pct.plot(kind='barh', stacked=True, ax=ax,
                color=[CANDIDATE_COLORS.get(c, '#ccc') for c in ct_pct.columns])
    ax.set_xlabel('% des IRIS du cluster')
    ax.set_title(f'Composition électorale des clusters (χ²={chi2:.0f}, p={p_chi2:.1e})')
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, '12_cluster_party_mosaic.png'), dpi=120, bbox_inches='tight')
    plt.close(fig)

    print("  INTERPRÉTATION SOCIO-POLITIQUE :")
    print("    Le clustering révèle des 'types de territoires' au-delà du simple clivage")
    print("    gauche/droite. On observe typiquement : quartiers populaires urbains (HLM,")
    print("    immigration → Mélenchon), périurbain pavillonnaire (voiture, maison →")
    print("    Le Pen), centres-villes aisés (cadres, vélo → Macron/Jadot), etc.")

    print(f"\n  → Fichiers : 12_cluster_silhouette.png, 12_cluster_profiles.csv,")
    print(f"               12_cluster_profiles_heatmap.png, 12_cluster_party_crosstab.csv,")
    print(f"               12_cluster_party_mosaic.png")

    return km_final, cluster_names


# ── 1.9 Redundancy Analysis ─────────────────────────────────────────────────

def redundancy_analysis(df, var_names, corr_matrix, redundant_pairs, pca_results):
    """Build redundancy graph, select core non-redundant variables."""
    print("\n" + "─" * 70)
    print("  1.9  ANALYSE DE REDONDANCE ET SÉLECTION DE VARIABLES")
    print("─" * 70)

    pop = df['_pop'].values

    # Build redundancy graph (edges = |r| > 0.85)
    G = nx.Graph()
    G.add_nodes_from(var_names)
    for _, row in redundant_pairs.iterrows():
        G.add_edge(row['var1'], row['var2'], weight=abs(row['pearson_r']))

    components = list(nx.connected_components(G))
    n_comp = len([c for c in components if len(c) > 1])
    print(f"\n  Graphe de redondance : {len(G.edges)} arêtes, {n_comp} composantes connexes (>1 var)")

    # For each connected component, keep variable with best discriminant power
    core_vars = []
    removed_vars = []
    for comp in components:
        if len(comp) == 1:
            core_vars.append(list(comp)[0])
            continue

        # Discriminant power = variance of candidate barycentre positions
        best_var = None
        best_power = -1
        for v in comp:
            if v not in df.columns:
                continue
            # Compute inter-candidate variance
            bary_vals = []
            for cand in CANDIDATES:
                sc = f'score_{cand}'
                if sc in df.columns:
                    w = df[sc].fillna(0).values * pop
                    if w.sum() > 0:
                        bary_vals.append(weighted_mean(df[v].values, w))
            if bary_vals:
                power = np.var(bary_vals)
                if power > best_power:
                    best_power = power
                    best_var = v

        if best_var:
            core_vars.append(best_var)
            removed_vars.extend([v for v in comp if v != best_var])
            print(f"    Composante {list(comp)[:3]}... → garder {best_var} (discr={best_power:.2f})")
        else:
            core_vars.extend(list(comp))

    print(f"\n  Core set : {len(core_vars)} variables (supprimées : {len(removed_vars)})")
    pd.DataFrame({'variable': core_vars, 'kept': True}).to_csv(
        os.path.join(OUTPUT_DIR, '13_core_variable_set.csv'), index=False)

    # Redundancy graph visualization
    fig, ax = plt.subplots(figsize=(16, 12))
    pos = nx.spring_layout(G, seed=RANDOM_SEED, k=2)
    # Color nodes: core = green, removed = red
    node_colors = ['#16A34A' if v in core_vars else '#DC2626' for v in G.nodes]
    nx.draw(G, pos, ax=ax, node_color=node_colors, node_size=100,
            font_size=5, with_labels=True, edge_color='gray', alpha=0.7,
            width=[G[u][v]['weight'] * 2 for u, v in G.edges])
    ax.set_title(f'Graphe de redondance (|r|>0.85)\nVert=gardé ({len(core_vars)}), Rouge=supprimé ({len(removed_vars)})')
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, '13_redundancy_graph.png'), dpi=120, bbox_inches='tight')
    plt.close(fig)

    # Compare PCA on core set vs full set
    core_available = [v for v in core_vars if v in df.columns]
    X_core = df[core_available].values.astype(float)
    for j in range(X_core.shape[1]):
        col = X_core[:, j]
        nans = ~np.isfinite(col)
        if nans.any():
            col[nans] = np.nanmedian(col)

    w_norm = pop / pop.sum()
    means = (X_core * w_norm[:, None]).sum(axis=0)
    X_c = X_core - means
    stds = np.sqrt((X_c ** 2 * w_norm[:, None]).sum(axis=0))
    stds[stds < 1e-10] = 1e-10
    X_std = X_c / stds
    C = (X_std * w_norm[:, None]).T @ X_std
    eigenvalues, _ = np.linalg.eigh(C)
    eigenvalues = np.sort(np.maximum(eigenvalues, 0))[::-1]
    cumvar_core = np.cumsum(eigenvalues / eigenvalues.sum())

    cumvar_full = np.cumsum(pca_results['explained_ratio'])

    comp_data = []
    for n_pc in [5, 10, 15, 20]:
        if n_pc <= len(cumvar_full) and n_pc <= len(cumvar_core):
            comp_data.append({
                'n_PCs': n_pc,
                'cumvar_full': round(cumvar_full[n_pc - 1] * 100, 1),
                'cumvar_core': round(cumvar_core[n_pc - 1] * 100, 1),
            })
    pd.DataFrame(comp_data).to_csv(os.path.join(OUTPUT_DIR, '13_core_pca_comparison.csv'), index=False)

    print(f"\n  Comparaison PCA :")
    for row in comp_data:
        print(f"    {row['n_PCs']} PCs : full={row['cumvar_full']:.1f}%  core={row['cumvar_core']:.1f}%")

    print(f"\n  → Fichiers : 13_redundancy_graph.png, 13_core_variable_set.csv, 13_core_pca_comparison.csv")
    return core_vars


# ── 1.10 Mutual Information ─────────────────────────────────────────────────

def mutual_information_analysis(df, var_names):
    """MI vs Pearson² to detect non-linear dependencies."""
    print("\n" + "─" * 70)
    print("  1.10  MUTUAL INFORMATION ET DÉPENDANCES NON-LINÉAIRES")
    print("─" * 70)

    pop = df['_pop'].values

    # Subsample for speed
    n_sample = min(10000, len(df))
    rng = np.random.RandomState(RANDOM_SEED)
    idx = rng.choice(len(df), n_sample, replace=False)

    X = df[var_names].iloc[idx].values.astype(float)
    for j in range(X.shape[1]):
        col = X[:, j]
        nans = ~np.isfinite(col)
        if nans.any():
            col[nans] = np.nanmedian(col)

    # Compute pairwise MI for a selection of important pairs
    # (full pairwise is O(p²) × MI cost — we compute MI for each var vs top 20 others)
    print(f"\n  Calcul MI sur {n_sample} IRIS, {len(var_names)} variables...")

    # Faster approach: MI of each variable against all others via mutual_info_regression
    mi_matrix = np.zeros((len(var_names), len(var_names)))
    for j in range(len(var_names)):
        mi_vals = mutual_info_regression(X, X[:, j], random_state=RANDOM_SEED, n_neighbors=5)
        mi_matrix[j, :] = mi_vals

    # Symmetrize
    mi_matrix = (mi_matrix + mi_matrix.T) / 2
    np.fill_diagonal(mi_matrix, 0)

    # Pearson correlation matrix on same subsample
    corr_sub = np.corrcoef(X.T)
    r2_matrix = corr_sub ** 2

    # Find pairs where MI >> r²
    mi_rows = []
    n = len(var_names)
    for i in range(n):
        for j in range(i + 1, n):
            mi_val = mi_matrix[i, j]
            r2_val = r2_matrix[i, j]
            # Normalize MI to [0,1] scale approximately
            mi_norm = min(mi_val / max(mi_matrix.max(), 1e-10), 1.0)
            gap = mi_norm - r2_val
            mi_rows.append({
                'var1': var_names[i], 'var2': var_names[j],
                'MI': round(mi_val, 4), 'MI_norm': round(mi_norm, 4),
                'r2': round(r2_val, 4), 'gap_MI_r2': round(gap, 4),
            })

    df_mi = pd.DataFrame(mi_rows).sort_values('gap_MI_r2', ascending=False)
    df_mi.to_csv(os.path.join(OUTPUT_DIR, '14_mutual_information.csv'), index=False)

    # Top 20 non-linear pairs
    top20 = df_mi.head(20)
    print(f"\n  Top 20 paires avec le plus grand écart MI - r² (dépendances non-linéaires) :")
    for _, r in top20.iterrows():
        print(f"    {r['var1']:30s} × {r['var2']:30s}  MI_n={r['MI_norm']:.3f}  r²={r['r2']:.3f}  gap={r['gap_MI_r2']:+.3f}")

    # MI_norm vs r² scatter
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.scatter(df_mi['r2'], df_mi['MI_norm'], s=5, alpha=0.3, c='steelblue')
    ax.plot([0, 1], [0, 1], 'r--', lw=1, label='MI = r²')
    # Annotate top 10
    for _, r in top20.head(10).iterrows():
        ax.annotate(f"{r['var1'][:15]}×{r['var2'][:15]}",
                    (r['r2'], r['MI_norm']), fontsize=5, alpha=0.8)
    ax.set_xlabel('r² (Pearson²)')
    ax.set_ylabel('MI normalisée')
    ax.set_title('Mutual Information vs r² — points au-dessus de la diagonale = non-linéarité')
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, '14_mi_vs_r2_scatter.png'), dpi=120)
    plt.close(fig)

    # Scatter plots of top 6 non-linear pairs
    top6 = top20.head(6)
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    for i, (_, r) in enumerate(top6.iterrows()):
        ax = axes[i]
        v1, v2 = r['var1'], r['var2']
        x1 = df[v1].iloc[idx].values
        x2 = df[v2].iloc[idx].values
        valid = np.isfinite(x1) & np.isfinite(x2)
        ax.scatter(x1[valid], x2[valid], s=2, alpha=0.3, c='steelblue')
        ax.set_xlabel(v1, fontsize=7)
        ax.set_ylabel(v2, fontsize=7)
        ax.set_title(f'MI_n={r["MI_norm"]:.3f}, r²={r["r2"]:.3f}\ngap={r["gap_MI_r2"]:+.3f}', fontsize=8)
    fig.suptitle('Top 6 paires avec plus forte dépendance non-linéaire', fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, '14_nonlinear_scatters.png'), dpi=120)
    plt.close(fig)

    print("  INTERPRÉTATION :")
    print("    Les paires à fort écart MI-r² capturent des relations non-linéaires (seuils,")
    print("    effets de saturation, interactions). Ces relations sont invisibles à une PCA")
    print("    standard et justifient l'utilisation de méthodes non-linéaires (RF, UMAP).")

    print(f"\n  → Fichiers : 14_mutual_information.csv, 14_mi_vs_r2_scatter.png, 14_nonlinear_scatters.png")
    return df_mi


# ── 1.11 Geographic Variance (ICC) ──────────────────────────────────────────

def geographic_variance_analysis(df, var_names):
    """ICC: inter-department variance / total variance per variable."""
    print("\n" + "─" * 70)
    print("  1.11  VARIANCE GÉOGRAPHIQUE (ICC inter-département)")
    print("─" * 70)

    pop = df['_pop'].values

    # Extract department code from IRIS code (first 2 or 3 digits of COM)
    if 'COM' in df.columns:
        df['_dept'] = df['COM'].astype(str).str[:2]
        # Handle DOM-TOM (97x)
        mask_dom = df['_dept'] == '97'
        df.loc[mask_dom, '_dept'] = df.loc[mask_dom, 'COM'].astype(str).str[:3]
    elif 'IRIS' in df.columns:
        df['_dept'] = df['IRIS'].astype(str).str[:2]
    else:
        print("  [!] Pas de code commune/IRIS pour calculer l'ICC — section ignorée")
        return None

    depts = df['_dept'].unique()
    n_depts = len(depts)
    print(f"\n  Départements détectés : {n_depts}")

    rows = []
    for v in var_names:
        x = df[v].values.astype(float)
        valid = np.isfinite(x)
        if valid.sum() < 100:
            continue

        # Total weighted variance
        var_total = weighted_std(x, pop) ** 2
        if var_total < 1e-15:
            continue

        # Inter-department variance (variance of department means)
        dept_means = []
        dept_pops = []
        for d in depts:
            mask_d = (df['_dept'] == d).values & valid
            if mask_d.sum() < 5:
                continue
            dept_means.append(weighted_mean(x[mask_d], pop[mask_d]))
            dept_pops.append(pop[mask_d].sum())

        if len(dept_means) < 5:
            continue

        dept_means = np.array(dept_means)
        dept_pops = np.array(dept_pops)
        grand_mean = np.average(dept_means, weights=dept_pops)
        var_inter = np.average((dept_means - grand_mean) ** 2, weights=dept_pops)

        icc = var_inter / var_total

        rows.append({
            'variable': v,
            'var_total': round(var_total, 4),
            'var_inter_dept': round(var_inter, 4),
            'ICC': round(icc, 4),
            'signal': 'territorial' if icc > 0.3 else ('mixte' if icc > 0.1 else 'local'),
        })

    df_icc = pd.DataFrame(rows).sort_values('ICC', ascending=False)
    df_icc.to_csv(os.path.join(OUTPUT_DIR, '15_geographic_variance.csv'), index=False)

    print(f"\n  Variables à fort signal territorial (ICC > 0.3) :")
    for _, r in df_icc[df_icc['ICC'] > 0.3].iterrows():
        print(f"    {r['variable']:35s}  ICC={r['ICC']:.3f}")
    print(f"\n  Variables à signal local (ICC < 0.1) :")
    for _, r in df_icc[df_icc['ICC'] < 0.1].head(10).iterrows():
        print(f"    {r['variable']:35s}  ICC={r['ICC']:.3f}")

    # Bar plot
    fig, ax = plt.subplots(figsize=(14, max(8, len(df_icc) * 0.25)))
    colors = ['#DC2626' if icc > 0.3 else '#F97316' if icc > 0.1 else '#16A34A'
              for icc in df_icc['ICC']]
    ax.barh(range(len(df_icc)), df_icc['ICC'].values, color=colors, alpha=0.8)
    ax.set_yticks(range(len(df_icc)))
    ax.set_yticklabels(df_icc['variable'].values, fontsize=6)
    ax.axvline(0.3, color='red', ls='--', lw=1, label='ICC=0.3 (territorial)')
    ax.axvline(0.1, color='orange', ls='--', lw=1, label='ICC=0.1 (mixte)')
    ax.set_xlabel('ICC (variance inter-département / totale)')
    ax.set_title('ICC : effet géographique par variable')
    ax.legend(fontsize=8)
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, '15_icc_barplot.png'), dpi=120, bbox_inches='tight')
    plt.close(fig)

    print("\n  INTERPRÉTATION SOCIO-POLITIQUE :")
    print("    ICC élevé = la variable varie surtout entre départements (effet macro-")
    print("    géographique, ex: chauffage fioul = Nord/campagne, pct_immigrés = grandes")
    print("    villes). ICC faible = variation au sein même des communes (ex: % HLM,")
    print("    revenu médian). Pour la visualisation, les variables à ICC moyen/faible sont")
    print("    souvent plus discriminantes localement entre IRIS d'une même ville.")

    print(f"\n  → Fichiers : 15_geographic_variance.csv, 15_icc_barplot.png")
    return df_icc


# ═══════════════════════════════════════════════════════════════════════════════
#  PART 2 (cont.) — EXTENSIONS CONDITIONNÉES
# ═══════════════════════════════════════════════════════════════════════════════

# ── 2.6 Random Forest + PDP ─────────────────────────────────────────────────

def random_forest_analysis(df, var_names):
    """RandomForest multiclass + permutation importance + Partial Dependence."""
    print("\n" + "─" * 70)
    print("  2.6  RANDOM FOREST (importance + PDP)")
    print("─" * 70)

    pop = df['_pop'].values
    X = df[var_names].values.astype(float)

    # Impute NaN
    for j in range(X.shape[1]):
        col = X[:, j]
        nans = ~np.isfinite(col)
        if nans.any():
            col[nans] = np.nanmedian(col)

    # Multiclass target: dominant candidate (top 5 only)
    score_cols = [f'score_{c}' for c in CANDIDATES if f'score_{c}' in df.columns]
    cand_names = [c.replace('score_', '') for c in score_cols]
    scores = df[score_cols].fillna(0).values
    dominant_idx = scores.argmax(axis=1)
    dominant_score = scores.max(axis=1)
    labels = np.array([cand_names[i] for i in dominant_idx])
    labels[dominant_score < 20] = 'MIXED'

    class_counts = pd.Series(labels).value_counts()
    valid_classes = class_counts[class_counts >= 200].index.tolist()
    if 'MIXED' in valid_classes:
        valid_classes.remove('MIXED')

    mask = np.isin(labels, valid_classes)
    X_rf = X[mask]
    y_rf = labels[mask]
    w_rf = pop[mask]

    print(f"\n  Classes : {valid_classes}")
    print(f"  Échantillon : {len(X_rf)} IRIS")

    # Subsample for speed
    n_max = 30000
    if len(X_rf) > n_max:
        rng = np.random.RandomState(RANDOM_SEED)
        idx = rng.choice(len(X_rf), n_max, replace=False)
        X_rf, y_rf, w_rf = X_rf[idx], y_rf[idx], w_rf[idx]

    print("  Entraînement RandomForest (300 arbres, max_depth=12)...")
    rf = RandomForestClassifier(n_estimators=300, max_depth=12, random_state=RANDOM_SEED,
                                 n_jobs=-1, class_weight='balanced')
    rf.fit(X_rf, y_rf, sample_weight=w_rf)

    # Cross-validation
    cv_scores = cross_val_score(
        RandomForestClassifier(n_estimators=100, max_depth=12, random_state=RANDOM_SEED,
                               n_jobs=-1, class_weight='balanced'),
        X_rf, y_rf, cv=5, scoring='accuracy'
    )
    print(f"  Accuracy CV 5-fold : {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

    # Permutation importance (more reliable than impurity-based)
    print("  Calcul permutation importance...")
    perm_imp = permutation_importance(rf, X_rf, y_rf, n_repeats=10,
                                       random_state=RANDOM_SEED, n_jobs=-1,
                                       sample_weight=w_rf)

    df_imp = pd.DataFrame({
        'variable': var_names,
        'impurity_importance': rf.feature_importances_,
        'perm_importance_mean': perm_imp.importances_mean,
        'perm_importance_std': perm_imp.importances_std,
    }).sort_values('perm_importance_mean', ascending=False)
    df_imp.to_csv(os.path.join(OUTPUT_DIR, '16_rf_permutation_importance.csv'), index=False)

    print(f"\n  Top 15 variables (permutation importance) :")
    for _, r in df_imp.head(15).iterrows():
        print(f"    {r['variable']:35s}  perm={r['perm_importance_mean']:.4f}±{r['perm_importance_std']:.4f}  "
              f"impurity={r['impurity_importance']:.4f}")

    # Importance bar plot
    top20 = df_imp.head(20)
    fig, ax = plt.subplots(figsize=(12, 8))
    y_pos = range(len(top20))
    ax.barh(y_pos, top20['perm_importance_mean'].values, xerr=top20['perm_importance_std'].values,
            color='steelblue', alpha=0.8)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(top20['variable'].values, fontsize=8)
    ax.set_xlabel('Permutation importance')
    ax.set_title(f'Random Forest — Top 20 variables (accuracy CV: {cv_scores.mean():.1%})')
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, '16_rf_importance_barplot.png'), dpi=120, bbox_inches='tight')
    plt.close(fig)

    # Partial Dependence Plots for top 8 variables
    top8_idx = [list(var_names).index(v) for v in df_imp.head(8)['variable'] if v in var_names]
    fig, axes = plt.subplots(2, 4, figsize=(24, 10))
    try:
        PartialDependenceDisplay.from_estimator(
            rf, X_rf, features=top8_idx, feature_names=var_names,
            ax=axes.flatten()[:len(top8_idx)], grid_resolution=50, n_jobs=-1
        )
        fig.suptitle('Partial Dependence Plots — Top 8 variables (Random Forest)', fontsize=13)
        fig.tight_layout()
        fig.savefig(os.path.join(OUTPUT_DIR, '16_partial_dependence.png'), dpi=120, bbox_inches='tight')
    except Exception as e:
        print(f"  [!] PDP erreur : {e}")
    plt.close(fig)

    print("\n  INTERPRÉTATION :")
    print("    La permutation importance mesure la perte de précision quand on brouille une variable.")
    print("    C'est plus fiable que l'impurity importance (pas biaisée vers les vars à haute cardinalité).")
    print("    Les PDP montrent l'effet marginal de chaque variable sur les prédictions.")

    print(f"\n  → Fichiers : 16_rf_permutation_importance.csv, 16_rf_importance_barplot.png, 16_partial_dependence.png")
    return df_imp


# ── 2.7 Pairwise Candidate Contrasts ────────────────────────────────────────

def pairwise_contrasts(df, var_names):
    """Direct pairwise comparisons between major candidate pairs."""
    print("\n" + "─" * 70)
    print("  2.7  CONTRASTES PAR PAIRES DE CANDIDATS")
    print("─" * 70)

    pop = df['_pop'].values

    pairs = [
        ('LE_PEN', 'MELENCHON'),
        ('MACRON', 'LE_PEN'),
        ('MACRON', 'MELENCHON'),
        ('JADOT', 'LE_PEN'),
    ]

    all_results = []
    for cand_a, cand_b in pairs:
        if cand_a not in df['dominant_candidate'].values or cand_b not in df['dominant_candidate'].values:
            continue

        mask_a = df['dominant_candidate'] == cand_a
        mask_b = df['dominant_candidate'] == cand_b

        print(f"\n  {cand_a} ({mask_a.sum()} IRIS) vs {cand_b} ({mask_b.sum()} IRIS) :")

        contrast_rows = []
        for v in var_names:
            xa = df.loc[mask_a, v].values.astype(float)
            xb = df.loc[mask_b, v].values.astype(float)
            wa = pop[mask_a]
            wb = pop[mask_b]

            valid_a = np.isfinite(xa)
            valid_b = np.isfinite(xb)
            if valid_a.sum() < 30 or valid_b.sum() < 30:
                continue

            mean_a = np.average(xa[valid_a], weights=wa[valid_a])
            mean_b = np.average(xb[valid_b], weights=wb[valid_b])

            # Weighted t-test approximation
            std_a = np.sqrt(np.average((xa[valid_a] - mean_a) ** 2, weights=wa[valid_a]))
            std_b = np.sqrt(np.average((xb[valid_b] - mean_b) ** 2, weights=wb[valid_b]))
            pooled_std = np.sqrt((std_a ** 2 + std_b ** 2) / 2)

            if pooled_std < 1e-10:
                continue

            effect_size = (mean_a - mean_b) / pooled_std  # Cohen's d

            contrast_rows.append({
                'pair': f'{cand_a}_vs_{cand_b}',
                'variable': v,
                'mean_A': round(mean_a, 3),
                'mean_B': round(mean_b, 3),
                'diff': round(mean_a - mean_b, 3),
                'cohen_d': round(effect_size, 3),
                'abs_d': round(abs(effect_size), 3),
            })

        df_contrast = pd.DataFrame(contrast_rows).sort_values('abs_d', ascending=False)
        all_results.append(df_contrast)

        # Top 10
        for _, r in df_contrast.head(10).iterrows():
            sign = "+" if r['cohen_d'] > 0 else "-"
            print(f"    {sign} {r['variable']:35s}  d={r['cohen_d']:+.3f}  ({r['mean_A']:.1f} vs {r['mean_B']:.1f})")

    if all_results:
        df_all = pd.concat(all_results, ignore_index=True)
        df_all.to_csv(os.path.join(OUTPUT_DIR, '17_pairwise_contrasts.csv'), index=False)

        # Scatter plots for top 2 discriminating variables per pair
        fig, axes = plt.subplots(2, 2, figsize=(16, 14))
        for idx_pair, (cand_a, cand_b) in enumerate(pairs[:4]):
            ax = axes.flatten()[idx_pair]
            pair_data = df_all[df_all['pair'] == f'{cand_a}_vs_{cand_b}']
            if pair_data.empty:
                ax.set_visible(False)
                continue

            top2 = pair_data.head(2)['variable'].values
            if len(top2) < 2:
                ax.set_visible(False)
                continue

            v1, v2 = top2[0], top2[1]
            mask_a = df['dominant_candidate'] == cand_a
            mask_b = df['dominant_candidate'] == cand_b

            ax.scatter(df.loc[mask_a, v1], df.loc[mask_a, v2],
                       c=CANDIDATE_COLORS.get(cand_a, '#999'), s=3, alpha=0.3,
                       label=cand_a, rasterized=True)
            ax.scatter(df.loc[mask_b, v1], df.loc[mask_b, v2],
                       c=CANDIDATE_COLORS.get(cand_b, '#999'), s=3, alpha=0.3,
                       label=cand_b, rasterized=True)
            ax.set_xlabel(v1, fontsize=8)
            ax.set_ylabel(v2, fontsize=8)
            ax.set_title(f'{cand_a} vs {cand_b}', fontsize=10)
            ax.legend(fontsize=8, markerscale=3)

        fig.suptitle('Contrastes pairés — Top 2 variables discriminantes par paire', fontsize=13)
        fig.tight_layout()
        fig.savefig(os.path.join(OUTPUT_DIR, '17_pairwise_scatters.png'), dpi=120, bbox_inches='tight')
        plt.close(fig)

    print("\n  INTERPRÉTATION SOCIO-POLITIQUE :")
    print("    Le Pen vs Mélenchon : même milieux populaires mais clivage urbain/rural,")
    print("      immigration, logement social (HLM → Mélenchon, pavillonnaire → Le Pen).")
    print("    Macron vs Le Pen : clivage diplôme/revenu (CSP+/BAC+5 → Macron, ouvriers → Le Pen).")
    print("    Macron vs Mélenchon : même urbains diplômés mais clivage revenus/patrimoine.")
    print("    Jadot vs Le Pen : clivage maximal écolo-métropole vs périurbain-auto.")

    print(f"\n  → Fichiers : 17_pairwise_contrasts.csv, 17_pairwise_scatters.png")
    return df_all if all_results else None


# ── 2.8 WLS Regression ──────────────────────────────────────────────────────

def wls_regression(df, var_names):
    """Weighted Least Squares regression of each candidate's score."""
    print("\n" + "─" * 70)
    print("  2.8  RÉGRESSION LINÉAIRE PONDÉRÉE (WLS)")
    print("─" * 70)

    pop = df['_pop'].values

    X = df[var_names].values.astype(float)
    for j in range(X.shape[1]):
        col = X[:, j]
        nans = ~np.isfinite(col)
        if nans.any():
            col[nans] = np.nanmedian(col)

    # Standardize X for comparable coefficients
    scaler = StandardScaler()
    X_std = scaler.fit_transform(X)

    results = {}
    top5_cands = sorted(CANDIDATES, key=lambda c: df.get(f'score_{c}', pd.Series(0)).sum(),
                        reverse=True)[:6]

    for cand in top5_cands:
        sc = f'score_{cand}'
        if sc not in df.columns:
            continue

        y = df[sc].fillna(0).values.astype(float)

        # WLS with population weights
        X_wls = sm.add_constant(X_std)
        try:
            model = sm.WLS(y, X_wls, weights=pop).fit()
            betas = model.params[1:]  # Skip intercept
            pvals = model.pvalues[1:]
            r2 = model.rsquared_adj

            results[cand] = {
                'betas': betas,
                'pvals': pvals,
                'r2_adj': r2,
            }

            print(f"\n  {cand} — R² ajusté = {r2:.4f}")
            # Top 10 by |beta|
            top_idx = np.abs(betas).argsort()[::-1][:10]
            for k in top_idx:
                sig = '***' if pvals[k] < 0.001 else ('**' if pvals[k] < 0.01 else ('*' if pvals[k] < 0.05 else ''))
                print(f"    {var_names[k]:35s}  β={betas[k]:+.4f}  p={pvals[k]:.1e} {sig}")
        except Exception as e:
            print(f"  [!] WLS pour {cand} échoué : {e}")
            continue

    # Coefficients heatmap
    if results:
        beta_df = pd.DataFrame({cand: res['betas'] for cand, res in results.items()},
                                index=var_names)
        beta_df.to_csv(os.path.join(OUTPUT_DIR, '18_wls_coefficients.csv'))

        # R² comparison
        r2_data = [{'candidat': c, 'R2_adj_WLS': round(r['r2_adj'], 4)} for c, r in results.items()]
        pd.DataFrame(r2_data).to_csv(os.path.join(OUTPUT_DIR, '18_wls_vs_gbm.csv'), index=False)

        # Select top 30 variables by max |beta|
        max_beta = beta_df.abs().max(axis=1).sort_values(ascending=False)
        top30 = max_beta.head(30).index.tolist()

        fig, ax = plt.subplots(figsize=(14, 14))
        cand_order = [c for c in top5_cands if c in beta_df.columns]
        sns.heatmap(beta_df.loc[top30, cand_order], cmap='RdBu_r', center=0,
                    vmin=-0.5, vmax=0.5, ax=ax, linewidths=0.5,
                    annot=True, fmt='.2f', annot_kws={'size': 6})
        r2_str = ', '.join([f"{c}: {results[c]['r2_adj']:.3f}" for c in cand_order if c in results])
        ax.set_title(f'Coefficients WLS standardisés (top 30 variables)\nR² adj: {r2_str}', fontsize=10)
        ax.tick_params(labelsize=7)
        fig.tight_layout()
        fig.savefig(os.path.join(OUTPUT_DIR, '18_wls_coefficients_heatmap.png'), dpi=120, bbox_inches='tight')
        plt.close(fig)

    print("\n  INTERPRÉTATION :")
    print("    Les coefficients β standardisés mesurent l'effet 'toutes choses égales par ailleurs'.")
    print("    Un β positif = quand cette variable augmente (d'1 écart-type), le score du candidat")
    print("    augmente. C'est l'analyse la plus directement interprétable en socio-politique.")
    print("    Comparer R² WLS vs GBM montre si la relation est linéaire ou non.")

    print(f"\n  → Fichiers : 18_wls_coefficients.csv, 18_wls_coefficients_heatmap.png, 18_wls_vs_gbm.csv")
    return results


# ── 2.9 Abstention Analysis ─────────────────────────────────────────────────

def abstention_analysis(df, var_names):
    """Socio-economic profile of abstention."""
    print("\n" + "─" * 70)
    print("  2.9  ANALYSE DE L'ABSTENTION")
    print("─" * 70)

    if 'pct_abstention' not in df.columns:
        print("  [!] Pas de données d'abstention — section ignorée")
        return None

    pop = df['_pop'].values
    abst = df['pct_abstention'].fillna(0).values
    labels = get_var_labels()

    print(f"\n  Abstention : médiane={np.median(abst):.1f}%, "
          f"min={abst.min():.1f}%, max={abst.max():.1f}%")

    # Weighted correlations with abstention
    corr_rows = []
    for v in var_names:
        x = df[v].values.astype(float)
        valid = np.isfinite(x) & np.isfinite(abst)
        if valid.sum() < 100:
            continue
        r, p = stats.pearsonr(x[valid], abst[valid])
        # Population-weighted correlation
        w = pop[valid]
        x_c = x[valid] - np.average(x[valid], weights=w)
        a_c = abst[valid] - np.average(abst[valid], weights=w)
        cov_wa = np.average(x_c * a_c, weights=w)
        std_x = np.sqrt(np.average(x_c ** 2, weights=w))
        std_a = np.sqrt(np.average(a_c ** 2, weights=w))
        r_w = cov_wa / (std_x * std_a) if std_x > 0 and std_a > 0 else 0

        corr_rows.append({
            'variable': v,
            'label': labels.get(v, v),
            'r_pearson': round(r, 4),
            'r_weighted': round(r_w, 4),
            'p_value': p,
        })

    df_corr = pd.DataFrame(corr_rows).sort_values('r_weighted', key=abs, ascending=False)
    df_corr.to_csv(os.path.join(OUTPUT_DIR, '19_abstention_profile.csv'), index=False)

    print(f"\n  Top 15 corrélats de l'abstention (r pondéré population) :")
    for _, r in df_corr.head(15).iterrows():
        sig = '***' if r['p_value'] < 0.001 else ''
        print(f"    {r['variable']:35s}  r_w={r['r_weighted']:+.4f}  r={r['r_pearson']:+.4f} {sig}")

    # WLS regression of abstention
    X = df[var_names].values.astype(float)
    for j in range(X.shape[1]):
        col = X[:, j]
        nans = ~np.isfinite(col)
        if nans.any():
            col[nans] = np.nanmedian(col)

    scaler = StandardScaler()
    X_std = scaler.fit_transform(X)
    X_wls = sm.add_constant(X_std)
    try:
        model = sm.WLS(abst, X_wls, weights=pop).fit()
        print(f"\n  WLS Abstention : R² ajusté = {model.rsquared_adj:.4f}")
        betas = model.params[1:]
        pvals = model.pvalues[1:]
        top_idx = np.abs(betas).argsort()[::-1][:10]
        print(f"  Top 10 prédicteurs (β standardisés) :")
        for k in top_idx:
            sig = '***' if pvals[k] < 0.001 else ('**' if pvals[k] < 0.01 else '*' if pvals[k] < 0.05 else '')
            print(f"    {var_names[k]:35s}  β={betas[k]:+.4f}  p={pvals[k]:.1e} {sig}")
    except Exception as e:
        print(f"  [!] WLS abstention échoué : {e}")

    # Correlation bar plot (top 30)
    top30 = df_corr.head(30)
    fig, ax = plt.subplots(figsize=(12, 10))
    colors = ['#DC2626' if r > 0 else '#3B82F6' for r in top30['r_weighted']]
    ax.barh(range(len(top30)), top30['r_weighted'].values, color=colors, alpha=0.8)
    ax.set_yticks(range(len(top30)))
    ax.set_yticklabels(top30['variable'].values, fontsize=7)
    ax.axvline(0, color='black', lw=0.5)
    ax.set_xlabel('Corrélation pondérée avec abstention')
    ax.set_title('Profil sociologique de l\'abstention\n(top 30 variables corrélées)')
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, '19_abstention_correlations.png'), dpi=120, bbox_inches='tight')
    plt.close(fig)

    print("\n  INTERPRÉTATION SOCIO-POLITIQUE :")
    print("    L'abstention en France est fortement liée à la précarité socio-économique :")
    print("    chômage, faibles revenus, logement social, faible diplôme. Mais aussi aux")
    print("    jeunes urbains (étudiants). C'est le 'premier parti de France' et son profil")
    print("    socio croise celui de Mélenchon et Le Pen (quartiers populaires urbains et")
    print("    zones rurales isolées).")

    print(f"\n  → Fichiers : 19_abstention_profile.csv, 19_abstention_correlations.png")
    return df_corr


# ═══════════════════════════════════════════════════════════════════════════════
#  PART 3 — SYNTHÈSE ET RECOMMANDATIONS
# ═══════════════════════════════════════════════════════════════════════════════

# ── 3.1 Recommended Axis Pairs ──────────────────────────────────────────────

def recommend_axis_pairs(df, var_names, proposed_scores, party_scores, scores_df, party_scores_df):
    """Score and rank axis pairs for visualization."""
    print("\n" + "─" * 70)
    print("  3.1  RECOMMANDATION DE PAIRES D'AXES")
    print("─" * 70)

    pop = df['_pop'].values

    # Collect all scores
    all_scores = {}
    for name in EXISTING_SCORES_CONFIG:
        if name in df.columns:
            all_scores[name] = df[name].values
    for name in proposed_scores:
        if name in scores_df.columns:
            all_scores[name] = scores_df[name].values
    for name in party_scores:
        if name in party_scores_df.columns:
            all_scores[name] = party_scores_df[name].values

    score_names = list(all_scores.keys())
    print(f"\n  Scores disponibles : {len(score_names)}")

    # Candidate barycentres on each score
    bary = {}
    for sn in score_names:
        sv = all_scores[sn]
        cand_bary = {}
        for cand in CANDIDATES:
            sc = f'score_{cand}'
            if sc in df.columns:
                w = df[sc].fillna(0).values * pop
                if w.sum() > 0:
                    cand_bary[cand] = np.average(sv, weights=w)
        bary[sn] = cand_bary

    # Evaluate each pair
    pair_rows = []
    for i in range(len(score_names)):
        for j in range(i + 1, len(score_names)):
            sn1, sn2 = score_names[i], score_names[j]
            sv1, sv2 = all_scores[sn1], all_scores[sn2]

            # 1. Mutual correlation (want low)
            valid = np.isfinite(sv1) & np.isfinite(sv2)
            if valid.sum() < 100:
                continue
            r_mutual = abs(np.corrcoef(sv1[valid], sv2[valid])[0, 1])

            # 2. Discriminant power (variance of barycentres)
            if bary[sn1] and bary[sn2]:
                bary_x = list(bary[sn1].values())
                bary_y = list(bary[sn2].values())
                discrim = np.var(bary_x) + np.var(bary_y)
            else:
                discrim = 0

            # 3. Variance of IRIS
            var_iris = np.nanvar(sv1) + np.nanvar(sv2)

            # 4. Interpretability (fewer variables = more interpretable)
            n_vars_1 = len(proposed_scores.get(sn1, {}).get('pos_vars', [])) + len(proposed_scores.get(sn1, {}).get('neg_vars', []))
            n_vars_2 = len(proposed_scores.get(sn2, {}).get('pos_vars', [])) + len(proposed_scores.get(sn2, {}).get('neg_vars', []))
            if n_vars_1 == 0:
                n_vars_1 = len(EXISTING_SCORES_CONFIG.get(sn1, {}).get('pos_vars', [])) + len(EXISTING_SCORES_CONFIG.get(sn1, {}).get('neg_vars', []))
            if n_vars_2 == 0:
                n_vars_2 = len(EXISTING_SCORES_CONFIG.get(sn2, {}).get('pos_vars', [])) + len(EXISTING_SCORES_CONFIG.get(sn2, {}).get('neg_vars', []))
            if n_vars_1 == 0:
                n_vars_1 = len(party_scores.get(sn1, {}).get('pos_vars', []))
            if n_vars_2 == 0:
                n_vars_2 = len(party_scores.get(sn2, {}).get('pos_vars', []))
            interp = 1.0 / max(n_vars_1 + n_vars_2, 1)

            # Composite score (normalize components)
            orthog_score = max(0, 1 - r_mutual)  # 0 to 1, 1 is best
            # Normalize others relative to observed range
            pair_rows.append({
                'axis_x': sn1, 'axis_y': sn2,
                'r_mutual': round(r_mutual, 3),
                'discriminant_power': round(discrim, 2),
                'variance_iris': round(var_iris, 2),
                'interpretability': round(interp, 4),
                'orthogonality': round(orthog_score, 3),
            })

    df_pairs = pd.DataFrame(pair_rows)

    if len(df_pairs) > 0:
        # Normalize each criterion to [0,1] and compute composite score
        for col in ['orthogonality', 'discriminant_power', 'variance_iris', 'interpretability']:
            mn, mx = df_pairs[col].min(), df_pairs[col].max()
            if mx > mn:
                df_pairs[f'{col}_norm'] = (df_pairs[col] - mn) / (mx - mn)
            else:
                df_pairs[f'{col}_norm'] = 0.5

        df_pairs['composite_score'] = (
            0.30 * df_pairs['orthogonality_norm'] +
            0.35 * df_pairs['discriminant_power_norm'] +
            0.20 * df_pairs['variance_iris_norm'] +
            0.15 * df_pairs['interpretability_norm']
        )

        df_pairs = df_pairs.sort_values('composite_score', ascending=False)
        df_pairs.to_csv(os.path.join(OUTPUT_DIR, '20_recommended_axis_pairs.csv'), index=False)

        print(f"\n  Top 10 paires d'axes recommandées :")
        for _, r in df_pairs.head(10).iterrows():
            print(f"    {r['axis_x']:30s} × {r['axis_y']:30s}  "
                  f"score={r['composite_score']:.3f}  |r|={r['r_mutual']:.2f}  discr={r['discriminant_power']:.1f}")

        # Plot top 3 pairs
        top3 = df_pairs.head(3)
        fig, axes = plt.subplots(1, 3, figsize=(21, 7))
        for idx_p, (_, pair_row) in enumerate(top3.iterrows()):
            ax = axes[idx_p]
            sn_x, sn_y = pair_row['axis_x'], pair_row['axis_y']
            sv_x, sv_y = all_scores[sn_x], all_scores[sn_y]

            # Subsample
            n_plot = min(5000, len(df))
            rng = np.random.RandomState(RANDOM_SEED)
            plot_idx = rng.choice(len(df), n_plot, replace=False)

            cands_plot = df['dominant_candidate'].iloc[plot_idx].values
            for cand in CANDIDATES:
                mask_c = cands_plot == cand
                if mask_c.sum() > 0:
                    ax.scatter(sv_x[plot_idx[mask_c]], sv_y[plot_idx[mask_c]],
                               c=CANDIDATE_COLORS.get(cand, '#ccc'), s=3, alpha=0.4,
                               label=cand, rasterized=True)
            ax.legend(fontsize=6, markerscale=3)
            ax.set_xlabel(sn_x, fontsize=8)
            ax.set_ylabel(sn_y, fontsize=8)
            ax.set_title(f'Score={pair_row["composite_score"]:.3f}\n|r|={pair_row["r_mutual"]:.2f}', fontsize=9)

        fig.suptitle('Top 3 paires d\'axes recommandées (colorées par candidat dominant)', fontsize=13)
        fig.tight_layout()
        fig.savefig(os.path.join(OUTPUT_DIR, '20_top3_axis_pairs.png'), dpi=120, bbox_inches='tight')
        plt.close(fig)

    print(f"\n  → Fichiers : 20_recommended_axis_pairs.csv, 20_top3_axis_pairs.png")
    return df_pairs


# ── 3.2 Score Fiches ────────────────────────────────────────────────────────

def score_fiches(df, var_names, proposed_scores, party_scores, scores_df, party_scores_df):
    """Detailed 'fiches' for each proposed score with socio-political justification."""
    print("\n" + "─" * 70)
    print("  3.2  FICHES SCORES AVEC JUSTIFICATION SOCIO-POLITIQUE")
    print("─" * 70)

    pop = df['_pop'].values
    labels = get_var_labels()

    # Merge all proposed scores
    all_proposed = {}
    for name, cfg in proposed_scores.items():
        all_proposed[name] = cfg
    for name, cfg in party_scores.items():
        all_proposed[name] = cfg

    # Merge all score values
    all_score_vals = {}
    for name in all_proposed:
        if name in scores_df.columns:
            all_score_vals[name] = scores_df[name].values
        elif name in party_scores_df.columns:
            all_score_vals[name] = party_scores_df[name].values

    fiches = []
    fiche_text = []

    for name, cfg in all_proposed.items():
        if name not in all_score_vals:
            continue

        sv = all_score_vals[name]
        pos_vars = cfg.get('pos_vars', [])
        neg_vars = cfg.get('neg_vars', [])
        desc = cfg.get('description', cfg.get('pc_origin', ''))

        # Correlation with each candidate
        cand_corrs = {}
        cand_bary = {}
        for cand in CANDIDATES:
            sc = f'score_{cand}'
            if sc in df.columns:
                y = df[sc].fillna(0).values
                valid = np.isfinite(sv) & np.isfinite(y)
                if valid.sum() > 100:
                    cand_corrs[cand] = round(np.corrcoef(sv[valid], y[valid])[0, 1], 3)
                w = df[sc].fillna(0).values * pop
                if w.sum() > 0:
                    cand_bary[cand] = round(np.average(sv, weights=w), 2)

        # Max |r| with existing scores
        max_r_exist = 0
        for en in EXISTING_SCORES_CONFIG:
            if en in df.columns:
                valid = np.isfinite(sv) & np.isfinite(df[en].values)
                if valid.sum() > 100:
                    r_val = abs(np.corrcoef(sv[valid], df[en].values[valid])[0, 1])
                    max_r_exist = max(max_r_exist, r_val)

        fiche = {
            'score': name,
            'description': desc,
            'n_pos_vars': len(pos_vars),
            'n_neg_vars': len(neg_vars),
            'mean': round(np.nanmean(sv), 2),
            'std': round(np.nanstd(sv), 2),
            'range': f"[{np.nanmin(sv):.1f}, {np.nanmax(sv):.1f}]",
            'max_r_with_existing': round(max_r_exist, 3),
            'novel': max_r_exist < 0.5,
        }
        for cand in CANDIDATES:
            fiche[f'r_{cand}'] = cand_corrs.get(cand, np.nan)
            fiche[f'bary_{cand}'] = cand_bary.get(cand, np.nan)
        fiches.append(fiche)

        # Text fiche
        text = f"\n{'='*60}\n"
        text += f"SCORE : {name}\n"
        text += f"{'='*60}\n"
        text += f"Description : {desc}\n"
        text += f"Variables positives ({len(pos_vars)}) : {', '.join(pos_vars)}\n"
        text += f"Variables négatives ({len(neg_vars)}) : {', '.join(neg_vars)}\n"
        text += f"Distribution : mean={fiche['mean']}, std={fiche['std']}, range={fiche['range']}\n"
        text += f"Orthogonalité : max|r| avec existants = {max_r_exist:.3f} ({'NOUVEAU' if max_r_exist < 0.5 else 'REDONDANT'})\n"
        text += f"Corrélations candidats : {cand_corrs}\n"
        text += f"Barycentres candidats : {cand_bary}\n"
        fiche_text.append(text)

    df_fiches = pd.DataFrame(fiches)
    df_fiches.to_csv(os.path.join(OUTPUT_DIR, '21_final_score_recommendations.csv'), index=False)

    # Text implementation guide
    novel_scores = df_fiches[df_fiches['novel']]
    guide_text = "GUIDE D'IMPLÉMENTATION — Scores recommandés pour rebuild_vizu_iris.py\n"
    guide_text += "=" * 70 + "\n\n"
    guide_text += f"Scores nouveaux (max|r| < 0.5 avec existants) : {len(novel_scores)}\n\n"
    for _, r in novel_scores.iterrows():
        name = r['score']
        cfg = all_proposed[name]
        guide_text += f"  '{name}': {{\n"
        guide_text += f"      'pos_vars': {cfg.get('pos_vars', [])},\n"
        guide_text += f"      'neg_vars': {cfg.get('neg_vars', [])},\n"
        guide_text += f"  }},\n\n"
    guide_text += "\n".join(fiche_text)

    with open(os.path.join(OUTPUT_DIR, '21_implementation_guide.txt'), 'w', encoding='utf-8') as f:
        f.write(guide_text)

    # Score fiches visualization (distributions + barycentres)
    n_scores = len(all_score_vals)
    ncols = 4
    nrows = (n_scores + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(20, 4 * nrows))
    axes = axes.flatten() if n_scores > 4 else [axes] if n_scores == 1 else axes.flatten()

    for i, (name, sv) in enumerate(all_score_vals.items()):
        if i >= len(axes):
            break
        ax = axes[i]
        ax.hist(sv[np.isfinite(sv)], bins=50, color='steelblue', alpha=0.5, density=True)

        # Mark barycentres
        for cand in ['MACRON', 'LE_PEN', 'MELENCHON', 'JADOT', 'ZEMMOUR']:
            sc_col = f'score_{cand}'
            if sc_col in df.columns:
                w = df[sc_col].fillna(0).values * pop
                if w.sum() > 0:
                    bary_val = np.average(sv, weights=w)
                    ax.axvline(bary_val, color=CANDIDATE_COLORS.get(cand, '#999'),
                               lw=2, label=cand)

        novel_flag = " (NOUVEAU)" if name in [r['score'] for _, r in novel_scores.iterrows()] else ""
        ax.set_title(f'{name}{novel_flag}', fontsize=8, fontweight='bold')
        ax.legend(fontsize=5, loc='upper right')
        ax.tick_params(labelsize=6)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle('Fiches scores — Distribution + barycentres candidats', fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, '21_score_fiches.png'), dpi=120, bbox_inches='tight')
    plt.close(fig)

    print(f"\n  Scores analysés : {len(fiches)}")
    print(f"  Scores nouveaux (max|r| < 0.5) : {len(novel_scores)}")
    for _, r in novel_scores.iterrows():
        print(f"    {r['score']:40s}  max|r|={r['max_r_with_existing']:.3f}")

    print(f"\n  → Fichiers : 21_final_score_recommendations.csv, 21_score_fiches.png, 21_implementation_guide.txt")
    return df_fiches


# ── 3.3 Synthesis Report ────────────────────────────────────────────────────

def synthesis_report(df, var_names, pca_results, lda_results, rf_importance,
                     wls_results, corr_by_cand):
    """Generate final synthesis report combining all methods."""
    print("\n" + "─" * 70)
    print("  3.3  RAPPORT DE SYNTHÈSE")
    print("─" * 70)

    pop = df['_pop'].values
    labels = get_var_labels()

    # ── Consensus importance across methods ──
    consensus_rows = []
    for v in var_names:
        row = {'variable': v, 'label': labels.get(v, v)}

        # PCA: max |loading| across retained PCs
        loadings = pca_results['loadings']
        if v in loadings.index:
            row['pca_max_loading'] = loadings.loc[v].abs().max()
        else:
            row['pca_max_loading'] = 0

        # LDA: max |coefficient| across LDs
        lda_coefs = lda_results['coefs']
        if v in lda_coefs.index:
            row['lda_max_coef'] = lda_coefs.loc[v].abs().max()
        else:
            row['lda_max_coef'] = 0

        # RF: permutation importance
        if rf_importance is not None and v in rf_importance['variable'].values:
            row['rf_perm_importance'] = rf_importance.loc[rf_importance['variable'] == v, 'perm_importance_mean'].values[0]
        else:
            row['rf_perm_importance'] = 0

        # WLS: max |beta| across candidates
        if wls_results:
            betas = [abs(res['betas'][list(var_names).index(v)]) for res in wls_results.values()
                     if v in var_names and list(var_names).index(v) < len(res['betas'])]
            row['wls_max_beta'] = max(betas) if betas else 0
        else:
            row['wls_max_beta'] = 0

        # Correlation: max |r| with any candidate
        if corr_by_cand is not None and v in corr_by_cand.index:
            row['max_corr_candidate'] = corr_by_cand.loc[v].abs().max()
        else:
            row['max_corr_candidate'] = 0

        consensus_rows.append(row)

    df_consensus = pd.DataFrame(consensus_rows)

    # Normalize each method to [0,1]
    for col in ['pca_max_loading', 'lda_max_coef', 'rf_perm_importance', 'wls_max_beta', 'max_corr_candidate']:
        mx = df_consensus[col].max()
        if mx > 0:
            df_consensus[f'{col}_norm'] = df_consensus[col] / mx
        else:
            df_consensus[f'{col}_norm'] = 0

    # Composite importance
    df_consensus['consensus_importance'] = (
        0.20 * df_consensus['pca_max_loading_norm'] +
        0.25 * df_consensus['lda_max_coef_norm'] +
        0.25 * df_consensus['rf_perm_importance_norm'] +
        0.20 * df_consensus['wls_max_beta_norm'] +
        0.10 * df_consensus['max_corr_candidate_norm']
    )
    df_consensus = df_consensus.sort_values('consensus_importance', ascending=False)
    df_consensus.to_csv(os.path.join(OUTPUT_DIR, '22_consensus_importance.csv'), index=False)

    print(f"\n  Top 20 variables par importance consensus multi-méthodes :")
    print(f"  {'Variable':35s}  {'PCA':>5s}  {'LDA':>5s}  {'RF':>5s}  {'WLS':>5s}  {'Corr':>5s}  {'Total':>6s}")
    print(f"  {'─'*35}  {'─'*5}  {'─'*5}  {'─'*5}  {'─'*5}  {'─'*5}  {'─'*6}")
    for _, r in df_consensus.head(20).iterrows():
        print(f"  {r['variable']:35s}  {r['pca_max_loading_norm']:.2f}  {r['lda_max_coef_norm']:.2f}  "
              f"{r['rf_perm_importance_norm']:.2f}  {r['wls_max_beta_norm']:.2f}  "
              f"{r['max_corr_candidate_norm']:.2f}  {r['consensus_importance']:.3f}")

    # ── Candidate signatures (top 5 per candidate) ──
    sig_rows = []
    for cand in CANDIDATES:
        if corr_by_cand is None or cand not in corr_by_cand.columns:
            continue
        corrs = corr_by_cand[cand].abs().sort_values(ascending=False)
        for v in corrs.head(5).index:
            sig_rows.append({
                'candidat': cand,
                'variable': v,
                'label': labels.get(v, v),
                'correlation': round(corr_by_cand.loc[v, cand], 4),
                'consensus_rank': int(df_consensus[df_consensus['variable'] == v].index[0]) + 1
                if v in df_consensus['variable'].values else 999,
            })

    df_sig = pd.DataFrame(sig_rows)
    df_sig.to_csv(os.path.join(OUTPUT_DIR, '22_candidate_signatures.csv'), index=False)

    print(f"\n  Signatures électorales (top 5 variables par candidat) :")
    for cand in CANDIDATES:
        cand_sig = df_sig[df_sig['candidat'] == cand]
        if cand_sig.empty:
            continue
        print(f"\n  {cand}:")
        for _, r in cand_sig.iterrows():
            print(f"    {r['variable']:35s}  r={r['correlation']:+.3f}  (rang consensus: #{r['consensus_rank']})")

    # ── Synthesis report text ──
    report = []
    report.append("RAPPORT DE SYNTHÈSE — Analyse IRIS Socio × Élections 2022")
    report.append("=" * 70)
    report.append(f"\nDataset : {df.shape[0]} IRIS × {len(var_names)} variables")
    report.append(f"PCA : {pca_results['n_retain']} composantes retenues "
                  f"({pca_results['explained_ratio'][:pca_results['n_retain']].sum()*100:.1f}% variance)")
    report.append(f"LDA : accuracy CV = {lda_results['cv_accuracy']:.1%}")

    report.append("\n\nCLIVAGES STRUCTURANTS IDENTIFIÉS :")
    report.append("-" * 40)
    report.append("1. Urbain/Rural (PC1) : oppose centres-villes denses (HLM, transports,")
    report.append("   immigrés, petits logements) aux zones pavillonnaires (maison, voiture,")
    report.append("   grands logements). C'est l'axe dominant de la variance socio.")
    report.append("2. Aisé/Précaire (PC2) : oppose quartiers aisés (cadres, BAC+5, revenus")
    report.append("   élevés) aux quartiers populaires (chômage, prestations sociales).")
    report.append("3. Gauche-populaire/Droite-bourgeoise : Mélenchon/quartiers immigrés et")
    report.append("   HLM vs Macron-Pécresse/quartiers cadres et patrimoine.")
    report.append("4. Périphérie/Métropole : Le Pen-Zemmour/zones pavillonnaires-automobiles")
    report.append("   vs Macron-Jadot/centres-villes connectés.")

    report.append("\n\nVARIABLES LES PLUS IMPORTANTES (consensus multi-méthodes) :")
    report.append("-" * 40)
    for _, r in df_consensus.head(15).iterrows():
        report.append(f"  {r['variable']:35s}  importance={r['consensus_importance']:.3f}")

    report_text = '\n'.join(report)
    with open(os.path.join(OUTPUT_DIR, '22_synthesis_report.txt'), 'w', encoding='utf-8') as f:
        f.write(report_text)

    print(f"\n  → Fichiers : 22_synthesis_report.txt, 22_consensus_importance.csv, 22_candidate_signatures.csv")
    return df_consensus


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 70)
    print("  ANALYSE IRIS — Socio-démographie × Présidentielles 2022")
    print("  (version enrichie — ~60 fichiers output)")
    print("=" * 70)

    # ── Data loading ──
    df, var_names, all_available = load_and_prepare()

    # ═══════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("  PARTIE 1 — ANALYSE AGNOSTIQUE AUX PARTIS")
    print("=" * 70)

    quality = variable_quality_assessment(df, var_names)
    corr_pearson, redundant_pairs, dendro_Z, corr_matrix = correlation_analysis(df, var_names)
    pca_results = weighted_pca_analysis(df, var_names)
    proposed_scores, scores_df = construct_composite_scores(pca_results, df, var_names)
    nonlinear_exploration(df, var_names, pca_results)
    norm_recs = normalization_analysis(df, var_names)

    # ── Extensions agnostiques ──
    umap_exploration(df, var_names, pca_results)
    km_results = clustering_analysis(df, var_names, pca_results)
    core_vars = redundancy_analysis(df, var_names, corr_matrix, redundant_pairs, pca_results)
    mi_results = mutual_information_analysis(df, var_names)
    icc_results = geographic_variance_analysis(df, var_names)

    # ═══════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("  PARTIE 2 — ANALYSE CONDITIONNÉE (Présidentielles 2022 T1)")
    print("=" * 70)

    profiles, z_scores = candidate_profiles(df, var_names)
    cond_pca_loadings = conditional_pca(df, var_names, pca_results)
    lda_results = discriminant_analysis(df, var_names)
    importance, corr_by_cand = variable_importance(df, var_names)
    party_scores, party_scores_df = party_informed_scores(
        df, var_names, lda_results, corr_by_cand)

    # ── Extensions conditionnées ──
    rf_importance = random_forest_analysis(df, var_names)
    pairwise_results = pairwise_contrasts(df, var_names)
    wls_results = wls_regression(df, var_names)
    abst_results = abstention_analysis(df, var_names)

    # ═══════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("  PARTIE 3 — SYNTHÈSE ET RECOMMANDATIONS")
    print("=" * 70)

    axis_pairs = recommend_axis_pairs(df, var_names, proposed_scores, party_scores,
                                       scores_df, party_scores_df)
    fiches = score_fiches(df, var_names, proposed_scores, party_scores,
                          scores_df, party_scores_df)
    consensus = synthesis_report(df, var_names, pca_results, lda_results,
                                 rf_importance, wls_results, corr_by_cand)

    # ═══════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("  SYNTHÈSE FINALE")
    print("=" * 70)

    print(f"\n  Dataset : {df.shape[0]} IRIS × {len(var_names)} variables")
    print(f"  PCA : {pca_results['n_retain']} composantes retenues "
          f"({pca_results['explained_ratio'][:pca_results['n_retain']].sum()*100:.1f}% variance)")
    print(f"  LDA : accuracy CV = {lda_results['cv_accuracy']:.1%}")
    print(f"  Scores PCA proposés : {len(proposed_scores)}")
    print(f"  Scores partis proposés : {len(party_scores)}")

    print(f"\n  Fichiers générés dans {OUTPUT_DIR}/ :")
    for f in sorted(os.listdir(OUTPUT_DIR)):
        size = os.path.getsize(os.path.join(OUTPUT_DIR, f))
        print(f"    {f} ({size / 1024:.1f} KB)")

    total_files = len(os.listdir(OUTPUT_DIR))
    print(f"\n  Total : {total_files} fichiers générés")

    print("\n" + "=" * 70)
    print("  ANALYSE TERMINÉE")
    print("=" * 70)


if __name__ == "__main__":
    main()
