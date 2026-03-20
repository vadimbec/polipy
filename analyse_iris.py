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
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler

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

    # ── Distribution plots ──
    n_vars = len(var_names)
    ncols = 6
    nrows = (n_vars + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(24, 3 * nrows))
    axes = axes.flatten()
    for i, v in enumerate(var_names):
        ax = axes[i]
        xv = df[v].dropna().values
        ax.hist(xv, bins=50, color='steelblue', alpha=0.7, density=True)
        ax.set_title(v, fontsize=7, fontweight='bold')
        ax.tick_params(labelsize=5)
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)
    fig.suptitle('Distributions de toutes les variables', fontsize=14, y=1.0)
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

    # ── Heatmap ──
    fig, ax = plt.subplots(figsize=(20, 18))
    mask = np.triu(np.ones_like(corr_p, dtype=bool), k=1)
    sns.heatmap(df_corr_p, mask=mask, cmap='RdBu_r', center=0, vmin=-1, vmax=1,
                ax=ax, square=True, linewidths=0.1,
                xticklabels=True, yticklabels=True,
                cbar_kws={'shrink': 0.6})
    ax.tick_params(labelsize=5)
    ax.set_title('Matrice de corrélation Pearson pondérée (population)', fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, '02_corr_heatmap.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)

    # ── Dendrogram ──
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

    # ── Biplot PC1 vs PC2 ──
    fig, ax = plt.subplots(figsize=(14, 12))

    # Subsample for scatter
    n_plot = min(5000, len(df))
    rng = np.random.RandomState(RANDOM_SEED)
    idx_plot = rng.choice(len(df), n_plot, replace=False)
    ax.scatter(scores[idx_plot, 0], scores[idx_plot, 1],
               c='lightgray', s=2, alpha=0.3, rasterized=True)

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
    ax.set_title('Biplot PCA pondérée — PC1 × PC2')
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

    fig, ax = plt.subplots(figsize=(14, 16))
    z_plot = df_z.loc[top_vars, cand_order]
    sns.heatmap(z_plot, cmap='RdBu_r', center=0, vmin=-3, vmax=3,
                ax=ax, linewidths=0.5, annot=True, fmt='.1f', annot_kws={'size': 6})
    ax.set_title('Profils candidats — Z-scores vs moyenne nationale\n(top 40 variables les plus discriminantes)',
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

    # ── Scatter LD1 × LD2 ──
    fig, ax = plt.subplots(figsize=(12, 10))
    for cls in valid_classes:
        mask_cls = y_lda == cls
        ax.scatter(X_lda_proj[mask_cls, 0], X_lda_proj[mask_cls, 1],
                   c=CANDIDATE_COLORS.get(cls, '#999'), s=3, alpha=0.3,
                   label=cls, rasterized=True)
    ax.legend(fontsize=9, markerscale=4)
    ld1_pct = f" ({ev_ratio[0]*100:.1f}%)" if ev_ratio is not None else ""
    ld2_pct = f" ({ev_ratio[1]*100:.1f}%)" if ev_ratio is not None and len(ev_ratio) > 1 else ""
    ax.set_xlabel(f'LD1{ld1_pct}')
    ax.set_ylabel(f'LD2{ld2_pct}')
    ax.set_title(f'LDA — Présidentielles 2022 T1\n(accuracy CV: {cv_scores.mean():.1%})')
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
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 70)
    print("  ANALYSE IRIS — Socio-démographie × Présidentielles 2022")
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

    # ═══════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("  SYNTHÈSE")
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

    print("\n" + "=" * 70)
    print("  ANALYSE TERMINÉE")
    print("=" * 70)


if __name__ == "__main__":
    main()
