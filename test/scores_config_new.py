"""
scores_config_new.py
Fonctions standalone pour tester les nouvelles configs de scores avant
implémentation dans build_iris_final.py.

Usage depuis le notebook :
    from scores_config_new import make_score_v2, SCORES_CONFIG_NEW, SCORES_CONFIG_OLD, compute_pca_scores, EMBEDDING_VARS
"""

import pandas as pd
import numpy as np


# =============================================================================
# Helpers
# =============================================================================

def _rang_pondere(series, pop):
    """Centile pondéré par population, centré à 0 (range ~ -50 à +50)."""
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


# =============================================================================
# make_score V1 (ancienne version — pour comparaison)
# =============================================================================

def make_score_v1(df, pos_vars, neg_vars, pop_col='pop_totale'):
    """Score composite par rang centile pondéré — version originale (moyenne simple)."""
    parts = []
    pop = df[pop_col]
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


# =============================================================================
# make_score V2 (nouvelle version — débiaisage par cluster de corrélation)
# =============================================================================

def make_score_v2(df, pos_vars, neg_vars, pop_col='pop_totale',
                  corr_threshold=0.75, score_name=None, verbose=True):
    """
    Score composite par rang centile pondéré par population.
    Les variables fortement corrélées (> corr_threshold) sont regroupées
    en clusters (liaison complète); chaque variable reçoit un poids
    1/taille_cluster pour éviter de surpondérer les dimensions redondantes.
    """
    all_vars  = [(v, +1) for v in pos_vars  if v in df.columns]
    all_vars += [(v, -1) for v in neg_vars  if v in df.columns]
    if not all_vars:
        return pd.Series(0.0, index=df.index)

    names = [v for v, _ in all_vars]
    signs = [s for _, s in all_vars]
    pop   = df[pop_col]

    # Matrice de corrélation (imputation médiane locale)
    data = df[names].copy()
    for col in names:
        data[col] = data[col].fillna(data[col].median())
    corr = data.corr().abs()

    # Clustering greedy (liaison complète intra-cluster)
    clusters: list[list[str]] = []
    assigned: dict[str, int] = {}
    for v in names:
        placed = False
        for c_idx, cluster in enumerate(clusters):
            if all(corr.loc[v, u] >= corr_threshold for u in cluster):
                cluster.append(v)
                assigned[v] = c_idx
                placed = True
                break
        if not placed:
            clusters.append([v])
            assigned[v] = len(clusters) - 1

    cluster_sizes = {v: len(clusters[assigned[v]]) for v in names}

    if verbose and score_name:
        multi = {v: cluster_sizes[v] for v in names if cluster_sizes[v] > 1}
        if multi:
            # Regrouper par cluster
            cluster_members = {}
            for v, c_idx in assigned.items():
                cluster_members.setdefault(c_idx, []).append(v)
            for c_idx, members in cluster_members.items():
                if len(members) > 1:
                    print(f"  [{score_name}] cluster {c_idx}: {members} (poids 1/{len(members)} chacune)")

    parts = []
    for v, sign in zip(names, signs):
        w = 1.0 / cluster_sizes[v]
        parts.append(sign * w * _rang_pondere(df[v], pop))

    total_weight = sum(1.0 / cluster_sizes[v] for v in names)
    return pd.concat(parts, axis=1).sum(axis=1) / total_weight


# =============================================================================
# SCORES_CONFIG_OLD — configuration actuelle (pour comparaison)
# =============================================================================

SCORES_CONFIG_OLD = {
    'score_exploitation': {
        'pos_vars': ['DISP_PPAT21', 'P21_NSAL15P_EMPLOY', 'pct_csp_plus', 'pct_csp_retraite',
                     'DISP_MED21', 'DISP_PPEN21', 'DISP_PBEN21'],
        'neg_vars': ['DISP_TP6021', 'DISP_PTSA21', 'P21_NSAL15P_AIDFAM', 'DISP_PPLOGT21'],
    },
    'score_domination': {
        'pos_vars': ['pct_csp_plus', 'pct_csp_intermediaire', 'pct_sup5', 'pct_cdi', 'P21_NSAL15P_EMPLOY'],
        'neg_vars': ['pct_csp_ouvrier', 'pct_csp_sans_emploi', 'pct_csp_employe', 'pct_csp_independant',
                     'pct_sans_diplome', 'pct_capbep', 'pct_cdd', 'pct_interim', 'pct_temps_partiel',
                     'pct_chomage', 'DISP_TP6021', 'DISP_PPSOC21', 'DISP_PPMINI21'],
    },
    'score_cap_cult': {
        'pos_vars': ['pct_csp_plus', 'pct_csp_intermediaire', 'pct_sup5', 'pct_actifs_velo'],
        'neg_vars': ['pct_csp_ouvrier', 'pct_sans_diplome', 'pct_csp_sans_emploi', 'pct_capbep',
                     'pct_interim', 'pct_temps_partiel', 'pct_chomage'],
    },
    'score_cap_eco': {
        'pos_vars': ['DISP_PPAT21', 'P21_NSAL15P_EMPLOY', 'pct_csp_plus', 'pct_csp_retraite',
                     'DISP_MED21', 'DISP_PPEN21', 'DISP_PBEN21'],
        'neg_vars': ['DISP_TP6021', 'DISP_PTSA21', 'P21_NSAL15P_AIDFAM', 'DISP_PPLOGT21', 'DISP_PCHO21'],
    },
    'score_precarite': {
        'pos_vars': ['DISP_TP6021', 'pct_csp_sans_emploi', 'DISP_PPSOC21', 'DISP_PPMINI21', 'pct_chomage'],
        'neg_vars': ['DISP_MED21', 'DISP_PPAT21'],
    },
    'score_rentier': {
        'pos_vars': ['DISP_PPAT21', 'DISP_PPEN21', 'pct_csp_retraite'],
        'neg_vars': ['DISP_PACT21', 'DISP_PPSOC21', 'pct_csp_employe'],
    },
    'score_ruralite': {
        'pos_vars': ['pct_csp_agriculteur', 'pct_sans_diplome', 'pct_actifs_voiture', 'P21_ACTOCC15P_ILT3'],
        'neg_vars': ['pct_immigres', 'pct_actifs_velo', 'pct_actifs_transports', 'pct_actifs_marche',
                     'pct_etudiants', 'P21_ACTOCC15P_ILT1'],
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
        'neg_vars': ['pct_suroccupation', 'pct_petits_logements', 'pct_hlm', 'pct_logvac', 'pct_studios'],
    },
    'score_equipement_public': {
        'pos_vars': ['bpe_total_pour1000', 'bpe_D_sante_pour1000',
                     'bpe_C_enseignement_pour1000', 'bpe_F_sports_culture_pour1000',
                     'bpe_A_services_pour1000', 'bpe_B_commerces_pour1000'],
        'neg_vars': [],
    },
    'score_peripherie_metropole': {
        'pos_vars': ['pct_capbep', 'pct_actifs_voiture', 'pct_maison',
                     'pct_voiture_2plus', 'nb_pieces_moyen', 'pct_chauffage_fioul'],
        'neg_vars': ['pct_bac_plus', 'pct_sup5', 'pct_csp_plus', 'DISP_GI21',
                     'DISP_PACT21', 'DISP_RD21', 'pct_actifs_velo', 'DISP_PTSA21',
                     'pct_studios', 'pct_petits_logements'],
    },
}


# =============================================================================
# SCORES_CONFIG_NEW — nouvelles configurations
# =============================================================================

SCORES_CONFIG_NEW = {

    # ── Axe X Saint-Graphique : composition du revenu (capital vs travail) ──
    'score_composition_capital': {
        'pos_vars': [
            'DISP_PPAT21',           # part revenus patrimoniaux
            'DISP_PBEN21',           # part bénéfices (indépendants)
            'P21_NSAL15P_EMPLOY',    # présence d'employeurs résidents
            'pct_proprietaires',     # stock de patrimoine immobilier
            'pct_logements_anciens', # patrimoine hérité (bâti <1919) — proxy urbain
        ],
        'neg_vars': [
            'DISP_PTSA21',           # part salaires — travail salarié pur
            'pct_csp_ouvrier',
            'pct_csp_employe',
            'pct_cdi',               # travail salarié stable
        ],
    },

    # ── Axe Y partagé Saint-Graphique + Bourdieu : position sociale totale ──
    'score_domination': {
        'pos_vars': [
            'DISP_MED21',
            'DISP_PPAT21',
            'pct_proprietaires',
            'pct_sup5',
            'pct_bac_plus',
            'pct_csp_plus',
            'pct_csp_intermediaire',
            'pct_cdi',
        ],
        'neg_vars': [
            'DISP_TP6021',
            'pct_sans_diplome',
            'pct_capbep',
            'pct_csp_ouvrier',
            'pct_csp_sans_emploi',
            'pct_chomage',
            'DISP_PPMINI21',
        ],
    },

    # ── Axe X Bourdieu : capital économique pur (sans diplôme/CSP) ──
    'score_cap_eco': {
        'pos_vars': [
            'DISP_MED21',
            'DISP_PPAT21',
            'pct_proprietaires',
            'surface_moyenne',
            'pct_grands_logements',
        ],
        'neg_vars': [
            'DISP_TP6021',
            'DISP_PPMINI21',
            'pct_hlm',
            'pct_suroccupation',
        ],
    },

    # ── Axe Y Bourdieu : capital culturel/scolaire (sans revenus) ──
    'score_cap_cult': {
        'pos_vars': [
            'pct_sup5',
            'pct_bac_plus',
            'pct_csp_intermediaire',
            'pct_actifs_velo',
            'bpe_ecole_privee_pour1000',
            'bpe_sport_indoor_pour1000',
        ],
        'neg_vars': [
            'pct_sans_diplome',
            'pct_capbep',
            'pct_csp_independant',
            'pct_csp_agriculteur',
            'pct_actifs_voiture',
        ],
    },

    # ── Inchangé ──
    'score_precarite': {
        'pos_vars': ['DISP_TP6021', 'pct_csp_sans_emploi', 'DISP_PPSOC21',
                     'DISP_PPMINI21', 'pct_chomage'],
        'neg_vars': ['DISP_MED21', 'DISP_PPAT21'],
    },

    # ── Modifié : ratio capital/travail pur ──
    'score_rentier': {
        'pos_vars': [
            'DISP_PPAT21',
            'DISP_PBEN21',
            'pct_proprietaires',
        ],
        'neg_vars': [
            'DISP_PTSA21',
            'pct_cdd',
            'pct_interim',
        ],
    },

    # ── Modifié : retrait pct_immigres ──
    'score_ruralite': {
        'pos_vars': [
            'pct_csp_agriculteur',
            'pct_sans_diplome',
            'pct_actifs_voiture',
            'P21_ACTOCC15P_ILT3',
        ],
        'neg_vars': [
            'pct_actifs_velo',
            'pct_actifs_transports',
            'pct_actifs_marche',
            'pct_etudiants',
            'P21_ACTOCC15P_ILT1',
            'bpe_total_pour1000',
        ],
    },

    # ── Inchangé ──
    'score_urbanite': {
        'pos_vars': ['pct_appart', 'pct_locataires', 'pct_petits_logements',
                     'pct_voiture_0', 'pct_chauffage_gaz_ville',
                     'bpe_E_transports_pour1000', 'bpe_total_pour1000'],
        'neg_vars': ['pct_maison', 'pct_voiture_2plus', 'pct_chauffage_fioul',
                     'pct_grands_logements', 'surface_moyenne', 'pct_garage'],
    },

    # ── Nouveau : zone de tension péri-urbaine ──
    'score_periurbain': {
        'pos_vars': [
            'pct_maison',
            'pct_actifs_voiture',
            'pct_voiture_2plus',
            'P21_ACTOCC15P_ILT3',
            'pct_chauffage_fioul',
            'pct_hlm',
        ],
        'neg_vars': [
            'bpe_E_transports_pour1000',
            'pct_actifs_transports',
            'pct_appart',
            'bpe_total_pour1000',
        ],
    },

    # ── Inchangé ──
    'score_confort_residentiel': {
        'pos_vars': ['pct_proprietaires', 'pct_grands_logements', 'surface_moyenne',
                     'pct_garage', 'nb_pieces_moyen', 'pct_logements_5p_plus'],
        'neg_vars': ['pct_suroccupation', 'pct_petits_logements', 'pct_hlm',
                     'pct_logvac', 'pct_studios'],
    },

    # ── Inchangé ──
    'score_equipement_public': {
        'pos_vars': ['bpe_total_pour1000', 'bpe_D_sante_pour1000',
                     'bpe_C_enseignement_pour1000', 'bpe_F_sports_culture_pour1000',
                     'bpe_A_services_pour1000', 'bpe_B_commerces_pour1000'],
        'neg_vars': [],
    },

    # ── Renommé depuis score_peripherie_metropole (variables inchangées) ──
    'score_france_pavillonnaire': {
        'pos_vars': ['pct_capbep', 'pct_actifs_voiture', 'pct_maison',
                     'pct_voiture_2plus', 'nb_pieces_moyen', 'pct_chauffage_fioul'],
        'neg_vars': ['pct_bac_plus', 'pct_sup5', 'pct_csp_plus', 'DISP_GI21',
                     'DISP_PACT21', 'DISP_RD21', 'pct_actifs_velo', 'DISP_PTSA21',
                     'pct_studios', 'pct_petits_logements'],
    },
}


# =============================================================================
# Calcul de tous les scores sur un dataframe
# =============================================================================

def compute_all_scores(df, config, score_fn, pop_col='pop_totale', verbose=True):
    """Calcule tous les scores d'une config sur df. Retourne df avec nouvelles colonnes."""
    df = df.copy()
    for score_name, cfg in config.items():
        if score_fn == make_score_v2:
            df[score_name] = score_fn(df, cfg['pos_vars'], cfg['neg_vars'],
                                      pop_col=pop_col, score_name=score_name, verbose=verbose)
        else:
            df[score_name] = score_fn(df, cfg['pos_vars'], cfg['neg_vars'], pop_col=pop_col)
        if verbose:
            s = df[score_name]
            print(f"  {score_name}: [{s.min():.1f}, {s.max():.1f}], mean={s.mean():.1f}")
    return df


# =============================================================================
# Vraie ACP pondérée par population
# =============================================================================

EMBEDDING_VARS = [
    'pct_etrangers', 'pct_immigres', 'age_moyen', 'pct_femmes',
    'taille_menage_moy', 'pct_hors_menage', 'ecart_csp_plus_hf',
    'pct_0_19', 'pct_20_64', 'pct_65_plus',
    'pct_csp_agriculteur', 'pct_csp_independant', 'pct_csp_plus',
    'pct_csp_intermediaire', 'pct_csp_employe', 'pct_csp_ouvrier',
    'pct_csp_retraite', 'pct_csp_sans_emploi',
    'DISP_MED21', 'DISP_TP6021', 'DISP_GI21', 'DISP_RD21', 'DISP_S80S2021',
    'DISP_PTSA21', 'DISP_PPAT21', 'DISP_PPEN21', 'DISP_PPSOC21',
    'DISP_PCHO21', 'DISP_PPFAM21', 'DISP_PPLOGT21', 'DISP_PPMINI21',
    'DISP_PIMPOT21', 'DISP_PACT21',
    'pct_sup5', 'pct_sans_diplome', 'pct_capbep', 'pct_bac_plus',
    'pct_chomage', 'pct_cdi', 'pct_cdd', 'pct_interim',
    'pct_temps_partiel', 'pct_inactif', 'pct_etudiants',
    'pct_actifs_voiture', 'pct_actifs_transports', 'pct_actifs_velo',
    'pct_actifs_2roues', 'pct_actifs_marche',
    'pct_proprietaires', 'pct_locataires', 'pct_hlm', 'pct_logvac',
    'pct_maison', 'pct_appart', 'pct_petits_logements', 'pct_grands_logements',
    'pct_logements_anciens', 'pct_logements_recents',
    'pct_voiture_0', 'pct_voiture_2plus', 'surface_moyenne', 'pct_suroccupation',
    'pct_chauffage_elec', 'pct_chauffage_fioul', 'pct_chauffage_gaz_ville',
    'pct_chauffage_gaz_bouteille', 'pct_chauffage_autre',
    'pct_garage', 'nb_pieces_moyen', 'pct_studios', 'pct_logements_5p_plus',
    'bpe_total_pour1000', 'bpe_A_services_pour1000', 'bpe_B_commerces_pour1000',
    'bpe_C_enseignement_pour1000', 'bpe_D_sante_pour1000',
    'bpe_E_transports_pour1000', 'bpe_F_sports_culture_pour1000',
    'bpe_G_tourisme_pour1000', 'bpe_educ_prioritaire_pour1000',
    'bpe_ecole_privee_pour1000', 'bpe_sport_indoor_pour1000',
    'pct_sport_accessible',
]


def compute_pca_scores(df, n_components=8, pop_col='pop_totale'):
    """
    Calcule les vraies composantes ACP pondérées par population.
    Retourne (df_with_scores, eigenvalues, eigenvectors, var_names).

    Les scores sont mis sur la même plage [-50, 50] que les autres scores
    via _rang_pondere().
    """
    var_names = [v for v in EMBEDDING_VARS if v in df.columns]
    print(f"  ACP : {len(var_names)} variables disponibles sur {len(EMBEDDING_VARS)}")

    pop = df[pop_col].values.astype(float)
    pop = np.where(np.isfinite(pop) & (pop > 0), pop, 1.0)

    X = df[var_names].values.astype(float)

    # Imputation médiane
    for j in range(X.shape[1]):
        col = X[:, j]
        nans = ~np.isfinite(col)
        if nans.any():
            valid = np.isfinite(col)
            col[nans] = np.median(col[valid]) if valid.sum() > 0 else 0.0

    # Standardisation pondérée par population
    w_norm = pop / pop.sum()
    w_means = (X * w_norm[:, None]).sum(axis=0)
    X_c = X - w_means
    w_stds = np.sqrt((X_c ** 2 * w_norm[:, None]).sum(axis=0))
    w_stds[w_stds < 1e-10] = 1e-10
    X_std = X_c / w_stds

    # Matrice de covariance pondérée + décomposition spectrale
    C = (X_std * w_norm[:, None]).T @ X_std
    eigenvalues, eigenvectors = np.linalg.eigh(C)
    idx = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    # Variance expliquée
    var_explained = eigenvalues / eigenvalues.sum() * 100
    print(f"  Variance expliquée par composante :")
    for k in range(min(n_components, len(eigenvalues))):
        print(f"    ACP-{k+1}: {var_explained[k]:.1f}%  (cumulée: {var_explained[:k+1].sum():.1f}%)")

    # Projections
    X_pca = X_std @ eigenvectors[:, :n_components]

    # Convertir en rang pondéré [-50, 50] pour cohérence avec make_score
    pop_series = pd.Series(pop, index=df.index)
    df = df.copy()
    for k in range(n_components):
        col_name = f'score_pca_{k+1}'
        s = pd.Series(X_pca[:, k], index=df.index)
        df[col_name] = _rang_pondere(s, pop_series)
        # Top variables qui chargent sur cet axe
        loadings = pd.Series(eigenvectors[:, k], index=var_names)
        top_pos = loadings.nlargest(5).index.tolist()
        top_neg = loadings.nsmallest(5).index.tolist()
        print(f"  {col_name}: + {top_pos}")
        print(f"  {col_name}: - {top_neg}")

    return df, eigenvalues, eigenvectors, var_names
