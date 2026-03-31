# shared_config.py — Configuration et fonctions partagées entre desktop et mobile
import pandas as pd
import numpy as np
import json as _json
import os

# ── CONFIGURATION PARTIS ──────────────────────────────────────────────────────
COULEURS = {
    "RN":        "#374151",
    "LFI":       "#DC2626",
    "PS":        "#EC4899",
    "ENS":       "#F97316",
    "EELV":      "#16A34A",
    "PCF":       "#9B1C1C",
    "LR":        "#1D4ED8",
    "REC":       "#0F172A",
    "AUTRE":     "#9CA3AF",
}
LABELS = {
    "RN":            "Rassemblement National",
    "LFI":           "La France Insoumise",
    "PS":            "Parti Socialiste",
    "ENS":           "Ensemble / Renaissance",
    "EELV":          "Europe Écologie",
    "PCF":           "Parti Communiste",
    "LR":            "Les Républicains",
    "REC":           "Reconquête",
    "AUTRE":         "Autres partis",
    "NFP":           "Nouveau Front Populaire",
    "NUPES":         "NUPES",
    "PS_PP":         "PS-Place Publique (Glucksmann)",
    "UG":            "Union de la Gauche",
    "UXD":           "Alliance LR-RN (Ciotti)",
    "DVD":           "Divers droite",
    "DVC":           "Divers centre",
    "DVG":           "Divers gauche",
    "EXG":           "Extrême gauche",
    "EXD":           "Extrême droite",
    "DLF":           "Debout la France",
    "MODEM":         "MoDem",
    "HOR":           "Horizons",
    "UDI":           "UDI",
    "REG":           "Régionalistes",
    "MACRON":        "Emmanuel Macron",
    "LE_PEN":        "Marine Le Pen",
    "MELENCHON":     "Jean-Luc Mélenchon",
    "FILLON":        "François Fillon",
    "HAMON":         "Benoît Hamon",
    "DUPONT_AIGNAN": "Nicolas Dupont-Aignan",
    "ZEMMOUR":       "Éric Zemmour",
    "PECRESSE":      "Valérie Pécresse",
    "JADOT":         "Yannick Jadot",
    "ROUSSEL":       "Fabien Roussel",
    "HIDALGO":       "Anne Hidalgo",
    # Présidentielles 2012
    "HOLLANDE":      "François Hollande",
    "SARKOZY":       "Nicolas Sarkozy",
    "BAYROU":        "François Bayrou",
    "JOLY":          "Éva Joly",
}
SHORT = {
    "RN":            "RN",
    "LFI":           "LFI",
    "PS":            "PS",
    "ENS":           "ENS",
    "EELV":          "EELV",
    "PCF":           "PCF",
    "LR":            "LR",
    "REC":           "RCQ",
    "AUTRE":         "Autre",
    "NFP":           "NFP",
    "NUPES":         "NUPES",
    "PS_PP":         "PS-PP",
    "UG":            "UG",
    "UXD":           "UXD",
    "DVD":           "DVD",
    "DVC":           "DVC",
    "DVG":           "DVG",
    "EXG":           "EXG",
    "EXD":           "EXD",
    "DLF":           "DLF",
    "MODEM":         "MDM",
    "HOR":           "HOR",
    "UDI":           "UDI",
    "REG":           "REG",
    "MACRON":        "Macron",
    "LE_PEN":        "Le Pen",
    "MELENCHON":     "Mélenchon",
    "FILLON":        "Fillon",
    "HAMON":         "Hamon",
    "DUPONT_AIGNAN": "DPA",
    "ZEMMOUR":       "Zemmour",
    "PECRESSE":      "Pécresse",
    "JADOT":         "Jadot",
    "ROUSSEL":       "Roussel",
    "HIDALGO":       "Hidalgo",
    # Présidentielles 2012
    "HOLLANDE":      "Hollande",
    "SARKOZY":       "Sarkozy",
    "BAYROU":        "Bayrou",
    "JOLY":          "Joly",
}
# RN/REC/ZEMMOUR/LE_PEN rendered semi-transparent
OPACITY = {
    "RN": 0.50, "REC": 0.50, "LE_PEN": 0.50, "ZEMMOUR": 0.50,
    "EXD": 0.50, "UXD": 0.50,
}
ORDER = ["LFI","PCF","EELV","PS","ENS","LR","RN","REC","AUTRE"]
# ALL_ORDER : tous les partis/candidats possibles (pour créer une trace Plotly par parti)
ALL_ORDER = [
    "LFI","MELENCHON","PCF","ROUSSEL","EXG","EELV","JADOT","JOLY","PS","DVG","HAMON","HOLLANDE",
    "NFP","NUPES","PS_PP","UG",
    "ENS","MODEM","HOR","UDI","MACRON",
    "LR","DVD","DVC","FILLON","PECRESSE","SARKOZY","BAYROU","UXD",
    "RN","LE_PEN","REC","EXD","DLF","ZEMMOUR","DUPONT_AIGNAN",
    "REG",
    "HIDALGO",
    "AUTRE",
]

# ── SCORES CONFIG PCA ──────────────────────────────────────────────────────────
SCORES_CONFIG_GROUPED_PCA = {
    'score_domination': [
        {'poids': 1.0, 'vars': {
            'pct_csp_plus': +1.0, 'pct_csp_intermediaire': +0.6,
            'pct_csp_ouvrier': -0.8, 'pct_csp_employe': -0.6, 'pct_csp_sans_emploi': -1.0,

            'pct_sup5': +0.7, 'pct_bac_plus': +0.4,
            'pct_sans_diplome': -0.8, 'pct_capbep': -0.5,

            'pct_cdi': +0.8,
            'pct_interim': -0.9, 'pct_cdd': -0.7, 'pct_temps_partiel': -0.5,
        }}
    ],
    'score_exploitation': [
        {'poids': 0.75, 'vars': {
            'DISP_PPAT21': +1.5, 'DISP_PBEN21': +1.0, 'DISP_PCHO21': -0.5,
        }},
        {'poids': 0.25, 'vars': {
            'pct_proprietaires': +1.5, 'pct_employeurs': +0.8,
            'pct_csp_ouvrier': -0.8, 'pct_csp_employe': -0.5,
        }}
    ],
    'score_cap_eco': [
        {'poids': 0.7, 'vars': {
            'DISP_MED21': +1.0, 'DISP_PPAT21': +0.8, 'DISP_PBEN21': +0.5,
            'DISP_TP6021': -1.0, 'DISP_PPMINI21': -1.0, 'DISP_PPLOGT21': -0.6,
        }},
        {'poids': 0.3, 'vars': {
            'pct_proprietaires': +0.8, 'pct_hlm': -0.8, 'pct_suroccupation': -0.5,
        }}
    ],
    'score_cap_cult': [
        {'poids': 1.0, 'vars': {
            'pct_sup5': +1.0, 'pct_bac_plus': +0.6, 'pct_csp_intermediaire': +0.5,
            'pct_actifs_velo': +0.3,
            'pct_sans_diplome': -1.0, 'pct_capbep': -0.6,
            'pct_csp_agriculteur': -0.3,
        }},
    ],
    'score_precarite': [
        {'poids': 0.6, 'vars': {
            'DISP_TP6021': +1.0, 'DISP_PPMINI21': +0.8, 'DISP_PPSOC21': +0.5,
            'DISP_PPLOGT21': +0.4, 'DISP_MED21': -1.0,
        }},
        {'poids': 0.4, 'vars': {
            'pct_csp_sans_emploi': +0.8, 'pct_chomage': +0.7, 'pct_interim': +0.5,
            'pct_cdd': +0.4, 'pct_temps_partiel': +0.3, 'pct_hlm': +0.4, 'pct_suroccupation': +0.5,
        }}
    ],
    'score_rentier': [
        {'poids': 1.0, 'vars': {
            'DISP_PPAT21': +1.0, 'DISP_PPEN21': +0.5,
            'DISP_PTSA21': -1.0, 'DISP_PACT21': -0.6,
        }}
    ],
    'score_urbanite': [
        {'poids': 0.6, 'vars': {
            'pct_appart': +1.0, 'pct_voiture_0': +0.9, 'pct_actifs_transports': +0.8,
            'pct_locataires': +0.7, 'pct_petits_logements': +0.6,
            'pct_actifs_velo': +0.5, 'pct_actifs_marche': +0.4,
            'pct_maison': -1.0, 'pct_voiture_2plus': -0.9, 'pct_garage': -0.7,
            'surface_moyenne': -0.6, 'pct_chauffage_fioul': -0.4,
            'pct_csp_agriculteur': -0.7, 'P21_ACTOCC15P_ILT3': -0.5,
        }},
        {'poids': 0.4, 'vars': {
            'bpe_total_pour1000': +0.7, 'bpe_B_commerces_pour1000': +0.4,
        }}
    ],
    'score_confort_residentiel': [
        {'poids': 1.0, 'vars': {
            'pct_proprietaires': +0.7, 'pct_grands_logements': +0.8,
            'surface_moyenne': +0.7, 'nb_pieces_moyen': +0.6, 'pct_garage': +0.5,
            'pct_logements_5p_plus': +0.5,
            'pct_suroccupation': -1.0, 'pct_hlm': -0.7, 'pct_petits_logements': -0.6,
            'pct_studios': -0.5,
        }}
    ],
    'score_equipement_public': [
        {'poids': 1.0, 'vars': {
            'bpe_D_sante_pour1000': +1.0, 'bpe_C_enseignement_pour1000': +0.8,
            'bpe_A_services_pour1000': +0.7,
            'bpe_F_sports_culture_pour1000': +0.6, 'bpe_B_commerces_pour1000': +0.5,
        }}
    ],
    'score_dependance_carbone': [
        {'poids': 0.55, 'vars': {
            'pct_actifs_voiture': +1.0, 'pct_voiture_2plus': +0.8,
            'pct_actifs_transports': -0.9, 'pct_actifs_velo': -0.7, 'pct_actifs_marche': -0.5,
        }},
        {'poids': 0.45, 'vars': {
            'pct_chauffage_fioul': +1.0, 'pct_chauffage_gaz_bouteille': +0.6,
            'pct_logements_recents': -0.6, 'pct_chauffage_elec': -0.5,
        }}
    ],
}

SCORES_PCA_ANCHORS = {
    'score_domination':           'pct_csp_plus',
    'score_exploitation':         'DISP_PPAT21',
    'score_cap_eco':              'DISP_MED21',
    'score_cap_cult':             'pct_sup5',
    'score_precarite':            'DISP_TP6021',
    'score_rentier':              'DISP_PPAT21',
    'score_urbanite':             'pct_appart',
    'score_confort_residentiel':  'surface_moyenne',
    'score_equipement_public':    'bpe_D_sante_pour1000',
    'score_dependance_carbone':   'pct_actifs_voiture',
}

# ── FONCTIONS SCORES ──────────────────────────────────────────────────────────

def _rang_pondere(series, pop):
    """Centile pondéré par population, centré à 0 (range ≈ -50 à +50)."""
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
    # Remplir les NaN par la médiane (50) puis centrer à 0
    return result.fillna(50.0) - 50.0


def _zscore_pondere(series, pop):
    """
    Zscore pondéré par population.
    Chaque valeur est centrée/réduite par rapport à la moyenne et l'écart-type
    pondérés par la population de chaque IRIS. NaN remplacés par 0.
    """
    s = series.copy().astype(float)
    p = pop.copy().astype(float)
    valid = s.notna() & p.notna() & (p > 0)
    if valid.sum() < 10:
        return pd.Series(0.0, index=s.index)
    s_v = s[valid]
    p_v = p[valid]
    w = p_v / p_v.sum()
    w_mean = (s_v * w).sum()
    w_std = np.sqrt(((s_v - w_mean) ** 2 * w).sum())
    if w_std < 1e-10:
        return pd.Series(0.0, index=s.index)
    result = pd.Series(np.nan, index=s.index)
    result[valid] = (s_v - w_mean) / w_std
    return result.fillna(0.0)


def _pca_weighted_pc1(X, pop_array):
    """PCA pondérée population sur matrice X (n_iris × n_vars). Retourne (pc1_values, variance_explained_ratio)."""
    w = np.where(np.isfinite(pop_array) & (pop_array > 0), pop_array, 1.0)
    w_norm = w / w.sum()
    means = (X * w_norm[:, None]).sum(axis=0)
    Xc = X - means
    C = (Xc * w_norm[:, None]).T @ Xc
    eigenvalues, eigenvectors = np.linalg.eigh(C)
    idx = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    var_exp = eigenvalues[0] / eigenvalues.sum() if eigenvalues.sum() > 1e-10 else 0.0
    pc1 = Xc @ eigenvectors[:, 0]
    return pc1, var_exp


def make_score_grouped(df, groupes):
    """
    Score composite en trois étapes :
      1. Pour chaque groupe : zscore pondéré population de chaque variable,
         puis somme pondérée par les poids théoriques → indice composite du groupe
      2. Rang centile pondéré population de chaque indice composite de groupe
      3. Moyenne pondérée des rangs centiles des groupes (poids entre groupes)

    Paramètre `groupes` : liste de dicts {'vars': {var: poids}, 'poids': float}
    Range résultant : ~ [-50, +50].
    """
    pop = df['_pop']
    rangs_groupes = []
    poids_groupes = []

    for groupe in groupes:
        var_poids = groupe['vars']
        poids_groupe = groupe['poids']

        parts = []
        total_poids = sum(abs(p) for p in var_poids.values())
        if total_poids == 0:
            continue
        for var, poids in var_poids.items():
            if var not in df.columns:
                continue
            z = _zscore_pondere(df[var], pop)
            parts.append((poids / total_poids) * z)

        if not parts:
            continue

        indice_groupe = pd.concat(parts, axis=1).sum(axis=1)
        rangs_groupes.append(_rang_pondere(indice_groupe, pop))
        poids_groupes.append(poids_groupe)

    if not rangs_groupes:
        return pd.Series(0.0, index=df.index)

    total_poids_groupes = sum(poids_groupes)
    score_final = sum(r * p for r, p in zip(rangs_groupes, poids_groupes))
    return score_final / total_poids_groupes


LOG_TRANSFORM_VARS = {'DISP_MED21', 'surface_moyenne'}  # valeurs absolues → log avant zscore


def make_score_pca_grouped(df, groupes, anchor_var=None, min_variance_explained=0.35):
    """
    Score composite par PCA à deux étages, même structure de groupes que make_score_grouped.

    Étage 1 : pour chaque groupe, PCA pondérée sur les zscores des variables → PC1.
    Étage 2 : PCA pondérée sur les PC1 des groupes (zscores) → PC1 = score final.
    Signe corrigé par corrélation avec anchor_var.
    Normalisation finale : _rang_pondere → range [-50, +50].

    Retourne (score_series, diagnostics_dict).
    """
    pop = df['_pop']
    group_scores = []
    diags = []

    for i, groupe in enumerate(groupes):
        vars_disponibles = [v for v in groupe['vars'] if v in df.columns]
        if not vars_disponibles:
            continue

        X_parts = []
        for var in vars_disponibles:
            s = df[var].copy().astype(float)
            if var in LOG_TRANSFORM_VARS:
                s = s.clip(lower=1e-6).apply(np.log)
            z = _zscore_pondere(s, pop)
            X_parts.append(z.values)

        X = np.column_stack(X_parts)
        pc1, var_exp = _pca_weighted_pc1(X, pop.values)
        diags.append({'groupe': i, 'n_vars': len(vars_disponibles), 'var_exp': var_exp})

        if var_exp < min_variance_explained:
            print(f"  [WARN] Groupe {i}: PC1 explique seulement {var_exp*100:.1f}% de variance")

        group_scores.append(pd.Series(pc1, index=df.index))

    if not group_scores:
        return pd.Series(0.0, index=df.index), {}

    if len(group_scores) == 1:
        final_pc1 = group_scores[0].values
        final_var_exp = diags[0]['var_exp']
    else:
        G = np.column_stack([_zscore_pondere(s, pop).values for s in group_scores])
        final_pc1, final_var_exp = _pca_weighted_pc1(G, pop.values)
        if final_var_exp < min_variance_explained:
            print(f"  [WARN] Score final: PC1 explique seulement {final_var_exp*100:.1f}% de variance")

    score = pd.Series(final_pc1, index=df.index)

    if anchor_var and anchor_var in df.columns:
        if score.corr(df[anchor_var]) < 0:
            score = -score

    return _rang_pondere(score, pop), {'group_diags': diags, 'final_var_exp': final_var_exp}


# ── Variables pour la vraie ACP pondérée ──────────────────────────────────────
_EMBEDDING_VARS_PCA = [
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


def compute_pca_vraie(df, n_components=8):
    """
    Calcule les vraies composantes ACP pondérées par population sur df.
    Produit les colonnes df['score_pca_1'] .. df['score_pca_{n_components}'].
    Modifie df en place.
    """
    var_names = [v for v in _EMBEDDING_VARS_PCA if v in df.columns]
    print(f"  ACP vraie : {len(var_names)} variables disponibles sur {len(_EMBEDDING_VARS_PCA)}")
    if len(var_names) < 5:
        print("  ACP vraie : pas assez de variables, score_pca_1..8 mis à 0")
        for k in range(1, n_components + 1):
            df[f'score_pca_{k}'] = 0.0
        return

    pop = df['_pop'].values.astype(float)
    pop = np.where(np.isfinite(pop) & (pop > 0), pop, 1.0)

    X = df[var_names].values.astype(float)
    for j in range(X.shape[1]):
        col = X[:, j]
        nans = ~np.isfinite(col)
        if nans.any():
            valid = np.isfinite(col)
            col[nans] = np.median(col[valid]) if valid.sum() > 0 else 0.0

    w_norm = pop / pop.sum()
    w_means = (X * w_norm[:, None]).sum(axis=0)
    X_c = X - w_means
    w_stds = np.sqrt((X_c ** 2 * w_norm[:, None]).sum(axis=0))
    w_stds[w_stds < 1e-10] = 1e-10
    X_std = X_c / w_stds

    C = (X_std * w_norm[:, None]).T @ X_std
    eigenvalues, eigenvectors = np.linalg.eigh(C)
    idx = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    var_explained = eigenvalues / eigenvalues.sum() * 100
    print("  Variance expliquée par composante :")
    for k in range(min(n_components, len(eigenvalues))):
        print(f"    ACP-{k+1}: {var_explained[k]:.1f}%  (cumulée: {var_explained[:k+1].sum():.1f}%)")

    X_pca = X_std @ eigenvectors[:, :n_components]
    pop_series = pd.Series(pop, index=df.index)
    for k in range(n_components):
        col_name = f'score_pca_{k+1}'
        s = pd.Series(X_pca[:, k], index=df.index)
        df[col_name] = _rang_pondere(s, pop_series)
        loadings = pd.Series(eigenvectors[:, k], index=var_names)
        top_pos = loadings.nlargest(3).index.tolist()
        top_neg = loadings.nsmallest(3).index.tolist()
        print(f"  {col_name}: + {top_pos} / − {top_neg}")


# ── PRESETS D'AXES ────────────────────────────────────────────────────────────
AXIS_PRESETS = [
    {
        'id': 'saint_graphique',
        'label': 'Domination × Exploitation',
        'emoji': '⚒️',
        'xVar': 'score_exploitation', 'xInvert': False,
        'yVar': 'score_domination',
        'xTitle': '← Exploité (prolétaire) ─── Position dans le rapport capital/travail ─── Exploiteur (bourgeois) →',
        'yTitle': '← Dominé ─── Domination sociale (Bourdieu) ─── Dominant →',
        'xRange': [-50, 50], 'yRange': [-50, 50],
        'corners': [
            {'pos': 'tl', 'text': 'ASCENSION<br>SOCIALE', 'color': '#E87070'},
            {'pos': 'tr', 'text': 'REPRODUCTION<br>DU CAPITAL', 'color': '#6B8FD4'},
            {'pos': 'bl', 'text': 'PROLÉTARIAT', 'color': '#60B87A'},
            {'pos': 'br', 'text': 'PETITE<br>BOURGEOISIE', 'color': '#C49A30'},
        ],
        'desc': {
            'title': 'Domination × Exploitation — Position de classe',
            'x': "<b>Axe X — Position dans le rapport capital/travail</b> : d'où vient le revenu de l'IRIS ? À droite : zones <em>exploiteuses</em> — revenus patrimoniaux, bénéfices, employeurs, cadres supérieurs. À gauche : zones <em>exploitées</em> — salaires, ouvriers, employés, sans diplôme, pauvreté.",
            'y': '<b>Axe Y — Domination sociale</b> : position dans la hiérarchie sociale totale combinant capital économique et culturel. En haut : dominants (revenus élevés, diplômés, cadres sup.). En bas : dominés (chômage, minimas sociaux, faibles revenus).',
            'quadrants': {
                'tr': '<b>Reproduction du capital</b> — Zones exploiteuses ET dominantes : beaux quartiers, arrondissements bourgeois.',
                'tl': "<b>Ascension sociale</b> — Zones exploitées mais dominantes : cadres salariés issus de milieux populaires.",
                'bl': '<b>Prolétariat</b> — Zones exploitées ET dominées : quartiers ouvriers, banlieues populaires.',
                'br': '<b>Petite bourgeoisie</b> — Zones exploiteuses mais dominées : petits commerçants, artisans propriétaires.',
            }
        }
    },
    {
        'id': 'bourdieu',
        'label': 'Bourdieu',
        'emoji': '🎓',
        'xVar': 'score_cap_eco', 'xInvert': False,
        'yVar': 'score_cap_cult',
        'xTitle': '← Pauvre ─── Capital Economique ─── Riche →',
        'yTitle': '← Peu diplômé · ouvrier ─── Capital culturel ─── Diplômé · cadre →',
        'xRange': [-50, 50], 'yRange': [-50, 50],
        'corners': [
            {'pos': 'tl', 'text': 'PAUVRE<br>DIPLOME', 'color': '#E87070'},
            {'pos': 'tr', 'text': 'RICHE<br>DIPLOME', 'color': '#6B8FD4'},
            {'pos': 'bl', 'text': 'PAUVRE<br>NON DIPLOME', 'color': '#60B87A'},
            {'pos': 'br', 'text': 'RICHE<br>NON DIPLOME', 'color': '#C49A30'},
        ],
        'desc': {
            'title': 'Espace bourdieusien — Capital économique × Capital culturel',
            'x': "<b>Axe X — Capital économique</b> : revenu médian, patrimoine (DISP_PPAT21), bénéfices (DISP_PBEN21), pensions, propriétaires, taille des logements. À droite : zones aisées. À gauche : zones pauvres (taux de pauvreté, minimas sociaux, faibles revenus d'activité).",
            'y': "<b>Axe Y — Capital culturel</b> : diplômes (BAC+5), % cadres supérieurs et professions intermédiaires, pratiques culturelles (vélo urbain). En haut : forte proportion de diplômés et cadres. En bas : zones peu qualifiées (sans diplôme, CAP-BEP, ouvriers).",
            'quadrants': {
                'tl': '<b>Intellectuels déclassés</b> — Diplômés mais aux revenus modestes : enseignants, chercheurs, travailleurs du secteur public.',
                'tr': '<b>Élite intégrée</b> — Riches ET diplômés : grandes écoles, professions libérales, hauts fonctionnaires.',
                'bl': '<b>Classe populaire</b> — Faibles capitaux économique et culturel : zones ouvrières, quartiers populaires.',
                'br': '<b>Bourgeoisie patrimoniale</b> — Aisés mais peu diplômés : artisans propriétaires, commerçants, rentiers.',
            }
        }
    },
    {
        'id': 'rentier',
        'label': 'Rentier',
        'emoji': '💰',
        'xVar': 'score_rentier', 'xInvert': False,
        'yVar': 'score_domination',
        'xTitle': '← Revenu du travail (salaires) ─── Rentier vs Travailleur ─── Revenu du capital →',
        'yTitle': '← Dominé ─── Domination sociale (Bourdieu) ─── Dominant →',
        'xRange': [-50, 50], 'yRange': [-50, 50],
        'corners': [
            {'pos': 'tl', 'text': 'INTELLECTUELS<br>FONCTIONNAIRES', 'color': '#E87070'},
            {'pos': 'tr', 'text': 'ÉLITE<br>RENTIÈRE', 'color': '#6B8FD4'},
            {'pos': 'bl', 'text': 'SALARIÉS<br>PRÉCAIRES', 'color': '#60B87A'},
            {'pos': 'br', 'text': 'PETITS<br>RENTIERS', 'color': '#C49A30'},
        ],
        'desc': {
            'title': 'Rente vs Travail × Domination sociale',
            'x': '<b>Axe X — Score rentier</b> : part du revenu tirée du capital (patrimoine, pensions, retraites) vs travail salarié. À droite : les rentiers. À gauche : les travailleurs.',
            'y': '<b>Axe Y — Domination sociale</b> : hiérarchie sociale totale combinant capital économique, culturel et position professionnelle. En haut : dominants (revenus élevés, diplômés, cadres sup.). En bas : dominés.',
            'quadrants': {
                'tl': '<b>Intellectuels / Fonctionnaires</b> — Dominants mais non-rentiers : hauts fonctionnaires, médecins.',
                'tr': '<b>Élite rentière</b> — Dominant ET rentier : vieille bourgeoisie, héritiers parisiens.',
                'bl': '<b>Salariés précaires</b> — Dominé ET non-rentier : classe laborieuse sans patrimoine.',
                'br': "<b>Petits rentiers</b> — Rentier mais pas dominant : retraités propriétaires provinciaux.",
            }
        }
    },
    {
        'id': 'demographie',
        'label': 'Démographie',
        'emoji': '👥',
        'xVar': 'pct_etrangers', 'xInvert': False,
        'yVar': 'age_moyen',
        'xTitle': "← Peu d'étrangers ─── % population étrangère ─── Beaucoup d'étrangers →",
        'yTitle': '← Jeune ─── Âge moyen ─── Vieux →',
        'xRange': [-1.0, 45.0], 'yRange': [25.0, 55.0],
        'corners': [
            {'pos': 'tl', 'text': 'VIEUX<br>NATIFS', 'color': '#6B7280'},
            {'pos': 'tr', 'text': 'VIEUX<br>IMMIGRÉS', 'color': '#9CA3AF'},
            {'pos': 'bl', 'text': 'JEUNES<br>NATIFS', 'color': '#3B82F6'},
            {'pos': 'br', 'text': 'JEUNES<br>IMMIGRÉS', 'color': '#EF4444'},
        ],
        'desc': {
            'title': 'Démographie — Âge × Origine',
            'x': "< b>Axe X — % population étrangère</b> : part des résidents de nationalité étrangère dans l'IRIS.",
            'y': '<b>Axe Y — Âge moyen</b> : âge moyen de la population résidente de l\'IRIS.',
            'quadrants': {
                'tl': '<b>Vieux natifs</b> — Zones âgées à faible immigration : France rurale profonde, littoral retraité.',
                'tr': '<b>Vieux immigrés</b> — Zones avec une immigration ancienne et installée.',
                'bl': '<b>Jeunes natifs</b> — Zones jeunes peu diversifiées : périurbain récent, villes moyennes.',
                'br': '<b>Jeunes immigrés</b> — Zones à forte immigration récente : banlieues denses, zones industrielles.',
            }
        }
    },
    {
        'id': 'ruralite',
        'label': 'Territoire × Précarité',
        'emoji': '🌾',
        'xVar': 'score_urbanite', 'xInvert': True,
        'yVar': 'score_precarite',
        'xTitle': '← Urbain dense (transports, apparts) ─── Axe territorial ─── Rural / pavillonnaire (voiture, maison) →',
        'yTitle': '← Sécurisé ─── Score précarité ─── Précaire →',
        'xRange': [-55.0, 55.0], 'yRange': [-55.0, 55.0],
        'corners': [
            {'pos': 'bl', 'text': 'URBAIN<br>SÉCURISÉ', 'color': '#6B8FD4'},
            {'pos': 'br', 'text': 'RURAL/PAVILLONNAIRE<br>SÉCURISÉ', 'color': '#059669'},
            {'pos': 'tl', 'text': 'URBAIN<br>PRÉCAIRE', 'color': '#E87070'},
            {'pos': 'tr', 'text': 'RURAL/PAVILLONNAIRE<br>PRÉCAIRE', 'color': '#C49A30'},
        ],
        'desc': {
            'title': 'Territoire × Précarité sociale',
            'x': '<b>Axe X — Score urbanité (inversé)</b> : fusion des anciens scores ruralité / urbanité / périphérie-métropole / France pavillonnaire. À droite : habitat pavillonnaire, voiture, maison, agriculteurs, navettes longues. À gauche : urbain dense, transports, appartements, services de proximité.',
            'y': '<b>Axe Y — Score précarité</b> : chômage, minimas sociaux, taux de pauvreté (haut = plus précaire).',
            'quadrants': {
                'tl': '<b>Urbain précaire</b> — Grands ensembles denses, chômage élevé, minimas sociaux.',
                'tr': '<b>Rural/périurbain précaire</b> — Zones pavillonnaires ou rurales en difficulté économique.',
                'bl': '<b>Urbain sécurisé</b> — Centre-ville dense, cadres, fonctionnaires, emploi stable.',
                'br': '<b>Rural/périurbain sécurisé</b> — Pavillonnaire aisé, retraités propriétaires, bourgs ruraux stables.',
            }
        }
    },
    {
        'id': 'urbanisme', 'label': 'Urbanisme', 'emoji': '\U0001f3d7\ufe0f',
        'xVar': 'score_urbanite', 'xInvert': False,
        'yVar': 'score_equipement_public',
        'xTitle': '\u2190 Rural / Pavillonnaire \u2500\u2500\u2500 Urbanit\u00e9 \u2500\u2500\u2500 Urbain dense \u2192',
        'yTitle': '\u2190 Sous-\u00e9quip\u00e9 \u2500\u2500\u2500 \u00c9quipement public \u2500\u2500\u2500 Bien \u00e9quip\u00e9 \u2192',
        'xRange': [-50, 50], 'yRange': [-50, 50],
        'corners': [
            {'pos': 'tl', 'text': 'RURAL<br>\u00c9QUIP\u00c9', 'color': '#6B8FD4'},
            {'pos': 'tr', 'text': 'URBAIN<br>\u00c9QUIP\u00c9', 'color': '#E87070'},
            {'pos': 'bl', 'text': 'RURAL<br>SOUS-\u00c9QUIP\u00c9', 'color': '#C49A30'},
            {'pos': 'br', 'text': 'URBAIN<br>SOUS-\u00c9QUIP\u00c9', 'color': '#60B87A'},
        ],
        'desc': {
            'title': 'Urbanisme \u2014 Urbanit\u00e9 \u00d7 \u00c9quipement public',
            'x': "<b>Axe X \u2014 Score d'urbanit\u00e9</b> : composite int\u00e9grant type de logement, mode de chauffage, motorisation et transports. Gauche = rural/pavillonnaire, droite = urbain dense.",
            'y': "<b>Axe Y \u2014 Score d'\u00e9quipement public</b> : composite sant\u00e9, enseignement, sport, commerces, services pour 1000 hab.",
            'quadrants': {
                'tl': "<b>Rural \u00e9quip\u00e9</b> \u2014 Bourgs-centres avec bons services malgr\u00e9 l'habitat pavillonnaire.",
                'tr': '<b>Urbain \u00e9quip\u00e9</b> \u2014 C\u0153urs de ville denses et bien dot\u00e9s.',
                'bl': '<b>Rural sous-\u00e9quip\u00e9</b> \u2014 P\u00e9riurbain \u00e9loign\u00e9, d\u00e9serts de services.',
                'br': '<b>Urbain sous-\u00e9quip\u00e9</b> \u2014 Quartiers denses type grands ensembles, peu de services de proximit\u00e9.',
            }
        }
    },
    {
        'id': 'confort', 'label': 'Confort r\u00e9sidentiel', 'emoji': '\U0001f3e0',
        'xVar': 'score_confort_residentiel', 'xInvert': False,
        'yVar': 'score_precarite',
        'xTitle': '\u2190 Parc d\u00e9grad\u00e9 \u2500\u2500\u2500 Confort r\u00e9sidentiel \u2500\u2500\u2500 Parc confortable \u2192',
        'yTitle': '\u2190 Ais\u00e9 \u2500\u2500\u2500 Pr\u00e9carit\u00e9 \u2500\u2500\u2500 Pr\u00e9caire \u2192',
        'xRange': [-50, 50], 'yRange': [-50, 50],
        'corners': [
            {'pos': 'tl', 'text': 'D\u00c9GRAD\u00c9<br>PR\u00c9CAIRE', 'color': '#E87070'},
            {'pos': 'tr', 'text': 'CONFORTABLE<br>PR\u00c9CAIRE', 'color': '#C49A30'},
            {'pos': 'bl', 'text': 'D\u00c9GRAD\u00c9<br>AIS\u00c9', 'color': '#60B87A'},
            {'pos': 'br', 'text': 'CONFORTABLE<br>AIS\u00c9', 'color': '#6B8FD4'},
        ],
        'desc': {
            'title': 'Confort r\u00e9sidentiel \u00d7 Pr\u00e9carit\u00e9',
            'x': '<b>Axe X \u2014 Confort r\u00e9sidentiel</b> : composite propri\u00e9t\u00e9, surface, garage, pi\u00e8ces vs suroccupation, HLM, studios, vacance.',
            'y': '<b>Axe Y \u2014 Pr\u00e9carit\u00e9</b> : composite ch\u00f4mage, prestations sociales, taux de pauvret\u00e9 vs revenu m\u00e9dian.',
            'quadrants': {
                'tl': '<b>Parc d\u00e9grad\u00e9 & pr\u00e9caire</b> \u2014 HLM, suroccupation, ch\u00f4mage : quartiers populaires en difficult\u00e9.',
                'tr': '<b>Confortable & pr\u00e9caire</b> \u2014 Propri\u00e9taires modestes, p\u00e9riurbain peu cher mais d\u00e9pendant de la voiture.',
                'bl': '<b>D\u00e9grad\u00e9 & ais\u00e9</b> \u2014 Studios \u00e9tudiants, petits logements en centre-ville ais\u00e9.',
                'br': '<b>Confortable & ais\u00e9</b> \u2014 Quartiers r\u00e9sidentiels bourgeois, grandes propri\u00e9t\u00e9s.',
            }
        }
    },
    {
        'id': 'energie', 'label': 'Transition énergétique', 'emoji': '⚡',
        'xVar': 'score_dependance_carbone', 'xInvert': False,
        'yVar': 'score_urbanite',
        'xTitle': '← Faible dépendance fossile ─── Score dépendance carbone ─── Fort dépendance fossile →',
        'yTitle': '← Rural / pavillonnaire ─── Score urbanité ─── Urbain dense →',
        'xRange': [-50, 50], 'yRange': [-50, 50],
        'corners': [
            {'pos': 'tl', 'text': 'RURAL<br>SOBRE', 'color': '#059669'},
            {'pos': 'tr', 'text': 'PAVILLONNAIRE<br>CARBONÉ', 'color': '#DC2626'},
            {'pos': 'bl', 'text': 'URBAIN<br>SOBRE', 'color': '#1D4ED8'},
            {'pos': 'br', 'text': 'URBAIN<br>CARBONÉ', 'color': '#F59E0B'},
        ],
        'desc': {
            'title': 'Transition énergétique — Dépendance carbone × Urbanité',
            'x': "<b>Axe X — Score dépendance carbone</b> : composite mobilité fossile (voiture, 2+ voitures vs transports, vélo, marche) + chauffage fossile (fioul, gaz bouteille vs électrique, logements récents). À droite : forte dépendance aux énergies fossiles.",
            'y': "<b>Axe Y — Score urbanité</b> : densité urbaine, forme de l'habitat, accessibilité. En haut : urbain dense bien équipé. En bas : rural/pavillonnaire.",
            'quadrants': {
                'tr': '<b>Pavillonnaire carboné</b> — Zones périurbaines : maison, 2+ voitures, chauffage fioul. Profil le plus difficile à décarboner.',
                'tl': '<b>Rural sobre</b> — Zones rurales éloignées mais avec peu de chauffage fossile (bois, solaire) et moins de déplacements motorisés.',
                'bl': '<b>Urbain sobre</b> — Centres-villes denses : transports collectifs, vélo, logements récents ou rénovés. Profil le plus favorable à la transition.',
                'br': '<b>Urbain carboné</b> — Zones denses mais avec chauffage fioul ou fort usage voiture : habitat ancien non rénové.',
            }
        }
    },
    {
        'id': 'pca_vraie',
        'label': 'PCA',
        'emoji': '🔢',
        'xVar': 'score_pca_1', 'xInvert': False,
        'yVar': 'score_pca_2',
        'xTitle': '← Locatif dense (appartements, immigrés) ─── PCA 1 : Logement & confort ─── Propriétaire pavillonnaire →',
        'yTitle': '← Diplômé aisé ─── PCA 2 : Composition sociale ─── Ouvrier précaire →',
        'xRange': [-55, 55], 'yRange': [-55, 55],
        'corners': [
            {'pos': 'tl', 'text': 'HLM<br>PRÉCAIRE', 'color': '#DC2626'},
            {'pos': 'tr', 'text': 'PAVILLONNAIRE<br>OUVRIER', 'color': '#374151'},
            {'pos': 'bl', 'text': 'URBAIN<br>DIPLÔMÉ', 'color': '#F59E0B'},
            {'pos': 'br', 'text': 'RÉSIDENTIEL<br>BOURGEOIS', 'color': '#1D4ED8'},
        ],
        'desc': {
            'title': 'Vraie ACP pondérée — PC1 × PC2',
            'x': "<b>Axe X — ACP PC1 (vraie ACP, ~20% de variance)</b> : 1ère composante principale d'une vraie ACP pondérée par population, calculée sur ~80 variables socio-économiques. Oppose l'habitat pavillonnaire propriétaire à l'habitat collectif dense. <b>+</b> surface, 2+ voitures, propriétaires, nb pièces, maison. <b>−</b> locataires, appartements, sans voiture, immigrés, petits logements.",
            'y': "<b>Axe Y — ACP PC2 (vraie ACP, ~12% de variance)</b> : 2ème composante principale, orthogonale à PC1. Sépare les zones ouvrières précaires des zones de cadres diplômés. <b>+</b> impôts faibles, sans diplôme, allocations familiales, prestations sociales, minimas sociaux, DISP_PPMINI21. <b>−</b> BAC+, revenu médian, cadres sup, BAC+5, revenus d'activité.",
            'quadrants': {
                'tr': '<b>Pavillonnaire ouvrier</b> — Lotissements périurbains, population peu qualifiée.',
                'tl': '<b>HLM précaire</b> — Grands ensembles denses avec forte précarité sociale.',
                'br': '<b>Résidentiel bourgeois</b> — Grandes propriétés, cadres supérieurs.',
                'bl': '<b>Urbain diplômé</b> — Centres-villes, jeunes actifs diplômés, locataires.',
            }
        }
    },
    {
        'id': 'tsne',
        'label': 't-SNE',
        'emoji': '🔬',
        'xVar': 'tsne_x', 'xInvert': False,
        'yVar': 'tsne_y',
        'xTitle': 't-SNE dimension 1',
        'yTitle': 't-SNE dimension 2',
        'xRange': None, 'yRange': None,
        'corners': [],
        'desc': {
            'title': 't-SNE \u2014 Carte des similarit\u00e9s entre IRIS',
            'x': "<b>Technique : t-SNE</b> (t-distributed Stochastic Neighbor Embedding) : algorithme de r\u00e9duction non-lin\u00e9aire qui projette les ~80 variables socio-\u00e9conomiques de chaque IRIS en 2 dimensions. Contrairement \u00e0 l'ACP (qui cherche les axes de plus grande <em>variance globale</em>), le t-SNE optimise la pr\u00e9servation des <em>voisinages locaux</em> : il place proches sur la carte les IRIS qui se ressemblent, m\u00eame si leurs profils ne varient pas beaucoup \u00e0 l'\u00e9chelle nationale. Param\u00e8tres utilis\u00e9s : perplexit\u00e9=30, 1000 it\u00e9rations, initialisation par ACP \u00e0 20 composantes.",
            'y': "<b>Comment lire cette carte</b> : les axes X et Y n'ont aucune signification propre (on ne peut pas dire \u00ab plus \u00e0 droite = plus riche \u00bb). Ce qui compte, ce sont les <em>regroupements visuels</em> : un amas compact = un type de territoire sociologiquement coh\u00e9rent. Les couleurs des partis dominants r\u00e9v\u00e8lent comment le vote s'organise dans cet espace. Limite : le t-SNE ne pr\u00e9serve pas bien les distances entre groupes \u00e9loign\u00e9s \u2014 voir UMAP pour \u00e7a.",
            'quadrants': {}
        }
    },
    {
        'id': 'umap',
        'label': 'UMAP',
        'emoji': '🌐',
        'xVar': 'umap_x', 'xInvert': False,
        'yVar': 'umap_y',
        'xTitle': 'UMAP dimension 1',
        'yTitle': 'UMAP dimension 2',
        'xRange': None, 'yRange': None,
        'corners': [],
        'desc': {
            'title': 'UMAP \u2014 Topologie des territoires fran\u00e7ais',
            'x': "<b>Technique : UMAP</b> (Uniform Manifold Approximation and Projection) : comme le t-SNE, l'UMAP projette les ~80 variables socio-\u00e9conomiques en 2D, mais avec deux avantages : il pr\u00e9serve \u00e0 la fois la structure <em>locale</em> (les IRIS similaires sont proches) ET <em>globale</em> (les distances entre groupes \u00e9loign\u00e9s restent interpr\u00e9tables). Deux amas s\u00e9par\u00e9s sur cette carte correspondent \u00e0 des types de territoires v\u00e9ritablement diff\u00e9rents. Param\u00e8tres : n_neighbors=15, min_dist=0.1, m\u00e9trique euclidienne apr\u00e8s r\u00e9duction PCA \u00e0 20 composantes.",
            'y': "<b>Comment lire cette carte</b> : les axes X et Y n'ont aucune signification propre. L'int\u00e9r\u00eat est dans la <em>topologie</em> : la forme des amas, leur s\u00e9paration, les ponts entre types de territoires. L'UMAP tend \u00e0 produire des groupes plus nets et plus s\u00e9par\u00e9s que le t-SNE, ce qui facilite l'identification des grands types de territoires fran\u00e7ais et de leur vote. Les couleurs r\u00e9v\u00e8lent comment l'espace sociologique se partitionne \u00e9lectoralement.",
            'quadrants': {}
        }
    },
]
