"""
utils_election.py
Helpers pour charger les données électorales par IRIS et les fusionner
avec le DataFrame socio. Adapté de _load_election_iris_data() dans rebuild_vizu_iris.py.
"""

import pandas as pd
import numpy as np
import os

# ── Couleurs des partis (copié de rebuild_vizu_iris.py) ──────────────────────
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
    "NFP":       "#8B0000",
    "NUPES":     "#7C3AED",
    "PS_PP":     "#DB2777",
    "UG":        "#6D28D9",
    "UXD":       "#1E3A5F",
    "DVD":       "#3B82F6",
    "DVC":       "#F59E0B",
    "DVG":       "#F97316",
    "EXG":       "#991B1B",
    "EXD":       "#111827",
    "DLF":       "#1E40AF",
    "MODEM":     "#F97316",
    "HOR":       "#FB923C",
    "UDI":       "#60A5FA",
    "REG":       "#10B981",
    "MACRON":    "#F97316",
    "LE_PEN":    "#374151",
    "MELENCHON": "#DC2626",
    "FILLON":    "#1D4ED8",
    "HAMON":     "#EC4899",
    "DUPONT_AIGNAN": "#4B5563",
    "ZEMMOUR":   "#0F172A",
    "PECRESSE":  "#2563EB",
    "JADOT":     "#16A34A",
    "ROUSSEL":   "#9B1C1C",
    "HIDALGO":   "#BE185D",
    "HOLLANDE":  "#EC4899",
    "SARKOZY":   "#1D4ED8",
    "BAYROU":    "#F59E0B",
    "JOLY":      "#16A34A",
}

LABELS = {
    "RN":        "Rassemblement National",
    "LFI":       "La France Insoumise",
    "PS":        "Parti Socialiste",
    "ENS":       "Ensemble / Renaissance",
    "EELV":      "Europe Écologie",
    "PCF":       "Parti Communiste",
    "LR":        "Les Républicains",
    "REC":       "Reconquête",
    "AUTRE":     "Autres partis",
    "NFP":       "Nouveau Front Populaire",
    "NUPES":     "NUPES",
    "PS_PP":     "PS-Place Publique",
    "UG":        "Union de la Gauche",
    "UXD":       "Alliance LR-RN",
    "DVD":       "Divers droite",
    "DVC":       "Divers centre",
    "DVG":       "Divers gauche",
    "EXG":       "Extrême gauche",
    "EXD":       "Extrême droite",
    "DLF":       "Debout la France",
    "MODEM":     "MoDem",
    "HOR":       "Horizons",
    "UDI":       "UDI",
    "REG":       "Régionalistes",
    "MACRON":    "Emmanuel Macron",
    "LE_PEN":    "Marine Le Pen",
    "MELENCHON": "Jean-Luc Mélenchon",
    "FILLON":    "François Fillon",
    "HAMON":     "Benoît Hamon",
    "DUPONT_AIGNAN": "Nicolas Dupont-Aignan",
    "ZEMMOUR":   "Éric Zemmour",
    "PECRESSE":  "Valérie Pécresse",
    "JADOT":     "Yannick Jadot",
    "ROUSSEL":   "Fabien Roussel",
    "HIDALGO":   "Anne Hidalgo",
    "HOLLANDE":  "François Hollande",
    "SARKOZY":   "Nicolas Sarkozy",
    "BAYROU":    "François Bayrou",
    "JOLY":      "Éva Joly",
}

ORDER = ["LFI", "PCF", "EELV", "PS", "ENS", "LR", "RN", "REC", "AUTRE"]

ELECTIONS_META = {
    '2022_pres_t1': 'Présidentielles 2022 — 1er tour',
    '2022_pres_t2': 'Présidentielles 2022 — 2e tour',
    '2024_legi_t1': 'Législatives 2024 — 1er tour',
    '2024_legi_t2': 'Législatives 2024 — 2e tour',
    '2024_euro_t1': 'Européennes 2024',
    '2022_legi_t1': 'Législatives 2022 — 1er tour',
    '2022_legi_t2': 'Législatives 2022 — 2e tour',
    '2017_pres_t1': 'Présidentielles 2017 — 1er tour',
    '2017_pres_t2': 'Présidentielles 2017 — 2e tour',
    '2019_euro_t1': 'Européennes 2019',
}


def load_election(election_id, df_iris, base_path='../iris/elections/'):
    """
    Charge un CSV élection et fusionne sur CODE_IRIS == df_iris.IRIS.

    Retourne un DataFrame avec les colonnes :
    - IRIS
    - parti_dominant : str (clé de COULEURS)
    - score_RN, score_LFI, ... : % des exprimés par parti
    - pct_abstention, inscrits, exprimes, ... (si disponibles)
    """
    path = os.path.join(base_path, f'{election_id}.csv')
    if not os.path.exists(path):
        raise FileNotFoundError(f"Fichier élection introuvable : {path}")

    elec = pd.read_csv(path, index_col='CODE_IRIS', dtype={'CODE_IRIS': str})

    score_cols = [c for c in elec.columns
                  if c.startswith('score_') and c not in ('score_blanc', 'score_nul')]
    if not score_cols:
        raise ValueError(f"Aucune colonne score_* dans {path}")

    extra_cols = [c for c in ['pct_abstention', 'inscrits', 'votants', 'exprimes', 'blancs', 'nuls']
                  if c in elec.columns]

    merged = df_iris[['IRIS']].merge(
        elec[score_cols + extra_cols],
        left_on='IRIS', right_index=True, how='left'
    )

    # Parti dominant
    scores_arr = merged[score_cols].fillna(0).values
    dominant_idx = scores_arr.argmax(axis=1)
    party_names = [c.replace('score_', '') for c in score_cols]
    merged['parti_dominant'] = [party_names[i] for i in dominant_idx]

    # Mettre à NaN les lignes où tous les scores sont 0 (IRIS sans données)
    all_zero = (scores_arr == 0).all(axis=1)
    merged.loc[all_zero, 'parti_dominant'] = None

    return merged


def weighted_corr(x, y, w):
    """Corrélation de Pearson pondérée."""
    mask = x.notna() & y.notna() & w.notna() & (w > 0)
    x, y, w = x[mask], y[mask], w[mask]
    if len(x) < 10:
        return np.nan
    w = w / w.sum()
    mx = (x * w).sum()
    my = (y * w).sum()
    cov = ((x - mx) * (y - my) * w).sum()
    sx = np.sqrt(((x - mx) ** 2 * w).sum())
    sy = np.sqrt(((y - my) ** 2 * w).sum())
    if sx < 1e-10 or sy < 1e-10:
        return np.nan
    return cov / (sx * sy)


def compute_score_party_corr(df, score_cols, election_df, party_cols, pop_col='pop_totale'):
    """
    Calcule la matrice de corrélations pondérées score × % vote parti.
    df : DataFrame IRIS avec scores et pop_totale
    score_cols : liste de noms de colonnes de scores
    election_df : DataFrame avec les score_* électoraux (% exprimés)
    party_cols : liste de colonnes score_* dans election_df
    Retourne DataFrame (scores en lignes, partis en colonnes).
    """
    combined = df[score_cols + [pop_col]].join(
        election_df[party_cols], how='inner'
    )
    pop = combined[pop_col]
    result = {}
    for sc in score_cols:
        result[sc] = {}
        for pc in party_cols:
            pname = pc.replace('score_', '')
            result[sc][pname] = weighted_corr(combined[sc], combined[pc], pop)
    return pd.DataFrame(result).T
