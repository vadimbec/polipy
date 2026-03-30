"""
Exploration script for improving score_domination.
Tests ~11 candidate configurations and produces quantitative + graphical analyses.

Run with: conda run -n vadim_env python test/domination_exploration.py
"""
import os, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['figure.dpi'] = 110
plt.rcParams['font.size'] = 10

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH    = os.path.join(SCRIPT_DIR, '..', 'iris', 'iris_final_socio_politique.csv')
ELECT_PATH   = os.path.join(SCRIPT_DIR, '..', 'iris', 'elections', '2022_pres_t2.csv')
OUT_DIR      = SCRIPT_DIR

# ─── 1. Load data ──────────────────────────────────────────────────────────────
print('Loading data...')
df = pd.read_csv(DATA_PATH, low_memory=False)
df['_pop'] = df['pop_totale'].fillna(df['pop_totale'].median())
print(f'  IRIS: {len(df)}  columns: {len(df.columns)}')

# ─── 2. Compute derived variables (same as _gen_notebook.py CELL_VARS) ─────────
_nscol = df['P21_NSCOL15P'].replace(0, float('nan'))
for col in ['pct_sup5', 'pct_sans_diplome', 'pct_bac_plus', 'pct_capbep']:
    if col not in df.columns:
        if col == 'pct_sup5':
            df[col] = df['P21_NSCOL15P_SUP5'] / _nscol * 100
        elif col == 'pct_sans_diplome':
            df[col] = df['P21_NSCOL15P_DIPLMIN'] / _nscol * 100
        elif col == 'pct_bac_plus':
            df[col] = (df['P21_NSCOL15P_SUP2'] + df['P21_NSCOL15P_SUP34'] + df['P21_NSCOL15P_SUP5']) / _nscol * 100
        elif col == 'pct_capbep':
            df[col] = df['P21_NSCOL15P_CAPBEP'] / _nscol * 100

if 'pct_chomage' not in df.columns:
    df['pct_chomage'] = df['P21_CHOM1564'] / df['P21_ACT1564'].replace(0, float('nan')) * 100

_sal = df['P21_SAL15P'].replace(0, float('nan'))
for col, raw in [('pct_cdi', 'P21_SAL15P_CDI'), ('pct_cdd', 'P21_SAL15P_CDD'),
                  ('pct_interim', 'P21_SAL15P_INTERIM'), ('pct_temps_partiel', 'P21_SAL15P_TP')]:
    if col not in df.columns:
        df[col] = df[raw] / _sal * 100

_act1564 = df['P21_ACT1564'].replace(0, float('nan'))
if 'pct_employeurs' not in df.columns:
    df['pct_employeurs'] = (df['P21_NSAL15P_EMPLOY'] / _act1564 * 100).clip(0, 100)

if 'ecart_csp_plus_hf' not in df.columns:
    pop_h15 = df['C21_H15P'].replace(0, float('nan'))
    pop_f15 = df['C21_F15P'].replace(0, float('nan'))
    df['ecart_csp_plus_hf'] = (df['C21_H15P_CS3'] / pop_h15 - df['C21_F15P_CS3'] / pop_f15) * 100

print('  Derived variables OK')

# ─── 3. PCA helper functions (copied from _gen_notebook.py) ────────────────────
LOG_TRANSFORM_VARS = {'DISP_MED21', 'surface_moyenne'}

def _zscore_pondere(series, pop):
    s = series.copy().astype(float)
    p = pop.copy().astype(float)
    valid = s.notna() & p.notna() & (p > 0)
    if valid.sum() < 10:
        return pd.Series(0.0, index=s.index)
    s_v, p_v = s[valid], p[valid]
    w = p_v / p_v.sum()
    w_mean = (s_v * w).sum()
    w_std = (((s_v - w_mean) ** 2 * w).sum()) ** 0.5
    if w_std < 1e-10:
        return pd.Series(0.0, index=s.index)
    result = pd.Series(float('nan'), index=s.index)
    result[valid] = (s_v - w_mean) / w_std
    return result.fillna(0.0)

def _rang_pondere(series, pop):
    s = series.copy().astype(float)
    p = pop.copy().astype(float)
    valid = s.notna() & p.notna() & (p > 0)
    if valid.sum() < 10:
        return pd.Series(0.0, index=s.index)
    s_v, p_v = s[valid], p[valid]
    order = s_v.argsort()
    p_sorted = p_v.iloc[order]
    cumsum = p_sorted.cumsum()
    centile = (cumsum - p_sorted / 2) / p_sorted.sum() * 100
    result = pd.Series(float('nan'), index=s.index)
    result.iloc[np.where(valid.values)[0][order.values]] = centile.values
    return result.fillna(50.0) - 50.0

def _pca_weighted_pc1(X, pop_array):
    w = np.where(np.isfinite(pop_array) & (pop_array > 0), pop_array, 1.0)
    w_norm = w / w.sum()
    means = (X * w_norm[:, None]).sum(axis=0)
    Xc = X - means
    C = (Xc * w_norm[:, None]).T @ Xc
    eigenvalues, eigenvectors = np.linalg.eigh(C)
    idx = eigenvalues.argsort()[::-1]
    eigenvalues, eigenvectors = eigenvalues[idx], eigenvectors[:, idx]
    total = eigenvalues.sum()
    var_exp_all = eigenvalues / total if total > 1e-10 else eigenvalues * 0
    return Xc @ eigenvectors[:, 0], var_exp_all, eigenvectors[:, 0]

# ─── 4. Score computation function ─────────────────────────────────────────────
def compute_score_pca(groupes, anchor_var='pct_csp_plus'):
    """
    Run grouped PCA (same as _gen_notebook.py CELL_ANALYSIS).
    Returns dict with: score (centile series), var_exp_g{i}, var_exp_final,
    loadings_g{i}, group_corr (if 2 groups).
    """
    pop = df['_pop']
    pop_arr = pop.values
    n_groupes = len(groupes)

    group_pc1_series = []
    group_var_exps = []
    group_loadings = []   # list of (vars_dispo, loadings array, poids_manuel dict)
    group_warnings = []   # list of warning vars per group

    for gi, groupe in enumerate(groupes):
        vars_dispo = [v for v in groupe['vars'] if v in df.columns]
        vars_absentes = [v for v in groupe['vars'] if v not in df.columns]
        if vars_absentes:
            print(f'    WARNING group {gi+1}: missing vars {vars_absentes}')

        if not vars_dispo:
            group_pc1_series.append(pd.Series(0.0, index=df.index))
            group_var_exps.append(np.array([0.0]))
            group_loadings.append(([], np.array([]), {}))
            group_warnings.append([])
            continue

        X_parts = []
        for var in vars_dispo:
            s = df[var].copy().astype(float)
            if var in LOG_TRANSFORM_VARS:
                s = s.clip(lower=1e-6).apply(np.log)
            X_parts.append(_zscore_pondere(s, pop).values)
        X = np.column_stack(X_parts)

        pc1, var_exp_all, loadings = _pca_weighted_pc1(X, pop_arr)
        pc1_s = pd.Series(pc1, index=df.index)

        if anchor_var and anchor_var in df.columns and pc1_s.corr(df[anchor_var]) < 0:
            pc1_s = -pc1_s
            loadings = -loadings

        group_pc1_series.append(pc1_s)
        group_var_exps.append(var_exp_all)

        # Track [OK] vs [!!] warnings (sign mismatch between poids_manuel and loading)
        warns = []
        for vi, var in enumerate(vars_dispo):
            poids = groupe['vars'][var]
            loading = loadings[vi]
            if poids * loading < 0:  # sign mismatch
                warns.append(var)
        group_loadings.append((vars_dispo, loadings, groupe['vars']))
        group_warnings.append(warns)

    # Stage 2: inter-group PCA if needed
    if n_groupes > 1:
        G = np.column_stack([_zscore_pondere(s, pop).values for s in group_pc1_series])
        final_pc1_arr, final_var_exp_all, _ = _pca_weighted_pc1(G, pop_arr)
        final_var_exp = final_var_exp_all[0]

        # Correlation between groups
        group_corr = group_pc1_series[0].corr(group_pc1_series[1])

        final_pc1_s = pd.Series(final_pc1_arr, index=df.index)
        if anchor_var and anchor_var in df.columns and final_pc1_s.corr(df[anchor_var]) < 0:
            final_pc1_s = -final_pc1_s
    else:
        final_pc1_s = group_pc1_series[0]
        final_var_exp = group_var_exps[0][0]
        group_corr = None

    score = _rang_pondere(final_pc1_s, pop)

    return {
        'score': score,
        'var_exp_g': [ve[0] for ve in group_var_exps],
        'var_exp_final': final_var_exp,
        'group_corr': group_corr,
        'loadings': group_loadings,   # [(vars, loadings, poids_dict), ...]
        'warnings': group_warnings,
    }

# ─── 5. Configurations ─────────────────────────────────────────────────────────
DOM_CONFIGS = {
    # --- Baselines ---
    'dom_A': {
        'desc': 'Actuelle (7 CSP + contrats)',
        'groupes': [{'poids': 1.0, 'vars': {
            'pct_csp_plus': +1.0, 'pct_csp_intermediaire': +0.8, 'pct_csp_independant': +0.6,
            'pct_employeurs': +0.7, 'pct_cdi': +0.3,
            'pct_csp_ouvrier': -0.8, 'pct_csp_employe': -0.5, 'pct_csp_sans_emploi': -1.0,
            'pct_chomage': -0.8, 'pct_interim': -0.5, 'pct_cdd': -0.4, 'pct_temps_partiel': -0.5,
        }}],
    },
    'dom_bis': {
        'desc': 'Actuelle bis (CSP seules)',
        'groupes': [{'poids': 1.0, 'vars': {
            'pct_csp_plus': +1.0, 'pct_employeurs': +0.7,
            'pct_csp_independant': +0.6, 'pct_csp_intermediaire': +0.8,
            'pct_csp_sans_emploi': -1.0, 'pct_csp_employe': -0.5, 'pct_csp_ouvrier': -0.8,
        }}],
    },

    # --- Hypothèse C: Minimaliste — hiérarchie pure (CSP3/4 vs CSP5/6/8) ---
    'dom_C': {
        'desc': 'Minimaliste — hiérarchie pure',
        'groupes': [{'poids': 1.0, 'vars': {
            'pct_csp_plus': +1.0, 'pct_csp_intermediaire': +0.5,
            'pct_csp_employe': -0.8, 'pct_csp_ouvrier': -1.0, 'pct_csp_sans_emploi': -0.7,
        }}],
    },

    # --- Hypothèse D: Marxiste stricte — propriétaires du capital vs travailleurs ---
    'dom_D': {
        'desc': 'Marxiste: capital vs travail',
        'groupes': [{'poids': 1.0, 'vars': {
            'pct_employeurs': +1.0, 'pct_csp_plus': +0.8,
            'pct_csp_ouvrier': -1.0, 'pct_csp_employe': -0.8, 'pct_csp_sans_emploi': -0.6,
        }}],
    },

    # --- Hypothèse E: Service class (Goldthorpe) — 2 groupes: CSP + rapports d'emploi ---
    'dom_E': {
        'desc': 'Goldthorpe service class (2G)',
        'groupes': [
            {'poids': 0.6, 'vars': {
                'pct_csp_plus': +1.0, 'pct_csp_intermediaire': +0.7,
                'pct_csp_sans_emploi': -1.0, 'pct_csp_ouvrier': -0.8, 'pct_csp_employe': -0.5,
            }},
            {'poids': 0.4, 'vars': {
                'pct_cdi': +0.8, 'pct_interim': -0.8, 'pct_cdd': -0.7,
            }},
        ],
    },

    # --- Hypothèse F: Hiérarchie salariale pure (sans indépendants ni agriculteurs) ---
    'dom_F': {
        'desc': 'Hiérarchie salariale pure (sans indép.)',
        'groupes': [{'poids': 1.0, 'vars': {
            'pct_csp_plus': +1.0, 'pct_csp_intermediaire': +0.7,
            'pct_csp_employe': -0.6, 'pct_csp_ouvrier': -0.8, 'pct_csp_sans_emploi': -1.0,
        }}],
    },

    # --- Hypothèse G: Domination genrée — ajouter l'écart H/F dans les CSP+ ---
    'dom_G': {
        'desc': 'Domination genrée (+ écart H/F CSP+)',
        'groupes': [{'poids': 1.0, 'vars': {
            'pct_csp_plus': +1.0, 'pct_csp_intermediaire': +0.6,
            'ecart_csp_plus_hf': +0.4,
            'pct_csp_ouvrier': -0.8, 'pct_csp_employe': -0.5, 'pct_csp_sans_emploi': -0.8,
        }}],
    },

    # --- Hypothèse H: 2 groupes: classe (CSP) + rapports de travail (contrats) ---
    'dom_H': {
        'desc': '2G: classe CSP + rapports de travail',
        'groupes': [
            {'poids': 0.65, 'vars': {
                'pct_csp_plus': +1.0, 'pct_csp_intermediaire': +0.7,
                'pct_csp_ouvrier': -0.9, 'pct_csp_employe': -0.6, 'pct_csp_sans_emploi': -1.0,
            }},
            {'poids': 0.35, 'vars': {
                'pct_cdi': +0.8, 'pct_interim': -0.9, 'pct_cdd': -0.7, 'pct_temps_partiel': -0.5,
            }},
        ],
    },

    # --- Hypothèse I: Cadres dirigeants vs dominés stricts (polarisation maximale) ---
    'dom_I': {
        'desc': 'Cadres/employeurs vs dominés stricts',
        'groupes': [{'poids': 1.0, 'vars': {
            'pct_csp_plus': +1.0, 'pct_employeurs': +0.6,
            'pct_csp_ouvrier': -1.0, 'pct_csp_sans_emploi': -0.9, 'pct_csp_employe': -0.5,
        }}],
    },

    # --- Hypothèse J: Sécurité emploi + position (CDI = confiance/intégration salariale) ---
    'dom_J': {
        'desc': 'Position + sécurité emploi (CDI)',
        'groupes': [{'poids': 1.0, 'vars': {
            'pct_csp_plus': +1.0, 'pct_csp_intermediaire': +0.5, 'pct_cdi': +0.6,
            'pct_csp_ouvrier': -0.8, 'pct_csp_employe': -0.5, 'pct_csp_sans_emploi': -0.9,
            'pct_interim': -0.5,
        }}],
    },

    # --- Hypothèse K: Capital culturel-hiérarchique (éducation comme proxy domination symbolique) ---
    'dom_K': {
        'desc': 'Capital culturel-hiérarchique (+éducation)',
        'groupes': [{'poids': 1.0, 'vars': {
            'pct_csp_plus': +1.0, 'pct_csp_intermediaire': +0.6,
            'pct_sup5': +0.7, 'pct_bac_plus': +0.4,
            'pct_csp_ouvrier': -0.8, 'pct_sans_diplome': -0.8, 'pct_capbep': -0.5,
        }}],
    },

    # --- Hypothèse L: Capital culturel-hiérarchique + classe (CSP) + rapports de travail (contrats) --- 
    'dom_L': {
        'desc': 'Capital culturel-hiérarchique + classe (CSP) + rapports de travail (contrats)',
        'groupes': [{'poids': 1.0, 'vars': {

            'pct_csp_plus': +1.0, 'pct_csp_intermediaire': +0.6,
            'pct_csp_ouvrier': -0.8, 'pct_csp_employe': -0.6, 'pct_csp_sans_emploi': -1.0,

            'pct_sup5': +0.7, 'pct_bac_plus': +0.4, 
            'pct_sans_diplome': -0.8, 'pct_capbep': -0.5, 

            'pct_cdi': +0.8, 
            'pct_interim': -0.9, 'pct_cdd': -0.7, 'pct_temps_partiel': -0.5,
        }}],
    },

    # --- Hypothèse M: Capital culturel-hiérarchique + classe (CSP) + rapports de travail (contrats) --- (Grouped)
    'dom_M': {
        'desc': 'Capital culturel-hiérarchique + classe (CSP) + rapports de travail (contrats) (Grouped)',
        'groupes': [
            {'poids': 0.4, 'vars': {
                'pct_csp_plus': +1.0, 'pct_csp_intermediaire': +0.6,
                'pct_csp_ouvrier': -0.8, 'pct_csp_employe': -0.6, 'pct_csp_sans_emploi': -1.0,
            }},
            {'poids': 0.3, 'vars': {
                'pct_sup5': +0.7, 'pct_bac_plus': +0.4, 
                'pct_sans_diplome': -0.8, 'pct_capbep': -0.5, 
            }},
            {'poids': 0.3, 'vars': {
                'pct_cdi': +0.8, 
                'pct_interim': -0.9, 'pct_cdd': -0.7, 'pct_temps_partiel': -0.5,     
            }},
        ],
    },
    


}

ANCHOR_VAR = 'pct_csp_plus'

# Config exploitation_bis pour score de référence Y
EXPLOIT_BIS_CONFIG = [
    {'poids': 0.75, 'vars': {'DISP_PPAT21': +1.5, 'DISP_PBEN21': +1.0, 'DISP_PCHO21': -0.5}},
    {'poids': 0.25, 'vars': {'pct_proprietaires': +1.5, 'pct_employeurs': +0.8,
                              'pct_csp_ouvrier': -0.8, 'pct_csp_employe': -0.5}},
]

# Configs de référence pour la matrice de corrélation
CAP_ECO_CONFIG = [
    {'poids': 0.7, 'vars': {
        'DISP_MED21': +1.0, 'DISP_PPAT21': +0.8, 'DISP_PBEN21': +0.5,
        'DISP_TP6021': -1.0, 'DISP_PPMINI21': -1.0, 'DISP_PPLOGT21': -0.6,
    }},
    {'poids': 0.3, 'vars': {'pct_proprietaires': +0.8, 'pct_hlm': -0.8, 'pct_suroccupation': -0.5}},
]
PRECARITE_CONFIG = [
    {'poids': 0.6, 'vars': {
        'DISP_TP6021': +1.0, 'DISP_PPMINI21': +0.8, 'DISP_PPSOC21': +0.5,
        'DISP_PPLOGT21': +0.4, 'DISP_MED21': -1.0,
    }},
    {'poids': 0.4, 'vars': {
        'pct_csp_sans_emploi': +0.8, 'pct_chomage': +0.7, 'pct_interim': +0.5,
        'pct_cdd': +0.4, 'pct_temps_partiel': +0.3, 'pct_hlm': +0.4, 'pct_suroccupation': +0.5,
    }},
]

# ─── 6. Compute all scores ──────────────────────────────────────────────────────
print('\n=== Computing scores ===')
print('Computing score_exploitation_bis...')
exploit_res = compute_score_pca(EXPLOIT_BIS_CONFIG, anchor_var='DISP_PPAT21')
df['score_exploitation_bis'] = exploit_res['score']

print('Computing score_cap_eco...')
capeco_res = compute_score_pca(CAP_ECO_CONFIG, anchor_var='DISP_MED21')
df['score_cap_eco'] = capeco_res['score']

print('Computing score_precarite...')
prec_res = compute_score_pca(PRECARITE_CONFIG, anchor_var='DISP_TP6021')
df['score_precarite'] = prec_res['score']

print('Computing domination configs...')
dom_results = {}
for name, cfg in DOM_CONFIGS.items():
    res = compute_score_pca(cfg['groupes'], anchor_var=ANCHOR_VAR)
    df[name] = res['score']
    dom_results[name] = res
    n_g = len(cfg['groupes'])
    ve = [f'{v*100:.1f}%' for v in res['var_exp_g']]
    print(f'  {name}: var_exp_g=[{", ".join(ve)}]  final={res["var_exp_final"]*100:.1f}%')

# ─── 7. Quantitative analysis ──────────────────────────────────────────────────
print('\n' + '='*80)
print('=== QUANTITATIVE RESULTS ===')
print('='*80)
baseline_score = df['dom_bis']

for name, res in dom_results.items():
    cfg = DOM_CONFIGS[name]
    n_g = len(cfg['groupes'])
    score = df[name]
    corr_baseline = score.corr(baseline_score) if name != 'dom_bis' else 1.0
    corr_exploit  = score.corr(df['score_exploitation_bis'])
    corr_capeco   = score.corr(df['score_cap_eco'])
    corr_precar   = score.corr(df['score_precarite'])
    n_warns = sum(len(w) for w in res['warnings'])

    ve = res['var_exp_g']
    ve_str = ', '.join([f'G{i+1}={v*100:.1f}%' for i, v in enumerate(ve)])
    if n_g > 1:
        ve_str += f' -> final={res["var_exp_final"]*100:.1f}%'
    else:
        ve_str = f'G1={ve[0]*100:.1f}%'

    print(f'\n--- {name}: {cfg["desc"]} ---')
    print(f'  Variance expliquée : {ve_str}')
    if n_g > 1 and res['group_corr'] is not None:
        print(f'  Corrélation inter-groupes : {res["group_corr"]:.3f}')
    print(f'  Corr. baseline (dom_bis): {corr_baseline:.3f}')
    print(f'  Corr. exploitation_bis  : {corr_exploit:.3f}')
    print(f'  Corr. cap_eco           : {corr_capeco:.3f}')
    print(f'  Corr. precarite         : {corr_precar:.3f}')
    print(f'  Nb [!!] warnings        : {n_warns}')

    for gi, (vars_dispo, loadings, poids_dict) in enumerate(res['loadings']):
        ve_g = res['var_exp_g'][gi]
        print(f'  Groupe {gi+1} (poids={cfg["groupes"][gi]["poids"]}, var_exp={ve_g*100:.1f}%):')
        for vi, var in enumerate(vars_dispo):
            loading = loadings[vi]
            poids = poids_dict[var]
            ok = '[OK]' if poids * loading >= 0 else '[!!]'
            print(f'    {ok} {var:<40} poids={poids:+.2f}  loading={loading:+.3f}')

# Tableau récapitulatif
print('\n' + '='*80)
print('=== SUMMARY TABLE ===')
header = f"{'config':<10} {'n_g':>3} {'var_exp':>8} {'corr_base':>10} {'corr_expl':>10} {'corr_capeco':>11} {'warns':>6}  desc"
print(header)
print('-' * len(header))
for name, res in dom_results.items():
    cfg = DOM_CONFIGS[name]
    n_g = len(cfg['groupes'])
    score = df[name]
    corr_baseline = score.corr(baseline_score) if name != 'dom_bis' else 1.0
    corr_exploit  = score.corr(df['score_exploitation_bis'])
    corr_capeco   = score.corr(df['score_cap_eco'])
    n_warns = sum(len(w) for w in res['warnings'])
    ve_final = res['var_exp_final'] * 100
    print(f"{name:<10} {n_g:>3} {ve_final:>7.1f}%  {corr_baseline:>+9.3f}  {corr_exploit:>+9.3f}  {corr_capeco:>+10.3f}  {n_warns:>5}  {cfg['desc'][:40]}")

# ─── 8. Election data ──────────────────────────────────────────────────────────
print('\nLoading election data...')
df_elect = pd.read_csv(ELECT_PATH, dtype={'CODE_IRIS': str})
df_elect = df_elect.set_index('CODE_IRIS')

# Build code_iris column in df if needed
iris_code_col = None
for c in ['CODE_IRIS', 'code_iris', 'IRIS', 'iris']:
    if c in df.columns:
        iris_code_col = c
        break
if iris_code_col:
    df_e = df.copy()
    df_e.index = df_e[iris_code_col].astype(str).str.zfill(9)
    df_e = df_e.join(df_elect[['score_LE_PEN', 'score_MACRON', 'exprimes']], how='left')
    df_e['winner'] = df_e.apply(
        lambda r: 'MACRON' if (pd.notna(r['score_MACRON']) and pd.notna(r['score_LE_PEN'])
                               and r['score_MACRON'] > r['score_LE_PEN']) else
                  ('LE_PEN' if pd.notna(r['score_LE_PEN']) else None),
        axis=1
    )
    COLOR_MAP = {'MACRON': '#2271b3', 'LE_PEN': '#b32222'}
    df_e['color_elect'] = df_e['winner'].map(COLOR_MAP).fillna('#aaaaaa')
    has_election = True
    n_matched = df_e['winner'].notna().sum()
    print(f'  Matched {n_matched} IRIS with election data')
else:
    has_election = False
    print('  WARNING: no CODE_IRIS column found — election scatter will be skipped')

# ─── 9. GRAPHICAL ANALYSES ─────────────────────────────────────────────────────

# ── A. Correlation heatmap ──────────────────────────────────────────────────────
print('\nPlot A: Correlation heatmap...')
all_score_cols = list(DOM_CONFIGS.keys()) + ['score_exploitation_bis', 'score_cap_eco', 'score_precarite']
corr_df = df[all_score_cols].corr()

fig, ax = plt.subplots(figsize=(14, 11))
mask = np.zeros_like(corr_df.values, dtype=bool)
np.fill_diagonal(mask, True)
sns.heatmap(corr_df, annot=True, fmt='.2f', cmap='RdBu_r', center=0, vmin=-1, vmax=1,
            ax=ax, linewidths=0.5, annot_kws={'size': 8}, mask=mask)
ax.set_title('Corrélations inter-scores (domination candidates + références)', fontsize=12, pad=12)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'domination_corr_heatmap.png'), dpi=90, bbox_inches='tight')
plt.close()
print('  -> domination_corr_heatmap.png')

# ── B. Loadings heatmap (variables × configs) ───────────────────────────────────
print('Plot B: Loadings heatmap...')
all_vars_in_configs = []
for cfg in DOM_CONFIGS.values():
    for g in cfg['groupes']:
        for v in g['vars']:
            if v not in all_vars_in_configs:
                all_vars_in_configs.append(v)

loading_matrix = pd.DataFrame(index=all_vars_in_configs, columns=list(DOM_CONFIGS.keys()), dtype=float)
for name, res in dom_results.items():
    for gi, (vars_dispo, loadings, _) in enumerate(res['loadings']):
        for vi, var in enumerate(vars_dispo):
            loading_matrix.loc[var, name] = loadings[vi]
loading_matrix = loading_matrix.fillna(0.0)

fig, ax = plt.subplots(figsize=(len(DOM_CONFIGS) * 1.3 + 2, len(all_vars_in_configs) * 0.5 + 2))
sns.heatmap(loading_matrix.astype(float), annot=True, fmt='.2f', cmap='RdBu_r', center=0,
            vmin=-0.8, vmax=0.8, ax=ax, linewidths=0.5, annot_kws={'size': 7})
ax.set_title('Loadings PCA PC1 par config × variable\n(blanc = variable absente de la config)',
             fontsize=11, pad=10)
ax.set_xlabel('Configuration')
ax.set_ylabel('Variable')
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'domination_loadings_heatmap.png'), dpi=90, bbox_inches='tight')
plt.close()
print('  -> domination_loadings_heatmap.png')

# ── C. Distribution comparison ──────────────────────────────────────────────────
print('Plot C: Distribution comparison...')
n_configs = len(DOM_CONFIGS)
ncols = 4
nrows = (n_configs + ncols - 1) // ncols
fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4, nrows * 3))
axes_flat = axes.flatten()
for i, (name, cfg) in enumerate(DOM_CONFIGS.items()):
    ax = axes_flat[i]
    res = dom_results[name]
    ve = res['var_exp_final'] * 100
    ax.hist(df[name].dropna(), bins=40, color='steelblue', alpha=0.7, density=True)
    ax.axvline(0, color='red', linewidth=0.8, linestyle='--')
    ax.set_title(f'{name}\nvar_exp={ve:.1f}%', fontsize=9)
    ax.set_xlabel('Centile (-50, +50)', fontsize=8)
for i in range(n_configs, len(axes_flat)):
    axes_flat[i].set_visible(False)
fig.suptitle('Distributions des scores domination (centiles pondérés)', fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'domination_distributions.png'), dpi=90, bbox_inches='tight')
plt.close()
print('  -> domination_distributions.png')

# ── D. Correlation heatmap entre variables dans les configs ─────────────────────
print('Plot D: Inter-variable correlation matrices...')
# Group by "pool" of variables used across all domination configs
dom_vars_unique = [v for v in all_vars_in_configs if v in df.columns]
if len(dom_vars_unique) >= 2:
    fig_size = max(10, len(dom_vars_unique) * 0.7)
    fig, ax = plt.subplots(figsize=(fig_size, fig_size * 0.85))
    corr_vars = df[dom_vars_unique].corr()
    sns.heatmap(corr_vars, annot=True, fmt='.2f', cmap='RdBu_r', center=0, vmin=-1, vmax=1,
                ax=ax, linewidths=0.5, annot_kws={'size': 7})
    ax.set_title('Corrélations inter-variables (pool domination configs)', fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'domination_vars_corr.png'), dpi=90, bbox_inches='tight')
    plt.close()
    print('  -> domination_vars_corr.png')

# ── E. Scatter "saint-graphique" style ─────────────────────────────────────────
if has_election:
    print('Plot E: Scatter saint-graphique style...')
    y_col = 'score_exploitation_bis'
    y_vals = df_e[y_col].values

    for name, cfg in DOM_CONFIGS.items():
        res = dom_results[name]
        ve = res['var_exp_final'] * 100
        corr_exploit = df[name].corr(df[y_col])

        x_vals = df_e[name].values
        colors = df_e['color_elect'].values
        sizes_raw = df_e['_pop'].values.copy()
        sizes_raw = np.where(np.isfinite(sizes_raw), sizes_raw, sizes_raw[np.isfinite(sizes_raw)].min())
        s_min, s_max = sizes_raw.min(), sizes_raw.max()
        sizes = 8 + 30 * (sizes_raw - s_min) / (s_max - s_min + 1e-9)

        fig, axes2 = plt.subplots(1, 2, figsize=(16, 7))
        fig.suptitle(f'{name} — {cfg["desc"]}\nvar_exp={ve:.1f}%  corr_exploit={corr_exploit:.3f}',
                     fontsize=11, fontweight='bold')

        # --- Subplot 1: Tous les IRIS ---
        ax1 = axes2[0]
        valid = np.isfinite(x_vals) & np.isfinite(y_vals)
        ax1.scatter(x_vals[valid], y_vals[valid], c=colors[valid], s=sizes[valid],
                    alpha=0.5, linewidths=0.0)
        ax1.axhline(0, color='gray', linewidth=0.6, linestyle='--')
        ax1.axvline(0, color='gray', linewidth=0.6, linestyle='--')
        ax1.set_xlabel(f'{name} (domination)', fontsize=10)
        ax1.set_ylabel('score_exploitation_bis', fontsize=10)
        ax1.set_title(f'Tous les IRIS (n={valid.sum()})', fontsize=10)
        # Legend
        for lbl, col in COLOR_MAP.items():
            ax1.scatter([], [], c=col, s=20, label=lbl)
        ax1.scatter([], [], c='#aaaaaa', s=20, label='Absent')
        ax1.legend(fontsize=8, loc='upper left')

        # --- Subplot 2: Barycentres candidats ---
        ax2 = axes2[1]
        bary_data = {}
        for cand in ['MACRON', 'LE_PEN']:
            score_cand = df_e[f'score_{cand}'].values / 100.0  # 0..1
            exprimes_cand = df_e['exprimes'].values.copy()
            exprimes_cand = np.where(np.isfinite(exprimes_cand), exprimes_cand, 0.0)
            weights = score_cand * exprimes_cand
            weights = np.where(np.isfinite(weights), weights, 0.0)
            valid2 = np.isfinite(x_vals) & np.isfinite(y_vals) & (weights > 0)
            if valid2.sum() > 0:
                w = weights[valid2]
                bary_x = np.sum(w * x_vals[valid2]) / w.sum()
                bary_y = np.sum(w * y_vals[valid2]) / w.sum()
                bary_data[cand] = (bary_x, bary_y)

        if bary_data:
            bary_xs = [v[0] for v in bary_data.values()]
            bary_ys = [v[1] for v in bary_data.values()]
            margin_x = max(abs(x) for x in bary_xs) * 1.8
            margin_y = max(abs(y) for y in bary_ys) * 1.8
            ax2.set_xlim(-margin_x, margin_x)
            ax2.set_ylim(-margin_y, margin_y)

            for cand, (bx, by) in bary_data.items():
                col = COLOR_MAP.get(cand, '#aaaaaa')
                ax2.scatter([bx], [by], c=col, s=200, zorder=5, edgecolors='black', linewidths=0.8)
                ax2.annotate(cand, (bx, by), textcoords='offset points', xytext=(8, 4),
                             fontsize=11, fontweight='bold', color=col)

        ax2.axhline(0, color='gray', linewidth=0.6, linestyle='--')
        ax2.axvline(0, color='gray', linewidth=0.6, linestyle='--')
        ax2.set_xlabel(f'{name} (domination)', fontsize=10)
        ax2.set_ylabel('score_exploitation_bis', fontsize=10)
        ax2.set_title('Barycentres candidats (pondérés votes×pop)', fontsize=10)

        plt.tight_layout()
        out_path = os.path.join(OUT_DIR, f'domination_scatter_{name}.png')
        plt.savefig(out_path, dpi=90, bbox_inches='tight')
        plt.close()
        print(f'  -> domination_scatter_{name}.png')

# ── F. Variance expliquée par config (bar chart) ────────────────────────────────
print('Plot F: Variance expliquée bar chart...')
fig, ax = plt.subplots(figsize=(12, 5))
names = list(DOM_CONFIGS.keys())
ve_vals = [dom_results[n]['var_exp_final'] * 100 for n in names]
bars = ax.bar(names, ve_vals, color=['#4575b4' if v >= 50 else '#fdae61' if v >= 40 else '#d73027' for v in ve_vals])
ax.axhline(50, color='green', linestyle='--', linewidth=1.5, label='objectif 50%')
ax.axhline(35.6, color='gray', linestyle=':', linewidth=1.5, label='baseline dom_A (35.6%)')
ax.set_ylim(0, 100)
ax.set_ylabel('Variance expliquée PC1 (%)')
ax.set_title('Variance expliquée par configuration de score_domination')
ax.legend()
for bar, v in zip(bars, ve_vals):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, f'{v:.1f}%',
            ha='center', va='bottom', fontsize=8)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'domination_variance_expliquee.png'), dpi=90, bbox_inches='tight')
plt.close()
print('  -> domination_variance_expliquee.png')

# ── G. Pairplot (sous-échantillon) ──────────────────────────────────────────────
print('Plot G: Pairplot (sample)...')
# Select a few interesting configs for pairplot
pairplot_cols = ['dom_A', 'dom_bis', 'dom_C', 'dom_F', 'dom_K', 'score_exploitation_bis', 'score_cap_eco']
pairplot_cols = ['dom_bis', 'dom_E', 'dom_H', 'dom_K', 'score_exploitation_bis']
pairplot_cols = [c for c in pairplot_cols if c in df.columns]
sample_size = min(2000, len(df))
df_sample = df[pairplot_cols].dropna().sample(sample_size, random_state=42)
fig = sns.pairplot(df_sample, diag_kind='kde', plot_kws={'alpha': 0.3, 's': 8})
fig.figure.suptitle('Pairplot: configs domination sélectionnées + références', y=1.01, fontsize=11)
plt.savefig(os.path.join(OUT_DIR, 'domination_pairplot.png'), dpi=80, bbox_inches='tight')
plt.close()
print('  -> domination_pairplot.png')

print('\n=== DONE ===')
print(f'Output files saved in: {OUT_DIR}')

# ── H. Correlation between all domination scores ─────────────────────────────────
print('Plot H: Correlation between all domination scores (dom A -> ... -> dom M) ...')
dom_cols = list(DOM_CONFIGS.keys())
dom_corr_df = df[dom_cols].corr()

fig, ax = plt.subplots(figsize=(12, 10))
mask = np.zeros_like(dom_corr_df.values, dtype=bool)
np.fill_diagonal(mask, True)
sns.heatmap(dom_corr_df, annot=True, fmt='.3f', cmap='RdBu_r', center=0, vmin=-1, vmax=1,
            mask=mask, square=True, linewidths=0.5, cbar_kws={'label': 'Corrélation'},
            ax=ax, annot_kws={'fontsize': 8})
ax.set_title('Corrélation entre configurations de score_domination', fontsize=12, fontweight='bold', pad=20)
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'domination_correlation_matrix.png'), dpi=90, bbox_inches='tight')
plt.close()
print('  -> domination_correlation_matrix.png')

