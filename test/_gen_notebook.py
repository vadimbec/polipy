"""Script de génération du notebook pca_scores_diagnostics.ipynb"""
import json, os

def code_cell(cell_id, src):
    return {"cell_type": "code", "id": cell_id, "metadata": {},
            "source": src, "outputs": [], "execution_count": None}

def md_cell(cell_id, src):
    return {"cell_type": "markdown", "id": cell_id, "metadata": {}, "source": src}

CELL_IMPORTS = """\
import sys, os
sys.path.insert(0, os.path.abspath('..'))
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
plt.rcParams['figure.dpi'] = 110
plt.rcParams['font.size'] = 10

df = pd.read_csv('../iris/iris_final_socio_politique.csv', low_memory=False)
df['_pop'] = df['pop_totale'].fillna(df['pop_totale'].median()) if 'pop_totale' in df.columns else 2000.0
print(f'IRIS : {len(df)}  colonnes : {len(df.columns)}')
"""

CELL_VARS = """\
_nscol = df['P21_NSCOL15P'].replace(0, float('nan'))
df['pct_sup5']         = df['P21_NSCOL15P_SUP5'] / _nscol * 100
df['pct_sans_diplome'] = df['P21_NSCOL15P_DIPLMIN'] / _nscol * 100
df['pct_bac_plus']     = (df['P21_NSCOL15P_SUP2'] + df['P21_NSCOL15P_SUP34'] + df['P21_NSCOL15P_SUP5']) / _nscol * 100
df['pct_capbep']       = df['P21_NSCOL15P_CAPBEP'] / _nscol * 100
df['pct_chomage']      = df['P21_CHOM1564'] / df['P21_ACT1564'].replace(0, float('nan')) * 100
df['pct_inactif']      = df['P21_INACT1564'] / df['P21_POP1564'].replace(0, float('nan')) * 100
df['pct_etudiants']    = df['P21_ETUD1564'] / df['P21_POP1564'].replace(0, float('nan')) * 100
_sal = df['P21_SAL15P'].replace(0, float('nan'))
df['pct_cdi']          = df['P21_SAL15P_CDI'] / _sal * 100
df['pct_cdd']          = df['P21_SAL15P_CDD'] / _sal * 100
df['pct_interim']      = df['P21_SAL15P_INTERIM'] / _sal * 100
df['pct_temps_partiel']= df['P21_SAL15P_TP'] / _sal * 100
_actocc = df['P21_ACTOCC15P'].replace(0, float('nan'))
df['pct_actifs_voiture']    = df['C21_ACTOCC15P_VOIT'] / _actocc * 100
df['pct_actifs_transports'] = df['C21_ACTOCC15P_TCOM'] / _actocc * 100
df['pct_actifs_velo']       = df['C21_ACTOCC15P_VELO'] / _actocc * 100
df['pct_actifs_marche']     = df['C21_ACTOCC15P_MAR'] / _actocc * 100
_act1564 = df['P21_ACT1564'].replace(0, float('nan'))
df['pct_employeurs'] = (df['P21_NSAL15P_EMPLOY'] / _act1564 * 100).clip(0, 100)
print('Variables derivees OK')
"""

CELL_HELPERS = """\
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

def make_score_grouped(groupes):
    pop = df['_pop']
    rangs, poids_list = [], []
    for groupe in groupes:
        parts = []
        total_p = sum(abs(p) for p in groupe['vars'].values())
        for var, poids in groupe['vars'].items():
            if var in df.columns:
                parts.append((poids / total_p) * _zscore_pondere(df[var], pop))
        if not parts:
            continue
        rangs.append(_rang_pondere(pd.concat(parts, axis=1).sum(axis=1), pop))
        poids_list.append(groupe['poids'])
    if not rangs:
        return pd.Series(0.0, index=df.index)
    return sum(r * p for r, p in zip(rangs, poids_list)) / sum(poids_list)

LOG_TRANSFORM_VARS = {'DISP_MED21', 'surface_moyenne'}
print('Fonctions OK')
"""

CELL_CONFIG = """\
SCORES_CONFIG_GROUPED = {
    'score_domination': [
        {'poids': 1.0, 'vars': {
            'pct_csp_plus': +1.0, 'pct_csp_intermediaire': +0.8, 'pct_csp_independant': +0.6,
            'pct_employeurs': +0.7, 'pct_cdi': +0.3,
            'pct_csp_ouvrier': -0.8, 'pct_csp_employe': -0.5, 'pct_csp_sans_emploi': -1.0,
            'pct_chomage': -0.8, 'pct_interim': -0.5, 'pct_cdd': -0.4, 'pct_temps_partiel': -0.5,
        }}
    ],
    'score_exploitation': [
        {'poids': 0.75, 'vars': {
            'DISP_PPAT21': +1.5, 'DISP_PBEN21': +1.0, 'DISP_PTSA21': -1.0, 'DISP_PCHO21': -0.5,
        }},
        {'poids': 0.25, 'vars': {
            'pct_proprietaires': +1.5, 'pct_employeurs': +0.8,
            'pct_csp_ouvrier': -0.8, 'pct_csp_employe': -0.5,
            'pct_cdi': -0.5, 'pct_cdd': -0.6, 'pct_interim': -0.7, 'pct_temps_partiel': -0.7,
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
        {'poids': 0.8, 'vars': {
            'pct_sup5': +1.0, 'pct_bac_plus': +0.6, 'pct_csp_intermediaire': +0.5,
            'pct_actifs_velo': +0.3,
            'pct_sans_diplome': -1.0, 'pct_capbep': -0.6,
            'pct_csp_independant': -0.3, 'pct_csp_agriculteur': -0.3,
        }},
        {'poids': 0.2, 'vars': {
            'bpe_ecole_privee_pour1000': +0.8, 'bpe_sport_indoor_pour1000': +0.4,
        }}
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
            'DISP_PPAT21': +1.0, 'DISP_PBEN21': +0.7, 'DISP_PPEN21': +0.5,
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
            'bpe_E_transports_pour1000': +0.9, 'bpe_total_pour1000': +0.7,
            'P21_ACTOCC15P_ILT1': +0.5, 'bpe_B_commerces_pour1000': +0.4,
        }}
    ],
    'score_confort_residentiel': [
        {'poids': 1.0, 'vars': {
            'pct_proprietaires': +0.7, 'pct_grands_logements': +0.8,
            'surface_moyenne': +0.7, 'nb_pieces_moyen': +0.6, 'pct_garage': +0.5,
            'pct_logements_5p_plus': +0.5,
            'pct_suroccupation': -1.0, 'pct_hlm': -0.7, 'pct_petits_logements': -0.6,
            'pct_studios': -0.5, 'pct_logvac': -0.3,
        }}
    ],
    'score_equipement_public': [
        {'poids': 1.0, 'vars': {
            'bpe_D_sante_pour1000': +1.0, 'bpe_C_enseignement_pour1000': +0.8,
            'bpe_A_services_pour1000': +0.7, 'bpe_E_transports_pour1000': +0.7,
            'bpe_F_sports_culture_pour1000': +0.6, 'bpe_B_commerces_pour1000': +0.5,
            'bpe_educ_prioritaire_pour1000': +0.4,
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
    'score_domination':          'pct_csp_plus',
    'score_exploitation':        'DISP_PPAT21',
    'score_cap_eco':             'DISP_MED21',
    'score_cap_cult':            'pct_sup5',
    'score_precarite':           'DISP_TP6021',
    'score_rentier':             'DISP_PPAT21',
    'score_urbanite':            'pct_appart',
    'score_confort_residentiel': 'surface_moyenne',
    'score_equipement_public':   'bpe_D_sante_pour1000',
    'score_dependance_carbone':  'pct_actifs_voiture',
}
print(f'{len(SCORES_CONFIG_GROUPED)} scores')
"""

CELL_GROUPED = """\
for score_name, groupes in SCORES_CONFIG_GROUPED.items():
    df[score_name] = make_score_grouped(groupes)
    print(f'  {score_name}: [{df[score_name].min():.1f}, {df[score_name].max():.1f}]')
"""

CELL_ANALYSIS = """\
# Analyse detaillee : loadings, inter-groupes, scatter grouped vs PCA
all_results    = {}
all_loadings   = []
all_group_corr = []
all_scores_df  = {}

pop = df['_pop']
pop_arr = pop.values

for score_name, groupes in SCORES_CONFIG_GROUPED.items():
    anchor_var = SCORES_PCA_ANCHORS.get(score_name)
    n_groupes  = len(groupes)
    n_extra    = 1 if n_groupes > 1 else 0   # colonne inter-groupes
    n_cols     = n_groupes + n_extra + 1       # +1 scatter/dist

    fig, axes = plt.subplots(2, n_cols, figsize=(4.2 * n_cols, 8))
    if n_cols == 1:
        axes = axes.reshape(2, 1)
    fig.suptitle(f'{score_name}  (ancre: {anchor_var})', fontsize=13, fontweight='bold')

    group_pc1_series = []
    group_var_exps   = []

    # -- Etage 1 : PCA par groupe --
    for gi, groupe in enumerate(groupes):
        vars_dispo    = [v for v in groupe['vars'] if v in df.columns]
        vars_absentes = [v for v in groupe['vars'] if v not in df.columns]

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
            pc1_s    = -pc1_s
            loadings = -loadings

        group_pc1_series.append(pc1_s)
        group_var_exps.append(var_exp_all)

        for vi, var in enumerate(vars_dispo):
            all_loadings.append({
                'score': score_name, 'groupe': gi + 1, 'poids_groupe': groupe['poids'],
                'variable': var, 'poids_manuel': groupe['vars'][var],
                'loading_pc1': loadings[vi], 'var_exp_pc1_groupe': var_exp_all[0],
            })

        # Subplot loadings (haut)
        ax_load = axes[0, gi]
        loading_df = pd.Series(loadings, index=vars_dispo).sort_values()
        ax_load.barh(loading_df.index, loading_df.values,
                     color=['#d73027' if v < 0 else '#1a9850' for v in loading_df])
        ax_load.axvline(0, color='black', linewidth=0.8)
        ve_str = '  '.join([f'PC{k+1}:{var_exp_all[k]*100:.0f}%' for k in range(min(3, len(var_exp_all)))])
        ax_load.set_title(f'Groupe {gi+1} (poids={groupe["poids"]})\\n{ve_str}', fontsize=9)
        if vars_absentes:
            ax_load.set_xlabel(f'absent: {", ".join(vars_absentes)}', fontsize=7, color='orange')

        # Subplot scree (bas)
        ax_var = axes[1, gi]
        n_pc = min(len(var_exp_all), 8)
        bars = ax_var.bar(
            [f'PC{k+1}' for k in range(n_pc)], var_exp_all[:n_pc] * 100,
            color=['#4575b4' if k == 0 else '#abd9e9' for k in range(n_pc)]
        )
        ax_var.axhline(35, color='orange', linestyle='--', linewidth=1, label='seuil 35%')
        ax_var.set_ylim(0, 100)
        ax_var.set_title(f'Scree groupe {gi+1}')
        ax_var.legend(fontsize=8)
        for bar, val in zip(bars, var_exp_all[:n_pc]):
            ax_var.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                        f'{val*100:.0f}%', ha='center', fontsize=8)

    # -- Inter-groupes et etage 2 --
    if n_groupes > 1:
        col_inter = n_groupes

        # Matrice de correlation inter-groupes (PC1 etage 1)
        group_corr_mat = pd.DataFrame(
            {f'G{i+1}': s for i, s in enumerate(group_pc1_series)}
        ).corr()

        # Subplot heatmap inter-groupes (haut)
        ax_inter = axes[0, col_inter]
        im = ax_inter.imshow(group_corr_mat.values, cmap='RdBu_r', vmin=-1, vmax=1)
        labels = [f'G{i+1}' for i in range(n_groupes)]
        ax_inter.set_xticks(range(n_groupes))
        ax_inter.set_yticks(range(n_groupes))
        ax_inter.set_xticklabels(labels)
        ax_inter.set_yticklabels(labels)
        plt.colorbar(im, ax=ax_inter, fraction=0.046, pad=0.04)
        for i in range(n_groupes):
            for j in range(n_groupes):
                ax_inter.text(j, i, f'{group_corr_mat.iloc[i,j]:.2f}',
                              ha='center', va='center', fontsize=12, fontweight='bold')
        ax_inter.set_title('Corr inter-groupes\\n(PC1 etage 1)', fontsize=9)

        for i in range(n_groupes):
            for j in range(i+1, n_groupes):
                all_group_corr.append({
                    'score': score_name, 'groupe_i': i+1, 'groupe_j': j+1,
                    'poids_i': groupes[i]['poids'], 'poids_j': groupes[j]['poids'],
                    'corr_inter_groupe': round(group_corr_mat.iloc[i, j], 4),
                })

        # PCA etage 2
        G = np.column_stack([_zscore_pondere(s, pop).values for s in group_pc1_series])
        final_pc1, final_var_exp_all, _ = _pca_weighted_pc1(G, pop_arr)
        final_var_exp = final_var_exp_all[0]

        # Scree etage 2 (bas)
        ax_s2 = axes[1, col_inter]
        n_pc2 = n_groupes
        bars2 = ax_s2.bar(
            [f'PC{k+1}' for k in range(n_pc2)], final_var_exp_all[:n_pc2] * 100,
            color=['#4575b4' if k == 0 else '#abd9e9' for k in range(n_pc2)]
        )
        ax_s2.axhline(35, color='orange', linestyle='--', linewidth=1)
        ax_s2.set_ylim(0, 100)
        ax_s2.set_title(f'Scree etage 2\\n({n_groupes}G -> 1 score)')
        for bar, val in zip(bars2, final_var_exp_all[:n_pc2]):
            ax_s2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                       f'{val*100:.0f}%', ha='center', fontsize=8)
    else:
        final_pc1     = group_pc1_series[0].values
        final_var_exp = group_var_exps[0][0]

    # Score PCA final (centile pondere)
    score_pca_s = pd.Series(final_pc1, index=df.index)
    if anchor_var and anchor_var in df.columns and score_pca_s.corr(df[anchor_var]) < 0:
        score_pca_s = -score_pca_s
    score_pca_final = _rang_pondere(score_pca_s, pop)
    col_pca = f'{score_name}_pca'
    df[col_pca] = score_pca_final

    # Scatter grouped vs PCA (haut droite)
    corr = df[score_name].corr(df[col_pca])
    ax_sc = axes[0, -1]
    ax_sc.scatter(df[score_name].values, df[col_pca].values, s=6, alpha=0.5, c='steelblue')
    ax_sc.set_xlabel(f'{score_name}\\n(grouped)')
    ax_sc.set_ylabel(col_pca)
    c = 'darkgreen' if corr > 0.85 else ('orange' if corr > 0.6 else 'red')
    ax_sc.set_title(f'Grouped vs PCA  r={corr:.3f}', fontweight='bold', color=c)

    # Distribution (bas droite)
    ax_d = axes[1, -1]
    ax_d.hist(df[score_name].values, bins=30, alpha=0.6, label='grouped', color='steelblue', density=True)
    ax_d.hist(df[col_pca].values,    bins=30, alpha=0.6, label='PCA',     color='tomato',    density=True)
    ax_d.set_xlabel('Score')
    ax_d.set_ylabel('Densite')
    ax_d.set_title('Distributions')
    ax_d.legend()

    plt.tight_layout()
    plt.savefig(f'pca_diag_{score_name}.png', dpi=90, bbox_inches='tight')
    plt.show()

    all_results[score_name] = {
        'n_groupes':        n_groupes,
        'var_exp_g1':       f'{group_var_exps[0][0]*100:.1f}%',
        'var_exp_g2':       f'{group_var_exps[1][0]*100:.1f}%' if n_groupes > 1 else '-',
        'var_exp_final':    f'{final_var_exp*100:.1f}%',
        'corr_grouped_pca': round(corr, 3),
        'anchor_var':       anchor_var,
    }
    all_scores_df[score_name] = df[score_name]
    all_scores_df[col_pca]    = df[col_pca]
"""

CELL_RECAP = """\
recap = pd.DataFrame(all_results).T
recap.index.name = 'Score'

def _color_corr(val):
    try:
        v = float(val)
        if v >= 0.90: return 'background-color: #c7e9c0'
        if v >= 0.80: return 'background-color: #fff7bc'
        if v >= 0.65: return 'background-color: #fec44f'
        return 'background-color: #fc8d59'
    except:
        return ''

try:
    display(recap.style.map(_color_corr, subset=['corr_grouped_pca']))
except AttributeError:
    display(recap.style.applymap(_color_corr, subset=['corr_grouped_pca']))

if all_group_corr:
    print('\\n-- Correlations inter-groupes (PC1 etage 1) --')
    display(pd.DataFrame(all_group_corr).set_index(['score', 'groupe_i', 'groupe_j']))
"""

CELL_ZOOM = """\
# Zoom : heatmap correlations inter-variables + poids manuels vs PCA
for score_name in SCORES_CONFIG_GROUPED:
    for gi, groupe in enumerate(SCORES_CONFIG_GROUPED[score_name]):
        vars_dispo = [v for v in groupe['vars'] if v in df.columns]
        if len(vars_dispo) < 2:
            continue
        corr_mat = df[vars_dispo].corr()
        n = len(vars_dispo)
        fig, axes = plt.subplots(1, 2, figsize=(max(6, n * 1.1 + 2), max(5, n * 0.75)))
        fig.suptitle(f'{score_name} -- Groupe {gi+1} (poids={groupe["poids"]})', fontweight='bold')

        ax = axes[0]
        im = ax.imshow(corr_mat.values, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
        ax.set_xticks(range(n)); ax.set_yticks(range(n))
        ax.set_xticklabels(vars_dispo, rotation=45, ha='right', fontsize=8)
        ax.set_yticklabels(vars_dispo, fontsize=8)
        plt.colorbar(im, ax=ax)
        ax.set_title('Correlations inter-variables')
        for i in range(n):
            for j in range(n):
                ax.text(j, i, f'{corr_mat.iloc[i,j]:.2f}', ha='center', va='center', fontsize=7)

        ax2 = axes[1]
        X_parts = []
        for var in vars_dispo:
            s = df[var].copy().astype(float)
            if var in LOG_TRANSFORM_VARS:
                s = s.clip(lower=1e-6).apply(np.log)
            X_parts.append(_zscore_pondere(s, pop).values)
        X = np.column_stack(X_parts)
        _, _, loadings = _pca_weighted_pc1(X, pop_arr)
        poids_manuels = np.array([groupe['vars'][v] for v in vars_dispo])
        poids_norm = poids_manuels / np.sqrt((poids_manuels**2).sum())
        if np.dot(loadings, poids_norm) < 0:
            loadings = -loadings
        x_pos = np.arange(n)
        ax2.bar(x_pos - 0.175, poids_norm, 0.35, label='Poids manuels (norm)', color='steelblue', alpha=0.8)
        ax2.bar(x_pos + 0.175, loadings,   0.35, label='Loadings PC1 (PCA)',   color='tomato',    alpha=0.8)
        ax2.axhline(0, color='black', linewidth=0.8)
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels(vars_dispo, rotation=45, ha='right', fontsize=8)
        ax2.set_title('Poids manuels vs Loadings PCA')
        ax2.legend(fontsize=9)
        plt.tight_layout()
        plt.savefig(f'pca_zoom_{score_name}_g{gi+1}.png', dpi=90, bbox_inches='tight')
        plt.show()
"""

CELL_EXPORT = """\
# Export CSV + TXT
recap_csv = pd.DataFrame(all_results).T
recap_csv.index.name = 'score'
recap_csv.to_csv('pca_recap.csv')
print('Exporte : pca_recap.csv')

loadings_df = pd.DataFrame(all_loadings)
loadings_df.to_csv('pca_loadings.csv', index=False)
print('Exporte : pca_loadings.csv')

group_corr_df = pd.DataFrame()
if all_group_corr:
    group_corr_df = pd.DataFrame(all_group_corr)
    group_corr_df.to_csv('pca_group_correlations.csv', index=False)
    print('Exporte : pca_group_correlations.csv')

id_cols = [c for c in ['IRIS', 'COM', 'LAB_IRIS', 'pop_totale'] if c in df.columns]
scores_export = df[id_cols].copy()
for col, vals in all_scores_df.items():
    scores_export[col] = vals.values
scores_export.to_csv('pca_scores_iris.csv', index=False)
print('Exporte : pca_scores_iris.csv')

lines = ['=== RAPPORT PCA SCORES DIAGNOSTICS ===', '']
lines += ['-- RECAP --', recap_csv.to_string(), '']
if not group_corr_df.empty:
    lines += ['-- CORRELATIONS INTER-GROUPES (PC1 etage 1) --', group_corr_df.to_string(index=False), '']
lines += ['-- LOADINGS PAR SCORE/GROUPE --']
for score_name in SCORES_CONFIG_GROUPED:
    sub = loadings_df[loadings_df['score'] == score_name]
    lines.append(f'\\n{score_name}:')
    for gi, grp in sub.groupby('groupe'):
        lines.append(
            f'  Groupe {gi} (poids={grp["poids_groupe"].iloc[0]}, '
            f'var_exp_pc1={grp["var_exp_pc1_groupe"].iloc[0]*100:.1f}%)'
        )
        for _, row in grp.sort_values('loading_pc1', ascending=False).iterrows():
            ok = 'OK' if np.sign(row['poids_manuel']) == np.sign(row['loading_pc1']) else '!!'
            lines.append(
                f'    [{ok}] {row["variable"]:<42}  poids_manuel={row["poids_manuel"]:+.2f}'
                f'  loading_pca={row["loading_pc1"]:+.3f}'
            )
with open('pca_rapport.txt', 'w', encoding='utf-8') as f:
    f.write('\\n'.join(lines))
print('Exporte : pca_rapport.txt')
print('\\nTous les fichiers dans test/')
"""

nb = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {"display_name": "vadim_env", "language": "python", "name": "vadim_env"},
        "language_info": {"name": "python", "version": "3.11.0"}
    },
    "cells": [
        md_cell("a1", ["# Diagnostics : Scores composites grouped vs PCA a deux etages\n",
                        "\n",
                        "- Par groupe : variance expliquee PC1, loadings\n",
                        "- Inter-groupes : correlations entre scores de groupe (PC1 etage 1)\n",
                        "- Score final : corr avec grouped, distribution\n",
                        "- Export pca_recap.csv / pca_loadings.csv / pca_group_correlations.csv / pca_scores_iris.csv / pca_rapport.txt"]),
        code_cell("a2", CELL_IMPORTS),
        code_cell("a3", CELL_VARS),
        code_cell("a4", CELL_HELPERS),
        code_cell("a5", CELL_CONFIG),
        code_cell("a6", CELL_GROUPED),
        code_cell("a7", CELL_ANALYSIS),
        code_cell("a8", CELL_RECAP),
        code_cell("a9", CELL_ZOOM),
        code_cell("a10", CELL_EXPORT),
    ]
}

out = os.path.join(os.path.dirname(__file__), "pca_scores_diagnostics.ipynb")
with open(out, "w", encoding="utf-8") as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print(f"Notebook ecrit : {out}")
