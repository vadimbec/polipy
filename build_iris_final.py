"""
build_iris_final.py
Script autonome qui recrée iris/iris_final_socio_politique.csv
depuis les fichiers sources bruts INSEE.

Usage : conda run -n vadim_env python build_iris_final.py [--with-embeddings]
"""

import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import argparse
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import r2_score, mean_absolute_error
from shared_config import *
from shared_config import (
    _rang_pondere,
)

pd.set_option('display.float_format', lambda x: '%.3f' % x)

PATH_IRIS = "iris/"
PATH_COMMUNE = "communes/"


# =============================================================================
# Utilitaire de lecture INSEE (sep=';', encodage auto)
# =============================================================================
def read_insee(filepath, dtype_dict=None):
    try:
        return pd.read_csv(filepath, sep=";", encoding='utf-8', dtype=dtype_dict, low_memory=False)
    except UnicodeDecodeError:
        return pd.read_csv(filepath, sep=";", encoding='latin1', dtype=dtype_dict, low_memory=False)


# =============================================================================
# ÉTAPE 1-2 : Census + FILO + fallback commune → iris_database_hybride_parfaite.csv
# =============================================================================
def build_hybride():
    print("=" * 60)
    print("ÉTAPE 1-2 : Census + FILO + fallback commune")
    print("=" * 60)

    # Chargement
    df_dipl = read_insee(os.path.join(PATH_IRIS, "base-ic-diplomes-formation-2021.CSV"),
                         {'IRIS': str, 'COM': str})
    df_act = read_insee(os.path.join(PATH_IRIS, "base-ic-activite-residents-2021.CSV"),
                        {'IRIS': str, 'COM': str})
    df_pop = read_insee(os.path.join(PATH_IRIS, "base-ic-evol-struct-pop-2021.CSV"),
                        {'IRIS': str, 'COM': str})
    df_filo_iris = read_insee(os.path.join(PATH_IRIS, "BASE_TD_FILO_IRIS_2021_DISP.csv"),
                              {'IRIS': str})
    df_filo_com = read_insee(os.path.join(PATH_COMMUNE, "FILO2021_DISP_COM.csv"),
                             {'CODGEO': str})
    df_pauvres_com = read_insee(os.path.join(PATH_COMMUNE, "FILO2021_DISP_PAUVRES_COM.csv"),
                                {'CODGEO': str})

    print(f"  Diplômes : {len(df_dipl)} IRIS")
    print(f"  Activité : {len(df_act)} IRIS")
    print(f"  Population 2021 : {len(df_pop)} IRIS")
    print(f"  FILO IRIS : {len(df_filo_iris)} IRIS")

    # Préparer fallback commune
    df_com_full = pd.merge(df_filo_com, df_pauvres_com, on='CODGEO', how='left')
    com_rename_map = {
        # Déjà présents
        'Q221': 'DISP_MED21',
        'TP6021': 'DISP_TP6021',
        'PACT21': 'DISP_PACT21',
        'PPAT21': 'DISP_PPAT21',
        'PPSOC21': 'DISP_PPSOC21',
        'PIMPOT21': 'DISP_PIMPOT21',
        # Déciles et indicateurs de distribution
        'Q121': 'DISP_Q121',
        'Q321': 'DISP_Q321',
        'Q3_Q1': 'DISP_EQ21',
        'RD': 'DISP_RD21',
        'D121': 'DISP_D121',
        'D221': 'DISP_D221',
        'D321': 'DISP_D321',
        'D421': 'DISP_D421',
        'D621': 'DISP_D621',
        'D721': 'DISP_D721',
        'D821': 'DISP_D821',
        'D921': 'DISP_D921',
        'S80S2021': 'DISP_S80S2021',
        'GI21': 'DISP_GI21',
        # Parts de revenus par source
        'PTSA21': 'DISP_PTSA21',
        'PCHO21': 'DISP_PCHO21',
        'PBEN21': 'DISP_PBEN21',
        'PPEN21': 'DISP_PPEN21',
        'PPFAM21': 'DISP_PPFAM21',
        'PPMINI21': 'DISP_PPMINI21',
        'PPLOGT21': 'DISP_PPLOGT21',
    }
    df_com_full.rename(columns=com_rename_map, inplace=True)
    cols_to_fill = list(com_rename_map.values())

    for col in cols_to_fill:
        if col in df_com_full.columns and not pd.api.types.is_numeric_dtype(df_com_full[col]):
            df_com_full[col] = df_com_full[col].astype(str).str.replace(',', '.').replace(['nd', 'ns', 's', 'nan', 'None'], np.nan)
            df_com_full[col] = pd.to_numeric(df_com_full[col], errors='coerce')

    # Nettoyer FILO IRIS (fix pandas 2.x : StringDtype != object)
    filo_cols = [c for c in df_filo_iris.columns if c != 'IRIS']
    for col in filo_cols:
        if not pd.api.types.is_numeric_dtype(df_filo_iris[col]):
            df_filo_iris[col] = df_filo_iris[col].astype(str).str.replace(',', '.').replace(['nd', 'ns', 's', 'nan', 'None'], np.nan)
            df_filo_iris[col] = pd.to_numeric(df_filo_iris[col], errors='coerce')

    # Fusion census (diplômes + activité + population 2021)
    merged_census = pd.merge(
        df_dipl,
        df_act.drop(columns=['COM', 'TYP_IRIS', 'LAB_IRIS'], errors='ignore'),
        on='IRIS', how='outer'
    )
    df_pop['IRIS'] = df_pop['IRIS'].astype(str).str.zfill(9)
    existing_cols = set(merged_census.columns) - {'IRIS'}
    pop_cols_to_drop = [c for c in df_pop.columns if c in existing_cols or c in ('COM', 'TYP_IRIS', 'LAB_IRIS')]
    merged_census = pd.merge(
        merged_census,
        df_pop.drop(columns=pop_cols_to_drop, errors='ignore'),
        on='IRIS', how='outer'
    )
    # + Logement 2022
    df_log = read_insee(os.path.join(PATH_IRIS, "base-ic-logement-2022.CSV"),
                        {'IRIS': str, 'COM': str})
    df_log['IRIS'] = df_log['IRIS'].astype(str).str.zfill(9)
    existing_cols = set(merged_census.columns) - {'IRIS'}
    log_cols_to_drop = [c for c in df_log.columns if c in existing_cols or c in ('COM', 'TYP_IRIS', 'LAB_IRIS')]
    merged_census = pd.merge(
        merged_census,
        df_log.drop(columns=log_cols_to_drop, errors='ignore'),
        on='IRIS', how='outer'
    )
    print(f"  Logement 2022 : {len(df_log)} IRIS ({len(df_log.columns) - len(log_cols_to_drop)} nouvelles cols)")

    # + BPE 2024 (format long → pivot par domaine)
    df_bpe = read_insee(os.path.join(PATH_IRIS, "ds_bpe_iris_2024_geo_2024.csv"),
                        {'GEO': str})
    df_bpe.rename(columns={'GEO': 'IRIS'}, inplace=True)
    df_bpe['IRIS'] = df_bpe['IRIS'].astype(str).str.zfill(9)
    DOM_NAMES = {'A': 'services', 'B': 'commerces', 'C': 'enseignement',
                 'D': 'sante', 'E': 'transports', 'F': 'sports_culture', 'G': 'tourisme'}
    bpe_domain = df_bpe.groupby(['IRIS', 'FACILITY_DOM'])['OBS_VALUE'].sum().unstack(fill_value=0)
    bpe_domain.columns = [f'bpe_{k}_{DOM_NAMES[k]}' for k in bpe_domain.columns]
    bpe_domain['bpe_total'] = bpe_domain.sum(axis=1)
    bpe_domain = bpe_domain.reset_index()
    merged_census = pd.merge(merged_census, bpe_domain, on='IRIS', how='left')
    bpe_cols = [c for c in merged_census.columns if c.startswith('bpe_')]
    merged_census[bpe_cols] = merged_census[bpe_cols].fillna(0)
    print(f"  BPE 2024 : {len(df_bpe)} lignes → {len(bpe_cols)} colonnes agrégées")

    # + BPE éducation enrichi (éducation prioritaire, privé)
    df_bpe_edu = read_insee(os.path.join(PATH_IRIS, "ds_bpe_education_iris_2024_geo_2024.csv"),
                            {'GEO': str})
    df_bpe_edu.rename(columns={'GEO': 'IRIS'}, inplace=True)
    df_bpe_edu['IRIS'] = df_bpe_edu['IRIS'].astype(str).str.zfill(9)
    bpe_ep = df_bpe_edu[df_bpe_edu['EP'].astype(str) == '1'].groupby('IRIS')['OBS_VALUE'].sum().rename('bpe_educ_prioritaire')
    bpe_pr = df_bpe_edu[df_bpe_edu['SCHOOL_SECTOR'].astype(str) == 'PR'].groupby('IRIS')['OBS_VALUE'].sum().rename('bpe_ecole_privee')
    merged_census = merged_census.join(bpe_ep, on='IRIS', how='left')
    merged_census = merged_census.join(bpe_pr, on='IRIS', how='left')
    merged_census[['bpe_educ_prioritaire', 'bpe_ecole_privee']] = merged_census[['bpe_educ_prioritaire', 'bpe_ecole_privee']].fillna(0)
    print(f"  BPE éducation : {len(df_bpe_edu)} lignes (EP, privé)")

    # + BPE sport/culture enrichi (indoor, accessibilité)
    df_bpe_sp = read_insee(os.path.join(PATH_IRIS, "ds_bpe_sport_culture_iris_2024_geo_2024.csv"),
                           {'GEO': str})
    df_bpe_sp.rename(columns={'GEO': 'IRIS'}, inplace=True)
    df_bpe_sp['IRIS'] = df_bpe_sp['IRIS'].astype(str).str.zfill(9)
    # Sport = sous-domaines F1 (sports) + F2 (loisirs)
    df_sport = df_bpe_sp[df_bpe_sp['FACILITY_SDOM'].isin(['F1', 'F2'])]
    bpe_indoor = df_sport[df_sport['INDOOR'].astype(str) == '1'].groupby('IRIS')['OBS_VALUE'].sum().rename('bpe_sport_indoor')
    bpe_access = df_sport[df_sport['PRACTICE_AREA_ACCESSIBILITY'].astype(str) == '1'].groupby('IRIS')['OBS_VALUE'].sum().rename('bpe_sport_accessible')
    bpe_sport_tot = df_sport.groupby('IRIS')['OBS_VALUE'].sum().rename('bpe_sport_total')
    merged_census = merged_census.join(bpe_indoor, on='IRIS', how='left')
    merged_census = merged_census.join(bpe_access, on='IRIS', how='left')
    merged_census = merged_census.join(bpe_sport_tot, on='IRIS', how='left')
    merged_census[['bpe_sport_indoor', 'bpe_sport_accessible', 'bpe_sport_total']] = merged_census[['bpe_sport_indoor', 'bpe_sport_accessible', 'bpe_sport_total']].fillna(0)
    print(f"  BPE sport/culture : {len(df_bpe_sp)} lignes (indoor, accessibilité)")

    # + FILO IRIS
    final_db = pd.merge(merged_census, df_filo_iris, on='IRIS', how='left')
    # + fallback commune
    final_db = pd.merge(
        final_db,
        df_com_full[['CODGEO'] + cols_to_fill],
        left_on='COM', right_on='CODGEO', how='left'
    )

    # Bouchage de trous : IRIS d'abord, commune en fallback
    final_db = final_db.copy()  # defragmente avant la boucle de fillna
    for col in cols_to_fill:
        col_iris = col + "_x"
        col_com = col + "_y"
        if col_iris in final_db.columns and col_com in final_db.columns:
            final_db[col] = final_db[col_iris].fillna(final_db[col_com])
            final_db.drop(columns=[col_iris, col_com], inplace=True)

    final_db.drop(columns=['CODGEO'], errors='ignore', inplace=True)

    output_path = os.path.join(PATH_IRIS, "iris_database_hybride_parfaite.csv")
    final_db.to_csv(output_path, index=False)
    print(f"  Sauvegardé : {output_path} ({len(final_db)} IRIS x {len(final_db.columns)} cols)")
    return output_path


# =============================================================================
# ÉTAPE 3 : Imputation ML des variables FILO censurées
# =============================================================================
def build_ml(hybride_path):
    print("\n" + "=" * 60)
    print("ÉTAPE 3 : Imputation ML des variables FILO censurées")
    print("=" * 60)

    df = pd.read_csv(hybride_path, low_memory=False)
    df_imputed = df.copy()

    # Préparation features
    df_num = df.select_dtypes(include=['int16', 'int32', 'int64', 'float16', 'float32', 'float64']).copy()

    if 'P21_ACT1564' in df_num.columns:
        cols_pcs = [c for c in df_num.columns if c.startswith('C21_ACT1564_CS')]
        for c in cols_pcs:
            df_num[f"PCT_{c}"] = (df_num[c] / df_num['P21_ACT1564']) * 100

    if 'P21_POP1564' in df_num.columns:
        cols_dipl = [c for c in df_num.columns if c.startswith('P21_NSCOL15P')]
        for c in cols_dipl:
            df_num[f"PCT_{c}"] = (df_num[c] / df_num['P21_POP1564']) * 100

    df_num = df_num.replace([np.inf, -np.inf], np.nan)

    # Détection cibles
    missing_pct = (df_num.isna().sum() / len(df_num)) * 100
    vars_cibles = missing_pct[(missing_pct > 10) & (missing_pct.index.str.startswith('DISP_'))].index.tolist()

    cols_a_exclure = [c for c in df_num.columns if c.startswith('DISP_') and c != 'DISP_MED21']
    X_cols = [c for c in df_num.columns if c not in vars_cibles and c not in cols_a_exclure]

    # Seuils
    r2_threshold = 0.65
    corr_threshold = 0.8
    err_rel_threshold = 25.0

    print(f"  {len(vars_cibles)} variables cibles détectées")
    print(f"  Critères : R² > {r2_threshold*100:.0f}% | Corr > {corr_threshold:.2f} | Err Rel < {err_rel_threshold:.1f}%")

    imputation_flags = {}  # {varname: boolean Series} — True = imputé par ML

    for cible in vars_cibles:
        mask_known = df_num[cible].notna()
        mask_unknown = df_num[cible].isna()

        if mask_unknown.sum() == 0:
            imputation_flags[cible] = pd.Series(False, index=df.index)
            continue

        X = df_num[X_cols]
        y = df_num[cible]

        X_train, X_test, y_train, y_test = train_test_split(
            X[mask_known], y[mask_known], test_size=0.2, random_state=42
        )

        model = HistGradientBoostingRegressor(
            max_iter=300, learning_rate=0.05, max_depth=12, random_state=42
        )
        model.fit(X_train, y_train)

        y_pred_test = model.predict(X_test)
        r2 = r2_score(y_test, y_pred_test)
        mae = mean_absolute_error(y_test, y_pred_test)
        moyenne_cible = y_test.mean()
        corr = np.corrcoef(y_test, y_pred_test)[0, 1] if np.std(y_pred_test) > 0 else 0
        erreur_relative = (mae / moyenne_cible) * 100 if moyenne_cible != 0 else np.nan

        if r2 > r2_threshold and corr > corr_threshold and erreur_relative < err_rel_threshold:
            trous = mask_unknown.sum()
            df_imputed.loc[mask_unknown, cible] = model.predict(X[mask_unknown])
            imputation_flags[cible] = mask_unknown.copy()  # True là où on a imputé
            print(f"  {cible}: ACCEPTÉ (R²={r2*100:.1f}%, Corr={corr:.2f}, Err={erreur_relative:.1f}%) — {trous} trous bouchés")
        else:
            imputation_flags[cible] = pd.Series(False, index=df.index)
            print(f"  {cible}: rejeté (R²={r2*100:.1f}%, Corr={corr:.2f}, Err={erreur_relative:.1f}%)")

    # Ajout des colonnes de flag ML au DataFrame
    n_flags = 0
    for varname, flag in imputation_flags.items():
        df_imputed[f'ml_imputed_{varname}'] = flag.astype(bool)
        n_imputed = flag.sum()
        if n_imputed > 0:
            n_flags += 1
            print(f"  Flag ml_imputed_{varname} : {n_imputed} IRIS imputés")
    print(f"  {n_flags} variables avec imputations ML flaggées")

    output_path = os.path.join(PATH_IRIS, "iris_database_machine_learning.csv")
    df_imputed.to_csv(output_path, index=False)
    print(f"  Sauvegardé : {output_path} ({len(df_imputed)} IRIS)")
    return output_path



def compute_scores(df):
    """Calcule tous les scores composites (méthode PCA groupée) depuis shared_config."""
    print("\n" + "=" * 60)
    print("SCORES : Calcul des scores composites (PCA groupée)")
    print("=" * 60)

    df = df.copy()  # defragmente le DataFrame avant d'ajouter des colonnes

    # Colonne population interne requise par les fonctions shared_config
    if '_pop' not in df.columns:
        pop_src = 'pop_totale' if 'pop_totale' in df.columns else None
        df['_pop'] = df[pop_src].fillna(df[pop_src].median()) if pop_src else 2000.0

    # Scores composites via SCORES_CONFIG_GROUPED_PCA (méthode PCA à deux étages)
    for score_name, groupes in SCORES_CONFIG_GROUPED_PCA.items():
        anchor = SCORES_PCA_ANCHORS.get(score_name)
        df[score_name], diag = make_score_pca_grouped(groupes, df, anchor_var=anchor)
        fve = diag.get('final_var_exp', 0)
        print(f"  {score_name}: var_exp={fve*100:.1f}%  min={df[score_name].min():.1f} max={df[score_name].max():.1f}")
    print(f"  {len(SCORES_CONFIG_GROUPED_PCA)} scores PCA calculés")

    # Vraie ACP pondérée (score_pca_1 .. score_pca_8) — modifie df en place
    print("\n  ACP vraie pondérée :")
    compute_pca_vraie(df, n_components=8)
    print("  ACP vraie calculée (score_pca_1 .. score_pca_8)")

    # ── Versions strictes (sans IRIS imputés ML) ─────────────────────────────
    print("\n  Calcul versions strictes (exclusion IRIS imputés ML)...")

    # Scores composites stricts
    for score_name, groupes in SCORES_CONFIG_GROUPED_PCA.items():
        strict_col = f'{score_name}_strict'
        disp_vars = [v for g in groupes for v in g.get('vars', {}) if v.startswith('DISP_')]
        if not disp_vars:
            # Pas de dépendance DISP_* : strict == full
            df[strict_col] = df[score_name]
            continue
        flag_cols = [f'ml_imputed_{v}' for v in disp_vars if f'ml_imputed_{v}' in df.columns]
        if not flag_cols:
            df[strict_col] = df[score_name]
            continue
        imputed_mask = df[flag_cols].any(axis=1)
        n_excl = imputed_mask.sum()
        df_sub = df[~imputed_mask].copy()
        if len(df_sub) < 50:
            df[strict_col] = np.nan
            print(f"  {strict_col}: sous-ensemble trop petit ({len(df_sub)}), NaN")
            continue
        anchor = SCORES_PCA_ANCHORS.get(score_name)
        strict_series, _ = make_score_pca_grouped(groupes, df_sub, anchor_var=anchor)
        df[strict_col] = np.nan
        df.loc[~imputed_mask, strict_col] = strict_series
        print(f"  {strict_col}: {len(df_sub)} IRIS ({n_excl} exclus)")

    # ACP stricte
    print("\n  ACP vraie pondérée (strict)...")
    _emb_disp = [v for v in EMBEDDING_VARS_PCA if v.startswith('DISP_')]
    flag_cols_pca = [f'ml_imputed_{v}' for v in _emb_disp if f'ml_imputed_{v}' in df.columns]
    if flag_cols_pca:
        pca_imputed_mask = df[flag_cols_pca].any(axis=1)
        n_excl_pca = pca_imputed_mask.sum()
        df_strict_pca = df[~pca_imputed_mask].copy()
        if len(df_strict_pca) >= 50:
            compute_pca_vraie(df_strict_pca, n_components=8)  # modifie en place
            for k in range(1, 9):
                df[f'score_pca_{k}_strict'] = np.nan
                df.loc[~pca_imputed_mask, f'score_pca_{k}_strict'] = df_strict_pca[f'score_pca_{k}']
            print(f"  ACP stricte : {len(df_strict_pca)} IRIS ({n_excl_pca} exclus)")
        else:
            for k in range(1, 9):
                df[f'score_pca_{k}_strict'] = np.nan
            print(f"  ACP stricte : sous-ensemble trop petit, NaN")
    else:
        # Pas de flags PCA disponibles : strict == full
        for k in range(1, 9):
            df[f'score_pca_{k}_strict'] = df.get(f'score_pca_{k}', np.nan)
        print("  ACP stricte : aucun flag DISP_* PCA trouvé, strict == full")

    return df


# =============================================================================
# ÉTAPE 4 : Calcul variables démographiques dérivées (sur df déjà fusionné)
# =============================================================================
def compute_demographics(df):
    """Ajoute les variables démographiques calculées au dataframe in-place."""
    print("\n" + "=" * 60)
    print("ÉTAPE 4 : Variables démographiques dérivées")
    print("=" * 60)

    AGE_MAP = {
        "P21_POP0002": 1,  "P21_POP0305": 4,  "P21_POP0610": 8,
        "P21_POP1117": 14, "P21_POP1824": 21, "P21_POP2539": 32,
        "P21_POP4054": 47, "P21_POP5564": 60, "P21_POP6579": 72, "P21_POP80P": 90
    }
    CSP_MAPPING = {
        "pct_csp_agriculteur":   "C21_POP15P_CS1",
        "pct_csp_independant":   "C21_POP15P_CS2",
        "pct_csp_plus":          "C21_POP15P_CS3",
        "pct_csp_intermediaire": "C21_POP15P_CS4",
        "pct_csp_employe":       "C21_POP15P_CS5",
        "pct_csp_ouvrier":       "C21_POP15P_CS6",
        "pct_csp_retraite":      "C21_POP15P_CS7",
        "pct_csp_sans_emploi":   "C21_POP15P_CS8",
    }

    pop = df['P21_POP'].replace(0, np.nan)
    pop15 = df['C21_POP15P'].replace(0, np.nan)

    # Age moyen pondéré
    df['age_moyen'] = sum(age * df[col] for col, age in AGE_MAP.items()) / pop
    df['pop_totale'] = df['P21_POP']

    # Tranches d'âge
    df['pct_0_19']    = df['P21_POP0019'] / pop * 100
    df['pct_20_64']   = df['P21_POP2064'] / pop * 100
    df['pct_65_plus'] = df['P21_POP65P']  / pop * 100

    # Immigration
    df['pct_etrangers'] = df['P21_POP_ETR'] / pop * 100
    df['pct_immigres']  = df['P21_POP_IMM'] / pop * 100

    # CSP
    for col_out, col_in in CSP_MAPPING.items():
        df[col_out] = df[col_in] / pop15 * 100

    # Nouvelles variables
    pop_h15 = df['C21_H15P'].replace(0, np.nan)
    pop_f15 = df['C21_F15P'].replace(0, np.nan)
    df['pct_femmes']        = df['P21_POPF'] / pop * 100
    df['taille_menage_moy'] = df['P21_POP'] / df['P21_PMEN'].replace(0, np.nan)
    df['pct_hors_menage']   = df['P21_PHORMEN'] / pop * 100
    df['ecart_csp_plus_hf'] = (df['C21_H15P_CS3'] / pop_h15 - df['C21_F15P_CS3'] / pop_f15) * 100

    # ── Logement (2022) ──
    rp = df['P22_RP'].replace(0, np.nan)
    log_tot = df['P22_LOG'].replace(0, np.nan)
    achtot = df['P22_RP_ACHTOT'].replace(0, np.nan)
    norme = df['C22_RP_NORME'].replace(0, np.nan)

    df['pct_proprietaires']     = df['P22_RP_PROP'] / rp * 100
    df['pct_locataires']        = df['P22_RP_LOC'] / rp * 100
    df['pct_hlm']               = df['P22_RP_LOCHLMV'] / rp * 100
    df['pct_logvac']            = df['P22_LOGVAC'] / log_tot * 100
    df['pct_maison']            = df['P22_MAISON'] / log_tot * 100
    df['pct_appart']            = df['P22_APPART'] / log_tot * 100
    df['pct_petits_logements']  = (df['P22_RP_M30M2'] + df['P22_RP_3040M2']) / rp * 100
    df['pct_grands_logements']  = df['P22_RP_120M2P'] / rp * 100
    df['pct_logements_anciens'] = df['P22_RP_ACH1919'] / achtot * 100
    df['pct_logements_recents'] = df['P22_RP_ACH2019'] / achtot * 100
    df['pct_voiture_0']         = (rp - df['P22_RP_VOIT1P']) / rp * 100
    df['pct_voiture_2plus']     = df['P22_RP_VOIT2P'] / rp * 100
    df['pct_suroccupation']     = (df['C22_RP_SUROCC_MOD'] + df['C22_RP_SUROCC_ACC']) / norme * 100

    # Clipper tous les pourcentages logement à [0, 100]
    pct_log_vars = ['pct_proprietaires', 'pct_locataires', 'pct_hlm',
                    'pct_logvac', 'pct_maison', 'pct_appart',
                    'pct_petits_logements', 'pct_grands_logements',
                    'pct_logements_anciens', 'pct_logements_recents',
                    'pct_voiture_0', 'pct_voiture_2plus', 'pct_suroccupation']
    for v in pct_log_vars:
        df[v] = df[v].clip(0, 100)

    SURF_BRACKETS = {
        'P22_RP_M30M2': 20, 'P22_RP_3040M2': 35, 'P22_RP_4060M2': 50,
        'P22_RP_6080M2': 70, 'P22_RP_80100M2': 90, 'P22_RP_100120M2': 110,
        'P22_RP_120M2P': 140
    }
    df['surface_moyenne'] = sum(df[col] * mid for col, mid in SURF_BRACKETS.items()) / rp
    
    # ── Énergie de chauffage ──
    df['pct_chauffage_elec']          = df['P22_RP_CELEC'] / rp * 100
    df['pct_chauffage_fioul']         = df['P22_RP_CFIOUL'] / rp * 100
    df['pct_chauffage_gaz_ville']     = df['P22_RP_CGAZV'] / rp * 100
    df['pct_chauffage_gaz_bouteille'] = df['P22_RP_CGAZB'] / rp * 100
    df['pct_chauffage_autre']         = df['P22_RP_CAUT'] / rp * 100

    # ── Confort du logement ──
    df['pct_garage']            = df['P22_RP_GARL'] / rp * 100
    df['nb_pieces_moyen']       = df['P22_NBPI_RP'] / rp
    df['pct_studios']           = df['P22_RP_1P'] / rp * 100
    df['pct_logements_5p_plus'] = df['P22_RP_5PP'] / rp * 100

    # Clipper les nouveaux pourcentages à [0, 100]
    pct_new_vars = ['pct_chauffage_elec', 'pct_chauffage_fioul',
                    'pct_chauffage_gaz_ville', 'pct_chauffage_gaz_bouteille',
                    'pct_chauffage_autre', 'pct_garage', 'pct_studios',
                    'pct_logements_5p_plus']
    for v in pct_new_vars:
        df[v] = df[v].clip(0, 100)

    # ── Équipements BPE pour 1000 hab ──
    # NaN pour IRIS < 50 hab (ratios instables), puis winsorisation au p99
    pop_k = (df['pop_totale'] / 1000).replace(0, np.nan)
    low_pop_mask = df['pop_totale'] < 50
    bpe_cols = [c for c in df.columns if c.startswith('bpe_') and not c.endswith('_pour1000')]
    for col in bpe_cols:
        df[f'{col}_pour1000'] = df[col] / pop_k
        df.loc[low_pop_mask, f'{col}_pour1000'] = np.nan
    # Winsoriser au p99
    bpe_pour1000_cols = [f'{c}_pour1000' for c in bpe_cols]
    for col in bpe_pour1000_cols:
        p99 = df[col].quantile(0.99)
        df[col] = df[col].clip(upper=p99)

    # ── Accessibilité sport (ratio interne, pas pour-1000) ──
    sport_tot = df['bpe_sport_total'].replace(0, np.nan)
    df['pct_sport_accessible'] = (df['bpe_sport_accessible'] / sport_tot * 100).clip(0, 100)

    # ── Diplômes ──
    _nscol = df['P21_NSCOL15P'].replace(0, np.nan)
    df['pct_sup5']         = (df['P21_NSCOL15P_SUP5'] / _nscol * 100).clip(0, 100)
    df['pct_sans_diplome'] = (df['P21_NSCOL15P_DIPLMIN'] / _nscol * 100).clip(0, 100)
    df['pct_bac_plus']     = ((df['P21_NSCOL15P_SUP2'] + df['P21_NSCOL15P_SUP34'] + df['P21_NSCOL15P_SUP5']) / _nscol * 100).clip(0, 100)
    df['pct_capbep']       = (df['P21_NSCOL15P_CAPBEP'] / _nscol * 100).clip(0, 100)

    # ── Emploi ──
    df['pct_chomage']   = (df['P21_CHOM1564'] / df['P21_ACT1564'].replace(0, np.nan) * 100).clip(0, 100)
    df['pct_inactif']   = (df['P21_INACT1564'] / df['P21_POP1564'].replace(0, np.nan) * 100).clip(0, 100)
    df['pct_etudiants'] = (df['P21_ETUD1564'] / df['P21_POP1564'].replace(0, np.nan) * 100).clip(0, 100)

    _sal = df['P21_SAL15P'].replace(0, np.nan)
    df['pct_cdi']           = (df['P21_SAL15P_CDI'] / _sal * 100).clip(0, 100)
    df['pct_cdd']           = (df['P21_SAL15P_CDD'] / _sal * 100).clip(0, 100)
    df['pct_interim']       = (df['P21_SAL15P_INTERIM'] / _sal * 100).clip(0, 100)
    df['pct_temps_partiel'] = (df['P21_SAL15P_TP'] / _sal * 100).clip(0, 100)

    # ── Transport des actifs ──
    _actocc = df['P21_ACTOCC15P'].replace(0, np.nan)
    df['pct_actifs_voiture']    = (df['C21_ACTOCC15P_VOIT'] / _actocc * 100).clip(0, 100)
    df['pct_actifs_transports'] = (df['C21_ACTOCC15P_TCOM'] / _actocc * 100).clip(0, 100)
    df['pct_actifs_velo']       = (df['C21_ACTOCC15P_VELO'] / _actocc * 100).clip(0, 100)
    df['pct_actifs_2roues']     = (df['C21_ACTOCC15P_2ROUESMOT'] / _actocc * 100).clip(0, 100)
    df['pct_actifs_marche']     = (df['C21_ACTOCC15P_MAR'] / _actocc * 100).clip(0, 100)

    # Employeurs en proportion des actifs (variable dérivée pour SCORES_CONFIG_GROUPED)
    df['pct_employeurs'] = (
        df['P21_NSAL15P_EMPLOY'] / df['P21_ACT1564'].replace(0, np.nan) * 100
    ).clip(0, 100)

    n_computed = 20 + 23 + len(bpe_cols) + 21
    print(f"  {n_computed} variables dérivées calculées (démo + logement + énergie + BPE + diplômes + emploi + transport)")
    return df


# =============================================================================
# Embeddings t-SNE / UMAP (optionnel, via --with-embeddings)
# =============================================================================
# Variables socio utilisées pour les embeddings

def compute_pca_scores(df, n_components=8, pop_col='pop_totale'):
    """
    Calcule les vraies composantes ACP pondérées par population.
    Produit les colonnes score_pca_1 .. score_pca_8 (rang centile pondéré, plage -50..+50).
    """
    var_names = [v for v in EMBEDDING_VARS_PCA if v in df.columns]
    print(f"\n  ACP vraie : {len(var_names)} variables disponibles sur {len(EMBEDDING_VARS_PCA)}")

    pop = df[pop_col].values.astype(float)
    pop = np.where(np.isfinite(pop) & (pop > 0), pop, 1.0)

    X = df[var_names].apply(lambda c: pd.to_numeric(c, errors='coerce')).values.astype(float)
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
    df = df.copy()
    for k in range(n_components):
        col_name = f'score_pca_{k+1}'
        s = pd.Series(X_pca[:, k], index=df.index)
        df[col_name] = _rang_pondere(s, pop_series)
        loadings = pd.Series(eigenvectors[:, k], index=var_names)
        top_pos = loadings.nlargest(5).index.tolist()
        top_neg = loadings.nsmallest(5).index.tolist()
        print(f"  {col_name}: + {top_pos}")
        print(f"  {col_name}: - {top_neg}")

    return df


def compute_embeddings(df):
    """Calcule les coordonnées t-SNE et UMAP sur les composantes PCA déjà calculées (score_pca_1..8)."""
    from sklearn.manifold import TSNE
    try:
        import umap
        HAS_UMAP = True
    except ImportError:
        HAS_UMAP = False
        print("  WARNING: umap-learn non installé, UMAP ignoré")

    print("\n" + "=" * 60)
    print("EMBEDDINGS : t-SNE + UMAP sur composantes PCA")
    print("=" * 60)

    # Utilise les score_pca_* déjà calculés par compute_pca_vraie (via compute_scores)
    pca_cols = [f'score_pca_{i}' for i in range(1, 9) if f'score_pca_{i}' in df.columns]
    if not pca_cols:
        print("  ERREUR: score_pca_* non disponibles — compute_scores doit être appelé avant")
        return df
    X_pca = df[pca_cols].values.astype(float)
    print(f"  Entrée : {len(pca_cols)} composantes PCA (score_pca_1..{len(pca_cols)})")

    # t-SNE (full)
    print("  t-SNE en cours (peut prendre quelques minutes)...")
    tsne = TSNE(n_components=2, perplexity=30, max_iter=1000,
                random_state=42, init='pca', learning_rate='auto')
    tsne_coords = tsne.fit_transform(X_pca)
    df['tsne_x'] = tsne_coords[:, 0]
    df['tsne_y'] = tsne_coords[:, 1]
    print(f"  t-SNE terminé : x=[{tsne_coords[:, 0].min():.1f}, {tsne_coords[:, 0].max():.1f}]")

    # UMAP (full)
    if HAS_UMAP:
        print("  UMAP en cours...")
        reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2,
                            random_state=42, metric='euclidean')
        umap_coords = reducer.fit_transform(X_pca)
        df['umap_x'] = umap_coords[:, 0]
        df['umap_y'] = umap_coords[:, 1]
        print(f"  UMAP terminé : x=[{umap_coords[:, 0].min():.1f}, {umap_coords[:, 0].max():.1f}]")

    # Versions strictes (basées sur score_pca_*_strict)
    strict_pca_cols = [f'score_pca_{i}_strict' for i in range(1, 9) if f'score_pca_{i}_strict' in df.columns]
    if strict_pca_cols:
        # Masque : IRIS avec au moins un score_pca_*_strict NaN sont exclus
        strict_nan_mask = df[strict_pca_cols].isna().any(axis=1)
        df_strict = df.loc[~strict_nan_mask, strict_pca_cols].copy()
        X_pca_strict = df_strict.values.astype(float)
        n_strict = len(df_strict)
        print(f"\n  t-SNE strict ({n_strict} IRIS, {strict_nan_mask.sum()} exclus)...")
        tsne_strict = TSNE(n_components=2, perplexity=30, max_iter=1000,
                           random_state=42, init='pca', learning_rate='auto')
        tsne_strict_coords = tsne_strict.fit_transform(X_pca_strict)
        df['tsne_x_strict'] = np.nan
        df['tsne_y_strict'] = np.nan
        df.loc[~strict_nan_mask, 'tsne_x_strict'] = tsne_strict_coords[:, 0]
        df.loc[~strict_nan_mask, 'tsne_y_strict'] = tsne_strict_coords[:, 1]
        print(f"  t-SNE strict terminé")

        if HAS_UMAP:
            print("  UMAP strict en cours...")
            reducer_strict = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2,
                                       random_state=42, metric='euclidean')
            umap_strict_coords = reducer_strict.fit_transform(X_pca_strict)
            df['umap_x_strict'] = np.nan
            df['umap_y_strict'] = np.nan
            df.loc[~strict_nan_mask, 'umap_x_strict'] = umap_strict_coords[:, 0]
            df.loc[~strict_nan_mask, 'umap_y_strict'] = umap_strict_coords[:, 1]
            print(f"  UMAP strict terminé")

    return df


# =============================================================================
# ÉTAPE 5-6 : Fusion finale + nom_commune + sauvegarde
# =============================================================================
def build_final(ml_path, with_embeddings=False):
    print("\n" + "=" * 60)
    print("ÉTAPE 5-6 : Variables démo + scores + nom_commune + sauvegarde")
    print("=" * 60)

    df_final = pd.read_csv(ml_path, dtype={'IRIS': str, 'COM': str}, low_memory=False)
    df_final = df_final.copy()  # defragmente avant les 77+ colonnes dérivées
    print(f"  Base ML : {len(df_final)} IRIS x {len(df_final.columns)} cols")

    # Calcul des variables démographiques dérivées (colonnes P21_/C21_ déjà présentes)
    df_final = compute_demographics(df_final)

    # Filtrage IRIS à population nulle
    n_before = len(df_final)
    df_final = df_final[df_final['pop_totale'].fillna(0) > 0].copy()
    n_removed = n_before - len(df_final)
    if n_removed > 0:
        print(f"  Filtrage pop==0 : {n_removed} IRIS supprimés, {len(df_final)} restants")

    # Calcul des scores composites
    df_final = compute_scores(df_final)

    # Calcul des scores PCA
    df_final = compute_pca_scores(df_final)

    # Calcul optionnel t-SNE / UMAP
    if with_embeddings:
        df_final = compute_embeddings(df_final)

    # Noms de communes (COG 2026)
    df_cog = pd.read_csv(
        "cog_ensemble_2026_csv/v_commune_2026.csv",
        dtype={'COM': str},
        usecols=['COM', 'LIBELLE']
    ).rename(columns={'LIBELLE': 'nom_commune'})
    df_cog['COM'] = df_cog['COM'].str.zfill(5)
    df_cog = df_cog.drop_duplicates(subset='COM', keep='first')

    # Ajouter nom_commune
    df_final['_COM5'] = df_final['COM'].astype(str).str.zfill(5)
    df_final = pd.merge(df_final, df_cog, left_on='_COM5', right_on='COM', how='left', suffixes=('', '_cog'))
    df_final.drop(columns=['_COM5', 'COM_cog'], errors='ignore', inplace=True)

    # Sauvegarde
    output_path = os.path.join(PATH_IRIS, "iris_final_socio_politique.csv")
    df_final.to_csv(output_path, index=False)
    print(f"  Résultat : {len(df_final)} IRIS x {len(df_final.columns)} colonnes")
    print(f"  Sauvegardé : {output_path}")

    # Vérifications
    print("\n  === VÉRIFICATIONS ===")
    print(f"  IRIS total : {len(df_final)} IRIS")

    cols_verif = [
        'pop_totale', 'age_moyen', 'pct_0_19', 'pct_20_64', 'pct_65_plus',
        'pct_etrangers', 'pct_immigres',
        'pct_csp_agriculteur', 'pct_csp_plus', 'pct_csp_ouvrier',
        'pct_csp_retraite', 'pct_csp_sans_emploi',
        'pct_femmes', 'taille_menage_moy', 'pct_hors_menage', 'ecart_csp_plus_hf',
        'DISP_MED21', 'DISP_TP6021', 'DISP_PPAT21',
        'pct_proprietaires', 'pct_hlm', 'pct_maison', 'surface_moyenne',
        'pct_suroccupation', 'bpe_total_pour1000', 'bpe_D_sante_pour1000',
        'pct_chauffage_elec', 'pct_chauffage_fioul', 'nb_pieces_moyen',
        'pct_garage', 'pct_studios',
        'bpe_educ_prioritaire_pour1000', 'bpe_ecole_privee_pour1000',
        'bpe_sport_indoor_pour1000', 'pct_sport_accessible',
        # Variables dérivées ajoutées
        'pct_bac_plus', 'pct_chomage', 'pct_actifs_voiture',
        # Scores composites (méthode PCA)
        'score_exploitation', 'score_domination', 'score_cap_eco', 'score_cap_cult',
        'score_precarite', 'score_rentier', 'score_urbanite',
        'score_confort_residentiel', 'score_equipement_public', 'score_dependance_carbone',
        # Vraie ACP
        'score_pca_1', 'score_pca_2', 'score_pca_3',
    ]
    for col in cols_verif:
        if col in df_final.columns:
            nan_pct = df_final[col].isna().mean() * 100
            print(f"    {col:<30} NaN: {nan_pct:.1f}%")
        else:
            print(f"    {col:<30} MANQUANT")



# =============================================================================
# MAIN
# =============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Reconstruction du CSV IRIS final")
    parser.add_argument('--with-embeddings', action='store_true',
                        help="Calculer les coordonnées t-SNE et UMAP (ajoute ~3-5 min)")
    args = parser.parse_args()

    print("=" * 60)
    print("  build_iris_final.py — Reconstruction complète")
    if args.with_embeddings:
        print("  (avec embeddings t-SNE / UMAP)")
    print("=" * 60)

    hybride_path = build_hybride()
    ml_path = build_ml(hybride_path)
    build_final(ml_path, with_embeddings=args.with_embeddings)

    print("\nTerminé.")
