"""
build_iris_final.py
Script autonome qui recrée iris/iris_final_socio_politique.csv
depuis les fichiers sources bruts INSEE.

Usage : conda run -n vadim_env python build_iris_final.py [--with-embeddings]
"""

import argparse
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import r2_score, mean_absolute_error

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
        'Q221': 'DISP_MED21',
        'PACT21': 'DISP_PACT21',
        'PPAT21': 'DISP_PPAT21',
        'PPSOC21': 'DISP_PPSOC21',
        'PIMPOT21': 'DISP_PIMPOT21',
        'TP6021': 'DISP_TP6021'
    }
    df_com_full.rename(columns=com_rename_map, inplace=True)
    cols_to_fill = list(com_rename_map.values())

    for col in cols_to_fill:
        if col in df_com_full.columns and df_com_full[col].dtype == object:
            df_com_full[col] = df_com_full[col].str.replace(',', '.').replace(['nd', 'ns', 's'], np.nan)
            df_com_full[col] = pd.to_numeric(df_com_full[col], errors='coerce')

    # Nettoyer FILO IRIS
    filo_cols = [c for c in df_filo_iris.columns if c != 'IRIS']
    for col in filo_cols:
        if df_filo_iris[col].dtype == object:
            df_filo_iris[col] = df_filo_iris[col].str.replace(',', '.').replace(['nd', 'ns', 's'], np.nan)
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
    r2_threshold = 0.73
    corr_threshold = 0.8
    err_rel_threshold = 25.0

    print(f"  {len(vars_cibles)} variables cibles détectées")
    print(f"  Critères : R² > {r2_threshold*100:.0f}% | Corr > {corr_threshold:.2f} | Err Rel < {err_rel_threshold:.1f}%")

    for cible in vars_cibles:
        mask_known = df_num[cible].notna()
        mask_unknown = df_num[cible].isna()

        if mask_unknown.sum() == 0:
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
            print(f"  {cible}: ACCEPTÉ (R²={r2*100:.1f}%, Corr={corr:.2f}, Err={erreur_relative:.1f}%) — {trous} trous bouchés")
        else:
            print(f"  {cible}: rejeté (R²={r2*100:.1f}%, Corr={corr:.2f}, Err={erreur_relative:.1f}%)")

    output_path = os.path.join(PATH_IRIS, "iris_database_machine_learning.csv")
    df_imputed.to_csv(output_path, index=False)
    print(f"  Sauvegardé : {output_path} ({len(df_imputed)} IRIS)")
    return output_path


# =============================================================================
# Fonctions de scoring (rang centile pondéré par population)
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


def make_score(df, pos_vars, neg_vars, pop_col='pop_totale'):
    """Score composite par rang centile pondéré par population."""
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


# Configuration de tous les scores composites
SCORES_CONFIG = {
    # ── Scores sociologiques existants ──
    'score_exploitation': {
        'pos_vars': ['DISP_PPAT21', 'P21_NSAL15P_EMPLOY', 'pct_csp_plus', 'pct_csp_retraite', 'DISP_MED21', 'DISP_PPEN21', 'DISP_PBEN21'],
        'neg_vars': ['DISP_TP6021', 'DISP_PTSA21', 'P21_NSAL15P_AIDFAM', 'DISP_PPLOGT21'],
    },
    'score_domination': {
        'pos_vars': ['pct_csp_plus', 'pct_csp_intermediaire', 'pct_sup5', 'pct_cdi', 'P21_NSAL15P_EMPLOY'],
        'neg_vars': ['pct_csp_ouvrier', 'pct_csp_sans_emploi', 'pct_csp_employe', 'pct_csp_independant', 'pct_sans_diplome', 'pct_capbep', 'pct_cdd', 'pct_interim', 'pct_temps_partiel', 'pct_chomage', 'DISP_TP6021', 'DISP_PPSOC21', 'DISP_PPMINI21'],
    },
    'score_cap_cult': {
        'pos_vars': ['pct_csp_plus', 'pct_csp_intermediaire', 'pct_sup5', 'pct_actifs_velo'],
        'neg_vars': ['pct_csp_ouvrier', 'pct_sans_diplome', 'pct_csp_sans_emploi', 'pct_capbep', 'pct_interim', 'pct_temps_partiel', 'pct_chomage'],
    },
    'score_cap_eco': {
        'pos_vars': ['DISP_PPAT21', 'P21_NSAL15P_EMPLOY', 'pct_csp_plus', 'pct_csp_retraite', 'DISP_MED21', 'DISP_PPEN21', 'DISP_PBEN21'],
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
        'neg_vars': ['pct_immigres', 'pct_actifs_velo', 'pct_actifs_transports', 'pct_actifs_marche', 'pct_etudiants', 'P21_ACTOCC15P_ILT1'],
    },
    'score_urbanite': {
        'pos_vars': ['pct_appart', 'pct_locataires', 'pct_petits_logements', 'pct_voiture_0', 'pct_chauffage_gaz_ville', 'bpe_E_transports_pour1000', 'bpe_total_pour1000'],
        'neg_vars': ['pct_maison', 'pct_voiture_2plus', 'pct_chauffage_fioul', 'pct_grands_logements', 'surface_moyenne', 'pct_garage'],
    },
    'score_confort_residentiel': {
        'pos_vars': ['pct_proprietaires', 'pct_grands_logements', 'surface_moyenne', 'pct_garage', 'nb_pieces_moyen', 'pct_logements_5p_plus'],
        'neg_vars': ['pct_suroccupation', 'pct_petits_logements', 'pct_hlm', 'pct_logvac', 'pct_studios'],
    },
    'score_equipement_public': {
        'pos_vars': ['bpe_total_pour1000', 'bpe_D_sante_pour1000', 'bpe_C_enseignement_pour1000', 'bpe_F_sports_culture_pour1000', 'bpe_A_services_pour1000', 'bpe_B_commerces_pour1000'],
        'neg_vars': [],
    },
    # ── Scores ACP (composantes principales) ──
    'score_pca_pc1_logement_confort': {
        'pos_vars': ['pct_grands_logements', 'pct_garage', 'pct_actifs_voiture', 'pct_logements_5p_plus', 'nb_pieces_moyen', 'pct_voiture_2plus', 'pct_proprietaires', 'pct_maison', 'surface_moyenne'],
        'neg_vars': ['pct_appart', 'pct_locataires', 'pct_voiture_0', 'pct_immigres', 'pct_petits_logements', 'pct_actifs_transports', 'pct_etrangers', 'pct_studios'],
    },
    'score_pca_pc2_composition_diplomes': {
        'pos_vars': ['pct_capbep', 'pct_interim', 'pct_chomage', 'pct_csp_ouvrier', 'DISP_TP6021', 'DISP_PPLOGT21', 'DISP_PPMINI21', 'DISP_PPFAM21', 'DISP_PPSOC21', 'pct_sans_diplome', 'DISP_PIMPOT21'],
        'neg_vars': ['DISP_MED21', 'pct_bac_plus', 'pct_csp_plus', 'pct_sup5', 'DISP_PACT21', 'DISP_PTSA21'],
    },
    'score_pca_pc3_equipements_demographie': {
        'pos_vars': ['pct_csp_intermediaire', 'DISP_PACT21', 'DISP_PTSA21', 'pct_0_19'],
        'neg_vars': ['pct_65_plus', 'age_moyen', 'DISP_PPEN21', 'bpe_total_pour1000', 'pct_csp_retraite', 'bpe_B_commerces_pour1000', 'bpe_G_tourisme_pour1000', 'bpe_D_sante_pour1000', 'pct_actifs_marche', 'bpe_A_services_pour1000'],
    },
    'score_pca_pc4_demographie_chauffage': {
        'pos_vars': ['age_moyen', 'pct_csp_retraite', 'pct_65_plus', 'DISP_PPEN21', 'pct_chauffage_gaz_ville', 'pct_femmes'],
        'neg_vars': ['pct_logements_anciens', 'bpe_A_services_pour1000', 'pct_20_64', 'pct_chauffage_autre', 'pct_chauffage_gaz_bouteille', 'pct_csp_agriculteur', 'bpe_total_pour1000', 'bpe_F_sports_culture_pour1000', 'pct_chauffage_fioul'],
    },
    'score_pca_pc5_equipements_csp': {
        'pos_vars': ['pct_grands_logements', 'pct_csp_sans_emploi', 'DISP_S80S2021', 'pct_temps_partiel', 'pct_inactif', 'pct_etudiants'],
        'neg_vars': ['bpe_total_pour1000', 'bpe_A_services_pour1000', 'bpe_B_commerces_pour1000', 'pct_csp_employe', 'bpe_D_sante_pour1000', 'pct_chauffage_elec', 'pct_logements_recents', 'pct_csp_intermediaire'],
    },
    'score_pca_pc6_equipements_diplomes': {
        'pos_vars': ['pct_inactif', 'bpe_sport_indoor_pour1000', 'bpe_ecole_privee_pour1000', 'bpe_C_enseignement_pour1000', 'pct_hors_menage', 'pct_etudiants'],
        'neg_vars': ['bpe_E_transports_pour1000', 'pct_immigres', 'pct_etrangers', 'pct_cdi', 'pct_actifs_transports', 'pct_actifs_2roues', 'pct_csp_independant', 'DISP_S80S2021'],
    },
    'score_pca_pc7_logement_csp': {
        'pos_vars': ['DISP_S80S2021', 'pct_logements_recents', 'DISP_GI21', 'bpe_A_services_pour1000', 'bpe_total_pour1000', 'pct_csp_sans_emploi', 'pct_inactif', 'pct_csp_independant', 'pct_0_19', 'DISP_PPAT21'],
        'neg_vars': ['pct_logvac', 'pct_logements_anciens', 'pct_csp_agriculteur', 'pct_20_64', 'pct_actifs_velo'],
    },
    'score_pca_pc8_equipements_logement': {
        'pos_vars': ['pct_cdd', 'pct_logements_recents', 'pct_chauffage_elec'],
        'neg_vars': ['bpe_sport_indoor_pour1000', 'bpe_C_enseignement_pour1000', 'bpe_ecole_privee_pour1000', 'pct_chauffage_gaz_ville', 'bpe_F_sports_culture_pour1000', 'pct_logvac', 'bpe_total_pour1000', 'bpe_D_sante_pour1000'],
    },
    # ── Score parti-informé ──
    'score_peripherie_metropole': {
        'pos_vars': ['pct_capbep', 'pct_actifs_voiture', 'pct_maison', 'pct_voiture_2plus', 'nb_pieces_moyen', 'pct_chauffage_fioul'],
        'neg_vars': ['pct_bac_plus', 'pct_sup5', 'pct_csp_plus', 'DISP_GI21', 'DISP_PACT21', 'DISP_RD21', 'pct_actifs_velo', 'DISP_PTSA21', 'pct_studios', 'pct_petits_logements'],
    },
}


def compute_scores(df):
    """Calcule tous les scores composites par rang centile pondéré."""
    print("\n" + "=" * 60)
    print("SCORES : Calcul des scores composites")
    print("=" * 60)

    for score_name, cfg in SCORES_CONFIG.items():
        df[score_name] = make_score(df, cfg['pos_vars'], cfg['neg_vars'])
        print(f"  {score_name}: min={df[score_name].min():.1f} max={df[score_name].max():.1f}")

    print(f"  {len(SCORES_CONFIG)} scores calculés")
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

    n_computed = 20 + 23 + len(bpe_cols) + 21
    print(f"  {n_computed} variables dérivées calculées (démo + logement + énergie + BPE + diplômes + emploi + transport)")
    return df


# =============================================================================
# Embeddings t-SNE / UMAP (optionnel, via --with-embeddings)
# =============================================================================
# Variables socio utilisées pour les embeddings (même liste que analyse_iris.py)
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


def compute_embeddings(df):
    """Calcule les coordonnées t-SNE et UMAP sur tous les IRIS."""
    from sklearn.manifold import TSNE
    try:
        import umap
        HAS_UMAP = True
    except ImportError:
        HAS_UMAP = False
        print("  WARNING: umap-learn non installé, UMAP ignoré")

    print("\n" + "=" * 60)
    print("EMBEDDINGS : t-SNE + UMAP sur tous les IRIS")
    print("=" * 60)

    # Sélectionner les variables disponibles
    var_names = [v for v in EMBEDDING_VARS if v in df.columns]
    print(f"  {len(var_names)} variables socio disponibles sur {len(EMBEDDING_VARS)}")

    pop = df['pop_totale'].values.astype(float)
    X = df[var_names].values.astype(float)

    # Remplacer pop NaN/0 par 1 (poids minimal)
    pop = np.where(np.isfinite(pop) & (pop > 0), pop, 1.0)

    # Imputer les NaN par la médiane
    for j in range(X.shape[1]):
        col = X[:, j]
        nans = ~np.isfinite(col)
        if nans.any():
            valid_mask = np.isfinite(col)
            med = np.median(col[valid_mask]) if valid_mask.sum() > 0 else 0.0
            col[nans] = med

    # Standardisation pondérée par population
    w_norm = pop / pop.sum()
    w_means = (X * w_norm[:, None]).sum(axis=0)
    X_c = X - w_means
    w_stds = np.sqrt((X_c ** 2 * w_norm[:, None]).sum(axis=0))
    w_stds[w_stds < 1e-10] = 1e-10
    X_std = X_c / w_stds

    # PCA réduction à 20 composantes
    C = (X_std * w_norm[:, None]).T @ X_std
    eigenvalues, eigenvectors = np.linalg.eigh(C)
    idx = eigenvalues.argsort()[::-1]
    eigenvectors = eigenvectors[:, idx]
    n_pca_dims = min(20, X_std.shape[1])
    X_pca = X_std @ eigenvectors[:, :n_pca_dims]
    print(f"  PCA : réduction à {n_pca_dims} composantes")

    # t-SNE
    print("  t-SNE en cours (peut prendre quelques minutes)...")
    tsne = TSNE(n_components=2, perplexity=30, max_iter=1000,
                random_state=42, init='pca', learning_rate='auto')
    tsne_coords = tsne.fit_transform(X_pca)
    df['tsne_x'] = tsne_coords[:, 0]
    df['tsne_y'] = tsne_coords[:, 1]
    print(f"  t-SNE terminé : x=[{tsne_coords[:, 0].min():.1f}, {tsne_coords[:, 0].max():.1f}]")

    # UMAP
    if HAS_UMAP:
        print("  UMAP en cours...")
        reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2,
                            random_state=42, metric='euclidean')
        umap_coords = reducer.fit_transform(X_pca)
        df['umap_x'] = umap_coords[:, 0]
        df['umap_y'] = umap_coords[:, 1]
        print(f"  UMAP terminé : x=[{umap_coords[:, 0].min():.1f}, {umap_coords[:, 0].max():.1f}]")

    return df


# =============================================================================
# ÉTAPE 5-6 : Fusion finale + nom_commune + sauvegarde
# =============================================================================
def build_final(ml_path, with_embeddings=False):
    print("\n" + "=" * 60)
    print("ÉTAPE 5-6 : Variables démo + scores + nom_commune + sauvegarde")
    print("=" * 60)

    df_final = pd.read_csv(ml_path, dtype={'IRIS': str, 'COM': str}, low_memory=False)
    print(f"  Base ML : {len(df_final)} IRIS x {len(df_final.columns)} cols")

    # Calcul des variables démographiques dérivées (colonnes P21_/C21_ déjà présentes)
    df_final = compute_demographics(df_final)

    # Calcul des scores composites (nécessite les variables dérivées)
    df_final = compute_scores(df_final)

    # Calcul optionnel des embeddings t-SNE / UMAP
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
        # Scores composites
        'score_exploitation', 'score_precarite', 'score_peripherie_metropole',
        'score_pca_pc1_logement_confort', 'score_peripherie_metropole',
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
