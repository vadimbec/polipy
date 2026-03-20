"""
build_iris_final.py
Script autonome qui recrée iris/iris_final_socio_politique_bis.csv
depuis les fichiers sources bruts INSEE.

Usage : conda run -n vadim_env python build_iris_final.py
"""

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

    n_computed = 20 + 23 + len(bpe_cols)
    print(f"  {n_computed} variables dérivées calculées (démo + logement + énergie + BPE)")
    return df


# =============================================================================
# ÉTAPE 5-6 : Fusion finale + nom_commune + sauvegarde
# =============================================================================
def build_final(ml_path):
    print("\n" + "=" * 60)
    print("ÉTAPE 5-6 : Variables démo + nom_commune + sauvegarde")
    print("=" * 60)

    df_final = pd.read_csv(ml_path, dtype={'IRIS': str, 'COM': str}, low_memory=False)
    print(f"  Base ML : {len(df_final)} IRIS x {len(df_final.columns)} cols")

    # Calcul des variables démographiques dérivées (colonnes P21_/C21_ déjà présentes)
    df_final = compute_demographics(df_final)

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
    output_path = os.path.join(PATH_IRIS, "iris_final_socio_politique_bis.csv")
    df_final.to_csv(output_path, index=False)
    print(f"  Résultat : {len(df_final)} IRIS x {len(df_final.columns)} colonnes")
    print(f"  Sauvegardé : {output_path}")

    # Vérifications
    print("\n  === VÉRIFICATIONS ===")
    print(f"  IRIS total : {len(df_final)} (ancien jolivet : 45 650 — gain : +{len(df_final)-45650})")

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
    print("=" * 60)
    print("  build_iris_final.py — Reconstruction complète")
    print("=" * 60)

    hybride_path = build_hybride()
    ml_path = build_ml(hybride_path)
    build_final(ml_path)

    print("\nTerminé.")
