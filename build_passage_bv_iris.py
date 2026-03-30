# -*- coding: utf-8 -*-
"""
Reconstruit la table de passage BV -> IRIS.
- Spatial join "within" pour les BV dont le point representatif tombe dans un IRIS
- Fallback "nearest" pour les BV qui ne matchent pas (point hors IRIS)
Resultat : table_passage_BV_IRIS.csv (colonnes: ID_BUREAU_VOTE, CODE_IRIS)
"""
import os
import geopandas as gpd
import pandas as pd

F_BUREAUX = "iris/bureau-de-vote-insee-reu-openstreetmap.gpkg"          # https://www.data.gouv.fr/datasets/reconstruction-automatique-de-la-geometrie-des-bureaux-de-vote-depuis-insee-reu-et-openstreetmap
F_IRIS    = "iris/contours_iris_2025.gpkg"
OUT_FILE  = "table_passage_BV_IRIS.csv"

# --- 1. Chargement ---
print("Chargement IRIS contours...")
gdf_iris = gpd.read_file(F_IRIS)
gdf_iris['CODE_IRIS'] = gdf_iris['CODE_IRIS'].astype(str).str.zfill(9)
gdf_iris = gdf_iris[['CODE_IRIS', 'geometry']].copy()
print(f"  {len(gdf_iris)} IRIS, CRS: {gdf_iris.crs}")

print("Chargement bureaux de vote...")
gdf_bv = gpd.read_file(F_BUREAUX)
print(f"  {len(gdf_bv)} BV, CRS: {gdf_bv.crs}, colonnes: {list(gdf_bv.columns)}")

# Aligner CRS
if gdf_bv.crs != gdf_iris.crs:
    gdf_bv = gdf_bv.to_crs(gdf_iris.crs)

# --- 2. Construction ID_BUREAU_VOTE ---
print("Construction des ID_BUREAU_VOTE...")

def format_bureau_id(row):
    insee = str(row['insee']).zfill(5)
    bureau = str(row['bureau'])
    if bureau in ('None', 'nan', 'NaN', ''):
        bureau = "0001"
    elif "_" in bureau:
        parts = bureau.split("_")
        code_postal = parts[0]
        bureau_num = parts[1] if len(parts) > 1 else "0001"
        if insee == "75056":  # Paris
            arrond = code_postal[3:6]
            bureau = arrond + bureau_num.zfill(2)
        else:
            bureau = bureau_num.zfill(4)
    else:
        bureau = bureau.zfill(4)
    return f"{insee}_{bureau}"

gdf_bv['ID_BUREAU_VOTE'] = gdf_bv.apply(format_bureau_id, axis=1)
gdf_bv = gdf_bv[['ID_BUREAU_VOTE', 'geometry']].copy()

# --- 3. Point representatif ---
print("Calcul des points representatifs...")
gdf_pts = gdf_bv.copy()
gdf_pts['geometry'] = gdf_pts.geometry.representative_point()

# --- 4. Spatial join "within" ---
print("Spatial join 'within'...")
joined = gpd.sjoin(gdf_pts, gdf_iris, how="left", predicate="within")
matched   = joined[joined['CODE_IRIS'].notna()][['ID_BUREAU_VOTE', 'CODE_IRIS']].copy()
unmatched = joined[joined['CODE_IRIS'].isna()][['ID_BUREAU_VOTE', 'geometry']].copy()
print(f"  Matches: {len(matched)}, Sans match: {len(unmatched)}")

# --- 5. Fallback "nearest" pour les BV non matches ---
if len(unmatched) > 0:
    print(f"Fallback 'nearest' pour {len(unmatched)} BV...")
    # sjoin_nearest cherche l'IRIS le plus proche du point
    gdf_iris_centroids = gdf_iris.copy()
    fallback = gpd.sjoin_nearest(
        unmatched[['ID_BUREAU_VOTE', 'geometry']],
        gdf_iris[['CODE_IRIS', 'geometry']],
        how="left"
    )[['ID_BUREAU_VOTE', 'CODE_IRIS']].copy()
    fallback = fallback[fallback['CODE_IRIS'].notna()]
    print(f"  Recuperes par nearest: {len(fallback)}")
    result = pd.concat([matched, fallback], ignore_index=True)
else:
    result = matched

# --- 6. Deduplication BV ---
result = result.drop_duplicates(subset='ID_BUREAU_VOTE')
result = result[result['CODE_IRIS'].notna()].reset_index(drop=True)
result['CODE_IRIS'] = result['CODE_IRIS'].astype(str).str.zfill(9)

# --- 7. Passe inverse : IRIS sans BV -> BV le plus proche ---
# Pour chaque IRIS du contour qui n'a encore aucun BV, on lui assigne le BV le plus proche.
# Le meme BV peut etre "donne" a plusieurs IRIS.
iris_couverts = set(result['CODE_IRIS'])
iris_sans_bv = gdf_iris[~gdf_iris['CODE_IRIS'].isin(iris_couverts)][['CODE_IRIS', 'geometry']].copy()
print(f"\nIRIS sans BV apres sjoin: {len(iris_sans_bv)}")

if len(iris_sans_bv) > 0:
    # Centroide de chaque IRIS non couvert
    iris_sans_bv = iris_sans_bv.copy()
    iris_sans_bv['geometry'] = iris_sans_bv.geometry.centroid
    # BV avec leur point representatif (deja calcule)
    bv_pts = gdf_pts[['ID_BUREAU_VOTE', 'geometry']].copy()
    bv_pts = bv_pts.to_crs(iris_sans_bv.crs) if bv_pts.crs != iris_sans_bv.crs else bv_pts
    # Nearest BV pour chaque IRIS sans couverture
    iris_fallback = gpd.sjoin_nearest(
        iris_sans_bv[['CODE_IRIS', 'geometry']],
        bv_pts[['ID_BUREAU_VOTE', 'geometry']],
        how='left'
    )[['CODE_IRIS', 'ID_BUREAU_VOTE']].copy()
    iris_fallback = iris_fallback.dropna(subset=['ID_BUREAU_VOTE'])
    print(f"  IRIS recuperes par nearest BV: {len(iris_fallback)}")
    result = pd.concat([result, iris_fallback], ignore_index=True)

print(f"\nTotal lignes BV->IRIS: {len(result)}")
result.to_csv(OUT_FILE, index=False)
print(f"Sauvegarde -> {OUT_FILE}")

# --- 8. Verification ---
df_socio = pd.read_csv('iris/iris_final_socio_politique_bis.csv', low_memory=False, dtype={'IRIS': str})
iris_socio = set(df_socio['IRIS'].astype(str))
iris_passage = set(result['CODE_IRIS'])
print(f"\nVerification:")
print(f"  IRIS dans socio: {len(iris_socio)}")
print(f"  IRIS dans nouvelle table passage: {len(iris_passage)}")
print(f"  IRIS socio couverts: {len(iris_socio & iris_passage)}")
print(f"  IRIS socio non couverts: {len(iris_socio - iris_passage)}")
