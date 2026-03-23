# -*- coding: utf-8 -*-
"""
Telecharge les Contours IRIS IGN 2025 (metropole + DOM),
les extrait et les fusionne en un seul GeoPackage WGS84.
Resultat : iris/contours_iris_2025.gpkg
"""
import os, subprocess, glob
import geopandas as gpd
import pandas as pd

SEVEN_ZIP = r"C:\Program Files\7-Zip\7z.exe"
OUT_DIR   = r"C:\tmp\iris_contours"   # chemin court pour eviter MAX_PATH Windows
OUT_FILE  = "iris/contours_iris_2025.gpkg"

SOURCES = [
    # (url, crs_hint)
    ("https://data.geopf.fr/telechargement/download/CONTOURS-IRIS/CONTOURS-IRIS_3-0__GPKG_LAMB93_FXX_2025-01-01/CONTOURS-IRIS_3-0__GPKG_LAMB93_FXX_2025-01-01.7z",       "EPSG:2154"),
    ("https://data.geopf.fr/telechargement/download/CONTOURS-IRIS/CONTOURS-IRIS_3-0__GPKG_RGAF09UTM20_GLP_2025-01-01/CONTOURS-IRIS_3-0__GPKG_RGAF09UTM20_GLP_2025-01-01.7z", "EPSG:5490"),
    ("https://data.geopf.fr/telechargement/download/CONTOURS-IRIS/CONTOURS-IRIS_3-0__GPKG_RGAF09UTM20_MTQ_2025-01-01/CONTOURS-IRIS_3-0__GPKG_RGAF09UTM20_MTQ_2025-01-01.7z", "EPSG:5490"),
    ("https://data.geopf.fr/telechargement/download/CONTOURS-IRIS/CONTOURS-IRIS_3-0__GPKG_UTM22RGFG95_GUF_2025-01-01/CONTOURS-IRIS_3-0__GPKG_UTM22RGFG95_GUF_2025-01-01.7z", "EPSG:2972"),
    ("https://data.geopf.fr/telechargement/download/CONTOURS-IRIS/CONTOURS-IRIS_3-0__GPKG_RGR92UTM40S_REU_2025-01-01/CONTOURS-IRIS_3-0__GPKG_RGR92UTM40S_REU_2025-01-01.7z", "EPSG:2975"),
    ("https://data.geopf.fr/telechargement/download/CONTOURS-IRIS/CONTOURS-IRIS_3-0__GPKG_RGM04UTM38S_MYT_2025-01-01/CONTOURS-IRIS_3-0__GPKG_RGM04UTM38S_MYT_2025-01-01.7z", "EPSG:4471"),
]

os.makedirs(OUT_DIR, exist_ok=True)

def download(url, dest):
    fname = os.path.join(dest, url.split("/")[-1])
    if os.path.exists(fname):
        print(f"  deja telecharge : {os.path.basename(fname)}")
        return fname
    print(f"  telechargement : {url.split('/')[-1]} ...")
    subprocess.run(
        ["curl", "-L", "-A", "Mozilla/5.0", "-o", fname, url],
        check=True
    )
    print(f"  ok : {fname}")
    return fname

def extract(archive, dest):
    print(f"  extraction : {os.path.basename(archive)} ...")
    subprocess.run([SEVEN_ZIP, "x", archive, f"-o{dest}", "-y"], check=True, capture_output=True)

def find_gpkg(directory):
    return glob.glob(os.path.join(directory, "**", "*.gpkg"), recursive=True)

# Download + extract + read
gdfs = []
for url, crs_hint in SOURCES:
    archive = download(url, OUT_DIR)
    basename = os.path.basename(archive).replace(".7z", "")
    subdir  = os.path.join(OUT_DIR, basename)
    if not os.path.exists(subdir):
        extract(archive, subdir)
    # Search recursively from subdir regardless of nesting
    gpkgs = find_gpkg(subdir)
    if not gpkgs:
        print(f"  WARNING: aucun .gpkg dans {subdir}")
        continue
    print(f"  lecture : {gpkgs[0]}")
    gdf = gpd.read_file(gpkgs[0])
    print(f"    {len(gdf)} IRIS, colonnes : {list(gdf.columns)}")

    # Find CODE_IRIS column (might be code_iris, CODE_IRIS, etc.)
    geom_col = gdf.geometry.name
    non_geom = [c for c in gdf.columns if c != geom_col]
    iris_col = next((c for c in non_geom if 'iris' in c.lower() and 'code' in c.lower()), None)
    if iris_col is None:
        iris_col = next((c for c in non_geom if 'iris' in c.lower()), None)
    if iris_col is None:
        print(f"  WARNING: colonne CODE_IRIS introuvable dans {list(gdf.columns)}")
        continue

    gdf = gdf.rename(columns={iris_col: 'CODE_IRIS'})
    gdf['CODE_IRIS'] = gdf['CODE_IRIS'].astype(str).str.zfill(9)
    gdf = gdf[['CODE_IRIS', geom_col]].copy()
    if geom_col != 'geometry':
        gdf = gdf.rename_geometry('geometry')

    if gdf.crs is None:
        gdf = gdf.set_crs(crs_hint)
    gdf = gdf.to_crs("EPSG:4326")
    gdfs.append(gdf)
    print(f"    {len(gdf)} IRIS en WGS84")

# Merge
print(f"\nFusion de {len(gdfs)} territoires...")
merged = pd.concat(gdfs, ignore_index=True)
merged = gpd.GeoDataFrame(merged, crs="EPSG:4326")
merged = merged.drop_duplicates(subset='CODE_IRIS')
print(f"Total IRIS : {len(merged)}")

print(f"Ecriture -> {OUT_FILE}")
merged.to_file(OUT_FILE, driver="GPKG")
print("Fait !")
