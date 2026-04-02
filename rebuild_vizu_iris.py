# rebuild_vizu_iris.py — Générateur HTML desktop
# Tout le code partagé est dans shared_config.py
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import json as _json
import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

from shared_config import *  # noqa: F401,F403
from shared_config import _build_electoral_axis_vars  # private, not exported by *

# Partis, scores, axes, variables, élections — depuis shared_config

_SKIP_BUILD = '--skipbuild' in sys.argv

# ── Chargement des données ────────────────────────────────────────────────────
if not _SKIP_BUILD:
    print("Chargement des données IRIS...")
    df = load_iris_base()
    df = df.assign(**_build_electoral_axis_vars(df)).copy()
    print("Chargement données électorales...")
    iris_election_data = load_election_data(df)
    df = compute_parti_dominant(df, iris_election_data)
    marker_size = compute_marker_sizes(df)
    print("Calcul jitter...")
    _strict_cols = [c for c in df.columns if c.endswith('_strict')]
    var_data_x, var_data_y = compute_jitter_vars(df, extra_vars=_strict_cols)
    print("Calcul barycentres...")
    iris_election_precomputed = compute_barycentres(df, iris_election_data, var_data_x, var_data_y)
else:
    print("Mode --skipbuild : données JSON existantes conservées.")

# ── 9. HELPER JS DATA ─────────────────────────────────────────────────────────
def _round0(arr):
    return [int(round(float(v))) for v in arr]

def _round1(arr):
    return [round(float(v), 1) for v in arr]

def _round2(arr):
    return [round(float(v), 2) for v in arr]

def _round3(arr):
    import math as _math
    return [None if v is None or (isinstance(v, float) and not _math.isfinite(v)) else round(float(v), 3) for v in arr]

# Composite score vars get 3 decimals; age_moyen gets 1 decimal; pct vars get 0 (integer %)
_COMPOSITE_VARS = set(VARS_BY_CAT.get('Scores composites', []))
_ONE_DECIMAL_VARS = {'age_moyen', 'pct_etrangers', 'pct_immigres'}
_COMPOSITE_VARS = _COMPOSITE_VARS | {'tsne_x', 'tsne_y', 'umap_x', 'umap_y'}



# ── 10. TRACES PLOTLY ─────────────────────────────────────────────────────────
# Colonnes pour IRIS_INFO (20 champs — scores partis lus depuis IRIS_ELECTION_DATA)
# [0] LAB_IRIS, [1] nom_commune, [2] pop_totale, [3] DISP_MED21,
# [4] pct_csp_plus, [5] pct_csp_ouvrier, [6] pct_csp_intermediaire,
# [7] DISP_PPAT21, [8] inscrits, [9] votants, [10] pct_abstention,
# [11] score_blanc, [12] score_nul,
# [13] pct_proprietaires, [14] pct_hlm,
# [15] pct_chomage, [16] pct_bac_plus, [17] pct_sans_diplome, [18] age_moyen

_CD_PARTY_SCORES = [f'score_{g}' for g in ALL_ORDER]



def _build_trace_data_single():
    """Desktop: 3 traces — trace 0 = densité pop. (empty), trace 1 = barycentres (empty), trace 2 = all IRIS (empty)."""
    traces = []
    # Trace 0: densité population (remplie par restyleDensity en JS)
    traces.append(go.Histogram2dContour(
        x=[], y=[], z=[],
        histfunc='sum',
        # colorscale=[[0, 'rgba(200,200,200,0)'], [1, 'rgba(80,80,80,0.4)']],
        # colorscale=[[0, 'rgba(220,220,220,0)'], [1, 'rgba(110,110,110,0.3)']],
        colorscale = [[0, 'rgba(230,230,230,0)'], [0.1, 'rgba(130,130,130,0.2)'], [1, 'rgba(20,20,20,0.8)']],
        showscale=False,

        ncontours=25, 

        contours=dict(
            coloring='fill', 
            showlines=False, 
        ),

        nbinsx=40, nbinsy=40, 
        hoverinfo='none',
        showlegend=True,
        name='densité pop.',
        opacity=0.25,
    ))    
    
    # Trace 1: barycentres (filled by applyElection)
    traces.append(go.Scattergl(
        x=[], y=[], mode="markers+text",
        marker=dict(symbol="cross-thin", size=[], color=[], line=dict(width=3, color=[])),
        text=[], textposition="top right",
        textfont=dict(size=9, color=[], family="Helvetica Neue, sans-serif"),
        hovertemplate="<b>Barycentre %{text}</b><br>X : <b>%{x:.3f}</b><br>Y : <b>%{y:.3f}</b><extra></extra>",
        showlegend=False, opacity=0.9, name="barycentres"
    ))
    # Trace 2: all IRIS points (filled by applyElection)
    traces.append(go.Scattergl(
        x=[], y=[], mode="markers", name="iris",
        marker=dict(color=[], size=[], opacity=0.75,
                    line=dict(width=0.4, color="rgba(255,255,255,0.4)")),
        customdata=[], hoverinfo="none", showlegend=False,
    ))
    return traces




# ── 12. BUILD DESKTOP HTML ─────────────────────────────────────────────────────
def build_desktop_html(skip_build=False):
    traces = _build_trace_data_single()

    fig = go.Figure()
    fig.add_vline(x=0, line_dash="dot", line_color="#BBBBBB", line_width=1.5)
    fig.add_hline(y=0, line_dash="dot", line_color="#BBBBBB", line_width=1.5)
    for tr in traces:
        fig.add_trace(tr)
    fig.update_layout(
        paper_bgcolor="#FAF9F7", plot_bgcolor="#FEFDFB",
        margin=dict(t=20, b=60, l=70, r=30),
        dragmode="pan",
        xaxis=dict(
            title=dict(text=AXIS_PRESETS[0]['xTitle'],
                       font=dict(size=11, color="#555", family="Helvetica Neue, sans-serif"), standoff=10),
            range=AXIS_PRESETS[0]['xRange'],
            showgrid=True, gridcolor="#EBEBEB", gridwidth=0.8, zeroline=False,
            tickfont=dict(size=9.5, color="#AAA"), linecolor="#DDD", linewidth=1),
        yaxis=dict(
            title=dict(text=AXIS_PRESETS[0]['yTitle'],
                       font=dict(size=11, color="#555", family="Helvetica Neue, sans-serif"), standoff=8),
            range=AXIS_PRESETS[0]['yRange'],
            showgrid=True, gridcolor="#EBEBEB", gridwidth=0.8, zeroline=False,
            tickfont=dict(size=9.5, color="#AAA"), linecolor="#DDD", linewidth=1),
        showlegend=False,
        hoverlabel=dict(bgcolor="rgba(255,255,255,0.97)", bordercolor="#E0E0E0",
                        font=dict(size=12, family="Helvetica Neue, sans-serif", color="#1a1a1a")),
    )
    fig_json = fig.to_json()

    # ── Sérialisation données ───────────────────────────────────────────────
    import math, os as _os

    if not skip_build:
        def _is_nan(v):
            return v is None or (isinstance(v, float) and math.isnan(v))

        # Données socio (IRIS_X / IRIS_Y) → fichiers JSON externes
        # Les colonnes _strict (scores composites stricts) utilisent _round3 ; NaN → null
        _strict_col_set = set(_strict_cols)
        iris_x_js = {}
        iris_y_js = {}
        for v in var_data_x:
            fn = _round3 if (v in _COMPOSITE_VARS or v in _strict_col_set) else _round2
            iris_x_js[v] = fn(var_data_x[v])
            iris_y_js[v] = fn(var_data_y[v])

        # Données électorales → fichiers JSON externes (un par élection)
        iris_elec = {}
        for eid_s, data_list in iris_election_data.items():
            iris_elec[eid_s] = {
                'colors': [d['color'] for d in data_list],
                'partis': [d['parti'] for d in data_list],
                'scores': [d['scores'] for d in data_list],
                'abst': [round(d['abst'], 1) if not _is_nan(d['abst']) else None for d in data_list],
                'inscrits': [int(d['inscrits']) if not _is_nan(d['inscrits']) else None for d in data_list],
                'exprimes': [int(d['exprimes']) if not _is_nan(d['exprimes']) else None for d in data_list],
                'blancs': [int(d['blancs']) if not _is_nan(d['blancs']) else None for d in data_list],
                'nuls': [int(d['nuls']) if not _is_nan(d['nuls']) else None for d in data_list],
                **iris_election_precomputed.get(eid_s, {}),
            }

        # IRIS info, populations, marker sizes
        all_customdata = make_customdata(df)
        iris_pops = [int(round(float(v))) for v in df['_pop'].fillna(1).tolist()]
        marker_sizes_list = [round(float(marker_size.loc[i]), 1) for i in df.index]
        group_indices = {g: df.index[df['parti_dominant'] == g].tolist() for g in ALL_ORDER}

        # ── Écriture des fichiers JSON dans data/ ──────────────────────────────
        def _build_geo_centroids(df_arg):
            import geopandas as gpd
            # Source principale : contours_iris_2025.gpkg (EPSG:4326, contient DROM)
            print("  Calcul des centroïdes IRIS depuis contours_iris_2025.gpkg...")
            gdf = gpd.read_file('iris/contours_iris_2025.gpkg')
            centroids = gdf.to_crs('EPSG:2154').geometry.centroid.to_crs('EPSG:4326')
            gdf['lat'] = centroids.y
            gdf['lon'] = centroids.x
            lookup = dict(zip(gdf['CODE_IRIS'].astype(str), zip(gdf['lat'], gdf['lon'])))
            # Fallback : iris-stats.geojson pour les IRIS manquants
            missing = [str(c) for c in df_arg['IRIS'] if str(c) not in lookup]
            if missing:
                gdf2 = gpd.read_file('iris-stats.geojson')
                gdf2 = gdf2.set_crs('EPSG:2154', allow_override=True)
                centroids2 = gdf2.geometry.centroid.to_crs('EPSG:4326')
                gdf2['lat'] = centroids2.y
                gdf2['lon'] = centroids2.x
                for _, row in gdf2[gdf2['index'].astype(str).isin(set(missing))].iterrows():
                    lookup[str(row['index'])] = (row['lat'], row['lon'])
            lats, lons = [], []
            for iris_code in df_arg['IRIS']:
                coords = lookup.get(str(iris_code))
                if coords:
                    lats.append(round(float(coords[0]), 5))
                    lons.append(round(float(coords[1]), 5))
                else:
                    lats.append(None)
                    lons.append(None)
            return lats, lons

        _os.makedirs('data', exist_ok=True)

        # static.json : IRIS_INFO + IRIS_POPS + MARKER_SIZES + GROUP_INDICES
        _static = {
            'IRIS_INFO': all_customdata,
            'IRIS_POPS': iris_pops,
            'MARKER_SIZES': marker_sizes_list,
            'GROUP_INDICES': group_indices,
        }
        _path = 'data/static.json'
        with open(_path, 'w', encoding='utf-8') as _f:
            _json.dump(_static, _f, ensure_ascii=False, separators=(',', ':'))
        print(f"  data/static.json : {_os.path.getsize(_path)//1024} KB")

        _geo_path = 'data/geo.json'
        if not _os.path.exists(_geo_path):
            _lats, _lons = _build_geo_centroids(df)
            with open(_geo_path, 'w', encoding='utf-8') as _f:
                _json.dump({'lat': _lats, 'lon': _lons}, _f, separators=(',', ':'))
            print(f"  data/geo.json : {_os.path.getsize(_geo_path)//1024} KB")
        else:
            print(f"  data/geo.json : déjà présent ({_os.path.getsize(_geo_path)//1024} KB)")

        # Un fichier par élection
        for eid_s, elec_obj in iris_elec.items():
            _path = f'data/elec_{eid_s}.json'
            with open(_path, 'w', encoding='utf-8') as _f:
                _json.dump(elec_obj, _f, ensure_ascii=False, separators=(',', ':'))
            print(f"  data/elec_{eid_s}.json : {_os.path.getsize(_path)//1024} KB")

        # iris_x_desktop.json et iris_y_desktop.json
        _path = 'data/iris_x_desktop.json'
        with open(_path, 'w', encoding='utf-8') as _f:
            _json.dump(iris_x_js, _f, separators=(',', ':'))
        print(f"  data/iris_x_desktop.json : {_os.path.getsize(_path)//1024} KB")
        _path = 'data/iris_y_desktop.json'
        with open(_path, 'w', encoding='utf-8') as _f:
            _json.dump(iris_y_js, _f, separators=(',', ':'))
        print(f"  data/iris_y_desktop.json : {_os.path.getsize(_path)//1024} KB")

        # ml_flags.json : tableaux booléens (0/1) par variable DISP_* imputée
        ml_flags = {}
        for col in df.columns:
            if col.startswith('ml_imputed_'):
                varname = col[len('ml_imputed_'):]
                ml_flags[varname] = df[col].astype(int).tolist()
        _path = 'data/ml_flags.json'
        with open(_path, 'w', encoding='utf-8') as _f:
            _json.dump(ml_flags, _f, separators=(',', ':'))
        print(f"  data/ml_flags.json : {_os.path.getsize(_path)//1024} KB ({len(ml_flags)} variables flaggées)")

    # ── Métadonnées inline (petites, <500 KB) ─────────────────────────────
    elections_meta_str = _json.dumps(ELECTIONS_AVAILABLE, ensure_ascii=False, separators=(',', ':'))
    default_elec_str = _json.dumps(DEFAULT_ELECTION, separators=(',', ':'))
    presets_str = _json.dumps(AXIS_PRESETS, ensure_ascii=False)
    vars_str = _json.dumps(VARS_BY_CAT, ensure_ascii=False)
    var_labels_str = _json.dumps(VAR_LABELS, ensure_ascii=False)
    couleurs_str = _json.dumps(COULEURS, ensure_ascii=False, separators=(',', ':'))
    vote_parties_str = _json.dumps(VOTE_PARTIES_JS, separators=(',', ':'))
    all_parties_colors_str = _json.dumps(ALL_PARTIES_COLORS, ensure_ascii=False, separators=(',', ':'))

    if skip_build:
        import os as _os
        with open('data/static.json', encoding='utf-8') as _f:
            _static_cached = _json.load(_f)
        _group_indices_cached = _static_cached['GROUP_INDICES']
        buttons_data = [{'key': g, 'short': SHORT.get(g,g), 'label': LABELS.get(g,g),
                         'color': ALL_PARTIES_COLORS.get(g,'#9CA3AF'),
                         'count': len(_group_indices_cached.get(g, []))} for g in ALL_ORDER]
        n_iris = len(_static_cached['IRIS_POPS'])
    else:
        buttons_data = [{'key': g, 'short': SHORT.get(g,g), 'label': LABELS.get(g,g),
                         'color': ALL_PARTIES_COLORS.get(g,'#9CA3AF'),
                         'count': int((df['parti_dominant'] == g).sum())} for g in ALL_ORDER]
        n_iris = len(df)
    btns_str = _json.dumps(buttons_data, ensure_ascii=False, separators=(',', ':'))
    order_str = _json.dumps(ALL_ORDER, ensure_ascii=False, separators=(',', ':'))

    sg = AXIS_PRESETS[0]

    html = f"""<!DOCTYPE html>
<html lang="fr">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Sociologie des IRIS</title>
<style>
* {{ margin: 0; padding: 0; box-sizing: border-box; }}
html, body {{ background: #FAF9F7; font-family: 'Helvetica Neue', system-ui, sans-serif; color: #1a1a1a; }}
.page-header {{ text-align: center; padding: 20px 20px 8px; }}
.page-header h1 {{ font-size: 24px; font-weight: 900; letter-spacing: -0.5px; margin-bottom: 4px; }}
.page-header p {{ font-size: 12px; color: #888; }}

.axis-bar {{ display: flex; align-items: center; gap: 10px; padding: 10px 20px;
             background: #fff; border-bottom: 1px solid #E8E8E8; flex-wrap: wrap; }}
.axis-bar-label {{ font-size: 12px; font-weight: 700; color: #666; white-space: nowrap; }}
.preset-btns {{ display: flex; gap: 6px; flex-wrap: wrap; }}
.preset-btn {{ padding: 5px 13px; border-radius: 20px; border: 1.5px solid #D0D0D0;
               background: transparent; font-size: 12px; font-weight: 600; color: #555;
               font-family: inherit; cursor: pointer; transition: all 0.15s; white-space: nowrap; }}
.preset-btn:hover {{ border-color: #888; color: #222; }}
.preset-btn.active {{ background: #1a1a1a; border-color: #1a1a1a; color: #fff; }}
.custom-toggle {{ padding: 5px 13px; border-radius: 20px; border: 1.5px dashed #C0C0C0;
                  background: transparent; font-size: 12px; font-weight: 600; color: #888;
                  font-family: inherit; cursor: pointer; transition: all 0.15s; white-space: nowrap; margin-left: 4px; }}
.custom-toggle:hover {{ border-color: #888; color: #444; }}
.custom-toggle.open {{ border-color: #1a1a1a; color: #1a1a1a; background: #f5f5f5; }}
.ml-toggle-btn {{ padding: 5px 13px; border-radius: 20px; border: 1.5px solid #D97706;
                  background: transparent; font-size: 12px; font-weight: 600; color: #D97706;
                  font-family: inherit; cursor: pointer; transition: all 0.15s; white-space: nowrap; margin-left: 4px; }}
.ml-toggle-btn:hover {{ background: #FEF3C7; }}
.ml-toggle-btn.full-mode {{ background: #FEF3C7; border-color: #D97706; color: #92400E; }}
.ml-flag {{ font-size: 10px; color: #D97706; font-style: italic; font-weight: 400; margin-left: 4px; }}
.custom-panel {{ display: none; padding: 10px 20px; background: #f9f9f9;
                 border-bottom: 1px solid #E8E8E8; gap: 16px; align-items: center; flex-wrap: wrap; }}
.custom-panel.open {{ display: flex; }}
.custom-panel label {{ font-size: 12px; font-weight: 600; color: #555; }}
.custom-panel select {{ padding: 5px 8px; border-radius: 8px; border: 1px solid #D0D0D0;
                        background: #fff; font-size: 12px; font-family: inherit; color: #222;
                        cursor: pointer; min-width: 220px; }}

.main-layout {{ display: flex; gap: 0; }}
.chart-col {{ flex: 1; min-width: 0; position: relative; }}
.sidebar {{ width: 300px; flex-shrink: 0; padding: 16px; border-left: 1px solid #EBEBEB;
            overflow-y: auto; }}

.group-filters {{ padding: 10px 20px 6px; display: flex; flex-wrap: wrap; gap: 5px; align-items: center; }}
.toggle-all-btn {{ padding: 3px 10px; border-radius: 12px; border: 1px solid #CCC;
                   background: transparent; font-size: 10px; font-weight: 700; color: #888;
                   font-family: inherit; cursor: pointer; }}
.grp-btn {{ padding: 3px 10px; border-radius: 12px; border: 2px solid; font-size: 10px;
            font-weight: 700; font-family: inherit; cursor: pointer; transition: all 0.12s; }}
.grp-btn.on {{ color: #fff; }}
.grp-btn.off {{ background: transparent !important; opacity: 0.3; }}

.chart-wrapper {{ position: relative; }}
#chartDiv {{ width: 100%; height: 640px; }}
#domMapsRow {{ display: none; gap: 6px; padding: 6px 0 0; flex-wrap: wrap; }}
.dom-map-wrap {{ flex: 1; min-width: 140px; max-width: 200px; position: relative; }}
.dom-map-label {{ position: absolute; top: 4px; left: 6px; z-index: 10; font-size: 10px;
                  font-weight: 800; color: #333; background: rgba(255,255,255,0.82);
                  padding: 1px 6px; border-radius: 8px; pointer-events: none; }}
.dom-map-canvas {{ width: 100%; height: 185px; border-radius: 8px; overflow: hidden;
                   border: 1px solid #E8E8E8; }}
.corner-label {{ position: absolute; font-size: 10px; font-weight: 800; line-height: 1.2;
                 pointer-events: none; opacity: 0.65; }}
.corner-tl {{ top: 12px; left: 80px; text-align: left; }}
.corner-tr {{ top: 12px; right: 40px; text-align: right; }}
.corner-bl {{ bottom: 70px; left: 80px; text-align: left; }}
.corner-br {{ bottom: 70px; right: 40px; text-align: right; }}

.info-card-desktop {{ background: #fff; border: 1px solid #E8E8E8; border-radius: 12px;
                      padding: 16px; font-size: 13px; line-height: 1.6; }}
.info-card-desktop .name {{ font-size: 15px; font-weight: 900; margin-bottom: 2px; }}
.info-card-desktop .party {{ font-weight: 700; font-size: 11px; margin-bottom: 10px; }}
.info-card-desktop .row {{ color: #555; margin-bottom: 2px; }}
.info-card-desktop .row b {{ color: #1a1a1a; }}
.info-card-desktop .lbl {{ color: #999; }}
.info-card-desktop.empty {{ color: #AAA; font-size: 12px; text-align: center; padding: 40px 16px; }}
.info-card-desktop .dynamic-row {{ color: #555; margin-bottom: 2px; border-top: 1px solid #F0F0F0; padding-top: 6px; margin-top: 4px; }}
.info-card-desktop .vote-bars {{ margin-top: 8px; }}
.info-card-desktop .vote-bar-row {{ display: flex; align-items: center; gap: 6px; margin-bottom: 3px; font-size: 11px; }}
.info-card-desktop .vote-bar-label {{ width: 36px; color: #888; font-size: 10px; font-weight: 700; text-align: right; }}
.info-card-desktop .vote-bar-bg {{ flex: 1; height: 5px; background: #F0F0F0; border-radius: 3px; overflow: hidden; }}
.info-card-desktop .vote-bar-fill {{ height: 100%; border-radius: 3px; }}
.info-card-desktop .vote-bar-pct {{ width: 30px; text-align: right; color: #555; font-size: 10px; }}
.info-card-desktop .stat-bar-row {{ display: flex; align-items: center; gap: 6px; margin-bottom: 3px; font-size: 11px; }}
.info-card-desktop .stat-bar-label {{ width: 80px; color: #888; font-size: 10px; text-align: right; flex-shrink: 0; }}
.info-card-desktop .stat-bar-bg {{ flex: 1; height: 5px; background: #F0F0F0; border-radius: 3px; overflow: hidden; }}
.info-card-desktop .stat-bar-fill {{ height: 100%; border-radius: 3px; }}
.info-card-desktop .stat-bar-pct {{ width: 32px; text-align: right; color: #555; font-size: 10px; }}
.info-card-desktop .section-title {{ font-size: 10px; font-weight: 800; text-transform: uppercase;
                                      letter-spacing: 0.5px; color: #BBB; margin: 8px 0 4px; }}


.footer {{ text-align: center; padding: 10px 20px 6px; font-size: 9px; color: #AAA; }}

.axis-desc {{ padding: 14px 20px 20px; font-size: 12px; color: #555; line-height: 1.65;
              border-top: 1px solid #EBEBEB; background: #FDFCFA; }}
.axis-desc:empty {{ display: none; }}
.axis-desc .desc-title {{ font-size: 14px; font-weight: 900; color: #1a1a1a; margin-bottom: 10px; }}
.axis-desc .desc-axes {{ display: grid; grid-template-columns: 1fr 1fr; gap: 14px; margin-bottom: 12px; }}
.axis-desc .desc-ax {{ background: #fff; border: 1px solid #E8E8E8; border-radius: 8px; padding: 10px 12px; }}
.axis-desc .desc-quadrants {{ background: #fff; border: 1px solid #E8E8E8; border-radius: 8px; padding: 10px 12px; }}
.axis-desc .desc-q-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 8px; margin-top: 6px; }}
.axis-desc .desc-q {{ font-size: 11px; padding: 6px 8px; border-radius: 6px; background: #F8F8F8; }}
.axis-desc .desc-q-label {{ font-size: 9px; font-weight: 800; text-transform: uppercase;
                             letter-spacing: 0.5px; opacity: 0.6; margin-bottom: 2px; }}

.election-bar {{ display: flex; align-items: center; gap: 10px; padding: 8px 20px;
                 background: #F8F7F5; border-bottom: 1px solid #E8E8E8; flex-wrap: wrap; }}
.election-bar-label {{ font-size: 11px; font-weight: 700; color: #888; white-space: nowrap; }}
.election-type-btns {{ display: flex; gap: 4px; flex-wrap: wrap; }}
.elec-type-btn {{ padding: 3px 10px; border-radius: 14px; border: 1.5px solid #D0D0D0;
                  background: transparent; font-size: 11px; font-weight: 600; color: #777;
                  font-family: inherit; cursor: pointer; transition: all 0.12s; white-space: nowrap; }}
.elec-type-btn:hover {{ border-color: #888; color: #333; }}
.elec-type-btn.active {{ background: #3B3B3B; border-color: #3B3B3B; color: #fff; }}
.elec-year-btns {{ display: flex; gap: 4px; flex-wrap: wrap; }}
.elec-year-btn {{ padding: 3px 10px; border-radius: 14px; border: 1.5px solid #D0D0D0;
                  background: transparent; font-size: 11px; font-weight: 600; color: #777;
                  font-family: inherit; cursor: pointer; transition: all 0.12s; white-space: nowrap; }}
.elec-year-btn:hover {{ border-color: #888; color: #333; }}
.elec-year-btn.active {{ background: #555; border-color: #555; color: #fff; }}
.tour-btns {{ display: flex; gap: 4px; margin-left: auto; }}
.tour-btn {{ padding: 3px 10px; border-radius: 14px; border: 1.5px solid #D0D0D0;
             background: transparent; font-size: 11px; font-weight: 700; color: #777;
             font-family: inherit; cursor: pointer; transition: all 0.12s; }}
.tour-btn:hover {{ border-color: #888; color: #333; }}
.tour-btn.active {{ background: #1a1a1a; border-color: #1a1a1a; color: #fff; }}
.tour-btn:disabled {{ opacity: 0.3; cursor: default; }}
.election-current-label {{ font-size: 10px; color: #AAA; font-style: italic; padding: 0 6px; }}
</style>
</head>
<body>

<div class="page-header">
  <h1>Sociologie des IRIS</h1>
  <p>Sociologie des {n_iris} zones IRIS · taille = population IRIS · données INSEE 2021</p>
</div>

<div class="axis-bar" id="axisBar">
  <span class="axis-bar-label">Axes :</span>
  <div class="preset-btns" id="presetBtns"></div>
  <button class="custom-toggle" id="customToggle">Personnaliser ▾</button>
  <button class="ml-toggle-btn" id="mlToggleBtn" title="Basculer entre mode strict (IRIS imputés ML masqués) et mode complet (tous les IRIS)">⚠ Imputés ML : masqués</button>
</div>

<div class="election-bar" id="electionBar">
  <span class="election-bar-label">Couleur :</span>
  <div class="election-type-btns" id="elecTypeBtns"></div>
  <div class="elec-year-btns" id="elecYearBtns"></div>
  <div class="tour-btns">
    <button class="tour-btn" id="tourBtn1" data-tour="1">Tour 1</button>
    <button class="tour-btn" id="tourBtn2" data-tour="2">Tour 2</button>
  </div>
  <span class="election-current-label" id="electionCurrentLabel"></span>
  <span id="elecSpinner" style="display:none;font-size:11px;color:#888;margin-left:6px">chargement…</span>
  <select id="colorVarSelect" disabled style="margin-left:10px;padding:3px 8px;border-radius:8px;border:1px solid #D0D0D0;font-size:11px;font-family:inherit;color:#555;background:#fff;cursor:pointer;min-width:180px"><option value="">Couleur : élection</option></select>
</div>

<div class="custom-panel" id="customPanel">
  <div>
    <label for="xSelect">Axe X : </label>
    <select id="xSelect"></select>
  </div>
  <div>
    <label for="ySelect">Axe Y : </label>
    <select id="ySelect"></select>
  </div>
  <label style="font-size:11px; display:flex; align-items:center; gap:4px;">
    <input type="checkbox" id="xInvertChk"> Inverser X
  </label>
</div>

<div class="main-layout">
  <div class="chart-col">
    <div style="padding:10px 20px 2px;display:flex;align-items:center;gap:8px">
      <button class="toggle-all-btn" id="toggleAllBtn">Tout décocher</button>
      <button class="toggle-all-btn" id="densityToggleBtn" style="border-color:#AAA;color:#888;">Densité : affichée</button>
    </div>
    <div class="group-filters" id="groupFilters"></div>
    <div class="chart-wrapper">
      <div id="chartDiv"></div>
      <div id="mapDiv" style="width:100%;height:640px;display:none;position:relative;">
        <div id="carteLoadingMsg" style="display:none;position:absolute;top:8px;left:50%;transform:translateX(-50%);background:rgba(255,255,255,0.92);padding:6px 16px;border-radius:20px;font-size:12px;color:#555;z-index:10;box-shadow:0 1px 4px rgba(0,0,0,0.1)">Chargement des coordonnées géographiques…</div>
        <button id="mapResetBtn" onclick="mapInstance && mapInstance.fitBounds([[-5.2,41.3],[9.6,51.2]],{{padding:20}})" style="position:absolute;bottom:16px;right:8px;z-index:20;background:rgba(255,255,255,0.95);border:1px solid #ccc;border-radius:6px;padding:6px 10px;font-size:13px;cursor:pointer;box-shadow:0 1px 4px rgba(0,0,0,0.15)">↺ Recentrer</button>
      </div>
      <div id="domMapsRow">
        <div class="dom-map-wrap"><div class="dom-map-label">Guadeloupe</div><div class="dom-map-canvas" id="domMap0"></div></div>
        <div class="dom-map-wrap"><div class="dom-map-label">Martinique</div><div class="dom-map-canvas" id="domMap1"></div></div>
        <div class="dom-map-wrap"><div class="dom-map-label">Guyane</div><div class="dom-map-canvas" id="domMap2"></div></div>
        <div class="dom-map-wrap"><div class="dom-map-label">La Réunion</div><div class="dom-map-canvas" id="domMap3"></div></div>
      </div>
      <div class="corner-label corner-tl" id="cornerTL" style="color:{sg['corners'][0]['color']}"></div>
      <div class="corner-label corner-tr" id="cornerTR" style="color:{sg['corners'][1]['color']}"></div>
      <div class="corner-label corner-bl" id="cornerBL" style="color:{sg['corners'][2]['color']}"></div>
      <div class="corner-label corner-br" id="cornerBR" style="color:{sg['corners'][3]['color']}"></div>
    </div>
    <div class="footer">⊕ = barycentre du groupe &nbsp;·&nbsp; taille = population IRIS &nbsp;·&nbsp; couleur = parti dominant &nbsp;·&nbsp; N={n_iris}</div>
    <div id="colorLegend" style="display:none;flex-direction:column;padding:6px 20px;background:#fff;border-top:1px solid #E8E8E8"></div>
    <div id="abst-stats-panel" style="padding:6px 20px 4px;font-size:12px;color:#555;border-top:1px solid #e5e7eb;margin-top:2px;display:none"></div>
    <div class="axis-desc" id="axisDesc"></div>
  </div>
  <div class="sidebar">
    <div id="sidebarSticky" style="display:none;position:sticky;top:0;background:#FAF9F7;border-bottom:1px solid #EEE;padding:5px 16px;font-size:11px;font-weight:700;z-index:10;border-radius:8px 8px 0 0"></div>
    <div class="info-card-desktop empty" id="infoCard">
      <p>Cliquez sur un point<br>pour voir les infos de l'IRIS</p>
    </div>
  </div>
</div>

<div id="loadingOverlay" style="position:fixed;inset:0;background:rgba(250,249,247,0.96);display:flex;flex-direction:column;align-items:center;justify-content:center;z-index:9999;font-family:'Helvetica Neue',system-ui,sans-serif">
  <div style="font-size:14px;color:#555;margin-bottom:12px" id="loadingMsg">Chargement des données…</div>
  <div style="width:280px;height:4px;background:#EEE;border-radius:2px">
    <div id="loadingBar" style="height:4px;background:#F97316;border-radius:2px;width:0%;transition:width 0.4s ease"></div>
  </div>
  <div style="font-size:11px;color:#AAA;margin-top:8px" id="loadingDetail"></div>
</div>

<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<link href='https://unpkg.com/maplibre-gl@4/dist/maplibre-gl.css' rel='stylesheet'/>
<script src='https://unpkg.com/maplibre-gl@4/dist/maplibre-gl.js'></script>
<script>
const figData = {fig_json};

// ── Métadonnées inline (petites, <500 KB) ────────────────────────────────
const ELECTIONS_META = {elections_meta_str};
const DEFAULT_ELECTION_ID = {default_elec_str};
const PRESETS = {presets_str};
const VARS = {vars_str};
const varLabels = {var_labels_str};
const COULEURS_JS = {couleurs_str};
const VOTE_PARTIES = {vote_parties_str};
const ALL_PARTIES_COLORS_JS = {all_parties_colors_str};
const btns = {btns_str};
const ORDER = {order_str};

// ── Données globales (chargées en async depuis data/) ─────────────────────
let IRIS_X = null, IRIS_Y = null;
let IRIS_LAT = null, IRIS_LON = null;
let mapInstance = null;
let mapReady = false;
let mapInitialized = false;
let isCarteActive = false;
const domMaps = [];       // instances MapLibre des mini-maps DOM-TOM
const domMapsReady = [];  // bool par mini-map
const DOM_TOM_CONFIGS = [
  {{ id: 'domMap0', center: [-61.55, 16.25], zoom: 8.5, bounds: [[-62.0,15.83],[-61.0,16.56]] }},
  {{ id: 'domMap1', center: [-61.0,  14.65], zoom: 9.0, bounds: [[-61.3,14.37],[-60.75,14.90]] }},
  {{ id: 'domMap2', center: [-53.1,   4.0],  zoom: 5.5, bounds: [[-54.6,2.1],[-51.5,5.8]] }},
  {{ id: 'domMap3', center: [55.55, -21.1],  zoom: 8.5, bounds: [[55.21,-21.4],[55.84,-20.87]] }},
];
let IRIS_INFO = null, IRIS_POPS = null, MARKER_SIZES = null, GROUP_INDICES = null;
let ML_FLAGS = null;      // chargé depuis data/ml_flags.json (tableaux 0/1 par var DISP_*)
let mlStrictMode = true;  // true = masquer les IRIS imputés ML (mode par défaut)
let densityVisible = true;
// Noms de variables résolus vers _strict si mlStrictMode est actif
let currentXVarData = 'score_exploitation';
let currentYVarData = 'score_domination';
const elecCache = {{}};  // cache élections déjà fetché

const chartDiv = document.getElementById('chartDiv');

let currentXVar = 'score_exploitation';
let currentYVar = 'score_domination';
let currentXInvert = false;
let currentPresetId = 'saint_graphique';
let currentXRange = (PRESETS[0].xRange || [-10, 10]).slice();
let currentYRange = (PRESETS[0].yRange || [-10, 10]).slice();
let currentCorners = PRESETS[0].corners;
let currentColorVar = null;   // null = couleur par élection, sinon nom de variable
let colorVarMin = 0, colorVarMax = 1;

// ── Corner labels ─────────────────────────────────────────────────────────
function setCorners(corners) {{
  const map = {{}};
  corners.forEach(c => map[c.pos] = c);
  const ids = {{tl:'cornerTL', tr:'cornerTR', bl:'cornerBL', br:'cornerBR'}};
  for (const [pos, elId] of Object.entries(ids)) {{
    const el = document.getElementById(elId);
    if (map[pos]) {{ el.innerHTML = map[pos].text; el.style.color = map[pos].color; }}
    else {{ el.innerHTML = ''; }}
  }}
}}
setCorners(currentCorners);

// ── Auto-range helper ─────────────────────────────────────────────────────
function computeDataRange_embeddings(varName, invert) {{
  if (!IRIS_X || !IRIS_Y) return [-1, 1];
  const arr = IRIS_X[varName] || IRIS_Y[varName];
  if (!arr) return [-1, 1];
  let mn = Infinity, mx = -Infinity;
  const scanArr = (a) => {{
    for (const v of a) {{
      if (v === null || v === undefined || isNaN(v)) continue;
      const val = invert ? -v : v;
      if (val < mn) mn = val;
      if (val > mx) mx = val;
    }}
  }};
  scanArr(arr);
  // Inclure la variante _strict pour que le range couvre les deux modes (full et strict ML)
  const arrStrict = (IRIS_X && IRIS_X[varName + '_strict']) || (IRIS_Y && IRIS_Y[varName + '_strict']);
  if (arrStrict) scanArr(arrStrict);
  if (mn === Infinity) return [-1, 1];
  const pad = (mx - mn) * 0.02;
  return [mn - pad, mx + pad];
}}

// ── Auto-range helper (Percentiles %) ──────────────────────────────
function computeDataRange(varName, invert) {{
  if (!IRIS_X || !IRIS_Y) return [-1, 1];
  
  const arr = IRIS_X[varName] || IRIS_Y[varName];
  if (!arr || arr.length === 0) return [-1, 1];

  // 1. Filtrage et inversion
  let cleanArr = [];
  for (let i = 0; i < arr.length; i++) {{
    const v = arr[i];
    if (v !== null && v !== undefined && !isNaN(v)) {{
      cleanArr.push(invert ? -v : v);
    }}
  }}

  if (cleanArr.length === 0) return [-1, 1];

  // 2. Tri (nécessaire pour les percentiles)
  cleanArr.sort((a, b) => a - b);

  // 3. Calcul des indices %
  const lowIdx = Math.floor(cleanArr.length * 0.0001);
  const highIdx = Math.max(0, Math.ceil(cleanArr.length * 0.9999) - 1);

  const p05 = cleanArr[lowIdx];
  const p95 = cleanArr[highIdx];

  // 4. Marge de sécurité réduite (2%)
  const diff = p95 - p05;
  const pad = diff === 0 ? 1 : diff * 0.02;

  return [p05 - pad, p95 + pad];
}}


// ── Colorscale variable ────────────────────────────────────────────────────
function lerp(a, b, t) {{ return a + (b - a) * t; }}
function varToHex(val, mn, mx) {{
  const t = Math.max(0, Math.min(1, (val - mn) / (mx - mn || 1)));
  let r, g, b;
  if (t < 0.5) {{
    const s = t * 2;
    r = Math.round(lerp(59, 255, s));
    g = Math.round(lerp(130, 255, s));
    b = Math.round(lerp(246, 255, s));
  }} else {{
    const s = (t - 0.5) * 2;
    r = Math.round(lerp(255, 239, s));
    g = Math.round(lerp(255, 68, s));
    b = Math.round(lerp(255, 68, s));
  }}
  return '#' + [r,g,b].map(x => x.toString(16).padStart(2,'0')).join('');
}}

function computeColorVarRange(varName) {{
  const arr = (IRIS_X && IRIS_X[varName]) ? IRIS_X[varName] : (IRIS_Y && IRIS_Y[varName] ? IRIS_Y[varName] : null);
  if (!arr) {{ colorVarMin = 0; colorVarMax = 1; return; }}
  const sorted = arr.filter(v => v !== null && !isNaN(v)).slice().sort((a,b) => a-b);
  if (!sorted.length) {{ colorVarMin = 0; colorVarMax = 1; return; }}
  const lo = Math.floor(sorted.length * 0.02);
  const hi = Math.ceil(sorted.length * 0.98) - 1;
  colorVarMin = sorted[Math.max(0,lo)];
  colorVarMax = sorted[Math.min(sorted.length-1,hi)];
}}

function buildColorVarSelect() {{
  const sel = document.getElementById('colorVarSelect');
  if (!sel) return;
  sel.innerHTML = '<option value="">Couleur : élection</option>';
  for (const [cat, vars] of Object.entries(VARS)) {{
    const avail = vars.filter(v => IRIS_X && IRIS_X[v] !== undefined);
    if (!avail.length) continue;
    const og = document.createElement('optgroup');
    og.label = cat;
    avail.forEach(v => {{
      const opt = document.createElement('option');
      opt.value = v;
      opt.textContent = (varLabels[v] || v).substring(0, 60);
      if (v === currentColorVar) opt.selected = true;
      og.appendChild(opt);
    }});
    sel.appendChild(og);
  }}
  sel.disabled = false;
}}

function updateColorLegend() {{
  const leg = document.getElementById('colorLegend');
  if (!leg) return;
  if (!currentColorVar) {{ leg.style.display = 'none'; return; }}
  const label = (varLabels[currentColorVar] || currentColorVar).substring(0, 80);
  const mn = colorVarMin.toFixed(2), mx = colorVarMax.toFixed(2);
  leg.style.display = 'flex';
  leg.innerHTML = `
    <div style="font-size:10px;color:#555;margin-bottom:3px;font-weight:600">${{label}}</div>
    <div style="display:flex;align-items:center;gap:8px">
      <span style="font-size:10px;color:#3B82F6">${{mn}}</span>
      <div style="flex:1;height:10px;border-radius:5px;background:linear-gradient(to right,#3B82F6,#fff,#EF4444)"></div>
      <span style="font-size:10px;color:#EF4444">${{mx}}</span>
    </div>`;
}}

// ── Description panel ─────────────────────────────────────────────────────
const descDiv = document.getElementById('axisDesc');

function updateDesc(preset, xVar, yVar) {{
  if (preset && preset.desc) {{
    const d = preset.desc;
    const qs = preset.corners;
    const qmap = {{}};
    if (preset.desc.quadrants) {{
      for (const [pos, txt] of Object.entries(preset.desc.quadrants)) qmap[pos] = txt;
    }}
    const qOrder = [{{'pos':'tl','label':'↖'}},{{'pos':'tr','label':'↗'}},{{'pos':'bl','label':'↙'}},{{'pos':'br','label':'↘'}}];
    const qHtml = qOrder.filter(q => qmap[q.pos]).map(q => `
      <div class="desc-q">
        <div class="desc-q-label">${{q.label}} ${{qs.find(c=>c.pos===q.pos)?.text?.replace('<br>',' ') || ''}}</div>
        ${{qmap[q.pos]}}
      </div>`).join('');
    descDiv.innerHTML = `
      <div class="desc-title">${{d.title}}</div>
      <div class="desc-axes">
        <div class="desc-ax">${{d.x}}</div>
        <div class="desc-ax">${{d.y}}</div>
      </div>
      ${{qHtml ? `<div class="desc-quadrants"><b style="font-size:11px;color:#888;text-transform:uppercase;letter-spacing:.5px">Quadrants</b><div class="desc-q-grid">${{qHtml}}</div></div>` : ''}}`;
  }} else {{
    const xl = varLabels[xVar] || xVar;
    const yl = varLabels[yVar] || yVar;
    descDiv.innerHTML = `
      <div class="desc-axes" style="grid-template-columns:1fr 1fr">
        <div class="desc-ax"><b>Axe X — ${{xVar}}</b><br>${{xl}}</div>
        <div class="desc-ax"><b>Axe Y — ${{yVar}}</b><br>${{yl}}</div>
      </div>`;
  }}
}}

// ── T1 fallback helper : renvoie le cache T1 si l'élection courante est un T2 ─
function getT1FallbackData(electionId) {{
  if (!electionId.endsWith('_t2')) return null;
  const t1Id = electionId.replace(/_t2$/, '_t1');
  return elecCache[t1Id] || null;
}}

// ── Restyle all IRIS points (single trace 1) ─────────────────────────────
function restyleIRIS() {{
  if (!IRIS_X || !IRIS_Y) return;  // axes pas encore chargés
  const data = elecCache[currentElectionId];
  if (!data) return;
  const t1Data = getT1FallbackData(currentElectionId);
  const xArr = IRIS_X[currentXVarData] || IRIS_X[currentXVar] || [];
  const yArr = IRIS_Y[currentYVarData] || IRIS_Y[currentYVar] || [];
  const colorArr = currentColorVar ? (IRIS_X[currentColorVar] || IRIS_Y[currentColorVar] || null) : null;
  const n = xArr.length;
  const knownKeys = new Set(btns.map(b => b.key));
  const fx = [], fy = [], fc = [], fs = [], fcd = [];
  for (let i = 0; i < n; i++) {{
    let parti = data.partis[i];
    let isFallback = false;
    if ((parti === null || parti === undefined) && t1Data) {{
      parti = t1Data.partis[i];
      isFallback = true;
    }}
    if (parti === null || parti === undefined) continue;
    const effectiveKey = knownKeys.has(parti) ? parti : 'AUTRE';
    if (!activeGroups.has(effectiveKey)) continue;
    const xv = xArr[i], yv = yArr[i];
    if (xv == null || yv == null) continue;
    fx.push(currentXInvert ? -xv : xv);
    fy.push(yv);
    if (colorArr && colorArr[i] !== null && colorArr[i] !== undefined) {{
      fc.push(varToHex(colorArr[i], colorVarMin, colorVarMax));
    }} else {{
      const baseColor = isFallback ? (t1Data.colors[i] || '#9CA3AF') : (data.colors[i] || '#9CA3AF');
      fc.push(colorArr ? '#CCCCCC' : baseColor);
    }}
    fs.push(MARKER_SIZES[i] || 3);
    fcd.push(i);
  }}
  Plotly.restyle(chartDiv, {{
    x: [fx], y: [fy],
    'marker.color': [fc],
    'marker.size': [fs],
    customdata: [fcd],
  }}, [2]);
}}

// ── Restyle barycentres (trace 0) ────────────────────────────────────────
function restyleBarycentres() {{
  const baryX = [], baryY = [], baryColors = [], baryTexts = [], barySzs = [];
  const topG = Object.keys(currentGroupIndices)
    .filter(g => {{
      if (g === 'AUTRE') return false;
      const b = btns.find(b2 => b2.key === g);
      return baryMeans[g] && b && b.pct > 2;
    }})
    .sort((a, b2) => {{
      const pa = btns.find(x => x.key === a)?.pct || 0;
      const pb = btns.find(x => x.key === b2)?.pct || 0;
      return pb - pa;
    }});
  topG.forEach(g => {{
    const xm = (baryMeans[g][currentXVarData] !== undefined ? baryMeans[g][currentXVarData] : null)
               ?? (baryMeans[g][currentXVar] !== undefined ? baryMeans[g][currentXVar] : 0);
    const ym = (baryMeans[g][currentYVarData] !== undefined ? baryMeans[g][currentYVarData] : null)
               ?? (baryMeans[g][currentYVar] !== undefined ? baryMeans[g][currentYVar] : 0);
    baryX.push(currentXInvert ? -xm : xm);
    baryY.push(ym);
    baryColors.push(ALL_PARTIES_COLORS_JS[g] || COULEURS_JS[g] || '#999');
    baryTexts.push(btns.find(b2 => b2.key === g)?.short || g);
    barySzs.push(currentBarySizeMap[g] || 22);
  }});
  const abstBary = computeAbstBary(currentElectionId);
  if (abstBary) {{
    const xm = abstBary[currentXVarData] ?? abstBary[currentXVar],
          ym  = abstBary[currentYVarData] ?? abstBary[currentYVar];
    if (xm !== null && ym !== null) {{
      baryX.push(currentXInvert ? -xm : xm); baryY.push(ym);
      baryColors.push('#9CA3AF'); baryTexts.push('Abst.');
      barySzs.push(currentBarySizeMap['__ABST__'] || 18);
    }}
  }}
  Plotly.restyle(chartDiv, {{
    x: [baryX], y: [baryY],
    'marker.color': [baryColors],
    'marker.line.color': [baryColors],
    'textfont.color': [baryColors],
    text: [baryTexts],
    'marker.size': [barySzs],
  }}, [1]);
}}

// ── Résolution mode strict ML ─────────────────────────────────────────────
function resolveVarStrict(varName) {{
  if (!mlStrictMode || !IRIS_X) return varName;
  const sname = varName + '_strict';
  return IRIS_X[sname] !== undefined ? sname : varName;
}}

// ── Restyle density contour (trace 0) ────────────────────────────────────
function restyleDensity() {{
  if (!IRIS_X || !IRIS_Y || !IRIS_POPS) return;
  if (!densityVisible) {{
    Plotly.restyle(chartDiv, {{x: [[]], y: [[]], z: [[]]}}, [0]);
    return;
  }}
  const xArr = IRIS_X[currentXVarData] || IRIS_X[currentXVar] || [];
  const yArr = IRIS_Y[currentYVarData] || IRIS_Y[currentYVar] || [];
  const n = xArr.length;
  const xs = [], ys = [], ws = [];
  for (let i = 0; i < n; i++) {{
    const xv = xArr[i], yv = yArr[i];
    if (xv == null || yv == null) continue;
    xs.push(currentXInvert ? -xv : xv);
    ys.push(yv);
    ws.push(IRIS_POPS[i] || 1);
  }}
  Plotly.restyle(chartDiv, {{x: [xs], y: [ys], z: [ws]}}, [0]);
}}

// ── Apply axes ────────────────────────────────────────────────────────────
function applyAxes(xVar, xInvert, yVar, preset) {{
  currentXVar = xVar;
  currentYVar = yVar;
  currentXInvert = xInvert;
  // Résoudre les noms de variables pour l'accès aux données (strict si disponible)
  currentXVarData = resolveVarStrict(xVar);
  currentYVarData = resolveVarStrict(yVar);
  const _rangeF = (preset && (preset.id === 'tsne' || preset.id === 'umap'))
    ? computeDataRange_embeddings : computeDataRange;
  if (preset) {{
    currentXRange = preset.xRange ? preset.xRange.slice() : _rangeF(xVar, xInvert);
    currentYRange = preset.yRange ? preset.yRange.slice() : _rangeF(yVar, false);
    currentCorners = preset.corners;
  }} else {{
    currentXRange = _rangeF(xVar, xInvert);
    currentYRange = _rangeF(yVar, false);
  }}

  activeGroups = new Set(btns.filter(b => b.count > 0).map(b => b.key));
  rebuildFilterButtons();
  updateToggleLabel();
  restyleDensity();
  restyleIRIS();
  restyleBarycentres();

  const xTitle = preset ? preset.xTitle : (varLabels[xVar] || xVar) + (xInvert ? ' (inversé)' : '');
  const yTitle = preset ? preset.yTitle : (varLabels[yVar] || yVar);
  Plotly.relayout(chartDiv, {{
    'xaxis.title.text': xTitle,
    'yaxis.title.text': yTitle,
    'xaxis.range': currentXRange.slice(),
    'yaxis.range': currentYRange.slice(),
  }});

  setCorners(preset ? preset.corners : []);
  updateDesc(preset, xVar, yVar);
  document.getElementById('xInvertChk').checked = xInvert;
}}

// ── Variables d'état globales (utilisées par restyleIRIS, restyleBarycentres, applyAxes, etc.) ──
let activeGroups;
const filtersDiv = document.getElementById('groupFilters');
const toggleAllBtn = document.getElementById('toggleAllBtn');
let allOn = true;
let top4Parties;
let currentElectionId;
let currentElectionType, currentElectionYear, currentElectionTour;
let currentClickedGlobalIdx = null;
let currentGroupIndices = {{}};
let elecByType = {{}};
const typeLabels = {{legi: 'Législatives', euro: 'Européennes', pres: 'Présidentielles', muni: 'Municipales'}};
const elecTypeBtns = document.getElementById('elecTypeBtns');
const elecYearBtns = document.getElementById('elecYearBtns');
const tourBtn1 = document.getElementById('tourBtn1');
const tourBtn2 = document.getElementById('tourBtn2');
const electionLabel = document.getElementById('electionCurrentLabel');
let baryMeans = {{}};
let currentBarySizeMap = {{}};

// ── Fonctions partagées (utilisées cross-scope) ──────────────────────────
function computeButtonPcts(electionId) {{
  const data = elecCache[electionId];
  if (!data || !data.buttonPcts) return;
  btns.forEach(b => {{
    b.pct = data.buttonPcts[b.key] || 0;
  }});
}}

function updateToggleLabel() {{
  const enabledBtns = btns.filter(b => b.count > 0);
  allOn = enabledBtns.every(b => activeGroups.has(b.key));
  toggleAllBtn.textContent = allOn ? 'Tout décocher' : 'Tout cocher';
}}

function setGroupVisible(b, visible) {{
  const el = filtersDiv.querySelector('[data-key="' + b.key + '"]');
  const xr = (chartDiv.layout && chartDiv.layout.xaxis) ? chartDiv.layout.xaxis.range.slice() : currentXRange.slice();
  const yr = (chartDiv.layout && chartDiv.layout.yaxis) ? chartDiv.layout.yaxis.range.slice() : currentYRange.slice();
  if (visible) {{
    activeGroups.add(b.key);
    if (el) {{ el.classList.replace('off','on'); el.style.backgroundColor = b.color; el.style.color = '#fff'; }}
  }} else {{
    activeGroups.delete(b.key);
    if (el) {{ el.classList.replace('on','off'); el.style.backgroundColor = 'transparent'; el.style.color = b.color; }}
  }}
  restyleIRIS();
  updateMapColors();
  if (!isCarteActive) Plotly.relayout(chartDiv, {{'xaxis.range': xr, 'yaxis.range': yr}});
}}

function rebuildFilterButtons() {{
  filtersDiv.innerHTML = '';
  const activeBtns = btns.filter(b => b.count > 0)
                         .sort((a, b2) => (b2.pct || 0) - (a.pct || 0));
  activeBtns.forEach(b => {{
    const el = document.createElement('button');
    const isOn = activeGroups.has(b.key);
    el.className = 'grp-btn ' + (isOn ? 'on' : 'off');
    el.dataset.key = b.key;
    el.style.borderColor = b.color;
    el.style.backgroundColor = isOn ? b.color : 'transparent';
    el.style.color = isOn ? '#fff' : b.color;
    el.innerHTML = b.short + ' <span class="grp-count" style="font-size:8px;font-weight:400;opacity:0.7">' + (b.pct || 0).toFixed(1) + '%</span>';
    el.addEventListener('click', () => {{
      setGroupVisible(b, !activeGroups.has(b.key));
      updateToggleLabel();
    }});
    filtersDiv.appendChild(el);
  }});
}}

function getElectionId(type, year, tour) {{
  const years = elecByType[type];
  if (!years || !years[year]) return null;
  const found = years[year].find(e => e.tour === tour);
  return found ? found.id : null;
}}

// ── Info card helpers ──────────────────────────────────────────────────────
function fmtPct(v) {{ return (v !== '' && v !== null && !isNaN(Number(v))) ? Number(v).toFixed(1) + '%' : '—'; }}
function fmtNum(v, suffix) {{ return (v !== '' && v !== null && !isNaN(Number(v))) ? Math.round(Number(v)).toLocaleString('fr-FR') + (suffix||'') : '—'; }}

function updateAbstPanel(electionId) {{
  const panel = document.getElementById('abst-stats-panel');
  if (!panel) return;
  const data = elecCache[electionId];
  if (!data || !data.abst) {{ panel.style.display = 'none'; return; }}
  const hasInscrits = data.inscrits && data.inscrits.some(v => v != null);
  const hasExprimes = data.exprimes && data.exprimes.some(v => v != null);
  const hasBlancs = data.blancs && data.blancs.some(v => v != null);
  const hasNuls = data.nuls && data.nuls.some(v => v != null);
  let totalInscrits = 0, totalAbst = 0, totalExprimes = 0, totalBlancs = 0, totalNuls = 0;
  for (let i = 0; i < data.abst.length; i++) {{
    const abst = data.abst[i];
    if (abst == null || isNaN(abst)) continue;
    const ins = hasInscrits ? (data.inscrits[i] || 0) : (IRIS_POPS[i] || 0);
    totalInscrits += ins;
    totalAbst += ins * abst / 100;
    if (hasExprimes) totalExprimes += data.exprimes[i] || 0;
    if (hasBlancs) totalBlancs += data.blancs[i] || 0;
    if (hasNuls) totalNuls += data.nuls[i] || 0;
  }}
  const pctAbst = totalInscrits > 0 ? totalAbst / totalInscrits * 100 : null;
  const partyTotals = {{}};
  for (let i = 0; i < data.scores.length; i++) {{
    const sc = data.scores[i];
    if (!sc || typeof sc !== 'object') continue;
    const exp = hasExprimes ? (data.exprimes[i] || 0) : (hasInscrits ? (data.inscrits[i] || 0) : (IRIS_POPS[i] || 0));
    Object.entries(sc).forEach(([g, s]) => {{
      if (s > 0) partyTotals[g] = (partyTotals[g] || 0) + exp * s / 100;
    }});
  }}
  const top7 = Object.entries(partyTotals).sort((a, b) => b[1] - a[1]).slice(0, 7);
  const fmt = n => n >= 1e6 ? (n/1e6).toFixed(2)+'M' : n >= 1000 ? (n/1000).toFixed(1)+'k' : Math.round(n).toString();
  const fmtP = p => p != null ? p.toFixed(1)+'%' : '–';
  let html = `<span style="font-weight:600">Abstention&nbsp;:</span> <b>${{fmtP(pctAbst)}}</b>`;
  if (totalInscrits > 0) html += ` (${{fmt(totalAbst)}} / ${{fmt(totalInscrits)}} ins.)`;
  if (hasBlancs || hasNuls) {{
    html += ' &nbsp;·&nbsp;';
    if (hasBlancs) {{
      const pctBlancsIns = totalInscrits > 0 ? totalBlancs / totalInscrits * 100 : 0;
      html += ` <span style="font-weight:600">Blancs&nbsp;:</span> ${{fmtP(pctBlancsIns)}} ins. (${{fmt(totalBlancs)}})`;
    }}
    if (hasNuls) {{
      const pctNulsIns = totalInscrits > 0 ? totalNuls / totalInscrits * 100 : 0;
      html += ` &nbsp;·&nbsp; <span style="font-weight:600">Nuls&nbsp;:</span> ${{fmtP(pctNulsIns)}} ins. (${{fmt(totalNuls)}})`;
    }}
  }}
  html += ' &nbsp;·&nbsp; <span style="font-weight:600">Top partis&nbsp;:</span> ';
  html += top7.map(([g, v]) => {{
    const pctExp = totalExprimes > 0 ? v / totalExprimes * 100 : 0;
    const pctIns = totalInscrits > 0 ? v / totalInscrits * 100 : 0;
    const color = ALL_PARTIES_COLORS_JS[g] || '#999';
    return `<span style="color:${{color}};font-weight:600">${{g}}</span>&nbsp;${{fmtP(pctExp)}} exp.&nbsp;/${{fmtP(pctIns)}} ins. (${{fmt(v)}})`;
  }}).join(' · ');
  panel.innerHTML = html;
  panel.style.display = 'block';
}}

function computeAbstBary(electionId) {{
  const data = elecCache[electionId];
  return (data && data.abstBary) ? data.abstBary : null;
}}

async function applyElection(electionId) {{
  const t1IdNeeded = electionId.endsWith('_t2') ? electionId.replace(/_t2$/, '_t1') : null;
  const fetches = [];
  if (!elecCache[electionId]) fetches.push(electionId);
  if (t1IdNeeded && !elecCache[t1IdNeeded]) fetches.push(t1IdNeeded);
  if (fetches.length > 0) {{
    const spinner = document.getElementById('elecSpinner');
    if (spinner) spinner.style.display = 'inline';
    try {{
      await Promise.all(fetches.map(id =>
        fetch('data/elec_' + id + '.json').then(r => r.json()).then(d => {{ elecCache[id] = d; }})
      ));
    }} finally {{
      if (spinner) spinner.style.display = 'none';
    }}
  }}
  const data = elecCache[electionId];
  if (!data) return;
  currentElectionId = electionId;
  computeButtonPcts(electionId);

  const newGroups = {{}};
  ORDER.forEach(g => newGroups[g] = []);
  data.partis.forEach((parti, i) => {{
    if (parti === null || parti === undefined) return;
    if (newGroups[parti] !== undefined) {{
      newGroups[parti].push(i);
    }} else if (newGroups['AUTRE'] !== undefined) {{
      newGroups['AUTRE'].push(i);
    }}
  }});
  const oldEnabledBtns = btns.filter(b => (currentGroupIndices[b.key] || []).length > 0);
  const wasAllOn = oldEnabledBtns.length > 0 && oldEnabledBtns.every(b => activeGroups.has(b.key));
  const wasAllOff = oldEnabledBtns.every(b => !activeGroups.has(b.key));

  currentGroupIndices = newGroups;

  const enabledBtns = btns.filter(b => (newGroups[b.key] || []).length > 0);
  const enabledKeys = new Set(enabledBtns.map(b => b.key));
  btns.forEach(b => {{ b.count = (newGroups[b.key] || []).length; }});

  if (wasAllOn) {{
    activeGroups = new Set(enabledKeys);
  }} else if (wasAllOff) {{
    activeGroups = new Set();
  }} else {{
    const intersection = new Set([...activeGroups].filter(k => enabledKeys.has(k)));
    const intersectionReal = new Set([...intersection].filter(k => k !== 'AUTRE'));
    activeGroups = intersectionReal.size > 0 ? intersection : new Set(enabledKeys);
  }}

  const xr = (chartDiv.layout && chartDiv.layout.xaxis) ? chartDiv.layout.xaxis.range.slice() : currentXRange.slice();
  const yr = (chartDiv.layout && chartDiv.layout.yaxis) ? chartDiv.layout.yaxis.range.slice() : currentYRange.slice();
  restyleIRIS();
  rebuildFilterButtons();
  Plotly.relayout(chartDiv, {{ 'xaxis.range': xr, 'yaxis.range': yr }});

  baryMeans = (data.baryMeans) ? data.baryMeans : {{}};
  currentBarySizeMap = (data.barySizes) ? data.barySizes : {{}};
  restyleBarycentres();

  updateAbstPanel(electionId);

  const meta = ELECTIONS_META[electionId];
  electionLabel.textContent = meta ? meta.label : electionId;
  updateToggleLabel();

  if (currentClickedGlobalIdx !== null) {{
    const partisData = elecCache[currentElectionId];
    const hasT2Data = partisData && partisData.partis[currentClickedGlobalIdx] !== null;
    const t1Fb = getT1FallbackData(currentElectionId);
    const hasT1Fallback = !hasT2Data && t1Fb && t1Fb.partis[currentClickedGlobalIdx] !== null;
    if (hasT2Data || hasT1Fallback) {{
      showDesktopCardElec(currentClickedGlobalIdx);
    }} else {{
      const card = document.getElementById('infoCard');
      card.classList.remove('empty');
      card.innerHTML = '<div style="color:#AAA;padding:16px;font-size:12px;text-align:center">Pas de données pour cet IRIS<br>dans cette élection.</div>';
    }}
  }}
  updateMapColors();
}}

function updateYearBtns(type) {{
  elecYearBtns.innerHTML = '';
  const years = elecByType[type] ? Object.keys(elecByType[type]).sort() : [];
  years.forEach(year => {{
    const btn = document.createElement('button');
    btn.className = 'elec-year-btn' + (parseInt(year) === currentElectionYear && type === currentElectionType ? ' active' : '');
    btn.textContent = year;
    btn.dataset.year = year;
    btn.addEventListener('click', () => {{
      currentElectionYear = parseInt(year);
      elecYearBtns.querySelectorAll('.elec-year-btn').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      updateTourBtns(type, parseInt(year));
      const eid = getElectionId(type, parseInt(year), currentElectionTour)
             || getElectionId(type, parseInt(year), 1);
      if (eid) {{ currentElectionTour = ELECTIONS_META[eid].tour; applyElection(eid); }}
    }});
    elecYearBtns.appendChild(btn);
  }});
}}

function updateTourBtns(type, year) {{
  const years = elecByType[type];
  const available = years && years[year] ? years[year].map(e => e.tour) : [];
  tourBtn1.disabled = !available.includes(1);
  tourBtn2.disabled = !available.includes(2);
  tourBtn1.classList.toggle('active', currentElectionTour === 1 && available.includes(1));
  tourBtn2.classList.toggle('active', currentElectionTour === 2 && available.includes(2));
}}

// ── buildSelect (scope global, appelé après chargement de IRIS_X) ─────────
function buildSelect(selectId, selectedVar) {{
  const sel = document.getElementById(selectId);
  sel.innerHTML = '';
  for (const [cat, vars] of Object.entries(VARS)) {{
    const availVars = vars.filter(v => IRIS_X[v] !== undefined);
    if (availVars.length === 0) continue;
    const og = document.createElement('optgroup');
    og.label = cat;
    availVars.forEach(v => {{
      const opt = document.createElement('option');
      opt.value = v;
      opt.textContent = v + (varLabels[v] ? ' — ' + varLabels[v].substring(0, 50) : '');
      if (v === selectedVar) opt.selected = true;
      og.appendChild(opt);
    }});
    sel.appendChild(og);
  }}
}}

// ── initUI : construit les boutons et initialise l'état ───────────────────
function initUI() {{
// ── Preset buttons ────────────────────────────────────────────────────────
const presetBtnsDiv = document.getElementById('presetBtns');
PRESETS.forEach(p => {{
  // Les presets t-SNE/UMAP seront masqués après chargement de IRIS_X si absents
  const btn = document.createElement('button');
  btn.className = 'preset-btn' + (p.id === 'saint_graphique' ? ' active' : '');
  btn.dataset.id = p.id;
  btn.textContent = p.emoji + ' ' + p.label;
  btn.addEventListener('click', () => {{
    hideCarte();
    document.querySelectorAll('.preset-btn').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
    currentPresetId = p.id;
    document.getElementById('xSelect').value = p.xVar;
    document.getElementById('ySelect').value = p.yVar;
    document.getElementById('xInvertChk').checked = p.xInvert;
    applyAxes(p.xVar, p.xInvert, p.yVar, p);
  }});
  presetBtnsDiv.appendChild(btn);
}});

const carteBtn = document.createElement('button');
carteBtn.className = 'preset-btn';
carteBtn.dataset.id = 'carte';
carteBtn.textContent = '🗺️ Carte';
carteBtn.addEventListener('click', async () => {{
  document.querySelectorAll('.preset-btn').forEach(b => b.classList.remove('active'));
  carteBtn.classList.add('active');
  currentPresetId = 'carte';
  await showCarte();
}});
presetBtnsDiv.appendChild(carteBtn);

const customToggle = document.getElementById('customToggle');
const customPanel = document.getElementById('customPanel');
customToggle.addEventListener('click', () => {{
  const open = customPanel.classList.toggle('open');
  customToggle.classList.toggle('open', open);
  customToggle.textContent = open ? 'Personnaliser ▴' : 'Personnaliser ▾';
}});

function onCustomChange() {{
  hideCarte();
  const xVar = document.getElementById('xSelect').value;
  const yVar = document.getElementById('ySelect').value;
  const xInvert = document.getElementById('xInvertChk').checked;
  document.querySelectorAll('.preset-btn').forEach(b => b.classList.remove('active'));
  currentPresetId = null;
  const matchPreset = PRESETS.find(p => p.xVar === xVar && p.yVar === yVar && p.xInvert === xInvert);
  if (matchPreset) {{
    document.querySelectorAll('.preset-btn').forEach(b => {{
      if (b.dataset.id === matchPreset.id) b.classList.add('active');
    }});
  }}
  applyAxes(xVar, xInvert, yVar, matchPreset || null);
}}

document.getElementById('xSelect').addEventListener('change', onCustomChange);
document.getElementById('ySelect').addEventListener('change', onCustomChange);
document.getElementById('xInvertChk').addEventListener('change', onCustomChange);

document.getElementById('colorVarSelect').addEventListener('change', function() {{
  const v = this.value;
  currentColorVar = v || null;
  if (currentColorVar) computeColorVarRange(currentColorVar);
  restyleIRIS();
  updateMapColors();
  updateColorLegend();
}});

// ── Toggle all button ─────────────────────────────────────────────────────
toggleAllBtn.addEventListener('click', () => {{
  const enabledBtns = btns.filter(b => b.count > 0);
  const xr = chartDiv.layout ? chartDiv.layout.xaxis.range.slice() : currentXRange.slice();
  const yr = chartDiv.layout ? chartDiv.layout.yaxis.range.slice() : currentYRange.slice();
  if (allOn) {{
    enabledBtns.forEach(b => {{
      activeGroups.delete(b.key);
      const el = filtersDiv.querySelector('[data-key="' + b.key + '"]');
      if (el) {{ el.classList.replace('on','off'); el.style.backgroundColor = 'transparent'; el.style.color = b.color; }}
    }});
  }} else {{
    enabledBtns.forEach(b => {{
      activeGroups.add(b.key);
      const el = filtersDiv.querySelector('[data-key="' + b.key + '"]');
      if (el) {{ el.classList.replace('off','on'); el.style.backgroundColor = b.color; el.style.color = '#fff'; }}
    }});
  }}
  restyleIRIS();
  updateMapColors();
  restyleBarycentres();
  Plotly.relayout(chartDiv, {{'xaxis.range': xr, 'yaxis.range': yr}});
  updateToggleLabel();
}});

// ── Tour buttons ──────────────────────────────────────────────────────────
tourBtn1.addEventListener('click', () => {{
  if (tourBtn1.disabled) return;
  currentElectionTour = 1;
  updateTourBtns(currentElectionType, currentElectionYear);
  const eid = getElectionId(currentElectionType, currentElectionYear, 1);
  if (eid) applyElection(eid);
}});
tourBtn2.addEventListener('click', () => {{
  if (tourBtn2.disabled) return;
  currentElectionTour = 2;
  updateTourBtns(currentElectionType, currentElectionYear);
  const eid = getElectionId(currentElectionType, currentElectionYear, 2);
  if (eid) applyElection(eid);
}});

// ── Init state ────────────────────────────────────────────────────────────
currentElectionId = DEFAULT_ELECTION_ID;
currentElectionType = ELECTIONS_META[DEFAULT_ELECTION_ID]?.type || 'legi';
currentElectionYear = ELECTIONS_META[DEFAULT_ELECTION_ID]?.year || 2022;
currentElectionTour = ELECTIONS_META[DEFAULT_ELECTION_ID]?.tour || 1;
currentGroupIndices = {{}};  // sera rempli après chargement de GROUP_INDICES (static.json)
activeGroups = new Set(btns.map(b => b.key));

// Grouper les élections par type → year
elecByType = {{}};
for (const [eid, meta] of Object.entries(ELECTIONS_META)) {{
  if (!elecByType[meta.type]) elecByType[meta.type] = {{}};
  if (!elecByType[meta.type][meta.year]) elecByType[meta.type][meta.year] = [];
  elecByType[meta.type][meta.year].push({{id: eid, tour: meta.tour}});
}}

// Build type buttons (après remplissage de elecByType)
for (const [type, label] of Object.entries(typeLabels)) {{
  if (!elecByType[type]) continue;
  const btn = document.createElement('button');
  btn.className = 'elec-type-btn' + (type === currentElectionType ? ' active' : '');
  btn.textContent = label;
  btn.dataset.type = type;
  btn.addEventListener('click', () => {{
    currentElectionType = type;
    elecTypeBtns.querySelectorAll('.elec-type-btn').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
    const years = Object.keys(elecByType[type] || {{}}).sort();
    currentElectionYear = parseInt(years[years.length - 1]) || 2022;
    currentElectionTour = 1;
    updateYearBtns(type);
    updateTourBtns(type, currentElectionYear);
    const eid = getElectionId(type, currentElectionYear, 1)
           || getElectionId(type, currentElectionYear, 2);
    if (eid) applyElection(eid);
  }});
  elecTypeBtns.appendChild(btn);
}}

// Init year/tour buttons for default election
updateYearBtns(currentElectionType);
updateTourBtns(currentElectionType, currentElectionYear);
electionLabel.textContent = ELECTIONS_META[DEFAULT_ELECTION_ID]?.label || DEFAULT_ELECTION_ID;

// Dropdowns axes désactivés jusqu'au chargement de IRIS_X/IRIS_Y
document.getElementById('xSelect').disabled = true;
document.getElementById('ySelect').disabled = true;

// Initialiser les filtres partis (btns.count sera mis à jour après applyElection)
top4Parties = new Set(btns.slice().sort((a,b) => b.count - a.count).slice(0,4).map(b => b.key));
rebuildFilterButtons();
updateToggleLabel();

// ── Bouton toggle ML ──────────────────────────────────────────────────────
const mlToggleBtn = document.getElementById('mlToggleBtn');
if (mlToggleBtn) {{
  mlToggleBtn.style.display = 'none'; // masqué par défaut, affiché si ML_FLAGS chargé
  mlToggleBtn.addEventListener('click', () => {{
    mlStrictMode = !mlStrictMode;
    mlToggleBtn.textContent = mlStrictMode ? '⚠ Imputés ML : masqués' : '⚠ Imputés ML : affichés';
    mlToggleBtn.classList.toggle('full-mode', !mlStrictMode);
    // Recalculer les variables résolues et rafraîchir le graphique
    currentXVarData = resolveVarStrict(currentXVar);
    currentYVarData = resolveVarStrict(currentYVar);
    restyleDensity();
    restyleIRIS();
    restyleBarycentres();
  }});
}}

// ── Bouton toggle densité ─────────────────────────────────────────────────
const densityToggleBtn = document.getElementById('densityToggleBtn');
if (densityToggleBtn) {{
  densityToggleBtn.addEventListener('click', () => {{
    densityVisible = !densityVisible;
    densityToggleBtn.textContent = densityVisible ? 'Densité : affichée' : 'Densité : masquée';
    densityToggleBtn.style.color = densityVisible ? '#888' : '#CCC';
    densityToggleBtn.style.borderColor = densityVisible ? '#AAA' : '#DDD';
    restyleDensity();
  }});
}}

}} // fin initUI

// ── Fonction helper ML flags ──────────────────────────────────────────────
function isMLImputed(irisIdx, dispVarName) {{
  return ML_FLAGS && ML_FLAGS[dispVarName] && ML_FLAGS[dispVarName][irisIdx] === 1;
}}

// ── Info card using election-specific scores (lire depuis IRIS_INFO) ──
function showDesktopCardElec(irisGlobalIdx) {{
  const cd = IRIS_INFO[irisGlobalIdx];
  if (!cd) return;
  currentClickedGlobalIdx = irisGlobalIdx;
  const card = document.getElementById('infoCard');
  card.classList.remove('empty');

  const xRaw = IRIS_X ? (IRIS_X[currentXVar] || [])[irisGlobalIdx] : undefined;
  const yRaw = IRIS_Y ? (IRIS_Y[currentYVar] || [])[irisGlobalIdx] : undefined;
  const xDisp = xRaw !== undefined ? (currentXInvert ? -xRaw : xRaw) : undefined;

  const elecData = elecCache[currentElectionId];
  const t1Fb = getT1FallbackData(currentElectionId);
  const hasT2 = elecData && elecData.partis[irisGlobalIdx] !== null;
  const activeData = hasT2 ? elecData : (t1Fb || elecData);
  const isT1Fallback = !hasT2 && !!t1Fb;

  const sticky = document.getElementById('sidebarSticky');
  if (sticky) {{ sticky.style.display = 'block'; sticky.style.color = activeData?.colors[irisGlobalIdx] || '#9CA3AF'; sticky.textContent = cd[1] || cd[0] || ''; }}
  const elecScores = activeData ? activeData.scores[irisGlobalIdx] : null;
  const currentMeta = ELECTIONS_META[currentElectionId];
  const currentColor = activeData?.colors[irisGlobalIdx] || '#9CA3AF';
  const currentParti = activeData?.partis[irisGlobalIdx] || '—';

  let voteBarsHtml = '';
  if (elecScores && Object.keys(elecScores).length > 0) {{
    const allScores = Object.entries(elecScores).filter(([,v]) => v > 0).sort((a,b) => b[1]-a[1]);
    voteBarsHtml = allScores.map(([p, score], idx) => {{
      const color = ALL_PARTIES_COLORS_JS[p] || '#9CA3AF';
      const pct = score.toFixed(1) + '%';
      const w = Math.min(100, score);
      const isTop = idx === 0;
      return `<div class="vote-bar-row">
        <div class="vote-bar-label" style="color:${{color}};${{isTop ? 'font-weight:900' : ''}}">${{p.replace('_',' ')}}</div>
        <div class="vote-bar-bg" style="${{isTop ? 'height:7px' : ''}}"><div class="vote-bar-fill" style="width:${{w}}%;background:${{color}}"></div></div>
        <div class="vote-bar-pct" style="${{isTop ? 'font-weight:700;color:#1a1a1a' : ''}}">${{pct}}</div>
      </div>`;
    }}).join('');
    if (isT1Fallback) voteBarsHtml = `<div style="color:#AAA;font-size:10px;margin-bottom:5px;font-style:italic">(remporté au T1)</div>` + voteBarsHtml;
  }}

  card.innerHTML = `
    <div class="name">${{cd[1] || cd[0] || 'IRIS inconnu'}} <span style="color:#AAA;font-weight:400;font-size:11px">${{cd[0] || ''}}</span></div>
    <div class="party" style="color:#666;font-size:12px">${{cd[19] || ''}}</div>
    <div class="row"><span class="lbl">Population :</span> <b>${{fmtNum(cd[2], ' hab.')}}</b> &nbsp;·&nbsp; <span class="lbl">Âge moyen :</span> <b>${{cd[18] !== '' ? Number(cd[18]).toFixed(1) + ' ans' : '—'}}</b></div>
    <div class="row"><span class="lbl">Revenu médian :</span> <b>${{fmtNum(cd[3], ' €/UC')}}</b>${{isMLImputed(irisGlobalIdx, 'DISP_MED21') ? '<span class="ml-flag">(imputé ML)</span>' : ''}}</div>
    <div class="section-title">Catégories socio-professionnelles</div>
    <div class="stat-bar-row"><div class="stat-bar-label">Cadres sup.</div><div class="stat-bar-bg"><div class="stat-bar-fill" style="width:${{Math.min(100,cd[4]||0)}}%;background:#5B8DB8"></div></div><div class="stat-bar-pct">${{fmtPct(cd[4])}}</div></div>
    <div class="stat-bar-row"><div class="stat-bar-label">Prof. interm.</div><div class="stat-bar-bg"><div class="stat-bar-fill" style="width:${{Math.min(100,cd[6]||0)}}%;background:#82AAC8"></div></div><div class="stat-bar-pct">${{fmtPct(cd[6])}}</div></div>
    <div class="stat-bar-row"><div class="stat-bar-label">Ouvriers</div><div class="stat-bar-bg"><div class="stat-bar-fill" style="width:${{Math.min(100,cd[5]||0)}}%;background:#C97A5A"></div></div><div class="stat-bar-pct">${{fmtPct(cd[5])}}</div></div>
    <div class="section-title">Formation &amp; Emploi</div>
    <div class="stat-bar-row"><div class="stat-bar-label">Bac+</div><div class="stat-bar-bg"><div class="stat-bar-fill" style="width:${{Math.min(100,cd[16]||0)}}%;background:#7AAD8F"></div></div><div class="stat-bar-pct">${{fmtPct(cd[16])}}</div></div>
    <div class="stat-bar-row"><div class="stat-bar-label">Sans diplôme</div><div class="stat-bar-bg"><div class="stat-bar-fill" style="width:${{Math.min(100,cd[17]||0)}}%;background:#C9A45A"></div></div><div class="stat-bar-pct">${{fmtPct(cd[17])}}</div></div>
    <div class="stat-bar-row"><div class="stat-bar-label">Chômage</div><div class="stat-bar-bg"><div class="stat-bar-fill" style="width:${{Math.min(100,cd[15]||0)}}%;background:#C95A5A"></div></div><div class="stat-bar-pct">${{fmtPct(cd[15])}}</div></div>
    <div class="section-title">Logement</div>
    <div class="stat-bar-row"><div class="stat-bar-label">Propriétaires</div><div class="stat-bar-bg"><div class="stat-bar-fill" style="width:${{Math.min(100,cd[13]||0)}}%;background:#7AA7D0"></div></div><div class="stat-bar-pct">${{fmtPct(cd[13])}}</div></div>
    <div class="stat-bar-row"><div class="stat-bar-label">HLM</div><div class="stat-bar-bg"><div class="stat-bar-fill" style="width:${{Math.min(100,cd[14]||0)}}%;background:#E8975A"></div></div><div class="stat-bar-pct">${{fmtPct(cd[14])}}</div></div>
    <div class="section-title">${{currentMeta ? currentMeta.label : currentElectionId}}</div>
    ${{(function() {{
      const abstVal = activeData ? activeData.abst[irisGlobalIdx] : null;
      const blancVal = activeData ? activeData.blancs[irisGlobalIdx] : null;
      const inscrVal = activeData ? activeData.inscrits[irisGlobalIdx] : null;
      const blancPct = (blancVal != null && inscrVal != null && inscrVal > 0) ? blancVal / inscrVal * 100 : null;
      let row = '';
      if (abstVal != null) row += `<span class="lbl">Abstention :</span> <b>${{fmtPct(abstVal)}}</b>`;
      if (blancPct != null) row += (row ? ' &nbsp;·&nbsp; ' : '') + `<span class="lbl">Blancs :</span> <b>${{fmtPct(blancPct)}}</b>`;
      return row ? `<div class="row">${{row}}</div>` : '';
    }})()}}
    <div class="vote-bars">${{voteBarsHtml || '<span style="color:#AAA;font-size:11px">Données non disponibles</span>'}}</div>
    <div class="dynamic-row"><span class="lbl">Axe X (${{currentXVar}}) :</span> <b>${{xDisp !== undefined ? Number(xDisp).toFixed(3) : '—'}}</b></div>
    <div class="dynamic-row"><span class="lbl">Axe Y (${{currentYVar}}) :</span> <b>${{yRaw !== undefined ? Number(yRaw).toFixed(3) : '—'}}</b></div>
  `;
}}

// ── Carte MapLibre ────────────────────────────────────────────────────────
function buildMapGeoJSON() {{
  const elecData = elecCache[currentElectionId];
  if (!elecData || !IRIS_LAT || !IRIS_LON) return null;
  const t1Data = getT1FallbackData(currentElectionId);
  const enabledKeys = new Set(btns.filter(b => activeGroups.has(b.key)).map(b => b.key));
  const knownPartiKeys = new Set(btns.map(b => b.key));
  const colorArr = currentColorVar ? (IRIS_X && (IRIS_X[currentColorVar] || null)) || (IRIS_Y && (IRIS_Y[currentColorVar] || null)) : null;
  const features = [];
  for (let i = 0; i < IRIS_LAT.length; i++) {{
    if (IRIS_LAT[i] === null) continue;
    let parti = elecData.partis[i];
    let isFallback = false;
    if ((parti === null || parti === undefined) && t1Data) {{
      parti = t1Data.partis[i];
      isFallback = true;
    }}
    const effectiveKey = knownPartiKeys.has(parti) ? parti : 'AUTRE';
    if (!enabledKeys.has(effectiveKey)) continue;
    let color;
    if (colorArr && colorArr[i] !== null && colorArr[i] !== undefined) {{
      color = varToHex(colorArr[i], colorVarMin, colorVarMax);
    }} else {{
      const baseColor = isFallback ? (t1Data.colors[i] || '#9CA3AF') : (elecData.colors[i] || '#9CA3AF');
      color = colorArr ? '#CCCCCC' : baseColor;
    }}
    features.push({{
      type: 'Feature',
      geometry: {{ type: 'Point', coordinates: [IRIS_LON[i], IRIS_LAT[i]] }},
      properties: {{ idx: i, color, size: MARKER_SIZES[i] || 3 }}
    }});
  }}
  return {{ type: 'FeatureCollection', features }};
}}

function updateMapColors() {{
  if (!isCarteActive || !mapReady) return;
  const gj = buildMapGeoJSON();
  if (!gj) return;
  mapInstance.getSource('iris').setData(gj);
  domMaps.forEach((m, i) => {{ if (domMapsReady[i]) m.getSource('iris').setData(gj); }});
}}

function _addIrisLayer(map, sourceId, layerId, clickCb) {{
  map.addSource(sourceId, {{ type: 'geojson', data: {{ type: 'FeatureCollection', features: [] }} }});
  map.addLayer({{
    id: layerId, type: 'circle', source: sourceId,
    paint: {{
      'circle-color': ['get', 'color'],
      'circle-radius': ['interpolate', ['linear'], ['zoom'],
        5, ['*', ['get', 'size'], 0.35],
        10, ['*', ['get', 'size'], 1.3],
        14, ['*', ['get', 'size'], 2.5]
      ],
      'circle-opacity': 0.85,
      'circle-stroke-width': 0.5,
      'circle-stroke-color': 'rgba(255,255,255,0.4)',
    }}
  }});
  if (clickCb) {{
    map.on('click', layerId, e => clickCb(e.features[0].properties.idx));
    map.on('mouseenter', layerId, () => map.getCanvas().style.cursor = 'pointer');
    map.on('mouseleave', layerId, () => map.getCanvas().style.cursor = '');
  }}
}}

function initMap() {{
  if (mapInitialized) return;
  mapInitialized = true;
  mapInstance = new maplibregl.Map({{
    container: 'mapDiv',
    style: 'https://tiles.openfreemap.org/styles/bright',
    bounds: [[-5.2, 41.3], [9.6, 51.2]],
    fitBoundsOptions: {{ padding: 20 }},
    minZoom: 4,
    maxZoom: 16,
  }});
  mapInstance.on('load', () => {{
    mapReady = true;
    const overlay = document.createElement('div');
    overlay.style.cssText = 'position:absolute;inset:0;background:rgba(255,255,255,0.22);pointer-events:none;z-index:2';
    document.getElementById('mapDiv').appendChild(overlay);
    _addIrisLayer(mapInstance, 'iris', 'iris-circles', idx => {{
      currentClickedGlobalIdx = idx;
      showDesktopCardElec(idx);
    }});
    updateMapColors();
  }});

  // Mini-maps DOM-TOM
  document.getElementById('domMapsRow').style.display = 'flex';
  DOM_TOM_CONFIGS.forEach((cfg, i) => {{
    const m = new maplibregl.Map({{
      container: cfg.id,
      style: 'https://tiles.openfreemap.org/styles/bright',
      bounds: cfg.bounds,
      fitBoundsOptions: {{ padding: 5 }},
      minZoom: 4, maxZoom: 16,
      attributionControl: false,
      navigationControl: false,
    }});
    m.addControl(new maplibregl.AttributionControl({{ compact: true }}));
    domMaps.push(m);
    domMapsReady.push(false);
    m.on('load', () => {{
      domMapsReady[i] = true;
      const ov = document.createElement('div');
      ov.style.cssText = 'position:absolute;inset:0;background:rgba(255,255,255,0.22);pointer-events:none;z-index:2';
      document.getElementById(cfg.id).appendChild(ov);
      _addIrisLayer(m, 'iris', 'iris-circles', idx => {{
        currentClickedGlobalIdx = idx;
        showDesktopCardElec(idx);
      }});
      const gj = buildMapGeoJSON();
      if (gj) m.getSource('iris').setData(gj);
    }});
  }});
}}

async function showCarte() {{
  isCarteActive = true;
  document.getElementById('chartDiv').style.display = 'none';
  document.querySelectorAll('.corner-label').forEach(el => el.style.display = 'none');
  document.getElementById('axisDesc').style.display = 'none';
  document.getElementById('mapDiv').style.display = 'block';
  if (!IRIS_LAT) {{
    document.getElementById('carteLoadingMsg').style.display = 'block';
    const geoData = await fetch('data/geo.json').then(r => r.json());
    IRIS_LAT = geoData.lat;
    IRIS_LON = geoData.lon;
    document.getElementById('carteLoadingMsg').style.display = 'none';
  }}
  if (!mapInitialized) {{
    initMap();  // initialise aussi domMapsRow
  }} else {{
    document.getElementById('domMapsRow').style.display = 'flex';
    updateMapColors();
  }}
}}

function hideCarte() {{
  if (!isCarteActive) return;
  isCarteActive = false;
  document.getElementById('mapDiv').style.display = 'none';
  document.getElementById('domMapsRow').style.display = 'none';
  document.getElementById('chartDiv').style.display = 'block';
  document.querySelectorAll('.corner-label').forEach(el => el.style.display = '');
  document.getElementById('axisDesc').style.display = '';
}}

// ── Helpers overlay de chargement ─────────────────────────────────────────
function setLoadingProgress(pct, msg, detail) {{
  const bar = document.getElementById('loadingBar');
  const msgEl = document.getElementById('loadingMsg');
  const detailEl = document.getElementById('loadingDetail');
  if (bar) bar.style.width = pct + '%';
  if (msg && msgEl) msgEl.textContent = msg;
  if (detailEl) detailEl.textContent = detail || '';
}}
function hideLoadingOverlay() {{
  const ov = document.getElementById('loadingOverlay');
  if (ov) {{ ov.style.opacity = '0'; ov.style.transition = 'opacity 0.4s'; setTimeout(() => ov.remove(), 400); }}
}}

// ── Initialisation async ──────────────────────────────────────────────────
initUI();

(async function init() {{
  try {{
  // Phase 1 : Plotly newPlot (figData inline, 8 KB)
  setLoadingProgress(5, 'Initialisation du graphique…');
  await Plotly.newPlot(chartDiv, figData.data, figData.layout, {{
    responsive: true, displayModeBar: true, scrollZoom: true,
    modeBarButtonsToRemove: ['select2d','lasso2d','autoScale2d'],
    doubleClick: 'reset',
  }});

  // Enregistrer le click handler
  chartDiv.on('plotly_click', function(eventData) {{
    if (!eventData || !eventData.points || eventData.points.length === 0) return;
    const pt = eventData.points[0];
    if (pt.curveNumber !== 2) return;
    const globalIdx = pt.customdata;
    if (globalIdx !== undefined && globalIdx !== null) {{
      showDesktopCardElec(globalIdx);
    }}
  }});

  setLoadingProgress(10, 'Chargement des données…', 'static.json + élection par défaut');

  // Phase 2a : static.json + élection par défaut en parallèle
  const [staticData, defaultElecData] = await Promise.all([
    fetch('data/static.json').then(r => r.json()),
    fetch('data/elec_' + DEFAULT_ELECTION_ID + '.json').then(r => r.json()),
  ]);

  IRIS_INFO = staticData.IRIS_INFO;
  IRIS_POPS = staticData.IRIS_POPS;
  MARKER_SIZES = staticData.MARKER_SIZES;
  GROUP_INDICES = staticData.GROUP_INDICES;
  elecCache[DEFAULT_ELECTION_ID] = defaultElecData;
  currentGroupIndices = Object.assign({{}}, GROUP_INDICES);

  setLoadingProgress(40, 'Données électorales chargées — chargement des axes…', 'iris_x_desktop.json + iris_y_desktop.json');

  // Appliquer l'élection par défaut (barycentres + carte IRIS cliquables)
  await applyElection(DEFAULT_ELECTION_ID);
  setCorners(PRESETS[0].corners);
  updateDesc(PRESETS[0], PRESETS[0].xVar, PRESETS[0].yVar);

  // Phase 2b : IRIS_X, IRIS_Y et ML_FLAGS en parallèle (les plus gros fichiers)
  const [xData, yData, mlFlagsData] = await Promise.all([
    fetch('data/iris_x_desktop.json').then(r => r.json()),
    fetch('data/iris_y_desktop.json').then(r => r.json()),
    fetch('data/ml_flags.json').then(r => r.json()).catch(() => null),
  ]);

  IRIS_X = xData;
  IRIS_Y = yData;
  ML_FLAGS = mlFlagsData;
  // Afficher/masquer le bouton ML selon la disponibilité des flags
  const mlBtn = document.getElementById('mlToggleBtn');
  if (mlBtn) mlBtn.style.display = ML_FLAGS && Object.keys(ML_FLAGS).length > 0 ? '' : 'none';

  // Construire les selects maintenant que IRIS_X est disponible
  buildSelect('xSelect', PRESETS[0].xVar);
  buildSelect('ySelect', PRESETS[0].yVar);
  buildColorVarSelect();
  updateColorLegend();
  document.getElementById('xSelect').disabled = false;
  document.getElementById('ySelect').disabled = false;
  document.getElementById('xInvertChk').checked = PRESETS[0].xInvert || false;

  // Masquer les presets tsne/umap si données absentes
  document.querySelectorAll('.preset-btn').forEach(btn => {{
    const pid = btn.dataset.id;
    if ((pid === 'tsne' && !IRIS_X['tsne_x']) || (pid === 'umap' && !IRIS_X['umap_x'])) {{
      btn.style.display = 'none';
    }}
  }});

  setLoadingProgress(90, 'Finalisation…');
  applyAxes(PRESETS[0].xVar, PRESETS[0].xInvert, PRESETS[0].yVar, PRESETS[0]);

  setLoadingProgress(100);
  hideLoadingOverlay();

  // Ouvrir la carte par défaut
  currentPresetId = 'carte';
  document.querySelectorAll('.preset-btn').forEach(b => b.classList.remove('active'));
  const carteBtnEl = document.querySelector('.preset-btn[data-id="carte"]');
  if (carteBtnEl) carteBtnEl.classList.add('active');
  await showCarte();
  }} catch(e) {{
    const msgEl = document.getElementById('loadingMsg');
    const detailEl = document.getElementById('loadingDetail');
    if (msgEl) msgEl.textContent = 'Erreur : ' + e.message;
    if (detailEl) detailEl.textContent = e.stack || '';
    console.error('Init error:', e);
  }}
}})();

</script>
</body>
</html>"""
    return html


desktop_html = build_desktop_html(skip_build=_SKIP_BUILD)
with open("saint_graphique_iris.html", "w", encoding="utf-8") as f:
    f.write(desktop_html)
print(f"Desktop → saint_graphique_iris.html ({len(desktop_html)//1024} KB)")
