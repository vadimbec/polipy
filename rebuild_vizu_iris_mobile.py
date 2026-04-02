# rebuild_vizu_iris_mobile.py — Générateur HTML mobile
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
_CD_PARTY_SCORES = [f'score_{g}' for g in ALL_ORDER]



def _build_trace_data_single():
    """Mobile: 3 traces — trace 0 = densité pop. (empty), trace 1 = barycentres (empty), trace 2 = all IRIS (empty)."""
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


# ── VOTE PARTIES JS CONFIG ────────────────────────────────────────────────────
VOTE_PARTIES_JS = [
    {'key': f'score_{g}', 'label': SHORT.get(g, g), 'color': ALL_PARTIES_COLORS.get(g, '#9CA3AF')}
    for g in ALL_ORDER
]





# ── 12. BUILD MOBILE HTML ─────────────────────────────────────────────────────
def build_mobile_html(skip_build=False):
    import math, os as _os

    traces = _build_trace_data_single()

    if not skip_build:
        def _is_nan(v):
            return v is None or (isinstance(v, float) and math.isnan(v))

        # Données socio (IRIS_X / IRIS_Y) → fichiers JSON externes (précision réduite mobile)
        # Les colonnes _strict utilisent _round3 ; NaN → null
        _strict_col_set = set(_strict_cols)
        iris_x_js = {}
        iris_y_js = {}
        for v in var_data_x:
            fn = _round3 if (v in _COMPOSITE_VARS or v in _strict_col_set) else _round2
            iris_x_js[v] = fn(var_data_x[v])
            iris_y_js[v] = fn(var_data_y[v])

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

        # static.json (partagé avec desktop — ne réécrire que si absent ou si desktop a déjà écrit)
        _static_path = 'data/static.json'
        if not _os.path.exists(_static_path):
            _static = {
                'IRIS_INFO': all_customdata,
                'IRIS_POPS': iris_pops,
                'MARKER_SIZES': marker_sizes_list,
                'GROUP_INDICES': group_indices,
            }
            with open(_static_path, 'w', encoding='utf-8') as _f:
                _json.dump(_static, _f, ensure_ascii=False, separators=(',', ':'))
            print(f"  data/static.json : {_os.path.getsize(_static_path)//1024} KB")
        else:
            print(f"  data/static.json : déjà présent ({_os.path.getsize(_static_path)//1024} KB)")

        _geo_path = 'data/geo.json'
        if not _os.path.exists(_geo_path):
            _lats, _lons = _build_geo_centroids(df)
            with open(_geo_path, 'w', encoding='utf-8') as _f:
                _json.dump({'lat': _lats, 'lon': _lons}, _f, separators=(',', ':'))
            print(f"  data/geo.json : {_os.path.getsize(_geo_path)//1024} KB")
        else:
            print(f"  data/geo.json : déjà présent ({_os.path.getsize(_geo_path)//1024} KB)")

        # Fichiers élection (partagés avec desktop)
        for eid_s, elec_obj in iris_elec.items():
            _path = f'data/elec_{eid_s}.json'
            if not _os.path.exists(_path):
                with open(_path, 'w', encoding='utf-8') as _f:
                    _json.dump(elec_obj, _f, ensure_ascii=False, separators=(',', ':'))
                print(f"  data/elec_{eid_s}.json : {_os.path.getsize(_path)//1024} KB")

        # iris_x_mobile.json et iris_y_mobile.json (précision réduite)
        _path = 'data/iris_x_mobile.json'
        with open(_path, 'w', encoding='utf-8') as _f:
            _json.dump(iris_x_js, _f, separators=(',', ':'))
        print(f"  data/iris_x_mobile.json : {_os.path.getsize(_path)//1024} KB")
        _path = 'data/iris_y_mobile.json'
        with open(_path, 'w', encoding='utf-8') as _f:
            _json.dump(iris_y_js, _f, separators=(',', ':'))
        print(f"  data/iris_y_mobile.json : {_os.path.getsize(_path)//1024} KB")

        # ml_flags.json (partagé avec desktop — écrire si absent)
        _ml_path = 'data/ml_flags.json'
        if not _os.path.exists(_ml_path):
            ml_flags = {}
            for col in df.columns:
                if col.startswith('ml_imputed_'):
                    varname = col[len('ml_imputed_'):]
                    ml_flags[varname] = df[col].astype(int).tolist()
            with open(_ml_path, 'w', encoding='utf-8') as _f:
                _json.dump(ml_flags, _f, separators=(',', ':'))
            print(f"  data/ml_flags.json : {_os.path.getsize(_ml_path)//1024} KB")
        else:
            print(f"  data/ml_flags.json : déjà présent")

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
        with open('data/static.json', encoding='utf-8') as _f:
            _static_cached = _json.load(_f)
        _group_indices_cached = _static_cached['GROUP_INDICES']
        buttons_data = [{'key': g, 'short': SHORT.get(g, g), 'label': LABELS.get(g, g),
                         'color': ALL_PARTIES_COLORS.get(g, '#9CA3AF'),
                         'count': len(_group_indices_cached.get(g, []))} for g in ALL_ORDER]
        n_iris = len(_static_cached['IRIS_POPS'])
    else:
        buttons_data = [{'key': g, 'short': SHORT.get(g, g), 'label': LABELS.get(g, g),
                         'color': ALL_PARTIES_COLORS.get(g, '#9CA3AF'),
                         'count': int((df['parti_dominant'] == g).sum())} for g in ALL_ORDER]
        n_iris = len(df)
    btns_str = _json.dumps(buttons_data, ensure_ascii=False, separators=(',', ':'))
    order_str = _json.dumps(ALL_ORDER, ensure_ascii=False, separators=(',', ':'))

    fig = go.Figure()
    fig.add_vline(x=0, line_dash="dot", line_color="#CCC", line_width=1)
    fig.add_hline(y=0, line_dash="dot", line_color="#CCC", line_width=1)
    for tr in traces:
        fig.add_trace(tr)
    fig.update_layout(
        paper_bgcolor="#FAF9F7", plot_bgcolor="#FEFDFB",
        margin=dict(t=8, b=8, l=52, r=10),
        dragmode=False,
        hovermode="closest",
        xaxis=dict(
            range=AXIS_PRESETS[0]['xRange'],
            showgrid=True, gridcolor="#EBEBEB", gridwidth=0.5, zeroline=False,
            tickfont=dict(size=8, color="#AAA"), linecolor="#DDD", fixedrange=False,
            title=dict(text=AXIS_PRESETS[0]['xTitle'],
                       font=dict(size=8.5, color="#888", family="Helvetica Neue, sans-serif"), standoff=4)),
        yaxis=dict(
            range=AXIS_PRESETS[0]['yRange'],
            showgrid=True, gridcolor="#EBEBEB", gridwidth=0.5, zeroline=False,
            tickfont=dict(size=8, color="#AAA"), linecolor="#DDD", fixedrange=False,
            title=dict(text=AXIS_PRESETS[0]['yTitle'],
                       font=dict(size=8.5, color="#888", family="Helvetica Neue, sans-serif"), standoff=2)),
        showlegend=False,
    )
    fig_json = fig.to_json()

    sg = AXIS_PRESETS[0]

    html = f"""<!DOCTYPE html>
<html lang="fr">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no">
<title>Sociologie des IRIS — Mobile</title>
<style>
* {{ margin: 0; padding: 0; box-sizing: border-box; }}
html, body {{ background: #FAF9F7; font-family: 'Helvetica Neue', system-ui, sans-serif;
               color: #1a1a1a; overflow-x: hidden; -webkit-text-size-adjust: 100%; }}
.header {{ text-align: center; padding: 12px 12px 4px; }}
.header h1 {{ font-size: 17px; font-weight: 900; letter-spacing: -0.5px; margin-bottom: 2px; }}
.header p {{ font-size: 9px; color: #888; line-height: 1.4; }}

.axis-bar {{ padding: 6px 10px; background: #fff; border-bottom: 1px solid #EEE; }}
.axis-bar-label {{ font-size: 10px; font-weight: 700; color: #666; margin-bottom: 4px; }}
.preset-btns {{ display: flex; flex-wrap: wrap; gap: 4px; margin-bottom: 4px; }}
.preset-btn {{ padding: 4px 8px; border-radius: 14px; border: 1.5px solid #D0D0D0;
               background: transparent; font-size: 10px; font-weight: 600; color: #555;
               font-family: inherit; cursor: pointer; -webkit-tap-highlight-color: transparent;
               white-space: nowrap; }}
.preset-btn.active {{ background: #1a1a1a; border-color: #1a1a1a; color: #fff; }}
.custom-toggle {{ padding: 4px 10px; border-radius: 14px; border: 1.5px dashed #C0C0C0;
                  background: transparent; font-size: 10px; font-weight: 600; color: #888;
                  font-family: inherit; cursor: pointer; -webkit-tap-highlight-color: transparent; }}
.ml-toggle-btn {{ padding: 4px 10px; border-radius: 14px; border: 1.5px solid #D97706;
                  background: transparent; font-size: 10px; font-weight: 600; color: #D97706;
                  font-family: inherit; cursor: pointer; -webkit-tap-highlight-color: transparent; margin-top: 4px; }}
.ml-toggle-btn.full-mode {{ background: #FEF3C7; color: #92400E; }}
.ml-flag {{ font-size: 9px; color: #D97706; font-style: italic; font-weight: 400; }}
.custom-panel {{ display: none; padding: 6px 0 2px; gap: 8px; flex-direction: column; }}
.custom-panel.open {{ display: flex; }}
.custom-panel select {{ padding: 4px 6px; border-radius: 6px; border: 1px solid #D0D0D0;
                        background: #fff; font-size: 11px; font-family: inherit; width: 100%; }}
.custom-panel label {{ font-size: 10px; font-weight: 600; color: #555; }}

.filters {{ display: flex; flex-wrap: wrap; gap: 4px; padding: 4px 10px; justify-content: center; }}
.toggle-all {{ display: block; width: calc(100% - 20px); margin: 2px 10px 4px;
               padding: 3px 0; border-radius: 8px; border: 1.5px solid #CCC;
               background: transparent; font-size: 9px; font-weight: 700; color: #888;
               font-family: inherit; cursor: pointer; -webkit-tap-highlight-color: transparent; text-align: center; }}
.fbtn {{ display: inline-flex; align-items: center; gap: 3px; padding: 2px 8px; border-radius: 12px;
         border: 2px solid; font-size: 9px; font-weight: 700; font-family: inherit;
         cursor: pointer; transition: all 0.12s; -webkit-tap-highlight-color: transparent; }}
.fbtn.on {{ color: #fff; }}
.fbtn.off {{ background: transparent !important; opacity: 0.3; }}

#chartWrap {{ position: relative; width: 100%; }}
#chart {{ width: 100%; aspect-ratio: 9 / 11; touch-action: none; }}
#domMapsRow {{ display: none; gap: 4px; padding: 4px 4px 0; flex-wrap: wrap; touch-action: pan-y; }}
.dom-map-wrap {{ flex: 1; min-width: 120px; position: relative; padding: 10px 0; touch-action: pan-y; }}
.dom-map-label {{ position: absolute; top: 3px; left: 5px; z-index: 10; font-size: 9px;
                  font-weight: 800; color: #333; background: rgba(255,255,255,0.82);
                  padding: 1px 5px; border-radius: 6px; pointer-events: none; }}
.dom-map-canvas {{ width: 100%; height: 150px; border-radius: 6px; overflow: hidden;
                   border: 1px solid #E8E8E8; }}
.corner-label {{ position: absolute; font-size: 8px; font-weight: 800; line-height: 1.2;
                 pointer-events: none; opacity: 0.65; }}
.corner-tl {{ top: 8px; left: 56px; text-align: left; }}
.corner-tr {{ top: 8px; right: 8px; text-align: right; }}
.corner-bl {{ bottom: 40px; left: 56px; text-align: left; }}
.corner-br {{ bottom: 40px; right: 8px; text-align: right; }}

#resetBtn {{ position: absolute; top: 6px; right: 6px; z-index: 50;
             background: rgba(255,255,255,0.92); border: 1px solid #DDD;
             border-radius: 6px; padding: 3px 8px; font-size: 9px; font-weight: 700;
             color: #888; font-family: inherit; cursor: pointer; display: none;
             -webkit-tap-highlight-color: transparent; box-shadow: 0 1px 4px rgba(0,0,0,0.08); }}
#resetBtn.show {{ display: block; }}

.info-card {{ position: fixed; bottom: 0; left: 0; right: 0;
              background: rgba(255,255,255,0.98); border-top: 1px solid #E0E0E0;
              padding: 12px 16px calc(env(safe-area-inset-bottom, 8px) + 12px);
              font-size: 11px; line-height: 1.55; box-shadow: 0 -4px 20px rgba(0,0,0,0.08);
              transform: translateY(100%); transition: transform 0.25s ease; z-index: 100;
              backdrop-filter: blur(8px); -webkit-backdrop-filter: blur(8px);
              max-height: 65vh; overflow-y: auto; }}
.info-card.show {{ transform: translateY(0); }}
.info-card .name {{ font-size: 14px; font-weight: 900; }}
.info-card .party {{ font-weight: 700; font-size: 10px; margin-bottom: 4px; }}
.info-card .row {{ color: #555; margin-bottom: 1px; }}
.info-card .row b {{ color: #1a1a1a; }}
.info-card .lbl {{ color: #999; }}
.info-card .close {{ position: absolute; top: 8px; right: 14px; font-size: 22px; color: #BBB;
                     cursor: pointer; -webkit-tap-highlight-color: transparent; padding: 4px 8px; }}
.info-card .dynamic-row {{ color: #555; border-top: 1px solid #F0F0F0; padding-top: 4px; margin-top: 4px; }}
.info-card .section-title {{ font-size: 9px; font-weight: 800; text-transform: uppercase;
                              letter-spacing: 0.5px; color: #BBB; margin: 6px 0 3px; }}
.info-card .vote-grid {{ display: flex; flex-direction: column; gap: 3px; margin-top: 4px; font-size: 10px; }}
.info-card .vote-cell {{ display: flex; align-items: center; gap: 4px; }}
.info-card .vote-parti {{ min-width: 52px; font-weight: 700; font-size: 9.5px; flex-shrink: 0; }}
.info-card .vote-bar-bg {{ flex: 1; background: #F0F0F0; border-radius: 2px; height: 5px; overflow: hidden; }}
.info-card .vote-bar-fill {{ height: 100%; border-radius: 2px; }}
.info-card .vote-score {{ min-width: 34px; text-align: right; color: #444; font-size: 9.5px; }}
.info-card .stat-row {{ display: flex; align-items: center; gap: 4px; margin-bottom: 3px; font-size: 10px; }}
.info-card .stat-lbl {{ min-width: 70px; color: #999; font-size: 9px; flex-shrink: 0; }}
.info-card .stat-bar-bg {{ flex: 1; background: #F0F0F0; border-radius: 2px; height: 4px; overflow: hidden; }}
.info-card .stat-bar-fill {{ height: 100%; border-radius: 2px; }}
.info-card .stat-pct {{ min-width: 30px; text-align: right; color: #444; font-size: 9.5px; }}
.footer {{ text-align: center; padding: 6px 12px 4px; font-size: 8px; color: #AAA; line-height: 1.6; }}

.axis-desc {{ padding: 12px 14px 20px; font-size: 11px; color: #555; line-height: 1.6;
              border-top: 1px solid #EEE; background: #FDFCFA; }}
.axis-desc:empty {{ display: none; }}
.axis-desc .desc-title {{ font-size: 13px; font-weight: 900; color: #1a1a1a; margin-bottom: 8px; }}
.axis-desc .desc-ax {{ background: #fff; border: 1px solid #E8E8E8; border-radius: 8px;
                        padding: 8px 10px; margin-bottom: 8px; }}
.axis-desc .desc-quadrants {{ background: #fff; border: 1px solid #E8E8E8; border-radius: 8px;
                               padding: 8px 10px; }}
.axis-desc .desc-q-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 6px; margin-top: 6px; }}
.axis-desc .desc-q {{ font-size: 10px; padding: 5px 7px; border-radius: 6px; background: #F8F8F8; }}
.axis-desc .desc-q-label {{ font-size: 8px; font-weight: 800; text-transform: uppercase;
                              letter-spacing: 0.4px; opacity: 0.6; margin-bottom: 2px; }}

.election-bar {{ display: flex; align-items: center; gap: 6px; padding: 5px 10px;
                 background: #F8F7F5; border-bottom: 1px solid #E8E8E8; flex-wrap: wrap; }}
.election-bar-label {{ font-size: 9px; font-weight: 700; color: #888; white-space: nowrap; }}
.election-type-btns {{ display: flex; gap: 3px; flex-wrap: wrap; }}
.elec-type-btn {{ padding: 2px 7px; border-radius: 12px; border: 1.5px solid #D0D0D0;
                  background: transparent; font-size: 9px; font-weight: 600; color: #777;
                  font-family: inherit; cursor: pointer; -webkit-tap-highlight-color: transparent; white-space: nowrap; }}
.elec-type-btn.active {{ background: #3B3B3B; border-color: #3B3B3B; color: #fff; }}
.elec-year-btns {{ display: flex; gap: 3px; flex-wrap: wrap; }}
.elec-year-btn {{ padding: 2px 7px; border-radius: 12px; border: 1.5px solid #D0D0D0;
                  background: transparent; font-size: 9px; font-weight: 600; color: #777;
                  font-family: inherit; cursor: pointer; -webkit-tap-highlight-color: transparent; white-space: nowrap; }}
.elec-year-btn.active {{ background: #555; border-color: #555; color: #fff; }}
.tour-btns {{ display: flex; gap: 3px; }}
.tour-btn {{ padding: 2px 7px; border-radius: 12px; border: 1.5px solid #D0D0D0;
             background: transparent; font-size: 9px; font-weight: 700; color: #777;
             font-family: inherit; cursor: pointer; -webkit-tap-highlight-color: transparent; }}
.tour-btn.active {{ background: #1a1a1a; border-color: #1a1a1a; color: #fff; }}
.tour-btn:disabled {{ opacity: 0.3; cursor: default; }}
.election-current-label {{ font-size: 8px; color: #AAA; font-style: italic; }}

#abst-stats-panel {{ padding: 5px 10px; font-size: 9px; color: #555;
                     border-top: 1px solid #E8E8E8; display: none; line-height: 1.6; word-break: break-word; }}
</style>
</head>
<body>

<div class="header">
  <h1>Sociologie des IRIS</h1>
  <p>Sociologie des {n_iris} zones IRIS · données INSEE 2021</p>
</div>

<div class="axis-bar">
  <div class="axis-bar-label">Axes :</div>
  <div class="preset-btns" id="presetBtns"></div>
  <button class="custom-toggle" id="customToggle">Personnaliser ▾</button>
  <button class="ml-toggle-btn" id="mlToggleBtn" style="display:none" title="Basculer mode strict/full ML">⚠ Imputés ML : masqués</button>
  <button class="ml-toggle-btn" id="densityToggleBtn" style="border-color:#9CA3AF;color:#9CA3AF;">Densité : affichée</button>
  <div class="custom-panel" id="customPanel">
    <div>
      <label>Axe X :</label>
      <select id="xSelect"></select>
    </div>
    <div>
      <label>Axe Y :</label>
      <select id="ySelect"></select>
    </div>
    <label style="font-size:10px; display:flex; align-items:center; gap:4px;">
      <input type="checkbox" id="xInvertChk"> Inverser X
    </label>
  </div>
</div>

<div class="election-bar" id="electionBar">
  <span class="election-bar-label">Couleur :</span>
  <div class="election-type-btns" id="elecTypeBtns"></div>
  <div class="elec-year-btns" id="elecYearBtns"></div>
  <div class="tour-btns">
    <button class="tour-btn" id="tourBtn1">T1</button>
    <button class="tour-btn" id="tourBtn2">T2</button>
  </div>
  <span class="election-current-label" id="electionCurrentLabel"></span>
  <span id="elecSpinner" style="display:none;font-size:10px;color:#888;margin-left:4px">…</span>
  <select id="colorVarSelect" disabled style="margin-left:6px;padding:2px 6px;border-radius:8px;border:1px solid #D0D0D0;font-size:9px;font-family:inherit;color:#555;background:#fff;cursor:pointer;max-width:160px"><option value="">Couleur : élection</option></select>
</div>

<button class="toggle-all" id="toggleAll">Tout décocher</button>
<div class="filters" id="filters"></div>

<div id="chartWrap">
  <div id="chart"></div>
  <div id="mapDiv" style="width:100%;height:calc(100vh - 180px);min-height:300px;display:none;position:relative;">
    <div id="carteLoadingMsg" style="display:none;position:absolute;top:8px;left:50%;transform:translateX(-50%);background:rgba(255,255,255,0.92);padding:6px 16px;border-radius:20px;font-size:12px;color:#555;z-index:10;box-shadow:0 1px 4px rgba(0,0,0,0.1)">Chargement des coordonnées géographiques…</div>
    <button id="mapResetBtn" onclick="mapInstance && mapInstance.fitBounds([[-5.2,41.3],[9.6,51.2]],{{padding:10}})" style="position:absolute;bottom:16px;right:8px;z-index:20;background:rgba(255,255,255,0.95);border:1px solid #ccc;border-radius:6px;padding:6px 10px;font-size:13px;cursor:pointer;box-shadow:0 1px 4px rgba(0,0,0,0.15)">↺ Recentrer</button>
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
  <button id="resetBtn">↺ Zoom</button>
</div>

<div class="info-card" id="infoCard">
  <span class="close" id="closeCard">×</span>
  <div class="name" id="cardName"></div>
  <div class="party" id="cardParty"></div>
  <div class="row"><span class="lbl">Population :</span> <b id="cardPop"></b> &nbsp;·&nbsp; <span class="lbl">Âge moyen :</span> <b id="cardAge"></b></div>
  <div class="row"><span class="lbl">Revenu médian :</span> <b id="cardRev"></b></div>
  <div class="section-title">CSP</div>
  <div id="cardCSP"></div>
  <div class="section-title">Formation &amp; Emploi</div>
  <div id="cardFormation"></div>
  <div class="section-title">Logement</div>
  <div id="cardHousing"></div>
  <div class="section-title" id="cardElecLabel"></div>
  <div class="row" id="cardAbstElecRow" style="display:none"><span class="lbl">Abst. élec. :</span> <b id="cardAbstElec"></b></div>
  <div class="vote-grid" id="cardVotes"></div>
  <div class="dynamic-row"><span class="lbl">Axe X (<span id="cardXVar"></span>) :</span> <b id="cardXVal"></b> &nbsp;·&nbsp; <span class="lbl">Axe Y (<span id="cardYVar"></span>) :</span> <b id="cardYVal"></b></div>
</div>

<div class="footer">⊕ = barycentre · taille = population IRIS · couleur = parti dominant · N={n_iris}</div>
<div id="colorLegend" style="display:none;flex-direction:column;padding:4px 10px;background:#fff;border-top:1px solid #E8E8E8"></div>
<div id="abst-stats-panel"></div>
<div class="axis-desc" id="axisDesc"></div>

<div id="loadingOverlay" style="position:fixed;inset:0;background:rgba(250,249,247,0.96);display:flex;flex-direction:column;align-items:center;justify-content:center;z-index:9999;font-family:'Helvetica Neue',system-ui,sans-serif">
  <div style="font-size:13px;color:#555;margin-bottom:12px" id="loadingMsg">Initialisation…</div>
  <div style="width:240px;height:4px;background:#EEE;border-radius:2px">
    <div id="loadingBar" style="height:4px;background:#F97316;border-radius:2px;width:0%;transition:width 0.4s ease"></div>
  </div>
  <div style="font-size:10px;color:#AAA;margin-top:8px" id="loadingDetail"></div>
</div>

<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<link href='https://unpkg.com/maplibre-gl@4/dist/maplibre-gl.css' rel='stylesheet'/>
<script src='https://unpkg.com/maplibre-gl@4/dist/maplibre-gl.js'></script>
<script>
window.onerror = function(msg, src, line, col, err) {{
  var el = document.getElementById('loadingMsg');
  if (el) el.textContent = 'JS Error: ' + msg + ' (line ' + line + ')';
  var el2 = document.getElementById('loadingDetail');
  if (el2) el2.textContent = src + ':' + line;
}};
const figData = {fig_json};

// ── Métadonnées inline (petites, <500 KB) ────────────────────────────────
const ELECTIONS_META = {elections_meta_str};
const DEFAULT_ELECTION_ID = {default_elec_str};
const PRESETS    = {presets_str};
const VARS       = {vars_str};
const varLabels  = {var_labels_str};
const COULEURS_JS = {couleurs_str};
const VOTE_PARTIES = {vote_parties_str};
const ALL_PARTIES_COLORS_JS = {all_parties_colors_str};
const btns       = {btns_str};
const ORDER = {order_str};

// ── Données globales (chargées en async depuis data/) ─────────────────────
let IRIS_X = null, IRIS_Y = null;
let IRIS_LAT = null, IRIS_LON = null;
let mapInstance = null;
let mapReady = false;
let mapInitialized = false;
let isCarteActive = false;
let IRIS_INFO = null, IRIS_POPS = null, MARKER_SIZES = null, GROUP_INDICES = null;
let ML_FLAGS = null;      // chargé depuis data/ml_flags.json (tableaux 0/1 par var DISP_*)
let mlStrictMode = true;  // true = masquer les IRIS imputés ML (mode par défaut)
let densityVisible = true;
let currentXVarData = 'score_exploitation';
let currentYVarData = 'score_domination';
const elecCache = {{}};  // cache élections déjà fetché
const domMaps = [];
const domMapsReady = [];
const DOM_TOM_CONFIGS = [
  {{ id: 'domMap0', center: [-61.55, 16.25], zoom: 8.5, bounds: [[-62.0,15.83],[-61.0,16.56]] }},
  {{ id: 'domMap1', center: [-61.0,  14.65], zoom: 9.0, bounds: [[-61.3,14.37],[-60.75,14.90]] }},
  {{ id: 'domMap2', center: [-53.1,   4.0],  zoom: 5.5, bounds: [[-54.6,2.1],[-51.5,5.8]] }},
  {{ id: 'domMap3', center: [55.55, -21.1],  zoom: 8.5, bounds: [[55.21,-21.4],[55.84,-20.87]] }},
];

let currentXVar = 'score_exploitation';
let currentYVar = 'score_domination';
let currentXInvert = false;
let currentPresetId = 'carte';
let currentColorVar = null;   // null = couleur par élection, sinon nom de variable
let colorVarMin = 0, colorVarMax = 1;  // percentile 2–98 de la variable
let currentXRange = (PRESETS[0].xRange || [-10, 10]).slice();
let currentYRange = (PRESETS[0].yRange || [-10, 10]).slice();
let currentCorners = PRESETS[0].corners;

let activeGroups;
let currentElectionId;
let currentElectionType, currentElectionYear, currentElectionTour;
let currentClickedGlobalIdx = null;
let currentGroupIndices = {{}};
let elecByType = {{}};
let baryMeans = {{}};
let currentBarySizeMap = {{}};
const typeLabels = {{legi:'Législatives', euro:'Européennes', pres:'Présidentielles', muni:'Municipales'}};

const chartDiv = document.getElementById('chart');

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

// ── Auto-range helper ─────────────────────────────────────────────────────
function computeDataRange_embeddings(varName, invert) {{
  if (!IRIS_X || !IRIS_Y) return [-1, 1];
  const arr = IRIS_X[varName] || IRIS_Y[varName];
  if (!arr) return [-1, 1];
  let mn = Infinity, mx = -Infinity;
  for (const v of arr) {{
    if (v === null || v === undefined || isNaN(v)) continue;
    const val = invert ? -v : v;
    if (val < mn) mn = val;
    if (val > mx) mx = val;
  }}
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
  // Diverging: blue (low) → white (mid) → red (high)
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
      opt.textContent = (varLabels[v] || v).substring(0, 55);
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
  const label = (varLabels[currentColorVar] || currentColorVar).substring(0, 60);
  const mn = colorVarMin.toFixed(2), mx = colorVarMax.toFixed(2);
  leg.style.display = 'flex';
  leg.innerHTML = `
    <div style="font-size:9px;color:#555;margin-bottom:2px;font-weight:600">${{label}}</div>
    <div style="display:flex;align-items:center;gap:6px">
      <span style="font-size:9px;color:#3B82F6">${{mn}}</span>
      <div style="flex:1;height:8px;border-radius:4px;background:linear-gradient(to right,#3B82F6,#fff,#EF4444)"></div>
      <span style="font-size:9px;color:#EF4444">${{mx}}</span>
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
      <div class="desc-ax">${{d.x}}</div>
      <div class="desc-ax" style="margin-top:6px">${{d.y}}</div>
      ${{qHtml ? `<div class="desc-quadrants" style="margin-top:8px"><b style="font-size:9px;color:#888;text-transform:uppercase;letter-spacing:.4px">Quadrants</b><div class="desc-q-grid">${{qHtml}}</div></div>` : ''}}`;
  }} else {{
    const xl = varLabels[xVar] || xVar;
    const yl = varLabels[yVar] || yVar;
    descDiv.innerHTML = `
      <div class="desc-ax"><b>Axe X — ${{xVar}}</b><br>${{xl}}</div>
      <div class="desc-ax" style="margin-top:6px"><b>Axe Y — ${{yVar}}</b><br>${{yl}}</div>`;
  }}
}}

// ── Shared helpers (copied from desktop) ──────────────────────────────────
function fmtPct(v) {{ return (v !== '' && v !== null && !isNaN(Number(v))) ? Number(v).toFixed(1) + '%' : '—'; }}
function fmtNum(v, suffix) {{ return (v !== '' && v !== null && !isNaN(Number(v))) ? Math.round(Number(v)).toLocaleString('fr-FR') + (suffix||'') : '—'; }}

// ── Info card ──────────────────────────────────────────────────────────────
function showCard(irisGlobalIdx) {{
  const cd = IRIS_INFO[irisGlobalIdx];
  if (!cd) return;
  currentClickedGlobalIdx = irisGlobalIdx;
  const card = document.getElementById('infoCard');

  const xRaw = IRIS_X ? (IRIS_X[currentXVar] || [])[irisGlobalIdx] : undefined;
  const yRaw = IRIS_Y ? (IRIS_Y[currentYVar] || [])[irisGlobalIdx] : undefined;
  const xDisp = xRaw !== undefined ? (currentXInvert ? -xRaw : xRaw) : undefined;

  const elecData = elecCache[currentElectionId];
  const t1Fb = getT1FallbackData(currentElectionId);
  const hasT2 = elecData && elecData.partis[irisGlobalIdx] !== null;
  const activeData = hasT2 ? elecData : (t1Fb || elecData);
  const isT1Fallback = !hasT2 && !!t1Fb;

  const elecScores = activeData ? activeData.scores[irisGlobalIdx] : null;
  const currentMeta = ELECTIONS_META[currentElectionId];
  const currentColor = activeData?.colors[irisGlobalIdx] || '#9CA3AF';
  const currentParti = activeData?.partis[irisGlobalIdx] || '—';

  const abstElec = activeData ? activeData.abst[irisGlobalIdx] : null;

  let voteGridHtml = '';
  if (elecScores && Object.keys(elecScores).length > 0) {{
    const allScores = Object.entries(elecScores).filter(([,v]) => v > 0).sort((a,b) => b[1]-a[1]);
    const maxScore = allScores[0]?.[1] || 1;
    voteGridHtml = allScores.map(([p, score]) => {{
      const color = ALL_PARTIES_COLORS_JS[p] || '#9CA3AF';
      const barW = Math.round(score / maxScore * 100);
      return `<div class="vote-cell">` +
        `<span class="vote-parti" style="color:${{color}}">${{p.replace('_',' ')}}</span>` +
        `<div class="vote-bar-bg"><div class="vote-bar-fill" style="width:${{barW}}%;background:${{color}}"></div></div>` +
        `<span class="vote-score">${{score.toFixed(1)}}%</span>` +
        `</div>`;
    }}).join('');
    if (isT1Fallback) voteGridHtml = `<div style="color:#AAA;font-size:9px;margin-bottom:4px;font-style:italic">(remporté au T1)</div>` + voteGridHtml;
  }}

  const irisCode = cd[0] || '';
  const communeName = cd[1] || cd[0] || 'IRIS inconnu';
  document.getElementById('cardName').innerHTML = communeName + (irisCode ? ` <span style="color:#AAA;font-weight:400;font-size:10px">${{irisCode}}</span>` : '');
  const partyEl = document.getElementById('cardParty');
  partyEl.textContent = cd[19] || '';
  partyEl.style.color = '#666';
  document.getElementById('cardPop').textContent = fmtNum(cd[2], ' hab.');
  document.getElementById('cardAge').textContent = cd[18] !== '' && cd[18] != null ? Number(cd[18]).toFixed(1) + ' ans' : '—';
  document.getElementById('cardRev').innerHTML = fmtNum(cd[3], ' €/UC') +
    (isMLImputed(irisGlobalIdx, 'DISP_MED21') ? ' <span style="font-size:10px;color:#D97706;font-style:italic;font-weight:400">(imputé ML)</span>' : '');
  document.getElementById('cardCSP').innerHTML =
    `<div class="stat-row"><span class="stat-lbl">Cadres sup.</span><div class="stat-bar-bg"><div class="stat-bar-fill" style="width:${{Math.min(100,cd[4]||0)}}%;background:#5B8DB8"></div></div><span class="stat-pct">${{fmtPct(cd[4])}}</span></div>` +
    `<div class="stat-row"><span class="stat-lbl">Prof. interm.</span><div class="stat-bar-bg"><div class="stat-bar-fill" style="width:${{Math.min(100,cd[6]||0)}}%;background:#82AAC8"></div></div><span class="stat-pct">${{fmtPct(cd[6])}}</span></div>` +
    `<div class="stat-row"><span class="stat-lbl">Ouvriers</span><div class="stat-bar-bg"><div class="stat-bar-fill" style="width:${{Math.min(100,cd[5]||0)}}%;background:#C97A5A"></div></div><span class="stat-pct">${{fmtPct(cd[5])}}</span></div>`;
  document.getElementById('cardFormation').innerHTML =
    `<div class="stat-row"><span class="stat-lbl">Bac+</span><div class="stat-bar-bg"><div class="stat-bar-fill" style="width:${{Math.min(100,cd[16]||0)}}%;background:#7AAD8F"></div></div><span class="stat-pct">${{fmtPct(cd[16])}}</span></div>` +
    `<div class="stat-row"><span class="stat-lbl">Sans diplôme</span><div class="stat-bar-bg"><div class="stat-bar-fill" style="width:${{Math.min(100,cd[17]||0)}}%;background:#C9A45A"></div></div><span class="stat-pct">${{fmtPct(cd[17])}}</span></div>` +
    `<div class="stat-row"><span class="stat-lbl">Chômage</span><div class="stat-bar-bg"><div class="stat-bar-fill" style="width:${{Math.min(100,cd[15]||0)}}%;background:#C95A5A"></div></div><span class="stat-pct">${{fmtPct(cd[15])}}</span></div>`;
  document.getElementById('cardHousing').innerHTML =
    `<div class="stat-row"><span class="stat-lbl">Propriétaires</span><div class="stat-bar-bg"><div class="stat-bar-fill" style="width:${{Math.min(100,cd[13]||0)}}%;background:#7AA7D0"></div></div><span class="stat-pct">${{fmtPct(cd[13])}}</span></div>` +
    `<div class="stat-row"><span class="stat-lbl">HLM</span><div class="stat-bar-bg"><div class="stat-bar-fill" style="width:${{Math.min(100,cd[14]||0)}}%;background:#E8975A"></div></div><span class="stat-pct">${{fmtPct(cd[14])}}</span></div>`;
  document.getElementById('cardElecLabel').textContent = currentMeta ? currentMeta.label : currentElectionId;
  const abstElecRow = document.getElementById('cardAbstElecRow');
  if (abstElec != null) {{
    document.getElementById('cardAbstElec').textContent = fmtPct(abstElec);
    abstElecRow.style.display = '';
  }} else {{
    abstElecRow.style.display = 'none';
  }}
  document.getElementById('cardVotes').innerHTML = voteGridHtml || '<span style="color:#AAA;font-size:10px">Données non disponibles</span>';
  document.getElementById('cardXVar').textContent = currentXVar;
  document.getElementById('cardYVar').textContent = currentYVar;
  document.getElementById('cardXVal').textContent = xDisp !== undefined ? Number(xDisp).toFixed(3) : '—';
  document.getElementById('cardYVal').textContent = yRaw !== undefined ? Number(yRaw).toFixed(3) : '—';

  card.classList.add('show');
}}

document.getElementById('closeCard').addEventListener('click', () => {{
  document.getElementById('infoCard').classList.remove('show');
  currentClickedGlobalIdx = null;
}});
(function() {{
  const card = document.getElementById('infoCard');
  let startY = 0;
  card.addEventListener('touchstart', e => {{ startY = e.touches[0].clientY; }}, {{passive: true}});
  card.addEventListener('touchend', e => {{
    const dy = e.changedTouches[0].clientY - startY;
    if (dy > 60) {{ card.classList.remove('show'); currentClickedGlobalIdx = null; }}
  }}, {{passive: true}});
}})();

function computeButtonPcts(electionId) {{
  const data = elecCache[electionId];
  if (!data || !data.buttonPcts) return;
  btns.forEach(b => {{ b.pct = data.buttonPcts[b.key] || 0; }});
}}

// ── T1 fallback helper ────────────────────────────────────────────────────
function getT1FallbackData(electionId) {{
  if (!electionId.endsWith('_t2')) return null;
  const t1Id = electionId.replace(/_t2$/, '_t1');
  return elecCache[t1Id] || null;
}}

// ── Restyle all IRIS (trace 1) ────────────────────────────────────────────
function restyleIRIS() {{
  if (!IRIS_X || !IRIS_Y) return;
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

// ── Restyle barycentres (trace 0) ─────────────────────────────────────────
function computeAbstBary(electionId) {{
  const data = elecCache[electionId];
  return (data && data.abstBary) ? data.abstBary : null;
}}

function restyleBarycentres() {{
  const baryX = [], baryY = [], baryColors = [], baryTexts = [], barySzs = [];
  const topG = Object.keys(currentGroupIndices)
    .filter(g => {{
      if (g === 'AUTRE') return false;
      const b = btns.find(b2 => b2.key === g);
      return baryMeans[g] && b && b.pct > 1;
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

// ── Abstention stats panel ─────────────────────────────────────────────────
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
    html += ' &nbsp;·&nbsp; ';
    if (hasBlancs) {{
      const pctBlancsIns = totalInscrits > 0 ? totalBlancs / totalInscrits * 100 : 0;
      html += `<span style="font-weight:600">Blancs&nbsp;:</span> ${{fmtP(pctBlancsIns)}} ins. (${{fmt(totalBlancs)}})`;
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

// ── Election switching ────────────────────────────────────────────────────
function getElectionId(type, year, tour) {{
  const years = elecByType[type];
  if (!years || !years[year]) return null;
  const found = years[year].find(e => e.tour === tour);
  return found ? found.id : null;
}}

const filtersDiv = document.getElementById('filters');
const toggleAllBtn = document.getElementById('toggleAll');
let allOn = true;

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
  const activeBtns = btns.filter(b => b.count > 0).sort((a, b2) => (b2.pct || 0) - (a.pct || 0));
  activeBtns.forEach(b => {{
    const el = document.createElement('button');
    const isOn = activeGroups.has(b.key);
    el.className = 'fbtn ' + (isOn ? 'on' : 'off');
    el.dataset.key = b.key;
    el.style.borderColor = b.color;
    el.style.backgroundColor = isOn ? b.color : 'transparent';
    el.style.color = isOn ? '#fff' : b.color;
    el.innerHTML = b.short + ' <span style="font-size:7px;font-weight:400;opacity:0.7">' + (b.pct || 0).toFixed(1) + '%</span>';
    el.addEventListener('click', (e) => {{ e.preventDefault(); setGroupVisible(b, !activeGroups.has(b.key)); updateToggleLabel(); }});
    filtersDiv.appendChild(el);
  }});
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
    if (newGroups[parti] !== undefined) {{ newGroups[parti].push(i); }}
    else if (newGroups['AUTRE'] !== undefined) {{ newGroups['AUTRE'].push(i); }}
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
  document.getElementById('electionCurrentLabel').textContent = meta ? meta.label : electionId;
  updateToggleLabel();

  if (currentClickedGlobalIdx !== null) {{
    const partisData = elecCache[currentElectionId];
    const hasT2Data = partisData && partisData.partis[currentClickedGlobalIdx] !== null;
    const t1Fb = getT1FallbackData(currentElectionId);
    const hasT1Fallback = !hasT2Data && t1Fb && t1Fb.partis[currentClickedGlobalIdx] !== null;
    if (hasT2Data || hasT1Fallback) {{
      showCard(currentClickedGlobalIdx);
    }} else {{
      document.getElementById('infoCard').classList.remove('show');
      currentClickedGlobalIdx = null;
    }}
  }}
  updateMapColors();
}}

function updateYearBtns(type) {{
  const elecYearBtns = document.getElementById('elecYearBtns');
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
  const tourBtn1 = document.getElementById('tourBtn1');
  const tourBtn2 = document.getElementById('tourBtn2');
  const years = elecByType[type];
  const available = years && years[year] ? years[year].map(e => e.tour) : [];
  tourBtn1.disabled = !available.includes(1);
  tourBtn2.disabled = !available.includes(2);
  tourBtn1.classList.toggle('active', currentElectionTour === 1 && available.includes(1));
  tourBtn2.classList.toggle('active', currentElectionTour === 2 && available.includes(2));
}}

// ── Résolution mode strict ML ─────────────────────────────────────────────
function resolveVarStrict(varName) {{
  if (!mlStrictMode || !IRIS_X) return varName;
  const sname = varName + '_strict';
  return IRIS_X[sname] !== undefined ? sname : varName;
}}

function isMLImputed(irisIdx, dispVarName) {{
  return ML_FLAGS && ML_FLAGS[dispVarName] && ML_FLAGS[dispVarName][irisIdx] === 1;
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

// ── Touch pan + pinch-zoom + tap ──────────────────────────────────────────
let gesture = null;
let touch1  = null;
let pinch   = null;
let rafId   = null;

function getRange() {{
  return {{
    x: chartDiv._fullLayout.xaxis.range.slice(),
    y: chartDiv._fullLayout.yaxis.range.slice(),
  }};
}}

function findNearest(cx, cy) {{
  const ax = chartDiv._fullLayout.xaxis;
  const ay = chartDiv._fullLayout.yaxis;
  let best = null, bestD = Infinity;
  for (let ti = 1; ti < chartDiv.data.length; ti++) {{
    const tr = chartDiv.data[ti];
    if (tr.visible === false || !tr.customdata) continue;
    for (let pi = 0; pi < tr.x.length; pi++) {{
      const sx = ax.d2p(tr.x[pi]) + ax._offset;
      const sy = ay.d2p(tr.y[pi]) + ay._offset;
      const d = Math.hypot(cx - sx, cy - sy);
      if (d < bestD && d < 30) {{ bestD = d; best = {{ti, pi}}; }}
    }}
  }}
  return best;
}}

function doRelayout(xr, yr) {{
  if (rafId) cancelAnimationFrame(rafId);
  rafId = requestAnimationFrame(() => {{
    Plotly.relayout(chartDiv, {{'xaxis.range': xr, 'yaxis.range': yr}});
    rafId = null;
  }});
}}

chartDiv.addEventListener('touchstart', function(e) {{
  e.stopPropagation();
  if (e.touches.length >= 2) {{
    gesture = 'pinch';
    const t1 = e.touches[0], t2 = e.touches[1];
    const rect = chartDiv.getBoundingClientRect();
    pinch = {{
      dist: Math.hypot(t2.clientX - t1.clientX, t2.clientY - t1.clientY),
      midPxX: (t1.clientX + t2.clientX) / 2 - rect.left,
      midPxY: (t1.clientY + t2.clientY) / 2 - rect.top,
      startRanges: getRange(),
    }};
    touch1 = null;
  }} else if (e.touches.length === 1) {{
    gesture = null;
    const t = e.touches[0];
    const rect = chartDiv.getBoundingClientRect();
    touch1 = {{
      clientX: t.clientX, clientY: t.clientY,
      time: Date.now(),
      startPxX: t.clientX - rect.left,
      startPxY: t.clientY - rect.top,
      startRanges: getRange(),
      moved: false,
    }};
  }}
}}, {{ passive: false, capture: true }});

chartDiv.addEventListener('touchmove', function(e) {{
  e.preventDefault();
  e.stopPropagation();
  if (e.touches.length >= 2 && pinch) {{
    gesture = 'pinch';
    const t1 = e.touches[0], t2 = e.touches[1];
    const rect = chartDiv.getBoundingClientRect();
    const newDist = Math.hypot(t2.clientX - t1.clientX, t2.clientY - t1.clientY);
    const newMidPxX = (t1.clientX + t2.clientX) / 2 - rect.left;
    const newMidPxY = (t1.clientY + t2.clientY) / 2 - rect.top;
    const scale = pinch.dist / newDist;
    const sr = pinch.startRanges;
    const ax = chartDiv._fullLayout.xaxis;
    const ay = chartDiv._fullLayout.yaxis;
    const xSpan0 = sr.x[1] - sr.x[0];
    const ySpan0 = sr.y[1] - sr.y[0];
    const plotW = ax._length;
    const plotH = ay._length;
    const anchorX = sr.x[0] + (pinch.midPxX - ax._offset) / plotW * xSpan0;
    const anchorY = sr.y[1] - (pinch.midPxY - ay._offset) / plotH * ySpan0;
    let x0 = anchorX + (sr.x[0] - anchorX) * scale;
    let x1 = anchorX + (sr.x[1] - anchorX) * scale;
    let y0 = anchorY + (sr.y[0] - anchorY) * scale;
    let y1 = anchorY + (sr.y[1] - anchorY) * scale;
    const newXSpan = x1 - x0;
    const newYSpan = y1 - y0;
    const panDx = (newMidPxX - pinch.midPxX) / plotW * newXSpan;
    const panDy = (newMidPxY - pinch.midPxY) / plotH * newYSpan;
    doRelayout([x0 - panDx, x1 - panDx], [y0 + panDy, y1 + panDy]);
  }} else if (e.touches.length === 1 && touch1 && gesture !== 'pinch') {{
    gesture = 'pan';
    touch1.moved = true;
    const t = e.touches[0];
    const rect = chartDiv.getBoundingClientRect();
    const curPxX = t.clientX - rect.left;
    const curPxY = t.clientY - rect.top;
    const sr = touch1.startRanges;
    const ax = chartDiv._fullLayout.xaxis;
    const ay = chartDiv._fullLayout.yaxis;
    const xSpan = sr.x[1] - sr.x[0];
    const ySpan = sr.y[1] - sr.y[0];
    const plotW = ax._length;
    const plotH = ay._length;
    const dx = (curPxX - touch1.startPxX) / plotW * xSpan;
    const dy = (curPxY - touch1.startPxY) / plotH * ySpan;
    doRelayout([sr.x[0] - dx, sr.x[1] - dx], [sr.y[0] + dy, sr.y[1] + dy]);
  }}
}}, {{ passive: false, capture: true }});

chartDiv.addEventListener('touchend', function(e) {{
  e.stopPropagation();
  if (gesture === 'pinch' && e.touches.length < 2) {{
    gesture = null;
    pinch = null;
    if (e.touches.length === 1) {{
      const t = e.touches[0];
      const rect = chartDiv.getBoundingClientRect();
      touch1 = {{
        clientX: t.clientX, clientY: t.clientY,
        time: Date.now(),
        startPxX: t.clientX - rect.left,
        startPxY: t.clientY - rect.top,
        startRanges: getRange(),
        moved: false,
      }};
    }} else {{
      touch1 = null;
    }}
    return;
  }}
  if (e.touches.length === 0 && touch1 && !touch1.moved) {{
    const t = e.changedTouches[0];
    const dx = Math.abs(t.clientX - touch1.clientX);
    const dy = Math.abs(t.clientY - touch1.clientY);
    const dt = Date.now() - touch1.time;
    if (dx <= 12 && dy <= 12 && dt <= 300) {{
      const rect = chartDiv.getBoundingClientRect();
      const hit = findNearest(t.clientX - rect.left, t.clientY - rect.top);
      if (hit) {{
        const globalIdx = chartDiv.data[hit.ti].customdata[hit.pi];
        if (globalIdx !== undefined && globalIdx !== null) showCard(globalIdx);
      }} else {{
        card.classList.remove('show');
      }}
    }}
  }}
  if (e.touches.length === 0) {{
    gesture = null; touch1 = null; pinch = null;
  }}
}}, {{ passive: false, capture: true }});

chartDiv.addEventListener('touchcancel', function() {{
  gesture = null; touch1 = null; pinch = null;
}}, {{ passive: true }});

// ── Axis controls ─────────────────────────────────────────────────────────
const presetBtnsDiv = document.getElementById('presetBtns');
function buildPresetButtons() {{
  presetBtnsDiv.innerHTML = '';
  PRESETS.forEach(p => {{
    if (p.id === 'tsne' && !(IRIS_X && IRIS_X['tsne_x'])) return;
    if (p.id === 'umap' && !(IRIS_X && IRIS_X['umap_x'])) return;
    const btn = document.createElement('button');
    btn.className = 'preset-btn' + (p.id === currentPresetId ? ' active' : '');
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
  // Bouton Carte (toujours présent)
  const carteBtn = document.createElement('button');
  carteBtn.className = 'preset-btn' + (currentPresetId === 'carte' ? ' active' : '');
  carteBtn.dataset.id = 'carte';
  carteBtn.textContent = '🗺️ Carte';
  carteBtn.addEventListener('click', async () => {{
    document.querySelectorAll('.preset-btn').forEach(b => b.classList.remove('active'));
    carteBtn.classList.add('active');
    currentPresetId = 'carte';
    await showCarte();
  }});
  presetBtnsDiv.appendChild(carteBtn);
}}
// Boutons au démarrage (tsne/umap exclus car IRIS_X pas encore chargé)
buildPresetButtons();

const customToggle = document.getElementById('customToggle');
const customPanel = document.getElementById('customPanel');
customToggle.addEventListener('click', () => {{
  const open = customPanel.classList.toggle('open');
  customToggle.classList.toggle('open', open);
  customToggle.textContent = open ? 'Personnaliser ▴' : 'Personnaliser ▾';
}});

const mlToggleBtn = document.getElementById('mlToggleBtn');
if (mlToggleBtn) {{
  mlToggleBtn.addEventListener('click', () => {{
    mlStrictMode = !mlStrictMode;
    mlToggleBtn.textContent = mlStrictMode ? '⚠ Imputés ML : masqués' : '⚠ Imputés ML : affichés';
    mlToggleBtn.classList.toggle('full-mode', !mlStrictMode);
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
    densityToggleBtn.classList.toggle('full-mode', !densityVisible);
    restyleDensity();
  }});
}}

function buildSelect(selectId, selectedVar) {{
  const sel = document.getElementById(selectId);
  sel.innerHTML = '';
  for (const [cat, vars] of Object.entries(VARS)) {{
    const availVars = vars.filter(v => IRIS_X[v] !== undefined);
    if (!availVars.length) continue;
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

// ── Init state et UI synchrone (métadonnées inline disponibles immédiatement) ──
currentElectionId = DEFAULT_ELECTION_ID;
currentElectionType = ELECTIONS_META[DEFAULT_ELECTION_ID]?.type || 'euro';
currentElectionYear = ELECTIONS_META[DEFAULT_ELECTION_ID]?.year || 2024;
currentElectionTour = ELECTIONS_META[DEFAULT_ELECTION_ID]?.tour || 1;
currentGroupIndices = {{}};  // rempli après chargement de static.json
activeGroups = new Set(btns.map(b => b.key));

elecByType = {{}};
for (const [eid, meta] of Object.entries(ELECTIONS_META)) {{
  if (!elecByType[meta.type]) elecByType[meta.type] = {{}};
  if (!elecByType[meta.type][meta.year]) elecByType[meta.type][meta.year] = [];
  elecByType[meta.type][meta.year].push({{id: eid, tour: meta.tour}});
}}

const elecTypeBtns = document.getElementById('elecTypeBtns');
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
    currentElectionYear = parseInt(years[years.length - 1]) || 2024;
    currentElectionTour = 1;
    updateYearBtns(type);
    updateTourBtns(type, currentElectionYear);
    const eid = getElectionId(type, currentElectionYear, 1)
           || getElectionId(type, currentElectionYear, 2);
    if (eid) applyElection(eid);
  }});
  elecTypeBtns.appendChild(btn);
}}

updateYearBtns(currentElectionType);
updateTourBtns(currentElectionType, currentElectionYear);
document.getElementById('electionCurrentLabel').textContent = ELECTIONS_META[DEFAULT_ELECTION_ID]?.label || DEFAULT_ELECTION_ID;

document.getElementById('tourBtn1').addEventListener('click', () => {{
  if (document.getElementById('tourBtn1').disabled) return;
  currentElectionTour = 1;
  updateTourBtns(currentElectionType, currentElectionYear);
  const eid = getElectionId(currentElectionType, currentElectionYear, 1);
  if (eid) applyElection(eid);
}});
document.getElementById('tourBtn2').addEventListener('click', () => {{
  if (document.getElementById('tourBtn2').disabled) return;
  currentElectionTour = 2;
  updateTourBtns(currentElectionType, currentElectionYear);
  const eid = getElectionId(currentElectionType, currentElectionYear, 2);
  if (eid) applyElection(eid);
}});

// Dropdowns axes désactivés jusqu'au chargement de IRIS_X/IRIS_Y
document.getElementById('xSelect').disabled = true;
document.getElementById('ySelect').disabled = true;

rebuildFilterButtons();
updateToggleLabel();

const resetBtn = document.getElementById('resetBtn');
function checkZoomed() {{
  const xr = chartDiv.layout.xaxis.range;
  const yr = chartDiv.layout.yaxis.range;
  const zoomed = Math.abs(xr[0]-currentXRange[0])>0.05 || Math.abs(xr[1]-currentXRange[1])>0.05 ||
                 Math.abs(yr[0]-currentYRange[0])>0.05 || Math.abs(yr[1]-currentYRange[1])>0.05;
  resetBtn.classList.toggle('show', zoomed);
}}
resetBtn.addEventListener('click', () => {{
  Plotly.relayout(chartDiv, {{'xaxis.range': currentXRange.slice(), 'yaxis.range': currentYRange.slice()}});
}});

// ── Carte géographique ────────────────────────────────────────────────────
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
    fitBoundsOptions: {{ padding: 10 }},
    minZoom: 4,
    maxZoom: 16,
    dragRotate: false,
  }});
  mapInstance.touchZoomRotate.disableRotation();
  mapInstance.on('load', () => {{
    mapReady = true;
    const overlay = document.createElement('div');
    overlay.style.cssText = 'position:absolute;inset:0;background:rgba(255,255,255,0.22);pointer-events:none;z-index:2';
    document.getElementById('mapDiv').appendChild(overlay);
    _addIrisLayer(mapInstance, 'iris', 'iris-circles', idx => {{
      currentClickedGlobalIdx = idx;
      showCard(idx);
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
      dragRotate: false,
    }});
    m.touchZoomRotate.disableRotation();
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
        showCard(idx);
      }});
      const gj = buildMapGeoJSON();
      if (gj) m.getSource('iris').setData(gj);
    }});
  }});
}}

async function showCarte() {{
  isCarteActive = true;
  document.getElementById('chart').style.display = 'none';
  document.querySelectorAll('.corner-label').forEach(el => el.style.display = 'none');
  const resetBtn = document.getElementById('resetBtn');
  if (resetBtn) resetBtn.style.display = 'none';
  document.querySelector('.footer') && (document.querySelector('.footer').style.display = 'none');
  document.getElementById('axisDesc') && (document.getElementById('axisDesc').style.display = 'none');
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
  document.getElementById('chart').style.display = 'block';
  document.querySelectorAll('.corner-label').forEach(el => el.style.display = '');
  const resetBtn = document.getElementById('resetBtn');
  if (resetBtn) resetBtn.style.display = '';
  document.querySelector('.footer') && (document.querySelector('.footer').style.display = '');
  document.getElementById('axisDesc') && (document.getElementById('axisDesc').style.display = '');
}}

// ── Initialisation async ──────────────────────────────────────────────────
(async function init() {{
  setLoadingProgress(5, 'Initialisation du graphique…');
  await Plotly.newPlot(chartDiv, figData.data, figData.layout, {{
    responsive: true, displayModeBar: false, scrollZoom: false,
    doubleClick: false, staticPlot: false,
  }});
  chartDiv.on('plotly_relayout', checkZoomed);

  setLoadingProgress(10, 'Chargement des données…', 'static.json + élection par défaut');

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

  setLoadingProgress(40, 'Données électorales chargées — chargement des axes…', 'iris_x_mobile.json + iris_y_mobile.json');

  await applyElection(DEFAULT_ELECTION_ID);
  setCorners(PRESETS[0].corners);
  updateDesc(PRESETS[0], PRESETS[0].xVar, PRESETS[0].yVar);

  const [xData, yData, mlFlagsData] = await Promise.all([
    fetch('data/iris_x_mobile.json').then(r => r.json()),
    fetch('data/iris_y_mobile.json').then(r => r.json()),
    fetch('data/ml_flags.json').then(r => r.json()).catch(() => null),
  ]);

  IRIS_X = xData;
  IRIS_Y = yData;
  ML_FLAGS = mlFlagsData;
  if (mlToggleBtn && ML_FLAGS && Object.keys(ML_FLAGS).length > 0) mlToggleBtn.style.display = '';

  buildPresetButtons();
  buildSelect('xSelect', PRESETS[0].xVar);
  buildSelect('ySelect', PRESETS[0].yVar);
  buildColorVarSelect();
  updateColorLegend();
  document.getElementById('xSelect').disabled = false;
  document.getElementById('ySelect').disabled = false;
  document.getElementById('xInvertChk').checked = PRESETS[0].xInvert || false;

  setLoadingProgress(90, 'Finalisation…');
  applyAxes(PRESETS[0].xVar, PRESETS[0].xInvert, PRESETS[0].yVar, PRESETS[0]);

  setLoadingProgress(100);
  hideLoadingOverlay();

  // Ouvrir la carte par défaut
  await showCarte();
}})().catch(function(err) {{
  document.getElementById('loadingMsg').textContent = 'Erreur : ' + err.message;
  document.getElementById('loadingDetail').textContent = err.stack || '';
  console.error('init error', err);
}});

</script>
</body>
</html>"""
    return html


mobile_html = build_mobile_html(skip_build=_SKIP_BUILD)
with open("saint_graphique_iris_mobile.html", "w", encoding="utf-8") as f:
    f.write(mobile_html)
print(f"Mobile  → saint_graphique_iris_mobile.html ({len(mobile_html)//1024} KB)")
