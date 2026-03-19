import pandas as pd
import numpy as np
import plotly.graph_objects as go
import json as _json
import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

# ── CONFIGURATION ─────────────────────────────────────────────────────────────
COULEURS = {
    "LFI-NFP": "#DC2626", "ECOS": "#16A34A", "GDR": "#9B1C1C",
    "SOC":     "#EC4899", "LIOT": "#D97706", "EPR": "#F97316",
    "DEM":     "#FB923C", "HOR":  "#0EA5E9", "DR":  "#1E40AF",
    "UDDPLR":  "#1E3A8A", "RN":   "#374151", "NI":  "#9CA3AF",
}
LABELS = {
    "LFI-NFP": "La France Insoumise", "ECOS": "Écologistes",  "GDR":    "GDR (PCF)",
    "SOC":     "Socialistes",          "LIOT": "LIOT",         "EPR":    "Ensemble / Renaissance",
    "DEM":     "MoDem",                "HOR":  "Horizons",     "DR":     "Droite Républicaine",
    "UDDPLR":  "UDR",                  "RN":   "Rassemblement National", "NI": "Non-inscrits",
}
SHORT = {
    "LFI-NFP": "LFI", "ECOS": "Écolos", "GDR": "GDR", "SOC": "PS",
    "LIOT": "LIOT", "EPR": "EPR", "DEM": "MoDem", "HOR": "Horizons",
    "DR": "DR", "UDDPLR": "UDR", "RN": "RN", "NI": "NI",
}
ORDER = ["LFI-NFP","ECOS","GDR","SOC","LIOT","EPR","DEM","HOR","DR","UDDPLR","RN","NI"]

AXIS_PRESETS = [
    {
        'id': 'saint_graal',
        'label': 'Saint-Graphique',
        'emoji': '⚒️',
        'xVar': 'score_exploitation', 'xInvert': True,
        'yVar': 'score_domination',
        'xTitle': '← Exploité (prolétaire) ─── Position dans le rapport capital/travail ─── Exploiteur (bourgeois) →',
        'yTitle': '← Dominé ─── Domination sociale (Bourdieu) ─── Dominant →',
        'xRange': [-2.5, 2.5], 'yRange': [-3.0, 3.0],
        'corners': [
            {'pos': 'tl', 'text': 'ASCENSION<br>SOCIALE', 'color': '#E87070'},
            {'pos': 'tr', 'text': 'REPRODUCTION<br>DU CAPITAL', 'color': '#6B8FD4'},
            {'pos': 'bl', 'text': 'PROLÉTARIAT', 'color': '#60B87A'},
            {'pos': 'br', 'text': 'PETITE<br>BOURGEOISIE', 'color': '#C49A30'},
        ],
        'desc': {
            'title': 'Le Saint-Graphique — Marx × Bourdieu',
            'x': "<b>Axe X — Position dans le rapport capital/travail (Marx)</b> : mesure d'où vient le revenu. À droite : les <em>exploiteurs</em> (bourgeois, rentiers, patrons) — leur revenu provient du capital, des rentes, de la propriété des moyens de production. À gauche : les <em>exploités</em> (prolétaires, employés, ouvriers) — leur revenu provient de la vente de leur force de travail. Variables clés : part patrimoine (PPAT), taux de pauvreté, part ouvriers/employés, locataires, prestations sociales.",
            'y': '<b>Axe Y — Domination sociale (Bourdieu)</b> : mesure la position dans la hiérarchie sociale totale (capital économique + culturel). En haut : les <em>dominants</em> (riches, très diplômés, cadres supérieurs). En bas : les <em>dominés</em> (pauvres, peu ou pas diplômés). Variables clés : part BAC+5, cadres, niveau de vie médian, taux de pauvreté.',
            'quadrants': {
                'tr': '<b>Reproduction du capital</b> — Exploiteur ET dominant : la grande bourgeoisie classique. Vit du capital, riche, diplômé. EPR, DR, UDR.',
                'tl': '<b>Ascension sociale</b> — Exploité mais dominant : cadres, ingénieurs, professions intellectuelles issus de milieux populaires. Paradoxe de la méritocratie.',
                'bl': '<b>Prolétariat</b> — Exploité ET dominé : classe ouvrière, employés précaires, zones industrielles. LFI, GDR, RN des périphéries.',
                'br': '<b>Petite bourgeoisie</b> — Exploiteur mais dominé : artisans, commerçants, petits patrons. Capital productif limité, statut social modeste.',
            }
        }
    },
    {
        'id': 'pca',
        'label': 'ACP',
        'emoji': '📊',
        'xVar': 'PC1', 'xInvert': False,
        'yVar': 'PC2',
        'xTitle': '← Urbain · diplômé · cadres ─── ACP Composante 1 ─── Rural · ouvrier · voiture →',
        'yTitle': '← Aisé · emploi ─── ACP Composante 2 ─── Pauvre · chômage →',
        'xRange': [-20, 12], 'yRange': [-10, 18],
        'corners': [
            {'pos': 'tl', 'text': 'PAUVRE<br>URBAIN', 'color': '#E87070'},
            {'pos': 'tr', 'text': 'PAUVRE<br>RURAL', 'color': '#6B8FD4'},
            {'pos': 'bl', 'text': 'RICHE<br>URBAIN', 'color': '#60B87A'},
            {'pos': 'br', 'text': 'RICHE<br>RURAL', 'color': '#C49A30'},
        ],
        'desc': {
            'title': 'Analyse en composantes principales (ACP)',
            'x': "<b>PC1 — Gradient urbain/rural et capital humain</b> : la première composante principale explique la plus grande part de variance des 53 variables de circonscription. À gauche : urbain, diplômé, cadres, transports en commun, vélo. À droite : rural, ouvrier, voiture, maisons, agriculture. C\'est l\'axe structurant le plus fort de la géographie sociale française.",
            'y': "<b>PC2 — Gradient pauvreté/richesse</b> : deuxième composante, orthogonale à PC1. En haut : taux de pauvreté élevé, chômage, faibles revenus. En bas : niveau de vie médian élevé, fort taux d\'emploi. Capture l\'inégalité économique indépendamment du type d\'espace (urbain ou rural).",
            'quadrants': {
                'tl': '<b>Pauvre urbain</b> — Banlieues populaires, quartiers défavorisés des grandes villes, Seine-Saint-Denis.',
                'tr': '<b>Pauvre rural</b> — France périphérique, zones rurales en déclin, ex-bassins industriels reconvertis.',
                'bl': '<b>Riche urbain</b> — Beaux quartiers parisiens, métropoles dynamiques, centres-villes aisés.',
                'br': '<b>Riche rural</b> — Campagnes aisées, zones résidentielles périurbaines, terroir agricole prospère.',
            }
        }
    },
    {
        'id': 'bourdieu',
        'label': 'Bourdieu',
        'emoji': '🎓',
        'xVar': 'score_cap_cult', 'xInvert': False,
        'yVar': 'score_cap_eco',
        'xTitle': '← Peu diplômé · ouvrier ─── Capital culturel (Bourdieu) ─── Diplômé · cadre →',
        'yTitle': '← Pauvre ─── Capital économique (Bourdieu) ─── Riche →',
        'xRange': [-2.0, 4.0], 'yRange': [-3.5, 3.5],
        'corners': [
            {'pos': 'tl', 'text': 'NOBLESSE<br>DU CAPITAL', 'color': '#E87070'},
            {'pos': 'tr', 'text': 'ÉLITE<br>INTÉGRÉE', 'color': '#6B8FD4'},
            {'pos': 'bl', 'text': 'CLASSE<br>POPULAIRE', 'color': '#60B87A'},
            {'pos': 'br', 'text': 'INTELLECTUELS<br>DÉCLASSÉS', 'color': '#C49A30'},
        ],
        'desc': {
            'title': 'Espace bourdieusien — Capital culturel × Capital économique',
            'x': '<b>Axe X — Capital culturel</b> : accumulation de savoirs, diplômes et distinction culturelle. À droite : forte proportion de BAC+5, cadres supérieurs, professions intellectuelles (médecins, enseignants-chercheurs, avocats). À gauche : peu ou pas diplômés, ouvriers, CAP/BEP. Bourdieu distingue ce capital comme source autonome de pouvoir social, irréductible à la richesse.',
            'y': "<b>Axe Y — Capital économique</b> : richesse, revenus et patrimoine. En haut : niveau de vie médian élevé, ménages aisés, forte part de patrimoine dans les revenus (PPAT). En bas : pauvreté, faibles revenus, dépendance aux prestations sociales. Bourdieu articule ce capital au capital culturel pour constituer l\'espace social.",
            'quadrants': {
                'tl': '<b>Noblesse du capital</b> — Riche mais peu diplômé : héritiers, patrons autodidactes, propriétaires fonciers. Capital économique sans légitimité culturelle.',
                'tr': '<b>Élite intégrée</b> — Riche ET diplômé : grandes écoles, hauts fonctionnaires, PDG. La fraction dominante de la classe dominante.',
                'bl': '<b>Classe populaire</b> — Pauvre ET peu diplômé : prolétariat au sens classique, cumul des handicaps sociaux.',
                'br': '<b>Intellectuels déclassés</b> — Diplômé mais peu riche : enseignants, chercheurs, artistes. Capital culturel élevé, faible capital économique.',
            }
        }
    },
    {
        'id': 'precarite',
        'label': 'Précarité',
        'emoji': '📍',
        'xVar': 'score_precarite', 'xInvert': False,
        'yVar': 'score_ruralite',
        'xTitle': '← Aisé · stable ─── Précarité multidimensionnelle ─── Précaire · pauvre →',
        'yTitle': '← Métropole ─── Ruralité / Périphérie ─── Rural profond →',
        'xRange': [-2.0, 4.0], 'yRange': [-3.0, 3.0],
        'corners': [
            {'pos': 'tl', 'text': 'RURAL<br>AISÉ', 'color': '#E87070'},
            {'pos': 'tr', 'text': 'PÉRIPHÉRIE<br>PRÉCAIRE', 'color': '#6B8FD4'},
            {'pos': 'bl', 'text': 'MÉTROPOLE<br>AISÉE', 'color': '#60B87A'},
            {'pos': 'br', 'text': 'BANLIEUE<br>PRÉCAIRE', 'color': '#C49A30'},
        ],
        'desc': {
            'title': 'Précarité multidimensionnelle × Ruralité',
            'x': '<b>Axe X — Précarité multidimensionnelle</b> : score composite combinant taux de pauvreté, chômage, locataires, familles monoparentales, faibles diplômes et dépendance aux prestations sociales. À droite : forte précarité cumulée. À gauche : stabilité économique et sociale.',
            'y': "<b>Axe Y — Ruralité / Périphérie</b> : mesure l\'éloignement des grands centres urbains et des services. En haut : rural profond, agriculture, chauffage au fioul, forte dépendance à la voiture, communes hors aire d\'attraction des villes. En bas : métropoles denses, transports en commun, économie tertiaire.",
            'quadrants': {
                'tl': "<b>Rural aisé</b> — Campagne prospère, agriculture productive, bourgeoisie rurale. Faible précarité malgré l\'éloignement.",
                'tr': '<b>Périphérie précaire</b> — La "France des oubliés" : zones rurales ou périurbaines pauvres, ex-bassins industriels, gilets jaunes.',
                'bl': '<b>Métropole aisée</b> — Centres-villes dynamiques, économie de la connaissance, hauts revenus urbains.',
                'br': '<b>Banlieue précaire</b> — Zones urbaines défavorisées, banlieues populaires, concentration de pauvreté en milieu dense.',
            }
        }
    },
    {
        'id': 'rentier',
        'label': 'Rentier',
        'emoji': '💰',
        'xVar': 'score_rentier', 'xInvert': False,
        'yVar': 'score_domination',
        'xTitle': '← Revenu du travail (salaires) ─── Rentier vs Travailleur ─── Revenu du capital (PPAT) →',
        'yTitle': '← Dominé ─── Domination sociale (Bourdieu) ─── Dominant →',
        'xRange': [-3.0, 6.0], 'yRange': [-3.0, 3.0],
        'corners': [
            {'pos': 'tl', 'text': 'INTELLECTUELS<br>FONCTIONNAIRES', 'color': '#E87070'},
            {'pos': 'tr', 'text': 'ÉLITE<br>RENTIÈRE', 'color': '#6B8FD4'},
            {'pos': 'bl', 'text': 'SALARIÉS<br>PRÉCAIRES', 'color': '#60B87A'},
            {'pos': 'br', 'text': 'PETITS<br>RENTIERS', 'color': '#C49A30'},
        ],
        'desc': {
            'title': 'Rente vs Travail × Domination sociale',
            'x': '<b>Axe X — Score rentier (rente vs travail)</b> : mesure la part du revenu tirée du capital (PPAT — patrimoine, loyers, dividendes) vs du travail salarié (PACT) ou des prestations. À droite : les <em>rentiers</em>, dont le revenu provient principalement du patrimoine et de ses fruits. À gauche : les <em>travailleurs</em> dépendant de leurs salaires ou des aides. Cet axe distingue ce que Marx appelait "bourgeoisie financière" (droite) des "salariés-cadres" bien payés mais non-rentiers (gauche-haut).',
            'y': '<b>Axe Y — Domination sociale (Bourdieu)</b> : même que le Saint-Graphique. En haut : dominant (riche + diplômé). En bas : dominé.',
            'quadrants': {
                'tl': '<b>Intellectuels / Fonctionnaires</b> — Dominants mais non-rentiers : hauts fonctionnaires, grandes écoles, médecins. Revenu du travail qualifié, peu de patrimoine relatif.',
                'tr': '<b>Élite rentière</b> — Dominant ET rentier : vieille bourgeoisie, héritiers, propriétaires immobiliers parisiens. Le cœur de la reproduction des inégalités.',
                'bl': '<b>Salariés précaires</b> — Dominé ET non-rentier : classe laborieuse sans patrimoine, dépend entièrement du travail ou des aides.',
                'br': '<b>Petits rentiers</b> — Rentier mais pas dominant socialement : retraités propriétaires, petits bailleurs provinciaux, héritiers modestes.',
            }
        }
    },
]

# Variables available in custom dropdowns, organized by category
VARS_BY_CAT = {
    'Scores composites': ['score_exploitation','score_domination','score_cap_eco','score_cap_cult','score_precarite','score_ruralite','score_rentier'],
    'ACP': ['PC1','PC2','PC3','PC4','PC5'],
    'Revenu et inégalités': ['nivvie_median_diff','tx_pauvrete60_diff','part_aises_diff','part_pauvres_diff','D9_diff','D1_diff','rpt_D9_D1_diff','PPAT','PACT','PPEN','PPSOC'],
    'Emploi et formation': ['actemp','actcho','act_cad','act_int','act_emp','act_ouv','act_art','act_agr','actdip_BAC5','actdip_PEU'],
    'Démographie': ['age_moyen','dec50','tvar_pop','inactret','inactetu'],
    'Logement et mobilité': ['proprio','locatai','maison','mfuel','modtrans_voit','modtrans_commun','modtrans_velo','pop_horsaav'],
    'Social et famille': ['men_monop','men_coupae','pop_urb','mobresid'],
}

ALL_VARS = []
for cat_vars in VARS_BY_CAT.values():
    for v in cat_vars:
        if v not in ALL_VARS:
            ALL_VARS.append(v)

# ── 1. CHARGEMENT ET MERGE DES DONNÉES ───────────────────────────────────────
df_new = pd.read_csv("deputes_enriched_circo.csv")
df_old = pd.read_csv("bloc_bourgeois_heritage_final.csv")

df = df_new.merge(
    df_old[['député','libelle_pcs','ville_naissance','pays_naissance']],
    on='député', how='left'
)
print(f"Données après merge : {len(df)} lignes × {len(df.columns)} colonnes")

# ── 2. TAILLE DES MARQUEURS basée sur nivvie_median_diff ─────────────────────
niv = df['nivvie_median_diff'].fillna(df['nivvie_median_diff'].mean())
q5  = niv.quantile(0.05)
q95 = niv.quantile(0.95)
niv_clipped = niv.clip(q5, q95)
marker_size = 6 + (niv_clipped - q5) / (q95 - q5) * (16 - 6)

np.random.seed(42)
N = len(df)
# Per-deputy jitter seed: fixed unit jitter in [-1, 1] per deputy, scaled per variable
jit_unit_x = np.random.uniform(-1, 1, N)
jit_unit_y = np.random.uniform(-1, 1, N)

def jittered(col_name, scale=0.03):
    """Return jittered values for a column, replacing NaN with mean."""
    vals = df[col_name].copy().astype(float)
    mean_val = vals.mean()
    vals = vals.fillna(mean_val)
    std_val = vals.std()
    if np.isnan(std_val) or std_val == 0:
        std_val = 1.0
    return vals + jit_unit_x * scale * std_val

# Build jittered data for all variables
var_data_x = {}  # var -> jittered array for x-use
var_data_y = {}  # var -> jittered array for y-use
for v in ALL_VARS:
    if v not in df.columns:
        print(f"  WARNING: variable {v} not in dataframe, skipping")
        continue
    vals = df[v].copy().astype(float)
    mean_val = vals.mean()
    vals_filled = vals.fillna(mean_val)
    std_val = vals_filled.std()
    if np.isnan(std_val) or std_val == 0:
        std_val = 1.0
    scale = 0.03
    var_data_x[v] = (vals_filled + jit_unit_x * scale * std_val).tolist()
    var_data_y[v] = (vals_filled + jit_unit_y * scale * std_val).tolist()

# Saint Graal default values
sg_x_raw = jittered('score_exploitation')
sg_y = jittered('score_domination')
sg_x = -sg_x_raw  # inverted for display

# barycentre means per group per variable
def group_means():
    bm = {}
    for g in ORDER:
        sub = df[df['groupe_politique'] == g]
        if sub.empty:
            continue
        bm[g] = {}
        for v in ALL_VARS:
            if v not in df.columns:
                continue
            vals = pd.to_numeric(sub[v], errors='coerce')
            bm[g][v] = float(vals.mean()) if vals.notna().any() else 0.0
    return bm

bary_means = group_means()

# Group data: groupData[g][v] = [jittered vals for deputies in group g]
def build_group_data():
    gd_x = {}
    gd_y = {}
    indices_by_group = {}
    for g in ORDER:
        mask = df['groupe_politique'] == g
        if not mask.any():
            continue
        indices_by_group[g] = df.index[mask].tolist()
        gd_x[g] = {}
        gd_y[g] = {}
        for v in ALL_VARS:
            if v not in var_data_x:
                continue
            idxs = indices_by_group[g]
            gd_x[g][v] = [var_data_x[v][i] for i in idxs]
            gd_y[g][v] = [var_data_y[v][i] for i in idxs]
    return gd_x, gd_y

group_data_x, group_data_y = build_group_data()

# var_labels from json (subset of our vars)
with open("var_labels.json", encoding='utf-8') as f:
    var_labels_all = _json.load(f)

var_labels = {}
for v in ALL_VARS:
    if v in var_labels_all:
        var_labels[v] = var_labels_all[v]
    else:
        var_labels[v] = v  # fallback

# ── 3. BARYCENTRES ────────────────────────────────────────────────────────────
bary_rows = []
for g in ORDER:
    sub = df[df['groupe_politique'] == g]
    if len(sub) < 2:
        continue
    bary_rows.append({
        'g': g,
        'x': float(-sub['score_exploitation'].mean()),
        'y': float(sub['score_domination'].mean()),
        'n': len(sub),
        'c': COULEURS.get(g, "#999"),
        'label': LABELS.get(g, g),
        'abbr': SHORT.get(g, g),
    })
bary = pd.DataFrame(bary_rows)

# ── 4. PLOTLY FIGURE (simple, pour affichage Jupyter) ─────────────────────────
def build_jupyter_fig():
    fig = go.Figure()

    fig.add_vline(x=0, line_dash="dot", line_color="#BBBBBB", line_width=1.5)
    fig.add_hline(y=0, line_dash="dot", line_color="#BBBBBB", line_width=1.5)

    # Corner annotations
    corners = AXIS_PRESETS[0]['corners']
    corner_map = {
        'tl': (-2.4,  2.85, 'left',  'top'),
        'tr': ( 2.4,  2.85, 'right', 'top'),
        'bl': (-2.4, -2.85, 'left',  'bottom'),
        'br': ( 2.4, -2.85, 'right', 'bottom'),
    }
    for c in corners:
        cx, cy, xa, ya = corner_map[c['pos']]
        fig.add_annotation(x=cx, y=cy, text=f"<b>{c['text']}</b>",
                           showarrow=False, xanchor=xa, yanchor=ya,
                           font=dict(size=9.5, color=c['color'], family="Helvetica Neue, sans-serif"))

    # Barycentres
    fig.add_trace(go.Scatter(
        x=bary['x'], y=bary['y'], mode="markers+text",
        marker=dict(symbol="cross-thin", size=18, color=bary['c'].tolist(),
                    line=dict(width=3, color=bary['c'].tolist())),
        text=bary['abbr'], textposition="top right",
        textfont=dict(size=9, color=bary['c'].tolist(), family="Helvetica Neue, sans-serif"),
        hovertemplate="<b>Barycentre %{text}</b><br>X moyen : <b>%{x:.3f}</b><br>Y moyen : <b>%{y:.3f}</b><extra></extra>",
        showlegend=False, opacity=0.75))

    for g in ORDER:
        sub = df[df['groupe_politique'] == g].copy()
        if sub.empty:
            continue
        c, lb = COULEURS.get(g, "#999"), LABELS.get(g, g)
        sub_idx = sub.index.tolist()
        x_vals = [-var_data_x['score_exploitation'][i] for i in sub_idx]
        y_vals = [var_data_y['score_domination'][i] for i in sub_idx]
        sz = [marker_size[i] for i in sub_idx]
        fig.add_trace(go.Scatter(
            x=x_vals, y=y_vals, mode="markers", name=lb,
            marker=dict(color=c, size=sz, opacity=0.82,
                        line=dict(width=0.7, color="rgba(255,255,255,0.6)")),
            customdata=sub[['député','libelle_pcs','nivvie_median_diff','score_domination',
                             'anciennete','ville_naissance','pays_naissance','circo_label',
                             'age','scoreParticipation','scoreLoyaute']].values,
            hovertemplate=(
                f"<span style='font-size:13px;font-weight:900'>%{{customdata[0]}}</span>"
                f"<br><span style='color:{c};font-weight:700;font-size:10px'>{lb}</span>"
                "<br><span style='color:#888'>Profession :</span> <b>%{customdata[1]}</b>"
                "<br><span style='color:#888'>Circonscription :</span> <b>%{customdata[7]}</b>"
                "<br><span style='color:#888'>Né·e à :</span> %{customdata[5]}"
                "<br><span style='color:#888'>Âge :</span> %{customdata[8]} ans"
                "<br><span style='color:#888'>Revenu circo :</span> <b>%{customdata[2]:,.0f}</b>"
                "<br><span style='color:#888'>Score domination :</span> <b>%{customdata[3]:.2f}</b>"
                "<br><span style='color:#888'>Participation :</span> %{customdata[9]:.0%}&ensp;·&ensp;"
                "<span style='color:#888'>Loyauté :</span> %{customdata[10]:.0%}"
                "<extra></extra>"),
            legendgroup=g))

    fig.update_layout(
        width=1200, height=800, paper_bgcolor="#FAF9F7", plot_bgcolor="#FEFDFB",
        margin=dict(t=100, b=120, l=95, r=30),
        title=dict(
            text="<b>Sociologie des députés — Saint Graal</b><br><sup style='color:#888;font-size:11px;font-weight:400'>Score exploitation (inversé) × Score domination — XVIIe législature</sup>",
            x=0.5, xanchor="center", y=0.985,
            font=dict(size=22, family="Helvetica Neue, sans-serif", color="#1a1a1a")),
        xaxis=dict(
            title=dict(
                text="← Exploité (prolétaire) ─── Position dans le rapport capital/travail ─── Exploiteur (bourgeois) →",
                font=dict(size=11, color="#555", family="Helvetica Neue, sans-serif"), standoff=10),
            range=[-2.5, 2.5],
            showgrid=True, gridcolor="#EBEBEB", gridwidth=0.8, zeroline=False,
            tickfont=dict(size=9.5, color="#AAA"), linecolor="#DDD", linewidth=1),
        yaxis=dict(
            title=dict(
                text="← Dominé ─── Domination sociale (Bourdieu) ─── Dominant →",
                font=dict(size=11, color="#555", family="Helvetica Neue, sans-serif"), standoff=8),
            range=[-3.0, 3.0],
            showgrid=True, gridcolor="#EBEBEB", gridwidth=0.8, zeroline=False,
            tickfont=dict(size=9.5, color="#AAA"), linecolor="#DDD", linewidth=1),
        legend=dict(orientation="h", x=0.5, xanchor="center", y=-0.17, yanchor="top",
                    font=dict(size=10.5, family="Helvetica Neue, sans-serif"),
                    bgcolor="rgba(0,0,0,0)", itemclick="toggle", itemdoubleclick="toggleothers"),
        hoverlabel=dict(bgcolor="rgba(255,255,255,0.97)", bordercolor="#E0E0E0",
                        font=dict(size=12, family="Helvetica Neue, sans-serif", color="#1a1a1a")),
    )
    return fig


# ── 5. HTML BUILDER HELPER ────────────────────────────────────────────────────
def _build_js_data():
    """Build embedded JS data strings for both HTML versions."""
    # Build per-group per-variable data arrays
    # Format: groupDataX[g][v] = [jittered x vals], groupDataY[g][v] = [jittered y vals]
    gd_x_js = {}
    gd_y_js = {}
    for g in ORDER:
        if g not in group_data_x:
            continue
        gd_x_js[g] = group_data_x[g]
        gd_y_js[g] = group_data_y[g]

    # bary means
    bm_js = bary_means

    # Presets config
    presets_js = AXIS_PRESETS

    # Vars config
    vars_js = VARS_BY_CAT

    return (
        _json.dumps(gd_x_js, ensure_ascii=False),
        _json.dumps(gd_y_js, ensure_ascii=False),
        _json.dumps(bm_js, ensure_ascii=False),
        _json.dumps(presets_js, ensure_ascii=False),
        _json.dumps(vars_js, ensure_ascii=False),
        _json.dumps(var_labels, ensure_ascii=False),
    )


def _build_trace_data():
    """Build initial Plotly trace data for embedding in HTML."""
    traces = []

    # Trace 0: barycentres
    bary_trace = go.Scatter(
        x=bary['x'].tolist(), y=bary['y'].tolist(),
        mode="markers+text",
        marker=dict(symbol="cross-thin", size=18, color=bary['c'].tolist(),
                    line=dict(width=3, color=bary['c'].tolist())),
        text=bary['abbr'].tolist(), textposition="top right",
        textfont=dict(size=9, color=bary['c'].tolist(), family="Helvetica Neue, sans-serif"),
        hovertemplate="<b>Barycentre %{text}</b><br>X moyen : <b>%{x:.3f}</b><br>Y moyen : <b>%{y:.3f}</b><extra></extra>",
        showlegend=False, opacity=0.75,
        name="barycentres"
    )
    traces.append(bary_trace)

    trace_group_map = {}
    trace_idx = 1
    for g in ORDER:
        sub = df[df['groupe_politique'] == g].copy()
        if sub.empty:
            continue
        c, lb = COULEURS.get(g, "#999"), LABELS.get(g, g)
        sub_idx = sub.index.tolist()
        x_vals = [-var_data_x['score_exploitation'][i] for i in sub_idx]
        y_vals = [var_data_y['score_domination'][i] for i in sub_idx]
        sz = [float(marker_size[i]) for i in sub_idx]

        niv_vals = sub['nivvie_median_diff'].fillna(sub['nivvie_median_diff'].mean()).tolist()

        customdata = sub[['député','libelle_pcs','anciennete','ville_naissance','pays_naissance',
                           'circo_label','age','scoreParticipation','scoreLoyaute']].copy()
        customdata.insert(0, 'nivvie_median_diff', niv_vals)
        customdata_list = customdata.fillna('').values.tolist()

        tr = go.Scatter(
            x=x_vals, y=y_vals, mode="markers", name=lb,
            marker=dict(color=c, size=sz, opacity=0.82,
                        line=dict(width=0.7, color="rgba(255,255,255,0.6)")),
            customdata=customdata_list,
            hoverinfo="none",
            showlegend=False,
            legendgroup=g,
        )
        traces.append(tr)
        trace_group_map[g] = trace_idx
        trace_idx += 1

    return traces, trace_group_map


# ── 6. BUILD DESKTOP HTML ────────────────────────────────────────────────────
def build_desktop_html():
    gd_x_js, gd_y_js, bm_js, presets_js, vars_js, vl_js = _build_js_data()
    traces, trace_group_map = _build_trace_data()

    buttons_data = []
    for g in ORDER:
        if g not in trace_group_map:
            continue
        buttons_data.append({
            'key': g, 'short': SHORT.get(g, g), 'label': LABELS.get(g, g),
            'color': COULEURS.get(g, '#999'), 'traceIdx': trace_group_map[g],
            'count': int(df[df['groupe_politique'] == g].shape[0]),
        })

    # Build plotly figure for desktop
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

    sg = AXIS_PRESETS[0]

    html = f"""<!DOCTYPE html>
<html lang="fr">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>L'Assemblée des classes — Heritage</title>
<style>
* {{ margin: 0; padding: 0; box-sizing: border-box; }}
html, body {{ background: #FAF9F7; font-family: 'Helvetica Neue', system-ui, sans-serif; color: #1a1a1a; }}
.page-header {{ text-align: center; padding: 20px 20px 8px; }}
.page-header h1 {{ font-size: 24px; font-weight: 900; letter-spacing: -0.5px; margin-bottom: 4px; }}
.page-header p {{ font-size: 12px; color: #888; }}

/* Axis selector bar */
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
.custom-panel {{ display: none; padding: 10px 20px; background: #f9f9f9;
                 border-bottom: 1px solid #E8E8E8; gap: 16px; align-items: center; flex-wrap: wrap; }}
.custom-panel.open {{ display: flex; }}
.custom-panel label {{ font-size: 12px; font-weight: 600; color: #555; }}
.custom-panel select {{ padding: 5px 8px; border-radius: 8px; border: 1px solid #D0D0D0;
                        background: #fff; font-size: 12px; font-family: inherit; color: #222;
                        cursor: pointer; min-width: 220px; }}

/* Main layout */
.main-layout {{ display: flex; gap: 0; }}
.chart-col {{ flex: 1; min-width: 0; position: relative; }}
.sidebar {{ width: 300px; flex-shrink: 0; padding: 16px; border-left: 1px solid #EBEBEB;
            overflow-y: auto; }}

/* Group filter buttons */
.group-filters {{ padding: 10px 20px 6px; display: flex; flex-wrap: wrap; gap: 5px; align-items: center; }}
.toggle-all-btn {{ padding: 3px 10px; border-radius: 12px; border: 1px solid #CCC;
                   background: transparent; font-size: 10px; font-weight: 700; color: #888;
                   font-family: inherit; cursor: pointer; }}
.grp-btn {{ padding: 3px 10px; border-radius: 12px; border: 2px solid; font-size: 10px;
            font-weight: 700; font-family: inherit; cursor: pointer; transition: all 0.12s; }}
.grp-btn.on {{ color: #fff; }}
.grp-btn.off {{ background: transparent !important; opacity: 0.3; }}

/* Chart wrapper and corners */
.chart-wrapper {{ position: relative; }}
#chartDiv {{ width: 100%; height: 640px; }}
.corner-label {{ position: absolute; font-size: 10px; font-weight: 800; line-height: 1.2;
                 pointer-events: none; opacity: 0.65; }}
.corner-tl {{ top: 12px; left: 80px; text-align: left; }}
.corner-tr {{ top: 12px; right: 40px; text-align: right; }}
.corner-bl {{ bottom: 70px; left: 80px; text-align: left; }}
.corner-br {{ bottom: 70px; right: 40px; text-align: right; }}

/* Info card (sidebar) */
.info-card-desktop {{ background: #fff; border: 1px solid #E8E8E8; border-radius: 12px;
                      padding: 16px; font-size: 13px; line-height: 1.6; }}
.info-card-desktop .name {{ font-size: 16px; font-weight: 900; margin-bottom: 2px; }}
.info-card-desktop .party {{ font-weight: 700; font-size: 11px; margin-bottom: 10px; }}
.info-card-desktop .row {{ color: #555; margin-bottom: 2px; }}
.info-card-desktop .row b {{ color: #1a1a1a; }}
.info-card-desktop .lbl {{ color: #999; }}
.info-card-desktop.empty {{ color: #AAA; font-size: 12px; text-align: center; padding: 40px 16px; }}
.info-card-desktop .dynamic-row {{ color: #555; margin-bottom: 2px; border-top: 1px solid #F0F0F0; padding-top: 6px; margin-top: 4px; }}

/* Legend */
.legend-area {{ padding: 10px 20px; display: flex; flex-wrap: wrap; gap: 10px; justify-content: center;
                border-top: 1px solid #EBEBEB; }}
.legend-item {{ display: flex; align-items: center; gap: 5px; font-size: 11px; color: #555; cursor: pointer; }}
.legend-dot {{ width: 10px; height: 10px; border-radius: 50%; flex-shrink: 0; }}

/* Footer */
.footer {{ text-align: center; padding: 10px 20px 6px; font-size: 9px; color: #AAA; }}

/* Axis description panel */
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
</style>
</head>
<body>

<div class="page-header">
  <h1>L'Assemblée des classes — Heritage</h1>
  <p>Sociologie des {len(df)} députés · XVIIe législature · taille = revenu circo</p>
</div>

<div class="axis-bar" id="axisBar">
  <span class="axis-bar-label">Axes :</span>
  <div class="preset-btns" id="presetBtns"></div>
  <button class="custom-toggle" id="customToggle">Personnaliser ▾</button>
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
    <div class="group-filters" id="groupFilters">
      <button class="toggle-all-btn" id="toggleAllBtn">Tout décocher</button>
    </div>
    <div class="chart-wrapper">
      <div id="chartDiv"></div>
      <div class="corner-label corner-tl" id="cornerTL" style="color:{sg['corners'][0]['color']}"></div>
      <div class="corner-label corner-tr" id="cornerTR" style="color:{sg['corners'][1]['color']}"></div>
      <div class="corner-label corner-bl" id="cornerBL" style="color:{sg['corners'][2]['color']}"></div>
      <div class="corner-label corner-br" id="cornerBR" style="color:{sg['corners'][3]['color']}"></div>
    </div>
    <div class="legend-area" id="legendArea"></div>
    <div class="footer">⊕ = barycentre du groupe &nbsp;·&nbsp; taille = revenu médian de la circonscription &nbsp;·&nbsp; N={len(df)}</div>
    <div class="axis-desc" id="axisDesc"></div>
  </div>
  <div class="sidebar">
    <div class="info-card-desktop empty" id="infoCard">
      <p>Cliquez sur un député<br>pour voir sa fiche</p>
    </div>
  </div>
</div>

<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<script>
const figData = {fig_json};
const groupDataX = {gd_x_js};
const groupDataY = {gd_y_js};
const baryMeans  = {bm_js};
const PRESETS    = {presets_js};
const VARS       = {vars_js};
const varLabels  = {vl_js};
const btns       = {_json.dumps(buttons_data)};
const ORDER      = {_json.dumps(ORDER)};
const COULEURS   = {_json.dumps(COULEURS)};
const LABELS     = {_json.dumps(LABELS)};

// Barycentre group order matches btns
let currentXVar = 'score_exploitation';
let currentYVar = 'score_domination';
let currentXInvert = true;
let currentPresetId = 'saint_graal';
let currentXRange = PRESETS[0].xRange.slice();
let currentYRange = PRESETS[0].yRange.slice();
let currentCorners = PRESETS[0].corners;

const chartDiv = document.getElementById('chartDiv');

Plotly.newPlot(chartDiv, figData.data, figData.layout, {{
  responsive: true, displayModeBar: true, scrollZoom: true,
  modeBarButtonsToRemove: ['select2d','lasso2d','autoScale2d'],
  doubleClick: 'reset',
}});

// ── Corner labels ─────────────────────────────────────────────────────────
function setCorners(corners) {{
  // Corners are defined in display space — no swapping needed
  const map = {{}};
  corners.forEach(c => map[c.pos] = c);
  const ids = {{tl:'cornerTL', tr:'cornerTR', bl:'cornerBL', br:'cornerBR'}};
  for (const [pos, elId] of Object.entries(ids)) {{
    const el = document.getElementById(elId);
    if (map[pos]) {{
      el.innerHTML = map[pos].text;
      el.style.color = map[pos].color;
    }} else {{
      el.innerHTML = '';
    }}
  }}
}}
setCorners(currentCorners);

// ── Auto-range helper ─────────────────────────────────────────────────────
function computeDataRange(dataObj, varName, invert) {{
  let mn = Infinity, mx = -Infinity;
  for (const g of Object.keys(dataObj)) {{
    const arr = dataObj[g][varName];
    if (!arr) continue;
    for (const v of arr) {{
      const val = invert ? -v : v;
      if (val < mn) mn = val;
      if (val > mx) mx = val;
    }}
  }}
  if (mn === Infinity) return [-1, 1];
  const pad = (mx - mn) * 0.08;
  return [mn - pad, mx + pad];
}}

// ── Description panel ─────────────────────────────────────────────────────
const descDiv = document.getElementById('axisDesc');

function updateDesc(preset, xVar, yVar) {{
  if (preset && preset.desc) {{
    const d = preset.desc;
    const qs = preset.corners;
    // Build quadrant map (data positions, not display — desc is always written for data coords)
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
    // Custom axes: just show full variable descriptions
    const xl = varLabels[xVar] || xVar;
    const yl = varLabels[yVar] || yVar;
    descDiv.innerHTML = `
      <div class="desc-axes" style="grid-template-columns:1fr 1fr">
        <div class="desc-ax"><b>Axe X — ${{xVar}}</b><br>${{xl}}</div>
        <div class="desc-ax"><b>Axe Y — ${{yVar}}</b><br>${{yl}}</div>
      </div>`;
  }}
}}

// ── Apply axes ────────────────────────────────────────────────────────────
function applyAxes(xVar, xInvert, yVar, preset) {{
  currentXVar = xVar;
  currentYVar = yVar;
  currentXInvert = xInvert;
  if (preset) {{
    currentXRange = preset.xRange.slice();
    currentYRange = preset.yRange.slice();
    currentCorners = preset.corners;
  }} else {{
    currentXRange = computeDataRange(groupDataX, xVar, xInvert);
    currentYRange = computeDataRange(groupDataY, yVar, false);
  }}

  // Update group traces (indices 1..N)
  btns.forEach((b, bi) => {{
    const traceIdx = b.traceIdx;
    const gx = groupDataX[b.key] && groupDataX[b.key][xVar] ? groupDataX[b.key][xVar] : [];
    const gy = groupDataY[b.key] && groupDataY[b.key][yVar] ? groupDataY[b.key][yVar] : [];
    const xVals = xInvert ? gx.map(v => -v) : gx;
    Plotly.restyle(chartDiv, {{x: [xVals], y: [gy]}}, [traceIdx]);
  }});

  // Update barycentre trace (index 0)
  const baryX = [];
  const baryY = [];
  ORDER.forEach(g => {{
    if (!baryMeans[g]) return;
    const xm = baryMeans[g][xVar] !== undefined ? baryMeans[g][xVar] : 0;
    const ym = baryMeans[g][yVar] !== undefined ? baryMeans[g][yVar] : 0;
    baryX.push(xInvert ? -xm : xm);
    baryY.push(ym);
  }});
  Plotly.restyle(chartDiv, {{x: [baryX], y: [baryY]}}, [0]);

  // Update axes
  const xTitle = preset ? preset.xTitle : (varLabels[xVar] || xVar) + (xInvert ? ' (inversé)' : '');
  const yTitle = preset ? preset.yTitle : (varLabels[yVar] || yVar);
  Plotly.relayout(chartDiv, {{
    'xaxis.title.text': xTitle,
    'yaxis.title.text': yTitle,
    'xaxis.range': currentXRange.slice(),
    'yaxis.range': currentYRange.slice(),
  }});

  // Always update corners (empty array clears them if no preset)
  setCorners(preset ? preset.corners : []);

  // Update description panel
  updateDesc(preset, xVar, yVar);

  // Update x invert checkbox
  document.getElementById('xInvertChk').checked = xInvert;
}}

// ── Preset buttons ────────────────────────────────────────────────────────
const presetBtnsDiv = document.getElementById('presetBtns');
PRESETS.forEach(p => {{
  const btn = document.createElement('button');
  btn.className = 'preset-btn' + (p.id === 'saint_graal' ? ' active' : '');
  btn.dataset.id = p.id;
  btn.textContent = p.emoji + ' ' + p.label;
  btn.addEventListener('click', () => {{
    document.querySelectorAll('.preset-btn').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
    currentPresetId = p.id;
    // Update selects
    document.getElementById('xSelect').value = p.xVar;
    document.getElementById('ySelect').value = p.yVar;
    document.getElementById('xInvertChk').checked = p.xInvert;
    applyAxes(p.xVar, p.xInvert, p.yVar, p);
  }});
  presetBtnsDiv.appendChild(btn);
}});

// ── Custom toggle ─────────────────────────────────────────────────────────
const customToggle = document.getElementById('customToggle');
const customPanel = document.getElementById('customPanel');
customToggle.addEventListener('click', () => {{
  const open = customPanel.classList.toggle('open');
  customToggle.classList.toggle('open', open);
  customToggle.textContent = open ? 'Personnaliser ▴' : 'Personnaliser ▾';
}});

// ── Build variable selects with optgroups ─────────────────────────────────
function buildSelect(selectId, selectedVar) {{
  const sel = document.getElementById(selectId);
  sel.innerHTML = '';
  for (const [cat, vars] of Object.entries(VARS)) {{
    const og = document.createElement('optgroup');
    og.label = cat;
    vars.forEach(v => {{
      const opt = document.createElement('option');
      opt.value = v;
      opt.textContent = v + (varLabels[v] ? ' — ' + varLabels[v].substring(0, 50) : '');
      if (v === selectedVar) opt.selected = true;
      og.appendChild(opt);
    }});
    sel.appendChild(og);
  }}
}}

buildSelect('xSelect', 'score_exploitation');
buildSelect('ySelect', 'score_domination');
document.getElementById('xInvertChk').checked = true;

function onCustomChange() {{
  const xVar = document.getElementById('xSelect').value;
  const yVar = document.getElementById('ySelect').value;
  const xInvert = document.getElementById('xInvertChk').checked;
  // Deactivate presets
  document.querySelectorAll('.preset-btn').forEach(b => b.classList.remove('active'));
  currentPresetId = null;
  // Find matching preset for range
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

// ── Group filter buttons ──────────────────────────────────────────────────
const activeGroups = new Set(btns.map(b => b.key));
const filtersDiv = document.getElementById('groupFilters');
const toggleAllBtn = document.getElementById('toggleAllBtn');
let allOn = true;

function updateToggleLabel() {{
  allOn = activeGroups.size === btns.length;
  toggleAllBtn.textContent = allOn ? 'Tout décocher' : 'Tout cocher';
}}

btns.forEach(b => {{
  const el = document.createElement('button');
  el.className = 'grp-btn on';
  el.dataset.key = b.key;
  el.style.borderColor = b.color;
  el.style.backgroundColor = b.color;
  el.innerHTML = b.short + ' <span style="font-size:8px;font-weight:400;opacity:0.7">' + b.count + '</span>';
  el.addEventListener('click', () => {{
    if (activeGroups.has(b.key)) {{
      activeGroups.delete(b.key);
      el.classList.replace('on','off');
      el.style.backgroundColor = 'transparent';
      el.style.color = b.color;
      Plotly.restyle(chartDiv, {{visible: false}}, [b.traceIdx]);
    }} else {{
      activeGroups.add(b.key);
      el.classList.replace('off','on');
      el.style.backgroundColor = b.color;
      el.style.color = '#fff';
      Plotly.restyle(chartDiv, {{visible: true}}, [b.traceIdx]);
    }}
    updateToggleLabel();
  }});
  filtersDiv.appendChild(el);
}});

toggleAllBtn.addEventListener('click', () => {{
  if (allOn) {{
    btns.forEach(b => {{
      if (activeGroups.has(b.key)) {{
        activeGroups.delete(b.key);
        const el = filtersDiv.querySelector('[data-key="' + b.key + '"]');
        el.classList.replace('on','off');
        el.style.backgroundColor = 'transparent';
        el.style.color = b.color;
        Plotly.restyle(chartDiv, {{visible: false}}, [b.traceIdx]);
      }}
    }});
  }} else {{
    btns.forEach(b => {{
      if (!activeGroups.has(b.key)) {{
        activeGroups.add(b.key);
        const el = filtersDiv.querySelector('[data-key="' + b.key + '"]');
        el.classList.replace('off','on');
        el.style.backgroundColor = b.color;
        el.style.color = '#fff';
        Plotly.restyle(chartDiv, {{visible: true}}, [b.traceIdx]);
      }}
    }});
  }}
  updateToggleLabel();
}});

// ── Legend ────────────────────────────────────────────────────────────────
const legendDiv = document.getElementById('legendArea');
btns.forEach(b => {{
  const item = document.createElement('div');
  item.className = 'legend-item';
  item.innerHTML = '<div class="legend-dot" style="background:' + b.color + '"></div><span>' + b.label + ' (' + b.count + ')</span>';
  item.addEventListener('click', () => {{
    const el = filtersDiv.querySelector('[data-key="' + b.key + '"]');
    if (el) el.click();
  }});
  legendDiv.appendChild(item);
}});

// ── Info card ──────────────────────────────────────────────────────────────
// customdata layout: [nivvie_median_diff, député, libelle_pcs, anciennete,
//                     ville_naissance, pays_naissance, circo_label, age,
//                     scoreParticipation, scoreLoyaute]
function showDesktopCard(traceData, pointIdx) {{
  const cd = traceData.customdata[pointIdx];
  if (!cd) return;
  const card = document.getElementById('infoCard');
  card.classList.remove('empty');

  const xRaw = traceData.x[pointIdx];
  const yRaw = traceData.y[pointIdx];
  const xLabel = varLabels[currentXVar] ? varLabels[currentXVar].substring(0, 40) : currentXVar;
  const yLabel = varLabels[currentYVar] ? varLabels[currentYVar].substring(0, 40) : currentYVar;

  const pays = cd[5] && cd[5] !== 'France' ? ' (' + cd[5] + ')' : '';
  const revStr = cd[0] !== '' && cd[0] !== null && !isNaN(Number(cd[0]))
    ? Number(cd[0]).toLocaleString('fr-FR', {{maximumFractionDigits:1}})
    : '—';
  const partic = cd[8] !== '' && cd[8] !== null ? Math.round(Number(cd[8]) * 100) + '%' : '—';
  const loyal  = cd[9] !== '' && cd[9] !== null ? Math.round(Number(cd[9]) * 100) + '%' : '—';

  card.innerHTML = `
    <div class="name">${{cd[1] || ''}}</div>
    <div class="party" style="color:${{traceData.marker.color || '#888'}}">${{traceData.name || ''}}</div>
    <div class="row"><span class="lbl">Profession :</span> <b>${{cd[2] || '—'}}</b></div>
    <div class="row"><span class="lbl">Circo :</span> <b>${{cd[6] || '—'}}</b></div>
    <div class="row"><span class="lbl">Né·e à :</span> ${{(cd[4] || '') + pays}}</div>
    <div class="row"><span class="lbl">Âge :</span> ${{cd[7] ? cd[7] + ' ans' : '—'}}</div>
    <div class="row"><span class="lbl">Ancienneté :</span> ${{cd[3] !== '' ? Number(cd[3]).toFixed(1) + ' ans' : '—'}}</div>
    <div class="row"><span class="lbl">Participation :</span> <b>${{partic}}</b> · <span class="lbl">Loyauté :</span> <b>${{loyal}}</b></div>
    <div class="row"><span class="lbl">Revenu circo :</span> <b>${{revStr}}</b></div>
    <div class="dynamic-row"><span class="lbl">Axe X (${{currentXVar}}) :</span> <b>${{xRaw !== undefined && xRaw !== null ? Number(xRaw).toFixed(2) : '—'}}</b></div>
    <div class="dynamic-row"><span class="lbl">Axe Y (${{currentYVar}}) :</span> <b>${{yRaw !== undefined && yRaw !== null ? Number(yRaw).toFixed(2) : '—'}}</b></div>
  `;
}}

chartDiv.on('plotly_click', function(eventData) {{
  if (!eventData || !eventData.points || eventData.points.length === 0) return;
  const pt = eventData.points[0];
  if (pt.curveNumber === 0) return; // barycentre
  const traceData = chartDiv.data[pt.curveNumber];
  showDesktopCard(traceData, pt.pointIndex);
}});

// ── Initial corners + description ────────────────────────────────────────
setCorners(PRESETS[0].corners);
updateDesc(PRESETS[0], PRESETS[0].xVar, PRESETS[0].yVar);

</script>
</body>
</html>"""
    return html


# ── 7. BUILD MOBILE HTML ──────────────────────────────────────────────────────
def build_mobile_html():
    gd_x_js, gd_y_js, bm_js, presets_js, vars_js, vl_js = _build_js_data()
    traces, trace_group_map = _build_trace_data()

    buttons_data = []
    for g in ORDER:
        if g not in trace_group_map:
            continue
        buttons_data.append({
            'key': g, 'short': SHORT.get(g, g), 'label': LABELS.get(g, g),
            'color': COULEURS.get(g, '#999'), 'traceIdx': trace_group_map[g],
            'count': int(df[df['groupe_politique'] == g].shape[0]),
        })

    fig = go.Figure()
    fig.add_vline(x=0, line_dash="dot", line_color="#CCC", line_width=1)
    fig.add_hline(y=0, line_dash="dot", line_color="#CCC", line_width=1)
    for tr in traces:
        # Scale marker size for mobile
        if hasattr(tr.marker, 'size') and isinstance(tr.marker.size, list):
            tr.marker.size = [s * 0.75 for s in tr.marker.size]
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
            title=dict(text="← Exploité ── Score exploitation (inv.) ── Exploiteur →",
                       font=dict(size=8.5, color="#888", family="Helvetica Neue, sans-serif"), standoff=4)),
        yaxis=dict(
            range=AXIS_PRESETS[0]['yRange'],
            showgrid=True, gridcolor="#EBEBEB", gridwidth=0.5, zeroline=False,
            tickfont=dict(size=8, color="#AAA"), linecolor="#DDD", fixedrange=False,
            title=dict(text="← Dominé ── Domination ── Dominant →",
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
<title>L'Assemblée des classes — Mobile</title>
<style>
* {{ margin: 0; padding: 0; box-sizing: border-box; }}
html, body {{ background: #FAF9F7; font-family: 'Helvetica Neue', system-ui, sans-serif;
               color: #1a1a1a; overflow-x: hidden; -webkit-text-size-adjust: 100%; }}
.header {{ text-align: center; padding: 12px 12px 4px; }}
.header h1 {{ font-size: 17px; font-weight: 900; letter-spacing: -0.5px; margin-bottom: 2px; }}
.header p {{ font-size: 9px; color: #888; line-height: 1.4; }}

/* Axis bar mobile */
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
              backdrop-filter: blur(8px); -webkit-backdrop-filter: blur(8px); }}
.info-card.show {{ transform: translateY(0); }}
.info-card .name {{ font-size: 14px; font-weight: 900; }}
.info-card .party {{ font-weight: 700; font-size: 10px; margin-bottom: 4px; }}
.info-card .row {{ color: #555; }}
.info-card .row b {{ color: #1a1a1a; }}
.info-card .lbl {{ color: #999; }}
.info-card .close {{ position: absolute; top: 8px; right: 14px; font-size: 22px; color: #BBB;
                     cursor: pointer; -webkit-tap-highlight-color: transparent; padding: 4px 8px; }}
.info-card .dynamic-row {{ color: #555; border-top: 1px solid #F0F0F0; padding-top: 4px; margin-top: 2px; }}
.footer {{ text-align: center; padding: 6px 12px 4px; font-size: 8px; color: #AAA; line-height: 1.6; }}

/* Axis description panel mobile */
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
</style>
</head>
<body>

<div class="header">
  <h1>L'Assemblée des classes</h1>
  <p>Sociologie des {len(df)} députés · XVIIe législature</p>
</div>

<div class="axis-bar">
  <div class="axis-bar-label">Axes :</div>
  <div class="preset-btns" id="presetBtns"></div>
  <button class="custom-toggle" id="customToggle">Personnaliser ▾</button>
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

<button class="toggle-all" id="toggleAll">Tout décocher</button>
<div class="filters" id="filters"></div>

<div id="chartWrap">
  <div id="chart"></div>
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
  <div class="row"><span class="lbl">Profession :</span> <b id="cardProf"></b></div>
  <div class="row"><span class="lbl">Circo :</span> <b id="cardCirco"></b></div>
  <div class="row"><span class="lbl">Né·e à :</span> <span id="cardVille"></span></div>
  <div class="row"><span class="lbl">Âge :</span> <span id="cardAge"></span></div>
  <div class="row"><span class="lbl">Ancienneté :</span> <span id="cardAnc"></span></div>
  <div class="row"><span class="lbl">Participation :</span> <b id="cardPartic"></b>&ensp;·&ensp;<span class="lbl">Loyauté :</span> <b id="cardLoyaute"></b></div>
  <div class="row"><span class="lbl">Revenu circo :</span> <b id="cardRev"></b></div>
  <div class="dynamic-row"><span class="lbl">Axe X (<span id="cardXVar"></span>) :</span> <b id="cardXVal"></b></div>
  <div class="dynamic-row"><span class="lbl">Axe Y (<span id="cardYVar"></span>) :</span> <b id="cardYVal"></b></div>
</div>

<div class="footer">⊕ = barycentre · taille = revenu circo · N={len(df)}</div>
<div class="axis-desc" id="axisDesc"></div>

<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<script>
const figData = {fig_json};
const groupDataX = {gd_x_js};
const groupDataY = {gd_y_js};
const baryMeans  = {bm_js};
const PRESETS    = {presets_js};
const VARS       = {vars_js};
const varLabels  = {vl_js};
const btns       = {_json.dumps(buttons_data)};
const ORDER_ARR  = {_json.dumps(ORDER)};
const INIT_X     = PRESETS[0].xRange.slice();
const INIT_Y     = PRESETS[0].yRange.slice();

let currentXVar = 'score_exploitation';
let currentYVar = 'score_domination';
let currentXInvert = true;
let currentPresetId = 'saint_graal';
let currentXRange = INIT_X.slice();
let currentYRange = INIT_Y.slice();

const chartDiv = document.getElementById('chart');
Plotly.newPlot(chartDiv, figData.data, figData.layout, {{
  responsive: true, displayModeBar: false, scrollZoom: false,
  doubleClick: false, staticPlot: false,
}});

// ── Corner labels ─────────────────────────────────────────────────────────
function setCorners(corners) {{
  const map = {{}};
  corners.forEach(c => map[c.pos] = c);
  const ids = {{tl:'cornerTL', tr:'cornerTR', bl:'cornerBL', br:'cornerBR'}};
  for (const [pos, elId] of Object.entries(ids)) {{
    const el = document.getElementById(elId);
    if (map[pos]) {{
      el.innerHTML = map[pos].text;
      el.style.color = map[pos].color;
    }} else {{
      el.innerHTML = '';
    }}
  }}
}}
setCorners(PRESETS[0].corners);

// ── Auto-range helper ─────────────────────────────────────────────────────
function computeDataRange(dataObj, varName, invert) {{
  let mn = Infinity, mx = -Infinity;
  for (const g of Object.keys(dataObj)) {{
    const arr = dataObj[g][varName];
    if (!arr) continue;
    for (const v of arr) {{
      const val = invert ? -v : v;
      if (val < mn) mn = val;
      if (val > mx) mx = val;
    }}
  }}
  if (mn === Infinity) return [-1, 1];
  const pad = (mx - mn) * 0.08;
  return [mn - pad, mx + pad];
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

// ── Apply axes ────────────────────────────────────────────────────────────
function applyAxes(xVar, xInvert, yVar, preset) {{
  currentXVar = xVar;
  currentYVar = yVar;
  currentXInvert = xInvert;
  if (preset) {{
    currentXRange = preset.xRange.slice();
    currentYRange = preset.yRange.slice();
  }} else {{
    currentXRange = computeDataRange(groupDataX, xVar, xInvert);
    currentYRange = computeDataRange(groupDataY, yVar, false);
  }}

  btns.forEach(b => {{
    const gx = groupDataX[b.key] && groupDataX[b.key][xVar] ? groupDataX[b.key][xVar] : [];
    const gy = groupDataY[b.key] && groupDataY[b.key][yVar] ? groupDataY[b.key][yVar] : [];
    const xVals = xInvert ? gx.map(v => -v) : gx;
    Plotly.restyle(chartDiv, {{x: [xVals], y: [gy]}}, [b.traceIdx]);
  }});

  const baryX = [], baryY = [];
  ORDER_ARR.forEach(g => {{
    if (!baryMeans[g]) return;
    const xm = baryMeans[g][xVar] !== undefined ? baryMeans[g][xVar] : 0;
    const ym = baryMeans[g][yVar] !== undefined ? baryMeans[g][yVar] : 0;
    baryX.push(xInvert ? -xm : xm);
    baryY.push(ym);
  }});
  Plotly.restyle(chartDiv, {{x: [baryX], y: [baryY]}}, [0]);

  const xLabel = varLabels[xVar] ? varLabels[xVar].substring(0, 55) : xVar;
  const yLabel = varLabels[yVar] ? varLabels[yVar].substring(0, 55) : yVar;
  const xTitle = preset ? preset.xTitle.substring(0, 60) : xLabel + (xInvert ? ' (inversé)' : '');
  const yTitle = preset ? preset.yTitle.substring(0, 60) : yLabel;
  Plotly.relayout(chartDiv, {{
    'xaxis.title.text': xTitle,
    'yaxis.title.text': yTitle,
    'xaxis.range': currentXRange.slice(),
    'yaxis.range': currentYRange.slice(),
  }});

  // Always update corners (empty array clears them when no preset)
  setCorners(preset ? preset.corners : []);
  updateDesc(preset, xVar, yVar);
  document.getElementById('xInvertChk').checked = xInvert;
}}

// ── Preset buttons ────────────────────────────────────────────────────────
const presetBtnsDiv = document.getElementById('presetBtns');
PRESETS.forEach(p => {{
  const btn = document.createElement('button');
  btn.className = 'preset-btn' + (p.id === 'saint_graal' ? ' active' : '');
  btn.dataset.id = p.id;
  btn.textContent = p.emoji + ' ' + p.label;
  btn.addEventListener('click', () => {{
    document.querySelectorAll('.preset-btn').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
    document.getElementById('xSelect').value = p.xVar;
    document.getElementById('ySelect').value = p.yVar;
    document.getElementById('xInvertChk').checked = p.xInvert;
    applyAxes(p.xVar, p.xInvert, p.yVar, p);
  }});
  presetBtnsDiv.appendChild(btn);
}});

const customToggle = document.getElementById('customToggle');
const customPanel = document.getElementById('customPanel');
customToggle.addEventListener('click', () => {{
  const open = customPanel.classList.toggle('open');
  customToggle.textContent = open ? 'Personnaliser ▴' : 'Personnaliser ▾';
}});

function buildSelect(selectId, selectedVar) {{
  const sel = document.getElementById(selectId);
  sel.innerHTML = '';
  for (const [cat, vars] of Object.entries(VARS)) {{
    const og = document.createElement('optgroup');
    og.label = cat;
    vars.forEach(v => {{
      const opt = document.createElement('option');
      opt.value = v;
      opt.textContent = v + (varLabels[v] ? ' — ' + varLabels[v].substring(0, 45) : '');
      if (v === selectedVar) opt.selected = true;
      og.appendChild(opt);
    }});
    sel.appendChild(og);
  }}
}}

buildSelect('xSelect', 'score_exploitation');
buildSelect('ySelect', 'score_domination');
document.getElementById('xInvertChk').checked = true;

function onCustomChange() {{
  const xVar = document.getElementById('xSelect').value;
  const yVar = document.getElementById('ySelect').value;
  const xInvert = document.getElementById('xInvertChk').checked;
  document.querySelectorAll('.preset-btn').forEach(b => b.classList.remove('active'));
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

// ── Reset zoom button ─────────────────────────────────────────────────────
const resetBtn = document.getElementById('resetBtn');
function checkZoomed() {{
  const xr = chartDiv.layout.xaxis.range;
  const yr = chartDiv.layout.yaxis.range;
  const zoomed = Math.abs(xr[0]-currentXRange[0])>0.05 || Math.abs(xr[1]-currentXRange[1])>0.05 ||
                 Math.abs(yr[0]-currentYRange[0])>0.05 || Math.abs(yr[1]-currentYRange[1])>0.05;
  resetBtn.classList.toggle('show', zoomed);
}}
chartDiv.on('plotly_relayout', checkZoomed);
resetBtn.addEventListener('click', () => {{
  Plotly.relayout(chartDiv, {{'xaxis.range': currentXRange.slice(), 'yaxis.range': currentYRange.slice()}});
}});

// ── Group filter buttons ──────────────────────────────────────────────────
const activeGroups = new Set(btns.map(b => b.key));
const filtersDiv = document.getElementById('filters');
const toggleAllBtn = document.getElementById('toggleAll');
let allOn = true;

function updateToggleLabel() {{
  allOn = activeGroups.size === btns.length;
  toggleAllBtn.textContent = allOn ? 'Tout décocher' : 'Tout cocher';
}}

toggleAllBtn.addEventListener('click', () => {{
  if (allOn) {{
    btns.forEach(b => {{
      if (activeGroups.has(b.key)) {{
        activeGroups.delete(b.key);
        const el = filtersDiv.querySelector('[data-key="' + b.key + '"]');
        el.classList.replace('on','off');
        el.style.backgroundColor = 'transparent';
        el.style.color = b.color;
        Plotly.restyle(chartDiv, {{visible: false}}, [b.traceIdx]);
      }}
    }});
  }} else {{
    btns.forEach(b => {{
      if (!activeGroups.has(b.key)) {{
        activeGroups.add(b.key);
        const el = filtersDiv.querySelector('[data-key="' + b.key + '"]');
        el.classList.replace('off','on');
        el.style.backgroundColor = b.color;
        el.style.color = '#fff';
        Plotly.restyle(chartDiv, {{visible: true}}, [b.traceIdx]);
      }}
    }});
  }}
  updateToggleLabel();
}});

btns.forEach(b => {{
  const el = document.createElement('button');
  el.className = 'fbtn on';
  el.dataset.key = b.key;
  el.style.borderColor = b.color;
  el.style.backgroundColor = b.color;
  el.innerHTML = b.short + ' <span style="font-size:7px;font-weight:400;opacity:0.7">' + b.count + '</span>';
  el.addEventListener('click', (e) => {{ e.preventDefault(); toggleGroup(b); updateToggleLabel(); }});
  filtersDiv.appendChild(el);
}});

function toggleGroup(b) {{
  const el = filtersDiv.querySelector('[data-key="' + b.key + '"]');
  if (activeGroups.has(b.key)) {{
    activeGroups.delete(b.key);
    el.classList.replace('on','off');
    el.style.backgroundColor = 'transparent';
    el.style.color = b.color;
    Plotly.restyle(chartDiv, {{visible: false}}, [b.traceIdx]);
  }} else {{
    activeGroups.add(b.key);
    el.classList.replace('off','on');
    el.style.backgroundColor = b.color;
    el.style.color = '#fff';
    Plotly.restyle(chartDiv, {{visible: true}}, [b.traceIdx]);
  }}
}}

// ── Info card ─────────────────────────────────────────────────────────────
const card = document.getElementById('infoCard');
const closeBtn = document.getElementById('closeCard');
closeBtn.addEventListener('click', () => card.classList.remove('show'));

function showCard(traceData, pointIdx) {{
  const cd = traceData.customdata[pointIdx];
  if (!cd) return;

  const xRaw = traceData.x[pointIdx];
  const yRaw = traceData.y[pointIdx];

  const pays = cd[5] && cd[5] !== 'France' ? ' (' + cd[5] + ')' : '';
  const revStr = cd[0] !== '' && cd[0] !== null && !isNaN(Number(cd[0]))
    ? Number(cd[0]).toLocaleString('fr-FR', {{maximumFractionDigits:1}})
    : '—';
  const partic = cd[8] !== '' && cd[8] !== null ? Math.round(Number(cd[8]) * 100) + '%' : '—';
  const loyal  = cd[9] !== '' && cd[9] !== null ? Math.round(Number(cd[9]) * 100) + '%' : '—';

  document.getElementById('cardName').textContent    = cd[1] || '';
  const partyEl = document.getElementById('cardParty');
  partyEl.textContent = traceData.name || '';
  partyEl.style.color = traceData.marker.color || '#888';
  document.getElementById('cardProf').textContent    = cd[2] || '—';
  document.getElementById('cardCirco').textContent   = cd[6] || '—';
  document.getElementById('cardVille').textContent   = (cd[4] || '') + pays;
  document.getElementById('cardAge').textContent     = cd[7] ? cd[7] + ' ans' : '—';
  document.getElementById('cardAnc').textContent     = cd[3] !== '' ? Number(cd[3]).toFixed(1) + ' ans' : '—';
  document.getElementById('cardPartic').textContent  = partic;
  document.getElementById('cardLoyaute').textContent = loyal;
  document.getElementById('cardRev').textContent     = revStr;
  document.getElementById('cardXVar').textContent    = currentXVar;
  document.getElementById('cardXVal').textContent    = xRaw !== undefined && xRaw !== null ? Number(xRaw).toFixed(2) : '—';
  document.getElementById('cardYVar').textContent    = currentYVar;
  document.getElementById('cardYVal').textContent    = yRaw !== undefined && yRaw !== null ? Number(yRaw).toFixed(2) : '—';
  card.classList.add('show');
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
    const newMidX = sr.x[0] + (newMidPxX - ax._offset) / plotW * xSpan0;
    const newMidY = sr.y[1] - (newMidPxY - ay._offset) / plotH * ySpan0;
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
        showCard(chartDiv.data[hit.ti], hit.pi);
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

// ── Initial description ────────────────────────────────────────────────
updateDesc(PRESETS[0], PRESETS[0].xVar, PRESETS[0].yVar);

</script>
</body>
</html>"""
    return html


# ── 8. OUTPUTS ────────────────────────────────────────────────────────────────
desktop_html = build_desktop_html()
with open("assemblee_des_classes_heritage.html", "w", encoding="utf-8") as f:
    f.write(desktop_html)
print(f"Desktop → assemblee_des_classes_heritage.html ({len(desktop_html)//1024} KB)")

mobile_html = build_mobile_html()
with open("assemblee_des_classes_heritage_mobile.html", "w", encoding="utf-8") as f:
    f.write(mobile_html)
print(f"Mobile  → assemblee_des_classes_heritage_mobile.html ({len(mobile_html)//1024} KB)")

fig_jupyter = build_jupyter_fig()
fig_jupyter.show()
print("Jupyter figure displayed.")
