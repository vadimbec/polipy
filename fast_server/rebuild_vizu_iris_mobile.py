import pandas as pd
import numpy as np
import plotly.graph_objects as go
import json as _json
import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

# ── CONFIGURATION ─────────────────────────────────────────────────────────────
COULEURS = {
    "RN":        "#374151",
    "LFI":       "#DC2626",
    "PS":        "#EC4899",
    "ENS":       "#F97316",
    "EELV":      "#16A34A",
    "PCF":       "#9B1C1C",
    "LR":        "#1D4ED8",
    "REC":       "#0F172A",
    "AUTRE":     "#9CA3AF",
}
LABELS = {
    "RN":            "Rassemblement National",
    "LFI":           "La France Insoumise",
    "PS":            "Parti Socialiste",
    "ENS":           "Ensemble / Renaissance",
    "EELV":          "Europe Écologie",
    "PCF":           "Parti Communiste",
    "LR":            "Les Républicains",
    "REC":           "Reconquête",
    "AUTRE":         "Autres partis",
    "NFP":           "Nouveau Front Populaire",
    "NUPES":         "NUPES",
    "PS_PP":         "PS-Place Publique (Glucksmann)",
    "UG":            "Union de la Gauche",
    "UXD":           "Alliance LR-RN (Ciotti)",
    "DVD":           "Divers droite",
    "DVC":           "Divers centre",
    "DVG":           "Divers gauche",
    "EXG":           "Extrême gauche",
    "EXD":           "Extrême droite",
    "DLF":           "Debout la France",
    "MODEM":         "MoDem",
    "HOR":           "Horizons",
    "UDI":           "UDI",
    "REG":           "Régionalistes",
    "MACRON":        "Emmanuel Macron",
    "LE_PEN":        "Marine Le Pen",
    "MELENCHON":     "Jean-Luc Mélenchon",
    "FILLON":        "François Fillon",
    "HAMON":         "Benoît Hamon",
    "DUPONT_AIGNAN": "Nicolas Dupont-Aignan",
    "ZEMMOUR":       "Éric Zemmour",
    "PECRESSE":      "Valérie Pécresse",
    "JADOT":         "Yannick Jadot",
    "ROUSSEL":       "Fabien Roussel",
    "HIDALGO":       "Anne Hidalgo",
    # Présidentielles 2012
    "HOLLANDE":      "François Hollande",
    "SARKOZY":       "Nicolas Sarkozy",
    "BAYROU":        "François Bayrou",
    "JOLY":          "Éva Joly",
}
SHORT = {
    "RN":            "RN",
    "LFI":           "LFI",
    "PS":            "PS",
    "ENS":           "ENS",
    "EELV":          "EELV",
    "PCF":           "PCF",
    "LR":            "LR",
    "REC":           "RCQ",
    "AUTRE":         "Autre",
    "NFP":           "NFP",
    "NUPES":         "NUPES",
    "PS_PP":         "PS-PP",
    "UG":            "UG",
    "UXD":           "UXD",
    "DVD":           "DVD",
    "DVC":           "DVC",
    "DVG":           "DVG",
    "EXG":           "EXG",
    "EXD":           "EXD",
    "DLF":           "DLF",
    "MODEM":         "MDM",
    "HOR":           "HOR",
    "UDI":           "UDI",
    "REG":           "REG",
    "MACRON":        "Macron",
    "LE_PEN":        "Le Pen",
    "MELENCHON":     "Mélenchon",
    "FILLON":        "Fillon",
    "HAMON":         "Hamon",
    "DUPONT_AIGNAN": "DPA",
    "ZEMMOUR":       "Zemmour",
    "PECRESSE":      "Pécresse",
    "JADOT":         "Jadot",
    "ROUSSEL":       "Roussel",
    "HIDALGO":       "Hidalgo",
    # Présidentielles 2012
    "HOLLANDE":      "Hollande",
    "SARKOZY":       "Sarkozy",
    "BAYROU":        "Bayrou",
    "JOLY":          "Joly",
}
# RN/REC/ZEMMOUR/LE_PEN rendered semi-transparent
OPACITY = {
    "RN": 0.50, "REC": 0.50, "LE_PEN": 0.50, "ZEMMOUR": 0.50,
    "EXD": 0.50, "UXD": 0.50,
}
ORDER = ["LFI","PCF","EELV","PS","ENS","LR","RN","REC","AUTRE"]
# ALL_ORDER : tous les partis/candidats possibles (pour créer une trace Plotly par parti)
ALL_ORDER = [
    "LFI","MELENCHON","PCF","ROUSSEL","EXG","EELV","JADOT","JOLY","PS","DVG","HAMON","HOLLANDE",
    "NFP","NUPES","PS_PP","UG",
    "ENS","MODEM","HOR","UDI","MACRON",
    "LR","DVD","DVC","FILLON","PECRESSE","SARKOZY","BAYROU","UXD",
    "RN","LE_PEN","REC","EXD","DLF","ZEMMOUR","DUPONT_AIGNAN",
    "REG",
    "HIDALGO",
    "AUTRE",
]

# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION — SCORES COMPOSITES
# Modifier ici les formules des scores. pos_vars = contribue positivement (signe +),
# neg_vars = contribue négativement (signe -). Chaque variable est normalisée par
# rang centile pondéré par population avant d'être combinée (plage résultat ≈ -50 à +50).
# ══════════════════════════════════════════════════════════════════════════════
SCORES_CONFIG = {    
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

# ── PRESETS D'AXES ────────────────────────────────────────────────────────────
AXIS_PRESETS = [
    {
        'id': 'saint_graphique',
        'label': 'Saint-Graphique',
        'emoji': '⚒️',
        'xVar': 'score_exploitation', 'xInvert': False,
        'yVar': 'score_domination',
        'xTitle': '← Exploité (prolétaire) ─── Position dans le rapport capital/travail ─── Exploiteur (bourgeois) →',
        'yTitle': '← Dominé ─── Domination sociale (Bourdieu) ─── Dominant →',
        'xRange': [-50, 50], 'yRange': [-50, 50],
        'corners': [
            {'pos': 'tl', 'text': 'ASCENSION<br>SOCIALE', 'color': '#E87070'},
            {'pos': 'tr', 'text': 'REPRODUCTION<br>DU CAPITAL', 'color': '#6B8FD4'},
            {'pos': 'bl', 'text': 'PROLÉTARIAT', 'color': '#60B87A'},
            {'pos': 'br', 'text': 'PETITE<br>BOURGEOISIE', 'color': '#C49A30'},
        ],
        'desc': {
            'title': 'Le Saint-Graphique — Marx × Bourdieu',
            'x': "<b>Axe X — Position dans le rapport capital/travail (Marx)</b> : d'où vient le revenu de l'IRIS ? À droite : zones <em>exploiteuses</em> (bourgeoisie, rentiers). À gauche : zones <em>exploitées</em> (prolétariat). Variables : % ouvriers, % employés, taux pauvreté, % sans diplôme.",
            'y': '<b>Axe Y — Domination sociale (Bourdieu)</b> : position dans la hiérarchie sociale totale (capital économique + culturel). En haut : dominants (riches, diplômés, cadres sup.). En bas : dominés.',
            'quadrants': {
                'tr': '<b>Reproduction du capital</b> — Zones exploiteuses ET dominantes : beaux quartiers, arrondissements bourgeois.',
                'tl': "<b>Ascension sociale</b> — Zones exploitées mais dominantes : cadres salariés issus de milieux populaires.",
                'bl': '<b>Prolétariat</b> — Zones exploitées ET dominées : quartiers ouvriers, banlieues populaires.',
                'br': '<b>Petite bourgeoisie</b> — Zones exploiteuses mais dominées : petits commerçants, artisans propriétaires.',
            }
        }
    },
    {
        'id': 'bourdieu',
        'label': 'Bourdieu',
        'emoji': '🎓',
        'xVar': 'score_cap_eco', 'xInvert': False,
        'yVar': 'score_cap_cult',
        'xTitle': '← Pauvre ─── Capital Economique ─── Riche →',
        'yTitle': '← Peu diplômé · ouvrier ─── Capital culturel ─── Diplômé · cadre →',
        'xRange': [-50, 50], 'yRange': [-50, 50],
        'corners': [
            {'pos': 'tl', 'text': 'PAUVRE<br>DIPLOME', 'color': '#E87070'},
            {'pos': 'tr', 'text': 'RICHE<br>DIPLOME', 'color': '#6B8FD4'},
            {'pos': 'bl', 'text': 'PAUVRE<br>NON DIPLOME', 'color': '#60B87A'},
            {'pos': 'br', 'text': 'RICHE<br>NON DIPLOME', 'color': '#C49A30'},
        ],
        'desc': {
            'title': 'Espace bourdieusien — Capital culturel × Domination',
            'x': '<b>Axe X — Capital culturel</b> : diplômes (BAC+5), % cadres sup. et professions intermédiaires. À droite : forte proportion de diplômés et cadres. À gauche : zones peu qualifiées.',
            'y': '<b>Axe Y — Domination sociale</b> : hiérarchie totale combinant capital économique et culturel.',
            'quadrants': {
                'tl': '<b>Noblesse du capital</b> — Riches mais peu diplômés : héritiers, propriétaires fonciers.',
                'tr': '<b>Élite intégrée</b> — Riches ET diplômés : grandes écoles, hauts fonctionnaires.',
                'bl': '<b>Classe populaire</b> — Pauvres ET peu diplômés : prolétariat classique.',
                'br': '<b>Intellectuels déclassés</b> — Diplômés mais peu riches : enseignants, chercheurs.',
            }
        }
    },
    {
        'id': 'rentier',
        'label': 'Rentier',
        'emoji': '💰',
        'xVar': 'score_rentier', 'xInvert': False,
        'yVar': 'score_domination',
        'xTitle': '← Revenu du travail (salaires) ─── Rentier vs Travailleur ─── Revenu du capital →',
        'yTitle': '← Dominé ─── Domination sociale (Bourdieu) ─── Dominant →',
        'xRange': [-50, 50], 'yRange': [-50, 50],
        'corners': [
            {'pos': 'tl', 'text': 'INTELLECTUELS<br>FONCTIONNAIRES', 'color': '#E87070'},
            {'pos': 'tr', 'text': 'ÉLITE<br>RENTIÈRE', 'color': '#6B8FD4'},
            {'pos': 'bl', 'text': 'SALARIÉS<br>PRÉCAIRES', 'color': '#60B87A'},
            {'pos': 'br', 'text': 'PETITS<br>RENTIERS', 'color': '#C49A30'},
        ],
        'desc': {
            'title': 'Rente vs Travail × Domination sociale',
            'x': '<b>Axe X — Score rentier</b> : part du revenu tirée du capital (patrimoine, pensions, retraites) vs travail salarié. À droite : les rentiers. À gauche : les travailleurs.',
            'y': '<b>Axe Y — Domination sociale</b> : même axe que le Saint-Graphique.',
            'quadrants': {
                'tl': '<b>Intellectuels / Fonctionnaires</b> — Dominants mais non-rentiers : hauts fonctionnaires, médecins.',
                'tr': '<b>Élite rentière</b> — Dominant ET rentier : vieille bourgeoisie, héritiers parisiens.',
                'bl': '<b>Salariés précaires</b> — Dominé ET non-rentier : classe laborieuse sans patrimoine.',
                'br': "<b>Petits rentiers</b> — Rentier mais pas dominant : retraités propriétaires provinciaux.",
            }
        }
    },
    {
        'id': 'demographie',
        'label': 'Démographie',
        'emoji': '👥',
        'xVar': 'pct_etrangers', 'xInvert': False,
        'yVar': 'age_moyen',
        'xTitle': '← Peu d\'étrangers ─── % population étrangère ─── Beaucoup d\'étrangers →',
        'yTitle': '← Jeune ─── Âge moyen ─── Vieux →',
        'xRange': [-1.0, 45.0], 'yRange': [25.0, 55.0],
        'corners': [
            {'pos': 'tl', 'text': 'VIEUX<br>NATIFS', 'color': '#6B7280'},
            {'pos': 'tr', 'text': 'VIEUX<br>IMMIGRÉS', 'color': '#9CA3AF'},
            {'pos': 'bl', 'text': 'JEUNES<br>NATIFS', 'color': '#3B82F6'},
            {'pos': 'br', 'text': 'JEUNES<br>IMMIGRÉS', 'color': '#EF4444'},
        ],
        'desc': {
            'title': 'Démographie — Âge × Origine',
            'x': '<b>Axe X — % population étrangère</b> : part des résidents de nationalité étrangère dans l\'IRIS.',
            'y': '<b>Axe Y — Âge moyen</b> : âge moyen de la population résidente de l\'IRIS.',
            'quadrants': {
                'tl': '<b>Vieux natifs</b> — Zones âgées à faible immigration : France rurale profonde, littoral retraité.',
                'tr': '<b>Vieux immigrés</b> — Zones avec une immigration ancienne et installée.',
                'bl': '<b>Jeunes natifs</b> — Zones jeunes peu diversifiées : périurbain récent, villes moyennes.',
                'br': '<b>Jeunes immigrés</b> — Zones à forte immigration récente : banlieues denses, zones industrielles.',
            }
        }
    },
    {
        'id': 'ruralite',
        'label': 'Ruralité × Précarité',
        'emoji': '🌾',
        'xVar': 'score_ruralite', 'xInvert': False,
        'yVar': 'score_precarite',
        'xTitle': '← Urbain ─── Score ruralité ─── Rural →',
        'yTitle': '← Sécurisé ─── Score précarité ─── Précaire →',
        'xRange': [-55.0, 55.0], 'yRange': [-55.0, 55.0],
        'corners': [
            {'pos': 'bl', 'text': 'URBAIN<br>SÉCURISÉ', 'color': '#6B8FD4'},
            {'pos': 'br', 'text': 'RURAL<br>SÉCURISÉ', 'color': '#059669'},
            {'pos': 'tl', 'text': 'URBAIN<br>PRÉCAIRE', 'color': '#E87070'},
            {'pos': 'tr', 'text': 'RURAL<br>PRÉCAIRE', 'color': '#C49A30'},
        ],
        'desc': {
            'title': 'Ruralité × Précarité sociale',
            'x': '<b>Axe X — Score ruralité</b> : profil agricole/retraité/voiture (droite = rural) vs tertiaire/diplômé/transports (gauche = urbain).',
            'y': '<b>Axe Y — Score précarité</b> : chômage, minimas sociaux, taux de pauvreté (haut = plus précaire).',
            'quadrants': {
                'tl': '<b>Urbain sécurisé</b> — Quartiers denses avec emploi stable : centre-ville, cadres, fonctionnaires.',
                'tr': '<b>Rural sécurisé</b> — Bourgs ruraux avec retraités et propriétaires fonciers aisés.',
                'bl': '<b>Urbain précaire</b> — Banlieues populaires denses : chômage élevé, minimas sociaux.',
                'br': '<b>Rural précaire</b> — Zones rurales défavorisées : désertification économique, faibles revenus.',
            }
        }
    },
    {
        'id': 'precarite_peripherie',
        'label': 'Fractures territoriales',
        'emoji': '🏘️',
        'xVar': 'score_peripherie_metropole', 'xInvert': False,
        'yVar': 'score_precarite',
        'xTitle': '← Metropole (Macron/Jadot) ─── Peripherie-Metropole ─── Peripherie (Le Pen) →',
        'yTitle': '← Aise ─── Precarite sociale ─── Precaire →',
        'xRange': [-50, 50], 'yRange': [-50, 50],
        'corners': [
            {'pos': 'tl', 'text': 'BANLIEUE<br>POPULAIRE', 'color': '#DC2626'},
            {'pos': 'tr', 'text': 'PERIPHERIE<br>PRECAIRE', 'color': '#374151'},
            {'pos': 'bl', 'text': 'METROPOLE<br>AISEE', 'color': '#F59E0B'},
            {'pos': 'br', 'text': 'PERIURBAIN<br>CONFORTABLE', 'color': '#3B82F6'},
        ],
        'desc': {
            'title': 'Fractures territoriales — Precarite x Peripherie-Metropole',
            'x': "<b>Axe X — Score peripherie-metropole</b> : construit a partir des variables qui separent le mieux les electorats Le Pen/Zemmour (peripherie) de ceux de Macron/Jadot (metropole). A droite : zones periurbaines (voiture, maison, CAP-BEP, fioul). A gauche : centres-villes connectes (BAC+5, cadres, velo, petits logements).",
            'y': "<b>Axe Y — Score precarite</b> : composite chomage, minimas sociaux, taux de pauvrete vs revenu median et patrimoine. En haut : zones precaires. En bas : zones aisees.",
            'quadrants': {
                'tr': '<b>Peripherie precaire</b> — Zones rurales ou periurbaines en difficulte economique. Terre d\'election du RN.',
                'tl': '<b>Banlieue populaire</b> — Grands ensembles et quartiers denses en difficulte. Fort vote LFI.',
                'br': '<b>Periurbain confortable</b> — Lotissements pavillonnaires aises. Vote droite traditionnelle ou RN modere.',
                'bl': '<b>Metropole aisee</b> — Centres-villes bourgeois et quartiers connectes. Vote Macron/EELV.',
            }
        }
    },
    {
        'id': 'acp',
        'label': 'ACP (PC1 x PC2)',
        'emoji': '📊',
        'xVar': 'score_pca_pc1_logement_confort', 'xInvert': False,
        'yVar': 'score_pca_pc2_composition_diplomes',
        'xTitle': '← Locatif dense ─── ACP PC1 : Logement & confort ─── Proprietaire pavillonnaire →',
        'yTitle': '← Diplome aise ─── ACP PC2 : Composition sociale ─── Ouvrier precaire →',
        'xRange': [-50, 50], 'yRange': [-50, 50],
        'corners': [
            {'pos': 'tl', 'text': 'HLM<br>PRECAIRE', 'color': '#DC2626'},
            {'pos': 'tr', 'text': 'PAVILLONNAIRE<br>OUVRIER', 'color': '#374151'},
            {'pos': 'bl', 'text': 'URBAIN<br>DIPLOME', 'color': '#F59E0B'},
            {'pos': 'br', 'text': 'RESIDENTIEL<br>BOURGEOIS', 'color': '#3B82F6'},
        ],
        'desc': {
            'title': 'Analyse en Composantes Principales — PC1 \u00d7 PC2',
            'x': "<b>Axe X — ACP PC1 : Logement & confort</b> (1\u00e8re composante). L'ACP cherche les axes de plus grande variabilit\u00e9 sans utiliser l'information \u00e9lectorale. PC1 r\u00e9sume l'opposition entre habitat pavillonnaire et habitat collectif dense. <br><b>+</b> grands logements, garage, voiture, maison, propri\u00e9taires. <br><b>\u2212</b> appartements, locataires, immigr\u00e9s, transports, studios.",
            'y': "<b>Axe Y — ACP PC2 : Composition sociale</b> (2\u00e8me composante, orthogonale \u00e0 PC1). S\u00e9pare zones ouvri\u00e8res pr\u00e9caires des zones de cadres dipl\u00f4m\u00e9s. <br><b>+</b> CAP-BEP, ch\u00f4mage, ouvriers, minimas sociaux, sans dipl\u00f4me. <br><b>\u2212</b> revenu m\u00e9dian, BAC+, cadres, BAC+5, revenus d'activit\u00e9.",
            'quadrants': {
                'tr': '<b>Pavillonnaire ouvrier</b> — Lotissements p\u00e9riurbains peu qualifi\u00e9s. Vote RN.',
                'tl': '<b>HLM pr\u00e9caire</b> — Grands ensembles avec forte pr\u00e9carit\u00e9. Vote LFI.',
                'br': '<b>R\u00e9sidentiel bourgeois</b> — Grandes propri\u00e9t\u00e9s, cadres sup\u00e9rieurs. Vote Macron.',
                'bl': '<b>Urbain dipl\u00f4m\u00e9</b> — Centres-villes, jeunes actifs. \u00c9lectorat EELV/Macron.',
            }
        }
    },
    {
        'id': 'tsne',
        'label': 't-SNE',
        'emoji': '🔬',
        'xVar': 'tsne_x', 'xInvert': False,
        'yVar': 'tsne_y',
        'xTitle': 't-SNE dimension 1',
        'yTitle': 't-SNE dimension 2',
        'xRange': None, 'yRange': None,
        'corners': [],
        'desc': {
            'title': 't-SNE \u2014 Carte des similarit\u00e9s entre IRIS',
            'x': "<b>Technique : t-SNE</b> (t-distributed Stochastic Neighbor Embedding) : projette les ~80 variables socio-\u00e9conomiques en 2D en pr\u00e9servant les <em>voisinages locaux</em>. Deux IRIS proches sur cette carte ont des profils sociologiques similaires.",
            'y': "<b>Comment lire</b> : les axes X et Y n'ont aucune signification propre. Ce qui compte, ce sont les <em>regroupements visuels</em>. Les couleurs r\u00e9v\u00e8lent comment le vote s'organise dans cet espace.",
            'quadrants': {}
        }
    },
    {
        'id': 'umap',
        'label': 'UMAP',
        'emoji': '🌐',
        'xVar': 'umap_x', 'xInvert': False,
        'yVar': 'umap_y',
        'xTitle': 'UMAP dimension 1',
        'yTitle': 'UMAP dimension 2',
        'xRange': None, 'yRange': None,
        'corners': [],
        'desc': {
            'title': 'UMAP \u2014 Topologie des territoires',
            'x': "<b>Technique : UMAP</b> (Uniform Manifold Approximation and Projection) : comme le t-SNE mais pr\u00e9serve aussi la structure <em>globale</em>. Deux amas \u00e9loign\u00e9s = territoires v\u00e9ritablement diff\u00e9rents.",
            'y': "<b>Comment lire</b> : les axes n'ont pas de signification propre. L'int\u00e9r\u00eat est dans la topologie : forme des amas, s\u00e9paration, ponts entre types de territoires.",
            'quadrants': {}
        }
    },
]

# Variables disponibles dans les dropdowns custom
VARS_BY_CAT = {
    'Scores composites': [
        'score_exploitation', 'score_domination', 'score_cap_eco', 'score_cap_cult',
        'score_precarite', 'score_rentier', 'score_ruralite',
        'score_urbanite', 'score_confort_residentiel', 'score_equipement_public',
        'score_pca_pc1_logement_confort', 'score_pca_pc2_composition_diplomes',
        'score_pca_pc3_equipements_demographie', 'score_pca_pc4_demographie_chauffage',
        'score_pca_pc5_equipements_csp', 'score_pca_pc6_equipements_diplomes',
        'score_pca_pc7_logement_csp', 'score_pca_pc8_equipements_logement',
        'score_peripherie_metropole',
    ],
    'Réductions dimensionnelles': [
        'tsne_x', 'tsne_y', 'umap_x', 'umap_y',
    ],
    'Démographie': [
        'pct_0_19', 'pct_20_64', 'pct_65_plus',
        'pct_etrangers', 'pct_immigres', 'age_moyen', ''
    ],
    'Revenus et inégalités': [
        'DISP_MED21', 'DISP_TP6021', 'DISP_GI21', 'DISP_S80S2021', 'DISP_RD21',
        'DISP_PPAT21', 'DISP_PPSOC21', 'DISP_PPMINI21', 'DISP_PACT21',
        'DISP_PPEN21', 'DISP_PBEN21', 'DISP_PPFAM21', 'DISP_PPLOGT21',
        'DISP_PCHO21', 'DISP_PTSA21',
        'DISP_D121', 'DISP_D221', 'DISP_D321', 'DISP_D421',
        'DISP_D621', 'DISP_D721', 'DISP_D821', 'DISP_D921',
        'DISP_Q121', 'DISP_Q321', 'DISP_PIMPOT21',
    ],
    'Diplômes et emploi': [
        'pct_sup5', 'pct_bac_plus', 'pct_sans_diplome', 'pct_capbep',
        'pct_chomage', 'pct_cdi', 'pct_cdd', 'pct_interim', 'pct_temps_partiel',
        'pct_inactif', 'pct_csp_agriculteur', 'pct_csp_independant', 'pct_csp_plus',
        'pct_csp_intermediaire', 'pct_csp_employe', 'pct_csp_ouvrier',
        'pct_csp_retraite', 'pct_csp_sans_emploi', 'pct_etudiants',
    ],
    'Mobilité': [
      'pct_actifs_voiture', 'pct_actifs_transports', 'pct_actifs_velo', 'pct_actifs_2roues',
    ],
}

VAR_LABELS = {
    'score_exploitation':       'Score exploitation (Marx) — part capital vs travail',
    'score_domination':         'Score domination (Bourdieu) — hiérarchie sociale totale',
    'score_cap_cult':           'Score capital culturel — diplômes + professions intellectuelles',
    'score_cap_eco':            'Score capital économique — richesse, patrimoine, revenus',
    'score_precarite':          'Score précarité sociale — chômage, minimas, pauvreté',
    'score_rentier':            'Score rentier — part patrimoine/capital dans les revenus',
    'score_ruralite':           'Score ruralité — profils agricoles/retraités vs urbains',
    'score_urbanite':           'Score urbanité — habitat dense, transports vs pavillonnaire, voiture',
    'score_confort_residentiel':'Score confort résidentiel — propriété, surface, garage vs HLM, suroccupation',
    'score_equipement_public':  'Score équipement public — densité santé, enseignement, commerces, services',
    'score_pca_pc1_logement_confort':       'Score ACP-PC1 — Logement & confort. <b>+</b> grands logements, garage, voiture, maison, propriétaires, surface. <b>−</b> appartements, locataires, sans-voiture, immigrés, transports, studios.',
    'score_pca_pc2_composition_diplomes':   'Score ACP-PC2 — Composition sociale. <b>+</b> CAP-BEP, intérim, chômage, ouvriers, taux pauvreté, minimas sociaux, sans diplôme. <b>−</b> revenu médian, BAC+, cadres, BAC+5, revenus d\'activité.',
    'score_pca_pc3_equipements_demographie':'Score ACP-PC3 — Équipements & démographie. <b>+</b> professions intermédiaires, revenus d\'activité, 0-19 ans. <b>−</b> 65+ ans, âge moyen, pensions, BPE total, retraités, commerces, tourisme, santé, marche, services.',
    'score_pca_pc4_demographie_chauffage':  'Score ACP-PC4 — Démographie & chauffage. <b>+</b> âge moyen, retraités, 65+, pensions, gaz de ville, femmes. <b>−</b> logements anciens, services, 20-64 ans, chauffage autre/bouteille/fioul, agriculteurs, BPE total, sport-culture.',
    'score_pca_pc5_equipements_csp':        'Score ACP-PC5 — Équipements & CSP. <b>+</b> grands logements, sans-emploi, inégalités S80/S20, temps partiel, inactifs, étudiants. <b>−</b> BPE total/services/commerces/santé, employés, chauffage élec, logements récents, prof. intermédiaires.',
    'score_pca_pc6_equipements_diplomes':   'Score ACP-PC6 — Éducation privée & diplômes. <b>+</b> inactifs, sport indoor, écoles privées, enseignement, hors-ménage, étudiants. <b>−</b> transports BPE, immigrés, étrangers, CDI, transports commun, 2-roues, indépendants, inégalités.',
    'score_pca_pc7_logement_csp':           'Score ACP-PC7 — Logement récent & inégalités. <b>+</b> inégalités S80/S20 et Gini, logements récents, services BPE, sans-emploi, inactifs, indépendants, 0-19 ans, patrimoine. <b>−</b> logements vacants/anciens, agriculteurs, 20-64 ans, vélo.',
    'score_pca_pc8_equipements_logement':   'Score ACP-PC8 — CDD & logements récents. <b>+</b> CDD, logements récents, chauffage élec. <b>−</b> sport indoor, enseignement, écoles privées, gaz de ville, sport-culture BPE, logements vacants, BPE total, santé.',
    'score_peripherie_metropole':           'Score périphérie-métropole — Périurbain voiture (Le Pen) vs métropole diplômée (Macron/Jadot)',
    'tsne_x':                               'Coordonnée t-SNE X (réduction non-linéaire)',
    'tsne_y':                               'Coordonnée t-SNE Y (réduction non-linéaire)',
    'umap_x':                               'Coordonnée UMAP X (réduction non-linéaire)',
    'umap_y':                               'Coordonnée UMAP Y (réduction non-linéaire)',
    'pct_csp_agriculteur':      '% agriculteurs exploitants (CSP 1)',
    'pct_csp_independant':      '% artisans, commerçants, chefs d\'entreprise (CSP 2)',
    'pct_csp_plus':             '% cadres et professions intellectuelles supérieures (CSP 3)',
    'pct_csp_intermediaire':    '% professions intermédiaires (CSP 4)',
    'pct_csp_employe':          '% employés (CSP 5)',
    'pct_csp_ouvrier':          '% ouvriers (CSP 6)',
    'pct_csp_retraite':         '% retraités (CSP 7)',
    'pct_csp_sans_emploi':      '% autres inactifs sans emploi (CSP 8)',
    'pct_0_19':                 '% population 0-19 ans',
    'pct_20_64':                '% population 20-64 ans',
    'pct_65_plus':              '% population 65 ans et plus',
    'pct_etrangers':            '% population de nationalité étrangère',
    'pct_immigres':             '% population immigrée',
    'age_moyen':                'Âge moyen de la population',
    'DISP_MED21':               'Revenu médian disponible (€/UC)',
    'DISP_TP6021':              'Taux de pauvreté (%)',
    'DISP_GI21':                'Indice de Gini (inégalités intra-IRIS)',
    'DISP_S80S2021':            'Ratio S80/S20 (inégalités)',
    'DISP_PPAT21':              '% revenus du patrimoine',
    'DISP_PPSOC21':             '% prestations sociales',
    'DISP_PPMINI21':            '% minimas sociaux (RSA, etc.)',
    'DISP_PACT21':              '% revenus d\'activité',
    'pct_etudiants':            '% étudiants parmi 15 ans et +',
    'pct_sup5':                 '% diplômés BAC+5 (parmi 15 ans et +)',
    'pct_bac_plus':             '% diplômés BAC et plus (parmi 15 ans et +)',
    'pct_sans_diplome':         '% sans diplôme (parmi 15 ans et +)',
    'pct_capbep':               '% diplômés CAP ou BEP',
    'pct_chomage':              '% chômeurs parmi actifs 15-64 ans',
    'pct_cdi':                  '% salariés en CDI',
    'pct_cdd':                  '% salariés en CDD',
    'pct_interim':              '% salariés en intérim',
    'pct_temps_partiel':        '% salariés à temps partiel',
    'pct_actifs_voiture':       '% actifs allant au travail en voiture',
    'pct_actifs_transports':    '% actifs allant au travail en transports en commun',
    'pct_actifs_velo':          '% actifs allant au travail en vélo',
    'pct_actifs_2roues':        '% actifs allant au travail en 2 roues motorisées',
    'pct_actifs_marche':        '% actifs allant au travail à pied',
    'DISP_PPEN21':              '% retraites et pensions dans le revenu disponible',
    'DISP_PBEN21':              '% bénéfices (revenus indépendants) dans le revenu',
    'DISP_PPFAM21':             '% prestations familiales dans le revenu',
    'DISP_PPLOGT21':            '% aides au logement dans le revenu',
    'DISP_PCHO21':              '% allocations chômage dans le revenu',
    'DISP_PTSA21':              'dont part des salaires et traitements dans le revenu (%)',
    'DISP_RD21':                'Rapport interdécile D9/D1 du revenu disponible',
    'DISP_EQ21':                'Écart interquartile rapporté à la médiane',
    'DISP_D121':                '1er décile du revenu disponible (€/UC)',
    'DISP_D221':                '2e décile du revenu disponible (€/UC)',
    'DISP_D321':                '3e décile du revenu disponible (€/UC)',
    'DISP_D421':                '4e décile du revenu disponible (€/UC)',
    'DISP_D621':                '6e décile du revenu disponible (€/UC)',
    'DISP_D721':                '7e décile du revenu disponible (€/UC)',
    'DISP_D821':                '8e décile du revenu disponible (€/UC)',
    'DISP_D921':                '9e décile du revenu disponible (€/UC)',
    'DISP_Q121':                '1er quartile du revenu disponible (€/UC)',
    'DISP_Q321':                '3e quartile du revenu disponible (€/UC)',
    'DISP_PIMPOT21':            'Part des impôts dans le revenu disponible (%)',
    # ── Population active et emploi (effectifs bruts) ──────────────────────────
    'P21_POP1564':              'Population 15-64 ans',
    'P21_POP1524':              'Population 15-24 ans',
    'P21_POP2554':              'Population 25-54 ans',
    'P21_POP5564':              'Population 55-64 ans',
    'P21_ACT1564':              'Actifs 15-64 ans',
    'P21_ACT1524':              'Actifs 15-24 ans',
    'P21_ACT2554':              'Actifs 25-54 ans',
    'P21_ACT5564':              'Actifs 55-64 ans',
    'P21_ACTOCC1564':           'Actifs occupés 15-64 ans',
    'P21_ACTOCC1524':           'Actifs occupés 15-24 ans',
    'P21_ACTOCC2554':           'Actifs occupés 25-54 ans',
    'P21_ACTOCC5564':           'Actifs occupés 55-64 ans',
    'P21_CHOM1564':             'Chômeurs 15-64 ans',
    'P21_CHOM1524':             'Chômeurs 15-24 ans',
    'P21_CHOM2554':             'Chômeurs 25-54 ans',
    'P21_CHOM5564':             'Chômeurs 55-64 ans',
    'P21_INACT1564':            'Inactifs 15-64 ans',
    'P21_ETUD1564':             'Élèves, étudiants et stagiaires non rémunérés 15-64 ans',
    'P21_RETR1564':             'Retraités ou préretraités 15-64 ans',
    'P21_AINACT1564':           'Autres inactifs 15-64 ans',
    # ── Actifs occupés par CSP (effectifs) ─────────────────────────────────────
    'C21_ACT1564':              'Actifs 15-64 ans (recensement)',
    'C21_ACT1564_CS1':          'Agriculteurs exploitants actifs 15-64 ans',
    'C21_ACT1564_CS2':          'Artisans, commerçants, chefs d\'entreprise actifs 15-64 ans',
    'C21_ACT1564_CS3':          'Cadres et professions intellectuelles supérieures actifs 15-64 ans',
    'C21_ACT1564_CS4':          'Professions intermédiaires actifs 15-64 ans',
    'C21_ACT1564_CS5':          'Employés actifs 15-64 ans',
    'C21_ACT1564_CS6':          'Ouvriers actifs 15-64 ans',
    'C21_ACTOCC1564':           'Actifs occupés 15-64 ans (recensement)',
    'C21_ACTOCC1564_CS1':       'Agriculteurs exploitants actifs occupés 15-64 ans',
    'C21_ACTOCC1564_CS2':       'Artisans, commerçants, chefs d\'entreprise actifs occupés 15-64 ans',
    'C21_ACTOCC1564_CS3':       'Cadres, professions intellectuelles supérieures actifs occupés 15-64 ans',
    'C21_ACTOCC1564_CS4':       'Professions intermédiaires actifs occupés 15-64 ans',
    'C21_ACTOCC1564_CS5':       'Employés actifs occupés 15-64 ans',
    'C21_ACTOCC1564_CS6':       'Ouvriers actifs occupés 15-64 ans',
    # ── Salariat et types de contrats (effectifs) ──────────────────────────────
    'P21_ACTOCC15P':            'Actifs occupés 15 ans ou plus',
    'P21_SAL15P':               'Salariés 15 ans ou plus',
    'P21_NSAL15P':              'Non-salariés 15 ans ou plus',
    'P21_ACTOCC15P_TP':         'Actifs occupés à temps partiel 15 ans ou plus',
    'P21_SAL15P_TP':            'Salariés à temps partiel 15 ans ou plus',
    'P21_SAL15P_CDI':           'Salariés en CDI ou fonction publique 15 ans ou plus',
    'P21_SAL15P_CDD':           'Salariés en CDD 15 ans ou plus',
    'P21_SAL15P_INTERIM':       'Salariés intérimaires 15 ans ou plus',
    'P21_SAL15P_EMPAID':        'Salariés en emploi aidé 15 ans ou plus',
    'P21_SAL15P_APPR':          'Salariés en apprentissage ou stagiaires 15 ans ou plus',
    'P21_NSAL15P_INDEP':        'Non-salariés indépendants 15 ans ou plus',
    'P21_NSAL15P_EMPLOY':       'Non-salariés employeurs 15 ans ou plus',
    'P21_NSAL15P_AIDFAM':       'Aides familiaux (non-salariés) 15 ans ou plus',
    # ── Mobilité domicile-travail ───────────────────────────────────────────────
    'P21_ACTOCC15P_ILT1':       'Actifs occupés travaillant dans leur commune de résidence',
    'P21_ACTOCC15P_ILT2P':      'Actifs occupés travaillant dans une autre commune',
    'P21_ACTOCC15P_ILT2':       'Actifs occupés travaillant dans un autre commune du même département',
    'P21_ACTOCC15P_ILT3':       'Actifs occupés travaillant dans un autre département de la même région',
    'P21_ACTOCC15P_ILT4':       'Actifs occupés travaillant dans une autre région en France métropolitaine',
    'P21_ACTOCC15P_ILT5':       'Actifs occupés travaillant hors de France métropolitaine',
    'C21_ACTOCC15P':            'Actifs occupés 15 ans ou plus (recensement)',
    'C21_ACTOCC15P_PAS':        'Actifs occupés n\'utilisant pas de moyen de transport pour travailler',
    'C21_ACTOCC15P_MAR':        'Actifs occupés allant travailler principalement à pied',
    'C21_ACTOCC15P_VELO':       'Actifs occupés allant travailler principalement à vélo',
    'C21_ACTOCC15P_2ROUESMOT':  'Actifs occupés allant travailler principalement en deux-roues motorisé',
    'C21_ACTOCC15P_VOIT':       'Actifs occupés allant travailler principalement en voiture',
    'C21_ACTOCC15P_TCOM':       'Actifs occupés allant travailler principalement en transports en commun',
    # ── Scolarisation et diplômes (effectifs) ──────────────────────────────────
    'P21_POP0205':              'Population 2-5 ans',
    'P21_POP0610':              'Population 6-10 ans',
    'P21_POP1114':              'Population 11-14 ans',
    'P21_POP1517':              'Population 15-17 ans',
    'P21_POP1824':              'Population 18-24 ans',
    'P21_POP2529':              'Population 25-29 ans',
    'P21_POP30P':               'Population 30 ans ou plus',
    'P21_SCOL0205':             'Scolarisés 2-5 ans',
    'P21_SCOL0610':             'Scolarisés 6-10 ans',
    'P21_SCOL1114':             'Scolarisés 11-14 ans',
    'P21_SCOL1517':             'Scolarisés 15-17 ans',
    'P21_SCOL1824':             'Scolarisés 18-24 ans',
    'P21_SCOL2529':             'Scolarisés 25-29 ans',
    'P21_SCOL30P':              'Scolarisés 30 ans ou plus',
    'P21_NSCOL15P':             'Non scolarisés 15 ans ou plus',
    'P21_NSCOL15P_DIPLMIN':     'Non scolarisés 15+ sans diplôme ou au plus CEP',
    'P21_NSCOL15P_BEPC':        'Non scolarisés 15+ titulaires BEPC / brevet des collèges / DNB',
    'P21_NSCOL15P_CAPBEP':      'Non scolarisés 15+ titulaires CAP ou BEP',
    'P21_NSCOL15P_BAC':         'Non scolarisés 15+ titulaires BAC ou brevet professionnel',
    'P21_NSCOL15P_SUP2':        'Non scolarisés 15+ titulaires BAC+2',
    'P21_NSCOL15P_SUP34':       'Non scolarisés 15+ titulaires BAC+3 ou BAC+4',
    'P21_NSCOL15P_SUP5':        'Non scolarisés 15+ titulaires BAC+5 ou plus',
    # ── Chômeurs par diplôme ────────────────────────────────────────────────────
    'P21_CHOM_DIPLMIN':         'Chômeurs 15-64 ans sans diplôme ou au plus CEP',
    'P21_CHOM_BEPC':            'Chômeurs 15-64 ans titulaires BEPC',
    'P21_CHOM_CAPBEP':          'Chômeurs 15-64 ans titulaires CAP ou BEP',
    'P21_CHOM_BAC':             'Chômeurs 15-64 ans titulaires BAC',
    'P21_CHOM_SUP2':            'Chômeurs 15-64 ans titulaires BAC+2',
    'P21_CHOM_SUP34':           'Chômeurs 15-64 ans titulaires BAC+3 ou BAC+4',
    'P21_CHOM_SUP5':            'Chômeurs 15-64 ans titulaires BAC+5 ou plus',
    # ── Actifs par diplôme ──────────────────────────────────────────────────────
    'P21_ACT_DIPLMIN':          'Actifs 15-64 ans sans diplôme ou au plus CEP',
    'P21_ACT_BEPC':             'Actifs 15-64 ans titulaires BEPC',
    'P21_ACT_CAPBEP':           'Actifs 15-64 ans titulaires CAP ou BEP',
    'P21_ACT_BAC':              'Actifs 15-64 ans titulaires BAC',
    'P21_ACT_SUP2':             'Actifs 15-64 ans titulaires BAC+2',
    'P21_ACT_SUP34':            'Actifs 15-64 ans titulaires BAC+3 ou BAC+4',
    'P21_ACT_SUP5':             'Actifs 15-64 ans titulaires BAC+5 ou plus',
}

ALL_VARS = []
for cat_vars in VARS_BY_CAT.values():
    for v in cat_vars:
        if v not in ALL_VARS:
            ALL_VARS.append(v)

# ── 1. CHARGEMENT DES DONNÉES ─────────────────────────────────────────────────
# Source unique : iris_final_socio_politique.csv contient toutes les colonnes nécessaires
# df_socio = pd.read_csv("iris/iris_final_socio_politique.csv", low_memory=False)
df_socio = pd.read_csv("iris/iris_final_socio_politique_bis.csv", low_memory=False)

# On garde aussi les coordonnées pour AXE_X / AXE_Y (optionnel)
# mais toutes les variables clés sont dans df_socio
df = df_socio.copy()
print(f"Données chargées : {len(df)} lignes × {len(df.columns)} colonnes")

# Colonnes de coordonnées si disponibles
try:
    df_coord = pd.read_csv("iris/iris_coordonnees_finales.csv", low_memory=False)
    coord_cols = [c for c in ['IRIS','LAB_IRIS','COM'] if c in df_coord.columns]
    for col in ['LAB_IRIS', 'COM']:
        if col not in df.columns and col in df_coord.columns:
            df = df.merge(df_coord[['IRIS', col]], on='IRIS', how='left')
    print("Colonnes LAB_IRIS/COM fusionnées depuis iris_coordonnees_finales.csv")
except Exception as e:
    print(f"iris_coordonnees_finales.csv non disponible : {e}")

# Assurer que LAB_IRIS et COM existent
if 'LAB_IRIS' not in df.columns:
    df['LAB_IRIS'] = df.get('IRIS', df.index.astype(str))
if 'COM' not in df.columns:
    df['COM'] = df.get('nom_commune', '')

# ── 2. POPULATION ─────────────────────────────────────────────────────────────
# Utiliser pop_totale (confirmée dans iris_final_socio_politique.csv)
if 'pop_totale' in df.columns:
    df['_pop'] = df['pop_totale'].fillna(df['pop_totale'].median())
elif 'TOTAL_POP_ESTIM' in df.columns:
    df['_pop'] = df['TOTAL_POP_ESTIM'].fillna(df['TOTAL_POP_ESTIM'].median())
else:
    df['_pop'] = 2000.0
print(f"Population : min={df['_pop'].min():.0f} max={df['_pop'].max():.0f} median={df['_pop'].median():.0f}")

# ── 3. CALCUL DES SCORES COMPOSITES ──────────────────────────────────────────

def _rang_pondere(series, pop):
    """Centile pondéré par population, centré à 0 (range ≈ -50 à +50)."""
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
    # Remplir les NaN par la médiane (50) puis centrer à 0
    return result.fillna(50.0) - 50.0

def make_score(pos_vars, neg_vars):
    """Score composite par rang centile pondéré par population."""
    parts = []
    pop = df['_pop']
    for v in pos_vars:
        if v not in df.columns:
            print(f"  SKIP {v} (missing)")
            continue
        parts.append(_rang_pondere(df[v], pop))
    for v in neg_vars:
        if v not in df.columns:
            print(f"  SKIP {v} (missing)")
            continue
        parts.append(-_rang_pondere(df[v], pop))
    if not parts:
        return pd.Series(0.0, index=df.index)
    return pd.concat(parts, axis=1).mean(axis=1)

# ── 3b. VARIABLES DÉRIVÉES POUR LES NOUVEAUX AXES ────────────────────────────
print("Calcul des variables dérivées...")
_nscol = df['P21_NSCOL15P'].replace(0, np.nan)
df['pct_sup5']        = df['P21_NSCOL15P_SUP5'] / _nscol * 100
df['pct_sans_diplome']= df['P21_NSCOL15P_DIPLMIN'] / _nscol * 100
df['pct_bac_plus']    = (df['P21_NSCOL15P_SUP2'] + df['P21_NSCOL15P_SUP34'] + df['P21_NSCOL15P_SUP5']) / _nscol * 100
df['pct_chomage']     = df['P21_CHOM1564'] / df['P21_ACT1564'].replace(0, np.nan) * 100
df['pct_inactif']     = df['P21_INACT1564'] / df['P21_POP1564'].replace(0, np.nan) * 100
df['pct_capbep']    = df['P21_NSCOL15P_CAPBEP'] / _nscol * 100
df['pct_etudiants'] = df['P21_ETUD1564'] / df['P21_POP1564'].replace(0, np.nan) * 100

_sal = df['P21_SAL15P'].replace(0, np.nan)
df['pct_cdi']             = df['P21_SAL15P_CDI'] / _sal * 100
df['pct_cdd']             = df['P21_SAL15P_CDD'] / _sal * 100
df['pct_interim']         = df['P21_SAL15P_INTERIM'] / _sal * 100
df['pct_temps_partiel']   = df['P21_SAL15P_TP'] / _sal * 100
_actocc = df['P21_ACTOCC15P'].replace(0, np.nan)
df['pct_actifs_voiture']     = df['C21_ACTOCC15P_VOIT'] / _actocc * 100
df['pct_actifs_transports']  = df['C21_ACTOCC15P_TCOM'] / _actocc * 100
df['pct_actifs_velo']       = df['C21_ACTOCC15P_VELO'] / _actocc * 100
df['pct_actifs_2roues']     = df['C21_ACTOCC15P_2ROUESMOT'] / _actocc * 100
df['pct_actifs_marche']     = df['C21_ACTOCC15P_MAR'] / _actocc * 100

print("Calcul des scores composites IRIS...")
for score_name, cfg in SCORES_CONFIG.items():
    df[score_name] = make_score(cfg['pos_vars'], cfg['neg_vars'])
    print(f"  {score_name}: min={df[score_name].min():.2f} max={df[score_name].max():.2f} mean={df[score_name].mean():.2f}")

# ── 4. PARTI DOMINANT PAR IRIS ────────────────────────────────────────────────
score_party_cols = [c for c in df.columns if c.startswith('score_') and c[6:] in ALL_ORDER]
score_party_cols = [c for c in score_party_cols if c in df.columns]
for c in score_party_cols:
    df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0.0)

if score_party_cols:
    party_map = {c: c[6:] for c in score_party_cols}  # 'score_RN' → 'RN'
    df['parti_dominant'] = df[score_party_cols].idxmax(axis=1).map(party_map).fillna('AUTRE')
    # IRIS with all-zero scores → AUTRE
    all_zero = (df[score_party_cols] == 0).all(axis=1)
    df.loc[all_zero, 'parti_dominant'] = 'AUTRE'
else:
    df['parti_dominant'] = 'AUTRE'

print("Distribution partis dominants (initial):")
print(df['parti_dominant'].value_counts().to_dict())

# ── 4b. CHARGEMENT DES DONNÉES ÉLECTORALES ────────────────────────────────────
import os as _os

# Couleurs pour tous les partis/candidats présents dans les élections
ALL_PARTIES_COLORS = {
    # Partis habituels
    'RN':           '#374151',
    'LFI':          '#DC2626',
    'PS':           '#EC4899',
    'ENS':          '#F97316',
    'EELV':         '#16A34A',
    'PCF':          '#9B1C1C',
    'LR':           '#1D4ED8',
    'REC':          '#0F172A',
    'AUTRE':        '#9CA3AF',
    # Coalitions de gauche
    'NFP':          '#B91C1C',
    'NUPES':        '#B91C1C',
    'PS_PP':        '#E86BA8',   # PS-Place Publique (Glucksmann, euro 2024)
    'UG':           '#C84B6E',   # Union de la Gauche (municipales)
    # Droite
    'UXD':          '#7C3AED',   # Ciotti / alliance LR-RN
    'DVD':          '#60A5FA',   # Divers droite
    'DVC':          '#A78BFA',   # Divers centre
    'DLF':          '#4B5563',   # Debout la France
    'EXD':          '#1F2937',   # Extrême droite hors RN
    # Centre
    'MODEM':        '#FB923C',   # MoDem
    'HOR':          '#FDBA74',   # Horizons
    'UDI':          '#93C5FD',   # UDI
    # Gauche
    'DVG':          '#F9A8D4',   # Divers gauche
    'EXG':          '#7F1D1D',   # Extrême gauche (LO, NPA)
    'REG':          "#89712F",   # Régionalistes
    # Présidentielles 2017
    'MACRON':       '#F97316',
    'LE_PEN':       '#374151',
    'MELENCHON':    '#DC2626',
    'FILLON':       '#1D4ED8',
    'HAMON':        '#EC4899',
    'DUPONT_AIGNAN':'#6B7280',
    # Présidentielles 2012
    'HOLLANDE':     '#EC4899',
    'SARKOZY':      '#1D4ED8',
    'BAYROU':       '#FB923C',
    'JOLY':         '#16A34A',
    # Présidentielles 2022
    'ZEMMOUR':      '#0F172A',
    'PECRESSE':     '#3B82F6',
    'JADOT':        '#16A34A',
    'ROUSSEL':      '#9B1C1C',
    'HIDALGO':      '#DB2777',
}

# Métadonnées des élections disponibles
# Structure : id_election → {type, year, tour, label, type_key}
ELECTIONS_AVAILABLE = {
    '2022_legi_t1': {'type': 'legi', 'year': 2022, 'tour': 1, 'label': 'Législatives 2022 — 1er tour'},
    '2022_legi_t2': {'type': 'legi', 'year': 2022, 'tour': 2, 'label': 'Législatives 2022 — 2e tour'},
    '2024_legi_t1': {'type': 'legi', 'year': 2024, 'tour': 1, 'label': 'Législatives 2024 — 1er tour'},
    '2024_legi_t2': {'type': 'legi', 'year': 2024, 'tour': 2, 'label': 'Législatives 2024 — 2e tour'},
    '2017_legi_t1': {'type': 'legi', 'year': 2017, 'tour': 1, 'label': 'Législatives 2017 — 1er tour'},
    '2017_legi_t2': {'type': 'legi', 'year': 2017, 'tour': 2, 'label': 'Législatives 2017 — 2e tour'},
    '2024_euro_t1': {'type': 'euro', 'year': 2024, 'tour': 1, 'label': 'Européennes 2024'},
    '2022_pres_t1': {'type': 'pres', 'year': 2022, 'tour': 1, 'label': 'Présidentielles 2022 — 1er tour'},
    '2022_pres_t2': {'type': 'pres', 'year': 2022, 'tour': 2, 'label': 'Présidentielles 2022 — 2e tour'},
    '2017_pres_t1': {'type': 'pres', 'year': 2017, 'tour': 1, 'label': 'Présidentielles 2017 — 1er tour'},
    '2017_pres_t2': {'type': 'pres', 'year': 2017, 'tour': 2, 'label': 'Présidentielles 2017 — 2e tour'},
    '2020_muni_t1': {'type': 'muni', 'year': 2020, 'tour': 1, 'label': 'Municipales 2020 — 1er tour'},
    '2020_muni_t2': {'type': 'muni', 'year': 2020, 'tour': 2, 'label': 'Municipales 2020 — 2e tour'},
    '2026_muni_t1': {'type': 'muni', 'year': 2026, 'tour': 1, 'label': 'Municipales 2026 — 1er tour'},
}

DEFAULT_ELECTION = '2022_legi_t1'


def _load_election_iris_data(election_id, df_iris):
    """Charge le CSV élection, merge sur CODE_IRIS, retourne liste de dicts par IRIS."""
    path = f"iris/elections/{election_id}.csv"
    if not _os.path.exists(path):
        print(f"  ⚠️  {path} introuvable, skip")
        return None
    elec = pd.read_csv(path, index_col='CODE_IRIS', dtype={'CODE_IRIS': str})
    score_cols = [c for c in elec.columns
                  if c.startswith('score_') and c not in ('score_blanc', 'score_nul')]
    if not score_cols:
        return None
    party_names = [c.replace('score_', '') for c in score_cols]
    extra_cols = [c for c in ['pct_abstention', 'inscrits', 'votants', 'exprimes', 'blancs', 'nuls'] if c in elec.columns]

    # Merge : pour chaque IRIS du df, chercher ses scores électoraux par CODE_IRIS=df.IRIS
    merged = df_iris[['IRIS']].merge(
        elec[score_cols + extra_cols],
        left_on='IRIS', right_index=True, how='left'
    )
    scores_arr = merged[score_cols].fillna(0).values
    dominant_idx = scores_arr.argmax(axis=1)
    has_abst = 'pct_abstention' in merged.columns
    has_inscrits = 'inscrits' in merged.columns
    has_exprimes = 'exprimes' in merged.columns
    has_blancs = 'blancs' in merged.columns
    has_nuls = 'nuls' in merged.columns
    result = []
    for i in range(len(df_iris)):
        row_sum = scores_arr[i].sum()
        abst = float(merged['pct_abstention'].iloc[i]) if has_abst and not pd.isna(merged['pct_abstention'].iloc[i]) else None
        inscrits_val = float(merged['inscrits'].iloc[i]) if has_inscrits and not pd.isna(merged['inscrits'].iloc[i]) else None
        exprimes_val = float(merged['exprimes'].iloc[i]) if has_exprimes and not pd.isna(merged['exprimes'].iloc[i]) else None
        blancs_val = float(merged['blancs'].iloc[i]) if has_blancs and not pd.isna(merged['blancs'].iloc[i]) else None
        nuls_val = float(merged['nuls'].iloc[i]) if has_nuls and not pd.isna(merged['nuls'].iloc[i]) else None
        if row_sum == 0:
            result.append({'parti': None, 'color': None, 'scores': {}, 'abst': abst, 'inscrits': inscrits_val, 'exprimes': exprimes_val, 'blancs': blancs_val, 'nuls': nuls_val})
            continue
        parti = party_names[dominant_idx[i]]
        color = ALL_PARTIES_COLORS.get(parti, '#9CA3AF')
        if row_sum > 101:  # voix brutes (1 seul candidat municipal) → convertir en %
            row_scores = {party_names[j]: round(float(scores_arr[i, j]) / row_sum * 100, 1) for j in range(len(party_names))}
        else:
            row_scores = {party_names[j]: round(float(scores_arr[i, j]), 1) for j in range(len(party_names))}
        result.append({'parti': parti, 'color': color, 'scores': row_scores, 'abst': abst, 'inscrits': inscrits_val, 'exprimes': exprimes_val, 'blancs': blancs_val, 'nuls': nuls_val})
    return result


print("Chargement des données électorales...")
iris_election_data = {}
for eid in ELECTIONS_AVAILABLE:
    data = _load_election_iris_data(eid, df)
    if data:
        iris_election_data[eid] = data
        print(f"  {eid}: OK ({len(data)} IRIS)")

# Remplacer parti_dominant par celui de l'élection par défaut
_default_elec_data = iris_election_data.get(DEFAULT_ELECTION)
if _default_elec_data:
    df['parti_dominant'] = [d['parti'] for d in _default_elec_data]
    # Remplacer aussi les couleurs dans COULEURS pour les nouveaux partis
    for parti, color in ALL_PARTIES_COLORS.items():
        if parti not in COULEURS:
            COULEURS[parti] = color
            LABELS[parti] = parti
            SHORT[parti] = parti[:4]
    print(f"Distribution partis dominants ({DEFAULT_ELECTION}):")
    print(df['parti_dominant'].value_counts().to_dict())

# ── 5. TAILLE DES MARQUEURS ────────────────────────────────────────────────────
pop = df['_pop'].copy()
q5  = pop.quantile(0.05)
q95 = pop.quantile(0.95)
pop_clipped = pop.clip(q5, q95)
lower_size = 1.2
upper_size = 2.5
marker_size = lower_size + (pop_clipped - q5) / (q95 - q5) * (upper_size - lower_size)  # 2.5–4px

# ── 6. JITTER ET DONNÉES PAR VARIABLE ─────────────────────────────────────────
np.random.seed(42)
N = len(df)
jit_unit_x = np.random.uniform(-1, 1, N)
jit_unit_y = np.random.uniform(-1, 1, N)

var_data_x = {}
var_data_y = {}
for v in ALL_VARS:
    if v not in df.columns:
        print(f"  WARNING: variable {v} not found, skipping")
        continue
    vals = df[v].copy().astype(float)
    mean_val = vals.mean()
    vals_filled = vals.fillna(mean_val)
    std_val = vals_filled.std()
    if np.isnan(std_val) or std_val == 0:
        std_val = 1.0
    scale = 0.015
    var_data_x[v] = (vals_filled + jit_unit_x * scale * std_val).tolist()
    var_data_y[v] = (vals_filled + jit_unit_y * scale * std_val).tolist()

# ── 6b. PRÉ-CALCUL BARYCENTRES PAR ÉLECTION ─────────────────────────────────
# Pré-calcule baryMeans, barySizes, abstBary, buttonPcts pour chaque élection
# afin d'éviter de les recalculer en JS à chaque changement d'élection.
print("Pré-calcul des barycentres par élection...")
pops = df['_pop'].fillna(1).values.astype(float)
iris_election_precomputed = {}

# Pré-construire la matrice des variables (n_iris × n_vars) — partagée entre élections
var_names = list(var_data_x.keys())
var_mat = np.column_stack([np.array(var_data_x[v], dtype=float) for v in var_names])
var_nan_mask = np.isnan(var_mat)
var_mat_clean = np.where(var_nan_mask, 0.0, var_mat)
var_valid = (~var_nan_mask).astype(float)  # 1 si valide, 0 si NaN

for eid, data in iris_election_data.items():
    n = len(data)
    # Collecter tous les partis présents
    parties_set = set()
    for d in data:
        if d['scores'] and isinstance(d['scores'], dict):
            parties_set.update(d['scores'].keys())
    parties_list = sorted(parties_set)
    p = len(parties_list)
    party_idx = {g: j for j, g in enumerate(parties_list)}

    # --- Construire matrice scores (n × p) ---
    score_mat = np.zeros((n, p))
    abst_arr = np.full(n, np.nan)
    exp_arr = np.zeros(n)
    for i, d in enumerate(data):
        if d['scores'] and isinstance(d['scores'], dict):
            for g, s in d['scores'].items():
                score_mat[i, party_idx[g]] = s
        abst_val = d.get('abst')
        if abst_val is not None:
            abst_arr[i] = float(abst_val)
        exp_val = d.get('exprimes')
        if exp_val is not None:
            exp_arr[i] = float(exp_val)

    # --- baryMeans: pondéré par score × pop (vectorisé) ---
    weights = score_mat * pops[:, np.newaxis]  # (n × p)
    # Pour ignorer les NaN dans var_mat : pondérer par valid_mask
    weighted_sums = weights.T @ var_mat_clean  # (p × v)
    weight_totals_per_var = weights.T @ var_valid  # (p × v) — total poids valides par variable
    weight_totals_per_var[weight_totals_per_var == 0] = np.nan

    bary_means_elec = {}
    for j, g in enumerate(parties_list):
        bary_means_elec[g] = {}
        for k, v in enumerate(var_names):
            wt = weight_totals_per_var[j, k]
            if not np.isnan(wt):
                bary_means_elec[g][v] = round(weighted_sums[j, k] / wt, 3)
            else:
                bary_means_elec[g][v] = 0

    # --- barySizes: sqrt scale 8-45 ---
    # totals par parti = sum(pop × score / 100)
    party_totals = (pops[:, np.newaxis] * score_mat / 100).sum(axis=0)  # (p,)
    totals = {g: party_totals[j] for j, g in enumerate(parties_list) if party_totals[j] > 0}
    # Abstention
    abst_valid = ~np.isnan(abst_arr)
    abst_total = np.sum(pops[abst_valid] * abst_arr[abst_valid] / 100) if abst_valid.any() else 0.0
    if abst_total > 0:
        totals['__ABST__'] = abst_total

    vals_list = [v for v in totals.values() if v > 0]
    bary_sizes_elec = {}
    if vals_list:
        min_v, max_v = min(vals_list), max(vals_list)
        for g, v in totals.items():
            bary_sizes_elec[g] = round(8 + (((v - min_v) / (max_v - min_v)) ** 0.5) * 37, 1) if max_v > min_v else 22

    # --- abstBary: pondéré par pop × abst (vectorisé) ---
    abst_weights = np.where(abst_valid, pops * abst_arr / 100, 0.0)  # (n,)
    abst_w_sums = abst_weights @ var_mat_clean  # (v,)
    abst_w_totals = abst_weights @ var_valid  # (v,)
    abst_bary_elec = {}
    for k, v in enumerate(var_names):
        if abst_w_totals[k] > 0:
            abst_bary_elec[v] = round(abst_w_sums[k] / abst_w_totals[k], 3)
        else:
            abst_bary_elec[v] = None

    # --- buttonPcts: % des exprimés par parti (vectorisé) ---
    total_exprimes = exp_arr.sum()
    button_pcts_elec = {}
    if total_exprimes > 0:
        party_voix = (exp_arr[:, np.newaxis] * score_mat / 100).sum(axis=0)  # (p,)
        for j, g in enumerate(parties_list):
            button_pcts_elec[g] = round(party_voix[j] / total_exprimes * 1000) / 10

    iris_election_precomputed[eid] = {
        'baryMeans': bary_means_elec,
        'barySizes': bary_sizes_elec,
        'abstBary': abst_bary_elec,
        'buttonPcts': button_pcts_elec,
    }
    print(f"  {eid}: {len(parties_set)} partis, {len(bary_sizes_elec)} tailles bary")

# ── 7. BARYCENTRES PAR PARTI ──────────────────────────────────────────────────
def group_means():
    bm = {}
    for g in ALL_ORDER:
        sub = df[df['parti_dominant'] == g]
        if sub.empty:
            continue
        bm[g] = {}
        pop_sub = df.loc[sub.index, '_pop'].fillna(1).values
        for v in ALL_VARS:
            if v not in df.columns:
                continue
            vals = pd.to_numeric(sub[v], errors='coerce').values
            valid = ~np.isnan(vals)
            if valid.any():
                bm[g][v] = float(np.average(vals[valid], weights=pop_sub[valid]))
            else:
                bm[g][v] = 0.0
    return bm

bary_means = group_means()

def build_group_data():
    gd_x = {}
    gd_y = {}
    for g in ORDER:
        mask = df['parti_dominant'] == g
        if not mask.any():
            continue
        idxs = df.index[mask].tolist()
        gd_x[g] = {}
        gd_y[g] = {}
        for v in ALL_VARS:
            if v not in var_data_x:
                continue
            gd_x[g][v] = [var_data_x[v][i] for i in idxs]
            gd_y[g][v] = [var_data_y[v][i] for i in idxs]
    return gd_x, gd_y

group_data_x, group_data_y = build_group_data()

# ── 8. BARYCENTRES PLOTLY ──────────────────────────────────────────────────────
bary_rows = []
for g in ORDER:
    sub = df[df['parti_dominant'] == g]
    if len(sub) < 2:
        continue
    bary_rows.append({
        'g': g,
        'x': float(-sub['score_exploitation'].mean()),
        'y': float(sub['score_domination'].mean()),
        'n': len(sub),
        'c': COULEURS.get(g, '#999'),
        'label': LABELS.get(g, g),
        'abbr': SHORT.get(g, g),
    })
bary = pd.DataFrame(bary_rows)

# ── 9. HELPER JS DATA ─────────────────────────────────────────────────────────
def _round0(arr):
    return [int(round(float(v))) for v in arr]

def _round1(arr):
    return [round(float(v), 1) for v in arr]

def _round2(arr):
    return [round(float(v), 2) for v in arr]

def _round3(arr):
    return [round(float(v), 3) for v in arr]

def _round_bary(bm):
    result = {}
    for g, vars_dict in bm.items():
        result[g] = {v: round(float(val), 3) for v, val in vars_dict.items()}
    return result

# Composite score vars get 3 decimals; age_moyen gets 1 decimal; pct vars get 0 (integer %)
_COMPOSITE_VARS = set(VARS_BY_CAT.get('Scores composites', [])) | {'tsne_x', 'tsne_y', 'umap_x', 'umap_y'}
_ONE_DECIMAL_VARS = {'age_moyen', 'pct_etrangers', 'pct_immigres'}

def _build_js_data():
    gd_x_js = {}
    gd_y_js = {}
    for g in ORDER:
        if g not in group_data_x:
            continue
        gd_x_js[g] = {}
        gd_y_js[g] = {}
        for v, arr in group_data_x[g].items():
            fn = _round3 if v in _COMPOSITE_VARS else (_round1 if v in _ONE_DECIMAL_VARS else _round0)
            gd_x_js[g][v] = fn(arr)
        for v, arr in group_data_y[g].items():
            fn = _round3 if v in _COMPOSITE_VARS else (_round1 if v in _ONE_DECIMAL_VARS else _round0)
            gd_y_js[g][v] = fn(arr)
    return (
        _json.dumps(gd_x_js, ensure_ascii=False, separators=(',', ':')),
        _json.dumps(gd_y_js, ensure_ascii=False, separators=(',', ':')),
        _json.dumps(_round_bary(bary_means), ensure_ascii=False, separators=(',', ':')),
        _json.dumps(AXIS_PRESETS, ensure_ascii=False),
        _json.dumps(VARS_BY_CAT, ensure_ascii=False),
        _json.dumps(VAR_LABELS, ensure_ascii=False),
    )

_DEP_NAMES = {
    '01':'Ain','02':'Aisne','03':'Allier','04':'Alpes-de-Haute-Provence','05':'Hautes-Alpes',
    '06':'Alpes-Maritimes','07':'Ardèche','08':'Ardennes','09':'Ariège','10':'Aube',
    '11':'Aude','12':'Aveyron','13':'Bouches-du-Rhône','14':'Calvados','15':'Cantal',
    '16':'Charente','17':'Charente-Maritime','18':'Cher','19':'Corrèze','2A':'Corse-du-Sud',
    '2B':'Haute-Corse','21':'Côte-d\'Or','22':'Côtes-d\'Armor','23':'Creuse','24':'Dordogne',
    '25':'Doubs','26':'Drôme','27':'Eure','28':'Eure-et-Loir','29':'Finistère',
    '30':'Gard','31':'Haute-Garonne','32':'Gers','33':'Gironde','34':'Hérault',
    '35':'Ille-et-Vilaine','36':'Indre','37':'Indre-et-Loire','38':'Isère','39':'Jura',
    '40':'Landes','41':'Loir-et-Cher','42':'Loire','43':'Haute-Loire','44':'Loire-Atlantique',
    '45':'Loiret','46':'Lot','47':'Lot-et-Garonne','48':'Lozère','49':'Maine-et-Loire',
    '50':'Manche','51':'Marne','52':'Haute-Marne','53':'Mayenne','54':'Meurthe-et-Moselle',
    '55':'Meuse','56':'Morbihan','57':'Moselle','58':'Nièvre','59':'Nord',
    '60':'Oise','61':'Orne','62':'Pas-de-Calais','63':'Puy-de-Dôme','64':'Pyrénées-Atlantiques',
    '65':'Hautes-Pyrénées','66':'Pyrénées-Orientales','67':'Bas-Rhin','68':'Haut-Rhin','69':'Rhône',
    '70':'Haute-Saône','71':'Saône-et-Loire','72':'Sarthe','73':'Savoie','74':'Haute-Savoie',
    '75':'Paris','76':'Seine-Maritime','77':'Seine-et-Marne','78':'Yvelines','79':'Deux-Sèvres',
    '80':'Somme','81':'Tarn','82':'Tarn-et-Garonne','83':'Var','84':'Vaucluse',
    '85':'Vendée','86':'Vienne','87':'Haute-Vienne','88':'Vosges','89':'Yonne',
    '90':'Territoire de Belfort','91':'Essonne','92':'Hauts-de-Seine','93':'Seine-Saint-Denis',
    '94':'Val-de-Marne','95':'Val-d\'Oise','971':'Guadeloupe','972':'Martinique',
    '973':'Guyane','974':'La Réunion','976':'Mayotte',
}

def _dep_label(iris_code):
    s = str(iris_code)
    if s.startswith('97') and len(s) >= 3:
        key = s[:3]
    elif s[:2].upper() in ('2A', '2B'):
        key = s[:2].upper()
    else:
        key = s[:2]
    name = _DEP_NAMES.get(key, '')
    return f'{name} ({key})' if name else ''

# ── 10. TRACES PLOTLY ─────────────────────────────────────────────────────────
# Colonnes pour IRIS_INFO (20 champs — scores partis lus depuis IRIS_ELECTION_DATA)
# [0] LAB_IRIS, [1] nom_commune, [2] pop_totale, [3] DISP_MED21,
# [4] pct_csp_plus, [5] pct_csp_ouvrier, [6] pct_csp_intermediaire,
# [7] DISP_PPAT21, [8] inscrits, [9] votants, [10] pct_abstention,
# [11] score_blanc, [12] score_nul,
# [13] pct_proprietaires, [14] pct_hlm,
# [15] pct_chomage, [16] pct_bac_plus, [17] pct_sans_diplome, [18] age_moyen

_CD_PARTY_SCORES = [f'score_{g}' for g in ALL_ORDER]

def _make_customdata(sub):
    commune_col = 'nom_commune' if 'nom_commune' in df.columns else 'COM'
    pop_col     = 'pop_totale'  if 'pop_totale'  in df.columns else '_pop'

    cd_cols = ['LAB_IRIS', commune_col, pop_col, 'DISP_MED21',
               'pct_csp_plus', 'pct_csp_ouvrier', 'pct_csp_intermediaire',
               'DISP_PPAT21', 'inscrits', 'votants', 'pct_abstention',
               'score_blanc', 'score_nul',
               'pct_proprietaires', 'pct_hlm',
               'pct_chomage', 'pct_bac_plus', 'pct_sans_diplome', 'age_moyen']

    # Only keep cols that exist
    existing = [c for c in cd_cols if c in sub.columns]
    cd_df = sub[existing].copy()

    # Fill missing columns with ''
    for c in cd_cols:
        if c not in cd_df.columns:
            cd_df[c] = ''

    cd_df = cd_df[cd_cols]
    numeric_cols = [c for c in cd_cols if c not in ('LAB_IRIS', commune_col)]
    for col in numeric_cols:
        try:
            cd_df[col] = pd.to_numeric(cd_df[col], errors='coerce').round(2)
        except Exception:
            pass
    rows = cd_df.fillna('').values.tolist()
    # [19] dep_label — dérivé du code IRIS (colonne 'IRIS' du df global)
    iris_codes = sub['IRIS'].tolist() if 'IRIS' in sub.columns else [''] * len(rows)
    for i, row in enumerate(rows):
        row.append(_dep_label(iris_codes[i]))
    return rows


def _build_trace_data(size_scale=1.0, include_data=True):
    traces = []
    trace_group_map = {}

    # Trace 0: barycentres
    if include_data:
        bary_trace = go.Scattergl(
            x=bary['x'].tolist(), y=bary['y'].tolist(),
            mode="markers+text",
            marker=dict(symbol="cross-thin", size=22, color=bary['c'].tolist(),
                        line=dict(width=3, color=bary['c'].tolist())),
            text=bary['abbr'].tolist(), textposition="top right",
            textfont=dict(size=9, color=bary['c'].tolist(), family="Helvetica Neue, sans-serif"),
            hovertemplate="<b>Barycentre %{text}</b><br>X : <b>%{x:.3f}</b><br>Y : <b>%{y:.3f}</b><extra></extra>",
            showlegend=False, opacity=0.9,
            name="barycentres"
        )
    else:
        bary_trace = go.Scattergl(
            x=[], y=[],
            mode="markers+text",
            marker=dict(symbol="cross-thin", size=[], color=[],
                        line=dict(width=3, color=[])),
            text=[], textposition="top right",
            textfont=dict(size=9, color=[], family="Helvetica Neue, sans-serif"),
            hovertemplate="<b>Barycentre %{text}</b><br>X : <b>%{x:.3f}</b><br>Y : <b>%{y:.3f}</b><extra></extra>",
            showlegend=False, opacity=0.9,
            name="barycentres"
        )
    traces.append(bary_trace)
    trace_idx = 1

    # Scatter traces per party (in ALL_ORDER = render order)
    for g in ALL_ORDER:
        sub = df[df['parti_dominant'] == g].copy()
        if sub.empty:
            tr = go.Scattergl(
                x=[], y=[], mode="markers", name=LABELS.get(g, g),
                marker=dict(color=ALL_PARTIES_COLORS.get(g, '#999'), size=[]),
                customdata=[],
                hoverinfo="none",
                showlegend=False,
                legendgroup=g,
            )
            traces.append(tr)
            trace_group_map[g] = trace_idx
            trace_idx += 1
            continue
        c, lb = ALL_PARTIES_COLORS.get(g, "#999"), LABELS.get(g, g)
        op = OPACITY.get(g, 0.75)

        if include_data:
            sub_idx = sub.index.tolist()
            x_vals = [var_data_x['score_exploitation'][i] for i in sub_idx]
            y_vals = [var_data_y['score_domination'][i] for i in sub_idx]
            sz = [float(marker_size.loc[i]) * size_scale for i in sub_idx]
            customdata_list = _make_customdata(sub)
            tr = go.Scattergl(
                x=x_vals, y=y_vals, mode="markers", name=lb,
                marker=dict(color=c, size=sz, opacity=op,
                            line=dict(width=0.4, color="rgba(255,255,255,0.4)")),
                customdata=customdata_list,
                hoverinfo="none",
                showlegend=False,
                legendgroup=g,
            )
        else:
            tr = go.Scattergl(
                x=[], y=[], mode="markers", name=lb,
                marker=dict(color=c, size=[], opacity=op,
                            line=dict(width=0.4, color="rgba(255,255,255,0.4)")),
                customdata=[],
                hoverinfo="none",
                showlegend=False,
                legendgroup=g,
            )
        traces.append(tr)
        trace_group_map[g] = trace_idx
        trace_idx += 1

    # No density contour traces
    contour_group_map = {g: None for g in ALL_ORDER}

    return traces, trace_group_map, contour_group_map


def _build_trace_data_single():
    """Desktop: 2 traces only — trace 0 = barycentres (empty), trace 1 = all IRIS (empty)."""
    traces = []
    # Trace 0: barycentres (filled by applyElection)
    traces.append(go.Scattergl(
        x=[], y=[], mode="markers+text",
        marker=dict(symbol="cross-thin", size=[], color=[], line=dict(width=3, color=[])),
        text=[], textposition="top right",
        textfont=dict(size=9, color=[], family="Helvetica Neue, sans-serif"),
        hovertemplate="<b>Barycentre %{text}</b><br>X : <b>%{x:.3f}</b><br>Y : <b>%{y:.3f}</b><extra></extra>",
        showlegend=False, opacity=0.9, name="barycentres"
    ))
    # Trace 1: all IRIS points (filled by applyElection)
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
def build_mobile_html():
    import math
    def _is_nan(v):
        return v is None or (isinstance(v, float) and math.isnan(v))

    traces = _build_trace_data_single()

    # Données socio (IRIS_X / IRIS_Y) → fichiers JSON externes (précision réduite mobile)
    import math as _math2, os as _os
    iris_x_js = {}
    iris_y_js = {}
    for v in var_data_x:
        fn = _round3 if v in _COMPOSITE_VARS else (_round1 if v in _ONE_DECIMAL_VARS else _round0)
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

    all_customdata = _make_customdata(df)
    iris_pops = [int(round(float(v))) for v in df['_pop'].fillna(1).tolist()]
    marker_sizes_list = [round(float(marker_size.loc[i]), 1) for i in df.index]
    group_indices = {g: df.index[df['parti_dominant'] == g].tolist() for g in ALL_ORDER}

    # ── Écriture des fichiers JSON dans data/ ──────────────────────────────
    def _build_geo_centroids(df_arg):
        import geopandas as gpd
        print("  Calcul des centroïdes IRIS depuis iris-stats.geojson...")
        gdf = gpd.read_file('iris-stats.geojson')
        gdf = gdf.set_crs('EPSG:2154', allow_override=True)
        gdf_wgs = gdf.to_crs('EPSG:4326')
        gdf_wgs = gdf_wgs.copy()
        gdf_wgs['lat'] = gdf_wgs.geometry.centroid.y
        gdf_wgs['lon'] = gdf_wgs.geometry.centroid.x
        lookup = dict(zip(gdf_wgs['index'].astype(str), zip(gdf_wgs['lat'], gdf_wgs['lon'])))
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

    # ── Métadonnées inline (petites, <500 KB) ─────────────────────────────
    elections_meta_str = _json.dumps(ELECTIONS_AVAILABLE, ensure_ascii=False, separators=(',', ':'))
    default_elec_str = _json.dumps(DEFAULT_ELECTION, separators=(',', ':'))
    presets_str = _json.dumps(AXIS_PRESETS, ensure_ascii=False)
    vars_str = _json.dumps(VARS_BY_CAT, ensure_ascii=False)
    var_labels_str = _json.dumps(VAR_LABELS, ensure_ascii=False)
    couleurs_str = _json.dumps(COULEURS, ensure_ascii=False, separators=(',', ':'))
    vote_parties_str = _json.dumps(VOTE_PARTIES_JS, separators=(',', ':'))
    all_parties_colors_str = _json.dumps(ALL_PARTIES_COLORS, ensure_ascii=False, separators=(',', ':'))

    buttons_data = [{'key': g, 'short': SHORT.get(g, g), 'label': LABELS.get(g, g),
                     'color': ALL_PARTIES_COLORS.get(g, '#9CA3AF'),
                     'count': int((df['parti_dominant'] == g).sum())} for g in ALL_ORDER]
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
    n_iris = len(df)

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
  <div id="mapDiv" style="width:100%;height:calc(100vh - 160px);min-height:300px;display:none;position:relative;">
    <div id="carteLoadingMsg" style="display:none;position:absolute;top:8px;left:50%;transform:translateX(-50%);background:rgba(255,255,255,0.92);padding:6px 16px;border-radius:20px;font-size:12px;color:#555;z-index:10;box-shadow:0 1px 4px rgba(0,0,0,0.1)">Chargement des coordonnées géographiques…</div>
    <button id="mapResetBtn" onclick="mapInstance && mapInstance.fitBounds([[-5.2,41.3],[9.6,51.2]],{{padding:10}})" style="position:absolute;bottom:16px;right:8px;z-index:20;background:rgba(255,255,255,0.95);border:1px solid #ccc;border-radius:6px;padding:6px 10px;font-size:13px;cursor:pointer;box-shadow:0 1px 4px rgba(0,0,0,0.15)">↺ Recentrer</button>
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
const elecCache = {{}};  // cache élections déjà fetché

let currentXVar = 'score_exploitation';
let currentYVar = 'score_domination';
let currentXInvert = false;
let currentPresetId = 'carte';
let currentColorVar = null;   // null = couleur par élection, sinon nom de variable
let colorVarMin = 0, colorVarMax = 1;  // percentile 2–98 de la variable
let currentXRange = PRESETS[0].xRange.slice();
let currentYRange = PRESETS[0].yRange.slice();
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
function computeDataRange(varName, invert) {{
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
  const pad = (mx - mn) * 0.08;
  return [mn - pad, mx + pad];
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
  const elecScores = elecData ? elecData.scores[irisGlobalIdx] : null;
  const currentMeta = ELECTIONS_META[currentElectionId];
  const currentColor = elecData?.colors[irisGlobalIdx] || '#9CA3AF';
  const currentParti = elecData?.partis[irisGlobalIdx] || '—';

  const abstElec = elecData ? elecData.abst[irisGlobalIdx] : null;

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
  }}

  document.getElementById('cardName').textContent = cd[1] || cd[0] || 'IRIS inconnu';
  const partyEl = document.getElementById('cardParty');
  partyEl.textContent = cd[19] || '';
  partyEl.style.color = '#666';
  document.getElementById('cardPop').textContent = fmtNum(cd[2], ' hab.');
  document.getElementById('cardAge').textContent = cd[18] !== '' && cd[18] != null ? Number(cd[18]).toFixed(1) + ' ans' : '—';
  document.getElementById('cardRev').textContent = fmtNum(cd[3], ' €/UC');
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

// ── Restyle all IRIS (trace 1) ────────────────────────────────────────────
function restyleIRIS() {{
  if (!IRIS_X || !IRIS_Y) return;
  const data = elecCache[currentElectionId];
  if (!data) return;
  const xArr = IRIS_X[currentXVar] || [];
  const yArr = IRIS_Y[currentYVar] || [];
  const colorArr = currentColorVar ? (IRIS_X[currentColorVar] || IRIS_Y[currentColorVar] || null) : null;
  const n = xArr.length;
  const knownKeys = new Set(btns.map(b => b.key));
  const fx = [], fy = [], fc = [], fs = [], fcd = [];
  for (let i = 0; i < n; i++) {{
    const parti = data.partis[i];
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
      fc.push(colorArr ? '#CCCCCC' : (data.colors[i] || '#9CA3AF'));
    }}
    fs.push(MARKER_SIZES[i] || 3);
    fcd.push(i);
  }}
  Plotly.restyle(chartDiv, {{
    x: [fx], y: [fy],
    'marker.color': [fc],
    'marker.size': [fs],
    customdata: [fcd],
  }}, [1]);
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
    const xm = baryMeans[g][currentXVar] !== undefined ? baryMeans[g][currentXVar] : 0;
    const ym = baryMeans[g][currentYVar] !== undefined ? baryMeans[g][currentYVar] : 0;
    baryX.push(currentXInvert ? -xm : xm);
    baryY.push(ym);
    baryColors.push(ALL_PARTIES_COLORS_JS[g] || COULEURS_JS[g] || '#999');
    baryTexts.push(btns.find(b2 => b2.key === g)?.short || g);
    barySzs.push(currentBarySizeMap[g] || 22);
  }});
  const abstBary = computeAbstBary(currentElectionId);
  if (abstBary) {{
    const xm = abstBary[currentXVar], ym = abstBary[currentYVar];
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
  }}, [0]);
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
  if (!elecCache[electionId]) {{
    const spinner = document.getElementById('elecSpinner');
    if (spinner) spinner.style.display = 'inline';
    try {{
      const r = await fetch('data/elec_' + electionId + '.json');
      elecCache[electionId] = await r.json();
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
    if (partisData && partisData.partis[currentClickedGlobalIdx] !== null) {{
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

// ── Apply axes ────────────────────────────────────────────────────────────
function applyAxes(xVar, xInvert, yVar, preset) {{
  currentXVar = xVar;
  currentYVar = yVar;
  currentXInvert = xInvert;
  if (preset) {{
    currentXRange = preset.xRange ? preset.xRange.slice() : computeDataRange(xVar, xInvert);
    currentYRange = preset.yRange ? preset.yRange.slice() : computeDataRange(yVar, false);
    currentCorners = preset.corners;
  }} else {{
    currentXRange = computeDataRange(xVar, xInvert);
    currentYRange = computeDataRange(yVar, false);
  }}

  activeGroups = new Set(btns.filter(b => b.count > 0).map(b => b.key));
  rebuildFilterButtons();
  updateToggleLabel();
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
  const enabledKeys = new Set(btns.filter(b => activeGroups.has(b.key)).map(b => b.key));
  const knownPartiKeys = new Set(btns.map(b => b.key));
  const colorArr = currentColorVar ? (IRIS_X && (IRIS_X[currentColorVar] || null)) || (IRIS_Y && (IRIS_Y[currentColorVar] || null)) : null;
  const features = [];
  for (let i = 0; i < IRIS_LAT.length; i++) {{
    if (IRIS_LAT[i] === null) continue;
    const parti = elecData.partis[i];
    const effectiveKey = knownPartiKeys.has(parti) ? parti : 'AUTRE';
    if (!enabledKeys.has(effectiveKey)) continue;
    let color;
    if (colorArr && colorArr[i] !== null && colorArr[i] !== undefined) {{
      color = varToHex(colorArr[i], colorVarMin, colorVarMax);
    }} else {{
      color = colorArr ? '#CCCCCC' : (elecData.colors[i] || '#9CA3AF');
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
  if (gj) mapInstance.getSource('iris').setData(gj);
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
    maxBounds: [[-7.0, 39.5], [11.5, 52.5]],
    dragRotate: false,
  }});
  mapInstance.touchZoomRotate.disableRotation();
  mapInstance.on('load', () => {{
    mapReady = true;
    // Overlay blanc semi-transparent pour atténuer le fond de carte
    const overlay = document.createElement('div');
    overlay.style.cssText = 'position:absolute;inset:0;background:rgba(255,255,255,0.22);pointer-events:none;z-index:2';
    document.getElementById('mapDiv').appendChild(overlay);
    mapInstance.addSource('iris', {{
      type: 'geojson',
      data: {{ type: 'FeatureCollection', features: [] }}
    }});
    mapInstance.addLayer({{
      id: 'iris-circles',
      type: 'circle',
      source: 'iris',
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
    mapInstance.on('click', 'iris-circles', e => {{
      const idx = e.features[0].properties.idx;
      currentClickedGlobalIdx = idx;
      showCard(idx);
    }});
    mapInstance.on('mouseenter', 'iris-circles', () => {{
      mapInstance.getCanvas().style.cursor = 'pointer';
    }});
    mapInstance.on('mouseleave', 'iris-circles', () => {{
      mapInstance.getCanvas().style.cursor = '';
    }});
    updateMapColors();
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
    initMap();
  }} else {{
    updateMapColors();
  }}
}}

function hideCarte() {{
  if (!isCarteActive) return;
  isCarteActive = false;
  document.getElementById('mapDiv').style.display = 'none';
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

  const [xData, yData] = await Promise.all([
    fetch('data/iris_x_mobile.json').then(r => r.json()),
    fetch('data/iris_y_mobile.json').then(r => r.json()),
  ]);

  IRIS_X = xData;
  IRIS_Y = yData;

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


mobile_html = build_mobile_html()
with open("saint_graphique_iris_mobile.html", "w", encoding="utf-8") as f:
    f.write(mobile_html)
print(f"Mobile  → saint_graphique_iris_mobile.html ({len(mobile_html)//1024} KB)")
