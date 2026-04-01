# shared_config.py — Configuration et fonctions partagées entre desktop et mobile
import pandas as pd
import numpy as np
import json as _json
import os as _os

# ── CONFIGURATION PARTIS ──────────────────────────────────────────────────────
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
    "RN":"Rassemblement National","LFI":"La France Insoumise","PS":"Parti Socialiste",
    "ENS":"Ensemble / Renaissance","EELV":"Europe Écologie","PCF":"Parti Communiste",
    "LR":"Les Républicains","REC":"Reconquête","AUTRE":"Autres partis",
    "NFP":"Nouveau Front Populaire","NUPES":"NUPES","PS_PP":"PS-Place Publique (Glucksmann)",
    "UG":"Union de la Gauche","UXD":"Alliance LR-RN (Ciotti)","DVD":"Divers droite",
    "DVC":"Divers centre","DVG":"Divers gauche","EXG":"Extrême gauche","EXD":"Extrême droite",
    "DLF":"Debout la France","MODEM":"MoDem","HOR":"Horizons","UDI":"UDI","REG":"Régionalistes",
    "MACRON":"Emmanuel Macron","LE_PEN":"Marine Le Pen","MELENCHON":"Jean-Luc Mélenchon",
    "FILLON":"François Fillon","HAMON":"Benoît Hamon","DUPONT_AIGNAN":"Nicolas Dupont-Aignan",
    "ZEMMOUR":"Éric Zemmour","PECRESSE":"Valérie Pécresse","JADOT":"Yannick Jadot",
    "ROUSSEL":"Fabien Roussel","HIDALGO":"Anne Hidalgo",
    "HOLLANDE":"François Hollande","SARKOZY":"Nicolas Sarkozy","BAYROU":"François Bayrou","JOLY":"Éva Joly",
}
SHORT = {
    "RN":"RN","LFI":"LFI","PS":"PS","ENS":"ENS","EELV":"EELV","PCF":"PCF","LR":"LR","REC":"RCQ",
    "AUTRE":"Autre","NFP":"NFP","NUPES":"NUPES","PS_PP":"PS-PP","UG":"UG","UXD":"UXD",
    "DVD":"DVD","DVC":"DVC","DVG":"DVG","EXG":"EXG","EXD":"EXD","DLF":"DLF","MODEM":"MDM",
    "HOR":"HOR","UDI":"UDI","REG":"REG","MACRON":"Macron","LE_PEN":"Le Pen",
    "MELENCHON":"Mélenchon","FILLON":"Fillon","HAMON":"Hamon","DUPONT_AIGNAN":"DPA",
    "ZEMMOUR":"Zemmour","PECRESSE":"Pécresse","JADOT":"Jadot","ROUSSEL":"Roussel",
    "HIDALGO":"Hidalgo","HOLLANDE":"Hollande","SARKOZY":"Sarkozy","BAYROU":"Bayrou","JOLY":"Joly",
}
OPACITY = {"RN":0.50,"REC":0.50,"LE_PEN":0.50,"ZEMMOUR":0.50,"EXD":0.50,"UXD":0.50}
ORDER = ["LFI","PCF","EELV","PS","ENS","LR","RN","REC","AUTRE"]
ALL_ORDER = [
    "LFI","MELENCHON","PCF","ROUSSEL","EXG","EELV","JADOT","JOLY","PS","DVG","HAMON","HOLLANDE",
    "NFP","NUPES","PS_PP","UG",
    "ENS","MODEM","HOR","UDI","MACRON",
    "LR","DVD","DVC","FILLON","PECRESSE","SARKOZY","BAYROU","UXD",
    "RN","LE_PEN","REC","EXD","DLF","ZEMMOUR","DUPONT_AIGNAN",
    "REG","HIDALGO","AUTRE",
]

SCORES_CONFIG_old = {
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

# ── Nouveaux scores v2 (débiaisés par clustering de corrélation) ──────────────
SCORES_CONFIG_V2_old = {
    # Axe X Saint-Graphique v2 : composition du revenu (capital vs travail)
    'score_composition_capital': {
        'pos_vars': ['DISP_PPAT21', 'DISP_PBEN21', 'P21_NSAL15P_EMPLOY', 'pct_proprietaires', 'pct_logements_anciens'],
        'neg_vars': ['DISP_PTSA21', 'pct_csp_ouvrier', 'pct_csp_employe', 'pct_cdi'],
    },
    # Axe Y partagé Saint-Graphique v2 + Bourdieu v2 : position sociale totale
    'score_domination_v2': {
        'pos_vars': ['DISP_MED21', 'DISP_PPAT21', 'pct_proprietaires', 'pct_sup5', 'pct_bac_plus', 'pct_csp_plus', 'pct_csp_intermediaire', 'pct_cdi'],
        'neg_vars': ['DISP_TP6021', 'pct_sans_diplome', 'pct_capbep', 'pct_csp_ouvrier', 'pct_csp_sans_emploi', 'pct_chomage', 'DISP_PPMINI21'],
    },
    # Axe X Bourdieu v2 : capital économique pur (sans diplôme/CSP)
    'score_cap_eco_v2': {
        'pos_vars': ['DISP_MED21', 'DISP_PPAT21', 'pct_proprietaires', 'surface_moyenne', 'pct_grands_logements'],
        'neg_vars': ['DISP_TP6021', 'DISP_PPMINI21', 'pct_hlm', 'pct_suroccupation'],
    },
    # Axe Y Bourdieu v2 : capital culturel/scolaire (sans revenus)
    'score_cap_cult_v2': {
        'pos_vars': ['pct_sup5', 'pct_bac_plus', 'pct_csp_intermediaire', 'pct_actifs_velo', 'bpe_ecole_privee_pour1000', 'bpe_sport_indoor_pour1000'],
        'neg_vars': ['pct_sans_diplome', 'pct_capbep', 'pct_csp_independant', 'pct_csp_agriculteur', 'pct_actifs_voiture'],
    },
    # Ratio capital/travail pur (v2)
    'score_rentier_v2': {
        'pos_vars': ['DISP_PPAT21', 'DISP_PBEN21', 'pct_proprietaires'],
        'neg_vars': ['DISP_PTSA21', 'pct_cdd', 'pct_interim'],
    },
    # Ruralité v2 (sans pct_immigres)
    'score_ruralite_v2': {
        'pos_vars': ['pct_csp_agriculteur', 'pct_sans_diplome', 'pct_actifs_voiture', 'P21_ACTOCC15P_ILT3'],
        'neg_vars': ['pct_actifs_velo', 'pct_actifs_transports', 'pct_actifs_marche', 'pct_etudiants', 'P21_ACTOCC15P_ILT1', 'bpe_total_pour1000'],
    },
    # Zone de tension péri-urbaine (nouveau)
    'score_periurbain': {
        'pos_vars': ['pct_maison', 'pct_actifs_voiture', 'pct_voiture_2plus', 'P21_ACTOCC15P_ILT3', 'pct_chauffage_fioul', 'pct_hlm'],
        'neg_vars': ['bpe_E_transports_pour1000', 'pct_actifs_transports', 'pct_appart', 'bpe_total_pour1000'],
    },
    # France pavillonnaire — renommé depuis score_peripherie_metropole (nouveau nom, variables inchangées)
    'score_france_pavillonnaire': {
        'pos_vars': ['pct_capbep', 'pct_actifs_voiture', 'pct_maison', 'pct_voiture_2plus', 'nb_pieces_moyen', 'pct_chauffage_fioul'],
        'neg_vars': ['pct_bac_plus', 'pct_sup5', 'pct_csp_plus', 'DISP_GI21', 'DISP_PACT21', 'DISP_RD21', 'pct_actifs_velo', 'DISP_PTSA21', 'pct_studios', 'pct_petits_logements'],
    },
}

# ── Scores grouped (nouvelle architecture : zscore→agrégation→rang centile par groupe) ──
SCORES_CONFIG_GROUPED_old = {

    # ── score_domination ─────────────────────────────────────────────────────────
    'score_domination': [
        {
            'poids': 1.0,
            'vars': {
                'pct_csp_plus':          +1.0,
                'pct_csp_intermediaire': +0.6,
                'pct_csp_independant':   +0.4,
                'pct_employeurs':        +0.5,
                'pct_cdi':               +0.3,
                'pct_csp_ouvrier':       -0.8,
                'pct_csp_employe':       -0.5,
                'pct_csp_sans_emploi':   -1.0,
                'pct_chomage':           -0.5,
                'pct_interim':           -0.4,
                'pct_cdd':               -0.3,
                'pct_temps_partiel':     -0.3,
            }
        }
    ],

    # ── score_composition_capital ─────────────────────────────────────────────────
    'score_composition_capital': [
        {
            'poids': 0.65,
            'vars': {
                'DISP_PPAT21':  +1.0,
                'DISP_PBEN21':  +0.8,
                'DISP_PTSA21':  -1.0,
                'DISP_PCHO21':  -0.3,
            }
        },
        {
            'poids': 0.35,
            'vars': {
                'pct_proprietaires':  +0.8,
                'pct_employeurs':     +0.5,
                'pct_csp_ouvrier':    -0.6,
                'pct_csp_employe':    -0.4,
                'pct_cdi':            -0.4,
                'pct_cdd':            -0.3,
                'pct_temps_partiel':  -0.2,
            }
        }
    ],

    # ── score_cap_eco ─────────────────────────────────────────────────────────────
    'score_cap_eco': [
        {
            'poids': 0.7,
            'vars': {
                'DISP_MED21':    +1.0,
                'DISP_PPAT21':   +0.8,
                'DISP_PBEN21':   +0.5,
                'DISP_TP6021':   -1.0,
                'DISP_PPMINI21': -0.7,
            }
        },
        {
            'poids': 0.3,
            'vars': {
                'pct_proprietaires':  +0.8,
                'pct_hlm':            -0.8,
                'pct_suroccupation':  -0.5,
            }
        }
    ],

    # ── score_cap_cult ────────────────────────────────────────────────────────────
    'score_cap_cult': [
        {
            'poids': 0.8,
            'vars': {
                'pct_sup5':              +1.0,
                'pct_bac_plus':          +0.6,
                'pct_csp_intermediaire': +0.5,
                'pct_actifs_velo':       +0.3,
                'pct_sans_diplome':      -1.0,
                'pct_capbep':            -0.6,
                'pct_csp_independant':   -0.3,
                'pct_csp_agriculteur':   -0.3,
            }
        },
        {
            'poids': 0.2,
            'vars': {
                'bpe_ecole_privee_pour1000':  +0.8,
                'bpe_sport_indoor_pour1000':  +0.4,
            }
        }
    ],

    # ── score_precarite ───────────────────────────────────────────────────────────
    'score_precarite': [
        {
            'poids': 0.6,
            'vars': {
                'DISP_TP6021':   +1.0,
                'DISP_PPMINI21': +0.8,
                'DISP_PPSOC21':  +0.5,
                'DISP_PPLOGT21': +0.4,
                'DISP_MED21':    -1.0,
            }
        },
        {
            'poids': 0.4,
            'vars': {
                'pct_csp_sans_emploi': +0.8,
                'pct_chomage':         +0.7,
                'pct_interim':         +0.5,
                'pct_cdd':             +0.4,
                'pct_temps_partiel':   +0.3,
                'pct_hlm':             +0.4,
                'pct_suroccupation':   +0.5,
            }
        }
    ],

    # ── score_rentier ─────────────────────────────────────────────────────────────
    'score_rentier': [
        {
            'poids': 1.0,
            'vars': {
                'DISP_PPAT21':  +1.0,
                'DISP_PBEN21':  +0.7,
                'DISP_PPEN21':  +0.5,
                'DISP_PTSA21':  -1.0,
                'DISP_PACT21':  -0.6,
            }
        }
    ],

    # score_ruralite retiré de grouped (valeur v1 conservée) — absorbé par score_urbanite

    # ── score_urbanite (amélioré, absorbe ruralité/périphérie) ───────────────────
    'score_urbanite': [
        {
            'poids': 0.6,
            'vars': {
                'pct_appart':            +1.0,
                'pct_voiture_0':         +0.9,
                'pct_actifs_transports': +0.8,
                'pct_locataires':        +0.7,
                'pct_petits_logements':  +0.6,
                'pct_actifs_velo':       +0.5,
                'pct_actifs_marche':     +0.4,
                'pct_maison':            -1.0,
                'pct_voiture_2plus':     -0.9,
                'pct_garage':            -0.7,
                'surface_moyenne':       -0.6,
                'pct_chauffage_fioul':   -0.4,
                'pct_csp_agriculteur':   -0.7,
                'P21_ACTOCC15P_ILT3':   -0.5,
            }
        },
        {
            'poids': 0.4,
            'vars': {
                'bpe_E_transports_pour1000': +0.9,
                'bpe_total_pour1000':        +0.7,
                'P21_ACTOCC15P_ILT1':        +0.5,
                'bpe_B_commerces_pour1000':  +0.4,
            }
        }
    ],

    # ── score_confort_residentiel ─────────────────────────────────────────────────
    'score_confort_residentiel': [
        {
            'poids': 1.0,
            'vars': {
                'pct_proprietaires':     +0.7,
                'pct_grands_logements':  +0.8,
                'surface_moyenne':       +0.7,
                'nb_pieces_moyen':       +0.6,
                'pct_garage':            +0.5,
                'pct_logements_5p_plus': +0.5,
                'pct_suroccupation':     -1.0,
                'pct_hlm':               -0.7,
                'pct_petits_logements':  -0.6,
                'pct_studios':           -0.5,
                'pct_logvac':            -0.3,
            }
        }
    ],

    # ── score_equipement_public ───────────────────────────────────────────────────
    'score_equipement_public': [
        {
            'poids': 1.0,
            'vars': {
                'bpe_D_sante_pour1000':          +1.0,
                'bpe_C_enseignement_pour1000':   +0.8,
                'bpe_A_services_pour1000':        +0.7,
                'bpe_E_transports_pour1000':      +0.7,
                'bpe_F_sports_culture_pour1000':  +0.6,
                'bpe_B_commerces_pour1000':       +0.5,
                'bpe_educ_prioritaire_pour1000':  +0.4,
            }
        }
    ],

    # score_france_pavillonnaire retiré de grouped (valeur v2 conservée) — absorbé par score_urbanite

    # ── Migration v1 → grouped ──────────────────────────────────────────────────

    'score_exploitation': [
        {
            'poids': 0.65,
            'vars': {
                'DISP_PPAT21':   +1.2,
                'DISP_PBEN21':   +1.0,
                'DISP_PPEN21':   +0.6,
                'DISP_MED21':    +0.5,
                'DISP_TP6021':   -1.0,
                'DISP_PTSA21':   -0.7,
                'DISP_PPLOGT21': -0.5,
            }
        },
        {
            'poids': 0.35,
            'vars': {
                'pct_csp_plus':       +0.8,
                'pct_csp_retraite':   +0.5,
                'P21_NSAL15P_EMPLOY': +0.7,
                'P21_NSAL15P_AIDFAM': -0.6,
            }
        }
    ],

    # ── Scores ACP — conversion grouped des approximations v1 PC1–PC8 ──────────

    'score_pca_pc1_logement_confort': [
        {
            'poids': 1.0,
            'vars': {
                'pct_grands_logements':  +1.0,
                'pct_garage':            +0.9,
                'pct_actifs_voiture':    +0.8,
                'pct_logements_5p_plus': +0.8,
                'nb_pieces_moyen':       +0.8,
                'pct_voiture_2plus':     +0.9,
                'pct_proprietaires':     +0.7,
                'pct_maison':            +0.9,
                'surface_moyenne':       +0.8,
                'pct_appart':            -1.0,
                'pct_locataires':        -0.8,
                'pct_voiture_0':         -0.9,
                'pct_immigres':          -0.5,
                'pct_petits_logements':  -0.8,
                'pct_actifs_transports': -0.7,
                'pct_etrangers':         -0.4,
                'pct_studios':           -0.7,
            }
        }
    ],

    'score_pca_pc2_composition_diplomes': [
        {
            'poids': 1.0,
            'vars': {
                'pct_capbep':       +0.8,
                'pct_interim':      +0.8,
                'pct_chomage':      +0.9,
                'pct_csp_ouvrier':  +0.9,
                'DISP_TP6021':      +1.0,
                'DISP_PPLOGT21':    +0.7,
                'DISP_PPMINI21':    +0.8,
                'DISP_PPFAM21':     +0.6,
                'DISP_PPSOC21':     +0.7,
                'pct_sans_diplome': +0.8,
                'DISP_PIMPOT21':    -0.6,
                'DISP_MED21':       -1.0,
                'pct_bac_plus':     -0.8,
                'pct_csp_plus':     -0.8,
                'pct_sup5':         -0.9,
                'DISP_PACT21':      -0.7,
                'DISP_PTSA21':      -0.7,
            }
        }
    ],

    'score_pca_pc3_equipements_demographie': [
        {
            'poids': 1.0,
            'vars': {
                'pct_csp_intermediaire':    +0.7,
                'DISP_PACT21':              +0.8,
                'DISP_PTSA21':              +0.7,
                'pct_0_19':                 +0.6,
                'pct_65_plus':              -1.0,
                'age_moyen':                -0.9,
                'DISP_PPEN21':              -0.7,
                'bpe_total_pour1000':       -0.8,
                'pct_csp_retraite':         -0.8,
                'bpe_B_commerces_pour1000': -0.6,
                'bpe_G_tourisme_pour1000':  -0.5,
                'bpe_D_sante_pour1000':     -0.5,
                'pct_actifs_marche':        -0.4,
                'bpe_A_services_pour1000':  -0.5,
            }
        }
    ],

    'score_pca_pc4_demographie_chauffage': [
        {
            'poids': 1.0,
            'vars': {
                'age_moyen':                   +0.8,
                'pct_csp_retraite':            +0.8,
                'pct_65_plus':                 +0.9,
                'DISP_PPEN21':                 +0.6,
                'pct_chauffage_gaz_ville':     +0.5,
                'pct_femmes':                  +0.4,
                'pct_logements_anciens':       -0.7,
                'bpe_A_services_pour1000':     -0.5,
                'pct_20_64':                   -0.8,
                'pct_chauffage_autre':         -0.4,
                'pct_chauffage_gaz_bouteille': -0.4,
                'pct_csp_agriculteur':         -0.5,
                'bpe_total_pour1000':          -0.6,
                'bpe_F_sports_culture_pour1000': -0.5,
                'pct_chauffage_fioul':         -0.5,
            }
        }
    ],

    'score_pca_pc5_equipements_csp': [
        {
            'poids': 1.0,
            'vars': {
                'pct_grands_logements':    +0.6,
                'pct_csp_sans_emploi':     +0.7,
                'DISP_S80S2021':           +0.6,
                'pct_temps_partiel':       +0.5,
                'pct_inactif':             +0.5,
                'pct_etudiants':           +0.5,
                'bpe_total_pour1000':      -0.9,
                'bpe_A_services_pour1000': -0.7,
                'bpe_B_commerces_pour1000': -0.6,
                'pct_csp_employe':         -0.5,
                'bpe_D_sante_pour1000':    -0.6,
                'pct_chauffage_elec':      -0.4,
                'pct_logements_recents':   -0.5,
                'pct_csp_intermediaire':   -0.6,
            }
        }
    ],

    'score_pca_pc6_equipements_diplomes': [
        {
            'poids': 1.0,
            'vars': {
                'pct_inactif':                 +0.5,
                'bpe_sport_indoor_pour1000':   +0.7,
                'bpe_ecole_privee_pour1000':   +0.7,
                'bpe_C_enseignement_pour1000': +0.6,
                'pct_hors_menage':             +0.5,
                'pct_etudiants':               +0.5,
                'bpe_E_transports_pour1000':   -0.7,
                'pct_immigres':                -0.5,
                'pct_etrangers':               -0.4,
                'pct_cdi':                     -0.5,
                'pct_actifs_transports':       -0.6,
                'pct_actifs_2roues':           -0.4,
                'pct_csp_independant':         -0.5,
                'DISP_S80S2021':               -0.5,
            }
        }
    ],

    'score_pca_pc7_logement_csp': [
        {
            'poids': 1.0,
            'vars': {
                'DISP_S80S2021':           +0.7,
                'pct_logements_recents':   +0.7,
                'DISP_GI21':               +0.6,
                'bpe_A_services_pour1000': +0.5,
                'bpe_total_pour1000':      +0.5,
                'pct_csp_sans_emploi':     +0.5,
                'pct_inactif':             +0.4,
                'pct_csp_independant':     +0.4,
                'pct_0_19':                +0.4,
                'DISP_PPAT21':             +0.5,
                'pct_logvac':              -0.5,
                'pct_logements_anciens':   -0.7,
                'pct_csp_agriculteur':     -0.4,
                'pct_20_64':               -0.5,
                'pct_actifs_velo':         -0.4,
            }
        }
    ],

    'score_pca_pc8_equipements_logement': [
        {
            'poids': 1.0,
            'vars': {
                'pct_cdd':                       +0.5,
                'pct_logements_recents':         +0.6,
                'pct_chauffage_elec':            +0.5,
                'bpe_sport_indoor_pour1000':     -0.8,
                'bpe_C_enseignement_pour1000':   -0.7,
                'bpe_ecole_privee_pour1000':     -0.7,
                'pct_chauffage_gaz_ville':       -0.6,
                'bpe_F_sports_culture_pour1000': -0.6,
                'pct_logvac':                    -0.5,
                'bpe_total_pour1000':            -0.7,
                'bpe_D_sante_pour1000':          -0.6,
            }
        }
    ],

    # ── Migration v2 → grouped ──────────────────────────────────────────────────

    'score_domination_v2': [
        {
            'poids': 0.5,
            'vars': {
                'DISP_MED21':        +1.0,
                'DISP_PPAT21':       +0.8,
                'pct_proprietaires': +0.5,
                'DISP_TP6021':       -1.0,
                'DISP_PPMINI21':     -0.8,
            }
        },
        {
            'poids': 0.5,
            'vars': {
                'pct_sup5':              +1.0,
                'pct_bac_plus':          +0.7,
                'pct_csp_plus':          +0.8,
                'pct_csp_intermediaire': +0.6,
                'pct_cdi':               +0.4,
                'pct_sans_diplome':      -1.0,
                'pct_capbep':            -0.7,
                'pct_csp_ouvrier':       -0.8,
                'pct_csp_sans_emploi':   -0.7,
                'pct_chomage':           -0.6,
            }
        }
    ],

    'score_cap_eco_v2': [
        {
            'poids': 0.65,
            'vars': {
                'DISP_MED21':    +1.0,
                'DISP_PPAT21':   +0.8,
                'DISP_TP6021':   -1.0,
                'DISP_PPMINI21': -0.8,
            }
        },
        {
            'poids': 0.35,
            'vars': {
                'pct_proprietaires':    +0.8,
                'surface_moyenne':      +0.7,
                'pct_grands_logements': +0.6,
                'pct_hlm':              -0.9,
                'pct_suroccupation':    -0.6,
            }
        }
    ],

    'score_cap_cult_v2': [
        {
            'poids': 0.75,
            'vars': {
                'pct_sup5':              +1.0,
                'pct_bac_plus':          +0.7,
                'pct_csp_intermediaire': +0.5,
                'pct_actifs_velo':       +0.4,
                'pct_sans_diplome':      -1.0,
                'pct_capbep':            -0.7,
                'pct_csp_independant':   -0.4,
                'pct_csp_agriculteur':   -0.3,
                'pct_actifs_voiture':    -0.5,
            }
        },
        {
            'poids': 0.25,
            'vars': {
                'bpe_ecole_privee_pour1000': +0.8,
                'bpe_sport_indoor_pour1000': +0.5,
            }
        }
    ],

    'score_rentier_v2': [
        {
            'poids': 1.0,
            'vars': {
                'DISP_PPAT21':       +1.0,
                'DISP_PBEN21':       +0.8,
                'pct_proprietaires': +0.6,
                'DISP_PTSA21':       -1.0,
                'pct_cdd':           -0.5,
                'pct_interim':       -0.6,
            }
        }
    ],

    # ── Transition énergétique ───────────────────────────────────────────────────

    'score_dependance_carbone': [
        {
            'poids': 0.55,
            'vars': {
                'pct_actifs_voiture':    +1.0,
                'pct_voiture_2plus':     +0.8,
                'pct_actifs_transports': -0.9,
                'pct_actifs_velo':       -0.7,
                'pct_actifs_marche':     -0.5,
            }
        },
        {
            'poids': 0.45,
            'vars': {
                'pct_chauffage_fioul':         +1.0,
                'pct_chauffage_gaz_bouteille': +0.6,
                'pct_logements_recents':       -0.6,
                'pct_chauffage_elec':          -0.5,
            }
        }
    ],
}

# ── SCORES CONFIG PCA ─────────────────────────────────────────────────────────
# Poids = vestige théorique. La vraie méthode est PCA (make_score_pca_grouped).
SCORES_CONFIG_GROUPED_PCA = {
    'score_domination': [
        {'poids': 1.0, 'vars': {
            'pct_csp_plus': +1.0, 'pct_csp_intermediaire': +0.6,
            'pct_csp_ouvrier': -0.8, 'pct_csp_employe': -0.6, 'pct_csp_sans_emploi': -1.0,
            'pct_sup5': +0.7, 'pct_bac_plus': +0.4,
            'pct_sans_diplome': -0.8, 'pct_capbep': -0.5,
            'pct_cdi': +0.8,
            'pct_interim': -0.9, 'pct_cdd': -0.7, 'pct_temps_partiel': -0.5,
        }}
    ],
    'score_exploitation': [
        {'poids': 0.75, 'vars': {
            'DISP_PPAT21': +1.5, 'DISP_PBEN21': +1.0, 'DISP_PCHO21': -0.5,
        }},
        {'poids': 0.25, 'vars': {
            'pct_proprietaires': +1.5, 'pct_employeurs': +0.8,
            'pct_csp_ouvrier': -0.8, 'pct_csp_employe': -0.5,
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
        {'poids': 1.0, 'vars': {
            'pct_sup5': +1.0, 'pct_bac_plus': +0.6, 'pct_csp_intermediaire': +0.5,
            'pct_actifs_velo': +0.3,
            'pct_sans_diplome': -1.0, 'pct_capbep': -0.6, 'pct_csp_agriculteur': -0.3,
        }},
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
            'DISP_PPAT21': +1.0, 'DISP_PPEN21': +0.5,
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
            'bpe_total_pour1000': +0.7, 'bpe_B_commerces_pour1000': +0.4,
        }}
    ],
    'score_confort_residentiel': [
        {'poids': 1.0, 'vars': {
            'pct_proprietaires': +0.7, 'pct_grands_logements': +0.8,
            'surface_moyenne': +0.7, 'nb_pieces_moyen': +0.6, 'pct_garage': +0.5,
            'pct_logements_5p_plus': +0.5,
            'pct_suroccupation': -1.0, 'pct_hlm': -0.7, 'pct_petits_logements': -0.6,
            'pct_studios': -0.5,
        }}
    ],
    'score_equipement_public': [
        {'poids': 1.0, 'vars': {
            'bpe_D_sante_pour1000': +1.0, 'bpe_C_enseignement_pour1000': +0.8,
            'bpe_A_services_pour1000': +0.7,
            'bpe_F_sports_culture_pour1000': +0.6, 'bpe_B_commerces_pour1000': +0.5,
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
    'score_domination':           'pct_csp_plus',
    'score_exploitation':         'DISP_PPAT21',
    'score_cap_eco':              'DISP_MED21',
    'score_cap_cult':             'pct_sup5',
    'score_precarite':            'DISP_TP6021',
    'score_rentier':              'DISP_PPAT21',
    'score_urbanite':             'pct_appart',
    'score_confort_residentiel':  'surface_moyenne',
    'score_equipement_public':    'bpe_D_sante_pour1000',
    'score_dependance_carbone':   'pct_actifs_voiture',
}

# ── ELECTIONS ─────────────────────────────────────────────────────────────────
ELECTIONS_AVAILABLE = {
    '2022_legi_t1': {'type':'legi','year':2022,'tour':1,'label':'Législatives 2022 — 1er tour'},
    '2022_legi_t2': {'type':'legi','year':2022,'tour':2,'label':'Législatives 2022 — 2e tour'},
    '2024_legi_t1': {'type':'legi','year':2024,'tour':1,'label':'Législatives 2024 — 1er tour'},
    '2024_legi_t2': {'type':'legi','year':2024,'tour':2,'label':'Législatives 2024 — 2e tour'},
    '2017_legi_t1': {'type':'legi','year':2017,'tour':1,'label':'Législatives 2017 — 1er tour'},
    '2017_legi_t2': {'type':'legi','year':2017,'tour':2,'label':'Législatives 2017 — 2e tour'},
    '2012_legi_t1': {'type':'legi','year':2012,'tour':1,'label':'Législatives 2012 — 1er tour'},
    '2012_legi_t2': {'type':'legi','year':2012,'tour':2,'label':'Législatives 2012 — 2e tour'},
    '2024_euro_t1': {'type':'euro','year':2024,'tour':1,'label':'Européennes 2024'},
    '2019_euro_t1': {'type':'euro','year':2019,'tour':1,'label':'Européennes 2019'},
    '2014_euro_t1': {'type':'euro','year':2014,'tour':1,'label':'Européennes 2014'},
    '2022_pres_t1': {'type':'pres','year':2022,'tour':1,'label':'Présidentielles 2022 — 1er tour'},
    '2022_pres_t2': {'type':'pres','year':2022,'tour':2,'label':'Présidentielles 2022 — 2e tour'},
    '2017_pres_t1': {'type':'pres','year':2017,'tour':1,'label':'Présidentielles 2017 — 1er tour'},
    '2017_pres_t2': {'type':'pres','year':2017,'tour':2,'label':'Présidentielles 2017 — 2e tour'},
    '2012_pres_t1': {'type':'pres','year':2012,'tour':1,'label':'Présidentielles 2012 — 1er tour'},
    '2012_pres_t2': {'type':'pres','year':2012,'tour':2,'label':'Présidentielles 2012 — 2e tour'},
    '2014_muni_t1': {'type':'muni','year':2014,'tour':1,'label':'Municipales 2014 — 1er tour'},
    '2014_muni_t2': {'type':'muni','year':2014,'tour':2,'label':'Municipales 2014 — 2e tour'},
    '2020_muni_t1': {'type':'muni','year':2020,'tour':1,'label':'Municipales 2020 — 1er tour'},
    '2020_muni_t2': {'type':'muni','year':2020,'tour':2,'label':'Municipales 2020 — 2e tour'},
    '2026_muni_t1': {'type':'muni','year':2026,'tour':1,'label':'Municipales 2026 — 1er tour'},
    '2026_muni_t2': {'type':'muni','year':2026,'tour':2,'label':'Municipales 2026 — 2e tour'},
}
DEFAULT_ELECTION = '2022_legi_t1'

ALL_PARTIES_COLORS = {
    'RN':'#374151','LFI':'#DC2626','PS':'#EC4899','ENS':'#F97316','EELV':'#16A34A',
    'PCF':'#9B1C1C','LR':'#1D4ED8','REC':'#0F172A','AUTRE':'#9CA3AF',
    'NFP':'#B91C1C','NUPES':'#B91C1C','PS_PP':'#E86BA8','UG':'#C84B6E',
    'UXD':'#7C3AED','DVD':'#60A5FA','DVC':'#A78BFA','DLF':'#4B5563','EXD':'#1F2937',
    'MODEM':'#FB923C','HOR':'#FDBA74','UDI':'#93C5FD',
    'DVG':'#F9A8D4','EXG':'#7F1D1D','REG':'#89712F',
    'MACRON':'#F97316','LE_PEN':'#374151','MELENCHON':'#DC2626','FILLON':'#1D4ED8',
    'HAMON':'#EC4899','DUPONT_AIGNAN':'#6B7280',
    'HOLLANDE':'#EC4899','SARKOZY':'#1D4ED8','BAYROU':'#FB923C','JOLY':'#16A34A',
    'ZEMMOUR':'#0F172A','PECRESSE':'#3B82F6','JADOT':'#16A34A','ROUSSEL':'#9B1C1C','HIDALGO':'#DB2777',
}

VOTE_PARTIES_JS = [
    {'key': f'score_{g}', 'label': SHORT.get(g, g), 'color': ALL_PARTIES_COLORS.get(g, '#9CA3AF')}
    for g in ALL_ORDER
]

# ── FONCTIONS DE SCORES ───────────────────────────────────────────────────────
# df et pop sont passés explicitement (pas de globals)

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
    return result.fillna(50.0) - 50.0


def _zscore_pondere(series, pop):
    """Zscore pondéré par population. NaN remplacés par 0."""
    s = series.copy().astype(float)
    p = pop.copy().astype(float)
    valid = s.notna() & p.notna() & (p > 0)
    if valid.sum() < 10:
        return pd.Series(0.0, index=s.index)
    s_v = s[valid]
    p_v = p[valid]
    w = p_v / p_v.sum()
    w_mean = (s_v * w).sum()
    w_std = np.sqrt(((s_v - w_mean) ** 2 * w).sum())
    if w_std < 1e-10:
        return pd.Series(0.0, index=s.index)
    result = pd.Series(np.nan, index=s.index)
    result[valid] = (s_v - w_mean) / w_std
    return result.fillna(0.0)


def _pca_weighted_pc1(X, pop_array):
    """PCA pondérée population sur matrice X (n_iris × n_vars). Retourne (pc1, variance_explained)."""
    w = np.where(np.isfinite(pop_array) & (pop_array > 0), pop_array, 1.0)
    w_norm = w / w.sum()
    means = (X * w_norm[:, None]).sum(axis=0)
    Xc = X - means
    C = (Xc * w_norm[:, None]).T @ Xc
    eigenvalues, eigenvectors = np.linalg.eigh(C)
    idx = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    var_exp = eigenvalues[0] / eigenvalues.sum() if eigenvalues.sum() > 1e-10 else 0.0
    pc1 = Xc @ eigenvectors[:, 0]
    return pc1, var_exp


LOG_TRANSFORM_VARS = {'DISP_MED21', 'surface_moyenne'}


def make_score_grouped(groupes, df):
    """Score grouped : zscore → agrégation → rang centile (méthode classique)."""
    pop = df['_pop']
    rangs_groupes = []
    poids_groupes = []
    for groupe in groupes:
        var_poids = groupe['vars']
        poids_groupe = groupe['poids']
        parts = []
        total_poids = sum(abs(p) for p in var_poids.values())
        if total_poids == 0:
            continue
        for var, poids in var_poids.items():
            if var not in df.columns:
                continue
            z = _zscore_pondere(df[var], pop)
            parts.append((poids / total_poids) * z)
        if not parts:
            continue
        indice_groupe = pd.concat(parts, axis=1).sum(axis=1)
        rangs_groupes.append(_rang_pondere(indice_groupe, pop))
        poids_groupes.append(poids_groupe)
    if not rangs_groupes:
        return pd.Series(0.0, index=df.index)
    total_poids_groupes = sum(poids_groupes)
    score_final = sum(r * p for r, p in zip(rangs_groupes, poids_groupes))
    return score_final / total_poids_groupes


def make_score_pca_grouped(groupes, df, anchor_var=None, min_variance_explained=0.35):
    """
    Score composite PCA à deux étages.
    Étage 1 : PCA pondérée sur les zscores de chaque groupe → PC1.
    Étage 2 : PCA pondérée sur les PC1 des groupes → score final.
    Retourne (score_series, diagnostics_dict).
    """
    pop = df['_pop']
    group_scores = []
    diags = []

    for i, groupe in enumerate(groupes):
        vars_disponibles = [v for v in groupe['vars'] if v in df.columns]
        if not vars_disponibles:
            continue
        X_parts = []
        for var in vars_disponibles:
            s = pd.to_numeric(df[var], errors='coerce')
            if var in LOG_TRANSFORM_VARS:
                s = s.clip(lower=1e-6).apply(np.log)
            z = _zscore_pondere(s, pop)
            X_parts.append(z.values)
        X = np.column_stack(X_parts)
        pc1, var_exp = _pca_weighted_pc1(X, pop.values)
        diags.append({'groupe': i, 'n_vars': len(vars_disponibles), 'var_exp': var_exp})
        if var_exp < min_variance_explained:
            print(f"  [WARN] Groupe {i}: PC1 explique seulement {var_exp*100:.1f}% de variance")
        group_scores.append(pd.Series(pc1, index=df.index))

    if not group_scores:
        return pd.Series(0.0, index=df.index), {}

    if len(group_scores) == 1:
        final_pc1 = group_scores[0].values
        final_var_exp = diags[0]['var_exp']
    else:
        G = np.column_stack([_zscore_pondere(s, pop).values for s in group_scores])
        final_pc1, final_var_exp = _pca_weighted_pc1(G, pop.values)
        if final_var_exp < min_variance_explained:
            print(f"  [WARN] Score final: PC1 explique seulement {final_var_exp*100:.1f}% de variance")

    score = pd.Series(final_pc1, index=df.index)
    if anchor_var and anchor_var in df.columns:
        if score.corr(pd.to_numeric(df[anchor_var], errors='coerce')) < 0:
            score = -score
    return _rang_pondere(score, pop), {'group_diags': diags, 'final_var_exp': final_var_exp}


_EMBEDDING_VARS_PCA = [
    'pct_etrangers','pct_immigres','age_moyen','pct_femmes',
    'taille_menage_moy','pct_hors_menage','ecart_csp_plus_hf',
    'pct_0_19','pct_20_64','pct_65_plus',
    'pct_csp_agriculteur','pct_csp_independant','pct_csp_plus',
    'pct_csp_intermediaire','pct_csp_employe','pct_csp_ouvrier',
    'pct_csp_retraite','pct_csp_sans_emploi',
    'DISP_MED21','DISP_TP6021','DISP_GI21','DISP_RD21','DISP_S80S2021',
    'DISP_PTSA21','DISP_PPAT21','DISP_PPEN21','DISP_PPSOC21',
    'DISP_PCHO21','DISP_PPFAM21','DISP_PPLOGT21','DISP_PPMINI21',
    'DISP_PIMPOT21','DISP_PACT21',
    'pct_sup5','pct_sans_diplome','pct_capbep','pct_bac_plus',
    'pct_chomage','pct_cdi','pct_cdd','pct_interim',
    'pct_temps_partiel','pct_inactif','pct_etudiants',
    'pct_actifs_voiture','pct_actifs_transports','pct_actifs_velo',
    'pct_actifs_2roues','pct_actifs_marche',
    'pct_proprietaires','pct_locataires','pct_hlm','pct_logvac',
    'pct_maison','pct_appart','pct_petits_logements','pct_grands_logements',
    'pct_logements_anciens','pct_logements_recents',
    'pct_voiture_0','pct_voiture_2plus','surface_moyenne','pct_suroccupation',
    'pct_chauffage_elec','pct_chauffage_fioul','pct_chauffage_gaz_ville',
    'pct_chauffage_gaz_bouteille','pct_chauffage_autre',
    'pct_garage','nb_pieces_moyen','pct_studios','pct_logements_5p_plus',
    'bpe_total_pour1000','bpe_A_services_pour1000','bpe_B_commerces_pour1000',
    'bpe_C_enseignement_pour1000','bpe_D_sante_pour1000',
    'bpe_E_transports_pour1000','bpe_F_sports_culture_pour1000',
    'bpe_G_tourisme_pour1000','bpe_educ_prioritaire_pour1000',
    'bpe_ecole_privee_pour1000','bpe_sport_indoor_pour1000','pct_sport_accessible',
]


def compute_pca_vraie(df, n_components=8):
    """
    Calcule les vraies ACP pondérées par population.
    Ajoute df['score_pca_1'] .. df['score_pca_{n_components}'] en place.
    """
    var_names = [v for v in _EMBEDDING_VARS_PCA if v in df.columns]
    print(f"  ACP vraie : {len(var_names)} variables disponibles sur {len(_EMBEDDING_VARS_PCA)}")
    if len(var_names) < 5:
        print("  ACP vraie : pas assez de variables, score_pca_1..8 mis à 0")
        for k in range(1, n_components + 1):
            df[f'score_pca_{k}'] = 0.0
        return

    pop = df['_pop'].values.astype(float)
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
    for k in range(n_components):
        col_name = f'score_pca_{k+1}'
        s = pd.Series(X_pca[:, k], index=df.index)
        df[col_name] = _rang_pondere(s, pop_series)
        loadings = pd.Series(eigenvectors[:, k], index=var_names)
        top_pos = loadings.nlargest(3).index.tolist()
        top_neg = loadings.nsmallest(3).index.tolist()
        print(f"  {col_name}: + {top_pos} / − {top_neg}")


def compute_scores_pca(df):
    """Calcule tous les scores SCORES_CONFIG_GROUPED_PCA et les ajoute au df en place."""
    for score_name, groupes in SCORES_CONFIG_GROUPED_PCA.items():
        anchor = SCORES_PCA_ANCHORS.get(score_name)
        df[score_name], diag = make_score_pca_grouped(groupes, df, anchor_var=anchor)
        fve = diag.get('final_var_exp', 0)
        print(f"  {score_name}: var_exp={fve*100:.1f}%  mean={df[score_name].mean():.2f}")


# ── PRESETS D'AXES ────────────────────────────────────────────────────────────
AXIS_PRESETS = [
    {
        'id': 'saint_graphique',
        'label': 'Domination × Exploitation',
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
            'title': 'Domination × Exploitation — Position de classe',
            'x': "<b>Axe X — Position dans le rapport capital/travail</b> : d'où vient le revenu de l'IRIS ? À droite : zones <em>exploiteuses</em> — revenus patrimoniaux, bénéfices, employeurs, cadres supérieurs. À gauche : zones <em>exploitées</em> — salaires, ouvriers, employés, sans diplôme, pauvreté.",
            'y': '<b>Axe Y — Domination sociale</b> : position dans la hiérarchie sociale totale combinant capital économique et culturel. En haut : dominants (revenus élevés, diplômés, cadres sup.). En bas : dominés (chômage, minimas sociaux, faibles revenus).',
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
            'title': 'Espace bourdieusien — Capital économique × Capital culturel',
            'x': "<b>Axe X — Capital économique</b> : revenu médian, patrimoine (DISP_PPAT21), bénéfices (DISP_PBEN21), pensions, propriétaires, taille des logements. À droite : zones aisées. À gauche : zones pauvres (taux de pauvreté, minimas sociaux, faibles revenus d'activité).",
            'y': "<b>Axe Y — Capital culturel</b> : diplômes (BAC+5), % cadres supérieurs et professions intermédiaires, pratiques culturelles (vélo urbain). En haut : forte proportion de diplômés et cadres. En bas : zones peu qualifiées (sans diplôme, CAP-BEP, ouvriers).",
            'quadrants': {
                'tl': '<b>Intellectuels déclassés</b> — Diplômés mais aux revenus modestes : enseignants, chercheurs, travailleurs du secteur public.',
                'tr': '<b>Élite intégrée</b> — Riches ET diplômés : grandes écoles, professions libérales, hauts fonctionnaires.',
                'bl': '<b>Classe populaire</b> — Faibles capitaux économique et culturel : zones ouvrières, quartiers populaires.',
                'br': '<b>Bourgeoisie patrimoniale</b> — Aisés mais peu diplômés : artisans propriétaires, commerçants, rentiers.',
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
            'y': '<b>Axe Y — Domination sociale</b> : hiérarchie sociale totale combinant capital économique, culturel et position professionnelle. En haut : dominants (revenus élevés, diplômés, cadres sup.). En bas : dominés.',
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
        'label': 'Territoire × Précarité',
        'emoji': '🌾',
        'xVar': 'score_urbanite', 'xInvert': True,
        'yVar': 'score_precarite',
        'xTitle': '← Urbain dense (transports, apparts) ─── Axe territorial ─── Rural / pavillonnaire (voiture, maison) →',
        'yTitle': '← Sécurisé ─── Score précarité ─── Précaire →',
        'xRange': [-55.0, 55.0], 'yRange': [-55.0, 55.0],
        'corners': [
            {'pos': 'bl', 'text': 'URBAIN<br>SÉCURISÉ', 'color': '#6B8FD4'},
            {'pos': 'br', 'text': 'RURAL/PAVILLONNAIRE<br>SÉCURISÉ', 'color': '#059669'},
            {'pos': 'tl', 'text': 'URBAIN<br>PRÉCAIRE', 'color': '#E87070'},
            {'pos': 'tr', 'text': 'RURAL/PAVILLONNAIRE<br>PRÉCAIRE', 'color': '#C49A30'},
        ],
        'desc': {
            'title': 'Territoire × Précarité sociale',
            'x': '<b>Axe X — Score urbanité (inversé)</b> : fusion des anciens scores ruralité / urbanité / périphérie-métropole / France pavillonnaire. À droite : habitat pavillonnaire, voiture, maison, agriculteurs, navettes longues. À gauche : urbain dense, transports, appartements, services de proximité.',
            'y': '<b>Axe Y — Score précarité</b> : chômage, minimas sociaux, taux de pauvreté (haut = plus précaire).',
            'quadrants': {
                'tl': '<b>Urbain précaire</b> — Grands ensembles denses, chômage élevé, minimas sociaux.',
                'tr': '<b>Rural/périurbain précaire</b> — Zones pavillonnaires ou rurales en difficulté économique.',
                'bl': '<b>Urbain sécurisé</b> — Centre-ville dense, cadres, fonctionnaires, emploi stable.',
                'br': '<b>Rural/périurbain sécurisé</b> — Pavillonnaire aisé, retraités propriétaires, bourgs ruraux stables.',
            }
        }
    },
    {
        'id': 'urbanisme', 'label': 'Urbanisme', 'emoji': '\U0001f3d7\ufe0f',
        'xVar': 'score_urbanite', 'xInvert': False,
        'yVar': 'score_equipement_public',
        'xTitle': '\u2190 Rural / Pavillonnaire \u2500\u2500\u2500 Urbanit\u00e9 \u2500\u2500\u2500 Urbain dense \u2192',
        'yTitle': '\u2190 Sous-\u00e9quip\u00e9 \u2500\u2500\u2500 \u00c9quipement public \u2500\u2500\u2500 Bien \u00e9quip\u00e9 \u2192',
        'xRange': [-50, 50], 'yRange': [-50, 50],
        'corners': [
            {'pos': 'tl', 'text': 'RURAL<br>\u00c9QUIP\u00c9', 'color': '#6B8FD4'},
            {'pos': 'tr', 'text': 'URBAIN<br>\u00c9QUIP\u00c9', 'color': '#E87070'},
            {'pos': 'bl', 'text': 'RURAL<br>SOUS-\u00c9QUIP\u00c9', 'color': '#C49A30'},
            {'pos': 'br', 'text': 'URBAIN<br>SOUS-\u00c9QUIP\u00c9', 'color': '#60B87A'},
        ],
        'desc': {
            'title': 'Urbanisme \u2014 Urbanit\u00e9 \u00d7 \u00c9quipement public',
            'x': '<b>Axe X \u2014 Score d\'urbanit\u00e9</b> : composite int\u00e9grant type de logement, mode de chauffage, motorisation et transports. Gauche = rural/pavillonnaire, droite = urbain dense.',
            'y': '<b>Axe Y \u2014 Score d\'\u00e9quipement public</b> : composite sant\u00e9, enseignement, sport, commerces, services pour 1000 hab.',
            'quadrants': {
                'tl': '<b>Rural \u00e9quip\u00e9</b> \u2014 Bourgs-centres avec bons services malgr\u00e9 l\'habitat pavillonnaire.',
                'tr': '<b>Urbain \u00e9quip\u00e9</b> \u2014 C\u0153urs de ville denses et bien dot\u00e9s.',
                'bl': '<b>Rural sous-\u00e9quip\u00e9</b> \u2014 P\u00e9riurbain \u00e9loign\u00e9, d\u00e9serts de services.',
                'br': '<b>Urbain sous-\u00e9quip\u00e9</b> \u2014 Quartiers denses type grands ensembles, peu de services de proximit\u00e9.',
            }
        }
    },
    {
        'id': 'confort', 'label': 'Confort r\u00e9sidentiel', 'emoji': '\U0001f3e0',
        'xVar': 'score_confort_residentiel', 'xInvert': False,
        'yVar': 'score_precarite',
        'xTitle': '\u2190 Parc d\u00e9grad\u00e9 \u2500\u2500\u2500 Confort r\u00e9sidentiel \u2500\u2500\u2500 Parc confortable \u2192',
        'yTitle': '\u2190 Ais\u00e9 \u2500\u2500\u2500 Pr\u00e9carit\u00e9 \u2500\u2500\u2500 Pr\u00e9caire \u2192',
        'xRange': [-50, 50], 'yRange': [-50, 50],
        'corners': [
            {'pos': 'tl', 'text': 'D\u00c9GRAD\u00c9<br>PR\u00c9CAIRE', 'color': '#E87070'},
            {'pos': 'tr', 'text': 'CONFORTABLE<br>PR\u00c9CAIRE', 'color': '#C49A30'},
            {'pos': 'bl', 'text': 'D\u00c9GRAD\u00c9<br>AIS\u00c9', 'color': '#60B87A'},
            {'pos': 'br', 'text': 'CONFORTABLE<br>AIS\u00c9', 'color': '#6B8FD4'},
        ],
        'desc': {
            'title': 'Confort r\u00e9sidentiel \u00d7 Pr\u00e9carit\u00e9',
            'x': '<b>Axe X \u2014 Confort r\u00e9sidentiel</b> : composite propri\u00e9t\u00e9, surface, garage, pi\u00e8ces vs suroccupation, HLM, studios, vacance.',
            'y': '<b>Axe Y \u2014 Pr\u00e9carit\u00e9</b> : composite ch\u00f4mage, prestations sociales, taux de pauvret\u00e9 vs revenu m\u00e9dian.',
            'quadrants': {
                'tl': '<b>Parc d\u00e9grad\u00e9 & pr\u00e9caire</b> \u2014 HLM, suroccupation, ch\u00f4mage : quartiers populaires en difficult\u00e9.',
                'tr': '<b>Confortable & pr\u00e9caire</b> \u2014 Propri\u00e9taires modestes, p\u00e9riurbain peu cher mais d\u00e9pendant de la voiture.',
                'bl': '<b>D\u00e9grad\u00e9 & ais\u00e9</b> \u2014 Studios \u00e9tudiants, petits logements en centre-ville ais\u00e9.',
                'br': '<b>Confortable & ais\u00e9</b> \u2014 Quartiers r\u00e9sidentiels bourgeois, grandes propri\u00e9t\u00e9s.',
            }
        }
    },
    {
        'id': 'energie', 'label': 'Transition énergétique', 'emoji': '⚡',
        'xVar': 'score_dependance_carbone', 'xInvert': False,
        'yVar': 'score_urbanite',
        'xTitle': '← Faible dépendance fossile ─── Score dépendance carbone ─── Fort dépendance fossile →',
        'yTitle': '← Rural / pavillonnaire ─── Score urbanité ─── Urbain dense →',
        'xRange': [-50, 50], 'yRange': [-50, 50],
        'corners': [
            {'pos': 'tl', 'text': 'RURAL<br>SOBRE', 'color': '#059669'},
            {'pos': 'tr', 'text': 'PAVILLONNAIRE<br>CARBONÉ', 'color': '#DC2626'},
            {'pos': 'bl', 'text': 'URBAIN<br>SOBRE', 'color': '#1D4ED8'},
            {'pos': 'br', 'text': 'URBAIN<br>CARBONÉ', 'color': '#F59E0B'},
        ],
        'desc': {
            'title': 'Transition énergétique — Dépendance carbone × Urbanité',
            'x': "<b>Axe X — Score dépendance carbone</b> : composite mobilité fossile (voiture, 2+ voitures vs transports, vélo, marche) + chauffage fossile (fioul, gaz bouteille vs électrique, logements récents). À droite : forte dépendance aux énergies fossiles.",
            'y': "<b>Axe Y — Score urbanité</b> : densité urbaine, forme de l'habitat, accessibilité. En haut : urbain dense bien équipé. En bas : rural/pavillonnaire.",
            'quadrants': {
                'tr': '<b>Pavillonnaire carboné</b> — Zones périurbaines : maison, 2+ voitures, chauffage fioul. Profil le plus difficile à décarboner.',
                'tl': '<b>Rural sobre</b> — Zones rurales éloignées mais avec peu de chauffage fossile (bois, solaire) et moins de déplacements motorisés.',
                'bl': '<b>Urbain sobre</b> — Centres-villes denses : transports collectifs, vélo, logements récents ou rénovés. Profil le plus favorable à la transition.',
                'br': '<b>Urbain carboné</b> — Zones denses mais avec chauffage fioul ou fort usage voiture : habitat ancien non rénové.',
            }
        }
    },

    {
        'id': 'pca_vraie',
        'label': 'PCA',
        'emoji': '🔢',
        'xVar': 'score_pca_1', 'xInvert': False,
        'yVar': 'score_pca_2',
        'xTitle': '← Locatif dense (appartements, immigrés) ─── PCA 1 : Logement & confort ─── Propriétaire pavillonnaire →',
        'yTitle': '← Diplômé aisé ─── PCA 2 : Composition sociale ─── Ouvrier précaire →',
        'xRange': [-55, 55], 'yRange': [-55, 55],
        'corners': [
            {'pos': 'tl', 'text': 'HLM<br>PRÉCAIRE', 'color': '#DC2626'},
            {'pos': 'tr', 'text': 'PAVILLONNAIRE<br>OUVRIER', 'color': '#374151'},
            {'pos': 'bl', 'text': 'URBAIN<br>DIPLÔMÉ', 'color': '#F59E0B'},
            {'pos': 'br', 'text': 'RÉSIDENTIEL<br>BOURGEOIS', 'color': '#1D4ED8'},
        ],
        'desc': {
            'title': 'Vraie ACP pondérée — PC1 × PC2',
            'x': "<b>Axe X — ACP PC1 (vraie ACP, ~20% de variance)</b> : 1ère composante principale d'une vraie ACP pondérée par population, calculée sur ~80 variables socio-économiques. Oppose l'habitat pavillonnaire propriétaire à l'habitat collectif dense. <b>+</b> surface, 2+ voitures, propriétaires, nb pièces, maison. <b>−</b> locataires, appartements, sans voiture, immigrés, petits logements.",
            'y': "<b>Axe Y — ACP PC2 (vraie ACP, ~12% de variance)</b> : 2ème composante principale, orthogonale à PC1. Sépare les zones ouvrières précaires des zones de cadres diplômés. <b>+</b> impôts faibles, sans diplôme, allocations familiales, prestations sociales, minimas sociaux, DISP_PPMINI21. <b>−</b> BAC+, revenu médian, cadres sup, BAC+5, revenus d'activité.",
            'quadrants': {
                'tr': '<b>Pavillonnaire ouvrier</b> — Lotissements périurbains, population peu qualifiée.',
                'tl': '<b>HLM précaire</b> — Grands ensembles denses avec forte précarité sociale.',
                'br': '<b>Résidentiel bourgeois</b> — Grandes propriétés, cadres supérieurs.',
                'bl': '<b>Urbain diplômé</b> — Centres-villes, jeunes actifs diplômés, locataires.',
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
            'x': "<b>Technique : t-SNE</b> (t-distributed Stochastic Neighbor Embedding) : algorithme de r\u00e9duction non-lin\u00e9aire qui projette les ~80 variables socio-\u00e9conomiques de chaque IRIS en 2 dimensions. Contrairement \u00e0 l'ACP (qui cherche les axes de plus grande <em>variance globale</em>), le t-SNE optimise la pr\u00e9servation des <em>voisinages locaux</em> : il place proches sur la carte les IRIS qui se ressemblent, m\u00eame si leurs profils ne varient pas beaucoup \u00e0 l'\u00e9chelle nationale. Param\u00e8tres utilis\u00e9s : perplexit\u00e9=30, 1000 it\u00e9rations, initialisation par ACP \u00e0 20 composantes.",
            'y': "<b>Comment lire cette carte</b> : les axes X et Y n'ont aucune signification propre (on ne peut pas dire \u00ab plus \u00e0 droite = plus riche \u00bb). Ce qui compte, ce sont les <em>regroupements visuels</em> : un amas compact = un type de territoire sociologiquement coh\u00e9rent. Les couleurs des partis dominants r\u00e9v\u00e8lent comment le vote s'organise dans cet espace. Limite : le t-SNE ne pr\u00e9serve pas bien les distances entre groupes \u00e9loign\u00e9s \u2014 voir UMAP pour \u00e7a.",
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
            'title': 'UMAP \u2014 Topologie des territoires fran\u00e7ais',
            'x': "<b>Technique : UMAP</b> (Uniform Manifold Approximation and Projection) : comme le t-SNE, l'UMAP projette les ~80 variables socio-\u00e9conomiques en 2D, mais avec deux avantages : il pr\u00e9serve \u00e0 la fois la structure <em>locale</em> (les IRIS similaires sont proches) ET <em>globale</em> (les distances entre groupes \u00e9loign\u00e9s restent interpr\u00e9tables). Deux amas s\u00e9par\u00e9s sur cette carte correspondent \u00e0 des types de territoires v\u00e9ritablement diff\u00e9rents. Param\u00e8tres : n_neighbors=15, min_dist=0.1, m\u00e9trique euclidienne apr\u00e8s r\u00e9duction PCA \u00e0 20 composantes.",
            'y': "<b>Comment lire cette carte</b> : les axes X et Y n'ont aucune signification propre. L'int\u00e9r\u00eat est dans la <em>topologie</em> : la forme des amas, leur s\u00e9paration, les ponts entre types de territoires. L'UMAP tend \u00e0 produire des groupes plus nets et plus s\u00e9par\u00e9s que le t-SNE, ce qui facilite l'identification des grands types de territoires fran\u00e7ais et de leur vote. Les couleurs r\u00e9v\u00e8lent comment l'espace sociologique se partitionne \u00e9lectoralement.",
            'quadrants': {}
        }
    },
]

# Variables disponibles dans les dropdowns custom
VARS_BY_CAT = {
    'Scores composites': [
        # Scores principaux (make_score_grouped — valeurs canoniques)
        'score_exploitation', 'score_domination', 'score_cap_eco', 'score_cap_cult',
        'score_precarite', 'score_rentier', 'score_composition_capital',
        'score_urbanite', 'score_confort_residentiel', 'score_equipement_public',
        'score_dependance_carbone',
        # Scores v2 (disponibles pour axes custom)
        'score_domination_v2', 'score_cap_eco_v2', 'score_cap_cult_v2',
        'score_rentier_v2', 'score_ruralite_v2',
        'score_periurbain', 'score_france_pavillonnaire',
        # Vraie ACP pondérée par population
        'score_pca_1', 'score_pca_2', 'score_pca_3', 'score_pca_4',
        'score_pca_5', 'score_pca_6', 'score_pca_7', 'score_pca_8',
    ],
    'Reductions dimensionnelles': [
        'tsne_x', 'tsne_y', 'umap_x', 'umap_y',
    ],
    'Démographie': [
        'pop_iris', 'pop_commune',
        'pct_etrangers', 'age_moyen',
        'pct_femmes', 'taille_menage_moy', 'pct_hors_menage', 'ecart_csp_plus_hf',
        'pct_0_19', 'pct_20_64', 'pct_65_plus', 'pct_immigres',
    ],
    'Revenus et inégalités': [
        'DISP_MED21', 'DISP_TP6021',
        'DISP_PPAT21', 'DISP_PPSOC21', 'DISP_PACT21',
        'DISP_PPEN21','DISP_PCHO21', 'DISP_PIMPOT21',
    ],
    'Diplômes et emploi': [
        'pct_sup5', 'pct_sans_diplome', 'pct_capbep',
        'pct_chomage', 'pct_cdi', 'pct_interim',
        'pct_inactif', 'pct_csp_retraite', 'pct_csp_sans_emploi', 'pct_etudiants',
    ],
    'Logement': [
        'pct_proprietaires', 'pct_locataires', 'pct_hlm',
        'pct_logvac', 'pct_maison', 'pct_appart',
        'pct_petits_logements', 'pct_grands_logements',
        'pct_logements_anciens', 'pct_logements_recents',
        'pct_voiture_0', 'pct_voiture_2plus',
        'surface_moyenne', 'pct_suroccupation',
        'pct_chauffage_elec', 'pct_chauffage_fioul',
        'pct_chauffage_gaz_ville', 'pct_chauffage_gaz_bouteille',
        'pct_chauffage_autre',
        'pct_garage', 'nb_pieces_moyen', 'pct_studios', 'pct_logements_5p_plus',
    ],
    'Équipements (BPE)': [
        'bpe_total_pour1000',
        'bpe_A_services_pour1000', 'bpe_B_commerces_pour1000',
        'bpe_C_enseignement_pour1000', 'bpe_D_sante_pour1000',
        'bpe_E_transports_pour1000', 'bpe_F_sports_culture_pour1000',
        'bpe_G_tourisme_pour1000',
        'bpe_educ_prioritaire_pour1000', 'bpe_ecole_privee_pour1000',
        'bpe_sport_indoor_pour1000', 'pct_sport_accessible',
    ],
    'Élections': [
        # Abstentions (chronologique)
        'pct_abst_legi17t1', 'pct_abst_legi17t2',
        'pct_abst_legi22t1', 'pct_abst_legi22t2',
        'pct_abst_legi24t1', 'pct_abst_legi24t2',
        'pct_abst_pres17t1', 'pct_abst_pres17t2',
        'pct_abst_pres22t1', 'pct_abst_pres22t2',
        'pct_abst_euro19t1', 'pct_abst_euro24t1',
        # RN
        'pct_RN_legi17t1', 'pct_RN_legi17t2',
        'pct_RN_legi22t1', 'pct_RN_legi22t2',
        'pct_RN_legi24t1', 'pct_RN_legi24t2',
        'pct_RN_euro19t1', 'pct_RN_euro24t1',
        # Gauche radicale (LFI / NUPES / NFP)
        'pct_LFI_legi17t1', 'pct_LFI_legi17t2',
        'pct_NUPES_legi22t1', 'pct_NUPES_legi22t2',
        'pct_NFP_legi24t1', 'pct_NFP_legi24t2',
        'pct_LFI_legi24t1',
        'pct_LFI_euro19t1', 'pct_LFI_euro24t1',
        # ENS / Macron
        'pct_ENS_legi17t1', 'pct_ENS_legi17t2',
        'pct_ENS_legi22t1',
        'pct_ENS_legi24t1', 'pct_ENS_legi24t2',
        'pct_ENS_euro19t1', 'pct_ENS_euro24t1',
        # LR
        'pct_LR_legi17t1', 'pct_LR_legi17t2',
        'pct_LR_legi22t1', 'pct_LR_legi22t2',
        'pct_LR_legi24t1', 'pct_LR_legi24t2',
        'pct_LR_euro19t1', 'pct_LR_euro24t1',
        # PS / Socialistes
        'pct_PS_legi17t1', 'pct_PS_legi17t2',
        'pct_PS_legi24t1',
        'pct_PS_PP_euro19t1', 'pct_PS_PP_euro24t1',
        # EELV
        'pct_EELV_legi17t1', 'pct_EELV_legi17t2',
        'pct_EELV_legi22t1',
        'pct_EELV_legi24t1',
        'pct_EELV_euro19t1', 'pct_EELV_euro24t1',
        # PCF
        'pct_PCF_legi17t1', 'pct_PCF_legi17t2',
        'pct_PCF_legi24t1',
        'pct_PCF_euro24t1',
        # Extrême-droite hors RN
        'pct_EXD_legi17t1', 'pct_EXD_legi17t2',
        'pct_EXD_legi24t1',
        'pct_UXD_legi24t1', 'pct_UXD_legi24t2',
        'pct_REC_legi22t1',
        'pct_REC_legi24t1',
        'pct_REC_euro24t1',
        # Présidentielles — Le Pen
        'pct_LE_PEN_pres17t1', 'pct_LE_PEN_pres17t2',
        'pct_LE_PEN_pres22t1', 'pct_LE_PEN_pres22t2',
        # Présidentielles — Macron
        'pct_MACRON_pres17t1', 'pct_MACRON_pres17t2',
        'pct_MACRON_pres22t1', 'pct_MACRON_pres22t2',
        # Présidentielles — Mélenchon
        'pct_MELENCHON_pres17t1', 'pct_MELENCHON_pres22t1',
        # Présidentielles — Fillon / Pécresse (droite)
        'pct_FILLON_pres17t1',
        'pct_PECRESSE_pres22t1',
        # Présidentielles — autres
        'pct_HAMON_pres17t1',
        'pct_ZEMMOUR_pres22t1',
        'pct_JADOT_pres22t1',
        'pct_ROUSSEL_pres22t1',
        'pct_HIDALGO_pres22t1',
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
    'score_dependance_carbone':             'Score dépendance carbone — mobilité fossile (voiture, 2+v) + chauffage fossile (fioul, gaz bouteille) vs transports/vélo/logements récents',
    'score_peripherie_metropole':           'Score périphérie-métropole — Périurbain voiture (Le Pen) vs métropole diplômée (Macron/Jadot)',
    # Scores v2/grouped
    'score_composition_capital':            'Score composition capital — revenus patrimoniaux/bénéfices/employeurs vs salaires/ouvriers/CDI',
    'score_domination_v2':                  'Score domination v2 (grouped) — capital éco (revenu, patrimoine) + capital culturel/position (diplômes, CSP)',
    'score_cap_eco_v2':                     'Score capital économique v2 (grouped) — richesse/revenus (65%) + capital immobilier (35%)',
    'score_cap_cult_v2':                    'Score capital culturel v2 (grouped) — diplômes/CSP (75%) + BPE culturels (25%)',
    'score_rentier_v2':                     'Score rentier v2 (grouped) — patrimoine/bénéfices/propriétaires vs salaires/CDD/intérim',
    'score_ruralite_v2':                    'Score ruralité v2 (débiaisé, sans immigrés) — agriculteurs, voiture, pendulaires vs transports, équipements, BPE',
    'score_periurbain':                     'Score périurbain — maison/voiture/HLM périphérique vs transports/appartements/équipements urbains',
    'score_france_pavillonnaire':           'Score France pavillonnaire — CAP-BEP/voiture/maison vs BAC+/vélo/cadres/métropole',
    # Vraie ACP pondérée par population
    'score_pca_1':  'PCA vraie PC1 — Logement & confort. <b>+</b> surface, 2+ voitures, propriétaires, nb pièces, maison. <b>−</b> locataires, appartements, sans voiture, immigrés, petits logements.',
    'score_pca_2':  'PCA vraie PC2 — Composition sociale. <b>+</b> impôts faibles, sans diplôme, allocations familiales, minimas sociaux. <b>−</b> BAC+, revenu médian, cadres sup, BAC+5, revenus d\'activité.',
    'score_pca_3':  'PCA vraie PC3 — Démographie active vs retraités. <b>+</b> 0-19 ans, revenus d\'activité, salaires, professions intermédiaires, employés. <b>−</b> 65+ ans, âge moyen, pensions, BPE total, retraités.',
    'score_pca_4':  'PCA vraie PC4 — Démographie & chauffage gaz. <b>+</b> gaz de ville, femmes, pensions, 65+, retraités. <b>−</b> logements anciens, chauffage autre/bouteille, services, 20-64 ans.',
    'score_pca_5':  'PCA vraie PC5 — Inégalités résidentielles. <b>+</b> inactifs, inégalités S80/S20, étudiants, temps partiel, sans-emploi. <b>−</b> BPE total/services/commerces/santé, employés, chauffage élec.',
    'score_pca_6':  'PCA vraie PC6 — Éducation privée & hors-ménage. <b>+</b> étudiants, hors-ménage, enseignement BPE, écoles privées, sport indoor. <b>−</b> transports BPE, immigrés, CDI, étrangers, indépendants.',
    'score_pca_7':  'PCA vraie PC7 — Logement récent & patrimoine. <b>+</b> patrimoine, 0-19 ans, logements récents, indépendants, Gini. <b>−</b> logements anciens/vacants, agriculteurs, 20-64 ans, chauffage fioul.',
    'score_pca_8':  'PCA vraie PC8 — CDD & chauffage électrique. <b>+</b> chauffage élec, logements récents, CDD, 2-roues, petits logements. <b>−</b> sport indoor, enseignement, écoles privées, gaz de ville, sport-culture BPE.',
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
    # ── Structure démographique 2020 (base-ic-evol-struct-pop-2020) ────────────
    'pop_totale':               'Population totale (2021)',
    'pop_iris':                 'Population totale de l\'IRIS (2021)',
    'pop_commune':              'Population totale de la commune (2021, somme des IRIS)',
    'age_moyen':                'Âge moyen de la population (2021)',
    'pct_0_19':                 '% population 0-19 ans (2021)',
    'pct_20_64':                '% population 20-64 ans (2021)',
    'pct_65_plus':              '% population 65 ans et plus (2021)',
    'pct_etrangers':            '% population de nationalité étrangère (2021)',
    'pct_immigres':             '% population immigrée (2021)',
    'pct_femmes':               '% population féminine (2021)',
    'taille_menage_moy':        'Taille moyenne des ménages (2021)',
    'pct_hors_menage':          '% vivant hors ménage (institutions, EHPAD, prisons…) (2021)',
    'ecart_csp_plus_hf':        'Écart H/F dans les cadres CSP+ (pts %, positif = plus d\'hommes) (2021)',
    # ── Effectifs bruts population 2020 ────────────────────────────────────────
    'P21_POP':                  'Population en 2021',
    'P21_POP0002':              'Nombre de personnes de 0 à 2 ans (2021)',
    'P21_POP0305':              'Nombre de personnes de 3 à 5 ans (2021)',
    'P21_POP0610':              'Nombre de personnes de 6 à 10 ans (2021)',
    'P21_POP1117':              'Nombre de personnes de 11 à 17 ans (2021)',
    'P21_POP1824':              'Nombre de personnes de 18 à 24 ans (2021)',
    'P21_POP2539':              'Nombre de personnes de 25 à 39 ans (2021)',
    'P21_POP4054':              'Nombre de personnes de 40 à 54 ans (2021)',
    'P21_POP5564':              'Nombre de personnes de 55 à 64 ans (2021)',
    'P21_POP6579':              'Nombre de personnes de 65 à 79 ans (2021)',
    'P21_POP80P':               'Nombre de personnes de 80 ans ou plus (2021)',
    'P21_POP0014':              'Nombre de personnes de 0 à 14 ans (2021)',
    'P21_POP1529':              'Nombre de personnes de 15 à 29 ans (2021)',
    'P21_POP3044':              'Nombre de personnes de 30 à 44 ans (2021)',
    'P21_POP4559':              'Nombre de personnes de 45 à 59 ans (2021)',
    'P21_POP6074':              'Nombre de personnes de 60 à 74 ans (2021)',
    'P21_POP75P':               'Nombre de personnes de 75 ans ou plus (2021)',
    'P21_POP0019':              'Nombre de personnes de 0 à 19 ans (2021)',
    'P21_POP2064':              'Nombre de personnes de 20 à 64 ans (2021)',
    'P21_POP65P':               'Nombre de personnes de 65 ans ou plus (2021)',
    'P21_POPH':                 'Nombre d\'hommes (2021)',
    'P21_H0014':                'Nombre d\'hommes de 0 à 14 ans (2021)',
    'P21_H1529':                'Nombre d\'hommes de 15 à 29 ans (2021)',
    'P21_H3044':                'Nombre d\'hommes de 30 à 44 ans (2021)',
    'P21_H4559':                'Nombre d\'hommes de 45 à 59 ans (2021)',
    'P21_H6074':                'Nombre d\'hommes de 60 à 74 ans (2021)',
    'P21_H75P':                 'Nombre d\'hommes de 75 ans ou plus (2021)',
    'P21_H0019':                'Nombre d\'hommes de 0 à 19 ans (2021)',
    'P21_H2064':                'Nombre d\'hommes de 20 à 64 ans (2021)',
    'P21_H65P':                 'Nombre d\'hommes de 65 ans ou plus (2021)',
    'P21_POPF':                 'Nombre de femmes (2021)',
    'P21_F0014':                'Nombre de femmes de 0 à 14 ans (2021)',
    'P21_F1529':                'Nombre de femmes de 15 à 29 ans (2021)',
    'P21_F3044':                'Nombre de femmes de 30 à 44 ans (2021)',
    'P21_F4559':                'Nombre de femmes de 45 à 59 ans (2021)',
    'P21_F6074':                'Nombre de femmes de 60 à 74 ans (2021)',
    'P21_F75P':                 'Nombre de femmes de 75 ans ou plus (2021)',
    'P21_F0019':                'Nombre de femmes de 0 à 19 ans (2021)',
    'P21_F2064':                'Nombre de femmes de 20 à 64 ans (2021)',
    'P21_F65P':                 'Nombre de femmes de 65 ans ou plus (2021)',
    'P21_POP_FR':               'Nombre de personnes de nationalité française (2021)',
    'P21_POP_ETR':              'Nombre de personnes étrangères (2021)',
    'P21_POP_IMM':              'Nombres de personnes immigrées (2021)',
    'P21_PMEN':                 'Population des ménages (2021)',
    'P21_PHORMEN':              'Population hors ménages (2021)',
    # ── CSP population 15 ans et plus 2021 (C21_) ──────────────────────────────
    'C21_POP15P':               'Nombre de personnes de 15 ans ou plus (2021)',
    'C21_POP15P_CS1':           'Nombre de personnes 15+ agriculteurs exploitants (2021)',
    'C21_POP15P_CS2':           'Nombre de personnes 15+ artisans, commerçants, chefs d\'entreprise (2021)',
    'C21_POP15P_CS3':           'Nombre de personnes 15+ cadres et professions intellectuelles supérieures (2021)',
    'C21_POP15P_CS4':           'Nombre de personnes 15+ professions intermédiaires (2021)',
    'C21_POP15P_CS5':           'Nombre de personnes 15+ employés (2021)',
    'C21_POP15P_CS6':           'Nombre de personnes 15+ ouvriers (2021)',
    'C21_POP15P_CS7':           'Nombre de personnes 15+ retraités (2021)',
    'C21_POP15P_CS8':           'Nombre de personnes 15+ autres sans activité professionnelle (2021)',
    'C21_H15P':                 'Nombre d\'hommes de 15 ans ou plus (2021)',
    'C21_H15P_CS1':             'Nombre d\'hommes 15+ agriculteurs exploitants (2021)',
    'C21_H15P_CS2':             'Nombre d\'hommes 15+ artisans, commerçants, chefs d\'entreprise (2021)',
    'C21_H15P_CS3':             'Nombre d\'hommes 15+ cadres et professions intellectuelles supérieures (2021)',
    'C21_H15P_CS4':             'Nombre d\'hommes 15+ professions intermédiaires (2021)',
    'C21_H15P_CS5':             'Nombre d\'hommes 15+ employés (2021)',
    'C21_H15P_CS6':             'Nombre d\'hommes 15+ ouvriers (2021)',
    'C21_H15P_CS7':             'Nombre d\'hommes 15+ retraités (2021)',
    'C21_H15P_CS8':             'Nombre d\'hommes 15+ autres sans activité professionnelle (2021)',
    'C21_F15P':                 'Nombre de femmes de 15 ans ou plus (2021)',
    'C21_F15P_CS1':             'Nombre de femmes 15+ agriculteurs exploitants (2021)',
    'C21_F15P_CS2':             'Nombre de femmes 15+ artisans, commerçants, chefs d\'entreprise (2021)',
    'C21_F15P_CS3':             'Nombre de femmes 15+ cadres et professions intellectuelles supérieures (2021)',
    'C21_F15P_CS4':             'Nombre de femmes 15+ professions intermédiaires (2021)',
    'C21_F15P_CS5':             'Nombre de femmes 15+ employés (2021)',
    'C21_F15P_CS6':             'Nombre de femmes 15+ ouvriers (2021)',
    'C21_F15P_CS7':             'Nombre de femmes 15+ retraités (2021)',
    'C21_F15P_CS8':             'Nombre de femmes 15+ autres sans activité professionnelle (2021)',
    # ── Logement (2022) ──
    'pct_proprietaires':        '% résidences principales occupées par propriétaires (2022)',
    'pct_locataires':           '% résidences principales en location (2022)',
    'pct_hlm':                  '% résidences principales en HLM (2022)',
    'pct_logvac':               '% logements vacants (2022)',
    'pct_maison':               '% maisons (vs appartements) (2022)',
    'pct_appart':               '% appartements (2022)',
    'pct_petits_logements':     '% logements < 40 m² (2022)',
    'pct_grands_logements':     '% logements ≥ 120 m² (2022)',
    'pct_logements_anciens':    '% logements construits avant 1919 (2022)',
    'pct_logements_recents':    '% logements construits 2006-2018 (2022)',
    'pct_voiture_0':            '% ménages sans voiture (2022)',
    'pct_voiture_2plus':        '% ménages avec 2 voitures ou plus (2022)',
    'surface_moyenne':          'Surface moyenne des résidences principales en m² (2022)',
    'pct_suroccupation':        '% résidences principales en suroccupation (2022)',
    'pct_chauffage_elec':       '% RP chauffées à l\'électricité (2022)',
    'pct_chauffage_fioul':      '% RP chauffées au fioul (2022)',
    'pct_chauffage_gaz_ville':  '% RP chauffées au gaz de ville / réseau de chaleur (2022)',
    'pct_chauffage_gaz_bouteille':'% RP chauffées au gaz bouteille/citerne (2022)',
    'pct_chauffage_autre':      '% RP chauffées à un autre combustible (bois, charbon…) (2022)',
    'pct_garage':               '% RP disposant d\'un stationnement réservé (2022)',
    'nb_pieces_moyen':          'Nombre moyen de pièces par résidence principale (2022)',
    'pct_studios':              '% RP d\'une seule pièce — studios (2022)',
    'pct_logements_5p_plus':    '% RP de 5 pièces ou plus (2022)',
    # ── Équipements BPE (2024) ──
    'bpe_total_pour1000':           'Équipements totaux pour 1000 habitants (BPE 2024)',
    'bpe_A_services_pour1000':      'Services (police, poste, banque…) pour 1000 hab. (BPE 2024)',
    'bpe_B_commerces_pour1000':     'Commerces (supermarchés, boulangeries…) pour 1000 hab. (BPE 2024)',
    'bpe_C_enseignement_pour1000':  'Enseignement (écoles, collèges…) pour 1000 hab. (BPE 2024)',
    'bpe_D_sante_pour1000':         'Santé (médecins, pharmacies…) pour 1000 hab. (BPE 2024)',
    'bpe_E_transports_pour1000':    'Transports (gares, taxis…) pour 1000 hab. (BPE 2024)',
    'bpe_F_sports_culture_pour1000':'Sports, loisirs, culture pour 1000 hab. (BPE 2024)',
    'bpe_G_tourisme_pour1000':      'Tourisme pour 1000 hab. (BPE 2024)',
    'bpe_educ_prioritaire_pour1000':'Établissements en éducation prioritaire pour 1000 hab. (BPE 2024)',
    'bpe_ecole_privee_pour1000':    'Écoles privées pour 1000 hab. (BPE 2024)',
    'bpe_sport_indoor_pour1000':    'Équipements sportifs couverts pour 1000 hab. (BPE 2024)',
    'pct_sport_accessible':         '% équipements sportifs accessibles PMR (BPE 2024)',
    # ── Abstentions ────────────────────────────────────────────────────────────
    'pct_abst_legi17t1':  '% abstention — Législatives 2017 T1',
    'pct_abst_legi17t2':  '% abstention — Législatives 2017 T2',
    'pct_abst_legi22t1':  '% abstention — Législatives 2022 T1',
    'pct_abst_legi22t2':  '% abstention — Législatives 2022 T2',
    'pct_abst_legi24t1':  '% abstention — Législatives 2024 T1',
    'pct_abst_legi24t2':  '% abstention — Législatives 2024 T2',
    'pct_abst_pres17t1':  '% abstention — Présidentielles 2017 T1',
    'pct_abst_pres17t2':  '% abstention — Présidentielles 2017 T2',
    'pct_abst_pres22t1':  '% abstention — Présidentielles 2022 T1',
    'pct_abst_pres22t2':  '% abstention — Présidentielles 2022 T2',
    'pct_abst_euro19t1':  '% abstention — Européennes 2019',
    'pct_abst_euro24t1':  '% abstention — Européennes 2024',
    # ── RN ─────────────────────────────────────────────────────────────────────
    'pct_RN_legi17t1':    '% RN — Législatives 2017 T1 (% des exprimés)',
    'pct_RN_legi17t2':    '% RN — Législatives 2017 T2 (% des exprimés)',
    'pct_RN_legi22t1':    '% RN — Législatives 2022 T1 (% des exprimés)',
    'pct_RN_legi22t2':    '% RN — Législatives 2022 T2 (% des exprimés)',
    'pct_RN_legi24t1':    '% RN — Législatives 2024 T1 (% des exprimés)',
    'pct_RN_legi24t2':    '% RN — Législatives 2024 T2 (% des exprimés)',
    'pct_RN_euro19t1':    '% RN — Européennes 2019 (% des exprimés)',
    'pct_RN_euro24t1':    '% RN — Européennes 2024 (% des exprimés)',
    # ── LFI / NUPES / NFP ──────────────────────────────────────────────────────
    'pct_LFI_legi17t1':   '% LFI — Législatives 2017 T1 (% des exprimés)',
    'pct_LFI_legi17t2':   '% LFI — Législatives 2017 T2 (% des exprimés)',
    'pct_NUPES_legi22t1': '% NUPES — Législatives 2022 T1 (% des exprimés)',
    'pct_NUPES_legi22t2': '% NUPES — Législatives 2022 T2 (% des exprimés)',
    'pct_NFP_legi24t1':   '% NFP — Législatives 2024 T1 (% des exprimés)',
    'pct_NFP_legi24t2':   '% NFP — Législatives 2024 T2 (% des exprimés)',
    'pct_LFI_legi24t1':   '% LFI — Législatives 2024 T1 (% des exprimés)',
    'pct_LFI_euro19t1':   '% LFI — Européennes 2019 (% des exprimés)',
    'pct_LFI_euro24t1':   '% LFI — Européennes 2024 (% des exprimés)',
    # ── ENS / Macron ───────────────────────────────────────────────────────────
    'pct_ENS_legi17t1':   '% ENS/LREM — Législatives 2017 T1 (% des exprimés)',
    'pct_ENS_legi17t2':   '% ENS/LREM — Législatives 2017 T2 (% des exprimés)',
    'pct_ENS_legi22t1':   '% Ensemble — Législatives 2022 T1 (% des exprimés)',
    'pct_ENS_legi24t1':   '% Ensemble — Législatives 2024 T1 (% des exprimés)',
    'pct_ENS_legi24t2':   '% Ensemble — Législatives 2024 T2 (% des exprimés)',
    'pct_ENS_euro19t1':   '% Renaissance — Européennes 2019 (% des exprimés)',
    'pct_ENS_euro24t1':   '% Renaissance — Européennes 2024 (% des exprimés)',
    # ── LR ─────────────────────────────────────────────────────────────────────
    'pct_LR_legi17t1':    '% LR — Législatives 2017 T1 (% des exprimés)',
    'pct_LR_legi17t2':    '% LR — Législatives 2017 T2 (% des exprimés)',
    'pct_LR_legi22t1':    '% LR — Législatives 2022 T1 (% des exprimés)',
    'pct_LR_legi22t2':    '% LR — Législatives 2022 T2 (% des exprimés)',
    'pct_LR_legi24t1':    '% LR — Législatives 2024 T1 (% des exprimés)',
    'pct_LR_legi24t2':    '% LR — Législatives 2024 T2 (% des exprimés)',
    'pct_LR_euro19t1':    '% LR — Européennes 2019 (% des exprimés)',
    'pct_LR_euro24t1':    '% LR — Européennes 2024 (% des exprimés)',
    # ── PS / Socialistes ───────────────────────────────────────────────────────
    'pct_PS_legi17t1':    '% PS — Législatives 2017 T1 (% des exprimés)',
    'pct_PS_legi17t2':    '% PS — Législatives 2017 T2 (% des exprimés)',
    'pct_PS_legi24t1':    '% PS — Législatives 2024 T1 (% des exprimés)',
    'pct_PS_PP_euro19t1': '% PS/Place Publique — Européennes 2019 (% des exprimés)',
    'pct_PS_PP_euro24t1': '% PS/Place Publique — Européennes 2024 (% des exprimés)',
    # ── EELV ───────────────────────────────────────────────────────────────────
    'pct_EELV_legi17t1':  '% EELV — Législatives 2017 T1 (% des exprimés)',
    'pct_EELV_legi17t2':  '% EELV — Législatives 2017 T2 (% des exprimés)',
    'pct_EELV_legi22t1':  '% EELV — Législatives 2022 T1 (% des exprimés)',
    'pct_EELV_legi24t1':  '% EELV — Législatives 2024 T1 (% des exprimés)',
    'pct_EELV_euro19t1':  '% EELV — Européennes 2019 (% des exprimés)',
    'pct_EELV_euro24t1':  '% EELV — Européennes 2024 (% des exprimés)',
    # ── PCF ────────────────────────────────────────────────────────────────────
    'pct_PCF_legi17t1':   '% PCF — Législatives 2017 T1 (% des exprimés)',
    'pct_PCF_legi17t2':   '% PCF — Législatives 2017 T2 (% des exprimés)',
    'pct_PCF_legi24t1':   '% PCF — Législatives 2024 T1 (% des exprimés)',
    'pct_PCF_euro24t1':   '% PCF — Européennes 2024 (% des exprimés)',
    # ── Extrême-droite hors RN ─────────────────────────────────────────────────
    'pct_EXD_legi17t1':   '% Extrême-droite — Législatives 2017 T1 (% des exprimés)',
    'pct_EXD_legi17t2':   '% Extrême-droite — Législatives 2017 T2 (% des exprimés)',
    'pct_EXD_legi24t1':   '% Extrême-droite — Législatives 2024 T1 (% des exprimés)',
    'pct_UXD_legi24t1':   '% Union Extrême-droite (Ciotti) — Législatives 2024 T1 (% des exprimés)',
    'pct_UXD_legi24t2':   '% Union Extrême-droite (Ciotti) — Législatives 2024 T2 (% des exprimés)',
    'pct_REC_legi22t1':   '% Reconquête — Législatives 2022 T1 (% des exprimés)',
    'pct_REC_legi24t1':   '% Reconquête — Législatives 2024 T1 (% des exprimés)',
    'pct_REC_euro24t1':   '% Reconquête — Européennes 2024 (% des exprimés)',
    # ── Présidentielles — Le Pen ───────────────────────────────────────────────
    'pct_LE_PEN_pres17t1':    '% Le Pen — Présidentielles 2017 T1 (% des exprimés)',
    'pct_LE_PEN_pres17t2':    '% Le Pen — Présidentielles 2017 T2 (% des exprimés)',
    'pct_LE_PEN_pres22t1':    '% Le Pen — Présidentielles 2022 T1 (% des exprimés)',
    'pct_LE_PEN_pres22t2':    '% Le Pen — Présidentielles 2022 T2 (% des exprimés)',
    # ── Présidentielles — Macron ───────────────────────────────────────────────
    'pct_MACRON_pres17t1':    '% Macron — Présidentielles 2017 T1 (% des exprimés)',
    'pct_MACRON_pres17t2':    '% Macron — Présidentielles 2017 T2 (% des exprimés)',
    'pct_MACRON_pres22t1':    '% Macron — Présidentielles 2022 T1 (% des exprimés)',
    'pct_MACRON_pres22t2':    '% Macron — Présidentielles 2022 T2 (% des exprimés)',
    # ── Présidentielles — Mélenchon ────────────────────────────────────────────
    'pct_MELENCHON_pres17t1': '% Mélenchon — Présidentielles 2017 T1 (% des exprimés)',
    'pct_MELENCHON_pres22t1': '% Mélenchon — Présidentielles 2022 T1 (% des exprimés)',
    # ── Présidentielles — Droite classique ────────────────────────────────────
    'pct_FILLON_pres17t1':    '% Fillon — Présidentielles 2017 T1 (% des exprimés)',
    'pct_PECRESSE_pres22t1':  '% Pécresse — Présidentielles 2022 T1 (% des exprimés)',
    # ── Présidentielles — autres ───────────────────────────────────────────────
    'pct_HAMON_pres17t1':     '% Hamon — Présidentielles 2017 T1 (% des exprimés)',
    'pct_ZEMMOUR_pres22t1':   '% Zemmour — Présidentielles 2022 T1 (% des exprimés)',
    'pct_JADOT_pres22t1':     '% Jadot — Présidentielles 2022 T1 (% des exprimés)',
    'pct_ROUSSEL_pres22t1':   '% Roussel — Présidentielles 2022 T1 (% des exprimés)',
    'pct_HIDALGO_pres22t1':   '% Hidalgo — Présidentielles 2022 T1 (% des exprimés)',
}

ALL_VARS = []
for cat_vars in VARS_BY_CAT.values():
    for v in cat_vars:
        if v not in ALL_VARS:
            ALL_VARS.append(v)

# ── FONCTIONS DE CHARGEMENT ET TRAITEMENT ────────────────────────────────────

def load_iris_base():
    """Charge iris_final_socio_politique.csv, fusionne coordonnées, retourne df avec _pop."""
    df = pd.read_csv("iris/iris_final_socio_politique.csv", low_memory=False)
    print(f"Données chargées : {len(df)} lignes × {len(df.columns)} colonnes")
    try:
        df_coord = pd.read_csv("iris/iris_coordonnees_finales.csv", low_memory=False)
        for col in ['LAB_IRIS', 'COM']:
            if col not in df.columns and col in df_coord.columns:
                df = df.merge(df_coord[['IRIS', col]], on='IRIS', how='left')
        print("Colonnes LAB_IRIS/COM fusionnées depuis iris_coordonnees_finales.csv")
    except Exception as e:
        print(f"iris_coordonnees_finales.csv non disponible : {e}")
    if 'LAB_IRIS' not in df.columns:
        df['LAB_IRIS'] = df.get('IRIS', df.index.astype(str))
    if 'COM' not in df.columns:
        df['COM'] = df.get('nom_commune', '')
    if 'pop_totale' in df.columns:
        df['_pop'] = df['pop_totale'].fillna(df['pop_totale'].median())
    elif 'TOTAL_POP_ESTIM' in df.columns:
        df['_pop'] = df['TOTAL_POP_ESTIM'].fillna(df['TOTAL_POP_ESTIM'].median())
    else:
        df['_pop'] = 2000.0
    print(f"Population : min={df['_pop'].min():.0f} max={df['_pop'].max():.0f} median={df['_pop'].median():.0f}")
    return df


def _build_electoral_axis_vars(df_iris):
    """Charge les CSV électoraux et retourne un dict col_name → array pour df.assign()."""
    new_cols = {}
    if 'COM' in df_iris.columns and 'pop_totale' in df_iris.columns:
        new_cols['pop_iris'] = df_iris['pop_totale'].fillna(0)
        new_cols['pop_commune'] = df_iris.groupby('COM')['pop_totale'].transform('sum').fillna(0)
    elif 'pop_totale' in df_iris.columns:
        new_cols['pop_iris'] = df_iris['pop_totale'].fillna(0)

    ELECTORAL_VARS = [
        ('2017_legi_t1','pct_abstention','pct_abst_legi17t1'),('2017_legi_t1','score_RN','pct_RN_legi17t1'),
        ('2017_legi_t1','score_LFI','pct_LFI_legi17t1'),('2017_legi_t1','score_PS','pct_PS_legi17t1'),
        ('2017_legi_t1','score_ENS','pct_ENS_legi17t1'),('2017_legi_t1','score_LR','pct_LR_legi17t1'),
        ('2017_legi_t1','score_EELV','pct_EELV_legi17t1'),('2017_legi_t1','score_PCF','pct_PCF_legi17t1'),
        ('2017_legi_t1','score_EXD','pct_EXD_legi17t1'),
        ('2017_legi_t2','pct_abstention','pct_abst_legi17t2'),('2017_legi_t2','score_RN','pct_RN_legi17t2'),
        ('2017_legi_t2','score_LFI','pct_LFI_legi17t2'),('2017_legi_t2','score_PS','pct_PS_legi17t2'),
        ('2017_legi_t2','score_ENS','pct_ENS_legi17t2'),('2017_legi_t2','score_LR','pct_LR_legi17t2'),
        ('2017_legi_t2','score_EELV','pct_EELV_legi17t2'),('2017_legi_t2','score_PCF','pct_PCF_legi17t2'),
        ('2017_legi_t2','score_EXD','pct_EXD_legi17t2'),
        ('2022_legi_t1','pct_abstention','pct_abst_legi22t1'),('2022_legi_t1','score_RN','pct_RN_legi22t1'),
        ('2022_legi_t1','score_NUPES','pct_NUPES_legi22t1'),('2022_legi_t1','score_ENS','pct_ENS_legi22t1'),
        ('2022_legi_t1','score_LR','pct_LR_legi22t1'),('2022_legi_t1','score_REC','pct_REC_legi22t1'),
        ('2022_legi_t1','score_EELV','pct_EELV_legi22t1'),
        ('2022_legi_t2','pct_abstention','pct_abst_legi22t2'),('2022_legi_t2','score_RN','pct_RN_legi22t2'),
        ('2022_legi_t2','score_NUPES','pct_NUPES_legi22t2'),('2022_legi_t2','score_LR','pct_LR_legi22t2'),
        ('2024_legi_t1','pct_abstention','pct_abst_legi24t1'),('2024_legi_t1','score_RN','pct_RN_legi24t1'),
        ('2024_legi_t1','score_NFP','pct_NFP_legi24t1'),('2024_legi_t1','score_ENS','pct_ENS_legi24t1'),
        ('2024_legi_t1','score_LR','pct_LR_legi24t1'),('2024_legi_t1','score_UXD','pct_UXD_legi24t1'),
        ('2024_legi_t1','score_REC','pct_REC_legi24t1'),('2024_legi_t1','score_EXD','pct_EXD_legi24t1'),
        ('2024_legi_t1','score_LFI','pct_LFI_legi24t1'),('2024_legi_t1','score_PS','pct_PS_legi24t1'),
        ('2024_legi_t1','score_EELV','pct_EELV_legi24t1'),('2024_legi_t1','score_PCF','pct_PCF_legi24t1'),
        ('2024_legi_t2','pct_abstention','pct_abst_legi24t2'),('2024_legi_t2','score_RN','pct_RN_legi24t2'),
        ('2024_legi_t2','score_NFP','pct_NFP_legi24t2'),('2024_legi_t2','score_UXD','pct_UXD_legi24t2'),
        ('2024_legi_t2','score_LR','pct_LR_legi24t2'),('2024_legi_t2','score_ENS','pct_ENS_legi24t2'),
        ('2017_pres_t1','pct_abstention','pct_abst_pres17t1'),('2017_pres_t1','score_LE_PEN','pct_LE_PEN_pres17t1'),
        ('2017_pres_t1','score_MELENCHON','pct_MELENCHON_pres17t1'),('2017_pres_t1','score_MACRON','pct_MACRON_pres17t1'),
        ('2017_pres_t1','score_FILLON','pct_FILLON_pres17t1'),('2017_pres_t1','score_HAMON','pct_HAMON_pres17t1'),
        ('2017_pres_t2','pct_abstention','pct_abst_pres17t2'),('2017_pres_t2','score_LE_PEN','pct_LE_PEN_pres17t2'),
        ('2017_pres_t2','score_MACRON','pct_MACRON_pres17t2'),
        ('2022_pres_t1','pct_abstention','pct_abst_pres22t1'),('2022_pres_t1','score_LE_PEN','pct_LE_PEN_pres22t1'),
        ('2022_pres_t1','score_MELENCHON','pct_MELENCHON_pres22t1'),('2022_pres_t1','score_MACRON','pct_MACRON_pres22t1'),
        ('2022_pres_t1','score_PECRESSE','pct_PECRESSE_pres22t1'),('2022_pres_t1','score_ZEMMOUR','pct_ZEMMOUR_pres22t1'),
        ('2022_pres_t1','score_JADOT','pct_JADOT_pres22t1'),('2022_pres_t1','score_ROUSSEL','pct_ROUSSEL_pres22t1'),
        ('2022_pres_t1','score_HIDALGO','pct_HIDALGO_pres22t1'),
        ('2022_pres_t2','pct_abstention','pct_abst_pres22t2'),('2022_pres_t2','score_LE_PEN','pct_LE_PEN_pres22t2'),
        ('2022_pres_t2','score_MACRON','pct_MACRON_pres22t2'),
        ('2019_euro_t1','pct_abstention','pct_abst_euro19t1'),('2019_euro_t1','score_RN','pct_RN_euro19t1'),
        ('2019_euro_t1','score_ENS','pct_ENS_euro19t1'),('2019_euro_t1','score_LFI','pct_LFI_euro19t1'),
        ('2019_euro_t1','score_EELV','pct_EELV_euro19t1'),('2019_euro_t1','score_LR','pct_LR_euro19t1'),
        ('2019_euro_t1','score_PS_PP','pct_PS_PP_euro19t1'),
        ('2024_euro_t1','pct_abstention','pct_abst_euro24t1'),('2024_euro_t1','score_RN','pct_RN_euro24t1'),
        ('2024_euro_t1','score_ENS','pct_ENS_euro24t1'),('2024_euro_t1','score_LFI','pct_LFI_euro24t1'),
        ('2024_euro_t1','score_EELV','pct_EELV_euro24t1'),('2024_euro_t1','score_LR','pct_LR_euro24t1'),
        ('2024_euro_t1','score_PS_PP','pct_PS_PP_euro24t1'),('2024_euro_t1','score_REC','pct_REC_euro24t1'),
        ('2024_euro_t1','score_PCF','pct_PCF_euro24t1'),
    ]
    loaded = {}
    for (csv_id, src_col, tgt_col) in ELECTORAL_VARS:
        if csv_id not in loaded:
            path = f"iris/elections/{csv_id}.csv"
            loaded[csv_id] = pd.read_csv(path, index_col='CODE_IRIS', dtype={'CODE_IRIS': str}) if _os.path.exists(path) else None
        elec = loaded[csv_id]
        if elec is None or src_col not in elec.columns:
            new_cols[tgt_col] = pd.Series(0.0, index=df_iris.index)
            continue
        merged = df_iris[['IRIS']].merge(elec[[src_col]], left_on='IRIS', right_index=True, how='left')
        new_cols[tgt_col] = merged[src_col].fillna(0).to_numpy()
    return new_cols


def _load_election_iris_data(election_id, df_iris):
    """Charge le CSV élection, merge sur CODE_IRIS, retourne liste de dicts par IRIS."""
    path = f"iris/elections/{election_id}.csv"
    if not _os.path.exists(path):
        print(f"  ⚠️  {path} introuvable, skip")
        return None
    elec = pd.read_csv(path, index_col='CODE_IRIS', dtype={'CODE_IRIS': str})
    score_cols = [c for c in elec.columns if c.startswith('score_') and c not in ('score_blanc', 'score_nul')]
    if not score_cols:
        return None
    party_names = [c.replace('score_', '') for c in score_cols]
    extra_cols = [c for c in ['pct_abstention', 'inscrits', 'votants', 'exprimes', 'blancs', 'nuls'] if c in elec.columns]
    merged = df_iris[['IRIS']].merge(elec[score_cols + extra_cols], left_on='IRIS', right_index=True, how='left')
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
        if row_sum > 101:
            row_scores = {party_names[j]: round(float(scores_arr[i, j]) / row_sum * 100, 1) for j in range(len(party_names))}
        else:
            row_scores = {party_names[j]: round(float(scores_arr[i, j]), 1) for j in range(len(party_names))}
        result.append({'parti': parti, 'color': color, 'scores': row_scores, 'abst': abst, 'inscrits': inscrits_val, 'exprimes': exprimes_val, 'blancs': blancs_val, 'nuls': nuls_val})
    return result


def load_election_data(df):
    """Charge toutes les élections disponibles. Retourne iris_election_data dict."""
    iris_election_data = {}
    for eid in ELECTIONS_AVAILABLE:
        data = _load_election_iris_data(eid, df)
        if data:
            iris_election_data[eid] = data
            print(f"  {eid}: OK ({len(data)} IRIS)")
    return iris_election_data


def compute_parti_dominant(df, iris_election_data):
    """Met à jour df['parti_dominant'] depuis l'élection par défaut."""
    _default = iris_election_data.get(DEFAULT_ELECTION)
    if _default:
        df['parti_dominant'] = [d['parti'] for d in _default]
        for parti, color in ALL_PARTIES_COLORS.items():
            if parti not in COULEURS:
                COULEURS[parti] = color
                LABELS[parti] = parti
                SHORT[parti] = parti[:4]
        print(f"Distribution partis dominants ({DEFAULT_ELECTION}):")
        print(df['parti_dominant'].value_counts().to_dict())
    else:
        score_party_cols = [c for c in df.columns if c.startswith('score_') and c[6:] in ALL_ORDER]
        if score_party_cols:
            for c in score_party_cols:
                df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0.0)
            party_map = {c: c[6:] for c in score_party_cols}
            df['parti_dominant'] = df[score_party_cols].idxmax(axis=1).map(party_map).fillna('AUTRE')
            all_zero = (df[score_party_cols] == 0).all(axis=1)
            df.loc[all_zero, 'parti_dominant'] = 'AUTRE'
        else:
            df['parti_dominant'] = 'AUTRE'
    return df


def compute_marker_sizes(df):
    """Retourne un array de tailles de marqueurs (2.5–6.5 px) pondéré par pop."""
    pop = df['_pop'].copy()
    q5, q95 = pop.quantile(0.05), pop.quantile(0.95)
    pop_clipped = pop.clip(q5, q95)
    return 2.5 + (pop_clipped - q5) / (q95 - q5) * (6.5 - 2.5)


def compute_jitter_vars(df):
    """Calcule var_data_x et var_data_y (avec jitter) pour toutes les variables."""
    np.random.seed(42)
    N = len(df)
    jit_unit_x = np.random.uniform(-1, 1, N)
    jit_unit_y = np.random.uniform(-1, 1, N)
    var_data_x, var_data_y = {}, {}
    _COMPOSITE = set(VARS_BY_CAT.get('Scores composites', [])) | {'tsne_x', 'tsne_y', 'umap_x', 'umap_y'}
    for v in ALL_VARS:
        if v not in df.columns:
            print(f"  WARNING: variable {v} not found, skipping")
            continue
        vals = pd.to_numeric(df[v], errors='coerce').copy().astype(float)
        mean_val = vals.mean()
        vals_filled = vals.fillna(0.0 if np.isnan(mean_val) else mean_val)
        std_val = vals_filled.std()
        if np.isnan(std_val) or std_val == 0:
            std_val = 1.0
        scale = 0.015
        var_data_x[v] = (vals_filled + jit_unit_x * scale * std_val).tolist()
        var_data_y[v] = (vals_filled + jit_unit_y * scale * std_val).tolist()
    return var_data_x, var_data_y


def compute_barycentres(df, iris_election_data, var_data_x, var_data_y):
    """Pré-calcule baryMeans, barySizes, abstBary, buttonPcts pour chaque élection."""
    pops = df['_pop'].fillna(1).values.astype(float)
    var_names = list(var_data_x.keys())
    var_mat = np.column_stack([np.array(var_data_x[v], dtype=float) for v in var_names])
    var_nan_mask = np.isnan(var_mat)
    var_mat_clean = np.where(var_nan_mask, 0.0, var_mat)
    var_valid = (~var_nan_mask).astype(float)
    iris_election_precomputed = {}
    for eid, data in iris_election_data.items():
        n = len(data)
        parties_set = set()
        for d in data:
            if d['scores'] and isinstance(d['scores'], dict):
                parties_set.update(d['scores'].keys())
        parties_list = sorted(parties_set)
        p_cnt = len(parties_list)
        party_idx = {g: j for j, g in enumerate(parties_list)}
        score_mat = np.zeros((n, p_cnt))
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
        weights = score_mat * pops[:, np.newaxis]
        weighted_sums = weights.T @ var_mat_clean
        weight_totals = weights.T @ var_valid
        weight_totals[weight_totals == 0] = np.nan
        bary_means_elec = {}
        for j, g in enumerate(parties_list):
            bary_means_elec[g] = {}
            for k, v in enumerate(var_names):
                wt = weight_totals[j, k]
                bary_means_elec[g][v] = round(weighted_sums[j, k] / wt, 3) if not np.isnan(wt) else 0
        party_totals = (pops[:, np.newaxis] * score_mat / 100).sum(axis=0)
        totals = {g: party_totals[j] for j, g in enumerate(parties_list) if party_totals[j] > 0}
        abst_valid = ~np.isnan(abst_arr)
        abst_total = np.sum(pops[abst_valid] * abst_arr[abst_valid] / 100) if abst_valid.any() else 0.0
        if abst_total > 0:
            totals['__ABST__'] = abst_total
        vals_list = [v for v in totals.values() if v > 0]
        bary_sizes_elec = {}
        if vals_list:
            min_v, max_v = min(vals_list), max(vals_list)
            for g, v in totals.items():
                bary_sizes_elec[g] = round(8 + (((v - min_v) / (max_v - min_v)) ** 0.5) * 34, 1) if max_v > min_v else 20
        abst_weights = np.where(abst_valid, pops * abst_arr / 100, 0.0)
        abst_w_sums = abst_weights @ var_mat_clean
        abst_w_totals = abst_weights @ var_valid
        abst_bary_elec = {}
        for k, v in enumerate(var_names):
            abst_bary_elec[v] = round(abst_w_sums[k] / abst_w_totals[k], 3) if abst_w_totals[k] > 0 else None
        total_exprimes = exp_arr.sum()
        button_pcts_elec = {}
        if total_exprimes > 0:
            party_voix = (exp_arr[:, np.newaxis] * score_mat / 100).sum(axis=0)
            for j, g in enumerate(parties_list):
                button_pcts_elec[g] = round(party_voix[j] / total_exprimes * 1000) / 10
        iris_election_precomputed[eid] = {
            'baryMeans': bary_means_elec,
            'barySizes': bary_sizes_elec,
            'abstBary': abst_bary_elec,
            'buttonPcts': button_pcts_elec,
        }
        print(f"  {eid}: {len(parties_set)} partis, {len(bary_sizes_elec)} tailles bary")
    return iris_election_precomputed


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


def make_customdata(df):
    """Retourne la liste des customdata (une liste par IRIS) pour le hover Plotly."""
    commune_col = 'nom_commune' if 'nom_commune' in df.columns else 'COM'
    pop_col = 'pop_totale' if 'pop_totale' in df.columns else '_pop'
    cd_cols = ['LAB_IRIS', commune_col, pop_col, 'DISP_MED21',
               'pct_csp_plus', 'pct_csp_ouvrier', 'pct_csp_intermediaire',
               'DISP_PPAT21', 'inscrits', 'votants', 'pct_abstention',
               'score_blanc', 'score_nul',
               'pct_proprietaires', 'pct_hlm',
               'pct_chomage', 'pct_bac_plus', 'pct_sans_diplome', 'age_moyen']
    existing = [c for c in cd_cols if c in df.columns]
    cd_df = df[existing].copy()
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
    iris_codes = df['IRIS'].tolist() if 'IRIS' in df.columns else [''] * len(rows)
    for i, row in enumerate(rows):
        row.append(_dep_label(iris_codes[i]))
    return rows


def group_means(df):
    """Barycentres pondérés par population par parti dominant."""
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


def build_group_data(df, var_data_x, var_data_y):
    """Retourne (group_data_x, group_data_y) : données par parti pour chaque variable."""
    gd_x, gd_y = {}, {}
    for g in ORDER:
        mask = df['parti_dominant'] == g
        if not mask.any():
            continue
        idxs = df.index[mask].tolist()
        gd_x[g], gd_y[g] = {}, {}
        for v in ALL_VARS:
            if v not in var_data_x:
                continue
            gd_x[g][v] = [var_data_x[v][i] for i in idxs]
            gd_y[g][v] = [var_data_y[v][i] for i in idxs]
    return gd_x, gd_y


def serialize_jitter_vars(var_data_x, var_data_y, suffix):
    """Écrit data/iris_x_{suffix}.json et data/iris_y_{suffix}.json."""
    import math
    _COMPOSITE = set(VARS_BY_CAT.get('Scores composites', [])) | {'tsne_x', 'tsne_y', 'umap_x', 'umap_y'}
    def _round2(arr): return [None if not math.isfinite(float(v)) else round(float(v), 2) for v in arr]
    def _round3(arr): return [None if not math.isfinite(float(v)) else round(float(v), 3) for v in arr]
    iris_x_js, iris_y_js = {}, {}
    for v in var_data_x:
        fn = _round3 if v in _COMPOSITE else _round2
        iris_x_js[v] = fn(var_data_x[v])
        iris_y_js[v] = fn(var_data_y[v])
    _os.makedirs('data', exist_ok=True)
    path_x = f'data/iris_x_{suffix}.json'
    path_y = f'data/iris_y_{suffix}.json'
    with open(path_x, 'w', encoding='utf-8') as f:
        _json.dump(iris_x_js, f, separators=(',', ':'))
    print(f"  {path_x} : {_os.path.getsize(path_x)//1024} KB")
    with open(path_y, 'w', encoding='utf-8') as f:
        _json.dump(iris_y_js, f, separators=(',', ':'))
    print(f"  {path_y} : {_os.path.getsize(path_y)//1024} KB")


def serialize_election_data(iris_election_data, iris_election_precomputed):
    """Écrit data/elec_{eid}.json pour chaque élection."""
    import math
    def _is_nan(v):
        return v is None or (isinstance(v, float) and math.isnan(v))
    _os.makedirs('data', exist_ok=True)
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
    for eid_s, elec_obj in iris_elec.items():
        path = f'data/elec_{eid_s}.json'
        with open(path, 'w', encoding='utf-8') as f:
            _json.dump(elec_obj, f, ensure_ascii=False, separators=(',', ':'))
        print(f"  data/elec_{eid_s}.json : {_os.path.getsize(path)//1024} KB")


def serialize_static(df, iris_election_data, marker_size):
    """Écrit data/static.json (IRIS_INFO, IRIS_POPS, MARKER_SIZES, GROUP_INDICES)."""
    _os.makedirs('data', exist_ok=True)
    all_customdata = make_customdata(df)
    iris_pops = [int(round(float(v))) for v in df['_pop'].fillna(1).tolist()]
    marker_sizes_list = [round(float(marker_size.loc[i]), 1) for i in df.index]
    group_indices = {g: df.index[df['parti_dominant'] == g].tolist() for g in ALL_ORDER}
    path = 'data/static.json'
    with open(path, 'w', encoding='utf-8') as f:
        _json.dump({'IRIS_INFO': all_customdata, 'IRIS_POPS': iris_pops,
                    'MARKER_SIZES': marker_sizes_list, 'GROUP_INDICES': group_indices},
                   f, ensure_ascii=False, separators=(',', ':'))
    print(f"  data/static.json : {_os.path.getsize(path)//1024} KB")
    return all_customdata, iris_pops, marker_sizes_list, group_indices


def build_inline_meta(df):
    """Retourne un dict avec toutes les métadonnées JSON à injecter dans le HTML."""
    buttons_data = [
        {'key': g, 'short': SHORT.get(g, g), 'label': LABELS.get(g, g),
         'color': ALL_PARTIES_COLORS.get(g, '#9CA3AF'),
         'count': int((df['parti_dominant'] == g).sum())}
        for g in ALL_ORDER
    ]
    return {
        'elections_meta': _json.dumps(ELECTIONS_AVAILABLE, ensure_ascii=False, separators=(',', ':')),
        'default_elec': _json.dumps(DEFAULT_ELECTION, separators=(',', ':')),
        'presets': _json.dumps(AXIS_PRESETS, ensure_ascii=False),
        'vars': _json.dumps(VARS_BY_CAT, ensure_ascii=False),
        'var_labels': _json.dumps(VAR_LABELS, ensure_ascii=False),
        'couleurs': _json.dumps(COULEURS, ensure_ascii=False, separators=(',', ':')),
        'vote_parties': _json.dumps(VOTE_PARTIES_JS, separators=(',', ':')),
        'all_parties_colors': _json.dumps(ALL_PARTIES_COLORS, ensure_ascii=False, separators=(',', ':')),
        'buttons': _json.dumps(buttons_data, ensure_ascii=False, separators=(',', ':')),
        'order': _json.dumps(ALL_ORDER, ensure_ascii=False, separators=(',', ':')),
    }