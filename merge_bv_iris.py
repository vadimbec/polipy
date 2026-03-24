"""
merge_bv_iris.py
================
Agrège les résultats électoraux par bureau de vote au niveau IRIS.

Source : fichiers parquet elections/candidats
  - resultats_elections/candidats_results.parquet  (voix par candidat par BV)
  - resultats_elections/general_results.parquet    (inscrits/votants par BV)
  - table_passage_BV_IRIS.csv                      (table de passage BV→IRIS)

Sortie : iris/elections/<id_election>.csv  (un fichier par élection+tour)

Usage :
  python merge_bv_iris.py
  python merge_bv_iris.py --elections 2022_legi_t1 2024_legi_t1
  python merge_bv_iris.py --list        # liste les élections disponibles
"""

import argparse
import os
import sys
import pandas as pd
import numpy as np

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

# ── CONFIGURATION ─────────────────────────────────────────────────────────────

# DEFAULT_CAND   = "resultats_elections/candidats_results.parquet"
# DEFAULT_GEN    = "resultats_elections/general_results.parquet"

DEFAULT_CAND   = "resultats_elections/candidats_results.parquet"
DEFAULT_GEN    = "resultats_elections/general_results.parquet"
DEFAULT_TABLE  = "table_passage_BV_IRIS.csv"
DEFAULT_OUTDIR = "iris/elections"

# Élections traitées par défaut (2019_euro_t1 skippée : voix=0 dans le parquet)
DEFAULT_ELECTIONS = [
    # Législatives
    "2012_legi_t1", "2012_legi_t2",
    "2017_legi_t1", "2017_legi_t2",
    "2022_legi_t1", "2022_legi_t2",
    "2024_legi_t1", "2024_legi_t2",
    # Européennes
    "2014_euro_t1", "2019_euro_t1", 
    "2024_euro_t1", 
    # Présidentielles
    "2012_pres_t1", "2012_pres_t2",
    "2017_pres_t1", "2017_pres_t2",
    "2022_pres_t1", "2022_pres_t2",
    # Municipales
    "2014_muni_t1", "2014_muni_t2",
    "2020_muni_t1", "2020_muni_t2",
    "2026_muni_t1", "2026_muni_t2"
]

# ── MAPPING NUANCES → PARTIS ───────────────────────────────────────────────────
# Mapping de base (commun à toutes les élections non-présidentielles).
# Certaines nuances sont remappées par élection via ELECTION_OVERRIDES.

NUANCE_TO_PARTI_BASE = {
    # ── RN ───────────────────────────────────────────────────────────────────
    'RN':   'RN',  'FN':   'RN',  'LRN':  'RN',  'LFN':  'RN',
    # ── UXD : Ciotti / alliance LR-RN 2024 ──────────────────────────────────
    'UXD':  'UXD', 'LUXD': 'UXD',
    # ── Reconquête ────────────────────────────────────────────────────────────
    'REC':  'REC', 'LREC': 'REC',
    # ── Extrême droite (hors RN) ─────────────────────────────────────────────
    'EXD':  'EXD', 'LEXD': 'EXD',
    # ── DLF (Debout la France / Dupont-Aignan) ──────────────────────────────
    'DLF':  'DLF', 'LDLF': 'DLF',
    # ── LR / Droite classique (UMP = ancienne appellation LR avant 2015) ────
    'LR':   'LR',  'LLR':  'LR',
    'UMP':  'LR',  'LUMP': 'LR',
    'DVD':  'DVD', 'LDVD': 'DVD', 'LUDR': 'DVD', 'LDSV': 'DVD',
    # ── PRV (Parti Radical Valoisien, apparenté UMP) ─────────────────────────
    'PRV':  'DVD',
    # ── ALLI (alliances locales droite/centre-droite, type Nouveau Centre-UMP) ─
    'ALLI': 'DVD',

    # ── DVC (divers centre) ──────────────────────────────────────────────────
    'DVC':  'DVC', 'LDVC': 'DVC',
    # ── CEN / NCE (Nouveau Centre / centre UDI-iste 2012) ────────────────────
    'CEN':  'DVC', 'NCE':  'DVC',
    # ── Ensemble / Renaissance ───────────────────────────────────────────────
    'ENS':  'ENS', 'REM':  'ENS', 'LREM': 'ENS', 'LENS': 'ENS', 'LREN': 'ENS',
    # ── Horizons ─────────────────────────────────────────────────────────────
    'HOR':  'HOR', 'LHOR': 'HOR',
    # ── MoDem ────────────────────────────────────────────────────────────────
    'MDM':  'MODEM', 'LMDM': 'MODEM',
    # ── UDI ──────────────────────────────────────────────────────────────────
    'UDI':  'UDI', 'LUDI': 'UDI',
    # ── Union de la gauche (label variable selon l'élection, cf OVERRIDES) ──
    'NUP':  'NUPES', 'UG': 'UG', 'LUG': 'UG',
    # ── LFI ──────────────────────────────────────────────────────────────────
    'FI':   'LFI', 'LFI':  'LFI',
    # ── Extrême gauche (LO, NPA, etc.) ───────────────────────────────────────
    'EXG':  'EXG', 'LEXG': 'EXG', 'LO':   'EXG', 'LLO':  'EXG', 'NPA':  'EXG', 'LNPA': 'EXG',
    # ── PS / Socialistes ─────────────────────────────────────────────────────
    'SOC':  'PS',  'LSOC': 'PS',
    # ── DVG (divers gauche) ──────────────────────────────────────────────────
    'DVG':  'DVG', 'LDVG': 'DVG',
    # ── PRG / Radicaux de gauche ─────────────────────────────────────────────
    'RDG':  'DVG', 'LRDG': 'DVG',
    # ── EELV / Écologistes ───────────────────────────────────────────────────
    'VEC':  'EELV', 'ECO':  'EELV', 'LVEC': 'EELV', 'LECO': 'EELV',
    # ── PCF / Front de Gauche ────────────────────────────────────────────────
    'COM':  'PCF', 'LCOM': 'PCF', 'FG':   'PCF', 'LFG':  'PCF',
    # ── LPG (Parti de Gauche, Mélenchon, fusionné dans FI) ───────────────────
    'LPG':  'PCF',
    # ── Régionalistes ────────────────────────────────────────────────────────
    'REG':  'REG', 'LREG': 'REG',
    # ── Union Divers Droite (maires sortants centre-droit, listes d'union locale) ─
    'LUD':  'DVD',
    # ── Union Citoyens / Union Centre (listes d'union centriste locale) ──────────
    'LUC':  'DVC',
    # ── Divers / DSV / DXD / DXG ─────────────────────────────────────────────
    'DIV':  'AUTRE', 'LDIV': 'AUTRE', 'DSV':  'AUTRE',
    'DXD':  'AUTRE', 'DXG':  'AUTRE',
    'LNC':  'AUTRE', 'LGJ':  'AUTRE',
    'NC':   'AUTRE',
    # ── AUT (divers ancienne codification) ───────────────────────────────────
    'AUT':  'AUTRE',
}

# Surcharges par élection : {id_election: {nuance: parti}}
# Permet de corriger les labels temporels (NUPES en 2022, NFP en 2024, etc.)
ELECTION_OVERRIDES = {
    # 2024 européennes : LUG = PS-Place Publique (Glucksmann), PAS NFP
    '2024_euro_t1': {
        'LUG': 'PS_PP',
    },
    # 2024 législatives : UG = NFP
    '2024_legi_t1': {'UG': 'NFP'},
    '2024_legi_t2': {'UG': 'NFP'},
}

# Mapping libellé de liste → parti pour les élections sans nuance codifiée
# Clé : libelle_abrege_liste (majuscules, strip)
LIBELLE_TO_PARTI = {
    # 2019 européennes
    'PRENEZ LE POUVOIR':        'RN',          # liste RN (Jordan Bardella)
    'RENAISSANCE':              'ENS',          # liste LREM (Nathalie Loiseau)
    "ENVIE D'EUROPE":           'PS_PP',        # liste PS-Place Publique (Glucksmann)
    'EUROPE ÉCOLOGIE':          'EELV',         # liste EELV (Yannick Jadot)
    'LA FRANCE INSOUMISE':      'LFI',          # liste LFI (Manon Aubry)
    'UNION DROITE-CENTRE':      'LR',           # liste LR (François-Xavier Bellamy)
    'DEBOUT LA FRANCE':         'DLF',          # liste DLF (Nicolas Dupont-Aignan)
    "POUR L'EUROPE DES GENS":   'PCF',          # liste PCF-PG (Ian Brossat)
    'LUTTE OUVRIÈRE':           'EXG',          # liste LO (Nathalie Arthaud)
    'ENSEMBLE POUR LE FREXIT':  'EXD',          # liste souverainiste droite
    'ENSEMBLE PATRIOTES':       'EXD',          # liste Philippot
    'LISTE DE LA RECONQUÊTE':   'EXD',          # liste Dupont-Moretti (pas Reconquête parti)
    'LISTE CITOYENNE':          'DVG',          # mouvement gilets jaunes / citoyens
    'ALLIANCE JAUNE':           'DVG',          # gilets jaunes
    'LES EUROPÉENS':            'DVC',          # liste pro-européen centre
    'URGENCE ÉCOLOGIE':         'EELV',         # liste écologiste dissidente
    'PARTI ANIMALISTE':         'AUTRE',
    'PARTI PIRATE':             'AUTRE',
    'UNE FRANCE ROYALE':        'AUTRE',
    'LA LIGNE CLAIRE':          'AUTRE',
    'DÉMOCRATIE REPRÉSENTATIVE':'AUTRE',
    'PACE':                     'AUTRE',
    "LES OUBLIES DE L'EUROPE":  'AUTRE',
    'PARTI FED. EUROPÉEN':      'AUTRE',
    'INITIATIVE CITOYENNE':     'AUTRE',
    'ALLONS ENFANTS':           'AUTRE',
    'DÉCROISSANCE 2019':        'AUTRE',
    'À VOIX ÉGALES':            'AUTRE',
    'NEUTRE ET ACTIF':          'AUTRE',
    'RÉVOLUTIONNAIRE':          'AUTRE',
    'ESPERANTO':                'AUTRE',
    'ÉVOLUTION CITOYENNE':      'AUTRE',
    'UDLEF':                    'AUTRE',
    'EUROPE AU SERVICE PEUPLES':'AUTRE',
}

# Élections utilisant libelle_abrege_liste plutôt que la colonne nuance
LIBELLE_ELECTIONS = {'2019_euro_t1'}

# Ordre canonique des partis pour le tri des colonnes de sortie
PARTIS_ORDER = [
    'RN', 'UXD', 'REC', 'EXD', 'DLF',
    'LR', 'DVD', 'DVC',
    'ENS', 'HOR', 'MODEM', 'UDI',
    'NFP', 'NUPES', 'PS_PP', 'UG',
    'LFI', 'PS', 'DVG', 'EELV', 'PCF', 'EXG',
    'REG',
    'AUTRE',
]

# ── MAPPING PRÉSIDENTIELLES (par nom de candidat) ─────────────────────────────
# Les présidentielles n'ont pas de nuance dans le parquet → on mappe par nom.
# Chaque candidat devient une "colonne parti" dans le CSV de sortie.

PRES_NOM_TO_PARTI = {
    # 2012
    'HOLLANDE':       'HOLLANDE',
    'SARKOZY':        'SARKOZY',
    'BAYROU':         'BAYROU',
    'JOLY':           'JOLY',
    'MÉLENCHON':      'MELENCHON',
    'MELENCHON':      'MELENCHON',
    'LE PEN':         'LE_PEN',
    'DUPONT-AIGNAN':  'DUPONT_AIGNAN',
    'POUTOU':         'AUTRE',
    'ARTHAUD':        'AUTRE',
    'CHEMINADE':      'AUTRE',
    # 2017
    'MACRON':         'MACRON',
    'FILLON':         'FILLON',
    'HAMON':          'HAMON',
    'LASSALLE':       'AUTRE',
    'ASSELINEAU':     'AUTRE',
    # 2022
    'ZEMMOUR':        'ZEMMOUR',
    'PÉCRESSE':       'PECRESSE',
    'PECRESSE':       'PECRESSE',
    'JADOT':          'JADOT',
    'ROUSSEL':        'ROUSSEL',
    'HIDALGO':        'HIDALGO',
}


# ── FONCTIONS ─────────────────────────────────────────────────────────────────

def load_data(cand_path: str, gen_path: str, table_path: str):
    print(f"Chargement candidats : {cand_path}")
    cand = pd.read_parquet(cand_path, columns=[
        'id_election', 'id_brut_miom', 'voix', 'nuance', 'nom', 'prenom',
        'libelle_abrege_liste'
    ])
    print(f"  {len(cand):,} lignes")

    print(f"Chargement général   : {gen_path}")
    gen = pd.read_parquet(gen_path, columns=[
        'id_election', 'id_brut_miom',
        'inscrits', 'votants', 'abstentions', 'blancs', 'nuls', 'exprimes'
    ])
    print(f"  {len(gen):,} lignes")

    print(f"Chargement passage   : {table_path}")
    passage = pd.read_csv(table_path)
    passage['CODE_IRIS'] = passage['CODE_IRIS'].astype(str).str.zfill(9)
    print(f"  {len(passage):,} correspondances BV→IRIS")

    return cand, gen, passage


def _assign_parti(cand: pd.DataFrame, id_election: str) -> pd.DataFrame:
    """Assigne la colonne 'parti' selon le type d'élection."""
    if 'pres' in id_election:
        # Présidentielles : mapper par nom (majuscule, strip)
        nom_upper = cand['nom'].str.upper().str.strip()
        cand = cand.copy()
        cand['parti'] = nom_upper.map(PRES_NOM_TO_PARTI).fillna('AUTRE')
        unmapped = nom_upper[~nom_upper.isin(PRES_NOM_TO_PARTI)].dropna().unique()
        if len(unmapped) > 0:
            print(f"  ⚠️  Noms candidats non mappés → AUTRE : {sorted(unmapped.tolist())}")
    elif id_election in LIBELLE_ELECTIONS:
        # Élections sans nuance codifiée : mapper par libelle_abrege_liste
        libelle_upper = cand['libelle_abrege_liste'].str.upper().str.strip()
        cand = cand.copy()
        cand['parti'] = libelle_upper.map(LIBELLE_TO_PARTI).fillna('AUTRE')
        unmapped = libelle_upper[~libelle_upper.isin(LIBELLE_TO_PARTI)].dropna().unique()
        if len(unmapped) > 0:
            print(f"  ⚠️  Libellés non mappés → AUTRE : {sorted(unmapped.tolist())}")
    else:
        mapping = {**NUANCE_TO_PARTI_BASE, **ELECTION_OVERRIDES.get(id_election, {})}
        cand = cand.copy()
        cand['parti'] = cand['nuance'].map(mapping).fillna('AUTRE')
        unmapped = cand.loc[
            cand['nuance'].notna() & ~cand['nuance'].isin(mapping), 'nuance'
        ].unique()
        if len(unmapped) > 0:
            print(f"  ⚠️  Nuances inconnues → AUTRE : {sorted(unmapped.tolist())}")
    return cand


def process_election(
    id_election: str,
    cand_all: pd.DataFrame,
    gen_all: pd.DataFrame,
    passage: pd.DataFrame,
) -> pd.DataFrame:
    """Agrège une élection au niveau IRIS."""

    # ── 1. Filtrer cette élection ──────────────────────────────────────────
    cand = cand_all[cand_all['id_election'] == id_election]
    gen  = gen_all[gen_all['id_election']  == id_election]

    if len(cand) == 0:
        print(f"  ⚠️  Aucune donnée candidats pour {id_election}")
        return None

    if 'exprimes' in gen.columns and gen['exprimes'].sum() == 0:
        if cand['voix'].sum() == 0:
            print(f"  ⚠️  exprimes = 0 et voix = 0 pour {id_election}, élection skippée")
            return None

    # ── 2. Mapper nuances/noms → partis ────────────────────────────────────
    cand = _assign_parti(cand, id_election)

    # ── 3. Voix par BV et parti ────────────────────────────────────────────
    bv_voix = (
        cand.groupby(['id_brut_miom', 'parti'])['voix']
        .sum()
        .unstack(fill_value=0)
    )

    # ── 4. Base inscrits/votants/exprimes ──────────────────────────────────
    base_cols = [c for c in ['inscrits', 'votants', 'abstentions', 'blancs', 'nuls', 'exprimes']
                 if c in gen.columns]
    bv_base = gen.set_index('id_brut_miom')[base_cols]

    # ── 5. Jointure BV ─────────────────────────────────────────────────────
    bv_all = bv_base.join(bv_voix, how='left').reset_index()
    bv_all = bv_all.rename(columns={'id_brut_miom': 'ID_BUREAU_VOTE'})

    # ── 6. Normaliser les anciens codes BV Paris (arr 2/3/4 renumérotés en 2024) ─
    # Paris a renuméroté arr2/3/4 entre 2022 et 2024 ; la table de passage utilise
    # les nouveaux codes. On traduit les anciens → nouveaux pour les élections antérieures.
    PARIS_BV_REMAP = {
        # Arr 2 : +10
        **{f'75056_{arr:02d}{old:02d}': f'75056_{arr:02d}{old+10:02d}'
           for arr in [2] for old in range(1, 11)},
        # Arr 3 : +20
        **{f'75056_{arr:02d}{old:02d}': f'75056_{arr:02d}{old+20:02d}'
           for arr in [3] for old in range(1, 16)},
        # Arr 4 : +35
        **{f'75056_{arr:02d}{old:02d}': f'75056_{arr:02d}{old+35:02d}'
           for arr in [4] for old in range(1, 15)},
    }
    bv_all['ID_BUREAU_VOTE'] = bv_all['ID_BUREAU_VOTE'].replace(PARIS_BV_REMAP)

    # ── 7. Merge avec table de passage ─────────────────────────────────────
    merged = bv_all.merge(passage, on='ID_BUREAU_VOTE', how='inner')
    n_matched = merged['ID_BUREAU_VOTE'].nunique()
    n_total   = len(bv_all)
    print(f"  BV matchés : {n_matched:,}/{n_total:,} ({n_matched/n_total*100:.1f}%)")

    # ── 7. Colonnes partis présentes dans cette élection ───────────────────
    partis_present = [p for p in PARTIS_ORDER if p in merged.columns]
    autres_partis  = [c for c in merged.columns if c not in
                      ['ID_BUREAU_VOTE', 'CODE_IRIS'] + base_cols + partis_present]

    agg_cols = base_cols + partis_present + autres_partis
    agg_cols = [c for c in agg_cols if c in merged.columns]

    # ── 8. Agréger au niveau IRIS ──────────────────────────────────────────
    iris_agg = merged.groupby('CODE_IRIS')[agg_cols].sum()
    print(f"  IRIS agrégés : {len(iris_agg):,}")

    # ── 9. Calcul scores (% des exprimés) ──────────────────────────────────
    # Dénominateur : exprimes si disponible, sinon votants (approximation)
    if 'exprimes' in iris_agg.columns:
        denom_score = iris_agg['exprimes'].replace(0, np.nan)
    else:
        denom_score = iris_agg['votants'].replace(0, np.nan)

    inscrits = iris_agg['inscrits'].replace(0, np.nan) if 'inscrits' in iris_agg.columns else None

    for p in partis_present + autres_partis:
        if p in iris_agg.columns:
            iris_agg[f'score_{p}'] = (iris_agg[p] / denom_score * 100).round(2)

    if 'abstentions' in iris_agg.columns and inscrits is not None:
        iris_agg['pct_abstention'] = (iris_agg['abstentions'] / inscrits * 100).round(2)
    if 'blancs' in iris_agg.columns:
        iris_agg['score_blanc'] = (iris_agg['blancs'] / denom_score * 100).round(2)
    if 'nuls' in iris_agg.columns:
        iris_agg['score_nul'] = (iris_agg['nuls'] / denom_score * 100).round(2)

    return iris_agg


def validate(id_election: str, iris_agg: pd.DataFrame) -> None:
    """Cross-checks rapides."""
    denom_col = 'exprimes' if 'exprimes' in iris_agg.columns else 'votants'

    checks = {
        '2024_legi_t1': ('RN',      29.3),
        '2022_legi_t1': ('RN',      18.7),
        '2017_legi_t1': ('RN',      13.2),
        '2024_euro_t1': ('RN',      31.4),
        '2022_pres_t1': ('LE_PEN',  23.2),
        '2022_pres_t1_macron': ('MACRON', 27.8),
    }

    for col in ['RN', 'NFP', 'NUPES', 'PS_PP', 'ENS', 'LR', 'LFI', 'LE_PEN', 'MACRON', 'MELENCHON']:
        if col in iris_agg.columns and denom_col in iris_agg.columns:
            total = iris_agg[col].sum()
            denom = iris_agg[denom_col].sum()
            pct = total / denom * 100
            print(f"  {col} national : {pct:.1f}%")

    if 'pct_abstention' in iris_agg.columns:
        print(f"  Abstention moy IRIS : {iris_agg['pct_abstention'].dropna().mean():.1f}%")


# ── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Agrège résultats BV → IRIS depuis parquet")
    parser.add_argument('--cand',      default=DEFAULT_CAND,   help="Fichier candidats parquet")
    parser.add_argument('--gen',       default=DEFAULT_GEN,    help="Fichier général parquet")
    parser.add_argument('--table',     default=DEFAULT_TABLE,  help="Table de passage BV→IRIS")
    parser.add_argument('--outdir',    default=DEFAULT_OUTDIR, help="Dossier de sortie")
    parser.add_argument('--elections', nargs='+', default=DEFAULT_ELECTIONS,
                        help="Liste des id_election à traiter")
    parser.add_argument('--list',      action='store_true',
                        help="Lister les élections disponibles et quitter")
    args = parser.parse_args()

    # ── Chargement ──────────────────────────────────────────────────────────
    cand, gen, passage = load_data(args.cand, args.gen, args.table)

    if args.list:
        available = sorted(cand['id_election'].unique().tolist())
        print(f"\n{len(available)} élections disponibles :")
        for e in available:
            print(f"  {e}")
        return

    # ── Traitement par élection ──────────────────────────────────────────────
    os.makedirs(args.outdir, exist_ok=True)

    for id_election in args.elections:
        print(f"\n{'='*60}")
        print(f"  {id_election}")
        print('='*60)

        iris_agg = process_election(id_election, cand, gen, passage)
        if iris_agg is None:
            continue

        validate(id_election, iris_agg)

        out_path = os.path.join(args.outdir, f"{id_election}.csv")
        iris_agg.to_csv(out_path, encoding='utf-8')
        print(f"\n  Sauvegarde → {out_path}  ({len(iris_agg):,} IRIS)")
        print(f"  Colonnes : {list(iris_agg.columns)}")

    print("\nTerminé.")


if __name__ == '__main__':
    main()
