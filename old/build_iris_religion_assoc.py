# -*- coding: utf-8 -*-
"""
build_iris_religion_assoc.py
Enrichit le dataset IRIS avec des variables religieuses et associatives :

  Mission 1 (BPE) :
    nb_lieux_culte_bpe(_pour1000)    — lieux de culte physiques
    bpe_grande_surface_pour1000      — grandes surfaces alimentaires (marqueur commercial)

  Mission 2 (RNA) :
    rna_culte_total/catholique/protestant/islam/juif/bouddhiste (+ _pour1000)
    — associations religieuses actives géocodées à l'IRIS via API BAN + spatial join

  Mission 3 (RNA) :
    rna_solidarite_active(_pour1000) — associations de solidarité/insertion
    rna_sport_distinction(_pour1000) — sports de distinction (golf, tennis, voile, équitation)

Pipeline géocodage :
  1. Filtre RNA : code 040000 (culte) ou codes solidarité/sport + filtre actif
  2. Adresse RNA → BAN API → lat/lon (batch 10k)
  3. lat/lon → point-in-polygon IRIS (geopandas sjoin)
  4. Fallback : nearest IRIS si hors-frontière
  5. Fallback proportionnel pop si adresse insuffisante ou score BAN < 0.5

Usage :
  conda run -n vadim_env python build_iris_religion_assoc.py

Sortie : iris/iris_religion_assoc.csv
"""

import os
import re
import io
import glob
import time
import requests
import numpy as np
import pandas as pd
import geopandas as gpd

# =============================================================================
# Chemins et constantes
# =============================================================================
PATH_IRIS          = "iris/"
RNA_DIR            = "rna_waldec/"
F_IRIS_REF         = os.path.join(PATH_IRIS, "iris_final_socio_politique.csv")
F_BPE              = os.path.join(PATH_IRIS, "ds_bpe_iris_2024_geo_2024.csv")
F_IRIS_POLYGONS    = os.path.join(PATH_IRIS, "contours_iris_2025.gpkg")
OUT_FILE           = os.path.join(PATH_IRIS, "iris_religion_assoc.csv")

BAN_URL            = "https://api-adresse.data.gouv.fr/search/csv/"
BAN_BATCH_SIZE     = 10_000
BAN_SCORE_MIN      = 0.5
MIN_POP_POUR1000   = 50   # IRIS sous ce seuil → NaN pour les variables /1000

# Codes RNA culte
CODE_CULTE = {'040000'}

# Codes RNA solidarité (action sociale, insertion, urgence)
CODE_SOLIDARITE = {
    '019000', '019005', '019010', '019012', '019014', '019016',
    '019020', '019025', '019030', '019032', '019035',
    '020000', '020005', '020010', '020015', '020020', '020025',
}

# Codes RNA sport de distinction
CODE_SPORT_DISTINCTION = {'011100', '011165', '011185'}  # golf, tennis, voile

# =============================================================================
# Patterns de classification RNA
# =============================================================================
NOISE_PATTERN = re.compile(
    r'\b(?:école|collège|lycée|université|sorbonne|patrimoine bati|'
    r'aide alimentaire|club sportif|association sportive)\b',
    re.IGNORECASE
)

DENOM_PATTERNS = {
    'catholique': re.compile(
        # Institutions et structures
        r'\b(?:paroisse|paroissial[e]?|dioc[eè]se|dioc[eé]sain[e]?|catholique|sanctuaire|abbaye|'
        r'chapelle|doyenn[eé]|prieur[eé]|cath[eé]drale|basilique|'
        # Ordres religieux
        r'carmel|carm[eé]lite|dominicain|franciscain|j[eé]suite|cistercien|b[eé]n[eé]dictin|'
        r'oblat[s]?|sal[eé]sien|assomptionniste|opus dei|capucin|clarisses?|'
        r'fr[eè]res? mineurs|petites? soeurs?|soeurs? de[- ](?:saint|la|notre)|'
        # Mouvements catholiques
        r'n[eé]o[- ]cat[eé]chum[eé]nal|focolare|foucauld|charles de foucauld|'
        # Pratique liturgique catholique spécifique
        r'messe|eucharistie|sacr[eé][- ]c[oe]ur|rosaire|'
        # pèlerin/pèlerinage : qualifier pour éviter le pèlerinage à La Mecque
        r'p[eè]lerin[s]? (?:catholique|chr[eé]tien|notre[- ]dame|saint|de lourdes|de rome|de compostelle)|'
        r'p[eè]lerinage (?:catholique|chr[eé]tien|notre[- ]dame|de lourdes|de rome|de compostelle|mariale)|'
        # aumônerie : qualifier pour éviter aumônerie musulmane, militaire générique
        r'aumon[eé]rie catholique|aum[oô]nerie catholique|'
        r'aumon[eé]rie (?:des [eé]tudiants|des lyc|des coll[eè]|des jeunes chr|universitaire chr)|'
        # Gestion scolaire catholique
        r'ogec|organisme de gestion.{0,20}catholique|[eé]cole catholique|'
        # Toponymie patronymique : exiger un contexte religieux explicite
        # (évite Saint-Étienne ville, Saint-Martin-de-Crau, etc.)
        r'notre[- ]dame|'
        # Prénom saint SEULEMENT après un mot religieux (paroisse, église, chapelle...)
        r'(?:paroisse|[eé]glise|chapelle|cath[eé]drale|prieur[eé]|abbaye|confrerie|confr[eé]rie|ogec)'
        r'[- \w]{0,20}(?:saint|sainte|st|ste)[- ](?:pierre|paul|jean|marie|joseph|nicolas|thomas|'
        r'fran[cç]ois|antoine|augustin|ignace|dominique|bernard|benoît|benoit|'
        r'vincent|louis|roch|anne|claire|th[eé]r[eè]se|scholastique|bruno|'
        r'martin|luc|marc|matthieu|matthias|barnab[eé]|philipp[e]?|ch[a]?rles)|'
        r'confrerie|confr[eé]rie)\b',
        re.IGNORECASE
    ),
    'evangelique': re.compile(
        # Termes avec et sans accents (titres RNA souvent en majuscules ASCII)
        r'\b(?:[eé]vang[eé]lique|evang[eé]lique|[eé]vangeli[sz]|evang[eé]li|'
        r'pentec[oô]tiste|pentecotiste|pentec[oô]te|pentecote|'
        r'assembl[eé]e[s]? de dieu|assemblee[s]? de dieu|'
        r'assembl[eé]e chr[eé]tienne|assemblee chretienne|'
        r'maranatha|baptiste|adventiste|'
        r'chr[eé]tien[s]?[- ][eé]vang[eé]lique|chretien[s]?[- ]evang[eé]lique|'
        r'[eé]glise [eé]vang[eé]lique|eglise evang[eé]lique|'
        r'fraternit[eé] [eé]vang[eé]lique|'
        r'christ [àa] l.oeuvre|bonne nouvelle|bon berger|'
        r'centre chr[eé]tien|centre chretien|'
        r'communaut[eé] chr[eé]tienne|communaute chretienne|'
        r'[eé]glise chr[eé]tienne|eglise chretienne|'
        r'[eé]glise du christ|eglise du christ|'
        r'mission chr[eé]tienne|mission chretienne|'
        r'chr[eé]tien[s]? pour|chretien[s]? pour|pour christ|'
        r'r[eé]union[s]? chr[eé]tienne[s]?|reunion[s]? chretienne[s]?|'
        r'association chr[eé]tienne|'
        r'arm[eé]e du salut|tabernacle|new life|r[eé]veil|reveil|'
        # Catch-all : "chrétien" seul dans le titre quand pas déjà catholique/protestant
        # (dénominations non-identifiées mais clairement chrétiennes)
        r'chr[eé]tien[s]?|chretien[s]?)\b',
        re.IGNORECASE
    ),
    'protestant': re.compile(
        r'\b(?:protestant[e]?[s]?|temple protestant|temple r[eé]form[eé]|'
        r'luth[eé]rien[s]?|lutherien[s]?|m[eé]thodiste[s]?|methodiste[s]?|'
        r'[eé]glise protestante unie|eglise protestante unie|epu[- ]\w|'
        r'[eé]glise r[eé]form[eé]e|eglise reformee|'
        r'consistoire protestant|f[eé]d[eé]ration protestante|federation protestante|'
        r'r[eé]form[eé][e]?|calviniste|presbyt[eé]rien|presbyterien|'
        r'huguenot|uepal|eer[f]?|cnef|famille[s]? protestante[s]?)\b',
        re.IGNORECASE
    ),
    'orthodoxe': re.compile(
        r'\b(?:orthodoxe|[eé]glise orthodoxe|eglise orthodoxe|paroisse orthodoxe|'
        r'patriarcat|m[eé]tropolite|metropolite|cath[eé]drale orthodoxe|'
        r'grec orthodoxe|russe orthodoxe|serbe orthodoxe|roumain[e]? orthodoxe|'
        r'bulgare orthodoxe|antiochien|constantinople|'
        r'copte|[eé]glise copte|eglise copte|syriaque|'
        r'arm[eé]nien.{0,5}apostolique|armenien.{0,5}apostolique|'
        r'[eé]glise arm[eé]nienne|eglise armenienne|'
        r'[eé]thiopien orthodoxe|ethiopien orthodoxe|[eé]glise orientale)\b',
        re.IGNORECASE
    ),
    'islam': re.compile(
        r'\b(?:mosqu[eé]e|mosquee|masjid|islamique|musulman[e]?[s]?|'
        r'centre islamique|association islamique|'
        r'sunnite|chiite|salafia|salafiste|soufisme|soufi|'
        r'al[- ]masjid|al[- ]sunna|ahl[- ]al|oumma|umma|'
        r'tabligh|da[wv]a|da[wv]ah|'
        r'cultuelle musulmane|culturelle musulmane|'
        r'foyer musulman|cercle musulman|union des musulmans|r[eé]union des musulmans|'
        r'coran|quran|'
        r'oratoire musulman|salle de pri[eè]re|salle de priere|'
        # Associations cultuelles de communautés d'origine musulmane sans le mot "musulman"
        r'cultuelle (?:marocaine|alg[eé]rienne|algerienne|tunisienne|turque|franco[- ]arabe|'
        r'franco[- ]turc|franco[- ]maroc|franco[- ]alg[eé]|'
        r'des marocains|des alg[eé]riens|des tunisiens|des turcs)|'
        r'firdaws|badr[- ]|al[- ]salam cultuelle|sakina[- ])\b',
        re.IGNORECASE
    ),
    'juif': re.compile(
        r'\b(?:synagogue|juif|juive|isra[eé]lite|israelite|consistoire|beth[- ]?din|'
        r'juda[iï]que|judaique|juda[iï]sme|judaisme|'
        r'beth[- ]?midrash|talmud[- ]?torah|talmud|torah|'
        r'hassidique|loubavitch|chabad|s[eé]farad[e]?|sefara[d]?|ashk[eé]naze|ashkenaze|'
        r'kehila|kehilla|shabbat|kippour|pessah|'
        r'bar[- ]mitsva|bat[- ]mitsva|bar[- ]mitzvah|bat[- ]mitzvah|'
        r'mikv[eé]|casher|kosher|masorti|ohel|'
        r'cultuelle isra[eé]lite|cultuelle israelite|'
        r'communaut[eé] isra[eé]lite|communaute israelite|'
        r'isra[eé]l\b(?! .{0,10}(?:etat|pays|ambassade|consulat|amis d))|'
        r'sioniste|mizrahi|hapoel|ahavat)\b',
        re.IGNORECASE
    ),
    'bouddhiste': re.compile(
        r'\b(?:bouddhiste[s]?|bouddha|bouddhisme|bouddhique|'
        r'zen|tib[eé]tain|tibetain|theravada|vajrayana|mahayana|hinayana|'
        r'sangha|dharma|pagode|centre bouddhiste|association bouddhiste|'
        r'dzogchen|kag[yu][uü]|kagyu|karmapa|karma[- ]\w+|gelug|nyingma|sakya|rimay|rim[eé]|'
        r'vipassana|vihara|stupa|lama[- ]\w+|rinpoch[eé]|rinpoche|'
        r'soka gakkai|soka|nichiren|shin bouddhisme|'
        r'f[eé]d[eé]ration.{0,15}bouddhiste|federation.{0,15}bouddhiste)\b',
        re.IGNORECASE
    ),
    'hindou': re.compile(
        r'\b(?:hindou[e]?[s]?|hindouisme|temple hindou|mandir|'
        r'sikh[s]?|sikhisme|gurdwara|'
        r'hare krishna|iskcon|vishnou|shiva|ganesh|ganesha|'
        r'sri[- ]\w+|sai baba|sathya sai|'
        r'alayam|kovil|ashram[- ]?)\b',
        re.IGNORECASE
    ),
    'mormon': re.compile(
        r'\b(?:mormon|latter[- ]day saint|'
        r'[eé]glise de j[eé]sus[- ]christ des saints des derniers jours|'
        r'eglise de jesus[- ]christ des saints des derniers jours|'
        r'saints des derniers jours|brigham|lds[- ]\w)\b',
        re.IGNORECASE
    ),
    'temoins_jehovah': re.compile(
        r'\b(?:t[eé]moin[s]? de j[eé]hovah|temoin[s]? de jehovah|'
        r'j[eé]hovah|jehovah|watchtower|tour de garde|salle du royaume|'
        r'association locale pour le culte des t[eé]moins)\b',
        re.IGNORECASE
    ),
    # Catch-all chrétien : tout ce qui est clairement chrétien sans dénomination précise.
    # Inclut tous les termes des sous-catégories + termes génériques chrétiens.
    # Utilisé pour rna_culte_chretien_total ; une asso peut matcher ici ET une sous-cat.
    'chretien_catchall': re.compile(
        # Termes théologiques / textes sacrés
        r'\b(?:bible|biblique|[eé]vangile[s]?|evangile[s]?|gospel|'
        r'j[eé]sus[- ]?christ|jesus[- ]?christ|j[eé]sus\b|jesus\b|'
        r'christ\b|christianisme|chr[eé]tien[s]?|chretien[s]?|'
        # seigneur seul est inter-religieux → exiger contexte chrétien
        r'seigneur[- ](?:j[eé]sus|christ)|gloire[- ]au[- ]seigneur[- ](?:j[eé]sus|christ)|'
        r'r[eé]surrection|resurrection|croix\b|'
        r'esprit[- ]saint|saint[- ]esprit|trinit[eé]|trinite|'
        r'[eé]vang[eé]lisation|evangelisation|[eé]vangili[sz]|'
        r'ap[oô]tre[s]?|apotre[s]?|'
        # prophète seul est inter-religieux (islam, juif...) → exiger contexte chrétien
        r'proph[eè]te[s]?[- ](?:chr[eé]|[eé]vang|de christ|du christ)|'
        r'grâce divine|gr[aâ]ce[- ]de[- ]dieu|gloire[- ]de[- ]dieu|'
        r'parole[- ]de[- ]dieu|r[eè]gne[- ]de[- ]dieu|'
        r'salut[- ](?:chr[eé]tien|[eé]ternel)|lou(?:ange|er)[- ](?:dieu|christ|seigneur)|'
        # Lieux bibliques utilisés comme noms d'associations
        r'gethsemani|bethesda|beth[eé]l\b|nazareth|b[eé]thl[eé]em|'
        r'sion[- ](?:chr|[eé]v|miss)|[eé]glise[- ]sion|mont[- ]sion|'
        # Structures ecclésiastiques génériques
        r'[eé]glise\b|eglise\b|apostolique|oecum[eé]nique|ecumenique|'
        r'pasteur[al]?\b|pastorale?|minist[eè]re[- ](?:chr[eé]|[eé]vang|miss)|'
        r'mission[- ](?:chr[eé]|[eé]vang|internat)|missionnaire[s]?|'
        r'monast[eè]re|monastique|monastere|'
        r'catéch[eè]se|catechese|catéchisme|catechisme|'
        # Mouvements / associations chrétiennes génériques (termes toujours qualifiés)
        r'foi[- ]et[- ](?:lumi[eè]re|vie|lib)|foyer[- ]chr[eé]|'
        r'pri[eè]re[- ](?:chr[eé]|[eé]vang)|ateliers[- ]de[- ]pri[eè]re|'
        r'alliance[- ](?:chr[eé]|bibl)|biblique[- ]franc|'
        r'r[eé]veil[- ]chr[eé]|[eé]glise[- ]du[- ]r[eé]veil|'
        r'louange[s]?[- ](?:chr[eé]|[eé]vang|de[- ]dieu|de[- ]christ)|'
        r'adoration[- ](?:chr|[eé]v)|'
        r'communion[- ]chr[eé]|fraternit[eé][- ]chr[eé]|'
        r'gloire[- ](?:chr[eé]|de[- ]dieu|de[- ]christ|du[- ]seigneur)|'
        r'b[eé]n[eé]diction[- ](?:chr[eé]|de[- ]dieu)|'
        r'onction[- ](?:chr[eé]|du[- ]saint[- ]esprit)|'
        r'intercession[- ](?:chr[eé]|[eé]vang))\b',
        re.IGNORECASE
    ),
}

SOLIDARITE_TEXT = re.compile(
    r'\b(?:aide alimentaire|banque alimentaire|restos? du c[oe]ur|solidarité|'
    r'urgence sociale|défense des droits|insertion|réinsertion|secours|'
    r'distribut|épicerie sociale|vestiaire|hébergement d[. ]urgence|accueil urgence)\b',
    re.IGNORECASE
)

SPORT_DISTINCTION_TEXT = re.compile(
    r'\b(?:golf|équitation|polo|yacht|voile club|tennis club|cercle équestre|'
    r'hippique|country club|club de golf|golf club)\b',
    re.IGNORECASE
)


# =============================================================================
# Utilitaire de lecture RNA (sep=';', encodage auto)
# =============================================================================
def read_rna(filepath):
    try:
        df = pd.read_csv(
            filepath, sep=";", encoding="utf-8", low_memory=False,
            dtype=str, on_bad_lines="skip"
        )
    except UnicodeDecodeError:
        df = pd.read_csv(
            filepath, sep=";", encoding="latin1", low_memory=False,
            dtype=str, on_bad_lines="skip"
        )
    df.columns = [c.lstrip('\ufeff').strip() for c in df.columns]
    return df


# =============================================================================
# Chargement de la référence IRIS (COM, pop_totale)
# =============================================================================
def load_iris_ref():
    print(f"Chargement référence IRIS : {F_IRIS_REF}...")
    df = pd.read_csv(F_IRIS_REF, dtype={'IRIS': str, 'COM': str}, low_memory=False,
                     usecols=['IRIS', 'COM', 'pop_totale'])
    df['IRIS'] = df['IRIS'].astype(str).str.zfill(9)
    df['COM']  = df['COM'].astype(str).str.zfill(5)
    df['pop_totale'] = pd.to_numeric(df['pop_totale'], errors='coerce').fillna(0)
    print(f"  {len(df)} IRIS référencés")
    return df


# =============================================================================
# Construction de la table commune → [(iris, poids), ...]
# =============================================================================
def load_iris_mapping(df_iris_ref):
    """Retourne {com_5digits: [(iris_9digits, poids), ...]} pondéré par pop_totale."""
    df = df_iris_ref[['IRIS', 'COM', 'pop_totale']].copy()
    com_pop = df.groupby('COM')['pop_totale'].transform('sum')
    df['weight'] = df['pop_totale'] / com_pop.replace(0, np.nan)
    # Fallback poids égaux si pop_commune == 0
    n_iris_in_com = df.groupby('COM')['IRIS'].transform('count')
    df['weight'] = df['weight'].fillna(1.0 / n_iris_in_com)

    mapping = {}
    for com, grp in df.groupby('COM'):
        mapping[com] = list(zip(grp['IRIS'].tolist(), grp['weight'].tolist()))
    return mapping


# =============================================================================
# Mission 1 — BPE : lieux de culte + grandes surfaces
# =============================================================================
def build_bpe(df_iris_ref):
    print(f"\nChargement BPE : {F_BPE}...")
    try:
        df_bpe = pd.read_csv(F_BPE, sep=";", encoding="utf-8", dtype={'GEO': str}, low_memory=False)
    except UnicodeDecodeError:
        df_bpe = pd.read_csv(F_BPE, sep=";", encoding="latin1", dtype={'GEO': str}, low_memory=False)
    df_bpe['GEO'] = df_bpe['GEO'].astype(str).str.zfill(9)
    df_bpe['OBS_VALUE'] = pd.to_numeric(df_bpe['OBS_VALUE'], errors='coerce').fillna(0)
    print(f"  {len(df_bpe)} lignes BPE chargées")

    # Lieux de culte : FACILITY_SDOM == 'F3'
    culte = df_bpe[df_bpe['FACILITY_SDOM'] == 'F3']
    nb_culte = culte.groupby('GEO')['OBS_VALUE'].sum().rename('nb_lieux_culte_bpe')
    print(f"  Lieux de culte (F3) : {nb_culte.sum():.0f} dans {len(nb_culte)} IRIS")

    # Grandes surfaces alimentaires : FACILITY_SDOM == 'B2'
    gs = df_bpe[df_bpe['FACILITY_SDOM'] == 'B2']
    nb_gs = gs.groupby('GEO')['OBS_VALUE'].sum().rename('nb_grande_surface_bpe')
    print(f"  Grandes surfaces (B2) : {nb_gs.sum():.0f} dans {len(nb_gs)} IRIS")

    result = df_iris_ref[['IRIS']].set_index('IRIS').copy()
    result = result.join(nb_culte.rename_axis('IRIS'), how='left')
    result = result.join(nb_gs.rename_axis('IRIS'), how='left')
    result['nb_lieux_culte_bpe']  = result['nb_lieux_culte_bpe'].fillna(0)
    result['nb_grande_surface_bpe'] = result['nb_grande_surface_bpe'].fillna(0)

    return result


# =============================================================================
# Construction de l'adresse BAN depuis colonnes RNA
# =============================================================================
def build_ban_address_col(df):
    def clean(col):
        return df[col].fillna('').astype(str).str.strip().replace({'_': ''}, regex=False)

    num   = clean('adrs_numvoie')
    typev = clean('adrs_typevoie')
    libv  = clean('adrs_libvoie')
    df = df.copy()
    df['adresse_ban'] = (num + ' ' + typev + ' ' + libv).str.strip()
    df['adresse_ok']  = df['adresse_ban'].str.len() >= 5
    df['postcode_ban'] = df['adrs_codepostal'].fillna('').astype(str).str.zfill(5)
    df['city_ban']     = df['adrs_libcommune'].fillna('').astype(str)
    df['adrs_codeinsee'] = df['adrs_codeinsee'].fillna('').astype(str).str.zfill(5)
    return df


# =============================================================================
# Géocodage BAN par batch CSV
# =============================================================================
def geocode_ban_batch(df_chunk, retries=3):
    ban_df = df_chunk[['id', 'adresse_ban', 'postcode_ban', 'city_ban']].copy()
    ban_df.columns = ['id', 'adresse', 'postcode', 'city']
    csv_buf = ban_df.to_csv(index=False).encode('utf-8')

    for attempt in range(retries):
        try:
            resp = requests.post(
                BAN_URL,
                files={'data': ('data.csv', csv_buf, 'text/csv')},
                data=[('columns', 'adresse'), ('columns', 'city'), ('postcode', 'postcode')],
                timeout=120
            )
            resp.raise_for_status()
            result = pd.read_csv(io.StringIO(resp.text), dtype=str, low_memory=False)
            col_map = {}
            for c in result.columns:
                cl = c.lower()
                if cl == 'latitude':     col_map[c] = 'latitude'
                if cl == 'longitude':    col_map[c] = 'longitude'
                if cl == 'result_score': col_map[c] = 'result_score'
                if cl == 'result_type':  col_map[c] = 'result_type'
            result.rename(columns=col_map, inplace=True)
            return result
        except Exception as e:
            if attempt < retries - 1:
                wait = 2 ** attempt
                print(f"    [BAN] Erreur tentative {attempt+1}/{retries} : {e}. Retry {wait}s...")
                time.sleep(wait)
            else:
                print(f"    [BAN] Échec après {retries} tentatives : {e}")
                return pd.DataFrame()


# =============================================================================
# Spatial join BAN résultats → IRIS polygones
# =============================================================================
def sjoin_to_iris(df_geocoded, gdf_iris):
    """
    Retourne df avec colonnes CODE_IRIS et geocode_method ajoutées.
    Les lignes sans score valide reçoivent CODE_IRIS=None / method='fallback'.
    """
    df = df_geocoded.copy()
    df['latitude']     = pd.to_numeric(df.get('latitude',     pd.Series(dtype=str)), errors='coerce')
    df['longitude']    = pd.to_numeric(df.get('longitude',    pd.Series(dtype=str)), errors='coerce')
    df['result_score'] = pd.to_numeric(df.get('result_score', pd.Series(dtype=str)), errors='coerce').fillna(0)

    mask_valid = (df['result_score'] >= BAN_SCORE_MIN) & df['latitude'].notna() & df['longitude'].notna()
    df_valid   = df[mask_valid].copy()
    df_invalid = df[~mask_valid].copy()

    results = []

    if len(df_valid) > 0:
        gdf_pts = gpd.GeoDataFrame(
            df_valid,
            geometry=gpd.points_from_xy(df_valid['longitude'], df_valid['latitude']),
            crs="EPSG:4326"
        )
        if gdf_pts.crs != gdf_iris.crs:
            gdf_pts = gdf_pts.to_crs(gdf_iris.crs)

        # Join within
        joined = gpd.sjoin(gdf_pts, gdf_iris[['CODE_IRIS', 'geometry']], how='left', predicate='within')
        matched   = joined[joined['CODE_IRIS'].notna()].copy()
        unmatched = joined[joined['CODE_IRIS'].isna()].drop(columns=['index_right'], errors='ignore').copy()
        matched['geocode_method'] = 'ban_within'
        results.append(matched[['id', 'CODE_IRIS', 'geocode_method']])

        # Fallback nearest
        if len(unmatched) > 0:
            nearest = gpd.sjoin_nearest(
                unmatched[['id', 'geometry']],
                gdf_iris[['CODE_IRIS', 'geometry']],
                how='left'
            ).copy()
            nearest['geocode_method'] = 'ban_nearest'
            nearest = nearest[nearest['CODE_IRIS'].notna()]
            results.append(nearest[['id', 'CODE_IRIS', 'geocode_method']])

    # Non-géocodés
    df_invalid_out = df_invalid[['id']].copy()
    df_invalid_out['CODE_IRIS'] = None
    df_invalid_out['geocode_method'] = 'fallback'
    results.append(df_invalid_out)

    return pd.concat(results, ignore_index=True)


# =============================================================================
# Allocation proportionnelle (fallback commune → IRIS)
# =============================================================================
def allocate_proportional(df_assoc, iris_mapping, col_counts):
    """
    Répartit les associations non géocodées selon la population de chaque IRIS dans leur commune.
    df_assoc : DataFrame avec 'adrs_codeinsee' et colonnes booléennes dans col_counts
    col_counts : {colname: Series booléenne}
    Retourne : DataFrame indexé par IRIS avec float counts
    """
    result_dict = {}

    for com_code, iris_list in iris_mapping.items():
        mask = df_assoc['adrs_codeinsee'] == com_code
        com_rows = df_assoc[mask]
        if len(com_rows) == 0:
            continue
        for col, col_mask in col_counts.items():
            n = col_mask.reindex(com_rows.index, fill_value=False).sum()
            if n == 0:
                continue
            for iris_code, weight in iris_list:
                if iris_code not in result_dict:
                    result_dict[iris_code] = {}
                result_dict[iris_code][col] = result_dict[iris_code].get(col, 0.0) + n * weight

    if not result_dict:
        return pd.DataFrame(columns=['IRIS'] + list(col_counts.keys())).set_index('IRIS')
    df_out = pd.DataFrame.from_dict(result_dict, orient='index')
    df_out.index.name = 'IRIS'
    return df_out


# =============================================================================
# Traitement d'un fichier RNA : filtrage, classification, géocodage
# =============================================================================
def process_rna_file(fpath, gdf_iris, iris_mapping):
    """
    Traite un fichier RNA départemental.
    Retourne un DataFrame indexé par IRIS avec les colonnes de comptage.
    """
    df = read_rna(fpath)
    for col in ['objet_social1', 'objet_social2', 'date_disso', 'titre', 'objet',
                'adrs_numvoie', 'adrs_typevoie', 'adrs_libvoie',
                'adrs_codepostal', 'adrs_libcommune', 'adrs_codeinsee', 'id']:
        if col not in df.columns:
            df[col] = ''

    # Filtre actif
    mask_active = (
        df['date_disso'].isna() |
        df['date_disso'].str.strip().isin(['', '0001-01-01'])
    )

    # Catégories d'intérêt
    mask_culte = (
        df['objet_social1'].str.strip().isin(CODE_CULTE) |
        df['objet_social2'].str.strip().isin(CODE_CULTE)
    )
    mask_soli = (
        df['objet_social1'].str.strip().isin(CODE_SOLIDARITE) |
        df['objet_social2'].str.strip().isin(CODE_SOLIDARITE)
    )
    mask_sport = (
        df['objet_social1'].str.strip().isin(CODE_SPORT_DISTINCTION) |
        df['objet_social2'].str.strip().isin(CODE_SPORT_DISTINCTION)
    )

    # Solidarité : élargir avec text matching même hors code
    # Sport distinction : pareil
    df['_text'] = (df['titre'].fillna('') + ' ' + df['objet'].fillna('')).str.lower()
    mask_soli  = mask_soli  | df['_text'].str.contains(SOLIDARITE_TEXT, na=False)
    mask_sport = mask_sport | df['_text'].str.contains(SPORT_DISTINCTION_TEXT, na=False)

    mask_any = mask_active & (mask_culte | mask_soli | mask_sport)
    df_sel = df[mask_any].copy()
    if len(df_sel) == 0:
        return None

    # Anti-bruit culte
    def is_noise(row):
        return (
            bool(NOISE_PATTERN.search(str(row.get('titre', ''))))
            and bool(NOISE_PATTERN.search(str(row.get('objet', ''))))
        )
    # Appliquer uniquement sur les lignes culte pour ne pas supprimer solidarité/sport
    culte_idx = df_sel.index[
        df_sel['objet_social1'].str.strip().isin(CODE_CULTE) |
        df_sel['objet_social2'].str.strip().isin(CODE_CULTE)
    ]
    noise_mask = pd.Series(False, index=df_sel.index)
    if len(culte_idx) > 0:
        noise_mask[culte_idx] = df_sel.loc[culte_idx].apply(is_noise, axis=1)
    df_sel = df_sel[~noise_mask].copy()

    if len(df_sel) == 0:
        return None

    # Construction colonnes booléennes de classification
    df_sel['_text'] = (df_sel['titre'].fillna('') + ' ' + df_sel['objet'].fillna('')).str.lower()

    df_sel['c_culte_total']    = mask_culte.reindex(df_sel.index, fill_value=False)
    for denom, pat in DENOM_PATTERNS.items():
        if denom != 'chretien_catchall':
            df_sel[f'c_culte_{denom}'] = df_sel['_text'].str.contains(pat, na=False)
    df_sel['c_solidarite']     = mask_soli.reindex(df_sel.index, fill_value=False)
    df_sel['c_sport_distinct'] = mask_sport.reindex(df_sel.index, fill_value=False)

    # Total chrétien : union de toutes les sous-catégories chrétiennes + catch-all
    chretien_subcols = ['c_culte_catholique', 'c_culte_evangelique', 'c_culte_protestant',
                        'c_culte_orthodoxe', 'c_culte_mormon', 'c_culte_temoins_jehovah']
    catchall_hits = df_sel['_text'].str.contains(DENOM_PATTERNS['chretien_catchall'], na=False)
    df_sel['c_culte_chretien_total'] = (
        df_sel[chretien_subcols].any(axis=1) | catchall_hits
    )

    COLS = ['c_culte_total', 'c_culte_chretien_total',
            'c_culte_catholique', 'c_culte_evangelique',
            'c_culte_protestant', 'c_culte_orthodoxe',
            'c_culte_islam', 'c_culte_juif', 'c_culte_bouddhiste',
            'c_culte_hindou', 'c_culte_mormon', 'c_culte_temoins_jehovah',
            'c_solidarite', 'c_sport_distinct']

    # Construction adresses BAN
    df_sel = build_ban_address_col(df_sel)

    # Séparer les associations avec adresse valide de celles sans
    df_to_geo  = df_sel[df_sel['adresse_ok']].copy()
    df_no_addr = df_sel[~df_sel['adresse_ok']].copy()

    # --- Géocodage BAN ---
    all_geo_results = []
    for i in range(0, len(df_to_geo), BAN_BATCH_SIZE):
        chunk = df_to_geo.iloc[i:i + BAN_BATCH_SIZE]
        result = geocode_ban_batch(chunk)
        if len(result) > 0:
            # Ajouter adrs_codeinsee pour vérification
            id_col = 'id' if 'id' in result.columns else result.columns[0]
            result = result.rename(columns={id_col: 'id'})
            result = result.merge(
                df_to_geo[['id', 'adrs_codeinsee']],
                on='id', how='left'
            )
            all_geo_results.append(result)
        time.sleep(0.3)

    # --- Spatial join → CODE_IRIS ---
    located_iris = pd.DataFrame(columns=['id', 'CODE_IRIS', 'geocode_method'])
    if all_geo_results:
        df_geocoded = pd.concat(all_geo_results, ignore_index=True)
        located_iris = sjoin_to_iris(df_geocoded, gdf_iris)

    # Merge classification back
    df_to_geo_mapped = df_to_geo[['id'] + COLS].merge(located_iris, on='id', how='left')

    # Associations géocodées avec IRIS attribué → comptage direct
    df_direct = df_to_geo_mapped[df_to_geo_mapped['CODE_IRIS'].notna()].copy()

    # Associations sans IRIS (géocodage échoué) → fallback proportionnel
    df_geo_fallback = df_to_geo_mapped[df_to_geo_mapped['CODE_IRIS'].isna()].copy()
    df_geo_fallback['adrs_codeinsee'] = df_to_geo.set_index('id').reindex(
        df_geo_fallback['id']
    )['adrs_codeinsee'].values

    # Compter par IRIS pour les associations directement géocodées
    iris_counts = pd.DataFrame(0.0, index=pd.Index([], name='IRIS'), columns=COLS)
    if len(df_direct) > 0:
        direct_agg = df_direct.groupby('CODE_IRIS')[COLS].sum()
        direct_agg.index.name = 'IRIS'
        iris_counts = direct_agg.astype(float)

    # Fallback proportionnel pour : geo_fallback + no_addr
    df_fallback_all = pd.concat([
        df_geo_fallback[['id', 'adrs_codeinsee'] + COLS],
        df_no_addr[['id', 'adrs_codeinsee'] + COLS]
    ], ignore_index=True)

    if len(df_fallback_all) > 0:
        col_counts_map = {col: df_fallback_all[col].astype(bool) for col in COLS}
        prop_counts = allocate_proportional(df_fallback_all, iris_mapping, col_counts_map)
        if len(prop_counts) > 0:
            iris_counts = iris_counts.add(prop_counts, fill_value=0)

    return iris_counts


# =============================================================================
# Mission 2+3 — RNA : tous les fichiers départementaux
# =============================================================================
def build_rna(gdf_iris, iris_mapping):
    rna_files = sorted(glob.glob(os.path.join(RNA_DIR, "rna_waldec_*.csv")))
    if not rna_files:
        raise FileNotFoundError(f"Aucun fichier RNA dans {RNA_DIR}")
    print(f"\nTraitement RNA : {len(rna_files)} fichiers départementaux")

    all_chunks = []
    for i, fpath in enumerate(rna_files):
        dpt = os.path.basename(fpath).replace('rna_waldec_20260306_dpt_', '').replace('.csv', '')
        print(f"  [{i+1:3d}/{len(rna_files)}] dpt {dpt:3s}...", end=' ', flush=True)
        try:
            chunk = process_rna_file(fpath, gdf_iris, iris_mapping)
            if chunk is not None and len(chunk) > 0:
                all_chunks.append(chunk)
                print(f"{len(chunk)} IRIS touchés")
            else:
                print("(aucune association pertinente)")
        except Exception as e:
            print(f"ERREUR : {e}")

    if not all_chunks:
        print("  [WARN] Aucune donnée RNA collectée.")
        return pd.DataFrame()

    # Agréger tous les départements
    result = pd.concat(all_chunks, axis=0)
    result = result.groupby(result.index).sum()

    # Renommer les colonnes c_ → rna_
    rename_map = {
        'c_culte_total':           'rna_culte_total',
        'c_culte_chretien_total':  'rna_culte_chretien_total',
        'c_culte_catholique':     'rna_culte_catholique',
        'c_culte_evangelique':    'rna_culte_evangelique',
        'c_culte_protestant':     'rna_culte_protestant',
        'c_culte_orthodoxe':      'rna_culte_orthodoxe',
        'c_culte_islam':          'rna_culte_islam',
        'c_culte_juif':           'rna_culte_juif',
        'c_culte_bouddhiste':     'rna_culte_bouddhiste',
        'c_culte_hindou':         'rna_culte_hindou',
        'c_culte_mormon':         'rna_culte_mormon',
        'c_culte_temoins_jehovah':'rna_culte_temoins_jehovah',
        'c_solidarite':           'rna_solidarite_active',
        'c_sport_distinct':       'rna_sport_distinction',
    }
    result.rename(columns=rename_map, inplace=True)
    return result


# =============================================================================
# Normalisation /1000 habitants, cap p99
# =============================================================================
def normalize_pour1000(df_result, pop_map, raw_cols):
    pop = df_result.index.map(pop_map).astype(float)
    pop_k = (pop / 1000.0).where(pop >= MIN_POP_POUR1000)

    for col in raw_cols:
        if col not in df_result.columns:
            continue
        raw = df_result[col] / pop_k
        p99 = raw.quantile(0.99)
        df_result[f'{col}_pour1000'] = raw.clip(upper=p99)

    return df_result


# =============================================================================
# MAIN
# =============================================================================
def main():
    print("=" * 60)
    print("  build_iris_religion_assoc.py")
    print("=" * 60)

    # 1. Référence IRIS
    df_iris_ref = load_iris_ref()
    pop_map     = df_iris_ref.set_index('IRIS')['pop_totale'].to_dict()
    iris_mapping = load_iris_mapping(df_iris_ref)
    iris_index  = df_iris_ref['IRIS'].values

    # 2. IRIS polygones
    print(f"\nChargement polygones IRIS : {F_IRIS_POLYGONS}...")
    gdf_iris = gpd.read_file(F_IRIS_POLYGONS)
    gdf_iris['CODE_IRIS'] = gdf_iris['CODE_IRIS'].astype(str).str.zfill(9)
    gdf_iris = gdf_iris[['CODE_IRIS', 'geometry']].copy()
    print(f"  {len(gdf_iris)} IRIS, CRS: {gdf_iris.crs}")

    # 3. Mission 1 : BPE
    print("\n--- Mission 1 : BPE ---")
    df_bpe = build_bpe(df_iris_ref)

    # 4. Mission 2+3 : RNA
    print("\n--- Missions 2+3 : RNA ---")
    df_rna = build_rna(gdf_iris, iris_mapping)

    # 5. Merge sur la base IRIS de référence
    print("\nMerge final...")
    df_final = df_iris_ref[['IRIS']].set_index('IRIS').copy()
    df_final = df_final.join(df_bpe, how='left')

    if len(df_rna) > 0:
        df_final = df_final.join(df_rna, how='left')

    # Remplir les NaN des colonnes brutes par 0
    raw_bpe_cols = ['nb_lieux_culte_bpe', 'nb_grande_surface_bpe']
    raw_rna_cols = ['rna_culte_total', 'rna_culte_chretien_total',
                    'rna_culte_catholique', 'rna_culte_evangelique',
                    'rna_culte_protestant', 'rna_culte_orthodoxe',
                    'rna_culte_islam', 'rna_culte_juif', 'rna_culte_bouddhiste',
                    'rna_culte_hindou', 'rna_culte_mormon', 'rna_culte_temoins_jehovah',
                    'rna_solidarite_active', 'rna_sport_distinction']
    all_raw = raw_bpe_cols + raw_rna_cols
    for col in all_raw:
        if col in df_final.columns:
            df_final[col] = df_final[col].fillna(0)

    # 6. Normalisation /1000 habitants
    normalize_pour1000(df_final, pop_map, all_raw)
    # Renommer grande_surface
    if 'nb_grande_surface_bpe_pour1000' in df_final.columns:
        df_final.rename(columns={'nb_grande_surface_bpe_pour1000': 'bpe_grande_surface_pour1000'}, inplace=True)

    # 7. Indicateur de qualité géocodage (taux IRIS attribués via BAN vs fallback)
    # Non calculable après merge, on laisse None (info disponible en cours d'exécution)

    # 8. Sauvegarde
    df_final.reset_index(inplace=True)
    df_final.to_csv(OUT_FILE, index=False)
    print(f"\nSauvegardé : {OUT_FILE}")
    print(f"  {len(df_final)} IRIS x {len(df_final.columns)} colonnes")

    # 9. Vérifications rapides
    print("\n=== VÉRIFICATIONS ===")
    iris_ste = '421560000'  # Saint-Étienne centre (IRIS de référence)
    if iris_ste in df_final['IRIS'].values:
        row = df_final[df_final['IRIS'] == iris_ste].iloc[0]
        print(f"\nSaint-Étienne centre (IRIS {iris_ste}) :")
        for col in all_raw[:6]:
            if col in df_final.columns:
                print(f"  {col:<30} : {row.get(col, 'N/A')}")
    else:
        print(f"  IRIS {iris_ste} non trouvé (normal si hors périmètre du CSV de référence)")

    for col in all_raw:
        if col in df_final.columns:
            nan_pct = df_final[col].isna().mean() * 100
            nonzero = (df_final[col] > 0).sum()
            print(f"  {col:<30} NaN: {nan_pct:.1f}%  IRIS non-nuls: {nonzero}")

    print("\nTerminé.")


if __name__ == '__main__':
    main()
