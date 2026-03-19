import os
import xml.etree.ElementTree as ET
import csv
from datetime import datetime

# --- CONFIGURATION DES CHEMINS ---
DATA_DIR = "data" # On scanne tout le dossier data globalement
XML_ACTEURS = os.path.join("xml", "acteur")
DATE_REF_HISTO = datetime(2024, 6, 9)

def strip_ns(root):
    for elem in root.iter():
        if '}' in elem.tag:
            elem.tag = elem.tag.split('}', 1)[1]

def parse_date(date_str):
    if not date_str or date_str in ["None", ""]: return None
    try: return datetime.strptime(date_str, "%Y-%m-%d")
    except: return None

print("🚀 Lancement de l'analyse infaillible...")

# 1. SCAN GLOBAL DE TOUT LE DOSSIER 'DATA'
fichiers_pa = []
fichiers_pm = []
fichiers_po = []

for root_dir, dirs, files in os.walk(DATA_DIR):
    for f in files:
        if f.startswith("PA") and f.endswith(".xml"): fichiers_pa.append(os.path.join(root_dir, f))
        elif f.startswith("PM") and f.endswith(".xml"): fichiers_pm.append(os.path.join(root_dir, f))
        elif f.startswith("PO") and f.endswith(".xml"): fichiers_po.append(os.path.join(root_dir, f))

print(f"📊 Fichiers trouvés dans '{DATA_DIR}': {len(fichiers_pa)} Acteurs, {len(fichiers_pm)} Mandats, {len(fichiers_po)} Organes.")

if len(fichiers_pm) == 0:
    print("❌ ERREUR CRITIQUE : Aucun fichier mandat (PM...) trouvé dans 'data'. C'est pour ça que tout le monde est NI !")
    exit()

# 2. CARTOGRAPHIE DES ORGANES (PO -> NOM DU PARTI)
po_to_label = {}
for path in fichiers_po:
    try:
        tree = ET.parse(path)
        root = tree.getroot(); strip_ns(root)
        if root.findtext("codeType") == "GP":
            uid = root.findtext("uid")
            label = root.findtext("libelleAbrege") or root.findtext("libelle")
            po_to_label[uid] = label
    except: continue

print(f"✅ {len(po_to_label)} Groupes Politiques décodés.")

# 3. LIAISON DES MANDATS (PA -> PO)
pa_to_group = {}
for path in fichiers_pm:
    try:
        tree = ET.parse(path)
        root = tree.getroot(); strip_ns(root)
        
        # On cible les mandats de Groupe Politique (GP)
        if root.findtext("typeOrgane") == "GP":
            d_fin = root.findtext("dateFin")
            
            # Condition robuste pour vérifier si le mandat est EN COURS
            # Si d_fin est vide (""), None, ou "None", le mandat est actif
            if not d_fin or d_fin == "None" or str(d_fin).strip() == "":
                pa_ref = root.findtext("acteurRef")
                po_ref = root.find(".//organeRef").text if root.find(".//organeRef") is not None else None
                
                if pa_ref and po_ref in po_to_label:
                    pa_to_group[pa_ref] = po_to_label[po_ref]
    except: continue

print(f"✅ {len(pa_to_group)} députés reliés à un parti politique.")

# 4. CALCUL ANCIENNETÉ ET GÉNÉRATION DU CSV
def get_historical_seniority(pa_id):
    path = os.path.join(XML_ACTEURS, f"{pa_id}.xml")
    if not os.path.exists(path): return None
    try:
        tree = ET.parse(path)
        root = tree.getroot(); strip_ns(root)
        jours = 0
        for m in root.findall(".//mandat"):
            if m.findtext("typeOrgane") in ["ASSEMBLEE", "SENAT", "GOUVERNEMENT", "MINISTERE"]:
                d1 = parse_date(m.findtext("dateDebut"))
                d2 = parse_date(m.findtext("dateFin"))
                if d1:
                    fin = d2 if d2 else DATE_REF_HISTO
                    jours += (fin - d1).days
        return round(max(0, jours) / 365.25, 1)
    except: return None

resultats = []
for path in fichiers_pa:
    try:
        pa_id = os.path.basename(path).replace(".xml", "")
        tree = ET.parse(path)
        root = tree.getroot(); strip_ns(root)
        
        ident = root.find(".//ident")
        nom = f"{ident.findtext('prenom')} {ident.findtext('nom')}"
        
        groupe = pa_to_group.get(pa_id, "NI")
        anciennete = get_historical_seniority(pa_id)
        
        resultats.append({
            "député": nom,
            "anciennete": anciennete if anciennete is not None else "NaN",
            "groupe politique": groupe
        })
    except: continue

with open("anciennete_deputes_definitive.csv", "w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["député", "anciennete", "groupe politique"])
    writer.writeheader()
    writer.writerows(resultats)

print(f"🎉 Analyse terminée. Fichier généré : 'anciennete_deputes_definitive.csv' ({len(resultats)} députés)")