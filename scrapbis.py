import requests
from bs4 import BeautifulSoup
import json
import unicodedata
import re

def normalize_name(name):
    name = name.replace("M. ", "").replace("Mme ", "")
    name = unicodedata.normalize('NFKD', str(name)).encode('ASCII', 'ignore').decode('utf-8')
    return name.lower().replace("-", " ").strip()

print("Connexion à l'Assemblée Nationale (Liste 2024)...")
url = "https://www2.assemblee-nationale.fr/deputes/liste/alphabetique"
response = requests.get(url)
soup = BeautifulSoup(response.content, 'html.parser')

deputes = []
# Le site liste les députés dans des <li> dans le bloc alphabétique
for li in soup.find_all('li'):
    link = li.find('a', href=True)
    if link and '/deputes/fiche/OMC_PA' in link['href']:
        full_text = li.get_text(strip=True)
        # Format habituel : "Nom Prénom (Groupe)"
        match_groupe = re.search(r'\((.*?)\)', full_text)
        groupe = match_groupe.group(1) if match_groupe else "NI"
        
        pa_id = link['href'].split('/')[-1].replace('OMC_', '')
        nom_officiel = link.text.strip()
        
        deputes.append({
            "pa_id": pa_id,
            "nom_officiel": nom_officiel,
            "nom_normalise": normalize_name(nom_officiel),
            "groupe_actuel": groupe,
            "url": f"https://www2.assemblee-nationale.fr{link['href']}"
        })

with open("deputes_actuels.json", "w", encoding="utf-8") as f:
    json.dump(deputes, f, indent=4, ensure_ascii=False)

print(f"✅ {len(deputes)} députés de la 17ème législature sauvegardés avec leurs groupes.")