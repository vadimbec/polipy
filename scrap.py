import requests
from bs4 import BeautifulSoup
import json
import unicodedata

def normalize_name(name):
    """Normalise le nom pour qu'il matche parfaitement avec le script précédent."""
    # Enlever les civilités
    name = name.replace("M. ", "").replace("Mme ", "")
    # Enlever les accents et mettre en minuscules
    name = unicodedata.normalize('NFKD', str(name)).encode('ASCII', 'ignore').decode('utf-8')
    return name.lower().replace("-", " ").strip()

print("Connexion au site de l'Assemblée nationale...")
url = "https://www2.assemblee-nationale.fr/deputes/liste/alphabetique"

try:
    response = requests.get(url)
    response.raise_for_status() # Vérifie que la page a bien chargé
except requests.exceptions.RequestException as e:
    print(f"Erreur de connexion : {e}")
    exit()

print("Analyse de la page HTML...")
soup = BeautifulSoup(response.content, 'html.parser')

# Le site de l'Assemblée liste les députés dans des balises <a>
# pointant vers leurs fiches individuelles (/deputes/fiche/OMC_PA...)
deputes_actuels = []
noms_vus = set()

# On cherche tous les liens de la page
for link in soup.find_all('a', href=True):
    href = link['href']
    # On isole les liens qui mènent vers une fiche de député
    if '/deputes/fiche/OMC_PA' in href:
        nom_brut = link.text.strip()
        nom_normalise = normalize_name(nom_brut)
        
        # On évite les doublons et les liens vides
        if nom_normalise and nom_normalise not in noms_vus:
            noms_vus.add(nom_normalise)
            deputes_actuels.append({
                "nom_officiel": nom_brut,
                "nom_normalise": nom_normalise,
                "url": f"https://www2.assemblee-nationale.fr{href}"
            })

print(f"Extraction terminée : {len(deputes_actuels)} députés trouvés ! (Le nombre doit être proche de 577)")

# Sauvegarde au format JSON pour notre étape suivante
with open("deputes_actuels.json", "w", encoding="utf-8") as f:
    json.dump(deputes_actuels, f, ensure_ascii=False, indent=4)

print("✅ Les données sont sauvegardées dans 'deputes_actuels.json'")