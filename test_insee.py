import requests, zipfile, io, sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

url = "https://www.insee.fr/fr/statistiques/fichier/2560452/table-appartenance-geo-communes-24.zip"
print("Downloading...")
r = requests.get(url, timeout=60)
print(f"Status: {r.status_code}, size: {len(r.content):,} bytes")

zf = zipfile.ZipFile(io.BytesIO(r.content))
print("Files in ZIP:", zf.namelist())

# Read first file and show columns + first rows
for name in zf.namelist():
    if name.endswith('.csv') or name.endswith('.CSV'):
        import pandas as pd
        df = pd.read_csv(zf.open(name), sep=';', dtype=str, nrows=5)
        print(f"\nFile: {name}")
        print("Columns:", df.columns.tolist())
        print(df.head(3).to_string())
        break
