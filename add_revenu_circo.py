"""
Enrichit bloc_bourgeois_heritage_final.csv avec revenu_circonscription.
Méthode : moyenne pondérée par population des revenus médians communaux (FILO2021)
par circonscription (via circo_composition.csv).
"""
import pandas as pd
import sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

# --- 1. Charger FILO ---
filo = pd.read_csv(
    'FILO2021_DEC_COM.csv', sep=';', dtype=str,
    usecols=['CODGEO', 'Q221', 'NBPERS21']
)
filo['Q221'] = pd.to_numeric(filo['Q221'], errors='coerce')
filo['NBPERS21'] = pd.to_numeric(filo['NBPERS21'], errors='coerce')
filo = filo.dropna(subset=['Q221', 'NBPERS21'])
print(f"FILO: {len(filo)} communes avec données valides")

# --- 2. Charger circo_composition ---
circo = pd.read_csv(
    'circo_composition.csv', sep=';', dtype=str,
    usecols=['COMMUNE_RESID', 'circo']
)
circo = circo.rename(columns={'COMMUNE_RESID': 'CODGEO', 'circo': 'CIRLEG'})
print(f"circo_composition: {len(circo)} lignes")

# --- 3. Jointure FILO × circo ---
merged = circo.merge(filo, on='CODGEO', how='inner')
print(f"Communes matchées: {merged['CODGEO'].nunique()} communes uniques sur {len(merged)} lignes")

# Calcul revenu moyen pondéré par circo
merged['rev_pond'] = merged['Q221'] * merged['NBPERS21']
agg = merged.groupby('CIRLEG').agg(
    rev_total=('rev_pond', 'sum'),
    pop_total=('NBPERS21', 'sum')
).reset_index()
agg['revenu_circonscription'] = (agg['rev_total'] / agg['pop_total']).round(0).astype(int)
print(f"Circonscriptions calculées: {len(agg)}")
print(f"Distribution: min={agg['revenu_circonscription'].min()}, "
      f"médiane={agg['revenu_circonscription'].median():.0f}, "
      f"max={agg['revenu_circonscription'].max()}")

# --- 4. Construire le code circo de chaque député ---
actifs = pd.read_csv('deputes-active.csv', dtype=str,
                     usecols=['id', 'departementCode', 'circo'])

def build_cirleg(row):
    dep = str(row['departementCode']).strip()
    cir = str(row['circo']).strip()
    try:
        cir_int = int(float(cir))
    except (ValueError, TypeError):
        return None
    if len(dep) == 3:  # DOM
        return dep + str(cir_int).zfill(2)
    else:  # Métropole
        return dep.zfill(2) + str(cir_int).zfill(3)

actifs['CIRLEG'] = actifs.apply(build_cirleg, axis=1)
actifs = actifs.rename(columns={'id': 'pa_id'})
print(f"\ndeputes-active: {len(actifs)} députés, "
      f"{actifs['CIRLEG'].notna().sum()} avec code CIRLEG valide")
print("Exemples:", actifs[['pa_id', 'departementCode', 'circo', 'CIRLEG']].head(5).to_string())

# --- 5. Jointure sur le CSV final ---
df = pd.read_csv('bloc_bourgeois_heritage_final.csv', dtype=str)
print(f"\nbloc_bourgeois: {len(df)} lignes")

# Supprimer colonne existante si présente
if 'revenu_circonscription' in df.columns:
    df = df.drop(columns=['revenu_circonscription'])
if 'CIRLEG' in df.columns:
    df = df.drop(columns=['CIRLEG'])

# Joindre circo code
df = df.merge(actifs[['pa_id', 'CIRLEG']], on='pa_id', how='left')
n_cirleg = df['CIRLEG'].notna().sum()
print(f"Députés avec code CIRLEG: {n_cirleg}/{len(df)}")

# Joindre revenu
df = df.merge(agg[['CIRLEG', 'revenu_circonscription']], on='CIRLEG', how='left')
n_rev = df['revenu_circonscription'].notna().sum()
print(f"Députés avec revenu_circonscription: {n_rev}/{len(df)}")

# Supprimer colonne CIRLEG intermédiaire (optionnel - garder pour debug)
# df = df.drop(columns=['CIRLEG'])

df.to_csv('bloc_bourgeois_heritage_final.csv', index=False)
print(f"\n✓ Sauvegardé bloc_bourgeois_heritage_final.csv ({len(df)} lignes)")

# Vérification manuelle
check = df[['député', 'departementCode' if 'departementCode' in df.columns else 'CIRLEG',
            'CIRLEG', 'revenu_circonscription']].dropna(subset=['revenu_circonscription'])
print("\nQuelques exemples:")
print(check.head(10).to_string())
