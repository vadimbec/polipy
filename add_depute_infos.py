"""Enrichit bloc_bourgeois_heritage_final.csv avec infos de deputes-active."""
import pandas as pd, sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

actifs = pd.read_csv('deputes-active.csv', dtype=str, usecols=[
    'id', 'departementNom', 'departementCode', 'circo', 'age',
    'scoreParticipation', 'scoreLoyaute', 'nombreMandats'
]).rename(columns={'id': 'pa_id'})

df = pd.read_csv('bloc_bourgeois_heritage_final.csv', dtype=str)

# Supprimer les colonnes si déjà présentes
for col in ['departementNom', 'departementCode', 'circo', 'age',
            'scoreParticipation', 'scoreLoyaute', 'nombreMandats']:
    if col in df.columns:
        df = df.drop(columns=[col])

df = df.merge(actifs, on='pa_id', how='left')

# Construire une colonne circo_label : "Yvelines (78), 10e circ."
def circo_label(row):
    dep = str(row.get('departementNom', '') or '')
    code = str(row.get('departementCode', '') or '')
    circ = str(row.get('circo', '') or '')
    if not dep or dep == 'nan': return ''
    try:
        n = int(float(circ))
        suffix = 're' if n == 1 else 'e'
        return f"{dep} ({code}), {n}{suffix} circ."
    except:
        return dep

df['circo_label'] = df.apply(circo_label, axis=1)

df.to_csv('bloc_bourgeois_heritage_final.csv', index=False)
print(f"OK — {len(df)} lignes, nouvelles colonnes : circo_label, age, scoreParticipation, scoreLoyaute")
print(df[['député', 'circo_label', 'age', 'scoreParticipation', 'scoreLoyaute']].head(6).to_string())
