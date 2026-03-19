import csv
files = [
    'iris/meta_BASE_TD_FILO_IRIS_2021_DISP.csv',
    'iris/meta_base-ic-activite-residents-2021.CSV',
    'iris/meta_base-ic-diplomes-formation-2021.CSV',
]
skip = {'', 'IRIS', 'COM', 'TYP_IRIS', 'LAB_IRIS'}
seen = {}
for f in files:
    with open(f, encoding='utf-8-sig') as fh:
        reader = csv.DictReader(fh, delimiter=';')
        for row in reader:
            cod = row.get('COD_VAR','').strip()
            if cod in skip or cod in seen:
                continue
            lib = (row.get('LIB_VAR_LONG','') or row.get('LIB_VAR','')).strip()
            seen[cod] = lib
for k,v in seen.items():
    print(repr(k), ':', repr(v))
