"""Update vizu.ipynb to add circo_label, age, scoreParticipation, scoreLoyaute to info cards."""
import json, sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

with open('vizu.ipynb', encoding='utf-8') as f:
    nb = json.load(f)

src = ''.join(nb['cells'][0]['source'])

# ── 1. Desktop customdata (indices 0-6 → 0-10) ───────────────────────────────
src = src.replace(
    "customdata=sub[['député','libelle_pcs','rev','score_y','anciennete','ville_naissance','pays_naissance']].values,",
    "customdata=sub[['député','libelle_pcs','rev','score_y','anciennete','ville_naissance','pays_naissance','circo_label','age','scoreParticipation','scoreLoyaute']].values,"
)

# ── 2. Desktop hovertemplate: add circo, age, scores ─────────────────────────
src = src.replace(
    '"<br><span style=\'color:#888\'>Né·e à :</span> %{customdata[5]}"'
    '"<br><span style=\'color:#888\'>Revenu commune naissance :</span> <b>%{customdata[2]:,.0f} €</b>"'
    '"<br><span style=\'color:#888\'>Score domination :</span> <b>%{customdata[3]:.1f} / 10</b>"'
    '"<br><span style=\'color:#888\'>Ancienneté :</span> %{customdata[4]:.1f} ans<extra></extra>")',
    '"<br><span style=\'color:#888\'>Circonscription :</span> <b>%{customdata[7]}</b>"'
    '"<br><span style=\'color:#888\'>Né·e à :</span> %{customdata[5]}"'
    '"<br><span style=\'color:#888\'>Âge :</span> %{customdata[8]} ans"'
    '"<br><span style=\'color:#888\'>Revenu circo :</span> <b>%{customdata[2]:,.0f} €</b>"'
    '"<br><span style=\'color:#888\'>Score domination :</span> <b>%{customdata[3]:.1f} / 10</b>"'
    '"<br><span style=\'color:#888\'>Participation :</span> %{customdata[9]:.0%}  ·  <span style=\'color:#888\'>Loyauté :</span> %{customdata[10]:.0%}"'
    '"<br><span style=\'color:#888\'>Ancienneté :</span> %{customdata[4]:.1f} ans<extra></extra>")'
)

# ── 3. Mobile customdata ──────────────────────────────────────────────────────
src = src.replace(
    "customdata=sub[['député','libelle_pcs','rev','score_y',\n                             'anciennete','ville_naissance','pays_naissance']].values.tolist(),",
    "customdata=sub[['député','libelle_pcs','rev','score_y',\n                             'anciennete','ville_naissance','pays_naissance',\n                             'circo_label','age','scoreParticipation','scoreLoyaute']].values.tolist(),"
)

# ── 4. Mobile info card HTML: add new rows ────────────────────────────────────
src = src.replace(
    '  <div class="row"><span class="lbl">Né·e à :</span> <span id="cardVille"></span></div>\n'
    '  <div class="row"><span class="lbl">Revenu circo :</span> <b id="cardRev"></b></div>\n'
    '  <div class="row"><span class="lbl">Score domination :</span> <b id="cardScore"></b></div>\n'
    '  <div class="row"><span class="lbl">Ancienneté :</span> <span id="cardAnc"></span></div>',
    '  <div class="row"><span class="lbl">Circonscription :</span> <b id="cardCirco"></b></div>\n'
    '  <div class="row"><span class="lbl">Né·e à :</span> <span id="cardVille"></span></div>\n'
    '  <div class="row"><span class="lbl">Âge :</span> <span id="cardAge"></span></div>\n'
    '  <div class="row"><span class="lbl">Revenu circo :</span> <b id="cardRev"></b></div>\n'
    '  <div class="row"><span class="lbl">Score domination :</span> <b id="cardScore"></b></div>\n'
    '  <div class="row"><span class="lbl">Participation :</span> <b id="cardPartic"></b>&ensp;·&ensp;<span class="lbl">Loyauté :</span> <b id="cardLoyaute"></b></div>\n'
    '  <div class="row"><span class="lbl">Ancienneté :</span> <span id="cardAnc"></span></div>'
)

# ── 5. Mobile showCard JS: populate new fields ────────────────────────────────
src = src.replace(
    "  document.getElementById('cardVille').textContent = cd[5] || '';\n"
    "  document.getElementById('cardRev').textContent = Number(cd[2]).toLocaleString('fr-FR') + ' €';\n"
    "  document.getElementById('cardScore').textContent = Number(cd[3]).toFixed(1) + ' / 10';\n"
    "  document.getElementById('cardAnc').textContent = Number(cd[4]).toFixed(1) + ' ans';",
    "  document.getElementById('cardCirco').textContent = cd[7] || '';\n"
    "  document.getElementById('cardVille').textContent = (cd[5] || '') + (cd[6] && cd[6] !== 'France' ? ' (' + cd[6] + ')' : '');\n"
    "  document.getElementById('cardAge').textContent = cd[8] ? cd[8] + ' ans' : '';\n"
    "  document.getElementById('cardRev').textContent = cd[2] ? Number(cd[2]).toLocaleString('fr-FR') + ' €' : '—';\n"
    "  document.getElementById('cardScore').textContent = Number(cd[3]).toFixed(1) + ' / 10';\n"
    "  const partic = cd[9] != null ? Math.round(Number(cd[9]) * 100) + '%' : '—';\n"
    "  const loyal  = cd[10] != null ? Math.round(Number(cd[10]) * 100) + '%' : '—';\n"
    "  document.getElementById('cardPartic').textContent = partic;\n"
    "  document.getElementById('cardLoyaute').textContent = loyal;\n"
    "  document.getElementById('cardAnc').textContent = Number(cd[4]).toFixed(1) + ' ans';"
)

nb['cells'][0]['source'] = [src]

with open('vizu.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print("OK - vizu.ipynb updated")
