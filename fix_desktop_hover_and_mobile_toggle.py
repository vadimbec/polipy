"""
1. Fix desktop hovertemplate (add circo, age, participation, loyauté)
2. Add select-all / deselect-all toggle button on mobile
"""
import json, sys, re
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

with open('vizu.ipynb', encoding='utf-8') as f:
    nb = json.load(f)

src = ''.join(nb['cells'][0]['source'])

# ════════════════════════════════════════════════════════════════════════
# 1. Desktop hovertemplate
# ════════════════════════════════════════════════════════════════════════
OLD_HOVER = (
    "\"<br><span style='color:#888'>Profession :</span> <b>%{customdata[1]}</b>\"\n"
    "                \"<br><span style='color:#888'>Né·e à :</span> %{customdata[5]}\"\n"
    "                \"<br><span style='color:#888'>Revenu commune naissance :</span> <b>%{customdata[2]:,.0f} €</b>\"\n"
    "                \"<br><span style='color:#888'>Score domination :</span> <b>%{customdata[3]:.1f} / 10</b>\"\n"
    "                \"<br><span style='color:#888'>Ancienneté :</span> %{customdata[4]:.1f} ans<extra></extra>\")"
)

NEW_HOVER = (
    "\"<br><span style='color:#888'>Profession :</span> <b>%{customdata[1]}</b>\"\n"
    "                \"<br><span style='color:#888'>Circonscription :</span> <b>%{customdata[7]}</b>\"\n"
    "                \"<br><span style='color:#888'>Né·e à :</span> %{customdata[5]}\"\n"
    "                \"<br><span style='color:#888'>Âge :</span> %{customdata[8]} ans\"\n"
    "                \"<br><span style='color:#888'>Revenu circo :</span> <b>%{customdata[2]:,.0f} €</b>\"\n"
    "                \"<br><span style='color:#888'>Score domination :</span> <b>%{customdata[3]:.1f} / 10</b>\"\n"
    "                \"<br><span style='color:#888'>Participation :</span> %{customdata[9]:.0%}&ensp;·&ensp;\"\n"
    "                \"<span style='color:#888'>Loyauté :</span> %{customdata[10]:.0%}\"\n"
    "                \"<br><span style='color:#888'>Ancienneté :</span> %{customdata[4]:.1f} ans<extra></extra>\")"
)

if OLD_HOVER in src:
    src = src.replace(OLD_HOVER, NEW_HOVER)
    print("✓ Desktop hovertemplate updated")
else:
    print("✗ Desktop hovertemplate: pattern NOT found — checking what's there")
    idx = src.find("Revenu commune naissance")
    if idx >= 0:
        print(repr(src[idx-100:idx+200]))

# ════════════════════════════════════════════════════════════════════════
# 2. Mobile: add "Tout / Aucun" toggle button
# ════════════════════════════════════════════════════════════════════════

# CSS: add toggle button style
OLD_CSS_FILTERS = "  .filters {{ display: flex; flex-wrap: wrap; gap: 5px; padding: 4px 10px 8px; justify-content: center; }}"
NEW_CSS_FILTERS = (
    "  .filters {{ display: flex; flex-wrap: wrap; gap: 5px; padding: 4px 10px 4px; justify-content: center; }}\n"
    "  .toggle-all {{ display: block; width: calc(100% - 20px); margin: 0 10px 6px;\n"
    "                 padding: 4px 0; border-radius: 8px; border: 1.5px solid #CCC;\n"
    "                 background: transparent; font-size: 10px; font-weight: 700;\n"
    "                 color: #888; font-family: inherit; cursor: pointer;\n"
    "                 -webkit-tap-highlight-color: transparent; text-align: center; }}"
)

if OLD_CSS_FILTERS in src:
    src = src.replace(OLD_CSS_FILTERS, NEW_CSS_FILTERS)
    print("✓ Mobile CSS: toggle-all button added")
else:
    print("✗ Mobile CSS: pattern not found")

# HTML: add button above filters div
OLD_FILTERS_DIV = '<div class="filters" id="filters"></div>'
NEW_FILTERS_DIV = (
    '<button class="toggle-all" id="toggleAll">Tout décocher</button>\n'
    '<div class="filters" id="filters"></div>'
)

if OLD_FILTERS_DIV in src:
    src = src.replace(OLD_FILTERS_DIV, NEW_FILTERS_DIV)
    print("✓ Mobile HTML: toggle button inserted")
else:
    print("✗ Mobile HTML: filters div not found")

# JS: add toggle-all logic after the filter buttons loop
OLD_TOGGLE_JS = "btns.forEach(b => {{"
NEW_TOGGLE_JS = (
    "// ── Bouton Tout / Aucun ─────────────────────────────────────────────────\n"
    "const toggleAllBtn = document.getElementById('toggleAll');\n"
    "let allOn = true;\n\n"
    "function updateToggleLabel() {{\n"
    "  allOn = activeGroups.size === btns.length;\n"
    "  toggleAllBtn.textContent = allOn ? 'Tout décocher' : 'Tout cocher';\n"
    "}}\n\n"
    "toggleAllBtn.addEventListener('click', function() {{\n"
    "  if (allOn) {{\n"
    "    // Décocher tous\n"
    "    btns.forEach(b => {{\n"
    "      if (activeGroups.has(b.key)) {{\n"
    "        activeGroups.delete(b.key);\n"
    "        const el = filtersDiv.querySelector('[data-key=\"' + b.key + '\"]');\n"
    "        el.classList.replace('on','off');\n"
    "        el.style.backgroundColor = 'transparent';\n"
    "        el.style.color = b.color;\n"
    "        Plotly.restyle(chartDiv, {{ visible: false }}, [b.traceIdx]);\n"
    "      }}\n"
    "    }});\n"
    "  }} else {{\n"
    "    // Cocher tous\n"
    "    btns.forEach(b => {{\n"
    "      if (!activeGroups.has(b.key)) {{\n"
    "        activeGroups.add(b.key);\n"
    "        const el = filtersDiv.querySelector('[data-key=\"' + b.key + '\"]');\n"
    "        el.classList.replace('off','on');\n"
    "        el.style.backgroundColor = b.color;\n"
    "        el.style.color = '#fff';\n"
    "        Plotly.restyle(chartDiv, {{ visible: true }}, [b.traceIdx]);\n"
    "      }}\n"
    "    }});\n"
    "  }}\n"
    "  updateToggleLabel();\n"
    "}});\n\n"
    "btns.forEach(b => {{"
)

if "btns.forEach(b => {{" in src:
    src = src.replace(OLD_TOGGLE_JS, NEW_TOGGLE_JS, 1)
    print("✓ Mobile JS: toggle-all logic added")
else:
    print("✗ Mobile JS: btns.forEach not found")

# Update toggleLabel after individual toggleGroup call
OLD_TOGGLE_GROUP = (
    "  el.addEventListener('click', (e) => {{ e.preventDefault(); toggleGroup(b.key, b.traceIdx, el, b.color); }});"
)
NEW_TOGGLE_GROUP = (
    "  el.addEventListener('click', (e) => {{ e.preventDefault(); toggleGroup(b.key, b.traceIdx, el, b.color); updateToggleLabel(); }});"
)

if OLD_TOGGLE_GROUP in src:
    src = src.replace(OLD_TOGGLE_GROUP, NEW_TOGGLE_GROUP)
    print("✓ Mobile JS: updateToggleLabel hooked into toggleGroup")
else:
    print("✗ Mobile JS: toggleGroup listener not found")

# ════════════════════════════════════════════════════════════════════════
nb['cells'][0]['source'] = [src]
with open('vizu.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)
print("\nDone — vizu.ipynb saved")
