import json, sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

with open('eda_circos.ipynb', encoding='utf-8') as f:
    nb = json.load(f)

code_cells_idx = [i for i, c in enumerate(nb['cells']) if c['cell_type'] == 'code']
cell_idx = code_cells_idx[6]

new_src = (
    "# Convert key raw vars to numeric for the plots\n"
    "for v in ['tx_pauvrete60_diff', 'rpt_D9_D1_diff', 'act_ouv', 'act_emp',\n"
    "          'nivvie_median_diff', 'actdip_BAC5', 'actcho', 'inactret']:\n"
    "    if v in df_ana.columns:\n"
    "        df_ana[v] = pd.to_numeric(df_ana[v], errors='coerce')\n"
    "\n"
    "AXIS_PAIRS = [\n"
    "    {'x': 'score_exploitation', 'y': 'score_domination',\n"
    "     'title': 'Saint Graal marxiste - Domination (Y) x Exploitation subie (X)',\n"
    "     'xlabel': 'Exploitation subie (ouv/precaire)', 'ylabel': 'Domination (riche + diplome)'},\n"
    "    {'x': 'PC1', 'y': 'PC2',\n"
    "     'title': 'ACP - PC1 x PC2 (data-driven)',\n"
    "     'xlabel': 'PC1 (richesse/diplome vs pauvrete)', 'ylabel': 'PC2 (pauvrete vs emploi/age)'},\n"
    "    {'x': 'score_cap_cult', 'y': 'score_cap_eco',\n"
    "     'title': 'Espace bourdieusien - Cap. eco (Y) x Cap. culturel (X)',\n"
    "     'xlabel': 'Capital culturel (diplome + intel.)', 'ylabel': 'Capital economique (revenu + patri.)'},\n"
    "    {'x': 'tx_pauvrete60_diff', 'y': 'rpt_D9_D1_diff',\n"
    "     'title': 'Pauvrete x Inegalite - Tx pauvrete (X) x D9/D1 (Y)',\n"
    "     'xlabel': 'Taux de pauvrete (%)', 'ylabel': 'Inegalite D9/D1'},\n"
    "    {'x': 'act_ouv', 'y': 'nivvie_median_diff',\n"
    "     'title': 'Simple et lisible - Revenu median (Y) x Part ouvriers (X)',\n"
    "     'xlabel': 'Part ouvriers (%)', 'ylabel': 'Revenu median (euros)'},\n"
    "    {'x': 'actdip_BAC5', 'y': 'nivvie_median_diff',\n"
    "     'title': 'Diplome x Richesse - Revenu median (Y) x Part BAC+5 (X)',\n"
    "     'xlabel': 'Part BAC+5 (%)', 'ylabel': 'Revenu median (euros)'},\n"
    "    {'x': 'actcho', 'y': 'inactret',\n"
    "     'title': 'Structure demo - Retraites (Y) x Chomage (X)',\n"
    "     'xlabel': 'Taux de chomage (%)', 'ylabel': 'Part retraites (%)'},\n"
    "]\n"
    "\n"
    "AXIS_PAIRS = [p for p in AXIS_PAIRS if p['x'] in df_ana.columns and p['y'] in df_ana.columns]\n"
    "print(f'{len(AXIS_PAIRS)} paires axes disponibles')\n"
    "\n"
    "n_pairs = len(AXIS_PAIRS)\n"
    "ncols = 3\n"
    "nrows = (n_pairs + ncols - 1) // ncols\n"
    "\n"
    "fig, axes = plt.subplots(nrows, ncols, figsize=(18, nrows * 5.5))\n"
    "axes_flat = list(axes.flatten())\n"
    "\n"
    "for i, pair in enumerate(AXIS_PAIRS):\n"
    "    ax = axes_flat[i]\n"
    "    x_data = pd.to_numeric(df_ana[pair['x']], errors='coerce')\n"
    "    y_data = pd.to_numeric(df_ana[pair['y']], errors='coerce')\n"
    "    ax.scatter(x_data, y_data, c=df_ana['color'], alpha=0.65, s=25,\n"
    "               edgecolors='white', lw=0.3)\n"
    "    ax.set_title(pair['title'], fontsize=9)\n"
    "    ax.set_xlabel(pair['xlabel'], fontsize=8)\n"
    "    ax.set_ylabel(pair['ylabel'], fontsize=8)\n"
    "    ax.tick_params(labelsize=7)\n"
    "\n"
    "for i in range(n_pairs, nrows * ncols):\n"
    "    axes_flat[i].set_visible(False)\n"
    "\n"
    "plt.suptitle(\"Paires d'axes candidates pour la visualisation\", fontsize=13)\n"
    "plt.tight_layout()\n"
    "plt.savefig('eda_axis_pairs.png', dpi=120)\n"
    "plt.show()\n"
    "print('Saved eda_axis_pairs.png')\n"
)

nb['cells'][cell_idx]['source'] = [new_src]
with open('eda_circos.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)
print('Fixed cell 7 OK')
