#!/bin/bash
set -e
PYTHON="C:/Users/vbecquet/AppData/Local/miniconda3/envs/vadim_env/python.exe"

echo "=== Régénération des HTML + JSON ==="
$PYTHON fast_server/rebuild_vizu_iris.py
$PYTHON fast_server/rebuild_vizu_iris_mobile.py

echo "=== Déploiement sur gh-pages ==="
git checkout gh-pages
cp saint_graphique_iris.html carte.html
cp saint_graphique_iris_mobile.html carte-mobile.html
git add carte.html carte-mobile.html data/
git commit -m "Deploy $(date '+%Y-%m-%d %H:%M')"
git push origin gh-pages

echo "=== Retour sur main ==="
git checkout main
echo "Done. https://vadimbec.github.io/polipy/"
