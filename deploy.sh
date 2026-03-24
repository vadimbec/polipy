#!/bin/bash
# Deploy vers gh-pages sans jamais quitter main.
# Utilise git worktree : gh-pages est checké dans un dossier temporaire séparé.
set -e

REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"
WORKTREE_DIR="$REPO_ROOT/.gh-pages-deploy"

echo "=== Vérification des fichiers à déployer ==="
for f in \
  "$REPO_ROOT/saint_graphique_iris.html" \
  "$REPO_ROOT/saint_graphique_iris_mobile.html" \
  "$REPO_ROOT/data/static.json" \
  "$REPO_ROOT/data/geo.json"; do
  if [ ! -f "$f" ]; then
    echo "ERREUR : fichier manquant : $f"
    echo "Lance d'abord rebuild_vizu_iris.py et rebuild_vizu_iris_mobile.py"
    exit 1
  fi
done

echo "=== Préparation du worktree gh-pages ==="
# Supprimer l'ancien worktree s'il existe (nettoyage propre)
if [ -d "$WORKTREE_DIR" ]; then
  git worktree remove --force "$WORKTREE_DIR" 2>/dev/null || true
fi
# Synchroniser gh-pages local avec remote avant de créer le worktree
git fetch origin gh-pages
git branch -f gh-pages origin/gh-pages
git worktree add "$WORKTREE_DIR" gh-pages

echo "=== Copie des fichiers ==="
cp "$REPO_ROOT/saint_graphique_iris.html"        "$WORKTREE_DIR/carte.html"
cp "$REPO_ROOT/saint_graphique_iris_mobile.html" "$WORKTREE_DIR/carte-mobile.html"
# Copier tout data/ (json générés)
mkdir -p "$WORKTREE_DIR/data"
cp "$REPO_ROOT/data/"*.json "$WORKTREE_DIR/data/"

echo "=== Commit et push ==="
cd "$WORKTREE_DIR"
git add carte.html carte-mobile.html data/
# Committer seulement s'il y a des changements
if git diff --cached --quiet; then
  echo "Aucun changement à déployer."
else
  git commit -m "Deploy $(date '+%Y-%m-%d %H:%M')"
  git push origin gh-pages
  echo "=== Push terminé ==="
fi

echo "=== Nettoyage du worktree ==="
cd "$REPO_ROOT"
git worktree remove --force "$WORKTREE_DIR"

echo "Done — stayed on main."
echo "https://vadimbec.github.io/polipy/"
