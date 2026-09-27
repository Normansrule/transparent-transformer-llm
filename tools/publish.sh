#!/usr/bin/env bash
# Test, commit and push in one go:   ./tools/publish.sh "what changed"
set -euo pipefail
cd "$(dirname "$0")/.."
python -m pytest -q
python classroom/check.py --solutions > /dev/null && echo "reference solutions: 10/10"
git add -A
git commit -m "${1:-update}" || { echo "nothing to commit"; exit 0; }
git push
echo "pushed. Pages rebuilds in about a minute: https://normansrule.github.io/transparent-transformer-llm/"
