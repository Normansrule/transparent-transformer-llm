#!/usr/bin/env bash
# One-time personalisation + first commit.
#   ./setup.sh <github-username> [ssh-host]
#   ./setup.sh Normansrule github-normansrule     # if ~/.ssh/config has a Host alias for that account
#   ./setup.sh someuser                           # plain github.com over SSH
set -euo pipefail
USER_NAME="${1:?usage: ./setup.sh <github-username> [ssh-host]}"
SSH_HOST="${2:-github.com}"

grep -rIl --exclude-dir=.git --exclude=setup.sh "YOUR_GITHUB_USERNAME" . | xargs sed -i "s/YOUR_GITHUB_USERNAME/${USER_NAME}/g"
echo "links now point at github.com/${USER_NAME}/transparent-transformer-llm"

[ -d .git ] || git init -q -b main
git add -A
git commit -q -m "transparent-transformer-llm: a language model you can see all the way through" || true
git remote remove origin 2>/dev/null || true
git remote add origin "git@${SSH_HOST}:${USER_NAME}/transparent-transformer-llm.git"

cat <<MSG

Ready. Two steps left:

  1. Create an EMPTY public repository named  transparent-transformer-llm  at https://github.com/new
     (no README, no license, no .gitignore: this folder already has them)

  2. git push -u origin main

Then switch the classroom on (all in the repository's Settings unless noted):
  a. Pages   -> Source "Deploy from a branch", Branch main, Folder /docs -> Save      (the live website)
  b. Actions tab -> enable workflows if asked                                         (tests, homework, ask-the-model)
  c. General -> Features -> tick "Discussions"                                        (optional: a place for questions)
  d. Test the robot: Issues -> New issue -> "Ask the model"

The website will appear at https://${USER_NAME}.github.io/transparent-transformer-llm/
MSG
