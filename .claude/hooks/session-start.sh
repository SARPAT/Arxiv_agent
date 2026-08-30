#!/bin/bash
set -euo pipefail

# Every fresh clone/session gets Claude Code's default global git identity
# (Claude <noreply@anthropic.com>), which would show up as the commit author
# on GitHub. Force this repo's local identity on every session start so
# commits are always attributed to the repo owner instead.
git config user.name "Saransh Patel"
git config user.email "saranshappy@gmail.com"
