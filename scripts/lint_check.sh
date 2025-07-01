#!/usr/bin/env bash
set -exo

TARGET_REF=$1
echo "Target Ref: ${TARGET_REF}"

PROJ_DIR="$( cd "$( dirname ${BASH_SOURCE[0]} )/../" &> /dev/null && pwd )"
cd $PROJ_DIR

source ${PROJ_DIR}/scripts/install_githooks.sh

# pre-commit run --files $(git diff --diff-filter=ACMR --name-only ${TARGET_REF})
pre-commit run --from-ref ${TARGET_REF} --to-ref HEAD

echo "Lint Check Succeed!"
