#!/usr/bin/env bash
set -exo

TARGET_BRANCH_NAME=$1
echo "Target Branch: ${TARGET_BRANCH_NAME}"

PROJ_DIR="$( cd "$( dirname ${BASH_SOURCE[0]} )/../../.." &> /dev/null && pwd )"
cd $PROJ_DIR
source ${PROJ_DIR}/scripts/SourceMe
function lint_check() {
  # fetch the HEAD of origin target branch.
  git fetch --depth=1 origin ${TARGET_BRANCH_NAME}

  pre-commit run --files $(git diff --diff-filter=ACMR --name-only origin/${TARGET_BRANCH_NAME})

}

lint_check
echo "Lint Check Succeed!"
