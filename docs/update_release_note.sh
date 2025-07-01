#!/bin/bash
set -e
if [ -n "$ZSH_VERSION" ]; then
  REPO_DIR=$(cd $(dirname ${(%):-%N})/..; pwd)
elif [ -n "$BASH_VERSION" ]; then
  REPO_DIR=$(cd $(dirname ${BASH_SOURCE[0]})/..; pwd)
fi

TMP_FILE=tmp.md
rm -rf $TMP_FILE || true
cat ${REPO_DIR}/docs/RELEASE_CURRENT.md > $TMP_FILE
echo >> $TMP_FILE
echo "---" >> $TMP_FILE
echo >> $TMP_FILE
cat ${REPO_DIR}/docs/RELEASE.md >> $TMP_FILE
cat $TMP_FILE > ${REPO_DIR}/docs/RELEASE.md
rm -rf $TMP_FILE
