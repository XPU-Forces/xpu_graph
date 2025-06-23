#!/bin/bash

# Check inputs
if [ "$#" -lt 2 ]; then
    echo "USAGE $0 <TARGET_REF> <BACKENDS>"
    echo "DEMO: $0 origin/master xpu1 xpu2"
    exit 1
fi

TARGET_REF="$1"
shift # remove the first argument
BACKENDS=$@
echo "CHECKING updates from target: $TARGET_REF"

CHANGED_FILES=$(git diff --name-only "$TARGET_REF")

if [ -z "$CHANGED_FILES" ]; then
    echo "NO files changed."
    exit 0
fi

IGNORED_PATTERNS=(
    '^doc/.*'
    '^docs/.*'
    '^README.md'
    '^.github/ISSUE_TEMPLATE/.*'
)

BACKEND_SPECIFIC_PATTERNS=(
    '^\.github/workflows/ci-for-([a-zA-Z0-9]+)\.yml'
    '^xpu_graph/backends/([a-zA-Z0-9]+)\.py'
    '^xpu_graph/passes/patterns/targets/([a-zA-Z0-9]+)/.*'
)

# CHECK all updated files
updated_files=()
for file in $CHANGED_FILES; do
    # Check if the updated file should be ignored
    current_matched=0
    for pattern in "${IGNORED_PATTERNS[@]}"; do
        if [[ "$file" =~ $pattern ]]; then
            # echo "DEBUG: ignore common file: $file"
            current_matched=1
            break
        fi
    done
    if [[ $current_matched -eq 1 ]]; then
        continue
    fi
    for pattern in "${BACKEND_SPECIFIC_PATTERNS[@]}"; do
        if [[ "$file" =~ $pattern ]]; then
            backend=${BASH_REMATCH[1]}
            for b in $BACKENDS; do
                if [[ $backend == $b ]]; then
                    # echo "DEBUG: updated $backend file: $file"
                    updated_files+=("$file")
                    current_matched=1
                    break
                fi
            done
            if [[ $current_matched -ne 1 ]]; then
                # echo "DEBUG: ignore other backend file: $file"
                current_matched=1
            fi
            break
        fi
    done
    if [[ $current_matched -ne 1 ]]; then
        # echo "DEBUG: updated common file: $file"
        updated_files+=("$file")
    fi
done

if [ ${#updated_files[@]} -ne 0 ]; then
    echo "FILES updated for backends: $BACKENDS"
    echo "updated files:"
    for file in "${updated_files[@]}"; do
        echo "- $file"
    done
    exit 1
else
    echo "NO files updated for backends: $BACKENDS"
    echo "changed files:"
    for file in "${CHANGED_FILES[@]}"; do
        echo "- $file"
    done
    exit 0
fi
