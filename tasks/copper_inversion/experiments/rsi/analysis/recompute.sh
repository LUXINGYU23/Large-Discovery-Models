#!/bin/sh
# Recompute a results bundle from a read-only archive and compare it byte for byte.
#
#   recompute.sh ARCHIVE OUTCOMES BUNDLE [OUT]
#
# ARCHIVE   directory with <exp>/<run_id>/ public run directories (may be mounted read-only)
# OUTCOMES  the outcome export (rsi-outcomes-v1) named in BUNDLE/provenance.json
# BUNDLE    the results bundle to check (this script's own directory, normally)
# OUT       new directory for the recomputed bundle (default: a fresh temporary directory)
#
# REPO selects the code checkout (default: code/<code_commit> on H20); PYTHON the interpreter.
set -eu
ARCHIVE=${1:?archive directory}
OUTCOMES=${2:?outcome export}
BUNDLE=${3:?results bundle}
OUT=${4:-$(mktemp -d)/results}
PYTHON=${PYTHON:-python3}
COMMIT=$("$PYTHON" -c 'import json,sys; print(json.load(open(sys.argv[1]))["code_commit"])' "$BUNDLE/provenance.json")
REPO=${REPO:-/Aidata/shared/xingyulu/rsi4geo/code/$COMMIT}
cd "$REPO"
"$PYTHON" -B -m tasks.copper_inversion.experiments.rsi.analysis.bundle \
    --archive "$ARCHIVE" --outcomes "$OUTCOMES" --out "$OUT" \
    --config "$BUNDLE/analysis-config.json" --code-commit "$COMMIT"
diff -r "$BUNDLE" "$OUT"
echo "identical: $OUT"
