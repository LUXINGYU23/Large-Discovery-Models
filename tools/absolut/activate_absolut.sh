#!/usr/bin/env bash
export ABSOLUT_PATH=/Aidata/shared/ldm/antibody-poc/env/sources/Absolut
if [[ ! -x "$ABSOLUT_PATH/src/bin/Absolut" ]]; then
  echo "Absolut executable missing under $ABSOLUT_PATH" >&2
  return 1 2>/dev/null || exit 1
fi
