#!/usr/bin/env bash
set -euo pipefail
root=${1:?Usage: bash install.sh /absolute/install/directory}
here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
commit=beb1821568f3ba5b7861255ab19172bbc65a7480
if [[ -e "$root" ]]; then echo "Destination must not exist: $root" >&2; exit 1; fi
git clone --filter=blob:none --no-checkout https://github.com/csi-greifflab/Absolut.git "$root"
git -C "$root" checkout --detach "$commit"
for patchfile in "$here"/patches/*.patch; do git -C "$root" apply "$patchfile"; done
make -C "$root/src" -o Makefile -j4 LFLAGS= AbsolutNoLib
mkdir -p "$root/src/bin" "$root/src/build-headless"
ln -s ../AbsolutNoLib "$root/src/bin/Absolut"
cp "$here/Makefile.headless" "$root/src/build-headless/Makefile"
make -C "$root/src/build-headless" -j4
printf "export ABSOLUT_PATH=%q\n" "$root" > "$root/activate.sh"
