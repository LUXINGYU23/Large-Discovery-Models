#!/bin/sh
set -eu

mkdir -p /workspace/research
python - <<'PY'
from pypdf import PdfReader, PdfWriter

path = "/workspace/research/smoke.pdf"
writer = PdfWriter()
writer.add_blank_page(width=72, height=72)
with open(path, "wb") as stream:
    writer.write(stream)
assert len(PdfReader(path).pages) == 1
PY
pdftotext /workspace/research/smoke.pdf - >/dev/null
printf 'pdf_and_cli_ok\n' > /workspace/research/proof.txt
test -s /workspace/research/proof.txt
