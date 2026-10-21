#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
if [[ "${1:-}" != "" && "${1:-}" != "--viz-only" ]]; then
    echo 'Usage: ./setup.sh [--viz-only]' >&2; exit 2
fi
PYTHON=${PYTHON:-.venv/bin/python}
if [ "$PYTHON" = .venv/bin/python ] && [ ! -x "$PYTHON" ]; then
    uv venv --python 3.11 .venv
fi
if [[ "${1:-}" == "--viz-only" ]]; then
    uv pip install --python "$PYTHON" 'numpy==2.4.6' 'opencv-python==5.0.0.93' 'Pillow==12.3.0'
    exit 0
fi
uv pip install --python "$PYTHON" --index-url https://download.pytorch.org/whl/cpu 'torch==2.14.1+cpu'
uv pip install --python "$PYTHON" --extra-index-url https://download.pytorch.org/whl/cpu --index-strategy unsafe-best-match -r requirements.txt
if [ ! -f weights/torch/deeplsd.pt ]; then
    SOURCE_CHECKPOINT=$(mktemp)
    trap 'rm -f "$SOURCE_CHECKPOINT"' EXIT
    curl -L --fail https://cvg-data.inf.ethz.ch/DeepLSD/deeplsd_md.tar -o "$SOURCE_CHECKPOINT"
    "$PYTHON" - "$SOURCE_CHECKPOINT" <<'PY'
import hashlib, json, pathlib, sys, torch
source = pathlib.Path(sys.argv[1])
expected = '56e9bae263977caa289ae49802987a67d4f00f376ae7e542176c8ae0a6cc2083'
assert hashlib.sha256(source.read_bytes()).hexdigest() == expected
state = torch.load(source, map_location='cpu', weights_only=False)['model']
folder = pathlib.Path('weights/torch'); folder.mkdir(parents=True, exist_ok=True)
path = folder / 'deeplsd.pt'
torch.save(dict(state), path)
restored = torch.load(path, map_location='cpu', weights_only=True)
assert all(torch.equal(t, restored[k]) for k, t in state.items())
(folder / 'provenance.json').write_text(json.dumps({
    'source_url': 'https://cvg-data.inf.ethz.ch/DeepLSD/deeplsd_md.tar',
    'source_sha256': expected, 'weights_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
    'license': 'MIT', 'tensor_only_conversion_verified': True,
    'upstream_commit': 'f7d9d6258c0cd25d4f6eea882853565403d289be'}, indent=2) + '\n')
PY
fi
