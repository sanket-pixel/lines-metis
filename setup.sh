#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
PYTHON=${PYTHON:-/home/sanket-shah/venvs/torch-cpu/bin/python}
uv pip install --python "$PYTHON" --index-url https://download.pytorch.org/whl/cpu 'torch==2.14.1+cpu'
uv pip install --python "$PYTHON" -r requirements.txt
if [ ! -f weights/torch/deeplsd.pt ]; then
    SOURCE_CHECKPOINT=$(mktemp)
    trap 'rm -f "$SOURCE_CHECKPOINT"' EXIT
    curl -L --fail https://cvg-data.inf.ethz.ch/DeepLSD/deeplsd_md.tar -o "$SOURCE_CHECKPOINT"
    TORCH_SITE=$("$PYTHON" -c 'import site; print(site.getsitepackages()[0])')
    PYTHONPATH="$TORCH_SITE${PYTHONPATH:+:$PYTHONPATH}" uv run --no-project --python "$PYTHON" --with omegaconf==2.4.0 python - "$SOURCE_CHECKPOINT" <<'PY'
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
