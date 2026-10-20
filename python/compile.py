"""Calibrate with real TUM images and compile the exported CNN for one Metis core."""
import argparse
import json
from pathlib import Path
import shutil
import tempfile
import cv2
import numpy as np
import onnx
from common import ROOT, images, preprocess, sha256

SETTINGS = dict(model_name='deeplsd', remove_output_dir=False, save_error_artifact=True,
    aipu_cores_used=1, resources_used=0.25, quantization_scheme='per_tensor_min_max',
    pipeline_spatial_tiles=True, pipeline_channel_tiles=True, inter_operator_async=True,
    use_hw_tokens=True, double_buffer=False, enable_buffer_promotion=True,
    enable_icr=True, enable_swicr=True, dma_dual_channel=True, run_graph_cleaner=False,
    dpu_allocation_algorithm='lazy', tiling_depth=1)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--seq', default='freiburg2_pioneer_slam')
    parser.add_argument('--images', type=int, default=10)
    args = parser.parse_args()
    available = images(args.seq, frames=None)
    if not 1 <= args.images <= len(available):
        parser.error('Calibration count must be between 1 and the sequence length')
    selected = [available[i] for i in np.linspace(0, len(available) - 1, args.images, dtype=int)]
    hashes = [sha256(path) for path in selected]
    if len(set(hashes)) != len(selected):
        raise ValueError('Calibration requires distinct images')
    model = ROOT / 'weights/onnx/deeplsd.onnx'
    onnx.checker.check_model(onnx.load(model), full_check=True)
    metadata = json.loads((model.parent / 'metadata.json').read_text())
    if sha256(model) != metadata['sha256']:
        raise ValueError('ONNX checksum mismatch; run make export')
    from axelera import compiler
    from axelera.compiler import CompilerConfig
    from importlib.metadata import version
    with tempfile.TemporaryDirectory(prefix='deeplsd-compile-') as folder:
        compiled = Path(folder) / 'compiled'
        config = CompilerConfig(**SETTINGS, output_dir=str(compiled))
        calibration = (preprocess(cv2.imread(str(path))) for path in selected)
        print(f'Calibrating with {len(selected)} images from {args.seq}', flush=True)
        quantized = compiler.quantize(model=str(model), calibration_dataset=calibration, config=config)
        compiler.compile(model=quantized, config=config, output_dir=compiled)
        provenance = {'onnx_sha256': sha256(model), 'compiler_version': version('axelera-compiler'),
            'calibration_sequence': args.seq, 'calibration_images': len(selected), 'settings': SETTINGS,
            'images': [{'name': p.name, 'sha256': h} for p, h in zip(selected, hashes)]}
        (compiled / 'provenance.json').write_text(json.dumps(provenance, indent=2) + '\n')
        manifest_path = compiled / 'manifest.json'
        manifest = json.loads(manifest_path.read_text())
        manifest.update(quantized_model_file=None, model_params_file=None, model_lib_file='model.json')
        manifest_path.write_text(json.dumps(manifest, indent=2) + '\n')
        for path in compiled.iterdir():
            if path.is_file() and path.name not in {'model.json', 'manifest.json', 'provenance.json'} and path.suffix not in {'.bin', '.elf', '.c'}:
                path.unlink()
        destination = ROOT / 'weights/aipu'
        if destination.exists():
            shutil.rmtree(destination)
        shutil.move(str(compiled), str(destination))
    print(f'Compiled weights saved to {destination}')


if __name__ == '__main__':
    main()
