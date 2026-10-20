"""Export the local CNN, fold static shape operations and validate the ONNX graph."""
from collections import Counter
import json
from pathlib import Path
import tempfile
import onnx
import onnxruntime as ort
import torch
from common import ROOT, OUTPUTS, sha256
from model import CHECKPOINT, load_model


def main():
    torch.set_num_threads(4)
    model = load_model()
    destination = ROOT / 'weights/onnx/deeplsd.onnx'
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as folder:
        raw = Path(folder) / 'deeplsd_320x240.onnx'
        with torch.inference_mode():
            torch.onnx.export(model, torch.zeros(1, 1, 240, 320), str(raw),
                input_names=['image'], output_names=OUTPUTS, opset_version=17,
                dynamo=False, external_data=False)
        options = ort.SessionOptions()
        options.intra_op_num_threads = 4
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
        options.optimized_model_filepath = str(destination)
        ort.InferenceSession(str(raw), options, providers=['CPUExecutionProvider'])
    graph = onnx.load(destination)
    onnx.checker.check_model(graph, full_check=True)
    assert not any(t.external_data for t in graph.graph.initializer)
    assert not any(n.op_type in {'Shape', 'Slice'} for n in graph.graph.node)
    assert all(n.domain in {'', 'ai.onnx'} for n in graph.graph.node)
    metadata = {'sha256': sha256(destination), 'torch_weights_sha256': sha256(CHECKPOINT),
                'input': [1, 1, 240, 320], 'outputs': OUTPUTS, 'opset': 17,
                'operators': dict(Counter(n.op_type for n in graph.graph.node))}
    (destination.parent / 'metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
    print(f'Exported and checked {destination}')


if __name__ == '__main__':
    main()
