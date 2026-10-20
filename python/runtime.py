"""AxRuntime buffers: manifest order, affine quantization and explicit NHWC padding."""
from contextlib import contextmanager
import json
from pathlib import Path
import time
import numpy as np
from common import ROOT, OUTPUTS, sha256


@contextmanager
def load_aipu(directory=ROOT / 'weights/aipu'):
    from axelera.runtime import Context
    directory = Path(directory)
    manifest = json.loads((directory / 'manifest.json').read_text())
    description = json.loads((directory / 'model.json').read_text())
    provenance = json.loads((directory / 'provenance.json').read_text())
    if provenance['onnx_sha256'] != sha256(ROOT / 'weights/onnx/deeplsd.onnx'):
        raise ValueError('Compiled weights do not match the ONNX file; compile again')
    if manifest['input_tensor_layout'] != 'NHWC' or manifest.get('preprocess_graph') or manifest.get('postprocess_graph'):
        raise ValueError('Expected an NHWC model without CPU split graphs')
    with Context() as context:
        model = context.load_model(directory / 'model.json')
        inputs, outputs = model.inputs(), model.outputs()
        if len(inputs) != 1 or len(outputs) != 2:
            raise ValueError('Expected one input and separate distance/orientation outputs')
        for info, shape, padding in zip(
            inputs + outputs,
            manifest['input_shapes'] + manifest['output_shapes'],
            manifest['n_padded_ch_inputs'] + manifest['n_padded_ch_outputs'],
        ):
            if (list(info.shape) != shape or tuple(info.unpadded_shape) != (1, 240, 320, 1)
                    or np.dtype(info.dtype) != np.dtype('int8') or info.scale <= 0
                    or [v for pair in info.padding for v in pair] != padding):
                raise ValueError(f'Unexpected runtime tensor: {info}')
        for info, params in zip(inputs + outputs, manifest['quantize_params'] + manifest['dequantize_params']):
            np.testing.assert_array_equal([info.scale, info.zero_point], params)
        if [o.name for o in outputs] != [o['name'] for o in description['outputs']]:
            raise ValueError('Runtime output IDs differ from model.json')
        mapping = [{'id': i, 'field': name, 'tensor': info.name, 'scale': float(info.scale),
                    'zero_point': int(info.zero_point)} for i, (name, info) in enumerate(zip(OUTPUTS, outputs))]
        info = inputs[0]
        buffers = [np.empty(t.shape, dtype=t.dtype) for t in outputs]
        with context.device_connect(num_sub_devices=1) as connection:
            with connection.load_model_instance(model, num_sub_devices=1, aipu_cores=1,
                    double_buffer=False, input_dmabuf=False, output_dmabuf=False) as instance:
                def run(image):
                    logical = image.transpose(0, 2, 3, 1)
                    rounded = np.rint(logical / np.float32(info.scale) + np.float32(info.zero_point))
                    clipping = float(np.mean((rounded < -128) | (rounded > 127)))
                    # Zero point may be outside int8. Saturate; never wrap -131 to +125.
                    padded = np.pad(np.clip(rounded, -128, 127).astype(np.int8), info.padding,
                                    constant_values=int(np.clip(info.zero_point, -128, 127)))
                    padded = np.ascontiguousarray(padded)
                    if padded.nbytes != info.size:
                        raise ValueError('Input buffer size mismatch')
                    start = time.perf_counter()
                    instance.run([padded], buffers)
                    milliseconds = (time.perf_counter() - start) * 1000
                    fields = []
                    # ABI IDs, not DDR offsets: distance is 0, orientation is 1.
                    for buffer, tensor in zip(buffers, outputs):
                        crop = tuple(slice(left, size - right) for size, (left, right) in zip(tensor.shape, tensor.padding))
                        logical = buffer[crop].astype(np.float32)
                        value = (logical - np.float32(tensor.zero_point)) * np.float32(tensor.scale)
                        fields.append(np.ascontiguousarray(value.transpose(0, 3, 1, 2)))
                    return fields, milliseconds, clipping
                yield run, mapping
