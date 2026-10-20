"""Run one TUM sequence and write a standalone line-overlay video."""
import argparse
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import subprocess
import time
import cv2
import numpy as np
from common import ROOT, OUTPUTS, images, preprocess, extract, map_lines


@contextmanager
def backend(name, threads=4):
    if name == 'aipu':
        from runtime import load_aipu
        with load_aipu() as loaded:
            yield loaded
        return
    if name == 'torch':
        import torch
        from model import load_model
        torch.set_num_threads(threads)
        model = load_model()
        def predict(image):
            with torch.inference_mode():
                return [t.numpy() for t in model(torch.from_numpy(image))]
    else:
        import onnxruntime as ort
        options = ort.SessionOptions()
        options.intra_op_num_threads = threads
        options.inter_op_num_threads = 1
        session = ort.InferenceSession(str(ROOT / 'weights/onnx/deeplsd.onnx'), options,
                                       providers=['CPUExecutionProvider'])
        def predict(image):
            return session.run(OUTPUTS, {'image': image})
    def run(image):
        start = time.perf_counter()
        fields = predict(image)
        return fields, (time.perf_counter() - start) * 1000, 0.
    yield run, None


def render(image, lines, name, fps, frame_index, total, sequence):
    frame = cv2.resize(image, (1280, 960))
    endpoints = map_lines(lines, frame.shape)
    for segment in endpoints:
        cv2.line(frame, tuple(np.rint(segment[0]).astype(int)), tuple(np.rint(segment[1]).astype(int)),
                 (106, 244, 129), 2, cv2.LINE_AA)
    shade = frame.copy()
    cv2.rectangle(shade, (0, 0), (1280, 62), (14, 17, 20), -1)
    cv2.rectangle(shade, (0, 886), (1280, 960), (14, 17, 20), -1)
    frame = cv2.addWeighted(shade, .82, frame, .18, 0)
    label = {'torch': 'PyTorch FP32', 'onnx': 'ONNX Runtime FP32', 'aipu': 'Axelera Metis INT8'}[name]
    cv2.putText(frame, f'ZEROPOINT VISION  |  DeepLSD  |  {label}', (24, 40),
                cv2.FONT_HERSHEY_SIMPLEX, .8, (240, 246, 245), 2, cv2.LINE_AA)
    cv2.putText(frame, f'Processing: {fps:.1f} FPS  |  {len(lines)} lines  |  {frame_index + 1}/{total}',
                (24, 915), cv2.FONT_HERSHEY_SIMPLEX, .75, (106, 244, 129), 2, cv2.LINE_AA)
    cv2.putText(frame, f'TUM RGB-D / {sequence} / CC BY 4.0  |  FPS excludes video I/O',
                (24, 946), cv2.FONT_HERSHEY_SIMPLEX, .57, (225, 228, 232), 1, cv2.LINE_AA)
    return frame


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--backend', choices=['torch', 'onnx', 'aipu'], default='torch')
    parser.add_argument('--seq', default='freiburg2_pioneer_slam')
    parser.add_argument('--frames', type=int, default=100)
    parser.add_argument('--start', type=int, default=0)
    parser.add_argument('--fps', type=int, default=30, help='Video playback FPS, independent of processing speed')
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--output', type=Path, help='Default: results/<backend>.mp4')
    args = parser.parse_args()
    parser.error("Line extraction is unavailable until the C++ extractor is implemented")
    if args.fps < 1 or args.threads < 1:
        parser.error('FPS and threads must be positive')
    cv2.setNumThreads(args.threads)
    paths = images(args.seq, args.frames, args.start)
    output = args.output or ROOT / 'results' / f'{args.backend}.mp4'
    output.parent.mkdir(parents=True, exist_ok=True)
    timings, core_times, clipping, counts = [], [], [], []
    with backend(args.backend, args.threads) as (run, mapping):
        first = preprocess(cv2.imread(str(paths[0])))
        for _ in range(3):
            extract(run(first)[0])
        writer = subprocess.Popen(['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo',
            '-pix_fmt', 'bgr24', '-s', '1280x960', '-r', str(args.fps), '-i', '-', '-an',
            '-c:v', 'libx264', '-threads', '2', '-preset', 'fast', '-crf', '18',
            '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(output)], stdin=subprocess.PIPE)
        try:
            for index, path in enumerate(paths):
                image = cv2.imread(str(path))
                if image is None:
                    raise ValueError(f'Cannot read {path}')
                start = time.perf_counter()
                prepared = preprocess(image)
                fields, core_ms, clipped = run(prepared)
                lines = extract(fields)
                timings.append((time.perf_counter() - start) * 1000)
                core_times.append(core_ms)
                clipping.append(clipped)
                counts.append(len(lines))
                frame = render(image, lines, args.backend, 1000 / np.mean(timings[-30:]),
                               index, len(paths), args.seq)
                writer.stdin.write(frame.tobytes())
                if (index + 1) % 50 == 0:
                    print(f'{args.backend}: {index + 1}/{len(paths)}', flush=True)
        finally:
            writer.stdin.close()
            status = writer.wait()
        if status:
            raise RuntimeError('Video encoding failed')
    result = {'sequence': args.seq, 'frames': len(paths), 'start': args.start,
              'frame_order_sha256': hashlib.sha256('\n'.join(p.name for p in paths).encode()).hexdigest(),
              'video': str(output.relative_to(ROOT)) if output.is_relative_to(ROOT) else str(output),
              'playback_fps': args.fps, 'processing_fps': 1000 / float(np.mean(timings)),
              'pipeline_ms_median': float(np.median(timings)), 'pipeline_ms_p95': float(np.percentile(timings, 95)),
              'model_ms_median': float(np.median(core_times)), 'mean_line_count': float(np.mean(counts)),
              'input_clipping_fraction_mean': float(np.mean(clipping)), 'warmup_frames': 3,
              'timing': 'Continuous CPU preprocessing + model + CPU line extraction; excludes image read, rendering and video encoding',
              'output_mapping': mapping}
    if args.output is not None:
        output.with_suffix('.json').write_text(json.dumps(result, indent=2) + '\n')
        print(json.dumps(result, indent=2))
        return
    metrics_path = ROOT / 'results/metrics.json'
    metrics = json.loads(metrics_path.read_text()) if metrics_path.exists() else {}
    metrics[args.backend] = result
    metrics_path.write_text(json.dumps(metrics, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
