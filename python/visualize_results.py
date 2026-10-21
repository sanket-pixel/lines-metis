#!/usr/bin/env python3
"""Render cached lines-metis detections without executing the model."""
import argparse
import json
from pathlib import Path
import subprocess

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

W, H = 1920, 1080
BG = (12, 18, 25)
FG = (236, 243, 246)
MUTED = (133, 154, 168)
CYAN = (99, 232, 218)
FONT = Path('/usr/share/fonts/truetype/ubuntu')


def font(size, mono=False, bold=False):
    name = 'UbuntuMono-R.ttf' if mono else ('Ubuntu-M.ttf' if bold else 'Ubuntu-R.ttf')
    return ImageFont.truetype(str(FONT / name), size)


def load_results(path):
    metadata, frames, summary = None, [], None
    with path.open() as stream:
        for line in stream:
            item = json.loads(line)
            if item['type'] == 'metadata':
                metadata = item
            elif item['type'] == 'frame':
                frames.append(item)
            elif item['type'] == 'summary':
                summary = item
    if not metadata or not summary or summary['frames'] != len(frames):
        raise ValueError('Expected a complete detection export')
    if metadata['coordinate_space'] != 'original_image_pixels':
        raise ValueError('Expected original-image endpoints')
    return metadata, frames


class Renderer:
    def __init__(self, metadata, frames, image_root):
        self.metadata, self.frames, self.image_root = metadata, frames, image_root
        self.fonts = {size: font(size) for size in (18, 20, 22, 24, 28, 32, 40, 64, 84)}
        self.mono = font(22, mono=True)
        self.base = Image.new('RGB', (W, H), BG)
        d = ImageDraw.Draw(self.base)
        d.text((48, 29), 'lines / metis', font=font(40, bold=True), fill=FG)
        d.text((48, 83), 'LEARNED LINE DETECTION  /  AXELERA METIS', font=self.fonts[18], fill=MUTED)
        d.rounded_rectangle((1603, 40, 1872, 82), radius=21, fill=(27, 50, 51))
        d.text((1625, 49), 'DeepLSD  /  INT8 AIPU', font=self.fonts[20], fill=CYAN)
        d.line((48, 123, 1872, 123), fill=(45, 59, 70))
        d.text((48, 151), '01  /  SCENE + DETECTED SEGMENTS', font=self.fonts[20], fill=MUTED)
        d.text((1360, 151), '02  /  STRUCTURE', font=self.fonts[20], fill=MUTED)
        d.text((1360, 522), '03  /  PIPELINE', font=self.fonts[20], fill=MUTED)
        d.text((48, 1028), 'TUM RGB-D · Freiburg 2 pioneer SLAM · J. Sturm et al. · CC BY 4.0', font=self.fonts[18], fill=MUTED)
        d.text((1190, 1028), 'cvg.cit.tum.de/data/datasets/rgbd-dataset', font=self.fonts[18], fill=MUTED)
        self.means = {key: np.mean([f['timings_ms'][key] for f in frames]) for key in frames[0]['timings_ms']}

    def render(self, index):
        frame = self.frames[index]
        source = Image.open(self.image_root / frame['file']).convert('RGB')
        canvas = self.base.copy()
        # Keep the complete source aspect ratio, and draw every saved segment.
        scene = source.resize((1056, 792), Image.Resampling.LANCZOS)
        scene = Image.blend(scene, Image.new('RGB', scene.size, (13, 26, 32)), .12)
        canvas.paste(scene, (144, 191))
        mask = np.zeros((H, W), dtype=np.uint8)
        for x1, y1, x2, y2 in frame['segments']:
            a = (round(144 + x1 * 1056 / frame['width']), round(191 + y1 * 792 / frame['height']))
            b = (round(144 + x2 * 1056 / frame['width']), round(191 + y2 * 792 / frame['height']))
            cv2.line(mask, a, b, 255, 2, cv2.LINE_AA)
        array = np.array(canvas)
        visible = mask > 0
        alpha = mask[visible, None] / 255
        array[visible] = (array[visible] * (1 - alpha) + np.array(CYAN) * alpha).astype(np.uint8)
        canvas = Image.fromarray(array)
        d = ImageDraw.Draw(canvas)
        # A geometry-only companion view shows exactly the same detections.
        d.rectangle((1360, 191, 1872, 481), fill=(17, 27, 36))
        for x in range(1360, 1873, 32):
            d.line((x, 191, x, 481), fill=(25, 37, 47))
        for y in range(191, 482, 32):
            d.line((1360, y, 1872, y), fill=(25, 37, 47))
        for x1, y1, x2, y2 in frame['segments']:
            d.line((1423+x1*384/frame['width'], 192+y1*288/frame['height'], 1423+x2*384/frame['width'], 192+y2*288/frame['height']), fill=CYAN, width=1)
        d.text((1360, 570), str(len(frame['segments'])), font=self.fonts[84], fill=FG)
        d.text((1520, 616), 'segments / frame', font=self.fonts[24], fill=MUTED)
        d.line((1360, 676, 1872, 676), fill=(45, 59, 70))
        d.text((1360, 700), f"{self.means['pipeline']:.1f} ms", font=self.fonts[40], fill=FG)
        d.text((1620, 713), f"{1000/self.means['pipeline']:.1f} FPS", font=self.fonts[28], fill=CYAN)
        d.text((1360, 752), 'Mean processing · capture / render excluded', font=self.fonts[18], fill=MUTED)
        colors = [(130,163,184), CYAN, (159,177,224), (227,188,124)]
        x = 1360
        keys = ['preprocessing', 'inference', 'postprocessing', 'extraction']
        for key, color in zip(keys, colors):
            width = 512 * self.means[key] / sum(self.means[k] for k in keys)
            d.rectangle((x, 799, x+width-2, 810), fill=color)
            x += width
        for n, (key, label, color) in enumerate(zip(keys, ['Pre', 'AIPU', 'Post', 'Lines'], colors)):
            x = 1360 + n*128
            d.text((x, 833), label, font=self.fonts[20], fill=color)
            d.text((x, 866), f'{self.means[key]:.1f} ms', font=self.mono, fill=FG)
        d.text((1360, 935), f"{self.metadata['preset'].upper()}  /  {self.metadata['model_provenance']['calibration_images']}-FRAME CALIBRATION", font=self.fonts[18], fill=MUTED)
        elapsed = frame['timestamp_seconds'] - self.frames[0]['timestamp_seconds']
        d.text((48, 991), f'FREIBURG 2   /   {elapsed:06.2f}s source', font=self.mono, fill=MUTED)
        d.text((870, 991), f'{index+1:04d} / {len(self.frames):04d}', font=self.mono, fill=MUTED)
        d.line((48, 1019, 1872, 1019), fill=(39, 54, 65), width=2)
        d.line((48, 1019, 48+1824*(index+1)/len(self.frames), 1019), fill=CYAN, width=2)
        return canvas


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, required=True)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--preview', type=Path)
    parser.add_argument('--preview-frame', type=int, default=0)
    parser.add_argument('--image-root', type=Path)
    parser.add_argument('--fps', type=float, default=30)
    parser.add_argument('--start', type=int, default=0)
    parser.add_argument('--frames', type=int)
    args = parser.parse_args()
    if args.fps <= 0 or args.start < 0 or (args.frames is not None and args.frames <= 0):
        parser.error('FPS and frame count must be positive; start must be nonnegative')
    metadata, frames = load_results(args.results)
    if args.start >= len(frames) or not 0 <= args.preview_frame < len(frames):
        parser.error('Frame index outside the export')
    renderer = Renderer(metadata, frames, args.image_root or Path(metadata['input']['path']))
    for path in (args.output, args.preview):
        if path and path.exists():
            parser.error(f'Refusing to overwrite {path}')
        if path:
            path.parent.mkdir(parents=True, exist_ok=True)
    if args.preview:
        renderer.render(args.preview_frame).save(args.preview)
    if not args.output:
        return
    selected = range(args.start, min(len(frames), args.start + args.frames) if args.frames else len(frames))
    command = ['ffmpeg', '-v', 'error', '-n', '-f', 'rawvideo', '-pixel_format', 'rgb24', '-video_size', f'{W}x{H}', '-framerate', str(args.fps), '-i', '-', '-an', '-c:v', 'libx264', '-preset', 'fast', '-crf', '18', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(args.output)]
    process = subprocess.Popen(command, stdin=subprocess.PIPE)
    try:
        for count, index in enumerate(selected):
            process.stdin.write(renderer.render(index).tobytes())
            if count % 100 == 0:
                print(f'Rendered {count+1}/{len(selected)}', flush=True)
        process.stdin.close()
        if process.wait():
            raise RuntimeError('FFmpeg encoding failed')
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()
    args.output.with_suffix('.json').write_text(json.dumps({'detections': str(args.results), 'rendered_frames': len(selected), 'first_frame': args.start, 'playback_fps': args.fps, 'timings': 'Full cached run mean; excludes capture and rendering', 'style': 'Exact endpoints, no tracking or synthetic heatmap'}, indent=2)+'\n')


if __name__ == '__main__':
    main()
