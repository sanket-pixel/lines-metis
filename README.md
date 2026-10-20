# lines-metis

DeepLSD line segment detection on Metis. The C++ port currently implements configuration loading, image preprocessing and
the AIPU backend, postprocessing, segment extraction and Pipeline orchestration. The legacy extractor and Python binding
have been removed. The app provides live overlays; a Python renderer creates polished videos from cached detections. Python model, preprocessing and AxRuntime logic remain as references.

```text
python/                 model, inference reference, export and compilation
cpp/app/                CLI arguments, input sources, result export and preview
cpp/include/            C++ module headers
cpp/src/                module implementations
weights/torch/          official MegaDepth weights and provenance
weights/onnx/           static FP32 model
weights/aipu/           compiled model and runtime manifest
data/tum/<sequence>/   PNG/JPEG frames
results/                standalone videos and measured timings
```

## Setup and commands

Run `./setup.sh` with uv and curl installed.
Setup uses `/home/sanket-shah/venvs/torch-cpu`. It does not modify the Voyager environment. Missing weights are downloaded from
the official source and verified against the recorded checkpoint hash.

```bash
make export                         # Embedded FP32 ONNX; checker and agreement check
make compile                        # Manual compilation with 10 calibration images
```

Parameters: `BACKEND=torch`, `SEQ=freiburg2_pioneer_slam`, `FRAMES=100`,
`START=0`, `FPS=30`, `THREADS=4`, `CALIB_IMAGES=10`. Override `PYTHON` or `SDK`
if needed. Calibration uses distinct real images and the inference preprocessing.
Compilation works in a temporary directory before updating `weights/aipu/`.

Build with OpenCV, yaml-cpp, nlohmann_json, OpenSSL and AxRuntime installed:

```bash
cmake -S . -B build
cmake --build build -j4
```

Set `-DAXRUNTIME_ROOT=/path/to/runtime` when needed. The `lines_metis` app currently loads and validates YAML settings, prints the resolved
`PipelineConfig`, and runs image, video or webcam input through the full pipeline:

```bash
./build/cpp/lines_metis --image data/tum/freiburg2_pioneer_slam/<frame>.png --config configs/default.yaml
./build/cpp/lines_metis --video /path/to/video.mp4 --config configs/default.yaml
./build/cpp/lines_metis --webcam 0 --config configs/default.yaml --show
```

Choose exactly one source: `--image <path>`, `--sequence <directory>`,
`--video <path>` or `--webcam <index>`. Sequences read PNG/JPEG files in sorted filename order.
The previous positional image-path syntax also works. Images are processed once;
video ends at EOF, and webcam input continues until stopped. `--max-frames N`
optionally limits processing; Ctrl+C stops the loop and releases its resources.
Add `--show` for a window with green segment overlays and the current detection
count and measured frame rate. Press Esc/Q or close the window to stop the preview.
Add `--profile` to print average preprocessing, inference, postprocessing and
segment-extraction times in milliseconds per frame when processing ends. The
pipeline total also includes coordinate mapping. These figures include the first
frame and exclude model loading, capture, rendering and console output. For a
live preview, stop with Esc/Q or Ctrl+C to see the summary.

Use a Release build (or the Release profile in CLion) for live input. Linux webcam
capture uses V4L2 and requests a small capture buffer to reduce lag.
Each frame prints its segment count and a few endpoint pairs. Input handling does
not add temporal state to the Pipeline.

In CLion, select the `lines_metis` target and put one source option plus `--config`
and its YAML path in Program arguments. Paths inside YAML files resolve
relative to their containing file. `segment_extractor.strictness` selects a preset
from `presets_file`; optional `min_length` and `max_join_gap` override its values.
Incomplete thresholds, invalid values and unknown keys are errors.

Preset comparison (distances and lengths in network pixels):

| Preset | Minimum support | Maximum mean distance | Maximum join gap | Minimum length |
| --- | ---: | ---: | ---: | ---: |
| Strict | 0.8 | 0.8 | 4 | 10 |
| Balanced | 0.7 | 1.1 | 8 | 10 |
| Relaxed | 0.6 | 1.5 | 12 | 8 |

Balanced is the default in `configs/default.yaml`. Strict remains the original
accepted baseline. Balanced and relaxed are experimental:
they accept weaker support and bridge larger gaps, recovering additional lines
but also adding clutter in the sampled scenes. Use `--config configs/balanced.yaml`
or `--config configs/relaxed.yaml` to compare them. These files use the preset
values directly; `configs/default.yaml` also uses balanced without overrides.
The previous strict default is preserved as `configs/strict.yaml`.
Full threshold definitions, including region/angle and lateral-join tolerances,
are in `configs/segment_presets.yaml`.

Export reusable detections for Python visualization without another inference run:

```bash
./build/cpp/lines_metis --sequence data/tum/freiburg2_pioneer_slam \
  --config configs/default.yaml \
  --output results/freiburg2_pioneer_slam_aipu_calib200_balanced_detections.jsonl --profile
```

`--output` refuses to overwrite an existing file. JSONL contains one metadata
record, one record per processed frame, then a summary with the exported count.
Metadata captures the input location, selected preset, resolved thresholds and
model provenance. Frame records contain zero-based `index`, `file`,
`timestamp_seconds`, `width`, `height`, `segments` and `timings_ms`.
Each segment is `[x1,y1,x2,y2]` in original-image pixels, already mapped once.
Freiburg timestamps come from the numeric filenames, in seconds; nonnumeric image
names yield null timestamps. Video timestamps use capture position, while webcam
timestamps measure elapsed time since opening the source. Timing values exclude
image loading, rendering and result serialization. A missing final summary means
the export did not finish successfully.

A Python renderer can load each frame record with `json.loads(line)`, open the
original image under the metadata input directory, and draw `segments` directly.
The saved source timestamps let it choose playback timing independently of
Pipeline processing speed.

The app constructs one `Pipeline(config)` per input and calls `Process(frame)`
for each frame. Pipeline owns the
AIPU context and all four modules, reuses its decoded-field buffers, and calls
preprocessing, inference, postprocessing and segment extraction in order. It then
maps endpoints once to original-image coordinates and returns `std::vector<Segment>`.
The app prints the segment count and first five endpoint pairs; the modules do not
print intermediate diagnostics.

Preprocessing converts BGR to grayscale, resizes to 320x240 with INTER_AREA,
normalizes with `/255`, then quantizes and pads the backend input using runtime
metadata. No stage-specific tensor settings are exposed in YAML.

Quantization matches Python's float32 affine arithmetic, ties-to-even rounding
and int8 saturation. All physical padding uses the saturated zero point. For the
current model this produces NHWC `[1,242,322,64]`, with padding value `-128` even
though the metadata zero point is `-131`.
The end-to-end entry point currently composes the CPU stages internally; a future
fused kernel can replace that implementation. The float reference and separate
`QuantizeAndPad()` methods remain available for numerical verification.
Postprocessing crops each output using its own padding, independently dequantizes
the heads, then decodes `distance = 5*exp(-distance_normalized)` and
`orientation = pi*orientation_fraction`. It reuses two contiguous row-major float
vectors of 240x320 samples, in network pixels and radians. Orientation matched
Python exactly on the verification frame; distance differed by at most
`4.77e-7` pixels due to floating-point exponential evaluation. No postprocessing
YAML settings are needed.

`SegmentExtractor` consumes `config.segment_extractor`, including the resolved
preset thresholds and YAML overrides. It grows orientation-consistent regions,
fits weighted PCA segments, validates sampled support and mean distance, optionally
joins collinear segments, and suppresses duplicates. The accepted strict preset
keeps `min_support=0.8` and `max_mean_distance=0.8` network pixels. `joining_enabled`
controls joining, while `min_length` and `max_join_gap` override the preset. The extractor returns endpoints in network coordinates. Pipeline alone maps them
using `(coordinate + 0.5) * scale - 0.5`; callers receive original-image coordinates.
Mapping was checked against Python on 640x480, 517x301 and 157x111 inputs derived
from a Freiburg frame, and repeated calls reuse the same Pipeline instance.

`AipuContext` is shared by reference across `AipuInference` objects and must outlive
them. Each backend owns its connection, model, instance and int8 buffers. It checks
runtime tensors against `manifest.json` and `model.json` before connecting a device.
Provenance verification reads `provenance.json` and the sibling
`../onnx/deeplsd.onnx` relative to the configured model directory. `num_cores`
controls model-instance allocation on one connected subdevice. `Process()` runs
already quantized/padded input buffers; output indices 0 and 1 are distance and
orientation, independent of their DDR offsets.

## Pipeline

The standard DeepLSD VGG U-Net receives grayscale FP32 `[1,1,240,320]`: BGR to
grayscale, `INTER_AREA` resize, then division by 255. Its separate heads produce
distance and orientation values. CPU decoding uses `5*exp(-distance_normalized)`
and `pi*orientation_fraction`.

The C++ extractor grows regions from these fields, fits weighted PCA segments,
validates field support, joins supported collinear pieces when enabled and
suppresses duplicates. Endpoints map back with the pixel-center
resize convention. See [the native API](cpp/include/segment_extractor/segment_extractor.h).

This extractor differs from upstream DeepLSD's guided LSD: it omits image-gradient
disambiguation, rectangle refinement and NFA significance testing. It can miss
weak lines or join nearby structures. There is no tracking or accuracy claim.
Non-4:3 images are stretched by the fixed resize.

AxRuntime checks the manifest, quantizes and pads NHWC int8 input, crops output
padding and dequantizes each head independently. Output ABI ID 0 is distance;
ID 1 is orientation. It uses one subdevice/core. The current local compiled model uses 200 Freiburg 2 calibration frames.
Checkpoint and compiled-model provenance remain in `weights/`.

## Results

Full C++ Pipeline videos use the same 200-frame calibrated model and all 2,921
Freiburg frames, rendered at 30 FPS. Each video has a JSON sidecar with resolved
settings, model hashes, input frame order and profiling measurements.

| Preset | Video | Mean segments/frame | Mean Pipeline FPS |
| --- | --- | ---: | ---: |
| Strict | [Video](results/freiburg2_pioneer_slam_aipu_calib200_full.mp4) | 62.1 | 34.8 |
| Balanced (experimental) | [Video](results/freiburg2_pioneer_slam_aipu_calib200_balanced_full.mp4) | 93.3 | 31.8 |
| Relaxed (experimental) | [Video](results/freiburg2_pioneer_slam_aipu_calib200_relaxed_full.mp4) | 116.1 | 30.0 |

The increased segment counts are not accuracy scores. Sampled frames show more
recovered structure and longer segments on average, with additional clutter.
Older `aipu.mp4` and `aipu_strict.mp4` results predate removal of the legacy extractor.
[The AIPU video](results/aipu.mp4) uses the first 300 Freiburg 2 frames, played at
30 FPS. Processing FPS includes preprocessing, inference and CPU line extraction;
it excludes image reads, rendering and encoding. Playback FPS is independent.
`results/metrics.json` records timings and the frame-order fingerprint.

Model weights are MIT; the C++ implementation is AGPL-3.0-or-later.
See [attribution and licenses](THIRD_PARTY_NOTICES.md).

### Python showcase renderer

Render cached endpoints without loading the model:

```bash
python python/visualize_results.py \
  --results results/freiburg2_pioneer_slam_aipu_calib200_balanced_detections.jsonl \
  --output results/my_showcase.mp4 \
  --preview results/my_showcase.png --preview-frame 600
```

Requires NumPy, OpenCV, Pillow, FFmpeg and Ubuntu fonts (installed system fonts).
The 1080p composition pairs the exact saved detections with a geometry-only view.
Performance labels use averages from the complete cached run, excluding capture,
rendering and encoding; playback defaults to 30 FPS independently. The source
clock uses dataset timestamps. No tracking, line filtering or heatmap is added.
Use `--start` and `--frames` for excerpts, `--image-root` to relocate source images,
and `--preview` without `--output` for a still. Existing files are never overwritten.

### Frozen showcase baseline

`showcase-v1` identifies the balanced C++ pipeline and Python visualization baseline.
[Baseline hashes](baselines/showcase_v1.json) identify the exact local models,
detections and showcase artifacts. Git contains source, configs, licenses and
provenance; datasets, model binaries, generated results, build products and IDE
state stay local. Result links above refer to locally generated files.
