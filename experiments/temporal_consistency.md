# Temporal consistency experiment

Branch: `experiment/temporal-consistency`. The frozen `showcase-v1` baseline remains
unchanged. This adds an optional C++ consumer after `Pipeline::Process()`; no model,
preprocessing, extraction or coordinate-mapping settings are changed.

`LineStabilizer::Process(image, segments, timestamp)` accepts original-image
endpoints and returns original-image endpoints. Each camera/sequence needs its
own instance. Call `Reset()` when switching streams. It is causal, with no future
frames or output frame queue.

## Method

1. Sample small neighborhoods along each previous segment. Forward/backward
   sparse pyramidal Lucas–Kanade flow rejects inconsistent sample motion.
2. Fit a local similarity transform to the consistent samples. This allows
   different structures to move differently under parallax; there is no single
   whole-scene homography.
3. Associate predicted segments with current detections using angle, normal
   distance and longitudinal overlap, with one-to-one greedy matching.
4. Blend motion-predicted and observed endpoints. If motion is unreliable, use
   the current observation directly rather than smoothing stale coordinates.
5. Confirm new lines on two consecutive detections. Confirmed missing lines may
   survive two frames only with reliable motion and current image edge support.
   Remove carried duplicates when matched observations already cover them.

Internal motion estimation uses grayscale images at a maximum width of 640
pixels, retaining aspect ratio and pixel-center coordinates. Final clipping
preserves each segment's direction. Track state resets on resolution changes,
non-increasing timestamps, gaps above 0.25 seconds and large appearance changes.
Scene-cut detection is heuristic; this is not a general occlusion or identity
tracker.

## Run

Build with the OpenCV `video` module in addition to the existing dependencies:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j4
ctest --test-dir build --output-on-failure
```

The current compact video app has no temporal or export switches. The earlier
app integration is preserved in commit `4329403`; the module and replay tool
remain available for cached experiments. Temporal settings are in their own YAML
file. Replay schema 2 retains `raw_segments`, stabilized `segments`, resolved
settings and separate `temporal_ms` measurements.

Replay the saved sequence without loading the model:

```bash
./build/cpp/stabilize_results \
  results/freiburg2_pioneer_slam_aipu_calib200_balanced_detections.jsonl \
  configs/temporal.yaml results/my_temporal.jsonl

python python/compare_temporal.py --results results/my_temporal.jsonl \
  --output results/my_comparison.mp4 --preview results/my_comparison.png

python python/measure_temporal.py --results results/my_temporal.jsonl \
  --output results/my_temporal_analysis.json
```

The renderer draws the raw and stabilized endpoints with identical styling and
synchronized source images. Its playback FPS is independent of inference speed.
The frozen showcase renderer is unchanged.

## Evaluation and limits

See [the evaluation record](temporal_v2_evaluation.json). All 2,921 Freiburg frames
are replayed from the same balanced detections. All original raw arrays are
retained exactly; no AIPU inference is performed by the replay tool. Evaluation
uses 584 adjacent frame pairs distributed across the full sequence. An independent
backward dense Farneback flow aligns previous line masks at 320x240. Symmetric
mask distances measure spatial change after motion compensation.

This is a stability proxy, not an accuracy or recall benchmark. Filtering away
unstable valid lines also improves it. New detections have a one-frame
confirmation delay by default, and short-lived valid lines can disappear. Brief
supported prediction can still produce stale or incorrect segments. Repeated
parallel structures, abrupt motion, blur and split/merged detections can break
association. There is no 3D reconstruction, loop closure or persistent identity
contract. The original extraction's missing lines are not recovered by this layer.

Tests cover confirmation, jitter reduction, endpoint reversal, bounded dropout,
translation, rotation/zoom, occlusion, timestamp and resolution resets, clipping,
resize-coordinate round trips and invalid settings. The test target also passes
AddressSanitizer/UndefinedBehaviorSanitizer with leak detection.
