# Segment extractor optimization

Branch: `optimization/segment-extractor`, based on `experiment/temporal-consistency`.
The public `Process()` interface and all extraction thresholds are unchanged.

## Changes

- Cache each segment's angle, length and direction during joining and duplicate
  suppression; refresh only when a merge changes the segment.
- Recompute the growing region's direction, center and trigonometry only after
  accepting a pixel, rather than for every rejected neighbor.
- Cache exact double-precision sine/cosine values for repeated orientation values.
  Cache keys use all float bits; values are not rounded or quantized.
- Use stable radix sorting for validated nonnegative finite float distances.
  Signed zeros share a key so equal-distance seeds retain raster order.
- Handle common modulo-pi angle differences directly, retaining the general
  remainder fallback for arbitrary input orientations.
- Store fitting regions in one contiguous buffer. Fit and validate independent
  regions through OpenCV's existing parallel backend, with four stripes, then
  collect results in original seed order. Small batches run inline. The library
  does not change the caller's global OpenCV thread count.
- Stop validation early only when acceptance is already impossible.

Region growth, PCA accumulation, merge selection, equal-length sorting and output
ordering retain their original arithmetic/order. No `fast-math`, approximate
trigonometry, threshold changes or architecture-specific build flags are used.

## Reproduce

Build with the existing dependencies and tests enabled (the default):

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j4
ctest --test-dir build --output-on-failure

./build/cpp/cache_extraction_fields configs/default.yaml \
  data/tum/freiburg2_pioneer_slam /tmp/freiburg_fields.bin

./build/cpp/benchmark_segment_extractor configs/balanced.yaml \
  /tmp/freiburg_fields.bin results/my_benchmark.json 1 3 4
```

The cache tool performs inference once to save decoded float fields. It does not
export, calibrate, compile or change the model. Its local binary cache has three
native uint32 header values (width, height, frame count), followed by row-major
float32 distance and orientation arrays per frame. All selected source PNGs are
sorted by filename. Cache and report tools refuse existing output paths.

Benchmark arguments after the report are stride, repeats and OpenCV threads.
Use stride 1 for every frame, and threads 1 to measure the serial path. Inference,
image loading, field-cache reads, rendering and temporal processing are outside
the measured region. Each frame alternates reference/optimized call order across
repeats after four initial warmups. Timings depend on CPU scheduling/cache state;
they differ from extraction timings interleaved with live AIPU inference.

The test-only reference implementation is frozen from commit `5d918af`. It is not
linked into the production app. The benchmark fails on any segment count,
endpoint bit pattern or output-order difference. Synthetic parity tests cover
297 cases across all three presets with 1 and 4 threads, including continuous
and quantized fields, dense regions, equal distances, signed zero, angle gates,
periodic orientations and invalid input. These establish parity on the tested
compiler/platform; they are not a cross-platform floating-point guarantee.

## Full-sequence result

AMD Ryzen 7 3700X, GCC Release (`-O3`), 320x240 fields, four OpenCV threads.
The same 200-frame-calibrated model supplies all 2,921 Freiburg images.

| Preset | Reference mean | Optimized mean | Speedup | Reference p95 | Optimized p95 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Strict | 3.28 ms | 1.80 ms | 1.82x | 4.95 ms | 2.61 ms |
| Balanced | 5.22 ms | 2.47 ms | 2.12x | 8.61 ms | 3.61 ms |
| Relaxed | 9.07 ms | 3.56 ms | 2.55x | 16.82 ms | 5.10 ms |

Every endpoint float bit and segment order matches the reference for every frame
and all three presets: 792,779 segments in total. Balanced uses three timed
repeats; strict and relaxed use two. The raw fields are shared between both
implementations in each pair. See [the numerical results](segment_extractor_performance.json).

## Live-pipeline check

A separate run processes the same first 600 Freiburg frames through real AIPU
inference, including the first frame. Balanced extraction drops from 9.12 to
4.49 ms/frame; Pipeline total drops from 31.82 to 27.33 ms/frame. All 600 complete
original-coordinate segment arrays and their order match exactly. The temporal
layer is disabled to isolate the extractor. Image reads and export are excluded
from these stage timings.

Additional OpenCV thread-limit runs (1 and 4) give approximately 4.55 ms for
extraction on these 600 frames. No global thread-limit changes are therefore
introduced into the production app. Parallel fitting helps slightly in the
cache benchmark; avoiding repeated scalar work delivers most of the gain.

Final checks also pass with AddressSanitizer/UndefinedBehaviorSanitizer and leak
detection. A temporal-enabled 120-frame app run reproduces both raw and stabilized
outputs exactly. The reusable field-cache tool reproduces the first cached frame
bit-for-bit for both heads. All hashes recorded by `showcase-v1` remain unchanged.
