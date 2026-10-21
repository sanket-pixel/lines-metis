SHELL := /bin/bash
.SHELLFLAGS := -eu -o pipefail -c
MAKEFLAGS += --no-print-directory
.DEFAULT_GOAL := build
PYTHON ?= .venv/bin/python
SDK ?= ../voyager-sdk
SDK_PYTHON ?= $(SDK)/venv/bin/python
SEQ := freiburg2_pioneer_slam
FPS ?= 30
DURATION ?= 30
CALIB_IMAGES ?= 200
BUILD_DIR ?= build
JOBS ?= 4
PRESET ?= balanced
CONFIG ?= configs/$(PRESET).yaml
VIDEO ?= data/sample.mp4
OUTPUT ?= results/$(PRESET)/video.mp4
DETECTIONS ?= results/$(PRESET)/detections.jsonl
SHOWCASE ?= results/$(PRESET)/showcase.mp4
GIF ?= assets/demo.gif
AXRUNTIME_ROOT ?= /opt/axelera

.PHONY: help setup data sample build video profile fancy render gif export compile

help:
	@printf '%s\n' \
	  'make setup    Python environment + official DeepLSD weights' \
	  'make data     Download Freiburg 2 pioneer SLAM RGB frames' \
	  'make sample   Create a lossless 30-second data/sample.mp4' \
	  'make export   Export and verify FP32 ONNX' \
	  'make compile  Calibrate on 200 Freiburg frames and compile for Metis' \
	  'make build    Build the Release C++ app' \
	  'make video    Save a simple line-overlay video' \
	  'make profile  Print stage timings without rendering' \
	  'make fancy    Run C++ inference + save the Python 1080p showcase' \
	  'make render   Restyle saved detections without running inference' \
	  'make gif      Create assets/demo.gif from the saved showcase' \
	  'Options: PRESET=balanced|strict|relaxed VIDEO=/path/to/video.mp4 CONFIG=/path/to/config.yaml'

setup:
	./setup.sh

data:
	@if [ "$$(find "data/tum/$(SEQ)" -maxdepth 1 -name '*.png' 2>/dev/null | wc -l)" -eq 2921 ]; then \
	  echo 'Freiburg RGB frames already present'; exit 0; fi; \
	mkdir -p data/tum; \
	work_dir=$$(mktemp -d data/tum/.download-XXXXXX); \
	trap 'rm -rf "$$work_dir"' EXIT; \
	curl -L --fail --retry 3 'https://cvg.cit.tum.de/rgbd/dataset/freiburg2/rgbd_dataset_freiburg2_pioneer_slam.tgz' -o "$$work_dir/sequence.tgz"; \
	mkdir "$$work_dir/rgb"; \
	tar -xzf "$$work_dir/sequence.tgz" -C "$$work_dir/rgb" --strip-components=2 --wildcards 'rgbd_dataset_freiburg2_pioneer_slam/rgb/*.png'; \
	[ "$$(find "$$work_dir/rgb" -name '*.png' | wc -l)" -eq 2921 ]; \
	mkdir -p "data/tum/$(SEQ)"; \
	cp -n "$$work_dir/rgb/"*.png "data/tum/$(SEQ)/"; \
	echo 'Saved 2921 RGB frames to data/tum/$(SEQ)'

sample:
	@mkdir -p data
	ffmpeg -v error -y -framerate "$(FPS)" -pattern_type glob -i 'data/tum/$(SEQ)/*.png' \
	  -t "$(DURATION)" -an -c:v libx264rgb -crf 0 -preset fast -movflags +faststart data/sample.partial.mp4
	mv data/sample.partial.mp4 data/sample.mp4
	@echo 'Saved sample: data/sample.mp4'

build:
	cmake -S . -B "$(BUILD_DIR)" -DCMAKE_BUILD_TYPE=Release -DAXRUNTIME_ROOT="$(AXRUNTIME_ROOT)"
	cmake --build "$(BUILD_DIR)" --target lines_metis --parallel "$(JOBS)"

video profile fancy:
	@test -f "$(VIDEO)" || { echo 'Video missing. Run make data && make sample, or set VIDEO=/path/to/video.mp4' >&2; exit 2; }
	+$(MAKE) build
	"$(BUILD_DIR)/cpp/lines_metis" --video "$(VIDEO)" --config "$(CONFIG)" \
	  $(if $(filter profile fancy,$@),--profile) $(if $(filter video,$@),--output "$(OUTPUT)") $(if $(filter fancy,$@),--detections "$(DETECTIONS)")
	$(if $(filter fancy,$@),$(MAKE) render,@:)

render:
	"$(PYTHON)" python/visualize_results.py --results "$(DETECTIONS)" --output "$(SHOWCASE)" \
	  --preview "$(basename $(SHOWCASE)).png" --overwrite

gif:
	@mkdir -p "$(dir $(GIF))"
	ffmpeg -v error -y -ss 0 -t 6 -i "$(SHOWCASE)" \
	  -filter_complex '[0:v]fps=10,scale=960:-1:flags=lanczos,split[a][b];[a]palettegen=max_colors=128[p];[b][p]paletteuse=dither=bayer:bayer_scale=3' \
	  -loop 0 "$(GIF)"
	@echo 'Saved README GIF: $(GIF)'

export:
	"$(PYTHON)" python/export_onnx.py

compile:
	source "$(SDK)/venv/bin/activate" && "$(SDK_PYTHON)" python/compile.py --seq "$(SEQ)" --images "$(CALIB_IMAGES)"
