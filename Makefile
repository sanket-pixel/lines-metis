SHELL := /bin/bash
MAKEFLAGS += --no-print-directory
.DEFAULT_GOAL := build
PYTHON ?= /home/sanket-shah/venvs/torch-cpu/bin/python
SDK ?= /home/sanket-shah/Desktop/Projects/axelera/voyager-sdk
BACKEND ?= torch
SEQ ?= freiburg2_pioneer_slam
FRAMES ?= 100
START ?= 0
FPS ?= 30
THREADS ?= 4
CALIB_IMAGES ?= 10
BUILD_DIR ?= build
JOBS ?= 4
CONFIG ?= configs/default.yaml
VIDEO ?=
AXRUNTIME_ROOT ?= /opt/axelera

.PHONY: build video profile infer export compile

build:
	cmake -S . -B "$(BUILD_DIR)" -DCMAKE_BUILD_TYPE=Release -DAXRUNTIME_ROOT="$(AXRUNTIME_ROOT)"
	cmake --build "$(BUILD_DIR)" --target lines_metis --parallel "$(JOBS)"

video profile:
	@if [ -z "$(VIDEO)" ]; then echo 'Usage: make $@ VIDEO=/path/to/video.mp4 [CONFIG=configs/default.yaml]' >&2; exit 2; fi
	@if [ ! -f "$(VIDEO)" ]; then echo 'Video file not found' >&2; exit 2; fi
	+$(MAKE) build
	"$(BUILD_DIR)/cpp/lines_metis" --video "$(VIDEO)" --config "$(CONFIG)" $(if $(filter profile,$@),--profile)

infer:
	@if [ "$(BACKEND)" = aipu ]; then \
		source "$(SDK)/venv/bin/activate" && "$(SDK)/venv/bin/python" python/infer.py --backend aipu --seq "$(SEQ)" --frames $(FRAMES) --start $(START) --fps $(FPS) --threads $(THREADS); \
	else \
		"$(PYTHON)" python/infer.py --backend "$(BACKEND)" --seq "$(SEQ)" --frames $(FRAMES) --start $(START) --fps $(FPS) --threads $(THREADS); \
	fi

export:
	"$(PYTHON)" python/export_onnx.py

compile:
	source "$(SDK)/venv/bin/activate" && "$(SDK)/venv/bin/python" python/compile.py --seq "$(SEQ)" --images $(CALIB_IMAGES)
