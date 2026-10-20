SHELL := /bin/bash
.DEFAULT_GOAL := infer
PYTHON ?= /home/sanket-shah/venvs/torch-cpu/bin/python
SDK ?= /home/sanket-shah/Desktop/Projects/axelera/voyager-sdk
BACKEND ?= torch
SEQ ?= freiburg2_pioneer_slam
FRAMES ?= 100
START ?= 0
FPS ?= 30
THREADS ?= 4
CALIB_IMAGES ?= 10

.PHONY: infer export compile
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
