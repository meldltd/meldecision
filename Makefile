# Laya in Go: ONNX Runtime + GoFiber.
#
#   make deps              download libonnxruntime + libtokenizers into third_party/
#   make export            export all three checkpoints to models/ (needs python + torch; ~5 GB)
#   make export-english    export one checkpoint
#   make build             build bin/layad
#   make run               run the server on :8080
#   make test              run the Go test suite (golden tests skip if models/ is missing)
#   make goldens           regenerate testdata/*.json from the upstream Python package

SHELL      := /bin/bash
PY         ?= python3
MODELS_DIR ?= models
ADDR       ?= :8080
ENV        := third_party/env.sh

.PHONY: deps export export-english export-multilingual export-typed-decisions build run test goldens clean help

help:
	@sed -n '3,10p' Makefile

third_party/env.sh:
	scripts/fetch_deps.sh

deps: third_party/env.sh

pydeps:
	$(PY) -m pip install -q torch transformers safetensors huggingface_hub onnx onnxscript onnxruntime tokenizers

export: export-english export-multilingual export-typed-decisions

export-english export-multilingual export-typed-decisions: export-%:
	$(PY) export/export_onnx.py $* --out $(MODELS_DIR)

build: deps
	source $(ENV) && go build -o bin/layad ./cmd/layad

run: build
	source $(ENV) && ./bin/layad -addr $(ADDR) -models $(MODELS_DIR)

test: deps
	source $(ENV) && LAYA_MODELS_DIR=$(abspath $(MODELS_DIR)) go test ./... -v -count=1

goldens:
	@test -n "$(LAYA_UPSTREAM)" || { echo "set LAYA_UPSTREAM=/path/to/NandhaKishorM/laya checkout"; exit 1; }
	PYTHONPATH=$(LAYA_UPSTREAM) $(PY) export/make_goldens.py --model english --out testdata/golden_english.json
	PYTHONPATH=$(LAYA_UPSTREAM) $(PY) export/make_goldens.py --model multilingual --out testdata/golden_multilingual.json

clean:
	rm -rf bin
