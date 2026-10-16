.PHONY: run build score

# Hack Apertus entry point: builds the image and runs the whole pipeline in Docker on a folder of scenes.
#   make run                                  # held-out scenes -> results_heldout.json
#   make run DATA=data/dataset OUT=results.json    # dev scenes
# Needs the NVIDIA container runtime and the weights in the Hugging Face cache (see README, "Weights").
DATA ?= data/heldout
OUT ?= results_heldout.json
HF_CACHE ?= $(HOME)/.cache/huggingface
IMAGE ?= claw-and-order

build:
	docker build -t $(IMAGE) .

run: build
	docker run --rm --gpus all --ipc=host \
		-v $(HF_CACHE):/root/.cache/huggingface \
		-v "$(CURDIR)":/work -w /work \
		$(IMAGE) --data $(DATA) --out $(OUT)

# Score a results file against the dev ground truth (no GPU needed).
score:
	python src/starter/evaluate.py results.json --data data/dataset
