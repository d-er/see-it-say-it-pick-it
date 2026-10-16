# The whole pipeline in one image: perception (Grounding DINO, SAM 2.1, SigLIP 2) and Apertus v1.5 8B on vLLM.
#
#   docker build -t claw-and-order .
#   docker run --rm --gpus all --ipc=host \
#       -v ~/.cache/huggingface:/root/.cache/huggingface \
#       -v "$PWD":/work -w /work \
#       claw-and-order --data data/heldout --out results_heldout.json
#
# Base: the official swiss-ai image for Apertus v1.5 (the vLLM and transformers fork commits that the model
# card names: vLLM a601a9d, transformers 3797303, torch 2.11.0+cu130), pinned by digest. Needs an NVIDIA
# driver >= 580 on the host (CUDA 13) and the NVIDIA container runtime.
# The weights are not in the image: they are read from the mounted Hugging Face cache (see README).
FROM ghcr.io/swiss-ai/vllm_apertus_1.5_release@sha256:634973c5d8a2bd542ae9f7b95bf7a271858a842cdc873b2214671079dd9c1ea9

# The only package the pipeline needs beyond the base image (same version as in the tested environment).
RUN uv pip install --python /opt/venv/bin/python --no-cache opencv-python-headless==5.0.0.93

WORKDIR /app
COPY src/run.py ./
COPY src/pipeline ./pipeline
COPY src/starter ./starter
COPY src/schema ./schema

# Weights come from the mounted cache only; nothing is downloaded at run time.
ENV HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    PYTHONUNBUFFERED=1

# Paths given to run.py (--data, --out) are relative to the working directory, /work in the command above.
ENTRYPOINT ["python", "/app/run.py"]
CMD ["--help"]
