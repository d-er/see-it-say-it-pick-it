# See It, Say It, Pick It — team Claw & Order

Daniel Alznauer, Giuseppe Prezzo

Our entry for the Hack Apertus 2026 ZHAW challenge. Given one photo of a UR5 workcell and one English
instruction, the pipeline answers with the pixel of the first object to pick and an ordered pick/place
plan, or asks back (`ask_user`), or refuses (`refuse`).

**Approach in one sentence:** open vision models turn the photo into a list of objects with measured facts
(name, colour, position, size, what lies on what), Apertus reads that list and the instruction and decides
what to do, and plain code checks every reply against the measured scene and sends a specific error back to
Apertus until the answer holds.

| Dev set (10 scenes, 50 instructions), `results.json` | Score |
|---|---|
| Grounding: `target_pixel` on the right object | 34.1 / 35 (39 of 40) |
| Safety: asked or refused right, no doable instruction refused | 30.0 / 30 (10 of 10, 0 of 40) |
| **Script total** | **64.1 / 65** |

The original challenge README (task, data, scoring, rules) is in [`docs/CHALLENGE.md`](docs/CHALLENGE.md).

## Run it

One command runs the whole pipeline on any folder of scenes (one subfolder per scene with `rgb.png` and
`instructions.txt`; `calibration.json` in the scene folder or its parent) and writes the results file:

```bash
python run.py --data heldout --out results_heldout.json
python run.py --data dataset --out results.json
python starter/evaluate.py results.json --data dataset      # score (only where ground truth exists)
```

`run.py` runs perception, releases the GPU, starts its own Apertus server (vLLM), answers every instruction,
stops the server and prints checks that need no ground truth. Next to the results file it writes
`<out>.trace.jsonl` (every Apertus reply and every correction) and `<out>.server.log`.

### With Docker (recommended)

Needs an NVIDIA GPU with ≥ 24 GB, an NVIDIA driver ≥ 580 (CUDA 13), and the NVIDIA container runtime.

```bash
docker build -t claw-and-order .
docker run --rm --gpus all --ipc=host \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    -v "$PWD":/work -w /work \
    claw-and-order --data heldout --out results_heldout.json
```

The image is the official swiss-ai image for Apertus v1.5 (the vLLM and transformers builds that the model card
names), pinned by digest, plus our code and OpenCV. The base image is an 8.9 GB download the first time; after
that the build takes seconds. Data and results stay on the host (`/work` is the current folder).

### Weights

Nothing is downloaded at run time (`HF_HUB_OFFLINE=1` in the image); the weights must be in the Hugging Face
cache that is mounted above. Fetch them once (Apertus is gated: accept its terms on the model page and
`hf auth login` first):

```bash
hf download swiss-ai/Apertus-v1.5-8B                                          # 18 GB; tested: ef4bbc54d6fe
hf download IDEA-Research/grounding-dino-base --revision 12bdfa3120f3e7ec7b434d90674b3396eccf88eb
hf download facebook/sam2.1-hiera-large --revision 665f8e2ad61cf5f53d65644ff27c8ee525124610
hf download google/siglip2-so400m-patch14-384 --revision e8e487298228002f3d8a82e0cd5c8ea9c567f57f
```

The three vision models are pinned to these revisions in `pipeline/config.py`.

### Without Docker

Apertus v1.5 needs forks of vLLM and transformers that are not released yet; the install follows the
official image step by step (Python 3.12, torch 2.11.0+cu130, `swiss-ai/vllm@a601a9d` with the precompiled
wheel, `swiss-ai/transformers@3797303`, `tokenizers<=0.23.0`, `opencv-python-headless`). See the
`Dockerfile` of [swiss-ai/model-launch](https://github.com/swiss-ai/model-launch/blob/main/images/vllm_apertus_1.5_release/Dockerfile).
With such an environment active, `python run.py ...` works as above.

### Options

| Option | Use |
|---|---|
| `--dtype auto\|bfloat16\|float32` | Apertus weights. `auto`: bfloat16, or float32 on GPUs without bfloat16 support (compute capability < 8.0, e.g. Quadro RTX 8000). float16 is not offered: Apertus overflows in float16 and answers nothing. |
| `--workers N` | send N instructions to Apertus at once (default 1). About four times faster with 8, but batching changes some of Apertus's decisions (dev: 61.8 instead of 64.1), so the submitted results use 1. |
| `--base-url URL` | use an Apertus server that is already running instead of starting one |
| `--scene-files DIR` | read ready-made scene files instead of running perception |
| `--cache DIR`, `--no-cache` | perception results are cached per photo and code version (default `cache/`) |
| `--limit N`, `--only ID,ID` | run a few instructions only, for a quick check |

If the Apertus server starts but cannot answer a test request, or every request fails for three instructions
in a row, `run.py` stops with an error instead of writing a file of fallback answers.

## Hardware and time

Measured with the Docker command above on an RTX 5090 Laptop GPU (24 GB, bfloat16), perception from scratch:

| Set | Perception | Apertus server start | Answering | Total |
|---|---|---|---|---|
| held-out (5 scenes, 40 instructions) | 104 s | 138 s | ~530 s | 775 s (12.9 min) |
| dev (10 scenes, 50 instructions) | 180 s | 136 s | ~440 s | 757 s (12.6 min) |

Not included: building the image, which first pulls the 8.9 GB base image.

| | |
|---|---|
| Memory | perception peaks at 13.3 GB, Apertus in bfloat16 takes 17.2 GB; they run one after the other, so 24 GB is enough |
| GPUs without bfloat16 | `--dtype auto` switches to float32 (about 34 GB of weights, so a 48 GB card is needed). **Not tested**: we have no such GPU, and float32 does not fit our 24 GB card. Expect it to be slower than bfloat16. |

We do not claim that the pipeline runs on an 8 GB GPU: we have not tested a quantised Apertus.

## How it works

```
photo ──► perception ──────────────► scene file ──► geometry ──► scene text ──► Apertus ──► rules ──► answer
          Grounding DINO: boxes      objects,        widths in mm,  one line per   reads the     checks the
          SAM 2.1: masks             masks, trays,   stacks, trays, object with    instruction,  reply against
          SigLIP 2: nouns            hand            neighbours,    its facts      decides and   the scene; an
          pixels: colours                            left/right                    plans         error goes back
                                                                                     ▲            to Apertus
                                                                                     └────────────┘ (≤ 2 times)
```

| Step | Module | Model | Does |
|---|---|---|---|
| Perception | `pipeline/perception.py` | Grounding DINO base, SAM 2.1 large, SigLIP 2 | finds every object with generic prompts ("object", "toy", "part", …) on the whole image and enlarged tiles, segments it, names it from a broad vocabulary, measures its colour from the pixels; finds the two trays and a hand. It never reads the instruction. |
| Geometry | `pipeline/geometry.py` | none | turns masks into facts on the table plane: width in mm (can the 85 mm gripper hold it?), what lies on what, which tray, held by a hand, order from the left, nearest neighbours, safe pick point (deepest point inside the mask). |
| Reasoning | `pipeline/reasoning.py` | **Apertus v1.5 8B** | reads the instruction and the scene text and returns one JSON reply, constrained to a schema whose ids are only those of this scene: the kind of answer (plan, ask, refuse), the object meant and why, and the ordered actions. |
| Checks | `pipeline/rules.py` | none | checks the reply against the measured facts (an object lies on the one to pick, the object is too wide for the gripper or held by a hand, the chosen object is not the kind the instruction names, "closest"/"left of" does not match the measured relation, the plan moves something the instruction forbids, …) and returns a sentence Apertus can act on. Apertus gets up to two corrections; only if it still fails does code build a fallback from the same checks. |

Apertus makes every decision: which object the words refer to, whether to plan, ask or refuse, and the plan.
The code only measures facts and checks replies; it never picks an object by itself while Apertus can still
answer. Every threshold is in `pipeline/config.py` with the reason for its value.

**Why not let Apertus look at the photo?** We tried first: given the image and asked for the pixel, Apertus
hit the right object once in 40 instructions (dev score 2.9 / 65). It sees roughly what is on the table, but
names gears "sun-shaped objects", places things vaguely and misses objects lying on others. With the same
model reading our measured scene list, the pipeline scores 64.1 / 65.

## Repository

| Path | What |
|---|---|
| `run.py` | the one command |
| `pipeline/` | perception, geometry, reasoning, rules, config |
| `Dockerfile` | the image above |
| `starter/`, `schema/`, `baseline/`, `examples/` | from the organisers: scorer, data loader, answer schema, naive baseline |
| `dataset/`, `heldout/` | public scenes with ground truth, held-out scenes without |
| `docs/` | the challenge README, data description, safety primer |

## License

Code: Apache 2.0 (`LICENSE`). Dataset: CC BY 4.0 (from the organisers).
