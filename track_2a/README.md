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

The dev set is also where we developed, so this score is optimistic. On 405 instructions of our own (the same
photos, new wording, one group per capability, written after development) 387 are right (96 %); the weakest groups
are absent colours ("the yellow cube" when there is none, 75 %) and objects lying on top of another (78 %).

The original challenge README (task, data, scoring, rules) is in [`docs/CHALLENGE.md`](docs/CHALLENGE.md).

**Hack Apertus submission (Track 2A, ZHAW).** The project root is `track_2a/`; run everything from there. `make run` builds the
Docker image and runs the pipeline on `data/heldout/` (`make run DATA=data/dataset OUT=results.json` for the dev set). The
technical report is [`technical_report.md`](technical_report.md); the report on Apertus's difficulties is
[`docs/report.pdf`](docs/report.pdf), the slides are [`docs/slides.pdf`](docs/slides.pdf). Our answers are
[`results.json`](results.json) (dev) and [`results_heldout.json`](results_heldout.json) (held-out).

## Run it

One command runs the whole pipeline on any folder of scenes (one subfolder per scene with `rgb.png` and
`instructions.txt`; `calibration.json` in the scene folder or its parent) and writes the results file:

```bash
python src/run.py --data data/heldout --out results_heldout.json
python src/run.py --data data/dataset --out results.json
python src/starter/evaluate.py results.json --data data/dataset      # score (only where ground truth exists)
```

`src/run.py` runs perception, releases the GPU, starts its own Apertus server (vLLM), answers every instruction,
stops the server and prints checks that need no ground truth. Next to the results file it writes
`<out>.trace.jsonl` (every Apertus reply and every correction) and `<out>.server.log`.

### With Docker (recommended)

Needs an NVIDIA GPU with ≥ 24 GB, an NVIDIA driver ≥ 580 (CUDA 13), and the NVIDIA container runtime.

```bash
docker pull ghcr.io/swiss-ai/vllm_apertus_1.5_release@sha256:634973c5d8a2bd542ae9f7b95bf7a271858a842cdc873b2214671079dd9c1ea9
docker build -t claw-and-order .
docker run --rm --gpus all --ipc=host \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    -v "$PWD":/work -w /work \
    claw-and-order --data data/heldout --out results_heldout.json
```

The image is the official swiss-ai image for Apertus v1.5 (the vLLM and transformers builds that the model card
names), pinned by digest, plus our code and OpenCV. The base image is an 8.9 GB download, about 4 minutes on our
connection: like the model weights, pull it once in advance (first line above). After that the build takes
seconds. Data and results stay on the host (`/work` is the current folder).

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

The three vision models are pinned to these revisions in `src/pipeline/config.py`.

### Without Docker

Apertus v1.5 needs forks of vLLM and transformers that are not released yet; the install follows the
official image step by step (Python 3.12, torch 2.11.0+cu130, `swiss-ai/vllm@a601a9d` with the precompiled
wheel, `swiss-ai/transformers@3797303`, `tokenizers<=0.23.0`, `opencv-python-headless`). See the
`Dockerfile` of [swiss-ai/model-launch](https://github.com/swiss-ai/model-launch/blob/main/images/vllm_apertus_1.5_release/Dockerfile).
With such an environment active, `python src/run.py ...` works as above.

### Options

| Option | Use |
|---|---|
| `--dtype auto\|bfloat16\|float32` | Apertus weights. `auto`: bfloat16, or float32 on GPUs without bfloat16 support (compute capability < 8.0, e.g. Quadro RTX 8000). float16 is not offered: Apertus overflows in float16 and answers nothing. |
| `--workers N` | send N instructions to Apertus at once (default 1). Answering is about 5.6 times faster with 8, but batching changes some of Apertus's decisions (dev: 61.8 instead of 64.1), so the submitted results use 1. |
| `--base-url URL` | use an Apertus server that is already running instead of starting one |
| `--scene-files DIR` | read ready-made scene files instead of running perception |
| `--cache DIR`, `--no-cache` | perception results are cached per photo and code version (default `cache/`) |
| `--limit N`, `--only ID,ID` | run a few instructions only, for a quick check |

If the Apertus server starts but cannot answer a test request, or every request fails for three instructions
in a row, `src/run.py` stops with an error instead of writing a file of fallback answers.

## Hardware and time

Measured from a fresh clone with the Docker commands above on an RTX 5090 Laptop GPU (24 GB, bfloat16),
perception from scratch:

| Step | held-out (5 scenes, 40 instructions) | dev (10 scenes, 50 instructions) |
|---|---|---|
| pull the base image (8.9 GB) | 238 s, once | — |
| `docker build` | 3 s | — |
| perception | 102 s | 180 s |
| Apertus server start | 136 s | 136 s |
| answering | ~512 s | ~460 s |
| **run total** | **752 s (12.5 min)** | 778 s (13.0 min) |

With the base image pulled in advance, a fresh clone gives the held-out results file in 12.6 minutes;
including the pull, about 16.6 minutes. Scoring takes a second.

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
| Perception | `src/pipeline/perception.py` | Grounding DINO base, SAM 2.1 large, SigLIP 2 | finds every object with generic prompts ("object", "toy", "part", …) on the whole image and enlarged tiles, segments it, names it from a broad vocabulary, measures its colour from the pixels; finds the two trays and a hand. It never reads the instruction. |
| Geometry | `src/pipeline/geometry.py` | none | turns masks into facts on the table plane: width in mm (can the 85 mm gripper hold it?), what lies on what, which tray, held by a hand, order from the left, nearest neighbours, safe pick point (deepest point inside the mask). |
| Reasoning | `src/pipeline/reasoning.py` | **Apertus v1.5 8B** | reads the instruction and the scene text and returns one JSON reply, constrained to a schema whose ids are only those of this scene: the kind of answer (plan, ask, refuse), the object meant and why, and the ordered actions. |
| Checks | `src/pipeline/rules.py` | none | checks the reply against the measured facts (an object lies on the one to pick, the object is too wide for the gripper or held by a hand, the chosen object is not the kind the instruction names, "closest"/"left of" does not match the measured relation, the plan moves something the instruction forbids, …) and returns a sentence Apertus can act on. Apertus gets up to two corrections; only if it still fails does code build a fallback from the same checks. |

Apertus makes every decision: which object the words refer to, whether to plan, ask or refuse, and the plan.
The code only measures facts and checks replies; it never picks an object by itself while Apertus can still
answer. Every threshold is in `src/pipeline/config.py` with the reason for its value.

**Why not let Apertus look at the photo?** We tried first: given the image and asked for the pixel, Apertus
hit the right object in 9 of 40 instructions (dev score 9.9 / 65). Three of its pixels lay outside the image it
saw, so it probably answers on a 0–1000 scale that its model card does not document; we read its pixels that way.
Still far from usable. It sees roughly what is on the table, but
names gears "sun-shaped objects", places things vaguely and misses objects lying on others. With the same
model reading our measured scene list, the pipeline scores 64.1 / 65.

## Repository

| Path | What |
|---|---|
| `src/run.py` | the one command |
| `src/pipeline/` | perception, geometry, reasoning, rules, config |
| `Makefile`, `Dockerfile` | `make run` and the image above |
| `technical_report.md` | the technical report (Hack Apertus template) |
| `src/starter/`, `src/schema/`, `src/baseline/`, `src/examples/` | from the organisers: scorer, data loader, answer schema, naive baseline |
| `data/dataset/`, `data/heldout/` | public scenes with ground truth, held-out scenes without |
| `docs/` | our report and slides; the challenge README, evaluation rules, data description, safety primer |
| `results.json`, `results_heldout.json` | our answers, dev and held-out |

## License

Code: Apache 2.0 ([`LICENSE`](../LICENSE)). Dataset: CC BY 4.0 (from the organisers).
