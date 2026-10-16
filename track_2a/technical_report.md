# Technical report — See It, Say It, Pick It (team Claw & Order)

Daniel Alznauer, Giuseppe Prezzo · Hack Apertus 2026, Track 2A, ZHAW challenge "See It, Say It, Pick It:
Vision-Language Grounding for a Real Robot Arm"

The two-page report on where Apertus had difficulty is [`docs/report.pdf`](docs/report.pdf), the five slides are
[`docs/slides.pdf`](docs/slides.pdf). This file follows the Hack Apertus template.

## 1. Summary

Given one photo of a UR5 workcell and one English instruction, the pipeline returns the pixel of the first object
to pick and an ordered pick/place plan, or asks back (`ask_user`), or refuses (`refuse`). Open vision models turn
the photo into a list of objects with measured facts (name, colour, position, width in mm, what lies on what, which
tray, a hand). Apertus v1.5 8B reads that list and the instruction and decides what to do. Plain code checks every
reply against the measured scene and sends a specific error back to Apertus, at most twice.

Dev set (10 scenes, 50 instructions): **64.1 / 65** from the official scorer (grounding 34.1 / 35, safety 30.0 / 30).

## 2. Architecture

```
photo ──► perception ──► scene file ──► geometry ──► scene text ──► Apertus ──► rules ──► answer
                                                                       ▲           │
                                                                       └───────────┘ correction (≤ 2 times)
```

| Step | Module | Model | Does |
|---|---|---|---|
| Perception | `src/pipeline/perception.py` | Grounding DINO base, SAM 2.1 large, SigLIP 2 | finds, segments and names every object with generic prompts; never reads the instruction |
| Geometry | `src/pipeline/geometry.py` | none | widths in mm, stacks, trays, hand, left/right order, neighbours, safe pick point |
| Reasoning | `src/pipeline/reasoning.py` | **Apertus v1.5 8B** | plan / ask / refuse, which object is meant and why, the ordered actions (JSON schema with this scene's ids only) |
| Checks | `src/pipeline/rules.py` | none | checks the reply against the facts and returns a sentence Apertus can act on; code builds a fallback only if Apertus still fails after two corrections |

`src/run.py` runs perception, frees the GPU, starts its own vLLM server for Apertus, answers every instruction one at a
time, and stops the server. Every threshold is in `src/pipeline/config.py` with the reason for its value.

## 3. Use of Apertus

Apertus makes every decision: which object the words refer to, whether to plan, ask or refuse, and the plan. It runs
in bfloat16, temperature 0, with JSON-schema constrained output. The code only measures facts and checks replies; it
never picks an object by itself while Apertus can still answer.

Other open-weight models and their role: **Grounding DINO** (object boxes), **SAM 2.1** (masks), **SigLIP 2** (nouns
for the objects). They do perception only and never see the instruction. No proprietary model or external API is used.

Apertus on its own, given the photo: it hit the right object in 1 of 40 instructions (2.9 / 65 as scored; probably
a 0–1000 coordinate scale, read that way 9 of 40, about 9.9 / 65). With the measured scene list: 64.1 / 65. This is
the "vision model + Apertus" against "Apertus alone" comparison the challenge asks for.

The template's `LLM_NAME`, `LLM_BASE_URL` and `LLM_API_KEY` are not needed: Apertus is served inside the container
from the mounted Hugging Face cache, and no API key is used. `src/run.py --base-url URL` uses an Apertus server that is
already running instead.

## 4. Data

We use the organisers' data only: `data/dataset/` (10 public scenes, 50 instructions, with ground truth) and `data/heldout/`
(5 scenes, 40 instructions, without answers), CC BY 4.0. No new dataset was created and nothing was trained: the
challenge is zero-shot. The ground truth and the category column of `instructions.txt` are never fed into the pipeline.

## 5. Evaluation

| Dev set, `results.json`, `src/starter/evaluate.py` | Score |
|---|---|
| Grounding: `target_pixel` on the right object | 34.1 / 35 (39 of 40) |
| Safety: asked or refused right, no doable instruction refused | 30.0 / 30 (10 of 10, 0 of 40) |
| **Script total** | **64.1 / 65** |

The dev set is where we developed, so this score is optimistic. On 405 instructions of our own (same photos, new
wording, one group per capability, written after development) 387 are right (96 %). The weakest groups are absent
colours ("the yellow cube" when there is none, 75 %) and objects lying on top of another (78 %). Changes were
judged against a noise floor (rewording the prompt without changing its meaning flips 22 of 405 answers) with paired
tests and a rule fixed before each run. Of eleven changes judged that way, one was accepted.

The held-out answers are in `results_heldout.json`, produced with `make run`.

## 6. Limitations

- Apertus cannot ground objects in the photo itself; every number it uses comes from perception and geometry.
- Only 39 % of first replies pass the checks; the correction loop is what makes plans physically possible.
- Absent objects: Apertus tends to pick by colour over noun ("yellow cube" → yellow gear).
- Answers are sensitive to wording and serving mode: batching 8 requests is 5.6 times faster but changes 3 of 50
  dev decisions (64.1 → 61.8), so the results use one request at a time.
- Widths come from one 2D photo (no depth); perspective enlarges them, so the 85 mm gripper limit uses a factor of 1.3.
- Not tested on GPUs without bfloat16: there Apertus must run in float32 (about 34 GB of weights).

Details and numbers: [`docs/report.pdf`](docs/report.pdf).

## 7. Reproducibility

Requirements: NVIDIA GPU with ≥ 24 GB (bfloat16; float32 needs about 48 GB), NVIDIA driver ≥ 580 (CUDA 13), the
NVIDIA container runtime, and the model weights in the Hugging Face cache (Apertus is gated: accept its terms and
`hf auth login` first). No API keys.

```bash
docker pull ghcr.io/swiss-ai/vllm_apertus_1.5_release@sha256:634973c5d8a2bd542ae9f7b95bf7a271858a842cdc873b2214671079dd9c1ea9
hf download swiss-ai/Apertus-v1.5-8B
hf download IDEA-Research/grounding-dino-base --revision 12bdfa3120f3e7ec7b434d90674b3396eccf88eb
hf download facebook/sam2.1-hiera-large --revision 665f8e2ad61cf5f53d65644ff27c8ee525124610
hf download google/siglip2-so400m-patch14-384 --revision e8e487298228002f3d8a82e0cd5c8ea9c567f57f

make run                                  # data/heldout/ -> results_heldout.json
make run DATA=data/dataset OUT=results.json    # dev set
make score                                # official scorer on results.json
```

`make run` builds the image from the `Dockerfile` (official swiss-ai Apertus v1.5 image, pinned by digest, plus
our code and OpenCV) and runs the pipeline in it. Nothing is downloaded at run time. On an RTX 5090 Laptop GPU
(24 GB) a fresh clone gives the held-out results in 12.6 minutes with the base image pulled in advance. The README
has the per-step timings and the install without Docker.

## 8. Next steps

- **Train, don't prompt.** Prompt text did not change Apertus's decisions; weights did. In a first test beyond this
  zero-shot challenge, a LoRA trained on 713 replies the check had accepted raised first-time-right replies from 70
  to 84 of 126 on three unseen photos (McNemar p = 0.0005), with final answers unchanged. Next: train on rejected and
  accepted pairs, weight the safety rows, more photos.
- **Production:** a quantised Apertus with the cell's adapter on an 8 GB GPU next to the robot, if quantisation
  passes the same pre-set rule; plus a safety gate the model cannot override, a depth camera, and a re-check after
  every pick.

## License

Code: Apache 2.0 ([`LICENSE`](../LICENSE)). Data in `data/dataset/` and `data/heldout/`: CC BY 4.0, from the organisers.

## References

- Challenge description: [`docs/CHALLENGE.md`](docs/CHALLENGE.md), [`docs/EVALUATION.md`](docs/EVALUATION.md)
- Apertus v1.5: https://huggingface.co/swiss-ai/Apertus-v1.5-8B
- Grounding DINO: https://huggingface.co/IDEA-Research/grounding-dino-base
- SAM 2.1: https://huggingface.co/facebook/sam2.1-hiera-large
- SigLIP 2: https://huggingface.co/google/siglip2-so400m-patch14-384
- Hack Apertus project template: https://github.com/HackApertus/project-template
