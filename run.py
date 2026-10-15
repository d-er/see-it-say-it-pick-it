#!/usr/bin/env python3
"""Run the pipeline on a folder of scenes and write the results file.

    python run.py --data dataset --out results.json
    python run.py --data heldout --out results_heldout.json

Per scene: perception (image -> scene file) -> geometry (measured facts) -> Apertus (one checked
reply per instruction). The Apertus server is started and stopped here; pass --base-url to use one
that is already running. --scene-files reads ready-made scene files instead of running perception.
Perception results are kept in --cache and reused while the photo and the perception code are unchanged.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO / "starter"))
from evaluate import answer_type  # noqa: E402
from scene_io import load_scene, pixel_to_table  # noqa: E402

from pipeline import config, geometry, perception, reasoning, rules  # noqa: E402
from pipeline.apertus import ApertusServer  # noqa: E402
from pipeline.scene import SceneFile  # noqa: E402

# Instructions in a row for which every request to Apertus failed (HTTP error, no reply) before the run stops:
# a broken server would otherwise turn the whole results file into fallback answers without a word.
SERVER_FAILURE_STOP = 3


def _perception_key(image: Path) -> str:
    """Changes when the photo or anything perception is made of changes."""
    h = hashlib.sha1(image.read_bytes())
    for name in ("perception.py", "vocabulary.py", "config.py", "scene.py"):
        h.update((REPO / "pipeline" / name).read_bytes())
    return h.hexdigest()[:12]


def prepare(folder: Path, scene_files: Path | None, cache: Path | None, only: set = frozenset()):
    """Everything about one scene that does not depend on the instruction."""
    scene = load_scene(folder)
    if scene_files is not None:
        scene_file = SceneFile.load(scene_files / f"{folder.name}.json")
    else:
        cached = cache / f"{folder.name}.{_perception_key(folder / 'rgb.png')}.json" if cache else None
        if cached and cached.exists():
            scene_file = SceneFile.load(cached)
        else:
            scene_file = perception.perceive(folder.name, scene.rgb)
            if cached:
                cache.mkdir(parents=True, exist_ok=True)
                scene_file.save(cached)
    def to_mm(u, v):
        return tuple(pixel_to_table(scene, u, v)[:2])

    gripper = scene.gripper_max_opening_mm or config.GRIPPER_MAX_OPENING_MM
    geometry.measure(scene_file, to_mm, gripper)
    relations = scene_file.relations = geometry.relations(scene_file, to_mm)
    instructions = [(iid, text) for iid, _category, text in scene.instructions]   # the category is a label: not used
    if only:
        instructions = [i for i in instructions if i[0] in only]
    return scene_file, reasoning.describe(scene_file, relations, gripper), gripper, instructions


def safe_answer(instruction_id: str, scene_file: SceneFile, gripper: float) -> dict:
    """Used when an instruction fails outright, so the results file is always complete."""
    return rules.to_answer(rules.fallback(None, scene_file, gripper), scene_file, instruction_id)


def check(scenes: list, results: list) -> list:
    """Checks that need no ground truth, so they also hold on scenes without answers. Returns warnings."""
    warnings = []
    for name, scene_file, _text, gripper, instructions in scenes:
        if len(scene_file.trays) != 2:
            warnings.append(f"{name}: {len(scene_file.trays)} trays found, expected 2")
        if not 1 <= len(scene_file.objects) <= 25:
            warnings.append(f"{name}: {len(scene_file.objects)} objects found")
    by_id = {}
    for entry in results:
        by_id.setdefault(entry["instruction_id"], []).append(entry)
    lookup = {iid: s for s in scenes for iid, _ in s[4]}
    for iid, (name, scene_file, _text, gripper, _instructions) in lookup.items():
        entries = by_id.get(iid, [])
        if len(entries) != 1:
            warnings.append(f"{iid}: {len(entries)} answers")
            continue
        entry = entries[0]
        if answer_type(entry)[0] is None:
            warnings.append(f"{iid}: answer does not match the result format: {answer_type(entry)[1]}")
            continue
        picks = [a["object"] for a in entry["actions"] if a["action"] == "pick"]
        if not picks:
            continue
        w, h = scene_file.image_size
        u, v = entry["target_pixel"]
        first = scene_file.get(picks[0])
        if not (0 <= u < w and 0 <= v < h and first is not None and first.mask[v, u]):
            warnings.append(f"{iid}: target_pixel {entry['target_pixel']} is not on {picks[0]}")
        moved = set()
        for a in entry["actions"]:
            o = scene_file.get(a["object"])
            if a["action"] == "pick":
                if o is None or not geometry.graspable(o, gripper) or o.held_by_hand:
                    warnings.append(f"{iid}: picks {a['object']}, which cannot be picked")
                elif set(o.under) - moved:
                    warnings.append(f"{iid}: picks {a['object']} while {', '.join(set(o.under) - moved)} lies on it")
                moved.add(a["object"])
    kinds = [e["actions"][0]["action"] for e in results]
    share = sum(k in ("ask_user", "refuse") for k in kinds) / max(len(kinds), 1)
    if share > config.NO_PLAN_ALARM_SHARE:
        warnings.append(f"{share:.0%} of the answers are ask_user or refuse: more than expected, check the scene files")
    return warnings


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, default=Path("dataset"), help="folder with one subfolder per scene (rgb.png + instructions.txt)")
    ap.add_argument("--out", type=Path, default=Path("results.json"))
    ap.add_argument("--team", default="Claw & Order")
    ap.add_argument("--scene-files", type=Path, help="folder with <scene>.json scene files; skips perception")
    ap.add_argument("--cache", type=Path, default=REPO / "cache", help="folder for perception results")
    ap.add_argument("--no-cache", action="store_true", help="run perception even if a cached result exists")
    ap.add_argument("--base-url", help="OpenAI-compatible URL of a running Apertus server; default: start one")
    ap.add_argument("--model", default=config.APERTUS_MODEL)
    ap.add_argument("--dtype", default="auto", choices=["auto", "bfloat16", "float32"],
                    help="dtype of the Apertus weights; auto: bfloat16, or float32 on GPUs without bfloat16 (Turing)")
    ap.add_argument("--trace", type=Path, help="per-instruction trace as JSON lines (default: <out>.trace.jsonl)")
    ap.add_argument("--limit", type=int, default=0, help="only the first N instructions (for a quick check)")
    ap.add_argument("--only", default="", help="comma-separated instruction ids to run (for debugging)")
    args = ap.parse_args()
    trace_path = args.trace or args.out.with_suffix(".trace.jsonl")

    folders = sorted(p for p in args.data.iterdir() if (p / "rgb.png").exists())
    if not folders:
        sys.exit(f"no <scene>/rgb.png under {args.data}")

    start = time.time()
    scenes = []
    for folder in folders:
        scenes.append((folder.name, *prepare(folder, args.scene_files, None if args.no_cache else args.cache,
                                             set(filter(None, args.only.split(","))))))
        print(f"{folder.name}: {len(scenes[-1][1].objects)} objects", flush=True)
    perception.release()   # Apertus needs the GPU memory
    print(f"scenes prepared in {time.time() - start:.0f} s", flush=True)

    results, fallbacks, done = [], 0, 0
    total = sum(len(s[4]) for s in scenes)
    if args.limit:
        total = min(total, args.limit)

    def run(base_url: str) -> None:
        nonlocal fallbacks, done
        failed_in_a_row = 0
        with trace_path.open("w") as trace_file:
            for name, scene_file, scene_text, gripper, instructions in scenes:
                for instruction_id, text in instructions:
                    if args.limit and done >= args.limit:
                        return
                    t = time.time()
                    try:
                        reply, trace = reasoning.answer(text, scene_file, scene_text, base_url, args.model, gripper)
                        entry = rules.to_answer(reply, scene_file, instruction_id)
                    except Exception as e:   # one bad instruction must never lose the results file
                        trace = [{"error": f"{type(e).__name__}: {e}"}, {"final": "fallback"}]
                        entry = safe_answer(instruction_id, scene_file, gripper)
                    fallbacks += trace[-1].get("final") == "fallback"
                    results.append(entry)
                    done += 1
                    args.out.write_text(json.dumps({"team": args.team, "results": results}, indent=1))
                    trace_file.write(json.dumps({"instruction_id": instruction_id, "instruction": text, "scene": scene_text,
                                                 "trace": trace, "answer": entry, "seconds": round(time.time() - t, 2)}) + "\n")
                    trace_file.flush()
                    first = entry["actions"][0]
                    print(f"[{done}/{total}] {instruction_id} {text!r} -> " + ", ".join(
                        f"{a['action']}({a.get('object', '')})" if "object" in a else a["action"] for a in entry["actions"])
                        + (f"  [{first.get('question') or first.get('reason')}]" if "object" not in first else "")
                        + ("  (fallback)" if trace[-1].get("final") == "fallback" else ""), flush=True)
                    attempts = [s for s in trace if "attempt" in s or "error" in s]
                    failed_in_a_row = failed_in_a_row + 1 if attempts and all("error" in s for s in attempts) else 0
                    if failed_in_a_row >= SERVER_FAILURE_STOP:
                        sys.exit(f"STOP: every request to Apertus failed for {failed_in_a_row} instructions in a row "
                                 f"(last: {attempts[-1]['error'][:300]}). The results file is incomplete. Check the "
                                 f"server log; on a GPU without bfloat16 run with --dtype float32.")

    if args.base_url:
        run(args.base_url)
    else:
        with ApertusServer(args.model, args.out.with_suffix(".server.log"), args.dtype) as server:
            run(server.base_url)

    kinds = [e["actions"][0]["action"] for e in results]
    no_plan = sum(k in ("ask_user", "refuse") for k in kinds)
    print(f"\n{len(results)} answers in {time.time() - start:.0f} s -> {args.out}   trace: {trace_path}")
    print(f"ask_user {kinds.count('ask_user')}, refuse {kinds.count('refuse')}, plans {len(kinds) - no_plan}; "
          f"validator fallbacks: {fallbacks}")
    if not args.limit:
        warnings = check(scenes, results)
        print(f"checks without ground truth: {len(warnings)} warning{'' if len(warnings) == 1 else 's'}")
        for w in warnings:
            print(f"  ! {w}")


if __name__ == "__main__":
    main()
