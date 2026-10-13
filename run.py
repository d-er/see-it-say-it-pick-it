#!/usr/bin/env python3
"""Run the pipeline on a folder of scenes and write the results file.

    python run.py --data dataset --out results.json
    python run.py --data heldout --out results_heldout.json

Per scene: perception (image -> scene file) -> geometry (measured facts) -> Apertus (one checked
reply per instruction). The Apertus server is started and stopped here; pass --base-url to use one
that is already running. --scene-files reads ready-made scene files instead of running perception.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO / "starter"))
from scene_io import load_scene, pixel_to_table  # noqa: E402

from pipeline import config, geometry, perception, reasoning, rules  # noqa: E402
from pipeline.apertus import ApertusServer  # noqa: E402
from pipeline.scene import SceneFile  # noqa: E402


def prepare(folder: Path, scene_files: Path | None, only: set = frozenset()):
    """Everything about one scene that does not depend on the instruction."""
    scene = load_scene(folder)
    if scene_files is not None:
        scene_file = SceneFile.load(scene_files / f"{folder.name}.json")
    else:
        scene_file = perception.perceive(folder.name, scene.rgb)
    def to_mm(u, v):
        return tuple(pixel_to_table(scene, u, v)[:2])

    gripper = scene.gripper_max_opening_mm or config.GRIPPER_MAX_OPENING_MM
    geometry.measure(scene_file, to_mm, gripper)
    relations = geometry.relations(scene_file, to_mm)
    instructions = [(iid, text) for iid, _category, text in scene.instructions]   # the category is a label: not used
    if only:
        instructions = [i for i in instructions if i[0] in only]
    return scene_file, reasoning.describe(scene_file, relations, gripper), gripper, instructions


def safe_answer(instruction_id: str, scene_file: SceneFile, gripper: float) -> dict:
    """Used when an instruction fails outright, so the results file is always complete."""
    return rules.to_answer(rules.fallback(None, scene_file, gripper), scene_file, instruction_id)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, default=Path("dataset"), help="folder with scene_* subfolders")
    ap.add_argument("--out", type=Path, default=Path("results.json"))
    ap.add_argument("--team", default="team")
    ap.add_argument("--scene-files", type=Path, help="folder with <scene>.json scene files; skips perception")
    ap.add_argument("--base-url", help="OpenAI-compatible URL of a running Apertus server; default: start one")
    ap.add_argument("--model", default=config.APERTUS_MODEL)
    ap.add_argument("--trace", type=Path, help="per-instruction trace as JSON lines (default: <out>.trace.jsonl)")
    ap.add_argument("--limit", type=int, default=0, help="only the first N instructions (for a quick check)")
    ap.add_argument("--only", default="", help="comma-separated instruction ids to run (for debugging)")
    args = ap.parse_args()
    trace_path = args.trace or args.out.with_suffix(".trace.jsonl")

    folders = sorted(p for p in args.data.glob("scene_*") if (p / "rgb.png").exists())
    if not folders:
        sys.exit(f"no scene_*/rgb.png under {args.data}")

    start = time.time()
    scenes = []
    for folder in folders:
        scenes.append((folder.name, *prepare(folder, args.scene_files, set(filter(None, args.only.split(","))))))
        print(f"{folder.name}: {len(scenes[-1][1].objects)} objects", flush=True)
    perception.release()   # Apertus needs the GPU memory
    print(f"scenes prepared in {time.time() - start:.0f} s", flush=True)

    results, fallbacks, done = [], 0, 0
    total = sum(len(s[4]) for s in scenes)
    if args.limit:
        total = min(total, args.limit)

    def run(base_url: str) -> None:
        nonlocal fallbacks, done
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

    if args.base_url:
        run(args.base_url)
    else:
        with ApertusServer(args.model, args.out.with_suffix(".server.log")) as server:
            run(server.base_url)

    kinds = [e["actions"][0]["action"] for e in results]
    no_plan = sum(k in ("ask_user", "refuse") for k in kinds)
    print(f"\n{len(results)} answers in {time.time() - start:.0f} s -> {args.out}   trace: {trace_path}")
    print(f"ask_user {kinds.count('ask_user')}, refuse {kinds.count('refuse')}, plans {len(kinds) - no_plan}; "
          f"validator fallbacks: {fallbacks}")


if __name__ == "__main__":
    main()
