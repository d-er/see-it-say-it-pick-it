"""Instruction + scene facts -> Apertus -> checked reply.

Apertus reads the instruction and a text list of the objects with their measured facts. It names the
action type, splits the referring expression, resolves it to object ids, decides plan / ask_user /
refuse and writes the ordered actions. rules.validate() checks the reply against the scene; a specific error
goes back to Apertus for at most config.MAX_RETRIES corrections.
"""
from __future__ import annotations

import json

from . import config, rules
from .apertus import chat
from .geometry import graspable
from .scene import SceneFile

ACTION_TYPES = ["pick", "pick_and_place", "throw", "push", "pull_out_from_under", "take_from_hand", "other"]
SELECTOR_TYPES = ["none", "order", "closest_to", "farthest_from", "left_of", "right_of", "lying_on", "lying_under", "in_tray", "other"]
ORDER_RANKS = ["none"] + [f"{n} from the {side}" for side in ("left", "right") for n in ("1st", "2nd", "3rd", "4th", "5th")] + ["middle"]
REFUSE_BECAUSE = ["not_refused", "unsafe_motion", "no_matching_object", "cannot_be_picked", "impossible_as_stated"]

SYSTEM = """You plan pick-and-place actions for a robot arm with a two-finger gripper. A camera looks down at a table. Another program has found the objects in the photo and measured them. You get that list and one instruction from a person. You do not see the photo, so rely on the list. The object names are the best guess of an image model: "name uncertain" means the guess was weak and the object may be another kind of thing of that colour; "could also be a ..." gives a second name the model nearly chose.

Left and right are as seen in the photo. The gripper holds one object at a time.

Fill in the answer fields in this order.

action_type: what the person wants the robot to do with the object.
  "pick": grasp it and lift it (pick up, grab, take, give me, hand me).
  "pick_and_place": grasp it and put it down somewhere (put, move, place, bring).
  "throw": throw it, toss it, or drop it from a height.
  "push": push, slide, shove, flick or knock it without grasping it.
  "pull_out_from_under": pull or drag an object out from beneath another object.
  "take_from_hand": take something that a person is holding.
  "other": anything else.
object_words: the words of the instruction that say what kind of object is meant, with its colour ("red gear", "pens", "everything"). Copy them from the instruction.
selector: the words of the instruction that single out which of those objects is meant, by position, order or relation to another object ("on the left", "second from the right", "closest to the bowl", "lying on the bolt"). "" if the instruction has no such words.
selector_type: what kind of selector it is.
  "none": selector is "".
  "order": a place in the row or in the photo ("on the left", "leftmost", "second from the right", "in the middle").
  "closest_to": closest to, nearest to, next to or beside another object.
  "farthest_from": farthest from another object.
  "left_of", "right_of": to the left or to the right of another object.
  "lying_on": lying on, or on top of, another object.
  "lying_under": under or beneath another object.
  "in_tray": inside one of the trays.
  "other": anything else ("with nothing on it", "the big one").
selector_reference_ids: ids of the other objects or of the tray that the selector names ("the bowl" in "closest to the bowl"). Empty for "none", "order" and "other".
order_rank: only for selector_type "order": the place that is meant, counted among the candidates. "on the left", "leftmost" and "the left one" are "1st from the left"; "rightmost" is "1st from the right"; "second from the right" is "2nd from the right"; "in the middle" is "middle". Otherwise "none".
object_kind: the noun in object_words that names the kind of object, in the singular ("gear", "pen", "cube"). Copy the instruction's own noun, also when the list calls the object by another word ("biro" for a pen). "" if the instruction uses no specific noun ("that one", "it", "the object", "the thing", "everything"). Do not guess a kind that the instruction does not say.
quantity: "one" if a single object is meant, "several" if more than one (all, both, everything, or two objects named).
destinations: where the instruction says the objects should be put: tray ids, ids of objects to put them on, or "table" if the instruction itself says table, or "human_hand" if the instruction says to put the object into or onto a person's hand or body. Empty if the instruction only asks to pick ("hand me", "give me" are picks).
forbidding_words: the words of the instruction that forbid moving or touching something ("without moving the block", "do not touch the bowl", "leave it where it is", "except the bowl"). Copy them from the instruction. "" if it has no such words, which is the usual case.
do_not_move_ids: ids of the objects that forbidding_words says must stay where they are. This can be the object to be moved itself, if the instruction also says to leave it where it is. Empty if forbidding_words is "".
candidate_ids: ids of all listed objects whose label and colour fit object_words. An object of another colour or another kind does not fit. If no listed object has the kind of object_kind as its name, an object of the right colour whose name is uncertain, or that could also be that kind, fits. If object_kind is "", every listed object is a candidate. Empty if none fits.
selector_check: if there is a selector, one entry per candidate: its id, the measured fact from the list that decides the selector for it, and whether it fits (true or false). For "closest to X", "farthest from X" and similar, read the distance line of X: the candidate that comes first in it is the closest, the one that comes last is the farthest, and exactly that one fits, however near or far it is. Empty list if there is no selector.
matching_ids: ids of the objects that are meant: the candidates whose selector_check says fits true, or all candidates if there is no selector. If none fits, leave the list empty.
kind:
  "refuse" if one of the four reasons under refuse_because holds.
  "ask_user" if object_kind and selector are both "" and quantity is "one" (the instruction does not say which object), or if quantity is "one" and matching_ids has more than one id.
  "plan" in every other case. Having to move another object out of the way first is normal and no reason to refuse.
refuse_because: "not_refused" unless kind is "refuse". Then one of:
  "unsafe_motion": action_type is throw, push, pull_out_from_under or take_from_hand, or destinations contains "human_hand".
  "no_matching_object": matching_ids is empty although the instruction names an object.
  "cannot_be_picked": an object in matching_ids that would have to be picked is marked "cannot be picked".
  "impossible_as_stated": the instruction contradicts itself, or it forbids moving an object that would have to be moved.
actions: for "plan", the ordered steps, each {"action": "pick", "object": id} or {"action": "place", "object": where}. "where" is one of the destinations, or "table" for an object that is only moved out of the way. If something lies on an object you need, first pick the upper object and place it on "table". Never pick an object while something lies on it. If the instruction only asks to pick an object, the last step is that pick. For "ask_user" and "refuse" the list is empty.
question: for "ask_user", the question to the person, naming the candidates. Otherwise "".
reason: for "refuse", one sentence saying why. Otherwise "".

Example scene:
Objects:
- blue_pen_1: blue pen. On the table. 12 mm wide, can be picked. 1st of 2 blue pens from the left (2nd from the right). Free: nothing lies on it.
- steel_bolt_1: steel bolt. On the table. 20 mm wide, can be picked. Lies under green_block_1.
- green_block_1: green block. On the table. 40 mm wide, can be picked. Lies on top of steel_bolt_1.
- blue_pen_2: blue pen. On the table. 12 mm wide, can be picked. 2nd of 2 blue pens from the left (1st from the right). Free: nothing lies on it.
- white_bowl_1: white bowl. In the black tray. 160 mm wide, too wide for the gripper: cannot be picked, but things can be placed on it. Free: nothing lies on it.
Neighbours in the photo (side by side, this says nothing about what lies on what):
- blue_pen_1: to its right steel_bolt_1.
- steel_bolt_1: to its left blue_pen_1, to its right blue_pen_2.
- green_block_1: to its left blue_pen_1, to its right blue_pen_2.
- blue_pen_2: to its left steel_bolt_1, to its right white_bowl_1.
- white_bowl_1: to its left blue_pen_2.
Distances from each object to the others, nearest first:
- blue_pen_1: steel_bolt_1 (60 mm), green_block_1 (65 mm), blue_pen_2 (150 mm), white_bowl_1 (240 mm).
- steel_bolt_1: green_block_1 (10 mm), blue_pen_1 (60 mm), blue_pen_2 (95 mm), white_bowl_1 (190 mm).
- green_block_1: steel_bolt_1 (10 mm), blue_pen_1 (65 mm), blue_pen_2 (90 mm), white_bowl_1 (185 mm).
- blue_pen_2: green_block_1 (90 mm), steel_bolt_1 (95 mm), white_bowl_1 (100 mm), blue_pen_1 (150 mm).
- white_bowl_1: blue_pen_2 (100 mm), green_block_1 (185 mm), steel_bolt_1 (190 mm), blue_pen_1 (240 mm).
Trays: grey_tray (left) holds nothing. black_tray (right) holds white_bowl_1.
No human hand in the photo.

Example answers for that scene:
"grab the pen nearest to the bowl" -> {"action_type": "pick", "object_words": "pen", "selector": "nearest to the bowl", "selector_type": "closest_to", "selector_reference_ids": ["white_bowl_1"], "order_rank": "none", "object_kind": "pen", "quantity": "one", "destinations": [], "forbidding_words": "", "do_not_move_ids": [], "candidate_ids": ["blue_pen_1", "blue_pen_2"], "selector_check": [{"id": "blue_pen_1", "fact": "240 mm from white_bowl_1", "fits": false}, {"id": "blue_pen_2", "fact": "100 mm from white_bowl_1, the nearest pen", "fits": true}], "matching_ids": ["blue_pen_2"], "kind": "plan", "refuse_because": "not_refused", "actions": [{"action": "pick", "object": "blue_pen_2"}], "question": "", "reason": ""}
"take the thing lying on the bolt" -> {"action_type": "pick", "object_words": "thing", "selector": "lying on the bolt", "selector_type": "lying_on", "selector_reference_ids": ["steel_bolt_1"], "order_rank": "none", "object_kind": "", "quantity": "one", "destinations": [], "forbidding_words": "", "do_not_move_ids": [], "candidate_ids": ["blue_pen_1", "steel_bolt_1", "green_block_1", "blue_pen_2", "white_bowl_1"], "selector_check": [{"id": "blue_pen_1", "fact": "free, lies on nothing", "fits": false}, {"id": "steel_bolt_1", "fact": "lies under green_block_1", "fits": false}, {"id": "green_block_1", "fact": "lies on top of steel_bolt_1", "fits": true}, {"id": "blue_pen_2", "fact": "free, lies on nothing", "fits": false}, {"id": "white_bowl_1", "fact": "free, lies on nothing", "fits": false}], "matching_ids": ["green_block_1"], "kind": "plan", "refuse_because": "not_refused", "actions": [{"action": "pick", "object": "green_block_1"}], "question": "", "reason": ""}
"pick up the pen to the right of the bolt" -> {"action_type": "pick", "object_words": "pen", "selector": "to the right of the bolt", "selector_type": "right_of", "selector_reference_ids": ["steel_bolt_1"], "order_rank": "none", "object_kind": "pen", "quantity": "one", "destinations": [], "forbidding_words": "", "do_not_move_ids": [], "candidate_ids": ["blue_pen_1", "blue_pen_2"], "selector_check": [{"id": "blue_pen_1", "fact": "to the left of steel_bolt_1", "fits": false}, {"id": "blue_pen_2", "fact": "to the right of steel_bolt_1", "fits": true}], "matching_ids": ["blue_pen_2"], "kind": "plan", "refuse_because": "not_refused", "actions": [{"action": "pick", "object": "blue_pen_2"}], "question": "", "reason": ""}
"take the left pen" -> {"action_type": "pick", "object_words": "pen", "selector": "left", "selector_type": "order", "selector_reference_ids": [], "order_rank": "1st from the left", "object_kind": "pen", "quantity": "one", "destinations": [], "forbidding_words": "", "do_not_move_ids": [], "candidate_ids": ["blue_pen_1", "blue_pen_2"], "selector_check": [{"id": "blue_pen_1", "fact": "1st of 2 blue pens from the left", "fits": true}, {"id": "blue_pen_2", "fact": "2nd of 2 blue pens from the left", "fits": false}], "matching_ids": ["blue_pen_1"], "kind": "plan", "refuse_because": "not_refused", "actions": [{"action": "pick", "object": "blue_pen_1"}], "question": "", "reason": ""}
"put the bolt in the grey tray" -> {"action_type": "pick_and_place", "object_words": "bolt", "selector": "", "selector_type": "none", "selector_reference_ids": [], "order_rank": "none", "object_kind": "bolt", "quantity": "one", "destinations": ["grey_tray"], "forbidding_words": "", "do_not_move_ids": [], "candidate_ids": ["steel_bolt_1"], "selector_check": [], "matching_ids": ["steel_bolt_1"], "kind": "plan", "refuse_because": "not_refused", "actions": [{"action": "pick", "object": "green_block_1"}, {"action": "place", "object": "table"}, {"action": "pick", "object": "steel_bolt_1"}, {"action": "place", "object": "grey_tray"}], "question": "", "reason": ""}
"move both pens into the bowl" -> {"action_type": "pick_and_place", "object_words": "both pens", "selector": "", "selector_type": "none", "selector_reference_ids": [], "order_rank": "none", "object_kind": "pen", "quantity": "several", "destinations": ["white_bowl_1"], "forbidding_words": "", "do_not_move_ids": [], "candidate_ids": ["blue_pen_1", "blue_pen_2"], "selector_check": [], "matching_ids": ["blue_pen_1", "blue_pen_2"], "kind": "plan", "refuse_because": "not_refused", "actions": [{"action": "pick", "object": "blue_pen_1"}, {"action": "place", "object": "white_bowl_1"}, {"action": "pick", "object": "blue_pen_2"}, {"action": "place", "object": "white_bowl_1"}], "question": "", "reason": ""}
"hand me the pen" -> {"action_type": "pick", "object_words": "pen", "selector": "", "selector_type": "none", "selector_reference_ids": [], "order_rank": "none", "object_kind": "pen", "quantity": "one", "destinations": [], "forbidding_words": "", "do_not_move_ids": [], "candidate_ids": ["blue_pen_1", "blue_pen_2"], "selector_check": [], "matching_ids": ["blue_pen_1", "blue_pen_2"], "kind": "ask_user", "refuse_because": "not_refused", "actions": [], "question": "Which pen do you mean, blue_pen_1 on the left or blue_pen_2 on the right?", "reason": ""}
"bring me this" -> {"action_type": "pick", "object_words": "this", "selector": "", "selector_type": "none", "selector_reference_ids": [], "order_rank": "none", "object_kind": "", "quantity": "one", "destinations": [], "forbidding_words": "", "do_not_move_ids": [], "candidate_ids": [], "selector_check": [], "matching_ids": [], "kind": "ask_user", "refuse_because": "not_refused", "actions": [], "question": "Which object do you mean?", "reason": ""}
"slide the block over to the bowl" -> {"action_type": "push", "object_words": "block", "selector": "", "selector_type": "none", "selector_reference_ids": [], "order_rank": "none", "object_kind": "block", "quantity": "one", "destinations": ["white_bowl_1"], "forbidding_words": "", "do_not_move_ids": [], "candidate_ids": ["green_block_1"], "selector_check": [], "matching_ids": ["green_block_1"], "kind": "refuse", "refuse_because": "unsafe_motion", "actions": [], "question": "", "reason": "Sliding an object without grasping it is an uncontrolled motion."}
"pick up the green pen" -> {"action_type": "pick", "object_words": "green pen", "selector": "", "selector_type": "none", "selector_reference_ids": [], "order_rank": "none", "object_kind": "pen", "quantity": "one", "destinations": [], "forbidding_words": "", "do_not_move_ids": [], "candidate_ids": [], "selector_check": [], "matching_ids": [], "kind": "refuse", "refuse_because": "no_matching_object", "actions": [], "question": "", "reason": "There is no green pen on the table, only blue pens."}
"put the bowl in the grey tray" -> {"action_type": "pick_and_place", "object_words": "bowl", "selector": "", "selector_type": "none", "selector_reference_ids": [], "order_rank": "none", "object_kind": "bowl", "quantity": "one", "destinations": ["grey_tray"], "forbidding_words": "", "do_not_move_ids": [], "candidate_ids": ["white_bowl_1"], "selector_check": [], "matching_ids": ["white_bowl_1"], "kind": "refuse", "refuse_because": "cannot_be_picked", "actions": [], "question": "", "reason": "The bowl is 160 mm wide, too wide for the gripper."}
"take the bolt but do not touch the block" -> {"action_type": "pick", "object_words": "bolt", "selector": "", "selector_type": "none", "selector_reference_ids": [], "order_rank": "none", "object_kind": "bolt", "quantity": "one", "destinations": [], "forbidding_words": "but do not touch the block", "do_not_move_ids": ["green_block_1"], "candidate_ids": ["steel_bolt_1"], "selector_check": [], "matching_ids": ["steel_bolt_1"], "kind": "refuse", "refuse_because": "impossible_as_stated", "actions": [], "question": "", "reason": "The block lies on the bolt, so the bolt cannot be picked without moving the block."}"""


def _article(noun: str) -> str:
    return f"{'an' if noun[0] in 'aeiou' else 'a'} {noun}"


def _naming(o) -> str:
    """The object's name with what the naming model was unsure about."""
    doubts = []
    if o.other_labels:
        doubts.append("could also be " + " or ".join(_article(n) for n in o.other_labels))
    if o.name_score is not None and o.name_score < config.NAME_SURE_SCORE:
        doubts.append("name uncertain")
    return o.name + (f" ({', '.join(doubts)})" if doubts else "")


def _ordinal(n: int) -> str:
    return f"{n}{'th' if 10 <= n % 100 <= 20 else {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10, 'th')}"


def describe(scene: SceneFile, relations: dict, gripper_mm: float = config.GRIPPER_MAX_OPENING_MM) -> str:
    """The scene as the text list Apertus reads."""
    lines, layout, distances = ["Objects:"], [], []
    for oid in relations["left_to_right"]:
        o, r = scene.get(oid), relations[oid]
        parts = [f"{_naming(o)}.", f"In the {o.in_tray} tray." if o.in_tray else "On the table."]
        if graspable(o, gripper_mm):
            parts.append(f"{o.width_mm:.0f} mm wide, can be picked.")
        else:
            parts.append(f"{o.width_mm:.0f} mm wide, too wide for the gripper: cannot be picked, but things can be placed on it.")
        if o.held_by_hand:
            parts.append("A human hand is holding or touching it: cannot be picked.")
        if r["rank"]:
            k = r["rank"]
            parts.append(f"{_ordinal(k['from_left'])} of {k['of']} {o.name}s from the left ({_ordinal(k['from_right'])} from the right).")
        if o.on_top_of:
            parts.append(f"Lies on top of {' and '.join(o.on_top_of)}.")
        if o.under:
            parts.append(f"Lies under {' and '.join(o.under)}.")
        elif not o.on_top_of:
            parts.append("Free: nothing lies on it.")
        lines.append(f"- {o.id}: " + " ".join(parts))
        sides = [f"{text} {r[key]}" for key, text in (("left_of", "to its left"), ("right_of", "to its right"),
                                                      ("above", "further up"), ("below", "further down")) if r[key]]
        if sides:
            layout.append(f"- {o.id}: " + ", ".join(sides) + ".")
        if r["nearest"]:
            distances.append(f"- {o.id}: " + ", ".join(f"{i} ({d} mm)" for i, d in r["nearest"]) + ".")
    if len(lines) == 1:
        lines.append("(no objects found)")
    if layout:
        lines += ["Neighbours in the photo (side by side, this says nothing about what lies on what):"] + layout
    if distances:
        lines += ["Distances from each object to the others, nearest first:"] + distances
    trays = []
    for t in scene.trays:
        inside = [o.id for o in scene.objects if o.in_tray and t.id.startswith(o.in_tray)]
        side = "left" if sum(p[0] for p in t.outline_px) / len(t.outline_px) < scene.image_size[0] / 2 else "right"
        trays.append(f"{t.id} ({side}) holds {', '.join(inside) if inside else 'nothing'}.")
    if trays:
        lines.append("Trays: " + " ".join(trays))
    lines.append("A human hand is in the photo." if scene.hand_present else "No human hand in the photo.")
    return "\n".join(lines)


def reply_schema(scene: SceneFile) -> dict:
    """JSON schema of one reply. Ids are limited to those of this scene, so Apertus cannot invent one."""
    ids = [o.id for o in scene.objects]
    an_id = {"enum": ids} if ids else {"type": "string"}
    return {
        "type": "object", "additionalProperties": False,
        "required": ["action_type", "object_words", "selector", "selector_type", "selector_reference_ids", "order_rank", "object_kind", "quantity", "destinations", "forbidding_words",
                     "do_not_move_ids", "candidate_ids", "selector_check", "matching_ids", "kind", "refuse_because", "actions", "question", "reason"],
        "properties": {
            "action_type": {"enum": ACTION_TYPES},
            "object_words": {"type": "string"},
            "selector": {"type": "string"},
            "selector_type": {"enum": SELECTOR_TYPES},
            "selector_reference_ids": {"type": "array", "items": {"enum": ids + [t.id for t in scene.trays]} if ids or scene.trays else {"type": "string"}},
            "order_rank": {"enum": ORDER_RANKS},
            "object_kind": {"type": "string"},
            "quantity": {"enum": ["one", "several"]},
            "destinations": {"type": "array", "items": {"enum": ids + rules.destinations(scene)}},
            "forbidding_words": {"type": "string"},
            "do_not_move_ids": {"type": "array", "items": an_id},
            "candidate_ids": {"type": "array", "items": an_id},
            "selector_check": {"type": "array", "items": {
                "type": "object", "additionalProperties": False, "required": ["id", "fact", "fits"],
                "properties": {"id": an_id, "fact": {"type": "string"}, "fits": {"type": "boolean"}}}},
            "matching_ids": {"type": "array", "items": an_id},
            "kind": {"enum": ["plan", "ask_user", "refuse"]},
            "refuse_because": {"enum": REFUSE_BECAUSE},
            "actions": {"type": "array", "items": {
                "type": "object", "additionalProperties": False, "required": ["action", "object"],
                "properties": {"action": {"enum": ["pick", "place"]},
                               "object": {"enum": ids + rules.destinations(scene)}}}},
            "question": {"type": "string"},
            "reason": {"type": "string"},
        },
    }


def answer(instruction: str, scene: SceneFile, scene_text: str, base_url: str, model: str,
           gripper_mm: float = config.GRIPPER_MAX_OPENING_MM) -> tuple[dict, list]:
    """(final reply, trace). The trace lists every attempt with the errors the validator found;
    its last entry says whether the final reply is Apertus's own or the fallback."""
    schema = reply_schema(scene)
    messages = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": f"Scene:\n{scene_text}\n\nInstruction: \"{instruction}\""}]
    trace, reply, doubted = [], None, False
    for attempt in range(config.MAX_RETRIES + 1):
        try:
            raw = chat(base_url, model, messages, schema)
            reply = json.loads(raw)
        except (OSError, ValueError, KeyError) as e:
            trace.append({"attempt": attempt, "error": f"{type(e).__name__}: {e}"})
            continue
        normalised = rules.normalise(reply)
        errors = rules.validate(reply, scene, gripper_mm, instruction)
        if not errors and not doubted:          # asked once: the second refusal stands
            doubt = rules.name_doubt(reply, scene)
            errors, doubted = ([doubt], True) if doubt else (errors, False)
        trace.append({"attempt": attempt, "reply": reply, "errors": errors, **({"normalised": normalised} if normalised else {})})
        if not errors:
            trace.append({"final": "apertus"})
            return reply, trace
        messages += [{"role": "assistant", "content": raw},
                     {"role": "user", "content": "A check of your answer against the scene found a problem:\n"
                                                 + "\n".join(errors) + "\nGive the corrected answer."}]
    reply = rules.fallback(reply, scene, gripper_mm)
    trace.append({"final": "fallback", "reply": reply})
    return reply, trace
