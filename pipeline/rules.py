"""Checks on what Apertus answered, and the conversion into the result format.

Apertus decides the kind of answer, the object and the plan. The code here only checks that answer
against the measured scene and says what is wrong, so the error can be sent back to Apertus.
Only when Apertus still fails after the retries does fallback() build an answer from the same checks.
"""
from __future__ import annotations

from . import config
from .geometry import graspable
from .scene import SceneFile

UNSAFE_ACTIONS = ("throw", "push", "pull_out_from_under", "take_from_hand")
TABLE = "table"


def destinations(scene: SceneFile) -> list:
    return [t.id for t in scene.trays] + [TABLE]


def _blockers(scene: SceneFile, object_id: str, seen=None) -> list:
    """Everything lying on an object, top first: what has to be moved before it can be picked."""
    seen = seen if seen is not None else set()
    out = []
    for upper in scene.get(object_id).under:
        if upper not in seen:
            seen.add(upper)
            out += _blockers(scene, upper, seen) + [upper]
    return out


def _cannot_pick(scene: SceneFile, object_id: str, gripper_mm: float) -> str | None:
    o = scene.get(object_id)
    if o.held_by_hand:
        return f"{o.id} is held by a human hand, so the robot must not take it"
    if not graspable(o, gripper_mm):
        return f"{o.id} is {o.width_mm:.0f} mm wide, too wide for the {gripper_mm:.0f} mm gripper, so it cannot be picked"
    return None


def _involved(scene: SceneFile, matching: list) -> list:
    """The matching objects plus everything that has to be moved off them, in the order to move them."""
    out = []
    for m in matching:
        for b in _blockers(scene, m) + [m]:
            if b not in out:
                out.append(b)
    return out


def validate(reply: dict, scene: SceneFile, gripper_mm: float = config.GRIPPER_MAX_OPENING_MM) -> list:
    """Problems with a reply, as sentences Apertus can act on. Empty list: the reply stands."""
    kind, action_type, because = reply["kind"], reply["action_type"], reply["refuse_because"]
    matching, frozen = list(dict.fromkeys(reply["matching_ids"])), set(reply["do_not_move_ids"])
    involved = _involved(scene, matching)
    unpickable = [p for p in (_cannot_pick(scene, i, gripper_mm) for i in involved) if p]

    if action_type in UNSAFE_ACTIONS:
        if kind != "refuse":
            return [f'action_type is "{action_type}", an unsafe or uncontrolled motion. kind must be "refuse" '
                    'with refuse_because "unsafe_motion".']
        return []
    if kind == "refuse":
        if not reply["reason"].strip():
            return ['kind is "refuse" but reason is empty.']
        if because == "unsafe_motion":
            return [f'refuse_because is "unsafe_motion", but action_type is "{action_type}", which is a normal grasp. '
                    "Either correct action_type or do not refuse for this reason."]
        if because == "no_matching_object" and matching:
            return [f'refuse_because is "no_matching_object", but matching_ids lists {", ".join(matching)}. '
                    "If these objects fit the instruction, do not refuse for this reason; if they do not fit, remove them."]
        if because == "cannot_be_picked" and not matching:
            return ['refuse_because is "cannot_be_picked", but matching_ids is empty. List the object that cannot be picked '
                    "in matching_ids."]
        if because == "cannot_be_picked" and not unpickable:
            return ['refuse_because is "cannot_be_picked", but every object involved can be picked: '
                    f'{", ".join(involved) or "none listed"}. An object lying on another one is no reason to refuse: '
                    'pick the upper object and place it on "table" first. kind should be "plan".']
        if because == "not_refused":
            return ['kind is "refuse", so refuse_because must name the reason.']
        return []

    errors = []
    if frozen and not reply["forbidding_words"].strip():
        return ['do_not_move_ids is filled but forbidding_words is "". Only words of the instruction that forbid moving '
                'something ("without moving", "do not touch", "leave", "except") justify do_not_move_ids; otherwise it '
                "must be empty."]
    if reply["forbidding_words"].strip() and not frozen:
        return [f'forbidding_words is "{reply["forbidding_words"].strip()}", but do_not_move_ids is empty. List the objects '
                'these words say must not be moved ("it" is the object named just before). If the words forbid nothing, '
                'set forbidding_words to "".']
    checked = {c["id"]: c["fits"] for c in reply["selector_check"]}
    fitting = [i for i, fits in checked.items() if fits]
    if reply["selector"].strip() and checked and set(fitting) != set(matching) and kind != "refuse":
        return [f"selector_check says fits true for {', '.join(fitting) or 'no candidate'}, but matching_ids is "
                f"{', '.join(matching) or 'empty'}. matching_ids must be exactly the candidates that fit the selector. "
                "Correct whichever of the two is wrong."]
    pointing = not reply["object_kind"].strip() and not reply["selector"].strip() and reply["quantity"] == "one"
    several = reply["quantity"] == "several" or (kind == "plan" and len(matching) > 1 and all(
        any(a["action"] == "pick" and a["object"] == m for a in reply["actions"]) for m in matching))
    if pointing:
        if kind != "ask_user":
            errors.append('object_kind and selector are both "": the instruction does not say which object is meant. '
                          'kind must be "ask_user". (If it does name a kind of object, fill in object_kind.)')
    elif not matching:
        errors.append('matching_ids is empty. If no listed object fits the instruction, kind must be "refuse" with '
                      'refuse_because "no_matching_object"; otherwise list the ids that fit.')
    elif not several and len(matching) > 1 and kind == "plan":
        if reply["selector"].strip():
            errors.append(f"The instruction means one object but matching_ids has {len(matching)}: {', '.join(matching)}. "
                          f'Use the selector "{reply["selector"].strip()}" and the measured facts to keep only the id '
                          'that fits. If the facts cannot decide, kind must be "ask_user".')
        else:
            errors.append(f"The instruction means one object, {len(matching)} objects fit ({', '.join(matching)}) and "
                          'selector is empty: nothing in the instruction says which one. kind must be "ask_user".')
    elif len(matching) == 1 and kind == "ask_user":
        errors.append(f'Exactly one object fits ({matching[0]}), so there is nothing to ask. kind must be "plan", '
                      'or "refuse" if it cannot be done.')
    if kind == "ask_user":
        if not errors and not reply["question"].strip():
            errors.append('kind is "ask_user" but question is empty.')
        return errors
    if errors:
        return errors

    # kind == "plan": walk through the actions
    actions = reply["actions"]
    if not actions:
        return ['kind is "plan" but actions is empty.']
    refuse_hint = 'If the task cannot be done without that, kind must be "refuse" with refuse_because '
    wanted = [d for d in reply["destinations"] if d != TABLE]
    holding, moved, picked = None, set(), []
    for n, a in enumerate(actions, 1):
        target = a["object"]
        if a["action"] == "pick":
            if holding:
                return [f"Action {n} picks {target} while the gripper still holds {holding}. It holds one object at a "
                        "time: place first."]
            if scene.get(target) is None:
                return [f"Action {n} picks {target}, which is a place, not an object that can be picked."]
            if target in frozen:
                return [f"Action {n} picks {target}, but do_not_move_ids says the instruction forbids moving it. "
                        + refuse_hint + '"impossible_as_stated".']
            problem = _cannot_pick(scene, target, gripper_mm)
            if problem:
                return [f"Action {n}: {problem}. " + refuse_hint + '"cannot_be_picked".']
            on_it = [u for u in scene.get(target).under if u not in moved]
            if on_it:
                return [f"Action {n} picks {target}, but {' and '.join(on_it)} still "
                        f"{'lies' if len(on_it) == 1 else 'lie'} on it. Moving an object out of the way is normal: first "
                        f'pick {on_it[0]} and place it on "{TABLE}", then pick {target}.'
                        + (f" But do_not_move_ids forbids moving {on_it[0]}: then kind must be \"refuse\" with "
                           'refuse_because "impossible_as_stated".' if on_it[0] in frozen else "")]
            holding = target
            picked.append(target)
        else:
            if not holding:
                return [f"Action {n} is a place, but the gripper holds nothing. Every place needs a pick before it."]
            if target == holding:
                return [f"Action {n} places {holding} onto itself."]
            if target in picked:
                return [f"Action {n} places {holding} onto {target}, which the plan has moved itself. Place it on a "
                        f'tray, on "{TABLE}" or on an object that stays where it is.']
            if holding in matching and wanted and target not in wanted:
                return [f"Action {n} places {holding} on {target}, but destinations says it should go to "
                        f"{' or '.join(wanted)}."]
            moved.add(holding)
            holding = None
    if wanted and holding in matching:
        return [f"The plan ends with {holding} in the gripper, but destinations says it should be placed on "
                f"{' or '.join(wanted)}. Add the place step."]
    never = [m for m in matching if m not in picked]
    if never:
        errors.append(f"matching_ids contains {', '.join(never)}, but the actions never pick "
                      f"{'it' if len(never) == 1 else 'them'}.")
    extra = [p for p in picked if p not in involved]
    if extra:
        errors.append(f"The actions pick {', '.join(extra)}, which is neither in matching_ids nor lying on a matching "
                      "object. Pick only what the instruction needs.")
    return errors


def _pixel(scene: SceneFile, name: str):
    o = scene.get(name)
    return list(o.pick_px) if o is not None else None


def to_answer(reply: dict, scene: SceneFile, instruction_id: str) -> dict:
    """A validated reply in the result format of schema/result.schema.json."""
    if reply["kind"] == "ask_user":
        return {"instruction_id": instruction_id, "target_pixel": None,
                "actions": [{"action": "ask_user", "question": reply["question"].strip()}]}
    if reply["kind"] == "refuse":
        return {"instruction_id": instruction_id, "target_pixel": None,
                "actions": [{"action": "refuse", "reason": reply["reason"].strip()}]}
    actions = []
    for a in reply["actions"]:
        step = {"action": a["action"], "object": a["object"]}
        if _pixel(scene, a["object"]):
            step["pixel"] = _pixel(scene, a["object"])
        actions.append(step)
    return {"instruction_id": instruction_id, "target_pixel": _pixel(scene, reply["actions"][0]["object"]),
            "actions": actions}


def fallback(reply: dict | None, scene: SceneFile, gripper_mm: float = config.GRIPPER_MAX_OPENING_MM) -> dict:
    """Last resort when Apertus did not produce a valid reply within the retries: a reply in the same
    format, built by the rules above from the slots of its last attempt. Logged as a fallback."""
    pickable = [o.id for o in scene.objects if _cannot_pick(scene, o.id, gripper_mm) is None]
    if reply is None:      # no usable reply at all: most instructions are plans, so guess one
        if not pickable:
            return {"kind": "refuse", "reason": "No object that the gripper can pick was found.", "actions": []}
        reply = {"action_type": "pick", "object_words": "", "selector": "", "object_kind": "object", "forbidding_words": "", "quantity": "one", "destinations": [], "do_not_move_ids": [],
                 "matching_ids": pickable[:1], "actions": []}
    base = {**reply, "refuse_because": "not_refused", "actions": [], "question": "", "reason": ""}
    matching = list(dict.fromkeys(reply["matching_ids"]))
    if reply["action_type"] in UNSAFE_ACTIONS:
        return {**base, "kind": "refuse", "refuse_because": "unsafe_motion",
                "reason": f"The instruction asks for an unsafe motion ({reply['action_type']})."}
    if not reply["object_kind"].strip() and not reply["selector"].strip() and reply["quantity"] == "one":
        return {**base, "kind": "ask_user", "question": "Which object do you mean?"}
    if not matching:
        return {**base, "kind": "refuse", "refuse_because": "no_matching_object",
                "reason": "No object on the table fits the instruction."}
    if reply["quantity"] == "one" and len(matching) > 1:
        return {**base, "kind": "ask_user", "question": f"Which one do you mean: {', '.join(matching)}?"}
    to_move = _involved(scene, matching)
    for m in to_move:
        problem = _cannot_pick(scene, m, gripper_mm)
        if problem:
            return {**base, "kind": "refuse", "refuse_because": "cannot_be_picked",
                    "reason": problem[0].upper() + problem[1:] + "."}
        if m in reply["do_not_move_ids"]:
            return {**base, "kind": "refuse", "refuse_because": "impossible_as_stated",
                    "reason": f"The instruction forbids moving {m}, but the task cannot be done without moving it."}
    wanted = [d for d in reply["destinations"] if d not in to_move + [TABLE]]
    actions = []
    for n, m in enumerate(to_move):
        actions.append({"action": "pick", "object": m})
        if m not in matching:
            actions.append({"action": "place", "object": TABLE})
        elif wanted:
            actions.append({"action": "place", "object": wanted[-1]})
        elif n < len(to_move) - 1:
            actions.append({"action": "place", "object": TABLE})
    return {**base, "kind": "plan", "actions": actions}
