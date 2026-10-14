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
# Words that point at something without saying what kind of object it is. object_kind may be "" only when
# object_words has nothing but these (and colours).
POINTING_WORDS = set("the a an this that these those it its one ones thing things object objects item items something "
                     "anything everything all them both other others stuff me my here there of and any each every "
                     "small big large little tiny smallest biggest largest".split())
COLOUR_WORDS = {name for name, _ in config.HUES} | {"white", "grey", "gray", "black", "brown", "silver"}
REFERRING_SELECTORS = ("closest_to", "farthest_from", "left_of", "right_of", "lying_on", "lying_under", "in_tray")
CHECKED_SELECTORS = ("closest_to", "farthest_from", "left_of", "right_of", "lying_on", "lying_under", "in_tray")
TABLE = "table"
HAND = "human_hand"      # a destination Apertus can name; nothing is ever placed there


def destinations(scene: SceneFile) -> list:
    return [t.id for t in scene.trays] + [TABLE, HAND]


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


def _wrong_kind(reply: dict, scene: SceneFile) -> str | None:
    """A candidate with another noun than object_kind, when the scene has objects with exactly that noun,
    or with another colour than object_words says, when a candidate has exactly that colour. A synonym
    ("cog" for a gear) or a colour that no candidate has is not judged here: that is left to Apertus."""
    ids = [scene.get(i) for i in dict.fromkeys(reply["candidate_ids"] + reply["matching_ids"])]
    kind = reply["object_kind"].strip().lower()
    if kind and any(o.label == kind for o in scene.objects):
        wrong = [o for o in ids if o.label != kind and kind not in o.other_labels]
        if wrong:
            return (f'object_kind is "{kind}", but ' + ", ".join(f"{o.id} is a {o.label}" for o in wrong)
                    + f". candidate_ids and matching_ids may hold only objects that are a {kind}.")
    same = [o for o in scene.objects if o.label == kind]
    colours_said = [w for w in _words(reply["object_words"]) if w in COLOUR_WORDS]
    if same and not ids and (not colours_said or any(o.colour in colours_said for o in same)):
        return (f'candidate_ids is empty, but the list has {", ".join(o.id for o in same)}, which '
                f'{"is" if len(same) == 1 else "are"} a {kind}. candidate_ids must hold the listed objects whose '
                "label and colour fit object_words.")
    said = [c for c in _words(reply["object_words"]) if any(o.colour == c for o in ids)]
    wrong = [o for o in ids if o.colour not in said] if len(said) == 1 else []
    if wrong:
        return (f'object_words is "{reply["object_words"].strip()}", but ' + ", ".join(f"{o.id} is {o.colour}" for o in wrong)
                + f". candidate_ids and matching_ids may hold only objects that are {said[0]}.")
    return None


def name_doubt(reply: dict, scene: SceneFile) -> str | None:
    """A refusal because nothing is called object_kind, while an object of the right colour has a name the naming
    model was unsure of, or "could also be" that kind. Apertus is asked once whether that object is meant."""
    kind = reply["object_kind"].strip().lower()
    if reply["kind"] != "refuse" or reply["refuse_because"] != "no_matching_object" or not kind:
        return None
    if any(o.label == kind for o in scene.objects):
        return None
    colours = [w for w in _words(reply["object_words"]) if w in COLOUR_WORDS]
    unsure = [o for o in scene.objects if (not colours or o.colour in colours) and (
        kind in o.other_labels or (o.name_score is not None and o.name_score < config.NAME_SURE_SCORE))]
    if not unsure:
        return None
    return (f'You refuse because no listed object is a {kind}. The names in the list are guesses of an image model, and '
            f'it was unsure about {", ".join(o.id for o in unsure)}. If {"it" if len(unsure) == 1 else "one of them"} '
            f'can be the {reply["object_words"].strip()} the instruction means, put {"its id" if len(unsure) == 1 else "the ids that can be"} '
            'in candidate_ids and matching_ids and answer again. If none can be, give the same refusal again.')


def _words(text: str) -> list:
    return "".join(c if c.isalnum() else " " for c in text.lower()).split()


def _reading(reply: dict, scene: SceneFile, instruction: str) -> str | None:
    """Slots that contradict each other or the instruction's own words. These are about how Apertus split
    the sentence, before any object is chosen."""
    object_words, selector, kind_noun = _words(reply["object_words"]), reply["selector"].strip(), reply["object_kind"].strip()
    colours = {o.colour for o in scene.objects} | COLOUR_WORDS
    naming = [w for w in object_words if w not in POINTING_WORDS and w not in colours]
    if not kind_noun and not selector and naming:
        return (f'object_words is "{reply["object_words"].strip()}" but object_kind is "". If "{naming[-1]}" names a kind '
                f'of object, object_kind must be "{naming[-1]}", also when the list calls such an object by another '
                'word. object_kind is "" only for words like "that one", "it", "the thing".')
    foreign = [w for w in _words(selector) if w not in _words(instruction)] if instruction else []
    if foreign:
        return (f'selector is "{selector}", but the instruction does not contain {", ".join(repr(w) for w in foreign)}. '
                'selector must be copied from the instruction, or be "" if the instruction has no words that single '
                "out one object.")
    stype, refs = reply["selector_type"], reply["selector_reference_ids"]
    if bool(selector) != (stype != "none"):
        return (f'selector is "{selector}" but selector_type is "{stype}". selector_type is "none" exactly when '
                'selector is "".')
    if stype in REFERRING_SELECTORS and not refs:
        return (f'selector_type is "{stype}", which refers to another object or a tray, but selector_reference_ids is '
                'empty. Name the object the selector refers to; if the selector is a place in the row ("on the right", '
                '"furthest to the left"), selector_type is "order".')
    if (stype == "order") != (reply["order_rank"] != "none"):
        return (f'selector_type is "{stype}" but order_rank is "{reply["order_rank"]}". order_rank is filled exactly '
                'when selector_type is "order".')
    # a listed object that the instruction mentions has to turn up in one of the slots
    said = set(_words(instruction))
    used = set(_words(" ".join([reply["object_words"], selector, reply["forbidding_words"]])))
    placed = {scene.get(d).label for d in reply["destinations"] if scene.get(d) is not None}
    for label in dict.fromkeys(o.label for o in scene.objects):
        forms = {label, label + "s"}
        if " " not in label and forms & said and not forms & used and label not in placed:
            return (f'The instruction mentions "{label}", but object_words, selector, forbidding_words and destinations '
                    f'do not. If the words around "{label}" single out the object that is meant ("closest to the '
                    f'{label}", "under the {label}"), copy them into selector.')
    return None


def _selector_facts(reply: dict, scene: SceneFile) -> tuple[set, str] | None:
    """What the measurements say about a selector whose type code can check: (ids that may be in
    matching_ids, the facts as a sentence). None when the selector is of a type that is left to Apertus.

    Apertus has named the type of the selector, the object it refers to and the candidates; the code only
    evaluates that relation on the measured scene."""
    kind = reply["selector_type"]
    refs = list(dict.fromkeys(reply["selector_reference_ids"]))
    if kind == "order":
        return _order_facts(reply, scene)
    if kind not in CHECKED_SELECTORS or not refs:
        return None
    tray_refs = [r for r in refs if scene.get(r) is None]
    if (kind == "in_tray") != bool(tray_refs) or (tray_refs and len(tray_refs) != len(refs)):
        return None
    pool = reply["candidate_ids"] if reply["object_kind"].strip() else [o.id for o in scene.objects]
    pool = [scene.get(i) for i in dict.fromkeys(pool) if i not in refs]
    if not pool:
        return None
    if kind == "in_tray":
        fits = {o.id for o in pool if o.in_tray and any(r.startswith(o.in_tray) for r in refs)}
        facts = [f"{o.id} is {'in the ' + o.in_tray + ' tray' if o.in_tray else 'in no tray'}" for o in pool]
    elif kind in ("lying_on", "lying_under"):
        field, words = ("on_top_of", "lies on top of") if kind == "lying_on" else ("under", "lies under")
        fits = {o.id for o in pool if set(getattr(o, field)) & set(refs)}
        facts = [f"{o.id} {words} {' and '.join(getattr(o, field))}" for o in scene.objects if getattr(o, field)] \
            or [f"no object {words} another"]
    elif len(refs) != 1 or scene.relations is None:
        return None
    elif kind in ("closest_to", "farthest_from"):
        gap = {i: d for i, d in scene.relations[refs[0]]["nearest"]}
        best = (min if kind == "closest_to" else max)(gap[o.id] for o in pool)
        fits = {o.id for o in pool if abs(gap[o.id] - best) <= config.DISTANCE_TIE_MM}   # a near-tie: either is accepted
        facts = [f"{o.id} is {gap[o.id]} mm from {refs[0]}" for o in sorted(pool, key=lambda o: gap[o.id])]
    else:   # left_of, right_of
        ref, sign = scene.get(refs[0]), (-1 if kind == "left_of" else 1)
        side = "left" if kind == "left_of" else "right"
        offset = {o.id: ((o.centre_px[0] - ref.centre_px[0]) * sign, abs(o.centre_px[1] - ref.centre_px[1])) for o in pool}
        beside = [i for i, (along, across) in offset.items() if along > 0 and along >= across]   # as in geometry.relations()
        on_side = beside or [i for i, (along, _) in offset.items() if along > 0]
        fits = {min(on_side, key=lambda i: offset[i][0] ** 2 + offset[i][1] ** 2)} if on_side else set()
        facts = [f"{i} is {'directly to the ' + side + ' of' if i in beside else 'further to the ' + side + ' than' if along > 0 else 'not to the ' + side + ' of'} {ref.id}"
                 for i, (along, _) in offset.items()]
        if len(beside) > 1:
            facts.append(f"of those directly to its {side}, {next(iter(fits))} is the nearest")
    return fits, "; ".join(facts)


def _order_facts(reply: dict, scene: SceneFile) -> tuple[set, str] | None:
    """The candidate at the place in the row that order_rank names, counted on the image column of the centres."""
    rank = reply["order_rank"]
    pool = reply["candidate_ids"] if reply["object_kind"].strip() else [o.id for o in scene.objects]
    row = sorted((scene.get(i) for i in dict.fromkeys(pool)), key=lambda o: o.centre_px[0])
    if rank == "none" or len(row) < 2:
        return None
    if rank == "middle":
        fits = {row[len(row) // 2].id} if len(row) % 2 else set()
    else:
        n, side = int(rank[0]), rank.split()[-1]
        fits = {(row if side == "left" else row[::-1])[n - 1].id} if n <= len(row) else set()
    return fits, "from left to right the candidates are " + ", ".join(o.id for o in row)


def validate(reply: dict, scene: SceneFile, gripper_mm: float = config.GRIPPER_MAX_OPENING_MM,
             instruction: str = "") -> list:
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
    if HAND in reply["destinations"]:
        if kind != "refuse" or because != "unsafe_motion":
            return [f'destinations contains "{HAND}": the robot would move onto a person. kind must be "refuse" with '
                    'refuse_because "unsafe_motion".'
                    + (f' (If the instruction only asks to be given an object ("hand me", "give me"), remove "{HAND}" '
                       "from destinations.)" if action_type == "pick" else "")]
        return []
    wrong_kind = _wrong_kind(reply, scene)
    if wrong_kind:
        return [wrong_kind]
    misread = _reading(reply, scene, instruction)
    if misread:
        return [misread]
    measured = _selector_facts(reply, scene)
    if measured is not None:
        fits, facts = measured
        tie = reply["selector_type"] in ("closest_to", "farthest_from") and len(fits) > 1
        if not (set(matching) <= fits and len(matching) == 1 if tie else set(matching) == fits):
            refs = ", ".join(dict.fromkeys(reply["selector_reference_ids"]))
            how = f'order_rank "{reply["order_rank"]}"' if reply["selector_type"] == "order" else f"reference {refs}"
            return [f'The selector "{reply["selector"].strip()}" was checked as selector_type "{reply["selector_type"]}" '
                    f"with {how}. Measured: {facts}. So "
                    + (f"{' and '.join(sorted(fits))} {'fits' if len(fits) == 1 else 'fit'}" if fits else "no candidate fits")
                    + f", but matching_ids is {', '.join(matching) or 'empty'}. Correct selector_check and matching_ids, "
                    "or correct candidate_ids, selector_type, selector_reference_ids or order_rank if those are wrong."]
    if kind == "refuse":
        if not reply["reason"].strip():
            return ['kind is "refuse" but reason is empty.']
        if because == "unsafe_motion" and HAND not in reply["destinations"]:
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
    elif not matching and kind == "ask_user" and len(set(reply["candidate_ids"])) > 1:
        errors.append('kind is "ask_user", so matching_ids must list the objects the person has to choose between: '
                      f'{", ".join(dict.fromkeys(reply["candidate_ids"]))} from candidate_ids, or those of them that fit.')
    elif not matching:
        errors.append('matching_ids is empty. The list may call an object by another word than the instruction does. If '
                      f'a listed object is a {reply["object_kind"].strip() or "match"} under another name, put its id in '
                      'candidate_ids and matching_ids. If no listed object fits, kind must be "refuse" with '
                      'refuse_because "no_matching_object".')
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
    wanted = [d for d in reply["destinations"] if d not in (TABLE, HAND)]
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
            on_it = [u for u in _blockers(scene, target) if u not in moved]
            if on_it:
                clear = ", then ".join(f'pick {u} and place it on "{TABLE}"' for u in on_it)
                barred = [u for u in on_it if u in frozen]
                return [f"Action {n} picks {target}, but {' and '.join(on_it)} still "
                        f"{'lies' if len(on_it) == 1 else 'lie'} on it. Moving an object out of the way is normal: first "
                        f"{clear}, then pick {target}."
                        + (f" But do_not_move_ids forbids moving {barred[0]}: then kind must be \"refuse\" with "
                           'refuse_because "impossible_as_stated".' if barred else "")]
            holding = target
            picked.append(target)
        else:
            if not holding:
                return [f"Action {n} is a place, but the gripper holds nothing. Every place needs a pick before it."]
            if target == HAND:
                return [f'Action {n} places {holding} on "{HAND}": the robot would move onto a person. kind must be '
                        '"refuse" with refuse_because "unsafe_motion", and destinations must contain "human_hand".']
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
        reply = {"action_type": "pick", "object_words": "", "selector": "", "selector_type": "none", "selector_reference_ids": [], "order_rank": "none", "candidate_ids": [], "object_kind": "object", "forbidding_words": "", "quantity": "one", "destinations": [], "do_not_move_ids": [],
                 "matching_ids": pickable[:1], "actions": []}
    base = {**reply, "refuse_because": "not_refused", "actions": [], "question": "", "reason": ""}
    matching = list(dict.fromkeys(reply["matching_ids"]))
    measured = _selector_facts(reply, scene)
    if measured is not None and not (set(matching) <= measured[0] and matching):
        matching = [o.id for o in scene.objects if o.id in measured[0]]      # what the measurements say fits the selector
        if reply["selector_type"] in ("closest_to", "farthest_from"):
            matching = matching[:1]
        base["matching_ids"] = matching
    if reply["action_type"] in UNSAFE_ACTIONS:
        return {**base, "kind": "refuse", "refuse_because": "unsafe_motion",
                "reason": f"The instruction asks for an unsafe motion ({reply['action_type']})."}
    if HAND in reply["destinations"]:
        return {**base, "kind": "refuse", "refuse_because": "unsafe_motion",
                "reason": "The instruction asks the robot to move onto a person."}
    candidates = list(dict.fromkeys(reply["candidate_ids"]))
    if not matching and len(candidates) > 1 and reply["quantity"] == "one" and not reply["selector"].strip():
        return {**base, "kind": "ask_user", "matching_ids": candidates, "question": f"Which one do you mean: {', '.join(candidates)}?"}
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
    wanted = [d for d in reply["destinations"] if d not in to_move + [TABLE, HAND]]
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
