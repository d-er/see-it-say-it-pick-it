"""Image -> scene file (masks, labels, colours, trays, hand).

Three open models, loaded one after the other and released before Apertus needs the GPU:

  Grounding DINO   boxes for anything that looks like an object, from generic prompts ("object",
                   "toy", "part", ...) on the whole image and on enlarged tiles; the grey tray; a hand
  SAM 2.1          one mask per box; and, from a grid of points, masks for what the detector missed
  SigLIP 2         a noun per mask, chosen from a broad vocabulary (vocabulary.py)

The colour name is measured from the pixels. Nothing here reads the instruction: which object is
meant is decided later, by Apertus.
"""
from __future__ import annotations

import cv2
import numpy as np

from . import config
from .scene import SceneFile, SceneObject, Tray
from .vocabulary import VOCABULARY

_models: dict = {}


def _load(kind: str):
    """Models are loaded on first use and kept until release()."""
    if kind not in _models:
        import torch
        from transformers import (AutoModel, AutoModelForZeroShotObjectDetection, AutoProcessor, Sam2Model,
                                  Sam2Processor)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model_id, revision = getattr(config, f"{kind.upper()}_MODEL")
        if kind == "detector":
            processor = AutoProcessor.from_pretrained(model_id, revision=revision)
            model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id, revision=revision)
        elif kind == "segmenter":
            processor = Sam2Processor.from_pretrained(model_id, revision=revision)
            model = Sam2Model.from_pretrained(model_id, revision=revision)
        else:
            processor = AutoProcessor.from_pretrained(model_id, revision=revision)
            model = AutoModel.from_pretrained(model_id, revision=revision)
        _models[kind] = (processor, model.to(device).eval(), device)
    return _models[kind]


def release() -> None:
    """Drop the models and give the GPU memory back (Apertus needs almost all of it)."""
    _models.clear()
    try:
        import gc
        import torch
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


# ---------------------------------------------------------------- detector

def _detect(image, prompt: str, threshold: float) -> list:
    """[(box [x1, y1, x2, y2], score)] for one short prompt. One prompt per call: with several nouns
    in one prompt the detector misses most of the gears, cubes and parts."""
    import torch
    processor, model, device = _load("detector")
    inputs = processor(images=image, text=f"{prompt}.", return_tensors="pt").to(device)
    with torch.no_grad():
        out = model(**inputs)
    res = processor.post_process_grounded_object_detection(out, threshold=threshold, text_threshold=0.0,
                                                           target_sizes=[image.shape[:2]])[0]
    return list(zip(res["boxes"].tolist(), res["scores"].tolist()))


def _views(w: int, h: int) -> list:
    """The whole image plus overlapping tiles that the detector sees enlarged."""
    views = [(0, 0, w, h)]
    n, share = config.TILES_PER_SIDE, config.TILE_SHARE
    tw, th = int(w * share), int(h * share)
    for j in range(n):
        for i in range(n):
            x0, y0 = round(i * (w - tw) / (n - 1)), round(j * (h - th) / (n - 1))
            views.append((x0, y0, x0 + tw, y0 + th))
    return views


def _iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter + 1e-9)


def propose(rgb: np.ndarray) -> list:
    """Boxes of everything that looks like an object: [(box, score, prompt)], best first, without
    near-duplicates. A box cut by the edge of a tile is dropped: the object is whole in another view."""
    h, w = rgb.shape[:2]
    found = []
    for x0, y0, x1, y1 in _views(w, h):
        view = rgb[y0:y1, x0:x1]
        for prompt in config.PROPOSAL_PROMPTS:
            for box, score in _detect(view, prompt, config.BOX_THRESHOLD):
                cut = ((box[0] < config.TILE_EDGE_PX and x0 > 0) or (box[1] < config.TILE_EDGE_PX and y0 > 0)
                       or (box[2] > x1 - x0 - config.TILE_EDGE_PX and x1 < w)
                       or (box[3] > y1 - y0 - config.TILE_EDGE_PX and y1 < h))
                if cut:
                    continue
                box = [float(np.clip(box[0] + x0, 0, w - 1)), float(np.clip(box[1] + y0, 0, h - 1)),
                       float(np.clip(box[2] + x0, 0, w - 1)), float(np.clip(box[3] + y0, 0, h - 1))]
                if min(box[2] - box[0], box[3] - box[1]) >= config.MIN_BOX_PX:
                    found.append((box, score, prompt))
    found.sort(key=lambda f: -f[1])
    kept = []
    for f in found:
        if all(_iou(f[0], k[0]) < config.BOX_DUPLICATE_IOU for k in kept):
            kept.append(f)
    return kept


# ---------------------------------------------------------------- segmenter

def segment(rgb: np.ndarray, boxes: list) -> list:
    """One boolean mask per box."""
    import torch
    if not boxes:
        return []
    processor, model, device = _load("segmenter")
    masks = []
    with torch.no_grad():
        pixels = processor(images=rgb, return_tensors="pt").to(device)
        embeddings = model.get_image_embeddings(pixels["pixel_values"])
        for i in range(0, len(boxes), config.SAM_BATCH):
            chunk = boxes[i:i + config.SAM_BATCH]
            inputs = processor(images=rgb, input_boxes=[chunk], return_tensors="pt").to(device)
            out = model(image_embeddings=embeddings, input_boxes=inputs["input_boxes"], multimask_output=False)
            full = processor.post_process_masks(out.pred_masks.cpu(), inputs["original_sizes"])[0]
            masks.extend(np.asarray(m[0], dtype=bool) for m in full)
    return [_largest_part(m) for m in masks]


def _largest_part(mask: np.ndarray) -> np.ndarray:
    """Drop specks: keep the connected pieces that hold at least a tenth of the largest one."""
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    if n <= 2:
        return mask
    areas = stats[1:, cv2.CC_STAT_AREA]
    keep = 1 + np.flatnonzero(areas >= 0.1 * areas.max())
    return np.isin(labels, keep)


# ---------------------------------------------------------------- trays and hand

def find_trays(rgb: np.ndarray) -> list:
    """Outlines of the two place targets, as rectangles from the detector. The grey tray is the best
    large "tray" box. The black tray is a mat, which the detector does not call a tray ("black tray"
    returns the grey one), so it is asked for by what it looks like."""
    h, w = rgb.shape[:2]

    def large_boxes(prompt):
        return sorted(((s, b) for b, s in _detect(rgb, prompt, config.TRAY_THRESHOLD)
                       if config.TRAY_MIN_SHARE <= (b[2] - b[0]) * (b[3] - b[1]) / (w * h) <= config.TRAY_MAX_SHARE),
                      reverse=True)

    def tray(tray_id, box):
        x0, y0, x1, y1 = (int(round(float(np.clip(v, 0, lim)))) for v, lim in zip(box, (w - 1, h - 1, w - 1, h - 1)))
        return Tray(tray_id, [[x0, y0], [x1, y0], [x1, y1], [x0, y1]])

    trays, grey = [], None
    boxes = large_boxes(config.GREY_TRAY_PROMPT)
    if boxes:
        grey = boxes[0][1]
        trays.append(tray("grey_tray", grey))
    for prompt in config.BLACK_TRAY_PROMPTS:
        boxes = [b for _, b in large_boxes(prompt) if grey is None or _iou(b, grey) < config.TRAY_SAME_IOU]
        if boxes:
            trays.append(tray("black_tray", boxes[0]))
            break
    return trays


def find_hand(rgb: np.ndarray) -> np.ndarray | None:
    """Mask of a human hand and arm, or None."""
    boxes = [b for prompt in config.HAND_PROMPTS for b, s in _detect(rgb, prompt, config.HAND_THRESHOLD)]
    if not boxes:
        return None
    return np.any(segment(rgb, boxes), axis=0)


# ---------------------------------------------------------------- which masks are objects

def _inside(point, outline, margin: float) -> bool:
    return cv2.pointPolygonTest(np.array(outline, dtype=np.float32), (float(point[0]), float(point[1])), True) >= -margin


def _contrast(lab: np.ndarray, mask: np.ndarray) -> float:
    """Colour difference (CIELAB) between the inside of a mask and the band around it."""
    m = mask.astype(np.uint8)
    inner = cv2.erode(m, np.ones((9, 9), np.uint8)).astype(bool)
    band = (cv2.dilate(m, np.ones((41, 41), np.uint8)) - cv2.dilate(m, np.ones((9, 9), np.uint8))).astype(bool)
    if not inner.any() or not band.any():
        return 0.0
    return float(np.linalg.norm(np.median(lab[inner], axis=0) - np.median(lab[band], axis=0)))


def _surrounded(inner: np.ndarray, outer: np.ndarray) -> float:
    """Share of the band around `inner` that belongs to `outer`."""
    m = inner.astype(np.uint8)
    band = (cv2.dilate(m, np.ones((41, 41), np.uint8)) - cv2.dilate(m, np.ones((9, 9), np.uint8))).astype(bool)
    return float((band & outer).sum() / max(1, band.sum()))


def select(rgb: np.ndarray, masks: list, scores: list, trays: list, hand: np.ndarray | None, log=None):
    """Which masks are objects. Returns (indices, enclosed): `enclosed` maps the index of a mask that
    lies wholly inside a larger one to that larger one; perceive() keeps it only if it gets a clear name.
    On return every pixel belongs to one object only.

    Dropped: masks outside the trays, tray-sized masks, a hand, marks that have the colour of the
    surface around them, duplicates, and a mask that is just several objects together.
    """
    h, w = rgb.shape[:2]
    log = log if log is not None else []
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    area = [int(m.sum()) for m in masks]
    tray_area = [cv2.contourArea(np.array(t.outline_px, dtype=np.float32)) for t in trays]
    alive = []
    for i, m in enumerate(masks):
        if area[i] < config.MIN_OBJECT_PX:
            log.append((i, "too small")); continue
        vs, us = np.nonzero(m)
        centre = (us.mean(), vs.mean())
        if trays and not any(_inside(centre, t.outline_px, config.TRAY_MARGIN_PX) for t in trays):
            log.append((i, "outside the trays")); continue
        if area[i] > config.MAX_OBJECT_SHARE * (min(tray_area) if tray_area else w * h):
            log.append((i, "as large as a tray")); continue
        if hand is not None and (m & hand).sum() > config.HAND_OVERLAP * area[i]:
            log.append((i, "hand")); continue
        contrast = _contrast(lab, m)
        if contrast < config.MIN_CONTRAST:
            log.append((i, f"same colour as around it ({contrast:.0f})")); continue
        alive.append(i)

    # duplicates: the better detector score stays
    alive.sort(key=lambda i: -scores[i])
    kept = []
    for i in alive:
        dup = next((k for k in kept if (masks[i] & masks[k]).sum() / (masks[i] | masks[k]).sum() > config.MASK_DUPLICATE_IOU), None)
        if dup is None:
            kept.append(i)
        else:
            log.append((i, f"duplicate of {dup}"))

    # a mask made of several others: each of them lies almost fully in it, together they fill it
    kept.sort(key=lambda i: -area[i])
    for i in list(kept):
        inner = [k for k in kept if k != i and area[k] < config.GROUP_MEMBER_MAX * area[i]
                 and (masks[k] & masks[i]).sum() > config.CONTAINED * area[k]]
        if len(inner) >= 2:
            union = np.any([masks[k] for k in inner], axis=0)
            if (union & masks[i]).sum() > config.GROUP_FILL * area[i]:
                kept.remove(i)
                log.append((i, f"group of {inner}"))

    # a mask inside a larger mask that also covers it. Surrounded on all sides: probably a piece of
    # that object (the middle of a saucer, the print on a plate). Sticking out over its edge: an
    # object lying on it.
    enclosed = {}
    for i in sorted(kept, key=lambda i: area[i]):
        outer = next((k for k in kept if k != i and area[k] > area[i]
                      and (masks[k] & masks[i]).sum() > config.CONTAINED * area[i]), None)
        if outer is not None and _surrounded(masks[i], masks[outer]) >= config.SURROUNDED:
            enclosed[i] = outer

    # visible pixels only: where two masks overlap, the smaller object is the one in front
    kept.sort(key=lambda i: area[i])
    taken = np.zeros((h, w), dtype=bool)
    for i in kept:
        if i not in enclosed:
            masks[i] = masks[i] & ~taken
            taken |= masks[i]
    return [i for i in kept if masks[i].sum() >= config.MIN_OBJECT_PX], enclosed


def recall_net(rgb: np.ndarray, known: list, trays: list, hand: np.ndarray | None, log=None) -> list:
    """Masks of objects the detector did not propose. SAM 2.1 segments everything from a grid of
    points; most of that is not an object, so a mask is only taken if it lies in a tray, stands out
    from the surface around it, is not a long strip (tray rim) or cut by the image border, and is
    neither covered nor surrounded by something already known (the hole of a part, print on a plate)."""
    from PIL import Image
    from transformers import pipeline
    processor, model, device = _load("segmenter")
    generator = pipeline("mask-generation", model=model, image_processor=processor.image_processor, device=model.device)
    out = generator(Image.fromarray(rgb), points_per_batch=64, points_per_crop=config.RECALL_POINTS_PER_SIDE)
    h, w = rgb.shape[:2]
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    tray_area = [cv2.contourArea(np.array(t.outline_px, dtype=np.float32)) for t in trays]
    taken = np.any(known, axis=0) if known else np.zeros((h, w), dtype=bool)
    log = log if log is not None else []
    added = []
    for m in sorted((np.asarray(m, dtype=bool) for m in out["masks"]), key=lambda m: -int(m.sum())):
        area = int(m.sum())
        if area < config.MIN_OBJECT_PX or area > config.MAX_OBJECT_SHARE * (min(tray_area) if tray_area else w * h):
            continue
        vs, us = np.nonzero(m)
        box = [int(us.min()), int(vs.min()), int(us.max()), int(vs.max())]
        if trays and not any(_inside((us.mean(), vs.mean()), t.outline_px, config.TRAY_MARGIN_PX) for t in trays):
            continue
        if box[0] <= 2 or box[1] <= 2 or box[2] >= w - 3 or box[3] >= h - 3:
            continue
        if hand is not None and (m & hand).sum() > config.HAND_OVERLAP * area:
            continue
        if (m & taken).sum() > config.RECALL_MAX_COVERED * area or _surrounded(m, taken) >= config.SURROUNDED:
            continue
        (_, _), (rw, rh), _ = cv2.minAreaRect(np.column_stack((us, vs)).astype(np.float32))
        if max(rw, rh) > config.RECALL_MAX_ASPECT * max(1.0, min(rw, rh)):
            log.append((box, "recall net: long strip")); continue
        contrast = _contrast(lab, m)
        if contrast < config.MIN_CONTRAST:
            log.append((box, f"recall net: same colour as around it ({contrast:.0f})")); continue
        m = _largest_part(m & ~taken)
        taken |= m
        added.append(m)
        log.append((box, f"recall net: added (contrast {contrast:.0f})"))
    return added


# ---------------------------------------------------------------- colour and name

def colour_name(rgb: np.ndarray, mask: np.ndarray) -> str | None:
    """Colour of most of the object, from its pixels. None if it has no colour and its lightness does
    not say whether it is white or grey (white_or_grey() decides that)."""
    inner = cv2.erode(mask.astype(np.uint8), np.ones((9, 9), np.uint8)).astype(bool)
    px = rgb[inner if inner.sum() > 200 else mask]
    hsv = cv2.cvtColor(px[None], cv2.COLOR_RGB2HSV)[0].astype(int)
    lab = cv2.cvtColor(px[None], cv2.COLOR_RGB2LAB)[0].astype(int)
    lightness = lab[:, 0] * 100 / 255
    chroma = np.hypot(lab[:, 1] - 128, lab[:, 2] - 128)
    coloured = chroma >= config.CHROMA_MIN
    if coloured.mean() < config.COLOURED_SHARE:
        light = float(np.median(lightness))
        return "black" if light < config.BLACK_MAX_L else "grey" if light < config.GREY_MAX_L \
            else "white" if light > config.WHITE_MIN_L else None
    hue = float(np.median(hsv[coloured, 0])) * 2          # degrees
    value = float(np.median(hsv[coloured, 2]))
    for name, upper in config.HUES:
        if hue < upper:
            return "brown" if name == "orange" and value < config.BROWN_MAX_VALUE else name
    return "red"


def _image_features(rgb: np.ndarray, masks: list):
    """One normalised SigLIP vector per mask: the crop around the object with its surroundings blurred."""
    import torch
    processor, model, device = _load("namer")
    h, w = rgb.shape[:2]
    crops = []
    for m in masks:
        vs, us = np.nonzero(m)
        cx, cy = (us.min() + us.max()) / 2, (vs.min() + vs.max()) / 2
        half = config.CROP_SCALE * max(us.max() - us.min(), vs.max() - vs.min()) / 2
        x0, y0, x1, y1 = int(max(0, cx - half)), int(max(0, cy - half)), int(min(w, cx + half)), int(min(h, cy + half))
        crop, inside = rgb[y0:y1, x0:x1].copy(), m[y0:y1, x0:x1]
        crop[~inside] = cv2.GaussianBlur(crop, (0, 0), config.CROP_BLUR_SIGMA)[~inside]
        crops.append(crop)
    with torch.no_grad():
        f = model.get_image_features(**processor(images=crops, return_tensors="pt").to(device))
        f = f if isinstance(f, torch.Tensor) else f.pooler_output
        return torch.nn.functional.normalize(f, dim=-1)


_text_cache: dict = {}


def _text_features(phrases: tuple):
    import torch
    if phrases not in _text_cache:
        processor, model, device = _load("namer")
        with torch.no_grad():
            t = processor(text=list(phrases), padding="max_length", max_length=64, return_tensors="pt").to(device)
            f = model.get_text_features(**t)
            f = f if isinstance(f, torch.Tensor) else f.pooler_output
            _text_cache[phrases] = torch.nn.functional.normalize(f, dim=-1)
    return _text_cache[phrases]


def name_objects(rgb: np.ndarray, masks: list) -> list:
    """[(noun, score, [(noun, score), ...top 3])] per mask, from every noun of the vocabulary. A noun
    scores as its best phrase."""
    import torch
    if not masks:
        return []
    _, model, _ = _load("namer")
    nouns = [n for n, _ in VOCABULARY]
    image = _image_features(rgb, masks)
    with torch.no_grad():
        sims = torch.stack([(image @ _text_features(tuple(f"a photo of a {p}" for p in phrases)).T).max(dim=1).values
                            for _, phrases in VOCABULARY], dim=1)
        probs = torch.sigmoid(sims * model.logit_scale.exp() + model.logit_bias).cpu().numpy()
    out = []
    for row in probs:
        order = np.argsort(-row)[:3]
        out.append((nouns[order[0]], float(row[order[0]]), [(nouns[k], round(float(row[k]), 3)) for k in order]))
    return out


def white_or_grey(rgb: np.ndarray, mask: np.ndarray) -> str:
    """For an object without colour whose lightness is between sure grey and sure white: the naming
    model says what it looks made of (white plastic or porcelain, or grey metal)."""
    image = _image_features(rgb, [mask])
    best = max(config.MATERIALS, key=lambda phrase: float((image @ _text_features((phrase,)).T)[0, 0]))
    return config.MATERIALS[best]


# ---------------------------------------------------------------- the stage

def perceive(name: str, rgb: np.ndarray, debug: dict | None = None) -> SceneFile:
    h, w = rgb.shape[:2]
    trays = find_trays(rgb)
    hand = find_hand(rgb)
    proposals = propose(rgb)
    masks = segment(rgb, [p[0] for p in proposals])
    log = []
    chosen, enclosed = select(rgb, masks, [p[1] for p in proposals], trays, hand, log)
    names = dict(zip(chosen, name_objects(rgb, [masks[i] for i in chosen])))
    for i, outer in enclosed.items():
        if i not in names:
            continue
        if names[i][1] >= config.ENCLOSED_NAME_MIN and names[i][0] != names[outer][0]:
            masks[outer] = masks[outer] & ~masks[i]          # an object lying in the middle of another one
        else:
            log.append((i, f"piece of {outer} ({names[i][0]} {names[i][1]:.2f})"))
            chosen.remove(i)
    pieces = [masks[i] for i in enclosed if i not in chosen]
    for m in recall_net(rgb, [masks[i] for i in chosen] + pieces, trays, hand, log):
        masks.append(m)
        proposals.append(([float(v) for v in cv2.boundingRect(m.astype(np.uint8))], 0.0, "recall net"))
        names[len(masks) - 1] = name_objects(rgb, [m])[0]
        chosen.append(len(masks) - 1)
    chosen.sort(key=lambda i: np.nonzero(masks[i])[1].mean())      # left to right
    objects, counts = [], {}
    for i in chosen:
        noun, score, _ = names[i]
        label = noun if score >= config.NAME_MIN_SCORE else config.UNKNOWN_LABEL
        colour = colour_name(rgb, masks[i]) or white_or_grey(rgb, masks[i])
        key = f"{colour}_{label}".replace(" ", "_")
        counts[key] = counts.get(key, 0) + 1
        objects.append(SceneObject(id=f"{key}_{counts[key]}", label=label, colour=colour, mask=masks[i]))
    if debug is not None:
        debug.update(proposals=proposals, masks=masks, dropped=log, chosen=chosen, names=[names[i] for i in chosen])
    return SceneFile(scene=name, image_size=[w, h], objects=objects, trays=trays, hand=hand)
