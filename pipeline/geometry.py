"""Masks -> measured facts. Pure geometry: no language, no model.

measure() fills the per-object fields of a SceneFile (pick point, width in mm, tray, stacking, hand).
relations() computes what depends on several objects (order from the left, neighbours, directions).
"""
from __future__ import annotations

import cv2
import numpy as np

from . import config
from .scene import SceneFile, SceneObject


def pick_point(mask: np.ndarray) -> list:
    """Pixel deepest inside the mask. Unlike the centroid it never falls into a hole or outside a
    curved object, and it is the middle of the widest part of the top surface."""
    dist = cv2.distanceTransform(np.pad(mask.astype(np.uint8), 1), cv2.DIST_L2, 5)[1:-1, 1:-1]
    v, u = np.unravel_index(int(np.argmax(dist)), dist.shape)
    return [int(u), int(v)]


def centroid(mask: np.ndarray) -> list:
    vs, us = np.nonzero(mask)
    return [int(round(us.mean())), int(round(vs.mean()))]


def size_mm(mask: np.ndarray, to_mm) -> list:
    """[length, width] on the table plane. Width is the smallest distance between two parallel lines
    that enclose the object (what the gripper has to span); length is the extent across that."""
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    pts = np.concatenate([c.reshape(-1, 2) for c in contours])
    hull = np.array([to_mm(u, v) for u, v in cv2.convexHull(pts).reshape(-1, 2)], dtype=float)
    best = None
    for i in range(len(hull)):
        edge = hull[(i + 1) % len(hull)] - hull[i]
        n = np.linalg.norm(edge)
        if n < 1e-6:
            continue
        along = edge / n
        across = (hull - hull[i]) @ np.array([-along[1], along[0]])
        width = float(across.max() - across.min())
        if best is None or width < best[1]:
            span = hull @ along
            best = (float(span.max() - span.min()), width)
    return [round(best[0], 1), round(best[1], 1)]


def _hull_mask(mask: np.ndarray) -> np.ndarray:
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    hull = cv2.convexHull(np.concatenate([c.reshape(-1, 2) for c in contours]))
    out = np.zeros(mask.shape, np.uint8)
    cv2.fillConvexPoly(out, hull, 1)
    return out.astype(bool)


def stacking(objects: list) -> dict:
    """{(top_id, bottom_id): cover}. Masks hold visible pixels only, so the object underneath has a
    piece missing where the upper one hides it: the upper object fills part of the lower one's convex
    hull, while the lower one fills almost none of the upper one's. The two masks must also touch."""
    hulls = {o.id: _hull_mask(o.mask) for o in objects}
    grown = {o.id: cv2.dilate(o.mask.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool) for o in objects}
    pairs = {}
    for top in objects:
        for bottom in objects:
            if top.id == bottom.id or int((grown[top.id] & grown[bottom.id]).sum()) < config.CONTACT_MIN_PX:
                continue
            covered = int((hulls[bottom.id] & top.mask).sum())
            reverse = int((hulls[top.id] & bottom.mask).sum())
            cover = covered / int(hulls[bottom.id].sum())
            if cover >= config.ON_TOP_MIN_COVER and covered >= config.ON_TOP_RATIO * reverse:
                pairs[(top.id, bottom.id)] = round(cover, 3)
    return pairs


def _in_outline(point, outline) -> bool:
    return cv2.pointPolygonTest(np.array(outline, dtype=np.float32), (float(point[0]), float(point[1])), False) >= 0


def measure(scene: SceneFile, to_mm, gripper_mm: float = config.GRIPPER_MAX_OPENING_MM) -> SceneFile:
    """Fill the measured fields of every object. to_mm(u, v) -> (x, y) on the table plane in mm."""
    hand_near = None
    if scene.hand_present:
        k = 2 * config.HAND_TOUCH_PX + 1
        hand_near = cv2.dilate(scene.hand.astype(np.uint8), np.ones((k, k), np.uint8)).astype(bool)
    for o in scene.objects:
        o.centre_px = centroid(o.mask)
        o.pick_px = pick_point(o.mask)
        o.size_mm = size_mm(o.mask, to_mm)
        o.width_mm = o.size_mm[1]
        o.in_tray = next((t.id.replace("_tray", "") for t in scene.trays if _in_outline(o.centre_px, t.outline_px)), None)
        o.on_top_of, o.under = [], []
        o.held_by_hand = bool(hand_near is not None and (hand_near & o.mask).any())
    for (top, bottom) in stacking(scene.objects):
        if not graspable(scene.get(top), gripper_mm):
            continue   # a dish is never "on" something: see config.py
        scene.get(top).on_top_of.append(bottom)
        scene.get(bottom).under.append(top)
    return scene


def graspable(o: SceneObject, gripper_mm: float = config.GRIPPER_MAX_OPENING_MM) -> bool:
    return o.width_mm <= gripper_mm * config.WIDTH_TOLERANCE


def relations(scene: SceneFile, to_mm) -> dict:
    """Facts that involve more than one object, keyed by object id:
        rank       {"of": n, "from_left": i, "from_right": j} among objects with the same colour and label
        nearest    [[other_id, mm between the centres], ...] nearest first
        left_of, right_of, above, below   id of the nearest object found in that direction from this one
                                          (within 45 degrees, as seen in the photo), or None
    plus "left_to_right": all ids ordered by the image column of their centre."""
    objs = scene.objects
    out = {o.id: {"rank": None, "nearest": []} for o in objs}
    groups: dict = {}
    for o in objs:
        groups.setdefault((o.colour, o.label), []).append(o)
    for members in groups.values():
        members.sort(key=lambda o: o.centre_px[0])
        for i, o in enumerate(members):
            if len(members) > 1:
                out[o.id]["rank"] = {"of": len(members), "from_left": i + 1, "from_right": len(members) - i}
    centre_mm = {o.id: np.array(to_mm(*o.centre_px), dtype=float) for o in objs}
    for a in objs:
        gaps = [(b.id, int(round(float(np.linalg.norm(centre_mm[b.id] - centre_mm[a.id]))))) for b in objs if b.id != a.id]
        out[a.id]["nearest"] = [list(g) for g in sorted(gaps, key=lambda g: g[1])]
        for name, (du, dv) in {"left_of": (-1, 0), "right_of": (1, 0), "above": (0, -1), "below": (0, 1)}.items():
            best = None
            for b in objs:
                if b.id == a.id:
                    continue
                x, y = b.centre_px[0] - a.centre_px[0], b.centre_px[1] - a.centre_px[1]
                along, across = x * du + y * dv, abs(x * dv) + abs(y * du)
                if along > 0 and along >= across and (best is None or along * along + across * across < best[0]):
                    best = (along * along + across * across, b.id)
            out[a.id][name] = best[1] if best else None
    out["left_to_right"] = [o.id for o in sorted(objs, key=lambda o: o.centre_px[0])]
    return out
