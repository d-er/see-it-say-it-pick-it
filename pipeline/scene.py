"""Scene file: the contract between perception and reasoning.

One JSON file per scene. Perception writes it, geometry adds the measured facts, reasoning reads it.
Nothing in here knows where the masks came from.

    {
      "scene": "scene_06",
      "image_size": [2448, 2048],                      # [width, height]
      "trays": [{"id": "grey_tray", "outline_px": [[u, v], ...]}, ...],
      "hand": {"size": [h, w], "counts": [...]} | null,   # mask of a human hand or arm, if one is visible
      "objects": [
        {
          "id": "red_gear_1",            # unique in the scene
          "label": "gear",               # noun, without the colour
          "colour": "red",
          "mask": {"size": [h, w], "counts": [...]},     # visible pixels only, run-length encoded
          # filled by geometry.measure():
          "centre_px": [u, v],           # centroid of the mask
          "pick_px": [u, v],             # point deepest inside the mask (distance transform)
          "size_mm": [long, short],      # sides of the minimum-area rectangle on the table plane
          "width_mm": 61.0,              # the short side: what the gripper has to span
          "in_tray": "grey" | "black" | null,
          "on_top_of": ["red_gear_2"],   # ids this object lies on
          "under": [],                   # ids lying on this object
          "held_by_hand": false
        }
      ]
    }

Pixels are [u, v] = [column, row], origin top-left, full-resolution image.
Mask encoding: row-major run lengths, starting with a run of zeros (which may be empty).
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np


def encode_mask(mask: np.ndarray) -> dict:
    flat = np.asarray(mask, dtype=bool).ravel()
    change = np.flatnonzero(flat[1:] != flat[:-1]) + 1
    counts = np.diff(np.concatenate(([0], change, [flat.size]))).tolist()
    if flat.size and flat[0]:
        counts.insert(0, 0)
    return {"size": list(mask.shape), "counts": counts}


def decode_mask(rle: dict) -> np.ndarray:
    counts = np.asarray(rle["counts"], dtype=np.int64)
    values = np.zeros(len(counts), dtype=bool)
    values[1::2] = True
    return np.repeat(values, counts).reshape(rle["size"])


@dataclass
class Tray:
    id: str
    outline_px: list


@dataclass
class SceneObject:
    id: str
    label: str
    colour: str
    mask: np.ndarray
    centre_px: list | None = None
    pick_px: list | None = None
    size_mm: list | None = None
    width_mm: float | None = None
    in_tray: str | None = None
    on_top_of: list = field(default_factory=list)
    under: list = field(default_factory=list)
    held_by_hand: bool = False

    @property
    def name(self) -> str:
        return f"{self.colour} {self.label}".strip()


@dataclass
class SceneFile:
    scene: str
    image_size: list
    objects: list
    trays: list = field(default_factory=list)
    hand: np.ndarray | None = None
    relations: dict | None = None      # geometry.relations(); measured at run time, not stored in the file

    @property
    def hand_present(self) -> bool:
        return self.hand is not None and bool(self.hand.any())

    def get(self, object_id: str) -> SceneObject | None:
        return next((o for o in self.objects if o.id == object_id), None)

    def to_json(self) -> dict:
        objects = []
        for o in self.objects:
            d = asdict(o)
            d["mask"] = encode_mask(o.mask)
            objects.append(d)
        return {"scene": self.scene, "image_size": list(self.image_size),
                "hand_present": self.hand_present,
                "trays": [asdict(t) for t in self.trays],
                "hand": encode_mask(self.hand) if self.hand is not None else None,
                "objects": objects}

    @classmethod
    def from_json(cls, d: dict) -> "SceneFile":
        objects = [SceneObject(**{**o, "mask": decode_mask(o["mask"])}) for o in d["objects"]]
        ids = [o.id for o in objects]
        if len(set(ids)) != len(ids):
            raise ValueError(f"{d['scene']}: object ids are not unique: {ids}")
        w, h = d["image_size"]
        for o in objects:
            if o.mask.shape != (h, w):
                raise ValueError(f"{d['scene']}: mask of {o.id} is {o.mask.shape}, image is {(h, w)}")
        return cls(scene=d["scene"], image_size=list(d["image_size"]), objects=objects,
                   trays=[Tray(**t) for t in d.get("trays", [])],
                   hand=decode_mask(d["hand"]) if d.get("hand") else None)

    def save(self, path) -> None:
        Path(path).write_text(json.dumps(self.to_json()))

    @classmethod
    def load(cls, path) -> "SceneFile":
        return cls.from_json(json.loads(Path(path).read_text()))


if __name__ == "__main__":   # python -m pipeline.scene examples/example_scene.json
    import sys

    s = SceneFile.load(sys.argv[1])
    print(f"{s.scene}: image {s.image_size[0]}x{s.image_size[1]}, trays {[t.id for t in s.trays]}, hand {s.hand_present}")
    for o in s.objects:
        print(f"  {o.id}: {o.name}, {int(o.mask.sum())} px, on_top_of={o.on_top_of}, under={o.under}")
