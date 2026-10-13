"""Image -> scene file (masks, labels, colours, trays, hand).

Stub: the real stage (open-vocabulary detector + segmenter) is not built yet. It reports a single
unnamed object in the middle of the image so the rest of the pipeline can run end to end.
"""
from __future__ import annotations

import numpy as np

from .scene import SceneFile, SceneObject


def perceive(name: str, rgb: np.ndarray) -> SceneFile:
    h, w = rgb.shape[:2]
    mask = np.zeros((h, w), dtype=bool)
    mask[h // 2 - 40:h // 2 + 40, w // 2 - 40:w // 2 + 40] = True
    return SceneFile(scene=name, image_size=[w, h],
                     objects=[SceneObject(id="object_1", label="object", colour="", mask=mask)])
