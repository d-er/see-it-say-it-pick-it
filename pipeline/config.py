"""Every threshold of the pipeline, each with the reason for its value."""

# Gripper opening from calibration.json; kept here as the default when a scene has no calibration value.
GRIPPER_MAX_OPENING_MM = 85.0

# Widths are measured on the table plane from a mask in a single 2D photo. The top of an object of
# height h appears larger by H / (H - h) with the camera H = 0.99 m above the table, and off-axis the
# mask also includes the visible side wall. Dev scenes: red gears 84 to 88 mm, mug 90 to 107 mm (both
# are pickable), saucer 112 to 126 mm, plate 185 mm. An object counts as too wide for the gripper only
# above this factor times the opening.
WIDTH_TOLERANCE = 1.3

# A lies on B when the masks are next to each other, A fills at least this share of B's estimated full
# outline, and B fills much less (in pixels) of A's convex hull. B's outline is the convex hull of its
# visible mask and, if B is round, also the smallest circle around it. Clear stacks in the dev scenes:
# 0.06 to 0.35. Objects standing side by side: at most 0.036. Between those lie a tall object that hides
# the edge of a flat neighbour in perspective (0.045 to 0.063); these count as "on top": moving the upper
# object first is then unnecessary but safe, while missing a real stack means grasping a buried object.
ON_TOP_MIN_COVER = 0.04
ON_TOP_RATIO = 3.0
CONTACT_GROW_PX = 10     # masks from the segmenter leave a gap of a few pixels (shadow) between touching objects
CONTACT_MIN_PX = 100     # overlap of the two masks after growing each by CONTACT_GROW_PX
ROUND_MIN_FILL = 0.75    # an object is round if it fills this share of its enclosing circle, counted where no
                         # other object is. Dev: plates, saucers, mandarins 0.93 to 0.98, red gears 0.77 to 0.87
# An object too wide for the gripper (plate, saucer) is never reported as lying on another object: the
# raised rim of a dish overhangs its neighbours in a top view, and the robot could not clear it anyway.

# A mask within this many pixels of the hand mask counts as held or touched by the hand (1 px = 0.25 mm).
HAND_TOUCH_PX = 12

# Apertus server (vLLM). Weights take 17.2 GB; the rest of the share is KV cache.
APERTUS_MODEL = "swiss-ai/Apertus-v1.5-8B"
GPU_MEMORY_UTILIZATION_MAX = 0.85   # weights 17.2 GB + 1.2 GB KV cache for MAX_MODEL_LEN; requests are text only
GPU_HEADROOM_GB = 2.0               # left free for whatever else uses the GPU
MAX_MODEL_LEN = 10000               # a scene with ten objects plus two corrections needs about 6,000 tokens
SERVER_START_TIMEOUT_S = 600

# Apertus requests.
TEMPERATURE = 0.0
SEED = 0
MAX_TOKENS = 700
REQUEST_TIMEOUT_S = 120
MAX_RETRIES = 2          # validator corrections sent back to Apertus per instruction

# ---------------------------------------------------------------- perception
# (model id, revision). Pinned so that a later upload to the Hub cannot change the result.
# Grounding DINO instead of LLMDet: with the transformers fork that Apertus v1.5 needs, LLMDet loads
# without a warning but returns misplaced boxes.
DETECTOR_MODEL = ("IDEA-Research/grounding-dino-base", "12bdfa3120f3e7ec7b434d90674b3396eccf88eb")
SEGMENTER_MODEL = ("facebook/sam2.1-hiera-large", "665f8e2ad61cf5f53d65644ff27c8ee525124610")
NAMER_MODEL = ("google/siglip2-so400m-patch14-384", "e8e487298228002f3d8a82e0cd5c8ea9c567f57f")

# Generic words, one detector pass each. None of them names a class of the public scenes: the
# detector only has to say "there is a thing here"; the noun comes from the naming model.
PROPOSAL_PROMPTS = ["object", "toy", "tool", "part", "fruit", "dish"]
BOX_THRESHOLD = 0.3          # 0.2 was tried: mugs split into pieces, gear hubs and plate print become objects
TILES_PER_SIDE = 3           # besides the whole image: 3 x 3 tiles, each this share of the width and
TILE_SHARE = 0.5             # height, so the detector sees small and stacked objects twice as large.
                             # 2 x 2 tiles of 2/3 missed an object lying on another one in two dev scenes.
TILE_EDGE_PX = 12            # a box this close to an inner tile edge is an object cut by the tile
MIN_BOX_PX = 40              # 10 mm on the table; smaller boxes are screw holes and print
BOX_DUPLICATE_IOU = 0.8
SAM_BATCH = 32

TRAY_THRESHOLD = 0.3
TRAY_MIN_SHARE = 0.10        # a tray covers between a tenth and
TRAY_MAX_SHARE = 0.60        # six tenths of the image
GREY_TRAY_PROMPT = "tray"
BLACK_TRAY_PROMPTS = ["black mat", "black surface"]   # tried in this order; the first large box that is not the grey tray
TRAY_SAME_IOU = 0.3
TRAY_MARGIN_PX = 40          # an object whose centre is this far outside a tray outline still counts

HAND_PROMPTS = ["hand", "arm"]
HAND_THRESHOLD = 0.5         # a real hand scores 0.86; the best false "hand" in the dev scenes 0.31
HAND_OVERLAP = 0.5           # a mask that lies half in the hand mask is part of the hand (a watch)

MIN_OBJECT_PX = 2500         # about 12 x 12 mm
MAX_OBJECT_SHARE = 0.5       # of the smaller tray: larger masks are the tray floor or the mat
MASK_DUPLICATE_IOU = 0.7
CONTAINED = 0.85             # share of a mask inside another one to count as lying in it
GROUP_MEMBER_MAX = 0.8       # a member of a group mask is at most this share of it
GROUP_FILL = 0.7             # the members together fill at least this share of the group mask
MIN_CONTRAST = 18            # CIELAB distance between a mask and the band around it. Dev: ejector marks on the
                             # tray floor 3 to 4, scratches on the mat 9; the weakest real object 32
SURROUNDED = 0.9             # share of the band around a mask that lies in a larger mask for it to count as
                             # enclosed. Dev: saucer middle and plate print 1.00; a gear lying on a gear 0.57
ENCLOSED_NAME_MIN = 0.5      # an enclosed mask is kept as an object only with a name at least this sure and
                             # different from the outer object's. Dev: enclosed pieces score at most 0.12

# Recall net: SAM 2.1 "segment everything" for objects the detector did not propose.
RECALL_POINTS_PER_SIDE = 32  # the SAM default
RECALL_MAX_COVERED = 0.3     # share of an automatic mask that may lie on known objects
RECALL_MAX_ASPECT = 8        # longer strips are tray rims (dev: 64 x 1570 px)

CROP_SCALE = 1.2             # naming crop: the object's box enlarged by this factor
CROP_BLUR_SIGMA = 25         # px; everything outside the mask is blurred, so that a neighbour or the object
                             # underneath is not named instead (a mug standing on a gear was called "gear")
NAME_MIN_SCORE = 0.0         # below this the object is called UNKNOWN_LABEL
UNKNOWN_LABEL = "object"

# Colour name from the pixels (CIELAB chroma, lightness 0 to 100; HSV hue in degrees).
CHROMA_MIN = 18              # below: no colour (white, grey, black)
COLOURED_SHARE = 0.5         # share of coloured pixels for the object to have a colour
BLACK_MAX_L = 25
GREY_MAX_L = 45              # darker than this (and not black): grey
WHITE_MIN_L = 75             # lighter than this: white. In between lightness does not decide: the dev scenes
                             # have grey metal parts at 55 to 66 and white objects at 60 to 70, because the
                             # light is uneven. There the naming model is asked for the material instead:
MATERIALS = {"a photo of something made of white plastic": "white",
             "a photo of something made of white porcelain": "white",
             "a photo of something made of grey metal": "grey"}
BROWN_MAX_VALUE = 110
HUES = [("red", 14), ("orange", 30), ("yellow", 70), ("green", 165), ("blue", 265), ("purple", 300),
        ("pink", 335), ("red", 361)]
