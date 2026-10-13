"""Every threshold of the pipeline, each with the reason for its value."""

# Gripper opening from calibration.json; kept here as the default when a scene has no calibration value.
GRIPPER_MAX_OPENING_MM = 85.0

# Widths are measured on the table plane from a mask in a single 2D photo. The top of an object of
# height h appears larger by H / (H - h) with the camera H = 0.99 m above the table, and off-axis the
# mask also includes the visible side wall. Dev scenes: red gears 84 to 88 mm, mug 90 to 102 mm (both
# are pickable), saucer 112 to 126 mm, plate 185 mm. An object counts as too wide for the gripper only
# above this factor times the opening.
WIDTH_TOLERANCE = 1.25

# A lies on B when the masks touch, A fills at least this share of the convex hull of B's visible
# mask, and B fills much less (in pixels) of A's hull. Clear stacks in the dev scenes: 0.07 to 0.21.
# Objects standing side by side: at most 0.02. Between those lie a tall object that hides the edge of
# a flat neighbour in perspective (0.05 to 0.07); these count as "on top": moving the upper object
# first is then unnecessary but safe, while missing a real stack means grasping a buried object.
ON_TOP_MIN_COVER = 0.04
ON_TOP_RATIO = 3.0
CONTACT_MIN_PX = 100     # overlap of the two masks after growing each by 2 px
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
