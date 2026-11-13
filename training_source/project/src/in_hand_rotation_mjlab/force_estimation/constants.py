"""Fixed orderings for the fingertip force estimator."""

from __future__ import annotations

# Output order requested for the estimator: [thumb, index, middle, ring].
FORCE_FINGERS = ("th", "if", "mf", "rf")
FORCE_FINGER_NAMES = ("thumb", "index", "middle", "ring")
NUM_FORCE_FINGERS = 4

# A "fingertip" is the whole distal link of each finger.  In the LEAP model the
# distal bodies ``<f>_ds`` carry four cube-collidable geoms: the ``<f>_tip`` mesh
# (the only geom used by the original ``fingertip_cube_contact`` sensor) and the
# boxes ``<f>_ds_collision_{1,2,3}``.  All of them are included.
DISTAL_BODIES = tuple(f"{f}_ds" for f in FORCE_FINGERS)

FORCE_SENSOR_NAME = "fe_fingertip_force"
# Contact slots kept per finger by the sensor (top-k by force).  The collector
# records the pre-reduction match count and flags any row where it exceeds this.
FORCE_SENSOR_SLOTS = 8

# Label used for "in contact" in evaluation (interval-mean normal force, N).
CONTACT_THRESHOLD_N = 0.05
