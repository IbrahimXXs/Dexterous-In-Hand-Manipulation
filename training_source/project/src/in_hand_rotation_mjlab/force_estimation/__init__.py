"""Sensorless fingertip contact-force estimation (independent of the cube-state study).

Input : the deployable 320-D actor observation (``estimation/layout.py``).
Output: compressive normal contact force between each fingertip and the cube,
        in newtons, ordered ``FORCE_FINGERS`` = (thumb, index, middle, ring).

MuJoCo contact forces are used only as supervised labels; they are never inputs.
"""

from .constants import FORCE_FINGERS, FORCE_FINGER_NAMES, NUM_FORCE_FINGERS  # noqa: F401
