"""Sensorless cube-state estimation for the continuous-yaw LEAP Hand task.

This subpackage is an additive research extension.  It never modifies the
original ``Mjlab-Leap-Right-HandCube-MBZUAI-Rotate`` task, its rewards, or the
pretrained ``model_400.pt`` checkpoint.

Modules
-------
- ``layout``     observation/label layouts, joint and fingertip ordering.
- ``rotations``  quaternion / rotation-matrix / 6D utilities and the cube's
                 24-element rotational symmetry group.
- ``model``      ``CubeStateEstimator`` (MLP baseline) and save/load helpers.
- ``losses``     combined supervised loss and evaluation metrics.
- ``dataset``    on-disk dataset format written by the collector.
- ``policy``     estimator-augmented actor for RSL-RL PPO (needs rsl_rl).
- ``migration``  checkpoint migration from the 320-input actor.
- ``deploy``     robot-side inference: joint history -> state + action.
"""

from .layout import (  # noqa: F401
  ACTOR_OBS_DIM,
  FINGERTIP_NAMES,
  HISTORY_LENGTH,
  JOINT_NAMES,
  NUM_FINGERTIPS,
  NUM_JOINTS,
)
