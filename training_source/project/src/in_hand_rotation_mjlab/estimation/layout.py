"""Fixed layouts shared by the collector, estimator, policy and deployment code.

Actor observation (320 values, verified against MJLab's ObservationManager in
``tests/test_estimation_env.py``)::

    [0:160]   joint_pos_rel           10 frames x 16 joints, oldest -> newest
                                      (measured joint position - default pose)
    [160:320] prev_commanded_joint_pos 10 frames x 16 joints, oldest -> newest
                                      (absolute commanded target, rad)

The group-level ``history_length=10`` in MJLab keeps a per-term history and
flattens each term before concatenation, which yields the term-major layout
above.  ``policy/yaw_policy.py`` (the hardware runner) uses the same layout.

Estimator labels (all expressed in the palm frame that the privileged critic
already uses: origin = centroid of the ``palm_collision_*`` geoms, axes = palm
body frame)::

    pos      (3)  cube centre position, m
    quat     (4)  cube orientation, wxyz, unit norm
    lin_vel  (3)  cube linear velocity relative to the palm centre, m/s
    ang_vel  (3)  cube angular velocity relative to the palm, rad/s
    contact  (4)  fingertip-cube contact indicator (if, mf, rf, th)
"""

from __future__ import annotations

NUM_JOINTS = 16
HISTORY_LENGTH = 10
ACTOR_OBS_DIM = 2 * NUM_JOINTS * HISTORY_LENGTH  # 320

# Policy / simulator joint order (``policy/yaw_policy.py`` "server order").
JOINT_NAMES = (
  "if_mcp", "if_rot", "if_pip", "if_dip",
  "mf_mcp", "mf_rot", "mf_pip", "mf_dip",
  "rf_mcp", "rf_rot", "rf_pip", "rf_dip",
  "th_cmc", "th_axl", "th_mcp", "th_ipl",
)

# Order of the ``fingertip_cube_contact`` sensor slots.  The task's own
# metrics (``if_contact`` ... ``th_contact``) use exactly this indexing; the
# collector additionally checks the resolved geom names at runtime.
FINGERTIP_NAMES = ("if", "mf", "rf", "th")
NUM_FINGERTIPS = len(FINGERTIP_NAMES)

# Slices of the actor observation.
JOINT_POS_REL_SLICE = slice(0, NUM_JOINTS * HISTORY_LENGTH)
PREV_COMMAND_SLICE = slice(NUM_JOINTS * HISTORY_LENGTH, ACTOR_OBS_DIM)

# Label field sizes in dataset files.
LABEL_FIELDS = {
  "pos": 3,
  "quat": 4,
  "lin_vel": 3,
  "ang_vel": 3,
  "contact": NUM_FINGERTIPS,
}

# Policy-feature layouts produced by ``CubeStateEstimator.policy_features``.
FEATURE_MODES = {
  # position(3) + 6D rotation(6) + lin vel(3) + ang vel(3) + contact prob(4)
  "full": 19,
  # contact probabilities only (Experiment D)
  "contact": NUM_FINGERTIPS,
}


def split_actor_obs(obs):
  """Return (joint_pos_rel, prev_command) as [..., 10, 16] views."""
  lead = obs.shape[:-1]
  q = obs[..., JOINT_POS_REL_SLICE].reshape(*lead, HISTORY_LENGTH, NUM_JOINTS)
  c = obs[..., PREV_COMMAND_SLICE].reshape(*lead, HISTORY_LENGTH, NUM_JOINTS)
  return q, c
