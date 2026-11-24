"""Digital twin rendering (offscreen MuJoCo) + camera-frame cube widget.

* Hand: the kit's ``HandScene`` (models/scene_right_mbzuai_cube.xml, unchanged), joints set
  from measured telemetry; no physics is stepped.
* Proprioceptive cube estimate: the scene's cube body, see-through orange (HandScene's own
  styling: face identity is not observable from proprioception).
* Vision cube: drawn as extra visual geoms (face plates in the layout colours, white quiet
  zones, black marker squares) at the camera measurement transformed to the palm frame -
  ONLY when the camera-to-palm transform is calibrated.  Otherwise it is not drawn in the
  hand view at all; the camera-frame widget shows its orientation, labelled unregistered.
"""

from __future__ import annotations

import os

import numpy as np

from .geometry import CubeMarkerModel
from .synthetic import CubeRenderer
from .intrinsics import STATUS_CHECKERBOARD, CameraIntrinsics
from .telemetry import kit_import


class TwinRenderer:
  def __init__(self, model: CubeMarkerModel, width: int = 480, height: int = 360):
    import mujoco

    self.mj = mujoco
    hs = kit_import("hand_scene")
    lv = kit_import("live_viewer")
    self.scene = hs.HandScene()
    self.default_q = self.scene.data.qpos[self.scene.qadr].copy()
    m = self.scene.model
    m.vis.global_.offwidth = max(m.vis.global_.offwidth, width)
    m.vis.global_.offheight = max(m.vis.global_.offheight, height)
    self.renderer = mujoco.Renderer(m, height, width)
    self.cam = lv.default_camera()
    self.cube_model = model
    self.cube_gid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "cube_geom")
    self.cube_rgba = m.geom_rgba[self.cube_gid].copy()
    self.size = (width, height)

  def render(self, q_server=None, proprio=None, vision_palm=None) -> np.ndarray:
    """q_server [16] or None; proprio (pos_palm, quat_palm_wxyz) or None; vision_palm (p, R) or None. RGB."""
    mj, sc = self.mj, self.scene
    sc.set_hand(self.default_q if q_server is None else np.asarray(q_server, np.float64))
    if proprio is not None:
      sc.model.geom_rgba[self.cube_gid] = self.cube_rgba
      sc.set_cube_palm(np.asarray(proprio[0]), np.asarray(proprio[1]))
    else:
      sc.model.geom_rgba[self.cube_gid][3] = 0.0
    self.renderer.update_scene(sc.data, camera=self.cam)
    if vision_palm is not None:
      c, Rp = sc.palm_frame()
      p_w = c + Rp @ np.asarray(vision_palm[0])
      R_w = Rp @ np.asarray(vision_palm[1])
      self._draw_cube(self.renderer.scene, p_w, R_w)
    return self.renderer.render()

  def _box(self, scn, pos, R, half_sizes, rgba):
    if scn.ngeom >= scn.maxgeom:
      return
    g = scn.geoms[scn.ngeom]
    scn.ngeom += 1
    self.mj.mjv_initGeom(g, self.mj.mjtGeom.mjGEOM_BOX, np.asarray(half_sizes, np.float64), np.asarray(pos, np.float64),
                         np.asarray(R, np.float64).reshape(-1), np.asarray(rgba, np.float32))

  def _draw_cube(self, scn, p_w, R_w):
    m = self.cube_model
    h, s = m.half_m, m.marker_size_m
    for f in m.faces.values():
      Rf = R_w @ f.R_cube_marker  # plate frame: x, y in the face, z = outward normal
      col = np.array(f.rgb) / 255.0
      self._box(scn, p_w + R_w @ f.face_center, Rf, (h, h, 0.0006), (*col, 0.95))
      self._box(scn, p_w + R_w @ (f.marker_center + 0.0007 * f.normal), Rf, (0.68 * s, 0.68 * s, 0.0002), (1, 1, 1, 1))
      self._box(scn, p_w + R_w @ (f.marker_center + 0.0010 * f.normal), Rf, (0.5 * s, 0.5 * s, 0.0002), (0.05, 0.05, 0.05, 1))


class CubeWidget:
  """The measured camera-frame orientation, drawn centred: what the camera sees, face letters + IDs."""

  def __init__(self, model: CubeMarkerModel, size: int = 240):
    import cv2

    self.cv2 = cv2
    self.model = model
    K = np.array([[size * 2.2, 0, size / 2 - 0.5], [0, size * 2.2, size / 2 - 0.5], [0, 0, 1.0]])
    self.intr = CameraIntrinsics(K, np.zeros(5), (size, size), STATUS_CHECKERBOARD, "widget")
    self.r = CubeRenderer(model, self.intr, supersample=2, background_bgr=(32, 32, 32))
    self.size = size

  def render(self, R: np.ndarray | None) -> np.ndarray:
    cv2 = self.cv2
    if R is None:
      img = np.full((self.size, self.size, 3), 32, np.uint8)
      cv2.putText(img, "no pose", (self.size // 2 - 40, self.size // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (160, 160, 160), 1,
                  cv2.LINE_AA)
      return img
    t = np.array([0.0, 0.0, 0.26])
    fr = self.r.render(np.asarray(R), t)
    img = fr.bgr.copy()
    K = self.intr.camera_matrix

    def proj(p):
      pc = np.asarray(R) @ p + t
      return (int(K[0, 0] * pc[0] / pc[2] + K[0, 2]), int(K[1, 1] * pc[1] / pc[2] + K[1, 2]))

    for i in fr.visible_faces:
      f = self.model.faces[i]
      u, v = proj(f.face_center - 0.6 * (f.marker_center - f.face_center))  # free part of the face, opposite the marker
      cv2.putText(img, f"{f.letter}{i}", (u - 12, v + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
      cv2.putText(img, f"{f.letter}{i}", (u - 12, v + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    o = proj(np.zeros(3))
    for k, col in enumerate(((0, 0, 255), (0, 200, 0), (255, 80, 0))):
      e = np.zeros(3)
      e[k] = 0.045
      cv2.arrowedLine(img, o, proj(e), col, 2, cv2.LINE_AA, tipLength=0.15)
    return img


def ensure_gl_backend() -> None:
  """Offscreen MuJoCo needs EGL/OSMesa when there is no display."""
  if "MUJOCO_GL" not in os.environ and not os.environ.get("DISPLAY") and not os.environ.get("WAYLAND_DISPLAY"):
    os.environ["MUJOCO_GL"] = "egl"
