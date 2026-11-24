"""Check the cube's physical marker layout against layout.json, from a recording of the real cube.

Frames showing two or more markers measure the relative pose of the faces; the camera need
not be calibrated.  Three analyses:

1. **Relative-rotation vote** (needs only a rough focal length): for every co-visible pair
   (i, j), the relative marker rotation R_i^T R_j (choosing the IPPE solutions closest to an
   axis-aligned rotation) is voted over all frames and over a range of focal lengths, and
   compared with layout.json's prediction.  A marker printed in another orientation, or
   stuck on another face, changes this vote; the in-face offset does not.
2. **Reconstruction** of every marker's frame in the cube frame from the votes (spanning
   tree + cycle checks), anchored so that the faces defining the cube frame (B = +X,
   Z = +Y, M = +Z by colour/letter) keep their layout.json normals.
3. **Bundle adjustment** (per-frame poses + focal f, principal point, radial k1 and the
   in-face marker offsets: shared or per face) under (a) layout.json's axes and (b) the
   reconstructed axes.  The residual says which geometry explains the images.

Nothing is written back to layout.json.  ``run`` writes a report and, separately, a
*candidate* measured layout (same schema as layout.json) + print layout with status
``measured_from_recording_unconfirmed``; the tracker uses it only when explicitly selected.
The intrinsics estimated here (status ``estimated_from_cube_markers``) are approximate.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np

from .geometry import (CORNERS_UNIT, LAYOUT_PATH, CubeMarkerModel, cube_symmetry_rotations, load_cube_model,
                       matrix_to_rotvec, rotation_angle_deg, rotvec_to_matrix)
from .intrinsics import STATUS_MARKER_ESTIMATE, CameraIntrinsics
from .tracker import CubeTracker, MarkerDetection, MarkerDetector, TrackerConfig

AXIS_NAMES = ("+X", "-X", "+Y", "-Y", "+Z", "-Z")


@dataclass
class FrameObs:
  index: int
  time_s: float
  dets: list[MarkerDetection]


def axis_name(v: np.ndarray) -> str:
  k = int(np.argmax(np.abs(v)))
  return ("+" if v[k] > 0 else "-") + "XYZ"[k]


def collect_detections(video: Path, model: CubeMarkerModel, max_frames: int | None = None):
  import cv2

  cap = cv2.VideoCapture(str(video))
  if not cap.isOpened():
    raise FileNotFoundError(f"cannot open video {video}")
  fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
  det = MarkerDetector(model)
  frames, size = [], None
  stats = {"frames": 0, "unknown_id_detections": 0, "markers_per_frame": Counter()}
  i = 0
  while True:
    ok, img = cap.read()
    if not ok or (max_frames and i >= max_frames):
      break
    size = (img.shape[1], img.shape[0])
    dets, unknown, _, _ = det.detect(img)
    stats["unknown_id_detections"] += len(unknown)
    stats["markers_per_frame"][len(dets)] += 1
    frames.append(FrameObs(i, i / fps, dets))
    i += 1
  stats["frames"] = i
  stats["fps"] = fps
  stats["markers_per_frame"] = {int(k): v for k, v in sorted(stats["markers_per_frame"].items())}
  return frames, size, stats


# ----------------------------------------------------------------------------- 1. votes
def relative_rotation_votes(frames, model: CubeMarkerModel, size, focal_factors=(0.8, 1.1, 1.4)) -> dict:
  import cv2

  P24 = cube_symmetry_rotations()
  s = model.marker_size_m
  sq = np.array([[-s / 2, s / 2, 0], [s / 2, s / 2, 0], [s / 2, -s / 2, 0], [-s / 2, -s / 2, 0]])
  out = {}
  for ff in focal_factors:
    f = ff * size[0]
    K = np.array([[f, 0, size[0] / 2], [0, f, size[1] / 2], [0, 0, 1.0]])
    votes: dict = defaultdict(Counter)
    resid: dict = defaultdict(list)
    for fr in frames:
      if len(fr.dets) < 2:
        continue
      sols = {}
      for d in fr.dets:
        _, rv, _, _ = cv2.solvePnPGeneric(sq, d.corners, K, None, flags=cv2.SOLVEPNP_IPPE_SQUARE)
        sols[d.marker_id] = [rotvec_to_matrix(r) for r in rv]
      ids = sorted(sols)
      for a in range(len(ids)):
        for b in range(a + 1, len(ids)):
          i, j = ids[a], ids[b]
          best = None
          for Ri in sols[i]:
            for Rj in sols[j]:
              Rr = Ri.T @ Rj
              angs = [rotation_angle_deg(Rr, P) for P in P24]
              k = int(np.argmin(angs))
              if best is None or angs[k] < best[0]:
                best = (angs[k], k)
          votes[(i, j)][best[1]] += 1
          resid[(i, j)].append(best[0])
    out[float(f)] = (votes, resid)
  pairs = {}
  for (i, j) in sorted({p for v, _ in out.values() for p in v}):
    allv = Counter()
    for v, _ in out.values():
      allv.update(v.get((i, j), {}))
    k, n = allv.most_common(1)[0]
    pred = model.faces[i].R_cube_marker.T @ model.faces[j].R_cube_marker
    kp = int(np.argmin([rotation_angle_deg(pred, P) for P in P24]))
    stable = all(v[(i, j)].most_common(1)[0][0] == k for v, _ in out.values() if (i, j) in v)
    pairs[f"{i}-{j}"] = {"frames": int(sum(out[next(iter(out))][0][(i, j)].values())), "observed": P24[k].tolist(),
                         "agreement": n / sum(allv.values()), "stable_across_focal": bool(stable),
                         "layout_json_predicts": P24[kp].tolist(), "matches_layout_json": bool(k == kp),
                         "median_residual_deg": float(np.median(out[next(iter(out))][1][(i, j)]))}
  return pairs


# ----------------------------------------------------------------------------- 2. reconstruction
def reconstruct_marker_frames(pairs: dict, model: CubeMarkerModel) -> dict:
  P24 = cube_symmetry_rotations()
  edges = {}
  for key, p in pairs.items():
    i, j = (int(x) for x in key.split("-"))
    edges[(i, j)] = (np.asarray(p["observed"]), p["frames"])
  nodes = sorted({i for e in edges for i in e})
  # spanning tree by decreasing support, from the most connected node
  deg = Counter()
  for (i, j), (_, n) in edges.items():
    deg[i] += n
    deg[j] += n
  root = max(nodes, key=lambda k: deg[k])
  F = {root: model.faces[root].R_cube_marker}
  tree, cycles = [], []
  for (i, j), (O, n) in sorted(edges.items(), key=lambda e: -e[1][1]):
    pass
  changed = True
  used = set()
  while changed:
    changed = False
    for (i, j), (O, n) in sorted(edges.items(), key=lambda e: -e[1][1]):
      if (i, j) in used:
        continue
      if i in F and j not in F:
        F[j] = F[i] @ O
      elif j in F and i not in F:
        F[i] = F[j] @ O.T
      else:
        continue
      used.add((i, j))
      tree.append(f"{i}-{j}")
      changed = True
  for (i, j), (O, n) in edges.items():
    if (i, j) not in used and i in F and j in F:
      cycles.append({"pair": f"{i}-{j}", "residual_deg": rotation_angle_deg(F[i].T @ F[j], O)})
  # anchor: global rotation G (24 options) making the frame-defining faces keep their layout normals
  frame_faces = [k for k, f in model.faces.items() if f.name in ("+X", "+Y", "+Z")]

  def score(G):
    a = sum(np.allclose(G @ F[k][:, 2], model.faces[k].normal) for k in frame_faces if k in F)
    b = sum(np.allclose(G @ F[k][:, 2], model.faces[k].normal) for k in F)
    c = sum(np.allclose(G @ F[k], model.faces[k].R_cube_marker) for k in F)
    return (a, b, c)

  G = max(P24, key=score)
  Fa = {k: G @ v for k, v in F.items()}
  per = {}
  for k in sorted(model.faces):
    f = model.faces[k]
    if k not in Fa:
      per[k] = {"observed": False}
      continue
    R = Fa[k]
    per[k] = {"observed": True, "letter": f.letter, "color": f.color,
              "layout_json": {"normal": f.name, "marker_x": axis_name(f.marker_x), "marker_y": axis_name(f.marker_y)},
              "recording": {"normal": axis_name(R[:, 2]), "marker_x": axis_name(R[:, 0]), "marker_y": axis_name(R[:, 1])},
              "normal_matches": bool(np.allclose(R[:, 2], f.normal)),
              "axes_match": bool(np.allclose(R, f.R_cube_marker)),
              "in_plane_rotation_deg": (float(np.degrees(np.arctan2((f.marker_x @ R[:, 1]), (f.marker_x @ R[:, 0]))))
                                        if np.allclose(R[:, 2], f.normal) else None)}
  return {"frames": {k: v.tolist() for k, v in Fa.items()}, "root": root, "tree": tree, "cycle_checks": cycles,
          "anchor_frame_faces_kept": int(score(G)[0]), "per_marker": per}


# ----------------------------------------------------------------------------- 3. bundle adjustment
def _K(g):
  return np.array([[g[0], 0, g[1]], [0, g[0], g[2]], [0, 0, 1.0]]), np.array([g[3], 0, 0, 0, 0.0])


def _corners(frames_R: dict, half: float, s: float, mid: int, off: np.ndarray) -> np.ndarray:
  R = frames_R[mid]
  mx, my, n = R[:, 0], R[:, 1], R[:, 2]
  mc = half * n + off[0] * mx + off[1] * my
  return np.array([mc + s * (u * mx + v * my) for u, v in CORNERS_UNIT])


def bundle_adjust(frames, size, frames_R: dict, half: float, s: float, g0, offsets0: dict, *, free_intr=("f", "cx", "cy", "k1"),
                  offset_mode: str = "per_face", iters: int = 60, huber_px: float = 1.5, min_markers: int = 2) -> dict:
  """offset_mode: 'fixed' | 'shared' | 'per_face'."""
  import cv2

  intr_names = ("f", "cx", "cy", "k1")
  g = np.asarray(g0, np.float64).copy()
  ids_all = sorted(frames_R)
  off = {k: np.asarray(offsets0.get(k, (0.0, 0.0)), np.float64).copy() for k in ids_all}
  use = [fr for fr in frames if sum(d.marker_id in frames_R for d in fr.dets) >= min_markers]
  # parameter layout of the global vector
  gi = [intr_names.index(n) for n in free_intr]
  if offset_mode == "per_face":
    seen = sorted({d.marker_id for fr in use for d in fr.dets if d.marker_id in frames_R})
    off_params = [(k, a) for k in seen for a in (0, 1)]
  elif offset_mode == "shared":
    off_params = [("all", 0), ("all", 1)]
  else:
    off_params = []
  G = len(gi) + len(off_params)

  def obj_for(fr, offd):
    dd = [d for d in fr.dets if d.marker_id in frames_R]
    return dd, np.concatenate([_corners(frames_R, half, s, d.marker_id, offd[d.marker_id]) for d in dd]), \
      np.concatenate([d.corners for d in dd])

  def model_for(gg):
    K, dist = _K(gg)
    intr = CameraIntrinsics(K, dist, size, STATUS_MARKER_ESTIMATE, "ba")
    return intr

  # init poses with the tracker under the starting geometry
  mdl = _model_from_frames(frames_R, half, s, off)
  tr = CubeTracker(mdl, model_for(g), TrackerConfig(inlier_marker_rms_px=8.0))
  X = {}
  for fr in use:
    dd = [d for d in fr.dets if d.marker_id in frames_R]
    pose, _ = tr.estimate(dd, 1e9)
    if pose is not None and len(pose.inlier_ids) >= min(2, len(dd)) and not pose.ambiguous:
      X[fr.index] = np.concatenate([matrix_to_rotvec(pose.R), pose.t])
  use = [fr for fr in use if fr.index in X]

  def cost_of(gg, offd, XX):
    K, dist = _K(gg)
    c = 0.0
    for fr in use:
      _, obj, img = obj_for(fr, offd)
      proj, _ = cv2.projectPoints(obj, XX[fr.index][:3], XX[fr.index][3:], K, dist)
      e = np.linalg.norm(proj.reshape(-1, 2) - img, axis=1)
      c += float(np.sum(np.where(e <= huber_px, 0.5 * e ** 2, huber_px * (e - 0.5 * huber_px))))
    return c

  def apply(gg, offd, dgl):
    g2 = gg.copy()
    o2 = {k: v.copy() for k, v in offd.items()}
    for j, k in enumerate(gi):
      g2[k] += dgl[j]
    for j, (k, a) in enumerate(off_params):
      if k == "all":
        for kk in o2:
          o2[kk][a] += dgl[len(gi) + j]
      else:
        o2[k][a] += dgl[len(gi) + j]
    return g2, o2

  def valid(offd):
    return all(np.all(np.abs(v) + 0.5 * s <= half + 1e-12) for v in offd.values())

  cost = cost_of(g, off, X)
  lam = 1e-3
  eps = 1e-6
  Sd = np.zeros((G, G))
  for _ in range(iters):
    K, dist = _K(g)
    S = np.zeros((G, G))
    sv = np.zeros(G)
    blocks = {}
    for fr in use:
      dd, obj, img = obj_for(fr, off)
      x6 = X[fr.index]
      proj, J = cv2.projectPoints(obj, x6[:3], x6[3:], K, dist)
      r = (proj.reshape(-1, 2) - img).reshape(-1)
      e = np.linalg.norm(r.reshape(-1, 2), axis=1)
      w = np.repeat(np.where(e <= huber_px, 1.0, huber_px / np.maximum(e, 1e-12)), 2)
      Jg = np.zeros((len(r), G))
      cols = {"f": J[:, 6] + J[:, 7], "cx": J[:, 8], "cy": J[:, 9], "k1": J[:, 10]}
      for j, k in enumerate(gi):
        Jg[:, j] = cols[intr_names[k]]
      for j, (k, a) in enumerate(off_params):
        d_off = {kk: v.copy() for kk, v in off.items()}
        if k == "all":
          for kk in d_off:
            d_off[kk][a] += eps
        elif any(d.marker_id == k for d in dd):
          d_off[k][a] += eps
        else:
          continue
        _, obj2, _ = obj_for(fr, d_off)
        p2, _ = cv2.projectPoints(obj2, x6[:3], x6[3:], K, dist)
        Jg[:, len(gi) + j] = (p2.reshape(-1) - proj.reshape(-1)) / eps
      Jp = J[:, :6]
      blocks[fr.index] = (Jp.T @ (w[:, None] * Jp), Jp.T @ (w[:, None] * Jg), Jp.T @ (w * r))
      S += Jg.T @ (w[:, None] * Jg)
      sv += Jg.T @ (w * r)
    improved = False
    for _ in range(10):
      Sd, sd, inv = S.copy(), sv.copy(), {}
      for k, (A, B, a) in blocks.items():
        Ai = np.linalg.inv(A + lam * np.diag(np.diag(A)) + 1e-12 * np.eye(6))
        inv[k] = Ai
        Sd -= B.T @ Ai @ B
        sd -= B.T @ Ai @ a
      dg = -np.linalg.solve(Sd + lam * np.diag(np.diag(Sd)) + 1e-12 * np.eye(G), sd) if G else np.zeros(0)
      g2, o2 = apply(g, off, dg)
      X2 = {k: X[k] - inv[k] @ (blocks[k][2] + blocks[k][1] @ dg) for k in X}
      if not valid(o2):
        lam *= 5
        continue
      c2 = cost_of(g2, o2, X2)
      if c2 < cost:
        g, off, X, cost = g2, o2, X2, c2
        lam = max(lam / 3, 1e-7)
        improved = True
        break
      lam *= 5
    if not improved or (G and np.max(np.abs(dg)) < 1e-10):
      break
  K, dist = _K(g)
  errs, per_face = [], defaultdict(list)
  for fr in use:
    dd, obj, img = obj_for(fr, off)
    proj, _ = cv2.projectPoints(obj, X[fr.index][:3], X[fr.index][3:], K, dist)
    e = np.linalg.norm(proj.reshape(-1, 2) - img, axis=1)
    errs.append(e)
    for k, d in enumerate(dd):
      per_face[d.marker_id].extend(e[4 * k:4 * k + 4].tolist())
  e = np.concatenate(errs) if errs else np.zeros(0)
  std = None
  if G and len(e):
    try:
      sigma2 = float(np.mean(np.minimum(e, 3 * huber_px) ** 2))
      std = np.sqrt(np.clip(np.diag(sigma2 * np.linalg.inv(Sd)), 0, None)).tolist()
    except np.linalg.LinAlgError:
      std = None
  names = [intr_names[k] for k in gi] + [f"off_{k}_{'xy'[a]}" for k, a in off_params]
  return {"intrinsics": dict(zip(intr_names, g.tolist())), "offsets_m": {int(k): v.tolist() for k, v in off.items()},
          "offset_mode": offset_mode, "free": names, "std": None if std is None else dict(zip(names, std)),
          "frames_used": len(use), "corners": int(len(e)),
          "rms_px": float(np.sqrt(np.mean(e ** 2))) if len(e) else float("nan"),
          "median_px": float(np.median(e)) if len(e) else float("nan"),
          "p95_px": float(np.percentile(e, 95)) if len(e) else float("nan"),
          "per_face_median_px": {int(k): float(np.median(v)) for k, v in sorted(per_face.items())},
          "per_face_corners": {int(k): len(v) for k, v in sorted(per_face.items())}}


def _model_from_frames(frames_R: dict, half: float, s: float, offsets: dict) -> CubeMarkerModel:
  """A CubeMarkerModel with arbitrary marker frames (analysis only; not written anywhere)."""
  from .geometry import FaceSpec, PrintLayout

  base = load_cube_model()
  faces = {}
  for k, R in frames_R.items():
    f = base.faces[k]
    mx, my, n = R[:, 0], R[:, 1], R[:, 2]
    off = np.asarray(offsets.get(k, (0, 0)), np.float64)
    mc = half * n + off[0] * mx + off[1] * my
    corners = np.array([mc + s * (u * mx + v * my) for u, v in CORNERS_UNIT])
    faces[k] = FaceSpec(k, axis_name(n), f.letter, f.color, f.rgb, mx, my, n, half * n, mc, corners)
  return CubeMarkerModel(base.name, base.cube_size_m, base.marker_size_m, base.mass_kg, base.dictionary_name,
                         base.body_frame, faces, PrintLayout(np.zeros(2), "analysis", "analysis", 0.0,
                                                             {k: np.asarray(v) for k, v in offsets.items()}),
                         base.layout_path, None)


def _best_focal_start(frames, size, frames_R, half, s, offsets) -> float:
  mdl = _model_from_frames(frames_R, half, s, offsets)
  use = [fr for fr in frames if len(fr.dets) >= 2]
  best = None
  for ff in np.linspace(0.6, 2.0, 15):
    g = np.array([ff * size[0], size[0] / 2, size[1] / 2, 0.0])
    K, dist = _K(g)
    tr = CubeTracker(mdl, CameraIntrinsics(K, dist, size, STATUS_MARKER_ESTIMATE, "ba"), TrackerConfig(inlier_marker_rms_px=8.0))
    rms = [p.reproj_rms_px for p, _ in (tr.estimate(fr.dets, 1e9) for fr in use) if p is not None and len(p.inlier_ids) >= 2]
    m = float(np.median(rms)) if rms else np.inf
    if best is None or m < best[0]:
      best = (m, ff * size[0])
  return best[1]


# ----------------------------------------------------------------------------- driver
REPO = Path(__file__).resolve().parents[4]


def _rel(p: Path) -> str:
  try:
    return str(Path(p).resolve().relative_to(REPO))
  except ValueError:
    return str(p)


def run(video: Path, out_dir: Path, *, write_candidate: bool = True, max_frames: int | None = None) -> dict:
  base = load_cube_model()
  video_label = _rel(video)
  half, s = base.half_m, base.marker_size_m
  frames, size, stats = collect_detections(video, base, max_frames)
  pairs = relative_rotation_votes(frames, base, size)
  recon = reconstruct_marker_frames(pairs, base)
  layout_R = {k: f.R_cube_marker for k, f in base.faces.items()}
  rec_R = {int(k): np.asarray(v) for k, v in recon["frames"].items()}
  file_off = {k: base.print_layout.offset_for(k) for k in base.faces}
  results = {}
  for geom_name, R_set in (("layout_json_axes", layout_R), ("recording_axes", rec_R)):
    f0 = _best_focal_start(frames, size, R_set, half, s, file_off)
    g0 = np.array([f0, size[0] / 2, size[1] / 2, 0.0])
    fixed_c = bundle_adjust(frames, size, R_set, half, s, g0, {k: (0, 0) for k in R_set}, offset_mode="fixed")
    fixed_p = bundle_adjust(frames, size, R_set, half, s, g0, file_off, offset_mode="fixed")
    g1 = np.array(list(fixed_p["intrinsics"].values()))
    shared = bundle_adjust(frames, size, R_set, half, s, g1, file_off, offset_mode="shared")
    per = bundle_adjust(frames, size, R_set, half, s, np.array(list(shared["intrinsics"].values())),
                        shared["offsets_m"], offset_mode="per_face")
    results[geom_name] = {"offset_centered": fixed_c, "offset_print_layout_file": fixed_p, "offset_shared_free": shared,
                          "offset_per_face_free": per}
  report = {"video": video_label, "image_size_wh": list(size), "detection_stats": stats,
            "relative_rotation_pairs": pairs, "reconstruction": recon, "bundle_adjustment": results,
            "layout_json": _rel(base.layout_path), "print_layout_file": _rel(base.print_layout_path),
            "notes": ["Intrinsics here are self-calibrated from the cube markers of one recording (approximate).",
                      "Pairs never seen together are not tested; faces seen only via one pair are weakly constrained."]}
  out_dir.mkdir(parents=True, exist_ok=True)
  stem = Path(video).stem
  (out_dir / f"layout_check_{stem}.json").write_text(json.dumps(report, indent=1, default=_json_default) + "\n")
  # topology: which faces are opposite each other (frame-independent)
  opp = {}
  for k, R in rec_R.items():
    for j, Rj in rec_R.items():
      if j > k and np.allclose(R[:, 2], -Rj[:, 2]):
        opp[f"{base.faces[k].letter}({k})"] = f"{base.faces[j].letter}({j})"
  report["recording_opposite_faces"] = opp
  report["layout_json_opposite_faces"] = {f"{base.faces[k].letter}({k})": f"{base.faces[j].letter}({j})"
                                          for k in base.faces for j in base.faces
                                          if j > k and np.allclose(base.faces[k].normal, -base.faces[j].normal)}
  (out_dir / f"layout_check_{stem}.json").write_text(json.dumps(report, indent=1, default=_json_default) + "\n")
  if write_candidate:
    # shared offset: one print template for all faces (per-face fit reported, but face 5 is weakly constrained)
    best = results["recording_axes"]["offset_shared_free"]
    write_candidate_layout(out_dir, stem, base, rec_R, best, video_label, recon,
                           results["recording_axes"]["offset_per_face_free"]["offsets_m"])
    g = best["intrinsics"]
    K, dist = _K(np.array([g["f"], g["cx"], g["cy"], g["k1"]]))
    CameraIntrinsics(K, dist, size, STATUS_MARKER_ESTIMATE, f"unknown camera of {Path(video).name}", best["rms_px"],
                     extra={"estimated_from": video_label, "method": "cube_vision.layout_check bundle adjustment "
                            "(recording axes, one shared marker offset)", "formal_std": best["std"],
                            "principal_point_spread_px": _pp_spread(results["recording_axes"]),
                            "warning": "Approximate self-calibration from the cube markers of one recording; NOT a "
                                       "board calibration. The principal point is poorly constrained (see "
                                       "principal_point_spread_px across fits); formal std underestimates the error. "
                                       "Valid only for this recording's resolution."}
                     ).save(out_dir / f"{stem}.estimated.intrinsics.json")
  return report


def _pp_spread(res: dict) -> dict:
  cx = [r["intrinsics"]["cx"] for r in res.values()]
  cy = [r["intrinsics"]["cy"] for r in res.values()]
  f = [r["intrinsics"]["f"] for r in res.values()]
  return {"f": [min(f), max(f)], "cx": [min(cx), max(cx)], "cy": [min(cy), max(cy)]}


def write_candidate_layout(out_dir: Path, stem: str, base: CubeMarkerModel, rec_R: dict, ba: dict, video, recon,
                           per_face_fit: dict | None = None) -> None:
  src = json.loads(Path(base.layout_path).read_text())
  faces = []
  for f in src["faces"]:
    k = int(f["id"])
    g = dict(f)
    if k in rec_R:
      R = rec_R[k]
      g["name"] = axis_name(R[:, 2])
      g["marker_x_in_cube"] = [int(round(v)) for v in R[:, 0]]
      g["marker_y_in_cube"] = [int(round(v)) for v in R[:, 1]]
      g["attachment"] = f"{f.get('letter')} / {f.get('color')} / {g['name']}: MEASURED from {Path(video).name}"
    else:
      g["attachment"] = "NOT OBSERVED in the recording; copied from layout.json"
    faces.append(g)
  names = sorted(x["name"] for x in faces)
  cand = dict(src)
  cand["faces"] = faces
  cand["name"] = src.get("name", "") + " (candidate measured from recording)"
  cand["status"] = "measured_from_recording_unconfirmed"
  cand["measured_from"] = str(video)
  cand["warning"] = ("Candidate only: derived from relative marker rotations in one recording. layout.json is unchanged. "
                     "Confirm on the physical cube before relying on it.")
  if names != sorted(AXIS_NAMES):
    cand["warning"] += f" The reconstructed normals do not cover all six faces ({names}); this file will not load."
  (out_dir / f"layout.measured_{stem}.json").write_text(json.dumps(cand, indent=2) + "\n")
  shared = ba["offset_mode"] == "shared"
  first = next(iter(ba["offsets_m"].values()))
  pl = {"description": "Marker offset inside the face, estimated by bundle adjustment of a recording (candidate).",
        "marker_center_offset_in_marker_axes_m": [round(first[0], 5), round(first[1], 5)] if shared else [0.0, 0.0],
        "per_face_offset_in_marker_axes_m": None if shared else
        {str(k): [round(v[0], 5), round(v[1], 5)] for k, v in ba["offsets_m"].items()},
        "offset_convention": "marker_centre_cube = face_centre + offset[0] * marker_x + offset[1] * marker_y",
        "status": "consistent_with_recording", "allowed_status_values": ["preview_derived_unverified",
                                                                          "consistent_with_recording", "caliper_measured"],
        "source": f"cube_vision.layout_check on {video}: rms {ba['rms_px']:.2f} px over {ba['corners']} corners",
        "fit_formal_std": ba["std"], "per_face_fit_m": per_face_fit,
        "uncertainty_m": 0.002}
  (out_dir / f"marker_print_layout.measured_{stem}.json").write_text(json.dumps(pl, indent=2, default=_json_default) + "\n")


def _json_default(o):
  if isinstance(o, np.ndarray):
    return o.tolist()
  if isinstance(o, (np.integer,)):
    return int(o)
  if isinstance(o, (np.floating,)):
    return float(o)
  raise TypeError(type(o))
