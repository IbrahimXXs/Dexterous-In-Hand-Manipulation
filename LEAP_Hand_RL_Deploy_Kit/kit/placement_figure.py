#!/usr/bin/env python3
"""Render the standard cube placement for real-hand trials (docs figure).

Target = grasp 49 of the 239-grasp training cache (experiments/hwlimit_ac/cache),
the most robust start found in grasp_rotation_v2 (keeps its gain under +-2 mm /
+-10 deg placement noise).  Panels: (1) open hand, where to put the cube;
(2) after the kit closes to its default grasp pose; (3) side view.

    uv run --with pillow python LEAP_Hand_RL_Deploy_Kit/kit/placement_figure.py --out docs/figures/cube_placement.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parent))
from hand_scene import HandScene  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / "training_source/project/experiments/hwlimit_ac/cache/hwlimit_grasp_cache.npz"
DEFAULT = np.array([0.1, -0.4, 1.3, 0.0, 0.1, 0.0, 1.3, 0.0, 0.1, 0.4, 1.3, 0.0, 1.45, 1.5, 0.579, 1.37])
W = H = 620


def project(r: mujoco.Renderer, p: np.ndarray) -> tuple[float, float]:
    cam = r.scene.camera[0]
    # the two stereo eyes are offset by +-ipd/2: use their midpoint
    pos = 0.5 * (np.array(r.scene.camera[0].pos) + np.array(r.scene.camera[1].pos))
    fwd, up = np.array(cam.forward), np.array(cam.up)
    right = np.cross(fwd, up)
    v = p - pos
    zc = v @ fwd
    x = (v @ right) / zc * cam.frustum_near
    y = (v @ up) / zc * cam.frustum_near
    half_h = 0.5 * (cam.frustum_top - cam.frustum_bottom)
    half_w = cam.frustum_width if cam.frustum_width > 0 else half_h * W / H  # 0 = from aspect ratio
    u = (x - (cam.frustum_center - half_w)) / (2 * half_w) * W
    w = (cam.frustum_top - y) / (cam.frustum_top - cam.frustum_bottom) * H
    return float(u), float(w)


def font(size: int):
    for f in ("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):
        if Path(f).exists():
            return ImageFont.truetype(f, size)
    return ImageFont.load_default()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--grasp", type=int, default=49)
    ap.add_argument("--out", type=Path, default=ROOT / "docs" / "figures" / "cube_placement.png")
    a = ap.parse_args()

    c = np.load(CACHE, allow_pickle=True)
    pw, qw = c["cube_pose_rel"][a.grasp][:3].astype(float), c["cube_pose_rel"][a.grasp][3:].astype(float)
    sc = HandScene()
    m, d = sc.model, sc.data
    m.geom_rgba[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "cube_visual")] = (1, 1, 1, 1)
    m.geom_rgba[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "cube_geom")] = (0, 0, 0, 0)
    m.vis.global_.offwidth = m.vis.global_.offheight = max(W, H)
    r = mujoco.Renderer(m, H, W)

    def scene(q_hand):
        sc.set_hand(q_hand)
        a_ = sc.cube_qadr
        d.qpos[a_:a_ + 3], d.qpos[a_ + 3:a_ + 7] = pw, qw
        mujoco.mj_kinematics(m, d)

    def shot(q_hand, az, el, dist, lookat):
        scene(q_hand)
        cam = mujoco.MjvCamera()
        cam.lookat[:], cam.distance, cam.azimuth, cam.elevation = lookat, dist, az, el
        r.update_scene(d, camera=cam)
        return Image.fromarray(r.render()), {f: project(r, p) for f, p in sc.tip_positions().items()}, project(r, pw)

    centre = np.array([pw[0] - 0.03, pw[1], pw[2]])
    panels = [
        ("1. PLACE: open hand, seen from above", *shot(np.zeros(16), 0.0, -89.9, 0.40, centre)),
        ("2. After the fingers close (5 s ramp)", *shot(DEFAULT, 0.0, -89.9, 0.40, centre)),
        ("3. Side view after closing (thumb side)", *shot(DEFAULT, 90.0, -15.0, 0.42, centre)),
    ]
    pad, title_h, foot_h = 16, 46, 246
    fig = Image.new("RGB", (3 * W + 4 * pad, H + title_h + foot_h + pad), "white")
    dr = ImageDraw.Draw(fig)
    for i, (title, img, tips, cpx) in enumerate(panels):
        x0 = pad + i * (W + pad)
        fig.paste(img, (x0, title_h))
        dr.text((x0, 10), title, fill="black", font=font(22))
        for f, (u, v) in tips.items():
            if not (0 <= u < W and 0 <= v < H):
                continue
            tw = dr.textlength(f, font=font(20))
            if i < 2:  # top views: centred just above / beside the fingertip, kept inside the panel
                tx, ty = u - tw / 2, v - 48
                if f == "thumb":
                    tx, ty = u - tw - 20, v - 40
            else:
                tx, ty = u + 10, v - 12
            tx = min(max(tx, 4), W - tw - 4)
            dr.text((x0 + tx, title_h + ty), f, fill=(255, 230, 0), font=font(20), stroke_width=2, stroke_fill="black")
        if i == 0:  # cross-hair on the target centre
            u, v = cpx
            u, v = x0 + u, title_h + v
            dr.ellipse((u - 7, v - 7, u + 7, v + 7), outline=(220, 0, 0), width=3)
            dr.line((u - 16, v, u + 16, v), fill=(220, 0, 0), width=2)
            dr.line((u, v - 16, u, v + 16), fill=(220, 0, 0), width=2)
            dr.text((x0 + 10, title_h + H - 34), "you stand here (wrist end)", fill="white", font=font(18),
                    stroke_width=2, stroke_fill="black")
            # trained rotation: +Z (yaw) = counter-clockwise seen from above (this view is not mirrored)
            rad = 118
            ts = np.radians(np.linspace(-35, 215, 60))
            pts = [(u + rad * np.cos(t), v - rad * np.sin(t)) for t in ts]
            dr.line(pts, fill=(0, 200, 255), width=6)
            ex, ey = pts[-1]
            t_end = ts[-1]
            tang = np.array([-np.sin(t_end), -np.cos(t_end)])  # screen direction of increasing angle
            nrm = np.array([tang[1], -tang[0]])
            tip = np.array([ex, ey]) + 14 * tang
            dr.polygon([tuple(tip), tuple(np.array([ex, ey]) - 12 * tang + 12 * nrm),
                        tuple(np.array([ex, ey]) - 12 * tang - 12 * nrm)], fill=(0, 200, 255))
            dr.text((x0 + 300, title_h + 470), "cube turns this way:\ncounter-clockwise\nseen from above (+Z)",
                    fill=(0, 220, 255), font=font(19), stroke_width=2, stroke_fill="black")
    lines = [
        "Same placement every trial (training grasp 49: the most robust start; tolerates about ±2 mm and ±10°):",
        "  • Hand mounted palm up, fingers pointing away from you, thumb on your right; place during the 5 s countdown (torque is off).",
        "  • Cube flat, its CENTRE right over the knuckle line (half on the palm, half on the first finger segments), in line with the MIDDLE finger (red cross).",
        "  • Edges parallel to the knuckle line; logo / taped face up. Hold it lightly until the fingers have closed, then let go.",
        "  • Expected rotation: counter-clockwise seen from above (+Z): the fingertip-side face moves left (to the ring finger).",
        "  • If the cube is pushed more than ~5 mm or tilts while the fingers close, note it; repeat the trial if it starts badly.",
    ]
    for k, s in enumerate(lines):
        dr.text((pad, title_h + H + 14 + 36 * k), s, fill="black", font=font(21 if k == 0 else 20))
    a.out.parent.mkdir(parents=True, exist_ok=True)
    fig.save(a.out)
    print(f"[placement] wrote {a.out}; cube centre in palm frame (mm): "
          f"{np.round(1000 * sc.world_to_palm(pw, np.eye(3))[0], 1)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
