#!/usr/bin/env python3
"""Live 3D view of the LEAP hand, the estimated cube and the fingertip forces.

Two sources:
  live    --udp 127.0.0.1:47800   packets streamed by deploy_direct.py --live-udp
  replay  --replay logs/X.npz     any run log; if it has no monitor fields (older
                                  logs) the estimators are re-run on the logged
                                  joints and commands (--monitor-dir)

What is drawn (all *estimates* from proprioception, except the hand pose):
  * hand: measured joint angles from the encoders (mirrors the real hand)
  * cube: estimator B's palm-frame pose, made continuous over the cube's 24
    symmetries; plain see-through box (face identity is not observable)
  * arrows on the cube: estimated fingertip normal force, pointing from each
    fingertip into the cube centre; length 3 cm per N (capped at 9 cm)
  * spheres on the fingertips: contact probability of the force estimator
  * text: estimated turns (integrated yaw), forces, contact probabilities,
    finger motor current and the force implied by that current (J^T F = Kt I)
  * figures: last 10 s of estimated forces and of current-implied forces

The viewer only reads; it never talks to the hand.  It needs mujoco (repo
.venv), the control loop does not.

    uv run python LEAP_Hand_RL_Deploy_Kit/kit/live_viewer.py --udp 127.0.0.1:47800
    uv run python LEAP_Hand_RL_Deploy_Kit/kit/live_viewer.py --replay LEAP_Hand_RL_Deploy_Kit/logs/physical_X.npz
    ... --replay X.npz --snapshot out.png --at 4.0     # offscreen frame, no window
"""

from __future__ import annotations

import argparse
import collections
import json
import socket
import struct
import sys
import time
import zlib
from pathlib import Path

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from hand_scene import CUBE_HALF, DEFAULT_SCENE, HandScene  # noqa: E402
from monitor import FINGERS, HandMonitor  # noqa: E402

KIT = Path(__file__).resolve().parents[1]
_HW_TO_SERVER = np.array([1, 0, 2, 3, 5, 4, 6, 7, 9, 8, 10, 11, 12, 13, 14, 15])
FINGER_RGB = {"thumb": (0.95, 0.25, 0.25), "index": (0.2, 0.75, 0.25), "middle": (0.25, 0.5, 1.0),
              "ring": (0.95, 0.8, 0.15)}
# finger -> motor ids (hardware order: index 0-3, middle 4-7, ring 8-11, thumb 12-15)
FINGER_MOTORS = {"index": [0, 1, 2, 3], "middle": [4, 5, 6, 7], "ring": [8, 9, 10, 11], "thumb": [12, 13, 14, 15]}
B_CONTACT_INDEX = {"index": 0, "middle": 1, "ring": 2, "thumb": 3}
ARROW_M_PER_N, ARROW_MAX_M = 0.03, 0.09


# ----------------------------------------------------------------------------- sources
class UdpSource:
    def __init__(self, addr: str):
        host, port = addr.rsplit(":", 1)
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind((host, int(port)))
        self.sock.setblocking(False)
        self.label = f"LIVE udp {addr}"

    def poll(self) -> dict | None:
        last = None
        while True:  # drain, keep newest
            try:
                data, _ = self.sock.recvfrom(65536)
            except BlockingIOError:
                return last
            try:
                last = json.loads(data)
            except ValueError:
                continue


def load_log_packets(path: Path, monitor_dir: Path) -> tuple[list[dict], str]:
    """Rebuild viewer packets from a run log (monitor fields re-computed if absent)."""
    d = np.load(path, allow_pickle=False)
    q = d["q_hw"][:, _HW_TO_SERVER]
    cmd = d["command_hw"][:, _HW_TO_SERVER]
    t = d["elapsed_s"]
    meta = json.loads(str(d["run_meta_json"])) if "run_meta_json" in d.files else {}
    logged = "mon_force_n" in d.files
    mon_rows = []
    if logged:
        keys = [k for k in d.files if k.startswith("mon_")]
        for i in range(len(t)):
            mon_rows.append({k[4:]: d[k][i] for k in keys})
        how = "monitor values logged during the run"
    else:
        mon = HandMonitor(monitor_dir)
        for i in range(len(t)):
            prev = q[0] if i == 0 else cmd[i - 1]  # bridge: reset from the measured pose, then applied commands
            obs = mon.push(q[i], prev, reset=(i == 0))
            mon_rows.append(mon.update(obs))
        how = "monitor re-computed from logged joints/commands"
    temp = np.nanmax(np.where(np.isfinite(d["temperature_c"]), d["temperature_c"], -np.inf), axis=1)
    temp = np.where(np.isfinite(temp), temp, np.nan)
    for i in range(1, len(temp)):  # temperature is only read every N steps
        if np.isnan(temp[i]):
            temp[i] = temp[i - 1]
    pk = []
    for i in range(len(t)):
        pk.append({"type": "step", "step": i, "t": float(t[i] - t[0]), "shadow": bool(meta.get("shadow", False)),
                   "port": meta.get("port"),
                   "policy": meta.get("policy_dir") or "unknown (older log)",
                   "q_server": q[i].tolist(), "cmd_server": cmd[i].tolist(),
                   "current_ma_hw": d["current_ma"][i].tolist(),
                   "temp_max_c": float(temp[i]),
                   "monitor": {k: np.asarray(v).tolist() for k, v in mon_rows[i].items()}})
    return pk, how


class ReplaySource:
    def __init__(self, path: Path, monitor_dir: Path, speed: float, loop: bool):
        self.packets, how = load_log_packets(path, monitor_dir)
        self.speed, self.loop = speed, loop
        self.t0 = time.monotonic()
        self.i = -1
        self.label = f"REPLAY {path.name} x{speed:g} ({how})"

    @property
    def done(self) -> bool:
        return not self.loop and self.i >= len(self.packets) - 1

    def poll(self) -> dict | None:
        tnow = (time.monotonic() - self.t0) * self.speed
        if self.loop and tnow > self.packets[-1]["t"] + 1.0:
            self.t0, self.i, tnow = time.monotonic(), -1, 0.0
        j = self.i
        while j + 1 < len(self.packets) and self.packets[j + 1]["t"] <= tnow:
            j += 1
        if j != self.i:
            self.i = j
            return self.packets[j]
        return None


# ----------------------------------------------------------------------------- state
class State:
    def __init__(self, kt: float, smooth: float, history_s: float = 10.0):
        self.kt, self.smooth = kt, smooth
        self.pkt: dict | None = None
        self.last_rx = 0.0
        self.pos = None
        self.quat = None
        self.hist = collections.deque(maxlen=int(history_s * 20) + 5)
        self.f_from_i: dict[str, float] = {f: 0.0 for f in FINGERS}

    def update(self, pkt: dict, scene: HandScene) -> None:
        if self.pkt is not None and pkt["step"] < self.pkt["step"]:
            self.hist.clear()  # new run / replay looped
            self.pos = self.quat = None
        self.pkt, self.last_rx = pkt, time.monotonic()
        scene.set_hand(np.asarray(pkt["q_server"]))
        mon = pkt.get("monitor")
        if mon:
            p = np.asarray(mon["cube_pos_palm"], np.float64)
            q = np.asarray(mon["cube_quat_palm"], np.float64)
            if self.pos is None:
                self.pos, self.quat = p, q
            else:
                a = 1.0 - self.smooth
                if q @ self.quat < 0:
                    q = -q
                self.pos = (1 - a) * self.pos + a * p
                self.quat = (1 - a) * self.quat + a * q
                self.quat /= np.linalg.norm(self.quat)
            scene.set_cube_palm(self.pos, self.quat)
        cur = np.asarray(pkt["current_ma_hw"], np.float64)
        tau_server = self.kt * 1e-3 * cur[_HW_TO_SERVER]
        cube_c = scene.cube_world()[0] if mon else None
        self.f_from_i = scene.tip_force_from_torque(tau_server, cube_c)
        if mon:
            self.hist.append((pkt["t"], np.asarray(mon["force_n"]), np.array([self.f_from_i[f] for f in FINGERS])))


# ----------------------------------------------------------------------------- drawing
def _add_geom(scn: mujoco.MjvScene) -> mujoco.MjvGeom | None:
    if scn.ngeom >= scn.maxgeom:
        return None
    g = scn.geoms[scn.ngeom]
    scn.ngeom += 1
    return g


def draw_overlay(scn: mujoco.MjvScene, scene: HandScene, st: State) -> None:
    mon = st.pkt.get("monitor") if st.pkt else None
    if not mon:
        return
    c = scene.cube_world()[0]
    tips = scene.tip_positions()
    forces = np.asarray(mon["force_n"])
    pc = np.asarray(mon["force_contact_prob"])
    for k, f in enumerate(FINGERS):
        rgb = FINGER_RGB[f]
        tip = tips[f]
        u = c - tip
        dist = np.linalg.norm(u)
        u = u / max(dist, 1e-9)
        p_contact = 0.0 if np.isnan(pc[k]) else float(pc[k])
        g = _add_geom(scn)  # fingertip sphere: contact probability
        if g is not None:
            mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_SPHERE, np.array([0.004 + 0.008 * p_contact, 0, 0]),
                                tip, np.eye(3).reshape(-1), np.array([*rgb, 0.25 + 0.7 * p_contact], np.float32))
        L = min(ARROW_MAX_M, ARROW_M_PER_N * float(forces[k]))
        if L > 0.002:
            # force on the cube: ends on the cube surface along the tip -> centre line
            end = c - u * CUBE_HALF
            g = _add_geom(scn)
            if g is not None:
                mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_ARROW, np.zeros(3), np.zeros(3), np.eye(3).reshape(-1),
                                    np.array([*rgb, 0.95], np.float32))
                mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_ARROW, 0.004, end - u * L, end)
                g.label = f"{f[0].upper()} {forces[k]:.2f}N"


def overlay_texts(st: State, source_label: str, finished: bool = False) -> list:
    pk = st.pkt
    font = mujoco.mjtFontScale.mjFONTSCALE_150
    if pk is None:
        return [(font, mujoco.mjtGridPos.mjGRID_TOPLEFT, "waiting for data", source_label)]
    stale = time.monotonic() - st.last_rx > 1.0
    mon = pk.get("monitor") or {}
    mode = "SHADOW (torque off)" if pk.get("shadow") else "POWERED"
    if pk.get("port") == "SIMULATED":
        mode = "SIMULATED motor bus - not the real hand"
    left = ["source", "policy", "mode", "time / step", "est. turns", "est. cube pos (mm)", "max temp"]
    status = "   [REPLAY FINISHED - close the window to exit]" if finished else ("   [NO NEW DATA]" if stale else "")
    right = [source_label + status, str(pk.get("policy")), mode,
             f"{pk['t']:.1f} s / {pk['step']}"]
    if mon:
        right += [f"{float(mon['est_yaw_rad']) / (2 * np.pi):+.2f}  (integrated estimate)",
                  " ".join(f"{1000 * v:+.0f}" for v in mon["cube_pos_palm"])]
    else:
        right += ["no monitor", "-"]
    right.append(f"{pk.get('temp_max_c', float('nan')):.0f} C")
    texts = [(font, mujoco.mjtGridPos.mjGRID_TOPLEFT, "\n".join(left), "\n".join(right))]
    if mon:
        cur = np.abs(np.asarray(pk["current_ma_hw"]))
        rows_l, rows_r = ["finger"], ["F_est N | p_contact | p_B | I mA | F(I) N"]
        for k, f in enumerate(FINGERS):
            pcf = mon["force_contact_prob"][k]
            rows_l.append(f)
            rows_r.append(f"{mon['force_n'][k]:5.2f} | {pcf:4.2f} | {mon['contact_prob_b'][B_CONTACT_INDEX[f]]:4.2f} | "
                          f"{cur[FINGER_MOTORS[f]].sum():4.0f} | {st.f_from_i[f]:+5.2f}")
        texts.append((font, mujoco.mjtGridPos.mjGRID_TOPRIGHT, "\n".join(rows_l), "\n".join(rows_r)))
    return texts


def make_figures(st: State, viewport) -> list:
    if not st.hist:
        return []
    t = np.array([h[0] for h in st.hist])
    F = np.stack([h[1] for h in st.hist])
    FI = np.stack([h[2] for h in st.hist])
    figs = []
    w, h = viewport.width // 3, viewport.height // 4
    for j, (title, data) in enumerate((("estimated fingertip force [N]", F),
                                       ("force implied by motor current [N] (approx.)", FI))):
        fig = mujoco.MjvFigure()
        mujoco.mjv_defaultFigure(fig)
        fig.title = title
        fig.flg_legend = 1
        fig.figurergba[3] = 0.6
        fig.range[0][:] = [t[-1] - 10.0, t[-1]]
        fig.range[1][:] = [0, 3] if j == 0 else [-3, 3]
        fig.flg_extend = 0
        for k, f in enumerate(FINGERS):
            n = min(len(t), fig.linedata.shape[1] // 2)
            xy = np.empty(2 * n)
            xy[0::2], xy[1::2] = t[-n:], data[-n:, k]
            fig.linedata[k][: 2 * n] = xy
            fig.linepnt[k] = n
            fig.linergb[k] = FINGER_RGB[f]
            fig.linename[k] = f
        rect = mujoco.MjrRect(viewport.left + j * w, viewport.bottom, w, h)
        figs.append((rect, fig))
    return figs


# ----------------------------------------------------------------------------- offscreen snapshot
def write_png(path: Path, img: np.ndarray) -> None:
    h, w, _ = img.shape
    raw = b"".join(b"\x00" + img[y].tobytes() for y in range(h))
    def chunk(tag, data):
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
    png = b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
    png += chunk(b"IDAT", zlib.compress(raw, 6)) + chunk(b"IEND", b"")
    path.write_bytes(png)


def default_camera() -> mujoco.MjvCamera:
    cam = mujoco.MjvCamera()
    cam.lookat[:] = [-0.01, 0.03, 0.24]
    cam.distance, cam.azimuth, cam.elevation = 0.42, 140.0, -35.0
    return cam


def snapshot(args, scene: HandScene, st: State, src: ReplaySource) -> None:
    for pk in src.packets:
        if pk["t"] > args.at:
            break
        st.update(pk, scene)
    scene.model.vis.global_.offwidth = max(scene.model.vis.global_.offwidth, 960)
    scene.model.vis.global_.offheight = max(scene.model.vis.global_.offheight, 720)
    r = mujoco.Renderer(scene.model, 720, 960)
    r.update_scene(scene.data, camera=default_camera())
    draw_overlay(r.scene, scene, st)
    write_png(args.snapshot, r.render())
    mon = st.pkt["monitor"]
    print(f"[viewer] wrote {args.snapshot} at t={st.pkt['t']:.2f}s: forces {np.round(mon['force_n'], 2)} N, "
          f"est. turns {float(mon['est_yaw_rad']) / (2 * np.pi):+.3f}, F(I) "
          + ", ".join(f"{f} {v:+.2f}" for f, v in st.f_from_i.items()))


# ----------------------------------------------------------------------------- main
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src_g = ap.add_mutually_exclusive_group(required=True)
    src_g.add_argument("--udp", help="host:port to listen on (live)")
    src_g.add_argument("--replay", type=Path, help="run log (.npz) to play back")
    ap.add_argument("--monitor-dir", type=Path, default=KIT / "monitors")
    ap.add_argument("--scene", type=Path, default=DEFAULT_SCENE)
    ap.add_argument("--speed", type=float, default=1.0)
    ap.add_argument("--loop", action="store_true")
    ap.add_argument("--smooth", type=float, default=0.5, help="display smoothing of the cube pose (0 = none)")
    ap.add_argument("--kt", type=float, default=0.5, help="motor torque constant [N m / A] for F(I) (approximate)")
    ap.add_argument("--snapshot", type=Path, default=None, help="replay only: write one offscreen PNG and exit")
    ap.add_argument("--at", type=float, default=3.0, help="replay time [s] for --snapshot")
    ap.add_argument("--max-seconds", type=float, default=0.0, help="close the window after this long (0 = never)")
    a = ap.parse_args()

    scene = HandScene(a.scene)
    st = State(a.kt, a.smooth)
    src = UdpSource(a.udp) if a.udp else ReplaySource(a.replay, a.monitor_dir, a.speed, a.loop)
    print(f"[viewer] {src.label}")
    if a.replay and a.snapshot is None:
        print("[viewer] close the window to exit (add --loop to repeat the replay)")
    if a.snapshot is not None:
        if not a.replay:
            ap.error("--snapshot needs --replay")
        snapshot(a, scene, st, src)
        return 0

    import mujoco.viewer

    t_start = time.monotonic()
    with mujoco.viewer.launch_passive(scene.model, scene.data, show_left_ui=False, show_right_ui=False) as v:
        cam = default_camera()
        v.cam.lookat[:], v.cam.distance = cam.lookat, cam.distance
        v.cam.azimuth, v.cam.elevation = cam.azimuth, cam.elevation
        while v.is_running():
            tick = time.monotonic()
            pk = src.poll()
            with v.lock():
                if pk is not None:
                    st.update(pk, scene)
                v.user_scn.ngeom = 0
                draw_overlay(v.user_scn, scene, st)
            v.set_texts(overlay_texts(st, src.label, finished=getattr(src, "done", False)))
            figs = make_figures(st, v.viewport) if v.viewport is not None else []
            if figs:
                v.set_figures(figs)
            v.sync()
            if a.max_seconds and time.monotonic() - t_start > a.max_seconds:
                break
            time.sleep(max(0.0, 1 / 60 - (time.monotonic() - tick)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
