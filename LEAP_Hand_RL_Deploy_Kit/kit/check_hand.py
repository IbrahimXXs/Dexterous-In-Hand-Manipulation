#!/usr/bin/env python3
"""Read-only LEAP hand check: ports, motor IDs, joint angles, temperatures.

Never enables torque. Safe to run with the hand powered and relaxed.
"""

from __future__ import annotations

import argparse
import glob
import sys

import numpy as np

try:
    from dynamixel_sdk import COMM_SUCCESS, PacketHandler, PortHandler
except ImportError:
    sys.exit("dynamixel_sdk not found - run ./install.sh first")

HW_JOINT_NAMES = (
    "if_rot", "if_mcp", "if_pip", "if_dip",
    "mf_rot", "mf_mcp", "mf_pip", "mf_dip",
    "rf_rot", "rf_mcp", "rf_pip", "rf_dip",
    "th_cmc", "th_axl", "th_mcp", "th_ipl",
)
HW_LIMIT_LO = np.array([-0.5236, -0.2618, -0.2618, -0.2618] * 3 + [0.0, 0.0, -0.3491, -0.3491])
HW_LIMIT_HI = np.array([0.5236, 1.9199, 1.9199, 1.9199] * 3 + [1.571, 1.571, 1.7453, 1.7453])
ADDR_TORQUE_ENABLE = 64
ADDR_PRESENT_POSITION = 132
ADDR_PRESENT_INPUT_VOLTAGE = 144
ADDR_PRESENT_TEMPERATURE = 146
ADDR_MODEL_NUMBER = 0


def list_ports() -> list[str]:
    by_id = sorted(glob.glob("/dev/serial/by-id/*"))
    ttys = sorted(glob.glob("/dev/ttyUSB*") + glob.glob("/dev/ttyACM*"))
    return by_id + ttys


def scan(port_name: str, bauds: list[int], ids: range) -> tuple[int | None, list[int]]:
    port = PortHandler(port_name)
    if not port.openPort():
        raise RuntimeError(
            f"Cannot open {port_name}. Is it the right device, is another program "
            "using it, and are you in the 'dialout' group (log out/in after install)?"
        )
    packet = PacketHandler(2.0)
    try:
        for baud in bauds:
            if not port.setBaudRate(baud):
                continue
            found = []
            for mid in ids:
                _, comm, err = packet.ping(port, mid)
                if comm == COMM_SUCCESS and err == 0:
                    found.append(mid)
            if found:
                return baud, found
        return None, []
    finally:
        port.closePort()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--port", default=None)
    ap.add_argument("--baudrate", type=int, default=4_000_000)
    ap.add_argument("--list", action="store_true", help="only list serial ports")
    args = ap.parse_args()

    ports = list_ports()
    print("Serial ports found:")
    for p in ports:
        print("   ", p)
    if not ports:
        print("    (none) - plug in the U2D2 / USB adapter and power the hand")
    if args.list:
        return 0
    port_name = args.port or (ports[0] if ports else None)
    if port_name is None:
        return 1
    print(f"\nUsing port {port_name}")

    baud, found = scan(port_name, [args.baudrate], range(16))
    if len(found) < 16:
        print(f"Only {len(found)}/16 motors answered at {args.baudrate} baud: {found}")
        print("Scanning other baud rates / IDs 0-30 to diagnose ...")
        alt_baud, alt = scan(port_name, [4_000_000, 3_000_000, 2_000_000, 1_000_000, 57_600], range(31))
        if alt:
            print(f"  -> found IDs {alt} at {alt_baud} baud")
            print("  The kit expects IDs 0..15 at 4,000,000 baud (LEAP default). Reconfigure the")
            print("  motors with Dynamixel Wizard, or pass --baudrate if you know what you're doing.")
        else:
            print("  -> no motors found. Check 12V power (or 5V for XL330 hands), cable, and port.")
        return 1

    port = PortHandler(port_name)
    port.openPort()
    port.setBaudRate(baud)
    pk = PacketHandler(2.0)
    rows = []
    q = np.zeros(16)
    for mid in range(16):
        model, _, _ = pk.read2ByteTxRx(port, mid, ADDR_MODEL_NUMBER)
        torque, _, _ = pk.read1ByteTxRx(port, mid, ADDR_TORQUE_ENABLE)
        raw, _, _ = pk.read4ByteTxRx(port, mid, ADDR_PRESENT_POSITION)
        volt, _, _ = pk.read2ByteTxRx(port, mid, ADDR_PRESENT_INPUT_VOLTAGE)
        temp, _, _ = pk.read1ByteTxRx(port, mid, ADDR_PRESENT_TEMPERATURE)
        if raw & 0x80000000:
            raw -= 1 << 32
        q[mid] = raw * 2 * np.pi / 4096 - np.pi
        rows.append((mid, HW_JOINT_NAMES[mid], model, torque, raw, q[mid], volt / 10, temp))
    port.closePort()

    print(f"\nAll 16 motors answered at {baud} baud.\n")
    print(" id joint    model torque  raw_pos   q[rad]  q[deg]  volt  temp  limit")
    warn = []
    for mid, name, model, torque, raw, qq, volt, temp in rows:
        inside = HW_LIMIT_LO[mid] - 0.15 <= qq <= HW_LIMIT_HI[mid] + 0.15
        flag = "ok" if inside else "OUT"
        if not inside:
            warn.append(name)
        print(f" {mid:2d} {name:7s} {model:6d} {torque:5d} {raw:9d} {qq:8.3f} {np.degrees(qq):7.1f} {volt:5.1f} {temp:4d}  {flag}")
    print()
    if any(r[3] for r in rows):
        print("NOTE: torque is currently ENABLED on some motors (another program used them).")
    if warn:
        print(f"WARNING: joints {warn} read far outside the LEAP joint range.")
        print("  The kit assumes the LEAP convention: raw 2048 (= 0 rad) is the flat, open hand.")
        print("  If the hand is relaxed and flat yet these read ~+/-180 deg or similar, the motor")
        print("  horn is mounted off by a multiple of 90 deg or the motor's homing offset is set.")
    else:
        print("Joint readings are inside the LEAP joint range.")
    print("Tip: lay the hand flat and open - every joint should read close to 0 rad.")
    hot = max(r[7] for r in rows)
    if hot >= 55:
        print(f"WARNING: hottest motor is {hot} C - let it cool before running a policy.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
