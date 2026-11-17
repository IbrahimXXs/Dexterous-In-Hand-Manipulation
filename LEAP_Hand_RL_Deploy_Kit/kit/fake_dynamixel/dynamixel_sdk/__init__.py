"""Simulated 16-motor Dynamixel bus (for offline tests of the hardware bridge).

Implements just the dynamixel_sdk surface used by deploy_direct.py. Motors
follow their goal position with a first-order lag while torque is enabled.
Never used on real hardware: it is only put on PYTHONPATH by `./leap.sh test`.
"""

from __future__ import annotations

import math
import time

COMM_SUCCESS = 0
COMM_TX_FAIL = -1001

_ADDR_TORQUE = 64
_ADDR_GOAL = 116
_ADDR_CURRENT = 126
_ADDR_VELOCITY = 128
_ADDR_POSITION = 132
_ADDR_TEMPERATURE = 146


class _Motor:
    def __init__(self, mid: int):
        self.regs = bytearray(256)
        # relaxed hand: slightly curled, ~raw 2048 = 0 rad
        self.pos = 2048 + (40 if mid % 4 else 0)
        self.vel = 0
        self._set(_ADDR_POSITION, 4, self.pos)
        self._set(_ADDR_GOAL, 4, self.pos)
        self._set(_ADDR_TEMPERATURE, 1, 34)
        self._t = time.monotonic()

    def _set(self, addr: int, size: int, value: int) -> None:
        self.regs[addr:addr + size] = (int(value) & ((1 << (8 * size)) - 1)).to_bytes(size, "little")

    def _get(self, addr: int, size: int) -> int:
        return int.from_bytes(self.regs[addr:addr + size], "little")

    def update(self) -> None:
        now = time.monotonic()
        dt, self._t = now - self._t, now
        goal = self._get(_ADDR_GOAL, 4)
        if goal & 0x80000000:
            goal -= 1 << 32
        if self.regs[_ADDR_TORQUE]:
            alpha = 1.0 - math.exp(-dt / 0.03)
            new = self.pos + alpha * (goal - self.pos)
            self.vel = (new - self.pos) / max(dt, 1e-3) * 60 / (0.229 * 4096)
            current = 2.69 * 0.4 * (goal - self.pos)  # crude load model
            self.pos = new
        else:
            self.vel, current = 0.0, 0.0
        self._set(_ADDR_POSITION, 4, int(round(self.pos)))
        self._set(_ADDR_VELOCITY, 4, int(round(self.vel)))
        self._set(_ADDR_CURRENT, 2, int(round(max(-300, min(300, current)))))


_BUS = {mid: _Motor(mid) for mid in range(16)}


class PortHandler:
    def __init__(self, name: str):
        self.name = name

    def openPort(self) -> bool:
        return True

    def setBaudRate(self, baud: int) -> bool:
        return True

    def clearPort(self) -> None:
        pass

    def closePort(self) -> None:
        pass


class PacketHandler:
    def __init__(self, version: float):
        pass

    def ping(self, port, mid):
        return (1200, COMM_SUCCESS, 0) if mid in _BUS else (0, COMM_TX_FAIL, 0)

    def write1ByteTxRx(self, port, mid, addr, value):
        _BUS[mid].update()
        _BUS[mid]._set(addr, 1, value)
        return COMM_SUCCESS, 0

    def read1ByteTxRx(self, port, mid, addr):
        _BUS[mid].update()
        return _BUS[mid]._get(addr, 1), COMM_SUCCESS, 0

    def read2ByteTxRx(self, port, mid, addr):
        _BUS[mid].update()
        return _BUS[mid]._get(addr, 2), COMM_SUCCESS, 0

    def read4ByteTxRx(self, port, mid, addr):
        _BUS[mid].update()
        return _BUS[mid]._get(addr, 4), COMM_SUCCESS, 0

    def getTxRxResult(self, comm):
        return f"simulated comm result {comm}"


class GroupSyncRead:
    def __init__(self, port, packet, addr, size):
        self.addr, self.size, self.ids = addr, size, []

    def addParam(self, mid):
        self.ids.append(mid)
        return True

    def txRxPacket(self):
        for mid in self.ids:
            _BUS[mid].update()
        return COMM_SUCCESS

    def isAvailable(self, mid, addr, size):
        return mid in self.ids and self.addr <= addr and addr + size <= self.addr + self.size

    def getData(self, mid, addr, size):
        return _BUS[mid]._get(addr, size)


class GroupSyncWrite:
    def __init__(self, port, packet, addr, size):
        self.addr, self.size, self.data = addr, size, {}

    def addParam(self, mid, data):
        self.data[mid] = bytes(data)
        return True

    def txPacket(self):
        for mid, data in self.data.items():
            _BUS[mid].update()
            _BUS[mid].regs[self.addr:self.addr + self.size] = data
        return COMM_SUCCESS

    def clearParam(self):
        self.data.clear()
