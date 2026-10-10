"""Simulated Booster loco RPC service: the subset of b1_loco_api brain_node uses."""

import json
import math
from dataclasses import dataclass

API_CHANGE_MODE = 2000
API_MOVE = 2001
API_ROTATE_HEAD = 2004
API_GET_UP = 2008
API_GET_MODE = 2017
API_VISUAL_KICK = 2038

MODE_DAMPING, MODE_PREPARE, MODE_WALKING, MODE_CUSTOM, MODE_SOCCER = 0, 1, 2, 3, 4
VALID_MODES = (MODE_DAMPING, MODE_PREPARE, MODE_WALKING, MODE_CUSTOM, MODE_SOCCER)
MOVING_MODES = (MODE_WALKING, MODE_SOCCER)
KICK_V1, KICK_V2 = 0, 1  # booster::robot::b1::VisualKickVersion

STATUS_OK = 0
# ponytail: the real SDK's error codes are undocumented; brain only checks status == 0.
STATUS_BAD_REQUEST = 1


@dataclass(frozen=True)
class VelocityLimits:
    vx_min: float = -0.7
    vx_max: float = 3.0
    vy_max: float = 0.8
    vyaw_max: float = 2.0


@dataclass
class LocoState:
    mode: int = MODE_SOCCER
    vx: float = 0.0
    vy: float = 0.0
    vyaw: float = 0.0
    head_pitch: float = 0.0
    head_yaw: float = 0.0
    getup_requested: bool = False
    visual_kick_active: bool = False
    visual_kick_version: int = KICK_V2

    def effective_velocity(self) -> tuple[float, float, float]:
        if self.mode in MOVING_MODES:
            return self.vx, self.vy, self.vyaw
        return 0.0, 0.0, 0.0

    def kicking(self) -> bool:
        return self.visual_kick_active and self.mode in MOVING_MODES


@dataclass(frozen=True)
class RpcReply:
    header: str
    body: str


def _finite(*values) -> tuple[float, ...]:
    out = tuple(float(v) for v in values)
    if not all(math.isfinite(v) for v in out):
        raise ValueError("non-finite parameter")
    return out


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def _ok(body: str = "") -> RpcReply:
    return RpcReply(json.dumps({"status": STATUS_OK}), body)


def handle_request(state: LocoState, header: str, body: str, limits: VelocityLimits = VelocityLimits()) -> RpcReply:
    try:
        api_id = int(json.loads(header)["api_id"])
        params = json.loads(body) if body else {}
        if api_id == API_MOVE:
            vx, vy, vyaw = _finite(params["vx"], params["vy"], params["vyaw"])
            state.vx = _clamp(vx, limits.vx_min, limits.vx_max)
            state.vy = _clamp(vy, -limits.vy_max, limits.vy_max)
            state.vyaw = _clamp(vyaw, -limits.vyaw_max, limits.vyaw_max)
        elif api_id == API_ROTATE_HEAD:
            state.head_pitch, state.head_yaw = _finite(params["pitch"], params["yaw"])
        elif api_id == API_CHANGE_MODE:
            mode = int(params["mode"])
            if mode not in VALID_MODES:
                raise ValueError(f"unknown mode {mode}")
            state.mode = mode
            state.vx = state.vy = state.vyaw = 0.0
            state.visual_kick_active = False
        elif api_id == API_GET_MODE:
            return _ok(json.dumps({"mode": state.mode}))
        elif api_id == API_GET_UP:
            state.getup_requested = True
        elif api_id == API_VISUAL_KICK:
            start = params["start"]
            if not isinstance(start, bool):
                raise ValueError("start must be a bool")
            version = int(params["version"])
            state.visual_kick_active = start
            state.visual_kick_version = version
        # Every other id: acknowledged no-op.
    except (KeyError, ValueError, TypeError):
        return RpcReply(json.dumps({"status": STATUS_BAD_REQUEST}), "")
    return _ok()
