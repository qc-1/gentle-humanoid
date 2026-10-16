"""Configurable force-pulse support for the MuJoCo sim2sim runner."""

from __future__ import annotations

import csv
import json
import math
import socket
import threading
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Deque, Dict, Iterable, List, Optional, Tuple

import mujoco
import numpy as np


PROTOCOL_VERSION = 1
DEFAULT_IMPULSE_PORT = 28563
TRAINING_FORCE_LIMIT_N = 30.0


class ImpulseValidationError(ValueError):
    """Raised when an impulse request is malformed or physically invalid."""


def _finite_float(value: Any, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ImpulseValidationError(f"'{name}' must be a number") from exc
    if not math.isfinite(result):
        raise ImpulseValidationError(f"'{name}' must be finite")
    return result


def _vector3(value: Any, name: str) -> np.ndarray:
    try:
        vector = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ImpulseValidationError(f"'{name}' must contain three numbers") from exc
    if vector.shape != (3,):
        raise ImpulseValidationError(f"'{name}' must contain exactly three numbers")
    if not np.all(np.isfinite(vector)):
        raise ImpulseValidationError(f"'{name}' must contain only finite values")
    return vector


@dataclass(frozen=True)
class ImpulseCommand:
    request_id: str
    body: str
    body_id: int
    force_n: float
    requested_duration_s: float
    steps: int
    actual_duration_s: float
    point_local_m: Tuple[float, float, float]
    direction: Tuple[float, float, float]
    direction_frame: str

    @property
    def impulse_ns(self) -> float:
        return self.force_n * self.actual_duration_s

    @property
    def exceeds_training_force(self) -> bool:
        return self.force_n > TRAINING_FORCE_LIMIT_N


@dataclass
class _ActiveImpulse:
    command: ImpulseCommand
    steps_applied: int = 0


@dataclass(frozen=True)
class ImpulseSnapshot:
    command: ImpulseCommand
    active: bool
    point_world_m: np.ndarray
    direction_world: np.ndarray
    force_world_n: np.ndarray
    elapsed_s: float
    remaining_s: float
    delivered_impulse_ns: float


def validate_impulse_request(
    request: Dict[str, Any], model: mujoco.MjModel, timestep: float
) -> ImpulseCommand:
    """Validate and normalize a preview/fire request."""
    if request.get("version") != PROTOCOL_VERSION:
        raise ImpulseValidationError(
            f"unsupported protocol version: {request.get('version')!r}"
        )

    request_id = str(request.get("id", "")).strip()
    if not request_id:
        raise ImpulseValidationError("request 'id' is required")

    body = str(request.get("body", "")).strip()
    if not body:
        raise ImpulseValidationError("'body' is required")
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body)
    if body_id < 0:
        raise ImpulseValidationError(f"unknown body: '{body}'")
    if body_id == 0:
        raise ImpulseValidationError("cannot apply an impulse to 'world'")

    force_n = _finite_float(request.get("force_n"), "force_n")
    if force_n <= 0.0:
        raise ImpulseValidationError("'force_n' must be greater than zero")

    requested_duration_s = _finite_float(
        request.get("duration_s"), "duration_s"
    )
    if requested_duration_s <= 0.0:
        raise ImpulseValidationError("'duration_s' must be greater than zero")
    if not math.isfinite(timestep) or timestep <= 0.0:
        raise ImpulseValidationError("simulation timestep must be positive and finite")

    point_local = _vector3(request.get("point_local_m"), "point_local_m")
    direction = _vector3(request.get("direction"), "direction")
    direction_norm = float(np.linalg.norm(direction))
    if direction_norm <= 1e-12:
        raise ImpulseValidationError("'direction' must be non-zero")
    direction /= direction_norm

    direction_frame = str(request.get("direction_frame", "")).strip().lower()
    if direction_frame not in {"world", "body"}:
        raise ImpulseValidationError("'direction_frame' must be 'world' or 'body'")

    # Quantize to the nearest physics step, while guaranteeing at least one step.
    step_count = requested_duration_s / timestep
    if not math.isfinite(step_count):
        raise ImpulseValidationError("'duration_s' is too large to quantize")
    steps = max(1, int(math.floor(step_count + 0.5)))
    actual_duration_s = steps * timestep
    if not math.isfinite(force_n * actual_duration_s):
        raise ImpulseValidationError("force-duration product is too large")

    return ImpulseCommand(
        request_id=request_id,
        body=body,
        body_id=body_id,
        force_n=force_n,
        requested_duration_s=requested_duration_s,
        steps=steps,
        actual_duration_s=actual_duration_s,
        point_local_m=tuple(float(v) for v in point_local),
        direction=tuple(float(v) for v in direction),
        direction_frame=direction_frame,
    )


def command_response(
    command: ImpulseCommand, message: str, *, status: str = "accepted"
) -> Dict[str, Any]:
    response: Dict[str, Any] = {
        "version": PROTOCOL_VERSION,
        "id": command.request_id,
        "status": status,
        "message": message,
        "actual_duration_s": command.actual_duration_s,
        "impulse_ns": command.impulse_ns,
    }
    if command.exceeds_training_force:
        response["warning"] = (
            f"{command.force_n:g} N exceeds the {TRAINING_FORCE_LIMIT_N:g} N "
            "training-force range; this is an out-of-distribution test"
        )
    return response


class ImpulseController:
    """Owns preview/active state and applies one force pulse at a time."""

    def __init__(self, model: mujoco.MjModel, timestep: float):
        self.model = model
        self.timestep = float(timestep)
        self.preview: Optional[ImpulseCommand] = validate_impulse_request(
            {
                "version": PROTOCOL_VERSION,
                "id": "default-preview",
                "body": "torso_link",
                "force_n": 30.0,
                "duration_s": 0.1,
                "point_local_m": [0.0, 0.0, 0.0],
                "direction": [1.0, 0.0, 0.0],
                "direction_frame": "world",
            },
            model,
            self.timestep,
        )
        self._active: Optional[_ActiveImpulse] = None

    @property
    def is_active(self) -> bool:
        return self._active is not None

    def handle_request(
        self, request: Dict[str, Any], *, control_active: bool
    ) -> Dict[str, Any]:
        request_id = str(request.get("id", "")).strip()
        if request.get("version") != PROTOCOL_VERSION:
            return self._rejected(request_id, "unsupported protocol version")
        if not request_id:
            return self._rejected(request_id, "request 'id' is required")

        action = str(request.get("command", "")).strip().lower()
        if action == "cancel":
            if self._active is None:
                return self._accepted(request_id, "no active impulse to cancel")
            cancelled = self._active.command
            self._active = None
            return {
                **self._accepted(request_id, f"cancelled impulse {cancelled.request_id}"),
                "cancelled_id": cancelled.request_id,
            }

        if action not in {"preview", "fire"}:
            return self._rejected(request_id, "'command' must be preview, fire, or cancel")

        try:
            command = validate_impulse_request(request, self.model, self.timestep)
        except ImpulseValidationError as exc:
            return self._rejected(request_id, str(exc))

        if action == "preview":
            if self._active is not None:
                return self._rejected(
                    request_id, "cannot change the preview while an impulse is active"
                )
            self.preview = command
            return command_response(command, "preview updated")

        if not control_active:
            return self._rejected(
                request_id, "the tracking-policy control loop is not active"
            )
        if self._active is not None:
            return self._rejected(
                request_id,
                f"impulse {self._active.command.request_id} is still active; cancel it first",
            )

        self.preview = command
        self._active = _ActiveImpulse(command=command)
        return command_response(command, "impulse scheduled for the next physics step")

    def sample_preview(self, data: mujoco.MjData) -> Optional[ImpulseSnapshot]:
        """Return the current preview in world coordinates without applying it."""
        if self.preview is None:
            return None
        return self._snapshot(self.preview, data, active=False, steps_applied=0)

    def step(self, data: mujoco.MjData) -> Optional[ImpulseSnapshot]:
        """Apply the active impulse for one physics step and return its state."""
        data.qfrc_applied[:] = 0.0
        if self._active is None:
            return self.sample_preview(data)

        active = self._active
        command = active.command
        point_world, direction_world = self._world_vectors(command, data)
        force_world = direction_world * command.force_n
        mujoco.mj_applyFT(
            self.model,
            data,
            force_world,
            np.zeros(3, dtype=np.float64),
            point_world,
            command.body_id,
            data.qfrc_applied,
        )

        active.steps_applied += 1
        snapshot = self._snapshot(
            command,
            data,
            active=True,
            steps_applied=active.steps_applied,
            point_world=point_world,
            direction_world=direction_world,
        )
        if active.steps_applied >= command.steps:
            self._active = None
        return snapshot

    def _snapshot(
        self,
        command: ImpulseCommand,
        data: mujoco.MjData,
        *,
        active: bool,
        steps_applied: int,
        point_world: Optional[np.ndarray] = None,
        direction_world: Optional[np.ndarray] = None,
    ) -> ImpulseSnapshot:
        if point_world is None or direction_world is None:
            point_world, direction_world = self._world_vectors(command, data)
        elapsed_s = min(steps_applied, command.steps) * self.timestep if active else 0.0
        return ImpulseSnapshot(
            command=command,
            active=active,
            point_world_m=point_world.copy(),
            direction_world=direction_world.copy(),
            force_world_n=(direction_world * command.force_n).copy(),
            elapsed_s=elapsed_s,
            remaining_s=max(0.0, command.actual_duration_s - elapsed_s),
            delivered_impulse_ns=command.force_n * elapsed_s,
        )

    @staticmethod
    def _accepted(request_id: str, message: str) -> Dict[str, Any]:
        return {
            "version": PROTOCOL_VERSION,
            "id": request_id,
            "status": "accepted",
            "message": message,
        }

    @staticmethod
    def _rejected(request_id: str, message: str) -> Dict[str, Any]:
        return {
            "version": PROTOCOL_VERSION,
            "id": request_id,
            "status": "rejected",
            "message": message,
        }

    @staticmethod
    def _world_vectors(
        command: ImpulseCommand, data: mujoco.MjData
    ) -> Tuple[np.ndarray, np.ndarray]:
        rotation = data.xmat[command.body_id].reshape(3, 3)
        point_world = data.xpos[command.body_id] + rotation @ np.asarray(
            command.point_local_m, dtype=np.float64
        )
        direction = np.asarray(command.direction, dtype=np.float64)
        if command.direction_frame == "body":
            direction = rotation @ direction
        direction /= np.linalg.norm(direction)
        return point_world, direction


@dataclass(frozen=True)
class PendingImpulseRequest:
    request: Dict[str, Any]
    address: Tuple[str, int]


class ImpulseUDPServer(threading.Thread):
    """Local JSON/UDP server; MuJoCo work remains on the simulation thread."""

    def __init__(self, host: str = "127.0.0.1", port: int = DEFAULT_IMPULSE_PORT):
        super().__init__(daemon=True)
        self.host = host
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.bind((host, port))
        self.port = int(self._sock.getsockname()[1])
        self._sock.settimeout(0.2)
        self._queue: Deque[PendingImpulseRequest] = deque()
        self._lock = threading.Lock()
        self._running = True
        print(f"[ImpulseUDPServer] Listening on udp://{host}:{self.port}")

    def run(self) -> None:
        while self._running:
            try:
                raw, address = self._sock.recvfrom(8192)
                try:
                    request = json.loads(raw.decode("utf-8"))
                    if not isinstance(request, dict):
                        raise ValueError("request must be a JSON object")
                except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
                    self.respond(
                        address,
                        {
                            "version": PROTOCOL_VERSION,
                            "id": "",
                            "status": "rejected",
                            "message": f"invalid JSON request: {exc}",
                        },
                    )
                    continue
                with self._lock:
                    self._queue.append(PendingImpulseRequest(request, address))
            except socket.timeout:
                continue
            except OSError:
                if self._running:
                    raise

    def pop_all(self) -> List[PendingImpulseRequest]:
        with self._lock:
            requests = list(self._queue)
            self._queue.clear()
        return requests

    def respond(self, address: Tuple[str, int], response: Dict[str, Any]) -> None:
        try:
            payload = json.dumps(response, allow_nan=False).encode("utf-8")
            self._sock.sendto(payload, address)
        except (OSError, TypeError, ValueError) as exc:
            if self._running:
                print(f"[ImpulseUDPServer] Failed to respond to {address}: {exc}")

    def stop(self) -> None:
        self._running = False
        try:
            self._sock.close()
        except OSError:
            pass


def impulse_csv_header(joint_names: Iterable[str]) -> List[str]:
    names = list(joint_names)
    header = [
        "sim_time_s",
        "wall_time_s",
        "event_id",
        "impulse_active",
        "body_name",
        "direction_frame",
        "requested_force_n",
        "requested_duration_s",
        "actual_duration_s",
        "elapsed_s",
        "remaining_s",
        "delivered_impulse_ns",
        "point_local_x_m",
        "point_local_y_m",
        "point_local_z_m",
        "point_world_x_m",
        "point_world_y_m",
        "point_world_z_m",
        "force_world_x_n",
        "force_world_y_n",
        "force_world_z_n",
        "base_x_m",
        "base_y_m",
        "base_z_m",
        "base_quat_w",
        "base_quat_x",
        "base_quat_y",
        "base_quat_z",
        "base_vx_m_s",
        "base_vy_m_s",
        "base_vz_m_s",
        "base_wx_rad_s",
        "base_wy_rad_s",
        "base_wz_rad_s",
    ]
    for prefix in ("q", "dq", "target", "tau"):
        header.extend(f"{prefix}_{name}" for name in names)
    return header


class ImpulseCSVLogger:
    """Write complete 200 Hz simulation state with impulse annotations."""

    def __init__(self, path: Path, joint_names: Iterable[str]):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.joint_names = list(joint_names)
        self._file = self.path.open("w", newline="")
        self._writer = csv.writer(self._file)
        self.header = impulse_csv_header(self.joint_names)
        self._writer.writerow(self.header)
        self._rows = 0
        self._last_state: Optional[Tuple[str, bool]] = None

    def write(
        self,
        *,
        data: mujoco.MjData,
        snapshot: Optional[ImpulseSnapshot],
        target_q: np.ndarray,
        wall_time_s: float,
    ) -> None:
        n = len(self.joint_names)
        q = np.asarray(data.qpos[7:7 + n], dtype=np.float64)
        dq = np.asarray(data.qvel[6:6 + n], dtype=np.float64)
        target = np.asarray(target_q, dtype=np.float64)
        tau = np.asarray(data.ctrl[:n], dtype=np.float64)
        arrays = (q, dq, target, tau)
        if any(array.shape != (n,) for array in arrays):
            raise ValueError("joint state, target, and torque arrays must match joint_names")

        if snapshot is None:
            impulse_values: List[Any] = [
                "", 0, "", "", 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
            ]
            impulse_values.extend([0.0] * 9)
            state = ("", False)
        else:
            command = snapshot.command
            applied_force = snapshot.force_world_n if snapshot.active else np.zeros(3)
            impulse_values = [
                command.request_id,
                int(snapshot.active),
                command.body,
                command.direction_frame,
                command.force_n,
                command.requested_duration_s,
                command.actual_duration_s,
                snapshot.elapsed_s,
                snapshot.remaining_s,
                snapshot.delivered_impulse_ns,
                *command.point_local_m,
                *snapshot.point_world_m,
                *applied_force,
            ]
            state = (command.request_id, snapshot.active)

        row: List[Any] = [float(data.time), float(wall_time_s), *impulse_values]
        row.extend(np.asarray(data.qpos[:7], dtype=np.float64).tolist())
        row.extend(np.asarray(data.qvel[:6], dtype=np.float64).tolist())
        row.extend(q.tolist())
        row.extend(dq.tolist())
        row.extend(target.tolist())
        row.extend(tau.tolist())
        if len(row) != len(self.header):
            raise RuntimeError(
                f"CSV row has {len(row)} columns, expected {len(self.header)}"
            )
        self._writer.writerow(row)
        self._rows += 1

        if self._last_state != state or self._rows % 200 == 0:
            self._file.flush()
        self._last_state = state

    def close(self) -> None:
        if not self._file.closed:
            self._file.flush()
            self._file.close()

    def __enter__(self) -> "ImpulseCSVLogger":
        return self

    def __exit__(self, *_args: Any) -> None:
        self.close()
