"""Interactive client for configuring force pulses in sim2sim."""

from __future__ import annotations

import argparse
import json
import math
import socket
import uuid
from pathlib import Path
from typing import Any, Dict, List

import mujoco
import numpy as np

from common.impulse import (
    DEFAULT_IMPULSE_PORT,
    PROTOCOL_VERSION,
    TRAINING_FORCE_LIMIT_N,
)
from paths import ASSETS_DIR, REAL_G1_ROOT


BANNER = """\
Impulse Sender
  list                     list available MuJoCo bodies
  show                     show the current pulse configuration
  body NAME                select a target body
  force N                  set force magnitude in newtons
  duration SEC             set pulse duration in seconds
  point X Y Z              set body-local application point in metres
  direction X Y Z          set the direction vector
  frame world|body         select the direction coordinate frame
  preview                  update the marker/arrow without applying force
  fire                     apply the configured force pulse
  cancel                   stop the active pulse at the next physics step
  help                     show this help
  quit                     exit this sender
"""


def resolve_model_path(path: str) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    under_assets = ASSETS_DIR / candidate
    if under_assets.exists():
        return under_assets
    return REAL_G1_ROOT / candidate


def available_bodies(model: mujoco.MjModel) -> List[str]:
    return [
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id)
        for body_id in range(1, model.nbody)
    ]


def request_payload(command: str, config: Dict[str, Any]) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "version": PROTOCOL_VERSION,
        "id": uuid.uuid4().hex,
        "command": command,
    }
    if command in {"preview", "fire"}:
        payload.update(config)
    return payload


def send_request(
    sock: socket.socket,
    host: str,
    port: int,
    payload: Dict[str, Any],
) -> Dict[str, Any]:
    encoded = json.dumps(payload, allow_nan=False).encode("utf-8")
    sock.sendto(encoded, (host, port))
    while True:
        raw, _ = sock.recvfrom(8192)
        response = json.loads(raw.decode("utf-8"))
        if response.get("id") in {payload["id"], ""}:
            return response


def parse_vector(parts: List[str], name: str) -> List[float]:
    if len(parts) != 3:
        raise ValueError(f"{name} requires exactly three values")
    vector = [float(v) for v in parts]
    if not all(math.isfinite(v) for v in vector):
        raise ValueError(f"{name} values must be finite")
    return vector


def validate_local_config(config: Dict[str, Any], bodies: List[str]) -> None:
    if config["body"] not in bodies:
        raise ValueError(f"unknown body: '{config['body']}'")
    if not math.isfinite(config["force_n"]) or config["force_n"] <= 0.0:
        raise ValueError("force must be a positive finite number")
    if not math.isfinite(config["duration_s"]) or config["duration_s"] <= 0.0:
        raise ValueError("duration must be a positive finite number")
    direction = np.asarray(config["direction"], dtype=np.float64)
    if not np.all(np.isfinite(direction)) or np.linalg.norm(direction) <= 1e-12:
        raise ValueError("direction must be finite and non-zero")


def print_config(config: Dict[str, Any]) -> None:
    requested_impulse = config["force_n"] * config["duration_s"]
    print("\nCurrent configuration")
    print(f"  body:       {config['body']}")
    print(f"  force:      {config['force_n']:g} N")
    print(f"  duration:   {config['duration_s']:g} s")
    print(f"  impulse:    {requested_impulse:g} N*s (before timestep quantization)")
    print(f"  local point:{config['point_local_m']}")
    print(f"  direction:  {config['direction']} ({config['direction_frame']} frame)\n")


def print_response(response: Dict[str, Any]) -> None:
    status = response.get("status", "unknown").upper()
    print(f"[{status}] {response.get('message', '')}")
    if "actual_duration_s" in response:
        print(
            f"  actual duration={response['actual_duration_s']:.6g} s, "
            f"impulse={response['impulse_ns']:.6g} N*s"
        )
    if response.get("warning"):
        print(f"  WARNING: {response['warning']}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Interactive MuJoCo impulse sender")
    parser.add_argument(
        "--xml_path", type=str, default=str(ASSETS_DIR / "g1" / "g1.xml")
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=DEFAULT_IMPULSE_PORT)
    parser.add_argument("--timeout", type=float, default=1.0)
    args = parser.parse_args()

    model_path = resolve_model_path(args.xml_path)
    model = mujoco.MjModel.from_xml_path(str(model_path))
    bodies = available_bodies(model)

    config: Dict[str, Any] = {
        "body": "torso_link",
        "force_n": 30.0,
        "duration_s": 0.1,
        "point_local_m": [0.0, 0.0, 0.0],
        "direction": [1.0, 0.0, 0.0],
        "direction_frame": "world",
    }

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.settimeout(args.timeout)
    print(BANNER)
    print(f"Target: udp://{args.host}:{args.port}")
    print_config(config)

    while True:
        try:
            user_input = input("impulse> ").strip()
            if not user_input:
                continue
            parts = user_input.split()
            command = parts[0].lower()
            values = parts[1:]

            if command in {"q", "quit", "exit"}:
                break
            if command in {"help", "h", "?"}:
                print(BANNER)
            elif command in {"list", "l"}:
                for index, body in enumerate(bodies, 1):
                    print(f"  {index:2d}. {body}")
            elif command == "show":
                print_config(config)
            elif command == "body":
                if len(values) != 1 or values[0] not in bodies:
                    raise ValueError("body requires one name from 'list'")
                config["body"] = values[0]
                print(f"body = {config['body']}")
            elif command == "force":
                if len(values) != 1:
                    raise ValueError("force requires one value in newtons")
                force_n = float(values[0])
                if not math.isfinite(force_n) or force_n <= 0.0:
                    raise ValueError("force must be a positive finite number")
                config["force_n"] = force_n
                print(f"force = {force_n:g} N")
                if force_n > TRAINING_FORCE_LIMIT_N:
                    print(
                        f"WARNING: values above {TRAINING_FORCE_LIMIT_N:g} N are "
                        "outside the training-force range"
                    )
            elif command == "duration":
                if len(values) != 1:
                    raise ValueError("duration requires one value in seconds")
                duration_s = float(values[0])
                if not math.isfinite(duration_s) or duration_s <= 0.0:
                    raise ValueError("duration must be a positive finite number")
                config["duration_s"] = duration_s
                print(f"duration = {duration_s:g} s")
            elif command == "point":
                config["point_local_m"] = parse_vector(values, "point")
                print(f"point_local_m = {config['point_local_m']}")
            elif command == "direction":
                direction = parse_vector(values, "direction")
                if np.linalg.norm(direction) <= 1e-12:
                    raise ValueError("direction must be non-zero")
                config["direction"] = direction
                print(f"direction = {direction}")
            elif command == "frame":
                if len(values) != 1 or values[0].lower() not in {"world", "body"}:
                    raise ValueError("frame must be 'world' or 'body'")
                config["direction_frame"] = values[0].lower()
                print(f"direction_frame = {config['direction_frame']}")
            elif command in {"preview", "fire"}:
                if values:
                    raise ValueError(f"{command} does not accept arguments")
                validate_local_config(config, bodies)
                response = send_request(
                    sock,
                    args.host,
                    args.port,
                    request_payload(command, config),
                )
                print_response(response)
            elif command == "cancel":
                if values:
                    raise ValueError("cancel does not accept arguments")
                response = send_request(
                    sock,
                    args.host,
                    args.port,
                    request_payload("cancel", config),
                )
                print_response(response)
            else:
                print(f"Unknown command: '{command}'. Type 'help' for usage.")
        except socket.timeout:
            print(
                f"[TIMEOUT] No response from udp://{args.host}:{args.port}; "
                "is sim2sim running?"
            )
        except (ValueError, json.JSONDecodeError) as exc:
            print(f"[ERROR] {exc}")
        except (KeyboardInterrupt, EOFError):
            print()
            break
        except OSError as exc:
            print(f"[ERROR] UDP request failed: {exc}")

    sock.close()
    print("Bye.")


if __name__ == "__main__":
    main()
