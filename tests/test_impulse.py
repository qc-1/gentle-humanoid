import csv
import json
import socket
import sys
import tempfile
import time
import unittest
from pathlib import Path

import mujoco
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from common.impulse import (  # noqa: E402
    ImpulseCSVLogger,
    ImpulseController,
    ImpulseUDPServer,
    ImpulseValidationError,
    PROTOCOL_VERSION,
    validate_impulse_request,
)


class ImpulseTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model_path = ROOT / "assets" / "g1" / "g1.xml"

    def setUp(self):
        self.model = mujoco.MjModel.from_xml_path(str(self.model_path))
        self.model.opt.timestep = 0.005
        self.data = mujoco.MjData(self.model)
        mujoco.mj_forward(self.model, self.data)
        self.controller = ImpulseController(self.model, 0.005)

    def request(self, command="fire", request_id="test", **overrides):
        request = {
            "version": PROTOCOL_VERSION,
            "id": request_id,
            "command": command,
            "body": "left_elbow_link",
            "force_n": 30.0,
            "duration_s": 0.1,
            "point_local_m": [0.05, 0.0, 0.0],
            "direction": [1.0, 0.0, 0.0],
            "direction_frame": "world",
        }
        request.update(overrides)
        return request

    def test_validation_and_duration_quantization(self):
        command = validate_impulse_request(self.request(), self.model, 0.005)
        self.assertEqual(command.steps, 20)
        self.assertAlmostEqual(command.actual_duration_s, 0.1)
        self.assertAlmostEqual(command.impulse_ns, 3.0)

        short = validate_impulse_request(
            self.request(duration_s=0.001), self.model, 0.005
        )
        self.assertEqual(short.steps, 1)
        self.assertAlmostEqual(short.actual_duration_s, 0.005)

    def test_invalid_requests(self):
        invalid_overrides = [
            {"body": "missing_link"},
            {"body": "world"},
            {"force_n": 0.0},
            {"force_n": float("nan")},
            {"duration_s": -0.1},
            {"point_local_m": [0.0, 0.0]},
            {"direction": [0.0, 0.0, 0.0]},
            {"direction_frame": "camera"},
        ]
        for overrides in invalid_overrides:
            with self.subTest(overrides=overrides):
                with self.assertRaises(ImpulseValidationError):
                    validate_impulse_request(
                        self.request(**overrides), self.model, 0.005
                    )

    def test_world_and_body_direction_frames(self):
        half_sqrt = np.sqrt(0.5)
        self.data.qpos[3:7] = [half_sqrt, 0.0, 0.0, half_sqrt]
        mujoco.mj_forward(self.model, self.data)

        body_response = self.controller.handle_request(
            self.request(
                command="preview",
                request_id="body",
                body="torso_link",
                point_local_m=[0.0, 0.0, 0.0],
                direction_frame="body",
            ),
            control_active=False,
        )
        self.assertEqual(body_response["status"], "accepted")
        body_snapshot = self.controller.sample_preview(self.data)
        np.testing.assert_allclose(
            body_snapshot.direction_world, [0.0, 1.0, 0.0], atol=1e-6
        )

        world_response = self.controller.handle_request(
            self.request(
                command="preview",
                request_id="world",
                body="torso_link",
                point_local_m=[0.0, 0.0, 0.0],
                direction_frame="world",
            ),
            control_active=False,
        )
        self.assertEqual(world_response["status"], "accepted")
        world_snapshot = self.controller.sample_preview(self.data)
        np.testing.assert_allclose(
            world_snapshot.direction_world, [1.0, 0.0, 0.0], atol=1e-6
        )

    def test_force_is_applied_at_offset_point(self):
        response = self.controller.handle_request(
            self.request(), control_active=True
        )
        self.assertEqual(response["status"], "accepted")
        snapshot = self.controller.step(self.data)
        self.assertTrue(snapshot.active)
        self.assertGreater(np.linalg.norm(self.data.qfrc_applied), 0.0)
        self.assertGreater(np.linalg.norm(self.data.qfrc_applied[6:]), 0.0)

    def test_timing_overlap_cancel_and_force_warning(self):
        response = self.controller.handle_request(
            self.request(force_n=45.0), control_active=True
        )
        self.assertEqual(response["status"], "accepted")
        self.assertIn("warning", response)

        self.controller.step(self.data)
        overlap = self.controller.handle_request(
            self.request(request_id="overlap"), control_active=True
        )
        self.assertEqual(overlap["status"], "rejected")

        cancel = self.controller.handle_request(
            {
                "version": PROTOCOL_VERSION,
                "id": "cancel",
                "command": "cancel",
            },
            control_active=True,
        )
        self.assertEqual(cancel["status"], "accepted")
        snapshot = self.controller.step(self.data)
        self.assertFalse(snapshot.active)
        self.assertEqual(np.count_nonzero(self.data.qfrc_applied), 0)

        response = self.controller.handle_request(
            self.request(request_id="complete"), control_active=True
        )
        self.assertEqual(response["status"], "accepted")
        snapshots = [self.controller.step(self.data) for _ in range(20)]
        self.assertTrue(all(snapshot.active for snapshot in snapshots))
        self.assertAlmostEqual(snapshots[-1].delivered_impulse_ns, 3.0)
        self.assertFalse(self.controller.is_active)

    def test_fire_rejected_before_control_loop(self):
        response = self.controller.handle_request(
            self.request(), control_active=False
        )
        self.assertEqual(response["status"], "rejected")
        self.assertFalse(self.controller.is_active)

        missing_id = self.controller.handle_request(
            {"version": PROTOCOL_VERSION, "command": "cancel"},
            control_active=False,
        )
        self.assertEqual(missing_id["status"], "rejected")

    def test_csv_shape_and_event_data(self):
        response = self.controller.handle_request(
            self.request(), control_active=True
        )
        self.assertEqual(response["status"], "accepted")
        snapshot = self.controller.step(self.data)
        joint_names = [
            mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, index)
            for index in range(1, self.model.njnt)
        ]
        self.assertEqual(len(joint_names), 29)

        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "impulse.csv"
            logger = ImpulseCSVLogger(path, joint_names)
            logger.write(
                data=self.data,
                snapshot=snapshot,
                target_q=np.zeros(29),
                wall_time_s=0.005,
            )
            logger.close()
            with path.open(newline="") as file:
                rows = list(csv.reader(file))

        self.assertEqual(len(rows), 2)
        self.assertEqual(len(rows[0]), len(rows[1]))
        event_index = rows[0].index("event_id")
        active_index = rows[0].index("impulse_active")
        self.assertEqual(rows[1][event_index], "test")
        self.assertEqual(rows[1][active_index], "1")


class ImpulseUDPServerTest(unittest.TestCase):
    def test_json_round_trip(self):
        try:
            server = ImpulseUDPServer("127.0.0.1", 0)
        except PermissionError as exc:
            self.skipTest(f"local UDP sockets are disabled in this environment: {exc}")
        server.start()
        client = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        client.settimeout(1.0)
        request = {
            "version": PROTOCOL_VERSION,
            "id": "round-trip",
            "command": "cancel",
        }
        client.sendto(json.dumps(request).encode(), ("127.0.0.1", server.port))

        pending = []
        deadline = time.monotonic() + 1.0
        while not pending and time.monotonic() < deadline:
            pending = server.pop_all()
            time.sleep(0.005)
        self.assertEqual(len(pending), 1)
        self.assertEqual(pending[0].request, request)

        response = {
            "version": PROTOCOL_VERSION,
            "id": "round-trip",
            "status": "accepted",
            "message": "ok",
        }
        server.respond(pending[0].address, response)
        raw, _ = client.recvfrom(8192)
        self.assertEqual(json.loads(raw.decode()), response)

        client.close()
        server.stop()
        server.join(timeout=1.0)


if __name__ == "__main__":
    unittest.main()
