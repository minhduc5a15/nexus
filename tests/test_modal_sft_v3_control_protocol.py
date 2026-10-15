from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts import modal_sft_v3_control_protocol as protocol


class ModalSftV3ControlProtocolTests(unittest.TestCase):
    def write(self, name: str, text: str) -> Path:
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        path = Path(directory.name) / name
        path.write_text(text, encoding="utf-8")
        return path

    def test_protocol_keeps_canary_cpu_only_and_paid_stages_blocked(self):
        summary = protocol.validate_protocol()
        self.assertEqual(
            summary["resources"], {"timeout": 300, "cpu": 0.25, "memory": 256}
        )
        self.assertTrue(summary["atomic_duplicate_guard"])
        self.assertFalse(summary["launcher_waits_for_remote_result"])
        self.assertFalse(summary["status_can_cancel"])
        self.assertFalse(summary["training_authorized"])
        self.assertFalse(summary["final_holdout_inference_authorized"])

    def test_gpu_canary_is_rejected(self):
        text = protocol.RUNNER.read_text(encoding="utf-8").replace(
            "timeout=300,\n    cpu=0.25,",
            'timeout=300,\n    gpu="T4",\n    cpu=0.25,',
            1,
        )
        with patch.object(protocol, "RUNNER", self.write("runner.py", text)):
            with self.assertRaisesRegex(
                protocol.ModalControlProtocolError, "CPU-only"
            ):
                protocol.validate_protocol()

    def test_cancel_capability_is_rejected(self):
        text = protocol.CONTROL.read_text(encoding="utf-8") + "\ncall.cancel()\n"
        with patch.object(protocol, "CONTROL", self.write("control.py", text)):
            with self.assertRaisesRegex(
                protocol.ModalControlProtocolError, "must not cancel"
            ):
                protocol.validate_protocol()

    def test_training_authorization_must_remain_closed(self):
        value = json.loads(protocol.PLAN.read_text(encoding="utf-8"))
        value["training_authorized"] = True
        path = self.write("plan.json", json.dumps(value))
        with patch.object(protocol, "PLAN", path):
            with self.assertRaisesRegex(
                protocol.ModalControlProtocolError, "Training must remain blocked"
            ):
                protocol.validate_protocol()


if __name__ == "__main__":
    unittest.main()
