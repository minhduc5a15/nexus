from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from modal.exception import TimeoutError as ModalTimeoutError

from scripts import modal_sft_v3_control as control


class FakeRegistry:
    def __init__(self):
        self.values = {}

    def put(self, key, value, *, skip_if_exists=False):
        if skip_if_exists and key in self.values:
            return False
        self.values[key] = value
        return True

    def get(self, key, default=None):
        return self.values.get(key, default)


class FakeFunction:
    def __init__(self, call_id="fc-test", error=None):
        self.call_id = call_id
        self.error = error
        self.calls = []

    def spawn(self, *args):
        self.calls.append(args)
        if self.error:
            raise self.error
        return type("Call", (), {"object_id": self.call_id})()


class FakeCall:
    def __init__(self, result=None, running=False):
        self.result = result
        self.running = running

    def get(self, timeout=None):
        if self.running:
            raise ModalTimeoutError("still running")
        return self.result


class BuiltinTimeoutCall:
    def get(self, timeout=None):
        raise TimeoutError()


class ModalSftV3ControlTests(unittest.TestCase):
    def record_path(self) -> Path:
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        return Path(directory.name) / "launch.json"

    def test_canary_reserves_before_single_spawn_and_persists_call_id(self):
        registry = FakeRegistry()
        function = FakeFunction()
        path = self.record_path()
        record = control.reserve_and_spawn_canary(
            "canary-1",
            30,
            registry=registry,
            function=function,
            now=lambda: "now",
            commit="abc",
            record_path=path,
        )
        self.assertEqual(function.calls, [("canary-1", 30)])
        self.assertEqual(record["state"], "spawned")
        self.assertEqual(record["function_call_id"], "fc-test")
        self.assertEqual(registry.values["canary-1"], record)
        self.assertIn('"function_call_id": "fc-test"', path.read_text())

    def test_duplicate_run_id_is_rejected_without_spawning(self):
        registry = FakeRegistry()
        registry.values["canary-1"] = {"state": "spawned"}
        function = FakeFunction()
        with self.assertRaisesRegex(control.ModalControlError, "duplicate spawn"):
            control.reserve_and_spawn_canary(
                "canary-1",
                30,
                registry=registry,
                function=function,
                commit="abc",
                record_path=self.record_path(),
            )
        self.assertEqual(function.calls, [])

    def test_spawn_failure_is_recorded_and_not_retried(self):
        registry = FakeRegistry()
        function = FakeFunction(error=RuntimeError("boom"))
        with self.assertRaisesRegex(RuntimeError, "boom"):
            control.reserve_and_spawn_canary(
                "canary-1",
                30,
                registry=registry,
                function=function,
                now=lambda: "now",
                commit="abc",
                record_path=self.record_path(),
            )
        self.assertEqual(registry.values["canary-1"]["state"], "spawn_failed")

    def test_status_poll_does_not_cancel_running_call(self):
        registry = FakeRegistry()
        registry.values["canary-1"] = {
            "run_id": "canary-1",
            "state": "remote_failed",
            "function_call_id": "fc-test",
            "failed_at": "earlier",
            "error": {"type": "TimeoutError", "message": ""},
        }
        call = FakeCall(running=True)
        record = control.poll_launch(
            "canary-1",
            registry=registry,
            call_from_id=lambda call_id: call,
            now=lambda: "later",
            record_path=self.record_path(),
        )
        self.assertEqual(record["state"], "running")
        self.assertEqual(record["function_call_id"], "fc-test")
        self.assertNotIn("failed_at", record)
        self.assertNotIn("error", record)
        self.assertFalse(hasattr(call, "cancel_called"))

    def test_builtin_timeout_from_modal_client_also_means_running(self):
        registry = FakeRegistry()
        registry.values["canary-1"] = {
            "run_id": "canary-1",
            "state": "spawned",
            "function_call_id": "fc-test",
        }
        record = control.poll_launch(
            "canary-1",
            registry=registry,
            call_from_id=lambda call_id: BuiltinTimeoutCall(),
            now=lambda: "later",
            record_path=self.record_path(),
        )
        self.assertEqual(record["state"], "running")

    def test_completed_status_persists_remote_result(self):
        registry = FakeRegistry()
        registry.values["canary-1"] = {
            "run_id": "canary-1",
            "state": "spawned",
            "function_call_id": "fc-test",
        }
        result = {"stage": "orchestration_canary", "gpu_used": False}
        record = control.poll_launch(
            "canary-1",
            registry=registry,
            call_from_id=lambda call_id: FakeCall(result=result),
            now=lambda: "later",
            record_path=self.record_path(),
        )
        self.assertEqual(record["state"], "complete")
        self.assertEqual(record["result"], result)

    def test_completed_status_is_idempotent_without_remote_poll(self):
        registry = FakeRegistry()
        existing = {
            "run_id": "canary-1",
            "state": "complete",
            "function_call_id": "fc-test",
            "completed_at": "first",
            "result": {"gpu_used": False},
        }
        registry.values["canary-1"] = existing

        def unexpected_call(call_id):
            self.fail("completed call should not be polled again")

        record = control.poll_launch(
            "canary-1",
            registry=registry,
            call_from_id=unexpected_call,
            now=lambda: "later",
            record_path=self.record_path(),
        )
        self.assertEqual(record, existing)

    def test_remote_failure_is_persisted(self):
        registry = FakeRegistry()
        registry.values["canary-1"] = {
            "run_id": "canary-1",
            "state": "spawned",
            "function_call_id": "fc-test",
        }

        class FailedCall:
            def get(self, timeout=None):
                raise RuntimeError("remote boom")

        with self.assertRaisesRegex(RuntimeError, "remote boom"):
            control.poll_launch(
                "canary-1",
                registry=registry,
                call_from_id=lambda call_id: FailedCall(),
                now=lambda: "later",
                record_path=self.record_path(),
            )
        self.assertEqual(registry.values["canary-1"]["state"], "remote_failed")
        self.assertEqual(
            registry.values["canary-1"]["error"]["type"], "RuntimeError"
        )

    def test_invalid_duration_and_run_id_are_rejected_before_reservation(self):
        for run_id, duration in (("bad/run", 30), ("ok", 29), ("ok", True)):
            registry = FakeRegistry()
            with self.subTest(run_id=run_id, duration=duration):
                with self.assertRaises(control.ModalControlError):
                    control.reserve_and_spawn_canary(
                        run_id,
                        duration,
                        registry=registry,
                        function=FakeFunction(),
                        commit="abc",
                        record_path=self.record_path(),
                    )
                self.assertEqual(registry.values, {})


if __name__ == "__main__":
    unittest.main()
