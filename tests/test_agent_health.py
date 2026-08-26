"""Tests for noticing, reporting, and repairing a missing event service.

Flask is only a bridge. With no Neuro SAN behind it, every utterance it accepts
is discarded, and the robot is indistinguishable from one that simply chose not
to answer. These cover the three things that stop that going unnoticed.
"""

import os
import unittest
from unittest.mock import MagicMock
from unittest.mock import patch
from urllib.error import HTTPError
from urllib.error import URLError

from apps.conscious_assistant import agent_runtime
from coded_tools.unigo2 import agent_events


class ServiceLivenessTests(unittest.TestCase):
    def test_a_serving_port_is_live(self):
        with patch.object(agent_runtime, "urlopen", MagicMock()):
            self.assertTrue(agent_runtime._service_is_live(8188))

    def test_an_http_error_still_proves_it_is_serving(self):
        # The endpoint only accepts POST, so refusing a GET is a healthy answer.
        error = HTTPError("http://127.0.0.1:8188", 405, "Method Not Allowed", {}, None)
        with patch.object(agent_runtime, "urlopen", side_effect=error):
            self.assertTrue(agent_runtime._service_is_live(8188))

    def test_a_refused_connection_is_not_live(self):
        with patch.object(
            agent_runtime, "urlopen", side_effect=URLError("Connection refused")
        ):
            self.assertFalse(agent_runtime._service_is_live(8188))

    def test_a_bound_but_dead_port_is_not_live(self):
        # A previous run still shutting down keeps the socket bound. Treating
        # that as a running service is how Flask ends up dispatching into
        # nothing for the rest of the session.
        with patch.object(agent_runtime, "_port_is_open", return_value=True), \
             patch.object(agent_runtime, "urlopen", side_effect=ConnectionRefusedError()):
            self.assertFalse(agent_runtime._service_is_live(8188))


class RuntimeStartupTests(unittest.TestCase):
    def setUp(self):
        # start() publishes the child's bridge token into this process so the
        # two sides agree on it. Snapshot the environment so that does not leak
        # into every later test's HTTP calls.
        patch.dict(os.environ).start()
        patch.object(agent_runtime, "_require_native_event_support").start()
        self.addCleanup(patch.stopall)

    def test_a_live_service_is_left_alone(self):
        runtime = agent_runtime.AgentRuntime()
        with patch.object(agent_runtime, "_service_is_live", return_value=True), \
             patch.object(agent_runtime.subprocess, "Popen") as popen:
            runtime.start()

        popen.assert_not_called()

    def test_a_bound_but_dead_port_does_not_block_startup(self):
        runtime = agent_runtime.AgentRuntime()
        process = MagicMock()
        process.poll.return_value = None
        # Dead on the pre-flight check, serving once our own process is up.
        with patch.object(agent_runtime, "_service_is_live", side_effect=[False, True]), \
             patch.object(agent_runtime.subprocess, "Popen", return_value=process) as popen, \
             patch.object(agent_runtime, "clear_navigation_awareness", create=True):
            runtime.start()

        popen.assert_called_once()

    def test_a_dead_service_is_restarted(self):
        runtime = agent_runtime.AgentRuntime()
        with patch.object(runtime, "is_alive", side_effect=[False, True]), \
             patch.object(runtime, "stop") as stop, \
             patch.object(runtime, "start") as start:
            self.assertTrue(runtime.ensure_running())

        stop.assert_called_once()
        start.assert_called_once()

    def test_a_healthy_service_is_not_restarted(self):
        runtime = agent_runtime.AgentRuntime()
        with patch.object(runtime, "is_alive", return_value=True), \
             patch.object(runtime, "start") as start:
            self.assertTrue(runtime.ensure_running())

        start.assert_not_called()

    def test_an_exited_process_is_not_alive(self):
        runtime = agent_runtime.AgentRuntime()
        runtime.process = MagicMock()
        runtime.process.poll.return_value = 1

        self.assertFalse(runtime.is_alive())


class DispatchFailureReportingTests(unittest.TestCase):
    def setUp(self):
        self.reported = []
        # Restore whatever was registered; clearing it would silently disarm
        # the app's own reporting for every test that runs after this one.
        previous = agent_events._DISPATCH_FAILURE_HOOK
        self.addCleanup(agent_events.set_dispatch_failure_hook, previous)
        agent_events.set_dispatch_failure_hook(
            lambda source, exc: self.reported.append((source, exc))
        )

    def test_an_undeliverable_event_is_reported_not_just_logged(self):
        with patch.object(
            agent_events, "_post_json", side_effect=URLError("Connection refused")
        ):
            accepted = agent_events.dispatch_agent_event("turn right", source="user")

        self.assertFalse(accepted)
        self.assertEqual(len(self.reported), 1)
        self.assertEqual(self.reported[0][0], "user")

    def test_an_undeliverable_event_is_logged_at_error(self):
        with patch.object(
            agent_events, "_post_json", side_effect=URLError("Connection refused")
        ):
            with self.assertLogs(agent_events.logger, level="ERROR"):
                agent_events.dispatch_agent_event("turn right", source="ambient")

    def test_a_delivered_event_reports_nothing(self):
        with patch.object(agent_events, "_post_json"):
            self.assertTrue(agent_events.dispatch_agent_event("hello", source="user"))

        self.assertEqual(self.reported, [])

    def test_a_raising_hook_cannot_break_dispatch(self):
        agent_events.set_dispatch_failure_hook(
            lambda source, exc: (_ for _ in ()).throw(RuntimeError("boom"))
        )
        with patch.object(
            agent_events, "_post_json", side_effect=URLError("Connection refused")
        ):
            self.assertFalse(agent_events.dispatch_agent_event("hi", source="user"))


if __name__ == "__main__":
    unittest.main()
