import unittest
from pathlib import Path
from unittest.mock import MagicMock
from unittest.mock import patch

from coded_tools.unigo2 import agent_events
from coded_tools.unigo2.robot_macros import RobotMacros
from apps.conscious_assistant import agent_runtime


ROOT = Path(__file__).resolve().parents[1]


class NativeEventRuntimeTests(unittest.TestCase):
    def test_flask_is_not_an_agent_scheduler(self):
        source = (ROOT / "apps" / "conscious_assistant" / "interface_flask.py").read_text()

        self.assertNotIn("conscious_thinking_process", source)
        self.assertNotIn("conscious_thinker", source)
        self.assertNotIn("StreamingInputProcessor", source)
        self.assertIn("dispatch_agent_event(user_input, source=\"user\")", source)

    def test_agent_output_is_an_explicit_tool_boundary(self):
        source = (ROOT / "registries" / "conscious_agent.hocon").read_text()

        self.assertIn('"invocation": "event"', source)
        self.assertIn('"ui_output"', source)
        self.assertNotIn('"scene_observer"', source)
        self.assertIn("only content that appears in the Thoughts pane", source)

    def test_navigation_has_one_event_aware_agent_tool(self):
        source = (ROOT / "registries" / "conscious_agent.hocon").read_text()

        self.assertIn('"name": "nav_planner"', source)
        self.assertNotIn('"name": "nav_status"', source)

    def test_internal_events_are_not_treated_as_user_speech(self):
        source = (ROOT / "registries" / "conscious_agent.hocon").read_text()

        self.assertIn("Only `user:` is a person speaking to you", source)
        self.assertIn("`observation:`", source)

    def test_scene_observer_is_a_dedicated_event_network(self):
        source = (ROOT / "registries" / "scene_observer_agent.hocon").read_text()

        self.assertNotIn('"parameters": {"type": "object", "properties": {}}', source)
        self.assertIn('"invocation": "event"', source)
        self.assertIn('"name": "capture_scene"', source)
        self.assertIn('"capture": {', source)

    def test_native_periodic_turns_are_manifest_owned(self):
        source = (ROOT / "registries" / "manifest.hocon").read_text()

        conscious_config = source.split('"conscious_agent.hocon"', 1)[1].split(
            '"scene_observer_agent.hocon"', 1
        )[0]
        observer_config = source.split('"scene_observer_agent.hocon"', 1)[1]

        self.assertNotIn('"periodic"', conscious_config)
        self.assertIn('"periodic"', observer_config)
        self.assertIn('"cron_schedule": "* * * * * */15"', observer_config)
        self.assertIn('"text": "system: [Silence]"', source)

    def test_native_event_turns_have_bounded_execution(self):
        source = (ROOT / "registries" / "conscious_agent.hocon").read_text()
        observer_source = (ROOT / "registries" / "scene_observer_agent.hocon").read_text()

        self.assertIn('"max_steps": 12', source)
        self.assertIn('"max_execution_seconds": 60', source)
        self.assertIn('"max_steps": 8', observer_source)
        self.assertIn('"max_execution_seconds": 60', observer_source)

    def test_dispatch_agent_event_posts_a_minimal_event(self):
        with patch.object(agent_events, "_post_json") as post:
            self.assertTrue(agent_events.dispatch_agent_event("go home", source="user"))

        endpoint, payload = post.call_args.args[:2]
        self.assertIn("/conscious_agent/streaming_chat", endpoint)
        self.assertEqual(payload["user_message"]["text"], "user: go home")
        self.assertEqual(payload["chat_filter"]["chat_filter_type"], "MINIMAL")

    def test_navigation_events_are_queued_off_the_control_loop(self):
        with patch.object(agent_events._EVENT_DISPATCHER, "submit") as submit:
            agent_events.queue_agent_event("waypoint reached", source="navigation")

        submit.assert_called_once_with(
            agent_events.dispatch_agent_event,
            "waypoint reached",
            source="navigation",
        )

    def test_event_bridge_consumes_the_full_native_acknowledgement(self):
        response = MagicMock()
        opener = MagicMock()
        opener.__enter__.return_value = response
        with patch.object(agent_events, "urlopen", return_value=opener):
            agent_events._post_json("http://127.0.0.1:8188/event", {"event": "wake"})

        response.read.assert_called_once_with()

    def test_event_bridge_accepts_only_loopback_self_signed_tls(self):
        response = MagicMock()
        opener = MagicMock()
        opener.__enter__.return_value = response
        with patch.object(agent_events, "urlopen", return_value=opener) as post:
            agent_events._post_json("https://127.0.0.1:5001/event", {"event": "wake"})

        self.assertIn("context", post.call_args.kwargs)
        self.assertIsNone(agent_events._local_ssl_context("https://example.com/event"))

    def test_native_event_runtime_requires_event_continuation_release(self):
        with patch.object(agent_runtime, "version", return_value="0.6.48"):
            with self.assertRaisesRegex(RuntimeError, "native event continuation"):
                agent_runtime._require_native_event_support()

        with patch.object(agent_runtime, "version", return_value="0.6.76"):
            agent_runtime._require_native_event_support()

    def test_interface_has_no_agent_input_queue(self):
        source = (ROOT / "apps" / "conscious_assistant" / "interface_flask.py").read_text()
        self.assertNotIn("user_input_queue", source)
        self.assertNotIn("THINKING_INTERVAL", source)

    def test_microphone_is_locked_only_during_tts(self):
        source = (ROOT / "apps" / "conscious_assistant" / "templates" / "index.html").read_text()

        self.assertIn("function isVoiceInputLocked()", source)
        self.assertIn("return isSpeaking;", source)
        self.assertNotIn("return isProcessing || isSpeaking;", source)

    def test_flask_owns_the_native_ui_callback_url(self):
        source = (ROOT / "apps" / "conscious_assistant" / "interface_flask.py").read_text()

        self.assertIn('os.environ["CONSCIOUS_UI_EVENT_ENDPOINT"] = _ui_event_endpoint()', source)
        self.assertIn('TLS_CERT = Path("/home/unitree/certs/cert.pem")', source)
        self.assertNotIn('CERT = "/home/unitree/certs/cert.pem"', source)

    def test_robot_actions_execute_in_the_native_agent_process(self):
        class FakeGo2:
            available = True

            def stop_move(self):
                self.stopped = True

        fake_go2 = FakeGo2()
        with patch("coded_tools.unigo2.robot_macros.Go2Macros", return_value=fake_go2):
            result = RobotMacros._execute("stop_move", {})

        self.assertTrue(fake_go2.stopped)
        self.assertEqual(result, "Action 'stop_move' completed successfully")


if __name__ == "__main__":
    unittest.main()
