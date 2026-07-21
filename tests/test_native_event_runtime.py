import unittest
from pathlib import Path
from unittest.mock import patch

from coded_tools.unigo2 import agent_events
from coded_tools.unigo2.robot_macros import RobotMacros


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
        self.assertIn('"scene_observer"', source)
        self.assertIn("only content that appears in the Thoughts pane", source)

    def test_native_periodic_turns_are_manifest_owned(self):
        source = (ROOT / "registries" / "manifest.hocon").read_text()

        self.assertIn('"periodic"', source)
        self.assertIn('"text": "system: [Silence]"', source)

    def test_dispatch_agent_event_posts_a_minimal_event(self):
        with patch.object(agent_events, "_post_json") as post:
            self.assertTrue(agent_events.dispatch_agent_event("go home", source="user"))

        endpoint, payload = post.call_args.args[:2]
        self.assertIn("/conscious_agent/streaming_chat", endpoint)
        self.assertEqual(payload["user_message"]["text"], "user: go home")
        self.assertEqual(payload["chat_filter"]["chat_filter_type"], "MINIMAL")

    def test_interface_has_no_agent_input_queue(self):
        source = (ROOT / "apps" / "conscious_assistant" / "interface_flask.py").read_text()
        self.assertNotIn("user_input_queue", source)
        self.assertNotIn("THINKING_INTERVAL", source)

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
