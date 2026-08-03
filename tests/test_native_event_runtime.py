import unittest
import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock
from unittest.mock import patch

from coded_tools.unigo2 import agent_events
from coded_tools.unigo2.robot_macros import RobotMacros
from apps.conscious_assistant import agent_runtime


ROOT = Path(__file__).resolve().parents[1]


class NativeEventRuntimeTests(unittest.TestCase):

    def test_navigation_updates_are_retained_silently(self):
        source = (ROOT / "coded_tools/unigo2/nav_planner.py").read_text()
        self.assertIn("remember_navigation_awareness", source)
        self.assertNotIn(
            'queue_agent_event(message, source="navigation")',
            source,
        )

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
        self.assertIn("Call ui_output at most once per event turn", source)
        self.assertIn("Never call ui_output with both fields empty", source)
        self.assertIn('"required": ["thought", "say", "heard"]', source)

    def test_navigation_has_one_event_aware_agent_tool(self):
        source = (ROOT / "registries" / "conscious_agent.hocon").read_text()

        self.assertIn('"name": "nav_planner"', source)
        self.assertNotIn('"name": "nav_status"', source)
        self.assertIn("Treat those events as authoritative", source)
        self.assertIn("call nav_planner with command `status` before answering", source)
        self.assertIn("Do not call set_location in response", source)
        self.assertIn("You must use command 'status'", source)

    def test_internal_events_are_not_treated_as_user_speech(self):
        source = (ROOT / "registries" / "conscious_agent.hocon").read_text()

        self.assertIn("Only `user:` is guaranteed to be direct input", source)
        self.assertIn("`observation:`", source)
        self.assertIn("For `observation:` and `system:` events, never use the `say` field", source)

    def test_ambient_speech_is_sent_to_the_agent_without_a_prefilter(self):
        source = (ROOT / "registries" / "conscious_agent.hocon").read_text()
        interface_source = (ROOT / "apps" / "conscious_assistant" / "interface_flask.py").read_text()

        self.assertIn("`ambient:` is an automatic transcription", source)
        self.assertIn("remain completely silent", source)
        self.assertIn("only the relevant addressed speech", source)
        self.assertIn('queue_agent_event(transcript, source="ambient")', interface_source)
        self.assertNotIn("ambient_llm_filter", interface_source)

    def test_ambient_audio_uses_a_persistent_realtime_stream(self):
        interface_source = (
            ROOT / "apps" / "conscious_assistant" / "interface_flask.py"
        ).read_text()
        browser_source = (
            ROOT / "apps" / "conscious_assistant" / "templates" / "index.html"
        ).read_text()

        self.assertIn('/api/realtime/transcription-token', interface_source)
        self.assertIn("new RTCPeerConnection()", browser_source)
        self.assertIn("conversation.item.input_audio_transcription.completed", browser_source)
        self.assertNotIn("AMBIENT_CHUNK_MS", browser_source)
        self.assertNotIn("ambientRecorder", browser_source)

    def test_raw_ambient_speech_is_thought_only_until_the_agent_responds(self):
        interface_source = (
            ROOT / "apps" / "conscious_assistant" / "interface_flask.py"
        ).read_text()
        browser_source = (
            ROOT / "apps" / "conscious_assistant" / "templates" / "index.html"
        ).read_text()

        self.assertIn("'assistant-thoughts'", browser_source)
        self.assertIn("'Heard: ' + data.data", browser_source)
        self.assertIn('payload.get("heard", "")', interface_source)
        self.assertIn('"update_user_input", {"data": heard.strip()}', interface_source)

    def test_scene_observer_is_a_native_python_service(self):
        server_source = (ROOT / "apps" / "conscious_assistant" / "native_server.py").read_text()
        manifest_source = (ROOT / "registries" / "manifest.hocon").read_text()

        self.assertIn("SceneObserverService", server_source)
        self.assertIn("observer_service.start()", server_source)
        self.assertIn("observer_service.stop()", server_source)
        self.assertNotIn("scene_observer_agent", manifest_source)
        self.assertNotIn('"periodic"', manifest_source)

    def test_native_event_turns_have_bounded_execution(self):
        source = (ROOT / "registries" / "conscious_agent.hocon").read_text()

        self.assertIn('"max_steps": 12', source)
        self.assertIn('"max_execution_seconds": 60', source)

    def test_dispatch_agent_event_posts_a_minimal_event(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            awareness_file = str(Path(temporary_directory) / "awareness.json")
            with (
                patch.dict(
                    os.environ,
                    {"CONSCIOUS_NAVIGATION_AWARENESS_FILE": awareness_file},
                ),
                patch.object(agent_events, "_post_json") as post,
            ):
                self.assertTrue(agent_events.dispatch_agent_event("go home", source="user"))

        endpoint, payload = post.call_args.args[:2]
        self.assertIn("/conscious_agent/streaming_chat", endpoint)
        self.assertEqual(payload["user_message"]["text"], "user: go home")
        self.assertEqual(payload["chat_filter"]["chat_filter_type"], "MINIMAL")

    def test_navigation_awareness_is_added_to_later_minimal_user_turns(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            awareness_file = str(Path(temporary_directory) / "awareness.json")
            with (
                patch.dict(
                    os.environ,
                    {"CONSCIOUS_NAVIGATION_AWARENESS_FILE": awareness_file},
                ),
                patch.object(agent_events, "_post_json") as post,
            ):
                agent_events.dispatch_agent_event(
                    "I arrived at kitchen.",
                    source="navigation",
                )
                agent_events.dispatch_agent_event(
                    "where are you?",
                    source="user",
                )

        payload = post.call_args.args[1]
        self.assertEqual(
            payload["user_message"]["text"],
            "user: where are you?\n"
            "system: Current navigation awareness (authoritative): "
            "I arrived at kitchen.",
        )

    def test_stale_navigation_awareness_is_not_added_to_user_turns(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            awareness_file = Path(temporary_directory) / "awareness.json"
            awareness_file.write_text(
                '{"text": "I arrived at kitchen.", "updated_at": 1}',
                encoding="utf-8",
            )
            with (
                patch.dict(
                    os.environ,
                    {
                        "CONSCIOUS_NAVIGATION_AWARENESS_FILE": str(awareness_file),
                        "CONSCIOUS_NAVIGATION_AWARENESS_MAX_AGE_SECONDS": "1",
                    },
                ),
                patch.object(agent_events, "_post_json") as post,
            ):
                agent_events.dispatch_agent_event("where are you?", source="user")

        payload = post.call_args.args[1]
        self.assertEqual(payload["user_message"]["text"], "user: where are you?")

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
