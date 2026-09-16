"""Offline speech-to-text fallback and how the transcribe route chooses."""

import io
import os
import threading
import unittest
from unittest.mock import patch

from apps.conscious_assistant import interface_flask
from coded_tools.unigo2 import local_stt


STT_VARS = ("GO2_STT_ENGINE", "GO2_STT_MODEL", "GO2_STT_DEVICE", "GO2_STT_LANGUAGE")


def stt_env(**overrides):
    """An environment with every STT variable under test control."""
    env = {name: "" for name in STT_VARS}
    env.update(overrides)
    return env


def upload():
    """A recording large enough to clear the route's minimum-size guard."""
    return {"audio": (io.BytesIO(b"\x00" * 4000), "speech.webm")}


class ConfigurationTests(unittest.TestCase):
    def test_auto_is_the_default(self):
        with patch.dict(os.environ, stt_env()):
            self.assertEqual(local_stt.engine(), "auto")

    def test_an_unrecognised_value_falls_back_to_auto(self):
        with patch.dict(os.environ, stt_env(GO2_STT_ENGINE="wat")):
            self.assertEqual(local_stt.engine(), "auto")

    def test_engine_choices_are_honoured(self):
        for value in ("auto", "openai", "local"):
            with patch.dict(os.environ, stt_env(GO2_STT_ENGINE=value)):
                self.assertEqual(local_stt.engine(), value)

    def test_it_stays_off_the_gpu_unless_asked(self):
        """The Orin is already carrying YOLO and deepface."""
        with patch.dict(os.environ, stt_env()):
            self.assertEqual(local_stt.device(), "cpu")
        with patch.dict(os.environ, stt_env(GO2_STT_DEVICE="cuda")):
            self.assertEqual(local_stt.device(), "cuda")

    def test_model_size_defaults_to_base(self):
        with patch.dict(os.environ, stt_env()):
            self.assertEqual(local_stt.model_size(), "base")
        with patch.dict(os.environ, stt_env(GO2_STT_MODEL="small")):
            self.assertEqual(local_stt.model_size(), "small")


class RouteSelectionTests(unittest.TestCase):
    """Which recogniser /api/transcribe reaches for, and when."""

    def post(self):
        with interface_flask.app.test_client() as client:
            return client.post("/api/transcribe", data=upload(),
                               content_type="multipart/form-data")

    def test_local_mode_never_calls_the_hosted_api(self):
        """A region with no hosted recogniser must not wait on a doomed call."""
        env = stt_env(GO2_STT_ENGINE="local")
        env["OPENAI_API_KEY"] = ""
        with patch.dict(os.environ, env):
            with patch.object(local_stt, "transcribe", return_value="hello there") as local:
                with patch.object(interface_flask.openai_provider, "create_client") as hosted:
                    response = self.post()
        self.assertEqual(response.get_json(), {"text": "hello there"})
        local.assert_called_once()
        hosted.assert_not_called()

    def test_local_mode_needs_no_hosted_credentials(self):
        """The 503 credential gate must not turn a local-only robot away."""
        env = stt_env(GO2_STT_ENGINE="local")
        env["OPENAI_API_KEY"] = ""
        env["AZURE_OPENAI_API_KEY"] = ""
        with patch.dict(os.environ, env):
            with patch.object(local_stt, "transcribe", return_value="no key needed"):
                response = self.post()
        self.assertEqual(response.status_code, 200)

    def test_auto_prefers_the_hosted_recogniser(self):
        env = stt_env(GO2_STT_ENGINE="auto")
        env["OPENAI_API_KEY"] = "sk-test"
        with patch.dict(os.environ, env):
            with patch.object(local_stt, "transcribe") as local:
                with patch.object(interface_flask.openai_provider, "create_client") as factory:
                    factory.return_value.audio.transcriptions.create.return_value.text = "hosted"
                    response = self.post()
        self.assertEqual(response.get_json(), {"text": "hosted"})
        local.assert_not_called()

    def test_auto_falls_back_when_the_hosted_call_fails(self):
        env = stt_env(GO2_STT_ENGINE="auto")
        env["OPENAI_API_KEY"] = "sk-test"
        with patch.dict(os.environ, env):
            with patch.object(local_stt, "available", return_value=True):
                with patch.object(local_stt, "transcribe", return_value="rescued") as local:
                    with patch.object(interface_flask.openai_provider, "create_client",
                                      side_effect=RuntimeError("region has no whisper")):
                        response = self.post()
        self.assertEqual(response.get_json(), {"text": "rescued"})
        local.assert_called_once()

    def test_openai_mode_does_not_fall_back(self):
        """Naming the hosted engine means you want to hear when it breaks."""
        env = stt_env(GO2_STT_ENGINE="openai")
        env["OPENAI_API_KEY"] = "sk-test"
        with patch.dict(os.environ, env):
            with patch.object(local_stt, "available", return_value=True):
                with patch.object(local_stt, "transcribe") as local:
                    with patch.object(interface_flask.openai_provider, "create_client",
                                      side_effect=RuntimeError("upstream down")):
                        response = self.post()
        self.assertEqual(response.status_code, 500)
        local.assert_not_called()

    def test_an_existing_robot_behaves_exactly_as_before(self):
        """
        No new variables set and no weights cached: the hosted path runs and
        nothing else happens. This is the upgrade case for robots already in
        the field, which must not change behaviour.
        """
        env = stt_env()
        env["OPENAI_API_KEY"] = "sk-test"
        with patch.dict(os.environ, env):
            with patch.object(local_stt, "is_model_cached", return_value=False):
                with patch.object(local_stt, "transcribe") as local:
                    with patch.object(interface_flask.openai_provider,
                                      "create_client") as factory:
                        factory.return_value.audio.transcriptions.create.return_value.text = "as before"
                        response = self.post()
        self.assertEqual(response.get_json(), {"text": "as before"})
        local.assert_not_called()

    def test_auto_never_downloads_weights_mid_request(self):
        """
        A hosted failure on a robot that has never used the fallback must not
        stall the request behind a few hundred megabytes of model download.
        """
        env = stt_env(GO2_STT_ENGINE="auto")
        env["OPENAI_API_KEY"] = "sk-test"
        with patch.dict(os.environ, env):
            with patch.object(local_stt, "is_model_cached", return_value=False):
                with patch.object(local_stt, "transcribe") as local:
                    with patch.object(interface_flask.openai_provider, "create_client",
                                      side_effect=RuntimeError("upstream down")):
                        response = self.post()
        self.assertEqual(response.status_code, 500)
        local.assert_not_called()

    def test_explicit_local_may_download_because_it_was_asked_for(self):
        env = stt_env(GO2_STT_ENGINE="local")
        with patch.dict(os.environ, env):
            with patch.object(local_stt, "is_model_cached", return_value=False):
                with patch.object(local_stt, "transcribe", return_value="downloaded") as local:
                    response = self.post()
        self.assertEqual(response.get_json(), {"text": "downloaded"})
        local.assert_called_once()

    def test_a_hosted_failure_with_no_local_model_still_reports_cleanly(self):
        env = stt_env(GO2_STT_ENGINE="auto")
        env["OPENAI_API_KEY"] = "sk-test"
        with patch.dict(os.environ, env):
            with patch.object(local_stt, "available", return_value=False):
                with patch.object(interface_flask.openai_provider, "create_client",
                                  side_effect=RuntimeError("upstream down")):
                    response = self.post()
        self.assertEqual(response.status_code, 500)
        # Still no traceback on the public surface, per the earlier hardening.
        self.assertEqual(response.get_json(), {"error": "Transcription failed"})


class AmbientModeTests(unittest.TestCase):
    """What the browser is told to do before it opens a microphone."""

    def mode(self, hosted_key, **env):
        with patch.dict(os.environ, stt_env(**env)):
            return local_stt.ambient_mode(hosted_key)

    def test_a_hosted_key_means_realtime(self):
        self.assertEqual(self.mode(True), "realtime")

    def test_local_engine_wins_over_a_hosted_key(self):
        """Chosen deliberately, so it is not second-guessed."""
        with patch.object(local_stt, "_importable", return_value=True):
            self.assertEqual(self.mode(True, GO2_STT_ENGINE="local"), "local")

    def test_local_engine_without_the_package_is_unavailable(self):
        with patch.object(local_stt, "_importable", return_value=False):
            self.assertEqual(self.mode(True, GO2_STT_ENGINE="local"), "unavailable")

    def test_openai_engine_never_falls_back(self):
        with patch.object(local_stt, "available", return_value=True):
            self.assertEqual(self.mode(False, GO2_STT_ENGINE="openai"), "unavailable")

    def test_auto_falls_back_to_local_when_there_is_no_hosted_key(self):
        with patch.object(local_stt, "available", return_value=True):
            self.assertEqual(self.mode(False, GO2_STT_ENGINE="auto"), "local")

    def test_auto_with_nothing_configured_is_unavailable(self):
        with patch.object(local_stt, "available", return_value=False):
            self.assertEqual(self.mode(False, GO2_STT_ENGINE="auto"), "unavailable")

    def test_the_route_reports_the_mode(self):
        env = stt_env()
        env["OPENAI_API_KEY"] = "sk-test"
        with patch.dict(os.environ, env):
            with interface_flask.app.test_client() as client:
                response = client.get("/api/speech-config")
        self.assertEqual(response.get_json(), {"ambient_mode": "realtime"})


class AmbientBrowserTests(unittest.TestCase):
    """The browser side of local ambient listening."""

    @classmethod
    def setUpClass(cls):
        from pathlib import Path

        root = Path(__file__).resolve().parents[1]
        cls.browser = (
            root / "apps" / "conscious_assistant" / "templates" / "index.html"
        ).read_text(encoding="utf-8")

    def test_it_asks_the_server_before_opening_a_session(self):
        self.assertIn("/api/speech-config", self.browser)
        self.assertIn("ambient_mode", self.browser)

    def test_local_utterances_reuse_the_existing_transcribe_route(self):
        self.assertIn("transcribeLocalAmbient", self.browser)
        self.assertIn("ambient.webm", self.browser)

    def test_local_transcripts_reach_the_agent_the_same_way(self):
        """Same socket event as the hosted path, so the server is unchanged."""
        self.assertIn("socket.emit('ambient_transcript'", self.browser)

    def test_it_segments_on_silence_rather_than_a_clock(self):
        """
        Ambient listening used to post a recording every five seconds and was
        changed away from that because it was slow and cut words in half. The
        fallback must not reintroduce a fixed interval.
        """
        self.assertIn("LOCAL_AMBIENT_SILENCE_MS", self.browser)
        self.assertNotIn("AMBIENT_CHUNK_MS", self.browser)
        self.assertNotIn("ambientRecorder", self.browser)

    def test_stopping_ambient_also_stops_the_local_loop(self):
        """Otherwise the mic and the poll timer outlive the session."""
        teardown = self.browser.split("function teardownAmbientTransport()")[1]
        self.assertIn("stopLocalAmbient()", teardown.split("\n        }")[0])


if __name__ == "__main__":
    unittest.main()


class SelfEchoTimingTests(unittest.TestCase):
    """
    The robot must not answer its own voice in ambient mode.

    The self-echo filter only considers audio heard while the robot was
    speaking, plus a short tail. The hosted recogniser returns while the audio
    is still streaming, so arrival time is a fair proxy for when it was heard.
    A recogniser running on the robot takes seconds, so by the time its
    transcript arrives the robot has stopped and arrival time says nothing --
    which let the robot's own words come back as a person and start a loop.
    """

    import time as _time

    def setUp(self):
        self.spoken = "Woof! Today I am patrolling the AI Studio."
        with interface_flask._speech_state_lock:
            interface_flask._active_speech_text = self.spoken
            interface_flask._speech_active = False
            interface_flask._active_speech_ended_at = self._time.monotonic()

    def finished_speaking(self, seconds_ago):
        with interface_flask._speech_state_lock:
            interface_flask._speech_active = False
            interface_flask._active_speech_ended_at = (
                self._time.monotonic() - seconds_ago)

    def still_speaking(self):
        with interface_flask._speech_state_lock:
            interface_flask._speech_active = True
            interface_flask._active_speech_ended_at = self._time.monotonic()

    def test_a_slow_local_transcript_of_its_own_voice_is_still_echo(self):
        """The case that made the robot talk to itself."""
        self.finished_speaking(4.0)
        heard = "Woof today I am patrolling the AI studio"
        self.assertFalse(
            interface_flask.is_self_echo(heard),
            "judged by arrival this looks like a person -- the old behaviour",
        )
        self.assertTrue(
            interface_flask.is_self_echo(heard, captured_ago=3.8),
            "told when it was heard, it must be recognised as echo",
        )

    def test_a_person_speaking_later_is_not_suppressed(self):
        """Being told the audio is old must not mute real conversation."""
        self.finished_speaking(4.0)
        self.assertFalse(
            interface_flask.is_self_echo("what time is the standup", captured_ago=0.2)
        )

    def test_barge_in_over_the_robot_still_gets_through(self):
        self.still_speaking()
        self.assertFalse(
            interface_flask.is_self_echo(
                "no stop that is not what I asked you to do", captured_ago=0.5)
        )

    def test_the_hosted_path_is_unchanged(self):
        """It sends no age, so the default keeps the previous behaviour."""
        self.finished_speaking(0.2)
        heard = "Woof today I am patrolling the AI studio"
        self.assertEqual(
            interface_flask.is_self_echo(heard),
            interface_flask.is_self_echo(heard, captured_ago=0.0),
        )

    def test_a_nonsense_age_does_not_break_the_handler(self):
        for value in ("abc", None, -500, [1]):
            with patch.object(interface_flask, "is_self_echo", return_value=True) as echo:
                interface_flask.handle_ambient_transcript(
                    {"data": "hello", "captured_ms_ago": value})
            self.assertGreaterEqual(echo.call_args.args[1], 0.0, repr(value))

    def test_the_browser_sends_the_age(self):
        from pathlib import Path

        browser = (Path(__file__).resolve().parents[1] / "apps" /
                   "conscious_assistant" / "templates" / "index.html").read_text()
        self.assertIn("captured_ms_ago", browser)
        self.assertIn("state.lastVoiceAt", browser)


class LocalAmbientDoesNotListenWhileSpeakingTests(unittest.TestCase):
    """
    On the local path the robot waits its turn instead of filtering echo.

    Transcribing on the robot takes seconds, so its own voice comes back long
    after it stopped and looks like somebody spoke. Not listening while it
    talks removes the problem rather than detecting it, and saves running
    Whisper over the robot's own speech. Interrupting mid-sentence is given up
    in exchange, which only applies here: the hosted path still barges in.
    """

    @classmethod
    def setUpClass(cls):
        from pathlib import Path

        root = Path(__file__).resolve().parents[1]
        cls.browser = (root / "apps" / "conscious_assistant"
                       / "templates" / "index.html").read_text(encoding="utf-8")
        start = cls.browser.index("function startLocalAmbient")
        cls.local = cls.browser[start:cls.browser.index(
            "async function transcribeLocalAmbient")]

    def test_the_local_loop_stops_capturing_while_the_robot_speaks(self):
        self.assertIn("isSpeaking", self.local)
        self.assertIn("state.mutedUntil", self.local)

    def test_it_also_stands_down_while_the_mic_button_is_in_use(self):
        """
        Both paths post to the same route and the robot decodes one clip at a
        time, so ambient work queued during a press is work the person waits
        behind. It would also transcribe the same speech twice.
        """
        self.assertIn("isRecording", self.local)
        self.assertIn("isTranscribing", self.local)

    def test_an_utterance_in_progress_is_thrown_away_not_transcribed(self):
        """Otherwise its tail carries the robot's voice into the transcript."""
        self.assertIn("state.discard = true", self.local)
        self.assertIn("discarded", self.browser)

    def test_it_keeps_ignoring_the_room_briefly_after_speech(self):
        """The speaker rings out and the room reverberates past the last word."""
        self.assertIn("LOCAL_AMBIENT_SPEAK_TAIL_MS", self.browser)

    def test_the_hosted_path_still_listens_while_speaking(self):
        """Barge-in is the whole reason the hosted mic stays open."""
        webrtc = self.browser[self.browser.index("ambientPeerConnection = new RTCPeerConnection"):]
        webrtc = webrtc[:webrtc.index("setRemoteDescription")]
        self.assertNotIn("isSpeaking", webrtc)

    def test_the_gate_is_not_applied_globally(self):
        """It belongs to the local loop, not to ambient listening as a whole."""
        self.assertEqual(
            self.local.count("isSpeaking || isRecording || isTranscribing"), 1)


class WarmUpTests(unittest.TestCase):
    """
    Loading the model at start-up, and only where it is actually used.

    On the robot's CPU the load takes long enough that paying for it inside the
    first request looks like a hang: the button sits on "transcribing" with
    nothing coming back. Doing it at start-up fixes that, but it must stay
    invisible to everyone on hosted speech, who would otherwise gain a model
    load and a few hundred megabytes of resident memory they never asked for.
    """

    def warm(self, **overrides):
        """Run the warm-up and report whether it started a thread."""
        with patch.dict(os.environ, stt_env(**overrides)):
            with patch.object(local_stt, "_importable", return_value=True):
                with patch.object(local_stt.threading, "Thread") as thread:
                    local_stt.warm_in_background()
        return thread.called

    def test_an_existing_hosted_robot_loads_nothing(self):
        """Nothing configured is the upgrade case: it must not change."""
        self.assertFalse(self.warm())

    def test_the_hosted_engine_loads_nothing(self):
        self.assertFalse(self.warm(GO2_STT_ENGINE="openai"))

    def test_auto_loads_nothing_because_it_prefers_the_hosted_call(self):
        """
        auto only reaches Whisper if a hosted call fails, which may never
        happen, so warming it up front would be cost for nothing.
        """
        self.assertFalse(self.warm(GO2_STT_ENGINE="auto"))

    def test_the_local_engine_warms_up(self):
        self.assertTrue(self.warm(GO2_STT_ENGINE="local"))

    def test_it_stays_quiet_when_the_package_is_missing(self):
        with patch.dict(os.environ, stt_env(GO2_STT_ENGINE="local")):
            with patch.object(local_stt, "_importable", return_value=False):
                with patch.object(local_stt.threading, "Thread") as thread:
                    local_stt.warm_in_background()
        self.assertFalse(thread.called)

    def test_a_failed_warm_up_does_not_stop_the_app(self):
        """A start-up nicety must never be the reason the robot will not boot."""
        with patch.dict(os.environ, stt_env(GO2_STT_ENGINE="local")):
            with patch.object(local_stt, "_importable", return_value=True):
                with patch.object(local_stt, "model",
                                  side_effect=RuntimeError("no weights")):
                    # assertLogs both proves it was reported and keeps the
                    # traceback out of the test run's output.
                    with self.assertLogs(level="ERROR") as logged:
                        local_stt.warm_in_background()
                        for thread in threading.enumerate():
                            if thread.name == "whisper-warmup":
                                thread.join(timeout=5)
        self.assertIn("Could not warm", "".join(logged.output))

    def test_it_runs_as_a_daemon_so_it_cannot_hold_up_shutdown(self):
        with patch.dict(os.environ, stt_env(GO2_STT_ENGINE="local")):
            with patch.object(local_stt, "_importable", return_value=True):
                with patch.object(local_stt.threading, "Thread") as thread:
                    local_stt.warm_in_background()
        self.assertTrue(thread.call_args.kwargs["daemon"])
