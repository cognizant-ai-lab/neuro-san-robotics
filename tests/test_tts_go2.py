import asyncio
import io
import subprocess
import sys
import threading
import time
import types
import unittest
import wave
from unittest.mock import patch

from coded_tools.unigo2 import tts_go2


class Go2TtsFallbackTests(unittest.TestCase):
    def test_onboard_device_resolves_to_ape_output(self):
        with patch.object(tts_go2, "ONBOARD_ALSA_DEVICE", "plughw:CARD=APE,DEV=0"):
            self.assertEqual(
                tts_go2._resolve_alsa_device("onboard"),
                "plughw:CARD=APE,DEV=0",
            )

    def test_auto_uses_usb_detection(self):
        with patch.object(tts_go2, "_detect_usb_audio_device", return_value="plughw:4,0"):
            self.assertEqual(tts_go2._resolve_alsa_device("auto"), "plughw:4,0")

    def test_usb_device_still_supports_auto_detection(self):
        with patch.object(tts_go2, "_detect_usb_audio_device", return_value="plughw:4,0"):
            self.assertEqual(tts_go2._resolve_alsa_device("usb"), "plughw:4,0")

    def test_explicit_alsa_device_is_unchanged(self):
        self.assertEqual(tts_go2._resolve_alsa_device("plughw:7,1"), "plughw:7,1")

    def test_onboard_pcm_is_upsampled_to_48khz(self):
        source = b"\x01\x00\x02\x00"
        prepared = tts_go2._prepare_openai_pcm_chunk(
            source,
            gain=1.0,
            device="plughw:CARD=APE,DEV=0",
        )
        self.assertEqual(prepared, b"\x01\x00\x01\x00\x02\x00\x02\x00")

    def test_usb_pcm_keeps_openai_sample_rate(self):
        source = b"\x01\x00\x02\x00"
        prepared = tts_go2._prepare_openai_pcm_chunk(
            source,
            gain=1.0,
            device="plughw:4,0",
        )
        self.assertEqual(prepared, source)

    def test_pcm_is_wrapped_as_48khz_mono_wav(self):
        source = b"\x01\x00\x02\x00"
        wav_data = tts_go2._pcm_to_wav_bytes(source, 48_000)
        with wave.open(io.BytesIO(wav_data), "rb") as wav_file:
            self.assertEqual(wav_file.getnchannels(), 1)
            self.assertEqual(wav_file.getsampwidth(), 2)
            self.assertEqual(wav_file.getframerate(), 48_000)
            self.assertEqual(wav_file.readframes(2), source)

    def test_say_non_chunked_falls_back_to_offline_tts_in_auto_mode(self):
        with (
            patch.object(tts_go2, "_should_use_openai", return_value=True),
            patch.object(tts_go2, "_should_runtime_fallback_to_offline_tts", return_value=True),
            patch.object(tts_go2, "_openai_say_streaming", side_effect=RuntimeError("timeout")),
            patch.object(tts_go2, "_say_single_chunk") as offline_say,
            patch.object(tts_go2.fcntl, "flock"),
        ):
            tts_go2.say("hello from cail-e", chunked=False)

        offline_say.assert_called_once()

    def test_say_non_chunked_raises_when_openai_is_explicit(self):
        with (
            patch.object(tts_go2, "_should_use_openai", return_value=True),
            patch.object(tts_go2, "_should_runtime_fallback_to_offline_tts", return_value=False),
            patch.object(tts_go2, "_openai_say_streaming", side_effect=RuntimeError("timeout")),
            patch.object(tts_go2, "_say_single_chunk") as offline_say,
            patch.object(tts_go2.fcntl, "flock"),
        ):
            with self.assertRaises(RuntimeError):
                tts_go2.say("hello from cail-e", chunked=False)

        offline_say.assert_not_called()

    def test_say_streaming_falls_back_to_chunked_offline_tts_in_auto_mode(self):
        with (
            patch.object(tts_go2, "_should_use_openai", return_value=True),
            patch.object(tts_go2, "_should_runtime_fallback_to_offline_tts", return_value=True),
            patch.object(tts_go2, "_openai_say_streaming", side_effect=RuntimeError("timeout")),
            patch.object(tts_go2, "_set_alsa_volume"),
            patch.object(tts_go2.fcntl, "flock"),
            patch.object(tts_go2.platform, "system", return_value="Linux"),
            patch.object(tts_go2, "_is_piper_available", return_value=False),
            patch.object(tts_go2, "split_into_chunks", return_value=["hello", "world"]),
            patch.object(tts_go2, "_say_single_chunk") as offline_say,
        ):
            tts_go2.say_streaming("hello world", max_chunk_size=5)

        self.assertEqual(offline_say.call_count, 2)

    def test_say_async_falls_back_to_offline_tts_in_auto_mode(self):
        with (
            patch.object(tts_go2, "_should_use_openai", return_value=True),
            patch.object(tts_go2, "_should_runtime_fallback_to_offline_tts", return_value=True),
            patch.object(tts_go2, "_openai_say_streaming_async", side_effect=RuntimeError("timeout")),
            patch.object(tts_go2, "_say_single_chunk") as offline_say,
        ):
            asyncio.run(tts_go2.say_async("hello from cail-e"))

        offline_say.assert_called_once()



class PlaybackCancellationTests(unittest.TestCase):
    """Cover the barge-in primitives that let speech be cut off mid-utterance."""

    def setUp(self):
        # Each test needs a controller nobody else has cancelled.
        self._controller = tts_go2._PlaybackController()
        patcher = patch.object(tts_go2, "_PLAYBACK", self._controller)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_cancel_bumps_the_generation_even_with_nothing_playing(self):
        before = self._controller.generation
        self.assertFalse(tts_go2.stop_speaking())
        self.assertGreater(self._controller.generation, before)

    def test_utterance_superseded_before_playback_is_discarded(self):
        generation = self._controller.generation
        tts_go2.stop_speaking()
        with self.assertRaises(tts_go2.SpeechInterrupted):
            tts_go2._raise_if_cancelled(generation)

    def test_a_fresh_utterance_survives_an_earlier_cancel(self):
        tts_go2.stop_speaking()
        generation = self._controller.generation
        tts_go2._raise_if_cancelled(generation)

    def test_cancel_terminates_a_registered_process(self):
        proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
        self.addCleanup(proc.kill)
        self._controller.register(proc)

        self.assertTrue(self._controller.is_active())
        self.assertTrue(tts_go2.stop_speaking())
        self.assertIsNotNone(proc.poll())
        self.assertFalse(self._controller.is_active())

    def test_run_playback_raises_when_cancelled_mid_command(self):
        def cancel_soon():
            time.sleep(0.2)
            tts_go2.stop_speaking()

        canceller = threading.Thread(target=cancel_soon, daemon=True)
        canceller.start()
        self.addCleanup(canceller.join)

        with self.assertRaises(tts_go2.SpeechInterrupted):
            tts_go2._run_playback([sys.executable, "-c", "import time; time.sleep(30)"])

    def test_run_playback_returns_normally_when_not_cancelled(self):
        result = tts_go2._run_playback([sys.executable, "-c", "pass"])
        self.assertEqual(result.returncode, 0)

    def test_duck_clamps_a_later_volume_change(self):
        applied = []
        with patch.object(tts_go2, "_apply_mixer_volume", applied.append):
            tts_go2.duck_playback(True)
            # A new utterance resetting the mixer must not undo the duck.
            tts_go2._set_alsa_volume(100)
            self.assertTrue(all(level <= tts_go2.DUCK_VOLUME_PERCENT for level in applied))

            tts_go2.duck_playback(False)
            self.assertEqual(applied[-1], tts_go2.DEFAULT_VOLUME_PERCENT)

    def test_cancel_clears_the_duck_and_restores_the_mixer(self):
        applied = []
        with patch.object(tts_go2, "_apply_mixer_volume", applied.append):
            tts_go2.duck_playback(True)
            self.assertTrue(self._controller.ducked)

            tts_go2.stop_speaking()

        self.assertFalse(self._controller.ducked)
        # Leaving the mixer down would mute the next utterance too.
        self.assertEqual(applied[-1], tts_go2.DEFAULT_VOLUME_PERCENT)

    def test_cancel_without_a_duck_leaves_the_mixer_alone(self):
        applied = []
        with patch.object(tts_go2, "_apply_mixer_volume", applied.append):
            tts_go2.stop_speaking()

        self.assertEqual(applied, [])

if __name__ == "__main__":
    unittest.main()


class PiperSampleRateTests(unittest.TestCase):
    """
    aplay plays raw PCM at whatever rate it is told. Piper's low-tier voices
    are 16 kHz where the medium and high tiers are 22.05 kHz, so the rate has
    to follow the installed voice's config rather than a constant.
    """

    def setUp(self):
        import json
        import os
        import tempfile

        handle = tempfile.NamedTemporaryFile("w", suffix=".onnx.json", delete=False)
        json.dump({"audio": {"sample_rate": 16000, "quality": "low"}}, handle)
        handle.close()
        self.config = handle.name
        self.addCleanup(lambda: os.unlink(self.config))

    @staticmethod
    def rate_passed_to_aplay(cmd):
        return cmd[cmd.index("-r") + 1]

    def test_cached_audio_plays_at_the_voice_sample_rate(self):
        with (
            patch.object(tts_go2, "PIPER_CONFIG", self.config),
            patch.object(tts_go2, "_run_playback") as run_playback,
        ):
            tts_go2._play_audio_bytes(b"\x00\x00", alsa_device="plughw:1,0")

        cmd = run_playback.call_args.args[0]
        self.assertEqual(cmd[0], "aplay")
        self.assertEqual(self.rate_passed_to_aplay(cmd), "16000")

    def test_direct_piper_speech_plays_at_the_voice_sample_rate(self):
        aplay = FakeAplay()
        with (
            patch.object(tts_go2, "PIPER_CONFIG", self.config),
            patch.object(tts_go2, "_PIPER_VOICE", FakeVoice([b"\x00\x00"])),
            patch.object(tts_go2, "_set_alsa_volume"),
            patch.object(tts_go2.subprocess, "Popen", return_value=aplay) as popen,
        ):
            tts_go2._linux_say_via_piper("hello", alsa_device="plughw:1,0")

        cmd = popen.call_args.args[0]
        self.assertEqual(cmd[0], "aplay")
        self.assertEqual(self.rate_passed_to_aplay(cmd), "16000")


class FakeAplay:
    """aplay reading raw PCM from a pipe; records what was flushed to it and when."""

    class _Stdin:
        def __init__(self):
            self.pending, self.flushed, self.closed = b"", b"", False

        def write(self, data):
            self.pending += data

        def flush(self):
            self.flushed, self.pending = self.flushed + self.pending, b""

        def close(self):
            self.flush()
            self.closed = True

    def __init__(self, returncode=0):
        self.stdin = self._Stdin()
        self.returncode = None
        self._exit_code = returncode
        self.terminated = False

    def poll(self):
        return self.returncode

    def wait(self, timeout=None):
        self.returncode = self._exit_code
        return self.returncode

    def communicate(self, input=None, timeout=None):
        self.wait()
        return None, None

    def terminate(self):
        self.terminated = True
        self.returncode = -15

    kill = terminate


class FakeVoice:
    """A loaded PiperVoice: yields one audio chunk per sentence, optionally running a hook first."""

    def __init__(self, chunks, before_chunk=None):
        self.chunks = chunks
        self.before_chunk = before_chunk or (lambda index: None)
        self.synthesised = 0

    def synthesize(self, text):
        for index, chunk in enumerate(self.chunks):
            self.before_chunk(index)
            self.synthesised += 1
            yield types.SimpleNamespace(audio_int16_bytes=chunk, phonemes=list(chunk.decode()))


class PiperStaysLoadedTests(unittest.TestCase):
    """
    Spawning the piper CLI per utterance reloaded the voice every time: 2.7 s of
    model load (plus process start) on the robot before the first sound, on
    every reply (#5). The voice is loaded once and kept; each sentence goes to
    aplay the moment it is synthesised.
    """

    def setUp(self):
        self.loads = 0
        self.voice = FakeVoice([b"one", b"two"])
        test = self

        class PiperVoice:  # stands in for piper.PiperVoice
            @staticmethod
            def load(model_path, config_path=None):
                test.loads += 1
                return test.voice

        fake_piper = types.ModuleType("piper")
        fake_piper.PiperVoice = PiperVoice
        for patcher in (
            patch.dict(sys.modules, {"piper": fake_piper}),
            patch.object(tts_go2, "_PIPER_VOICE", None),
            patch.object(tts_go2, "_piper_sample_rate", return_value=22050),
            patch.object(tts_go2, "_set_alsa_volume"),
            patch.object(tts_go2.os.path, "isfile", return_value=True),
        ):
            patcher.start()
            self.addCleanup(patcher.stop)
        self.aplays = []

    def speak(self, text="Hello there. How are you?", aplay_exit=0):
        def popen(cmd, **kwargs):
            self.assertEqual(cmd[0], "aplay", "only aplay may be spawned; piper runs in-process")
            self.assertIs(kwargs.get("stdin"), subprocess.PIPE)
            self.aplays.append(FakeAplay(aplay_exit))
            return self.aplays[-1]

        with patch.object(tts_go2.subprocess, "Popen", side_effect=popen):
            tts_go2._linux_say_via_piper(text, alsa_device="plughw:1,0")

    def test_the_voice_is_loaded_once_not_per_utterance(self):
        self.speak()
        self.speak()
        self.assertEqual(self.loads, 1)

    def test_each_sentence_reaches_aplay_before_the_next_is_synthesised(self):
        heard_before = {}
        self.voice.before_chunk = lambda i: heard_before.setdefault(i, self.aplays[0].stdin.flushed)
        self.speak()
        self.assertEqual(heard_before[1], b"one")
        self.assertEqual(self.aplays[0].stdin.flushed, b"onetwo")
        self.assertTrue(self.aplays[0].stdin.closed)

    def test_a_barge_in_stops_synthesis_and_is_reported_as_an_interruption(self):
        self.voice = FakeVoice([b"one", b"two", b"three"],
                               before_chunk=lambda i: i == 1 and tts_go2._PLAYBACK.cancel())
        with self.assertRaises(tts_go2.SpeechInterrupted):
            self.speak()
        self.assertTrue(self.aplays[0].terminated)
        self.assertLess(self.voice.synthesised, 3, "kept synthesising after the user interrupted")

    def test_an_aplay_failure_surfaces(self):
        with self.assertRaises(subprocess.CalledProcessError):
            self.speak(aplay_exit=1)

    def test_listeners_hear_when_the_first_audio_reaches_aplay(self):
        heard = []
        tts_go2.set_audio_started_listener(lambda **_: heard.append(self.aplays[0].stdin.flushed))
        self.addCleanup(tts_go2.set_audio_started_listener, None)
        self.speak()
        self.assertEqual(heard, [b""], "fire once, right before the first sentence is written")

    def test_listeners_hear_how_long_the_first_sentence_was(self):
        heard = []
        tts_go2.set_audio_started_listener(lambda **details: heard.append(details))
        self.addCleanup(tts_go2.set_audio_started_listener, None)
        self.speak()
        self.assertEqual(heard, [{"first_sentence_phonemes": 3}], "b'one' -> 3 phonemes")


class AudioStartedListenerTests(unittest.TestCase):
    def test_subprocess_engines_report_audio_when_playback_starts(self):
        heard = []
        tts_go2.set_audio_started_listener(lambda: heard.append(True))
        self.addCleanup(tts_go2.set_audio_started_listener, None)
        with patch.object(tts_go2.subprocess, "Popen", return_value=FakeAplay()):
            tts_go2._run_playback(["espeak-ng", "hi"])
        self.assertEqual(heard, [True])

    def test_a_failing_listener_does_not_stop_speech(self):
        def broken():
            raise RuntimeError("socket gone")
        tts_go2.set_audio_started_listener(broken)
        self.addCleanup(tts_go2.set_audio_started_listener, None)
        with patch.object(tts_go2.subprocess, "Popen", return_value=FakeAplay()):
            tts_go2._run_playback(["espeak-ng", "hi"])
