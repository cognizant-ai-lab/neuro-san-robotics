import asyncio
import io
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

    def test_auto_prefers_onboard_device(self):
        with patch.object(tts_go2, "ONBOARD_ALSA_DEVICE", "plughw:CARD=APE,DEV=0"):
            self.assertEqual(
                tts_go2._resolve_alsa_device("auto"),
                "plughw:CARD=APE,DEV=0",
            )

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


if __name__ == "__main__":
    unittest.main()
