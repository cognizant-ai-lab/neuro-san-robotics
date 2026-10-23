import importlib.util
import io
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

REPO = Path(__file__).resolve().parents[1]


def load_script():
    spec = importlib.util.spec_from_file_location(
        "tts_latency_check", REPO / "scripts" / "tts_latency_check.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class StarvationAnalysisTests(unittest.TestCase):
    """
    The aplay wrapper logs when each block of PCM arrived. Playback runs at a
    fixed rate from the first block, so audio arriving after the speaker has
    already played everything it had is a gap the listener hears as a stutter.
    """

    def setUp(self):
        self.check = load_script()
        self.rate = 16000
        self.second = 2 * self.rate  # bytes of 16-bit mono per second

    def test_first_audio_is_measured_from_the_start_mark(self):
        report = self.check.analyse(start=10.0, arrivals=[(10.8, self.second)], rate=self.rate)
        self.assertAlmostEqual(report.first_audio_s, 0.8)
        self.assertAlmostEqual(report.speech_s, 1.0)

    def test_audio_that_keeps_ahead_of_playback_has_no_gaps(self):
        # One second of audio at t=0, the next second arrives at t=0.5: fine.
        arrivals = [(0.0, self.second), (0.5, self.second), (1.2, self.second)]
        report = self.check.analyse(start=0.0, arrivals=arrivals, rate=self.rate)
        self.assertEqual(report.gaps, [])

    def test_audio_arriving_after_the_buffer_ran_dry_is_a_gap(self):
        # One second of audio at t=0 is exhausted at t=1.0; the next arrives at 1.5.
        arrivals = [(0.0, self.second), (1.5, self.second)]
        report = self.check.analyse(start=0.0, arrivals=arrivals, rate=self.rate)
        self.assertEqual(len(report.gaps), 1)
        position_s, gap_s = report.gaps[0]
        self.assertAlmostEqual(position_s, 1.0)
        self.assertAlmostEqual(gap_s, 0.5)

    def test_tiny_lateness_within_the_device_buffer_is_not_a_gap(self):
        arrivals = [(0.0, self.second), (1.01, self.second)]
        report = self.check.analyse(start=0.0, arrivals=arrivals, rate=self.rate)
        self.assertEqual(report.gaps, [])

    def test_no_audio_at_all_is_reported_not_crashed(self):
        report = self.check.analyse(start=0.0, arrivals=[], rate=self.rate)
        self.assertIsNone(report.first_audio_s)
        self.assertEqual(report.speech_s, 0.0)


class ProgressLineTests(unittest.TestCase):
    """
    While say() runs the script redraws one status line so the user can see
    that synthesis and playback are moving. Everything it shows comes from
    the same arrival trace the final report is built from.
    """

    def setUp(self):
        self.check = load_script()
        self.rate = 16000
        self.second = 2 * self.rate

    def test_before_first_audio_it_shows_the_wait_so_far(self):
        line = self.check.progress_line(now=10.8, start=10.0, arrivals=[], rate=self.rate,
                                        estimated_s=30.0)
        self.assertIn("waiting for first audio", line)
        self.assertIn("0.8 s", line)

    def test_after_first_audio_it_shows_how_much_has_played(self):
        arrivals = [(10.0, 5 * self.second)]
        line = self.check.progress_line(now=12.0, start=9.0, arrivals=arrivals, rate=self.rate,
                                        estimated_s=20.0)
        self.assertIn("played 2.0 s", line)
        self.assertIn("~20 s", line)

    def test_a_fed_speaker_is_reported_as_keeping_up(self):
        arrivals = [(10.0, 5 * self.second)]
        line = self.check.progress_line(now=12.0, start=9.0, arrivals=arrivals, rate=self.rate,
                                        estimated_s=20.0)
        self.assertIn("keeping up", line)

    def test_a_stalled_speaker_is_called_out(self):
        # One second of audio, exhausted at t=11; nothing more by t=12.
        arrivals = [(10.0, self.second)]
        line = self.check.progress_line(now=12.0, start=9.0, arrivals=arrivals, rate=self.rate,
                                        estimated_s=20.0)
        self.assertIn("STALLED", line)

    def test_the_bar_fills_with_played_time_and_never_overflows(self):
        arrivals = [(10.0, 40 * self.second)]
        half = self.check.progress_line(now=20.0, start=10.0, arrivals=arrivals, rate=self.rate,
                                        estimated_s=20.0)
        over = self.check.progress_line(now=45.0, start=10.0, arrivals=arrivals, rate=self.rate,
                                        estimated_s=20.0)
        self.assertEqual(half.count(self.check.BAR_FULL), self.check.BAR_WIDTH // 2)
        self.assertEqual(over.count(self.check.BAR_FULL), self.check.BAR_WIDTH)
        self.assertNotIn("100%", over)  # the estimate was wrong, not the audio

    def test_estimate_comes_from_the_word_count(self):
        self.assertAlmostEqual(self.check.estimate_speech_s("one two three four five six seven"),
                               7 / self.check.WORDS_PER_SECOND)


class ProgressThreadTests(unittest.TestCase):
    def test_it_redraws_from_the_trace_until_told_to_finish(self):
        check = load_script()
        with tempfile.TemporaryDirectory() as tmp:
            trace = Path(tmp) / "trace.txt"
            now = time.time()
            trace.write_text(f"sink silent-paced\n{now:.6f} 32000\n")
            stderr = io.StringIO()
            with mock.patch.object(check.sys, "stderr", stderr):
                progress = check.Progress(trace, start=now - 0.5, rate=16000, estimated_s=10.0)
                progress.start()
                time.sleep(0.35)
                progress.finish()
            self.assertFalse(progress.is_alive())
            self.assertIn("played", stderr.getvalue())
            self.assertTrue(stderr.getvalue().endswith("\r\x1b[K"), "line must be cleared at the end")


if __name__ == "__main__":
    unittest.main()
