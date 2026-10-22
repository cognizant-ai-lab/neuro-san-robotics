import importlib.util
import unittest
from pathlib import Path

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


if __name__ == "__main__":
    unittest.main()
