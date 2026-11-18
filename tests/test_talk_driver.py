"""Checks the scenario runner's pass/fail logic without a running app."""

import unittest

from scripts.talk_driver import check_turn
from scripts.talk_driver import turn_satisfied


class CheckTurnTest(unittest.TestCase):
    def test_reply_expected_but_silent(self):
        self.assertEqual(check_turn({}, [], "", None), ["expected a reply, got none"])

    def test_slow_reply_fails(self):
        replies = [("update_thoughts", "hmm")]
        self.assertTrue(check_turn({"max_reaction": 5}, replies, "", 9.0))
        self.assertEqual(check_turn({"max_reaction": 5}, replies, "", 2.0), [])

    def test_silence_expected(self):
        self.assertEqual(check_turn({"reply": False}, [], "", None), [])
        self.assertTrue(check_turn({"reply": False}, [("update_speech", "hi")], "", 1.0))

    def test_speech_regex_only_reads_speech(self):
        thought_only = [("update_thoughts", "I am CAIL-E")]
        self.assertTrue(check_turn({"speech_matches": "CAIL"}, thought_only, "", 1.0))
        spoken = thought_only + [("update_speech", "I'm CAIL-E")]
        self.assertEqual(check_turn({"speech_matches": "CAIL"}, spoken, "", 1.0), [])

    def test_log_regex(self):
        log = "[12:00:00] robot_macros: skipped 'dance' (robot control unavailable)\n"
        expect = {"log_matches": r"skipped 'dance\d?'"}
        self.assertEqual(check_turn(expect, [("update_thoughts", "x")], log, 1.0), [])
        self.assertTrue(check_turn(expect, [("update_thoughts", "x")], "", 1.0))

    def test_turn_waits_for_speech_after_thought(self):
        expect = {"speech_matches": "hi"}
        self.assertFalse(turn_satisfied(expect, [("update_thoughts", "x")], ""))
        self.assertTrue(turn_satisfied(expect, [("update_speech", "hi")], ""))


if __name__ == "__main__":
    unittest.main()
