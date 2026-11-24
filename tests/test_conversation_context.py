"""Short-term conversation and camera context attached to spoken turns."""

import unittest
from unittest.mock import patch

from coded_tools.unigo2 import agent_events


class TurnContextTest(unittest.TestCase):
    def setUp(self):
        self.now = 1000.0
        patches = [
            patch.object(agent_events, "_CONTEXT", agent_events.TurnContext()),
            patch.object(agent_events.time, "monotonic", lambda: self.now),
            patch.object(agent_events, "latest_navigation_awareness", lambda: ""),
            patch.object(agent_events, "_post_json"),
        ]
        for patcher in patches:
            patcher.start()
            self.addCleanup(patcher.stop)
        self.post = agent_events._post_json

    def sent_text(self) -> str:
        return self.post.call_args.args[1]["user_message"]["text"]

    def test_earlier_exchange_reaches_the_next_user_turn(self):
        agent_events.dispatch_agent_event("Ich heiße Alex", source="user")
        agent_events.remember_conversation("you", "Hallo Alex!")

        agent_events.dispatch_agent_event("Wie heiße ich?", source="user")

        self.assertEqual(
            self.sent_text(),
            "user: Wie heiße ich?\n"
            "system: Recent conversation (oldest first, authoritative):\n"
            "  user: Ich heiße Alex\n"
            "  you: Hallo Alex!",
        )

    def test_conversation_is_forgotten_after_five_quiet_minutes(self):
        agent_events.remember_conversation("you", "Hallo Alex!")
        self.now += 5 * 60 + 1

        agent_events.dispatch_agent_event("Wie heiße ich?", source="user")

        self.assertEqual(self.sent_text(), "user: Wie heiße ich?")

    def test_only_the_last_ten_lines_are_kept(self):
        for number in range(12):
            agent_events.remember_conversation("you", f"line {number}")

        agent_events.dispatch_agent_event("und?", source="user")

        self.assertNotIn("line 1\n", self.sent_text())
        self.assertIn("  you: line 2\n", self.sent_text())
        self.assertTrue(self.sent_text().endswith("  you: line 11"))

    def test_ambient_turns_see_the_conversation_but_are_not_recorded(self):
        agent_events.remember_conversation("you", "Hallo!")

        agent_events.dispatch_agent_event("redet ihr über mich?", source="ambient")
        self.assertIn("  you: Hallo!", self.sent_text())

        agent_events.dispatch_agent_event("was?", source="user")
        self.assertNotIn("redet ihr", self.sent_text())

    def test_internal_events_get_no_conversation(self):
        agent_events.remember_conversation("you", "Hallo!")

        agent_events.dispatch_agent_event("{}", source="observation")

        self.assertEqual(self.sent_text(), "observation: {}")

    def test_recent_camera_labels_reach_the_next_user_turn(self):
        agent_events.remember_observation(["tv", "person", "Alex"])
        self.now += 8

        agent_events.dispatch_agent_event("Was siehst du?", source="user")

        self.assertEqual(
            self.sent_text(),
            "user: Was siehst du?\nsystem: Camera labels 8 s ago: tv, person, Alex",
        )

    def test_an_empty_scene_is_reported_as_nothing_recognized(self):
        agent_events.remember_observation([])

        agent_events.dispatch_agent_event("Was siehst du?", source="user")

        self.assertIn("system: Camera labels 0 s ago: nothing recognized", self.sent_text())

    def test_camera_labels_older_than_a_minute_are_dropped(self):
        agent_events.remember_observation(["tv"])
        self.now += 61

        agent_events.dispatch_agent_event("Was siehst du?", source="user")

        self.assertEqual(self.sent_text(), "user: Was siehst du?")


if __name__ == "__main__":
    unittest.main()
