import unittest

from apps.conscious_assistant.agent_output import combine_speech_blocks
from apps.conscious_assistant.agent_output import parse_agent_output_blocks


class AgentOutputTests(unittest.TestCase):
    def test_parse_agent_output_blocks_splits_thoughts_and_speech(self):
        thoughts, speeches = parse_agent_output_blocks(
            "thought: planning a reply\n"
            "say: Hello there.\n"
            "say: How can I help?\n"
        )

        self.assertEqual(thoughts, ["planning a reply\n"])
        self.assertEqual(speeches, ["Hello there.\n", "How can I help?\n"])

    def test_parse_agent_output_blocks_accepts_case_and_indentation(self):
        thoughts, speeches = parse_agent_output_blocks(
            "  Thought: checking the route\n"
            "  Say: Starting now.\n"
        )

        self.assertEqual(thoughts, ["checking the route\n"])
        self.assertEqual(speeches, ["Starting now.\n"])

    def test_combine_speech_blocks_preserves_ui_lines_and_coalesces_tts(self):
        display_text, spoken_text = combine_speech_blocks(
            [
                "Hello there.",
                "I can help with that.\nLet us begin.",
            ]
        )

        self.assertEqual(
            display_text,
            "Hello there.\nI can help with that.\nLet us begin.",
        )
        self.assertEqual(
            spoken_text,
            "Hello there. I can help with that. Let us begin.",
        )


if __name__ == "__main__":
    unittest.main()
