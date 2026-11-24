"""The Piper first-sentence benchmark prints one line per test reply (#16)."""

import contextlib
import io
import types
import unittest
from unittest.mock import patch

from scripts import piper_first_sentence_bench as bench


class FakeVoice:
    def synthesize(self, text):
        yield types.SimpleNamespace(phonemes=list(text.split(".")[0]))
        raise AssertionError("only the first chunk may be synthesised")


class BenchTests(unittest.TestCase):
    def test_one_line_per_reply_with_its_length_and_first_chunk_time(self):
        out = io.StringIO()
        with (
            patch.object(bench.tts_go2, "_piper_voice", return_value=FakeVoice()),
            contextlib.redirect_stdout(out),
        ):
            bench.main(["Hi. There.", "Hello"])

        lines = out.getvalue().splitlines()
        self.assertEqual(len(lines), 2)
        self.assertRegex(lines[0], r"^\s+10 chars\s+2 phonemes  first chunk\s+\d+ ms  'Hi. There.'$")
        self.assertRegex(lines[1], r"^\s+5 chars\s+5 phonemes")


if __name__ == "__main__":
    unittest.main()
