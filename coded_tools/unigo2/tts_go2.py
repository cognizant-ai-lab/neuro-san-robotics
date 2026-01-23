"""
go2_tts.py — CodedTool wrapper for TTS functionality.

This module provides the CodedTool integration for the core TTS functionality.
For direct TTS usage without framework dependencies, import from tts_core.
"""

import argparse
import logging
from typing import Any, Dict

from neuro_san.interfaces.coded_tool import CodedTool
from coded_tools.unigo2.tts_core import say, TTSEngine


# ---------------------------------------------------------------------
# CodedTool wrapper
# ---------------------------------------------------------------------

class Go2TTSTool(CodedTool):
    """
    CodedTool wrapper for TTS with persistent engine.

    The engine is initialized once and reused for all TTS calls,
    avoiding model reload overhead.
    """

    def __init__(self):
        super().__init__()
        self._engine = None

    async def async_invoke(self, args: Dict[str, Any], sly_data: Dict[str, Any]) -> Any:
        text = args.get("text")
        if not isinstance(text, str) or not text.strip():
            return "Missing required 'text'"

        try:
            # Use persistent engine for better performance
            if self._engine is None:
                self._engine = TTSEngine()
                self._engine.__enter__()

            self._engine.say(
                text=text,
                rate=int(args.get("rate", 150)),
                volume=float(args.get("volume", 1.0)),
                voice=args.get("voice", "en-us+f3"),
                alsa_device=args.get("alsa_device"),
            )
            return f"TTS OK: {text}"
        except Exception as e:
            logging.exception("TTS failed")
            return f"TTS error: {e}"

    def __del__(self):
        """Cleanup engine on tool destruction."""
        if self._engine is not None:
            try:
                self._engine.__exit__(None, None, None)
            except Exception:
                pass


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser("Offline TTS for Unitree Go2")
    parser.add_argument("text", nargs="*", help="Text to speak")
    args = parser.parse_args()

    text = " ".join(args.text) if args.text else "Hello from Unitree!"
    say(text)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
