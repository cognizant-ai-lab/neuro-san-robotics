"""
Registry of text-to-speech engines and the order they are tried in.

Engine choice used to be spread across five places in tts_go2.py: a module
constant, two availability gates, a hardcoded Linux chain inside
_say_single_chunk, and a separate gate deciding whether failures may fall
through. Adding an engine meant editing all five, and the order was not
configurable at all. This module holds that decision in one place so an engine
is a small object registered once, and the order is an environment variable.

GO2_TTS_ENGINE selects what runs:

    unset / "auto"        try each engine in DEFAULT_ORDER, skipping the ones
                          that are not installed, until one speaks
    "piper"               use exactly that engine; if it fails, the failure is
                          raised rather than hidden behind a fallback
    "pocket,piper,espeak" try exactly these, in this order

The single-name form keeps its old meaning: naming an engine explicitly means
you want to know when it breaks, so it is a chain of one rather than a
preference. "auto" is the forgiving mode, where a timeout or a missing model
quietly moves to the next engine.
"""

import logging
import os
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List


# Hosted speech first when it is configured: it is the best quality and the
# robot is usually online. Everything after it is local, in descending order of
# how good it sounds. espeak-ng is last because it always works.
#
# "openai" rather than "hosted" because GO2_TTS_ENGINE=openai is already a
# documented setting on deployed robots, and it now covers Azure as well.
DEFAULT_ORDER = ("openai", "pocket", "piper", "say", "espeak")


@dataclass(frozen=True)
class TtsEngine:
    """One way of turning text into speech on this robot."""

    name: str
    available: Callable[[], bool]
    speak: Callable[..., None]
    description: str = ""
    # Engines that reach the network can fail for reasons that have nothing to
    # do with configuration, which is worth saying out loud in a log line.
    hosted: bool = False


@dataclass
class EngineRegistry:
    """The engines this build knows about, and the order to try them in."""

    engines: Dict[str, TtsEngine] = field(default_factory=dict)

    def register(self, engine: TtsEngine) -> None:
        """Add an engine. Re-registering a name replaces it, which lets a test
        swap in a stub without reaching into module internals."""
        self.engines[engine.name] = engine

    def known(self) -> List[str]:
        """Every registered engine name, in registration order."""
        return list(self.engines)

    def requested(self) -> List[str]:
        """Return the engine names asked for, before availability is checked."""
        raw = os.environ.get("GO2_TTS_ENGINE", "auto").strip().lower()
        if not raw or raw == "auto":
            return list(DEFAULT_ORDER)
        return [name for name in (part.strip() for part in raw.split(",")) if name]

    def is_forgiving(self) -> bool:
        """
        Whether a failing engine may fall through to the next one.

        Naming a single engine is a statement that you want its failures, so
        only "auto" and an explicit multi-engine chain fall through.
        """
        return len(self.requested()) > 1

    def chain(self) -> List[TtsEngine]:
        """
        Return the engines to try, in order, skipping unavailable ones.

        An engine named explicitly is kept even when it reports itself
        unavailable, so the caller gets that engine's own error rather than a
        silent no-op. In "auto" the unavailable ones are dropped, which is what
        makes a robot with no Piper voice fall through to espeak.
        """
        names = self.requested()
        explicit = names != list(DEFAULT_ORDER)
        chain: List[TtsEngine] = []

        for name in names:
            engine = self.engines.get(name)
            if engine is None:
                if explicit:
                    raise ValueError(
                        f"Unknown TTS engine {name!r} in GO2_TTS_ENGINE. "
                        f"Known engines: {', '.join(self.known())}"
                    )
                continue
            if explicit and len(names) == 1:
                chain.append(engine)
                continue
            try:
                if engine.available():
                    chain.append(engine)
            except Exception:
                logging.exception("TTS engine %s failed its availability check", name)

        return chain

    def speak(self, text: str, passthrough: tuple = (), **kwargs: Any) -> str:
        """
        Speak text with the first engine that manages it.

        Returns the name of the engine that spoke, so the caller can log what
        actually happened rather than what was configured.

        `passthrough` names exception types that mean "stop", not "this engine
        broke". A barge-in is the important one: it arrives as an exception out
        of the engine, but it is the user talking over the robot, and trying
        the next engine would restart the utterance they just interrupted.
        """
        chain = self.chain()
        if not chain:
            raise RuntimeError(
                "No usable TTS engine. Tried: "
                f"{', '.join(self.requested())}. Registered: "
                f"{', '.join(self.known()) or 'none'}."
            )

        last_error: Exception | None = None
        for index, engine in enumerate(chain):
            try:
                engine.speak(text, **kwargs)
                return engine.name
            except passthrough:
                raise
            except Exception as error:
                # The last engine has nowhere to fall through to, and a
                # single named engine was chosen precisely to surface errors.
                if index == len(chain) - 1 or not self.is_forgiving():
                    raise
                last_error = error
                logging.warning(
                    "TTS engine %s failed (%s); trying %s",
                    engine.name, error, chain[index + 1].name,
                )

        raise RuntimeError(f"Every TTS engine failed; last error: {last_error}")


# The registry tts_go2.py populates at import time.
REGISTRY = EngineRegistry()
