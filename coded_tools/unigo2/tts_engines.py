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
    "piper,espeak"        try exactly these, in this order

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
#
# Only engines that exist belong here. pocket-tts is a candidate but is not
# implemented, and listing a name nothing registers makes an explicit
# GO2_TTS_ENGINE="pocket" fail as though it were a typo.
DEFAULT_ORDER = ("openai", "piper", "say", "espeak")

# Names this registry recognises but does not own. The hosted engine streams
# audio as it arrives rather than handing back a finished utterance, so
# say_streaming() drives it directly and the registry never sees it. Listing it
# keeps an explicit GO2_TTS_ENGINE="openai" from being reported as unknown.
EXTERNALLY_HANDLED = frozenset({"openai"})


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

    def _is_available(self, engine: TtsEngine) -> bool:
        """Whether an engine reports itself usable, treating a broken check as no."""
        try:
            return bool(engine.available())
        except Exception:
            logging.exception("TTS engine %s failed its availability check", engine.name)
            return False

    def chain(self) -> List[TtsEngine]:
        """
        Return the engines to try, in order, skipping the ones not installed.

        An engine that is asked for but not installed on this machine is a
        different thing from one that is installed and fails. The first is a
        config that does not fit the host -- the same setmyenv.sh naming piper
        runs on the robot and on a laptop, where piper does not exist -- and
        falling through to something that works beats silence. The second is a
        real fault and is raised by speak().

        So an explicit request that matches nothing available here drops back to
        the default order rather than failing, and says so.

        The hosted engine never appears in the returned chain even though it is
        first in DEFAULT_ORDER. say_streaming() has already tried it by the time
        this runs, so a chain of ["piper", "espeak"] under "auto" means the full
        order is hosted, then piper, then espeak -- not that hosted was skipped.
        """
        names = self.requested()
        explicit = names != list(DEFAULT_ORDER)
        chain: List[TtsEngine] = []

        # A request naming only engines this registry does not own -- in
        # practice GO2_TTS_ENGINE="openai" -- is not ours to satisfy or to
        # substitute for. Returning nothing lets the hosted path keep its
        # meaning of "use the hosted model, and tell me when it fails".
        if explicit and all(name in EXTERNALLY_HANDLED for name in names):
            return []

        for name in names:
            if name in EXTERNALLY_HANDLED:
                continue
            engine = self.engines.get(name)
            if engine is None:
                if explicit:
                    known = ", ".join(sorted(set(self.known()) | EXTERNALLY_HANDLED))
                    raise ValueError(
                        f"Unknown TTS engine {name!r} in GO2_TTS_ENGINE. "
                        f"Known engines: {known}"
                    )
                continue
            if self._is_available(engine):
                chain.append(engine)

        if chain or not explicit:
            return chain

        # Nothing that was asked for exists here. Rather than go silent, use
        # whatever this host does have, and make the substitution visible.
        fallback = [
            engine
            for name in DEFAULT_ORDER
            if name not in EXTERNALLY_HANDLED
            and (engine := self.engines.get(name)) is not None
            and self._is_available(engine)
        ]
        if fallback:
            logging.warning(
                "TTS engine(s) %s not available here; using %s instead",
                ", ".join(names), fallback[0].name,
            )
        return fallback

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
