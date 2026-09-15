"""Engine selection and fallback order for text-to-speech."""

import os
import unittest
from unittest.mock import patch

from coded_tools.unigo2 import tts_engines
from coded_tools.unigo2.tts_engines import EngineRegistry, TtsEngine


def engine(name, *, available=True, fails=None, log=None):
    """Build a stub engine that records that it was asked to speak."""
    def speak(text, **kwargs):
        if log is not None:
            log.append(name)
        if fails is not None:
            raise fails
    return TtsEngine(name=name, available=lambda: available, speak=speak)


def registry(*engines):
    reg = EngineRegistry()
    for item in engines:
        reg.register(item)
    return reg


class Interrupted(RuntimeError):
    """Stands in for tts_go2.SpeechInterrupted, which is also a RuntimeError."""


class EngineOrderTests(unittest.TestCase):
    def test_auto_uses_the_default_order(self):
        with patch.dict(os.environ, {"GO2_TTS_ENGINE": "auto"}):
            self.assertEqual(
                registry().requested(), list(tts_engines.DEFAULT_ORDER)
            )

    def test_unset_behaves_as_auto(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("GO2_TTS_ENGINE", None)
            self.assertEqual(
                registry().requested(), list(tts_engines.DEFAULT_ORDER)
            )

    def test_an_explicit_chain_is_honoured_in_order(self):
        log = []
        reg = registry(
            engine("piper", available=False, log=log),
            engine("espeak", log=log),
        )
        with patch.dict(os.environ, {"GO2_TTS_ENGINE": "piper,espeak"}):
            self.assertEqual([e.name for e in reg.chain()], ["espeak"])
            self.assertEqual(reg.speak("hi"), "espeak")

    def test_unavailable_engines_are_skipped_in_auto(self):
        """A robot with no Piper voice must fall through, not go silent."""
        log = []
        reg = registry(
            engine("piper", available=False, log=log),
            engine("espeak", log=log),
        )
        with patch.dict(os.environ, {"GO2_TTS_ENGINE": "auto"}):
            self.assertEqual(reg.speak("hi"), "espeak")
            self.assertEqual(log, ["espeak"])

    def test_a_failing_engine_falls_through_to_the_next(self):
        log = []
        reg = registry(
            engine("piper", fails=RuntimeError("no voice"), log=log),
            engine("espeak", log=log),
        )
        with patch.dict(os.environ, {"GO2_TTS_ENGINE": "auto"}):
            self.assertEqual(reg.speak("hi"), "espeak")
            self.assertEqual(log, ["piper", "espeak"])

    def test_naming_one_engine_surfaces_its_failure(self):
        """Choosing an engine explicitly means you want to hear when it breaks."""
        reg = registry(
            engine("piper", fails=RuntimeError("no voice")),
            engine("espeak"),
        )
        with patch.dict(os.environ, {"GO2_TTS_ENGINE": "piper"}):
            self.assertFalse(reg.is_forgiving())
            with self.assertRaises(RuntimeError):
                reg.speak("hi")

    def test_a_named_engine_that_is_not_installed_here_is_substituted(self):
        """
        One setmyenv.sh runs on the robot and on a laptop. Naming piper on a
        host that has no piper should reach for whatever that host does have,
        because the alternative is the robot going silent. An engine that is
        installed and then fails is a different case, covered in
        HostPortabilityTests.
        """
        reg = registry(
            engine("piper", available=False),
            engine("espeak", available=True),
        )
        with patch.dict(os.environ, {"GO2_TTS_ENGINE": "piper"}):
            self.assertEqual([e.name for e in reg.chain()], ["espeak"])

    def test_an_unknown_named_engine_is_a_clear_error(self):
        reg = registry(engine("espeak"))
        with patch.dict(os.environ, {"GO2_TTS_ENGINE": "pipr"}):
            with self.assertRaises(ValueError) as caught:
                reg.chain()
        self.assertIn("pipr", str(caught.exception))
        self.assertIn("espeak", str(caught.exception))

    def test_no_usable_engine_says_what_was_tried(self):
        reg = registry(engine("piper", available=False))
        with patch.dict(os.environ, {"GO2_TTS_ENGINE": "auto"}):
            with self.assertRaises(RuntimeError) as caught:
                reg.speak("hi")
        self.assertIn("No usable TTS engine", str(caught.exception))


class HostPortabilityTests(unittest.TestCase):
    """
    One setmyenv.sh is copied to every machine, so the engine it names will
    not exist on all of them. Piper is Linux-only; a laptop has `say` instead.
    """

    def robot(self, piper=True, espeak=True):
        """A Linux robot: piper and espeak, no macOS `say`."""
        return registry(
            engine("piper", available=piper),
            engine("say", available=False),
            engine("espeak", available=espeak),
        )

    def mac(self):
        """A laptop: only `say`."""
        return registry(
            engine("piper", available=False),
            engine("say", available=True),
            engine("espeak", available=False),
        )

    def names(self, reg, value):
        with patch.dict(os.environ, {"GO2_TTS_ENGINE": value}):
            return [e.name for e in reg.chain()]

    def test_the_robot_is_unchanged(self):
        """Piper is installed there, so it must still be chosen, alone."""
        reg = self.robot()
        self.assertEqual(self.names(reg, "piper"), ["piper"])
        self.assertEqual(self.names(reg, "auto"), ["piper", "espeak"])
        self.assertEqual(self.names(reg, "espeak"), ["espeak"])
        self.assertEqual(self.names(reg, "piper,espeak"), ["piper", "espeak"])

    def test_a_laptop_falls_back_to_what_it_has(self):
        """GO2_TTS_ENGINE=piper on a Mac must speak, not raise and go silent."""
        self.assertEqual(self.names(self.mac(), "piper"), ["say"])

    def test_the_robot_without_a_piper_voice_still_speaks(self):
        reg = self.robot(piper=False)
        self.assertEqual(self.names(reg, "piper"), ["espeak"])

    def test_an_installed_engine_that_fails_is_still_raised(self):
        """Not-installed-here is a config mismatch; installed-and-broken is a fault."""
        reg = registry(
            engine("piper", available=True, fails=RuntimeError("piper segfaulted")),
            engine("espeak", available=True),
        )
        with patch.dict(os.environ, {"GO2_TTS_ENGINE": "piper"}):
            with self.assertRaises(RuntimeError):
                reg.speak("hi")

    def test_a_host_with_no_engines_at_all_still_reports_clearly(self):
        reg = registry(engine("piper", available=False))
        with patch.dict(os.environ, {"GO2_TTS_ENGINE": "piper"}):
            with self.assertRaises(RuntimeError) as caught:
                reg.speak("hi")
        self.assertIn("No usable TTS engine", str(caught.exception))


class BargeInTests(unittest.TestCase):
    """A barge-in must stop the utterance, not roll on to the next engine."""

    def test_an_interruption_is_not_treated_as_an_engine_failure(self):
        log = []
        reg = registry(
            engine("piper", fails=Interrupted("user spoke"), log=log),
            engine("espeak", log=log),
        )
        with patch.dict(os.environ, {"GO2_TTS_ENGINE": "auto"}):
            with self.assertRaises(Interrupted):
                reg.speak("hi", passthrough=(Interrupted,))
        # espeak must never run: doing so restarts the sentence the user
        # just talked over, which is the whole point of barge-in.
        self.assertEqual(log, ["piper"])

    def test_without_passthrough_it_would_fall_through(self):
        """Documents why passthrough exists: RuntimeError is otherwise caught."""
        log = []
        reg = registry(
            engine("piper", fails=Interrupted("user spoke"), log=log),
            engine("espeak", log=log),
        )
        with patch.dict(os.environ, {"GO2_TTS_ENGINE": "auto"}):
            reg.speak("hi")
        self.assertEqual(log, ["piper", "espeak"])


class RegistrationTests(unittest.TestCase):
    def test_registering_a_name_twice_replaces_it(self):
        """Lets a test or a site swap an engine without touching internals."""
        reg = registry(engine("piper"))
        replacement = engine("piper")
        reg.register(replacement)
        self.assertEqual(len(reg.known()), 1)
        self.assertIs(reg.engines["piper"], replacement)

    def test_the_real_registry_has_the_offline_engines(self):
        from coded_tools.unigo2 import tts_go2  # noqa: F401 - registers on import

        for name in ("piper", "espeak", "say"):
            self.assertIn(name, tts_engines.REGISTRY.known())

    def test_every_name_in_the_default_order_actually_exists(self):
        """
        A name in DEFAULT_ORDER that nothing registers is silently skipped in
        auto, but makes an explicit GO2_TTS_ENGINE of that name fail as though
        it were a typo. Listing an engine before implementing it did exactly
        that to "pocket".
        """
        from coded_tools.unigo2 import tts_go2  # noqa: F401 - registers on import

        accounted = set(tts_engines.REGISTRY.known()) | tts_engines.EXTERNALLY_HANDLED
        for name in tts_engines.DEFAULT_ORDER:
            self.assertIn(name, accounted, f"{name} is advertised but unreachable")

    def test_every_default_order_name_is_selectable(self):
        """Each advertised engine must resolve when named on its own."""
        from coded_tools.unigo2 import tts_go2  # noqa: F401 - registers on import

        for name in tts_engines.DEFAULT_ORDER:
            with patch.dict(os.environ, {"GO2_TTS_ENGINE": name}):
                try:
                    tts_engines.REGISTRY.chain()
                except ValueError as error:
                    self.fail(f"GO2_TTS_ENGINE={name} rejected: {error}")

    def test_the_hosted_engine_is_not_reported_as_a_typo(self):
        """GO2_TTS_ENGINE=openai predates this registry and must keep working."""
        from coded_tools.unigo2 import tts_go2  # noqa: F401 - registers on import

        with patch.dict(os.environ, {"GO2_TTS_ENGINE": "openai"}):
            self.assertEqual(tts_engines.REGISTRY.chain(), [])

    def test_a_genuine_typo_is_still_rejected(self):
        with patch.dict(os.environ, {"GO2_TTS_ENGINE": "openal"}):
            with self.assertRaises(ValueError):
                tts_engines.REGISTRY.chain()


if __name__ == "__main__":
    unittest.main()
