"""Tests for interrupting the robot mid-sentence and keeping the mic open.

The robot now hears itself, because the capture track is no longer muted while
it speaks. These tests cover the two mechanisms that make that safe: the speech
epoch, which discards superseded utterances and turns, and the self-echo
filter, which drops transcripts of the robot's own voice.
"""

import queue
import time
import unittest
from unittest.mock import patch

from apps.conscious_assistant import interface_flask
from coded_tools.unigo2 import agent_events


class SpeechStateTestCase(unittest.TestCase):
    """Base case that isolates the module-level speech state per test."""

    def setUp(self):
        self._saved = {
            name: getattr(interface_flask, name)
            for name in (
                "_speech_epoch",
                "_speech_active",
                "_active_speech_text",
                "_active_speech_ended_at",
                "_last_barge_in_at",
            )
        }
        # The imported module runs a real speech worker against the real queue;
        # give each test its own so draining it cannot race with playback.
        patcher = patch.object(interface_flask, "speech_queue", queue.Queue())
        self.speech_queue = patcher.start()
        self.addCleanup(patcher.stop)

    def tearDown(self):
        for name, value in self._saved.items():
            setattr(interface_flask, name, value)

    def set_speaking(self, text, *, active=True, ended_at=None):
        """Pretend the robot is (or just finished) saying `text`."""
        interface_flask._speech_active = active
        interface_flask._active_speech_text = text
        interface_flask._active_speech_ended_at = (
            time.monotonic() if ended_at is None else ended_at
        )


class SelfEchoTests(SpeechStateTestCase):
    def test_robot_hearing_itself_is_suppressed(self):
        self.set_speaking("The kitchen is down the hall past the main desk area")

        self.assertTrue(interface_flask.is_self_echo("the kitchen is down the hall"))

    def test_real_speech_during_playback_is_not_an_echo(self):
        self.set_speaking("The kitchen is down the hall past the main desk area")

        self.assertFalse(interface_flask.is_self_echo("stop walking and sit down"))

    def test_nothing_is_an_echo_when_the_robot_is_silent(self):
        interface_flask._speech_active = False
        interface_flask._active_speech_text = ""

        self.assertFalse(interface_flask.is_self_echo("the kitchen is down the hall"))

    def test_echo_window_closes_after_the_tail(self):
        stale = time.monotonic() - (interface_flask.SELF_ECHO_TAIL_SECONDS + 1.0)
        self.set_speaking("the kitchen is down the hall", active=False, ended_at=stale)

        # The same words much later are a person quoting the robot, not an echo.
        self.assertFalse(interface_flask.is_self_echo("the kitchen is down the hall"))

    def test_echo_still_matches_just_after_playback_ends(self):
        self.set_speaking("the kitchen is down the hall", active=False)

        self.assertTrue(interface_flask.is_self_echo("the kitchen is down the hall"))

    def test_talking_over_the_robot_is_not_mistaken_for_echo(self):
        # Barging in puts both voices in one transcript, so most of it really
        # is the robot. Only the words the robot never said give the person away.
        self.set_speaking("the kitchen is down the hall past the main desk area")

        self.assertFalse(
            interface_flask.is_self_echo("no wait the kitchen is down the hall")
        )

    def test_filler_words_do_not_fake_an_interruption(self):
        # Room-mic transcription of the robot's own voice routinely adds
        # leading fillers and swaps small function words. Counting those as a
        # person makes the robot cut itself off and feed its own sentence back.
        self.set_speaking("the kitchen is down the hall past the main desk area")

        for transcript in (
            "uh the kitchen is down the hall past the main desk area",
            "yeah so the kitchen is down the hall past the main desk area",
            "the kitchen is down the hall past the main desk in a area",
            "so uh the kitchen is down the hall",
        ):
            with self.subTest(transcript=transcript):
                self.assertTrue(interface_flask.is_self_echo(transcript))

    def test_substantive_words_still_signal_a_person(self):
        self.set_speaking("the kitchen is down the hall past the main desk area")

        for transcript in (
            "no wait the kitchen is down the hall",
            "stop walking and sit down",
            "actually never mind go to the charging station",
        ):
            with self.subTest(transcript=transcript):
                self.assertFalse(interface_flask.is_self_echo(transcript))

    def test_a_garbled_echo_is_still_an_echo(self):
        self.set_speaking("the kitchen is down the hall past the main desk area")

        # One dropped word is transcription noise, not a person speaking.
        self.assertTrue(interface_flask.is_self_echo("the kitchen is down hall"))


class CancelSpeechTests(SpeechStateTestCase):
    def test_cancel_bumps_the_epoch_and_drains_queued_speech(self):
        interface_flask.enqueue_speech("first")
        interface_flask.enqueue_speech("second")
        before = interface_flask._speech_epoch

        with patch.object(interface_flask, "tts_stop_speaking", return_value=True):
            interface_flask.cancel_speech(reason="test")

        self.assertGreater(interface_flask._speech_epoch, before)
        self.assertTrue(self.speech_queue.empty())

    def test_cancel_preserves_the_shutdown_sentinel(self):
        interface_flask.enqueue_speech("doomed")
        self.speech_queue.put(None)

        with patch.object(interface_flask, "tts_stop_speaking", return_value=False):
            interface_flask.cancel_speech(reason="test")

        # The worker still needs its stop signal after a barge-in.
        self.assertIs(self.speech_queue.get_nowait(), None)

    def test_queued_speech_carries_the_epoch_it_was_created_at(self):
        interface_flask.enqueue_speech("hello")
        job = self.speech_queue.get_nowait()

        self.assertEqual(job["epoch"], interface_flask._speech_epoch)

    def test_speech_queued_before_a_cancel_is_superseded(self):
        interface_flask.enqueue_speech("stale")
        stale_job = self.speech_queue.get_nowait()

        with patch.object(interface_flask, "tts_stop_speaking", return_value=False):
            interface_flask.cancel_speech(reason="test")
        interface_flask.enqueue_speech("fresh")
        fresh_job = self.speech_queue.get_nowait()

        self.assertLess(stale_job["epoch"], interface_flask._speech_epoch)
        self.assertEqual(fresh_job["epoch"], interface_flask._speech_epoch)


class BargeInThresholdTests(SpeechStateTestCase):
    def test_a_single_word_does_not_cut_the_robot_off(self):
        self.set_speaking("a long answer the user is listening to")

        self.assertFalse(interface_flask._should_barge_in("uh"))

    def test_a_real_sentence_cuts_the_robot_off(self):
        self.set_speaking("a long answer the user is listening to")

        self.assertTrue(interface_flask._should_barge_in("actually never mind"))

    def test_nothing_to_interrupt_when_the_robot_is_silent(self):
        interface_flask._speech_active = False

        self.assertFalse(interface_flask._should_barge_in("actually never mind"))

    def test_speech_still_queued_counts_as_interruptible(self):
        interface_flask._speech_active = False
        interface_flask.enqueue_speech("about to be said")

        self.assertTrue(interface_flask._should_barge_in("actually never mind"))


class AmbientEventCoalescingTests(unittest.TestCase):
    """Ambient transcripts must not queue up behind one another."""

    def setUp(self):
        agent_events._PENDING_BY_SOURCE.clear()
        self.addCleanup(agent_events._PENDING_BY_SOURCE.clear)

        self.submitted = []
        patcher = patch.object(
            agent_events._EVENT_DISPATCHER,
            "submit",
            lambda fn, *args, **kwargs: self.submitted.append((fn, args, kwargs)),
        )
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_a_newer_transcript_replaces_one_still_waiting(self):
        agent_events.queue_agent_event("first thing said", source="ambient")
        agent_events.queue_agent_event("second thing said", source="ambient")

        # One worker task, and it will pick up the newest text.
        self.assertEqual(len(self.submitted), 1)
        self.assertEqual(
            agent_events._PENDING_BY_SOURCE["ambient"], "second thing said"
        )

    def test_the_worker_dispatches_only_the_newest_text(self):
        agent_events.queue_agent_event("first thing said", source="ambient")
        agent_events.queue_agent_event("second thing said", source="ambient")

        with patch.object(agent_events, "dispatch_agent_event") as dispatch:
            agent_events._dispatch_latest("ambient")

        dispatch.assert_called_once_with("second thing said", source="ambient")

    def test_a_claimed_slot_lets_the_next_transcript_queue_again(self):
        agent_events.queue_agent_event("first thing said", source="ambient")
        with patch.object(agent_events, "dispatch_agent_event"):
            agent_events._dispatch_latest("ambient")

        agent_events.queue_agent_event("later thing said", source="ambient")

        self.assertEqual(len(self.submitted), 2)

    def test_user_turns_keep_strict_ordering(self):
        agent_events.queue_agent_event("first question", source="user")
        agent_events.queue_agent_event("second question", source="user")

        # Explicit turns are never superseded, so both are submitted directly.
        self.assertEqual(len(self.submitted), 2)
        self.assertEqual(self.submitted[0][1], ("first question",))
        self.assertEqual(self.submitted[1][1], ("second question",))

    def test_empty_transcripts_are_ignored(self):
        agent_events.queue_agent_event("   ", source="ambient")

        self.assertEqual(self.submitted, [])



class BargeInSocketFlowTests(SpeechStateTestCase):
    """End-to-end cover of the duck-then-decide sequence over Socket.IO."""

    def setUp(self):
        super().setUp()
        self.stop_speaking = patch.object(
            interface_flask, "tts_stop_speaking", return_value=True
        ).start()
        self.duck_playback = patch.object(interface_flask, "tts_duck_playback").start()
        self.addCleanup(patch.stopall)
        self.client = interface_flask.socketio.test_client(
            interface_flask.app, namespace="/chat"
        )
        self.addCleanup(self.client.disconnect, namespace="/chat")

    def test_voice_activity_ducks_without_cancelling(self):
        self.set_speaking("the kitchen is down the hall past the main desk")
        epoch = interface_flask._speech_epoch

        self.client.emit("barge_in", {}, namespace="/chat")

        self.duck_playback.assert_called_once_with(True)
        # Ducking must stay reversible, so nothing is cancelled yet.
        self.assertEqual(interface_flask._speech_epoch, epoch)

    def test_self_echo_never_reaches_the_agent(self):
        self.set_speaking("the kitchen is down the hall past the main desk")
        epoch = interface_flask._speech_epoch

        with patch.object(interface_flask, "queue_agent_event") as queue_event:
            self.client.emit(
                "ambient_transcript",
                {"data": "the kitchen is down the hall"},
                namespace="/chat",
            )

        queue_event.assert_not_called()
        self.assertEqual(interface_flask._speech_epoch, epoch)

    def test_real_speech_cancels_playback_and_wakes_the_agent(self):
        self.set_speaking("the kitchen is down the hall past the main desk")
        epoch = interface_flask._speech_epoch

        with patch.object(interface_flask, "queue_agent_event") as queue_event:
            self.client.emit(
                "ambient_transcript",
                {"data": "no wait stop and sit down"},
                namespace="/chat",
            )

        queue_event.assert_called_once()
        self.assertTrue(self.stop_speaking.called)
        self.assertGreater(interface_flask._speech_epoch, epoch)

    def test_push_to_talk_cancels_outright(self):
        self.set_speaking("a long answer nobody wants to wait through")
        epoch = interface_flask._speech_epoch

        self.client.emit("barge_in", {"confirmed": True}, namespace="/chat")

        self.assertGreater(interface_flask._speech_epoch, epoch)
        self.assertTrue(self.stop_speaking.called)

    def test_navigation_announcements_survive_a_barge_in(self):
        self.set_speaking("partway through an answer")
        self.client.emit("barge_in", {"confirmed": True}, namespace="/chat")

        # A navigation outcome is authoritative and belongs to no turn, so a
        # barge-in that happened to land near it must not silence it.
        response = interface_flask.app.test_client().post(
            "/api/agent-output",
            json={"say": "I arrived at the kitchen", "source": "navigation"},
        )

        self.assertEqual(response.status_code, 200)
        job = self.speech_queue.get_nowait()
        self.assertEqual(job["text"], "I arrived at the kitchen")

    def test_agent_speech_from_an_interrupted_turn_is_not_spoken(self):
        self.set_speaking("partway through an answer")
        self.client.emit("barge_in", {"confirmed": True}, namespace="/chat")

        # The agent finishes the turn the user already talked over.
        response = interface_flask.app.test_client().post(
            "/api/agent-output", json={"say": "the rest of the stale answer"}
        )

        self.assertEqual(response.status_code, 200)
        self.assertTrue(self.speech_queue.empty())
    def test_speech_from_a_turn_interrupted_by_talking_is_dropped(self):
        self.set_speaking("the kitchen is down the hall")

        with patch.object(interface_flask, "queue_agent_event"):
            self.client.emit(
                "ambient_transcript",
                {"data": "no wait stop and sit down"},
                namespace="/chat",
            )
        # The interrupted turn's in-flight output lands right after the cancel.
        response = interface_flask.app.test_client().post(
            "/api/agent-output", json={"say": "past the main desk on your left"}
        )

        self.assertEqual(response.status_code, 200)
        self.assertTrue(self.speech_queue.empty())

    def test_speech_from_a_typed_turn_interruption_is_dropped(self):
        self.set_speaking("the kitchen is down the hall")

        with patch.object(interface_flask, "dispatch_agent_event", return_value=True):
            self.client.emit("user_input", {"data": "never mind"}, namespace="/chat")
        response = interface_flask.app.test_client().post(
            "/api/agent-output", json={"say": "past the main desk on your left"}
        )

        self.assertEqual(response.status_code, 200)
        self.assertTrue(self.speech_queue.empty())

    def test_the_answer_to_the_interruption_is_still_spoken(self):
        self.set_speaking("the kitchen is down the hall")
        self.client.emit("barge_in", {"confirmed": True}, namespace="/chat")

        # A real reply needs a model round trip, so it lands past the grace
        # window. Suppressing that would be far worse than the stale line.
        interface_flask._last_barge_in_at = time.monotonic() - (
            interface_flask.SUPERSEDED_SPEECH_GRACE_SECONDS + 1.0
        )
        response = interface_flask.app.test_client().post(
            "/api/agent-output", json={"say": "okay, stopping here"}
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(self.speech_queue.get_nowait()["text"], "okay, stopping here")


if __name__ == "__main__":
    unittest.main()
