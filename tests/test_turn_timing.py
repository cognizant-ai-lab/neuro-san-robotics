"""Tests for per-turn voice latency timing (#13): id propagation and the report maths."""

import asyncio
import queue
import unittest
from unittest.mock import patch

from apps.conscious_assistant import interface_flask
from coded_tools.unigo2 import agent_events
from coded_tools.unigo2 import turn_timing
from coded_tools.unigo2.ui_output import UiOutputTool
from scripts import turn_latency_report as report


class TurnTimingTestCase(unittest.TestCase):
    def setUp(self):
        turn_timing._OPEN.clear()
        self.addCleanup(turn_timing._OPEN.clear)

    def logged_lines(self, action):
        with self.assertLogs(turn_timing.logger, level="INFO") as logs:
            action()
        return [line for line in logs.output if "turn_timing" in line]


class TurnLifecycleTests(TurnTimingTestCase):
    def test_first_audio_logs_every_step_relative_to_the_end_of_speech(self):
        with patch.object(turn_timing.time, "monotonic", return_value=100.0):
            turn_id = turn_timing.start("ptt", ended_ago_s=1.5)
        for step, at in (("dispatched", 100.1), ("ui_output", 102.0), ("audio", 102.5)):
            with patch.object(turn_timing.time, "monotonic", return_value=at):
                if step == "audio":
                    lines = self.logged_lines(lambda: turn_timing.mark(turn_id, "audio"))
                else:
                    turn_timing.mark(turn_id, step)

        self.assertEqual(len(lines), 1)
        turn = report.parse(lines)[0]
        self.assertEqual(turn["turn_id"], turn_id)
        self.assertEqual(turn["mode"], "ptt")
        self.assertEqual(turn["transcript_ms"], 1500)
        self.assertEqual(turn["dispatched_ms"], 1600)
        self.assertEqual(turn["ui_output_ms"], 3500)
        self.assertEqual(turn["audio_ms"], 4000)
        self.assertEqual(turn["reply"], "spoken")
        self.assertNotIn(turn_id, turn_timing._OPEN)

    def test_speak_start_and_first_sentence_length_land_on_the_turn_line(self):
        with patch.object(turn_timing.time, "monotonic", return_value=100.0):
            turn_id = turn_timing.start("ptt", ended_ago_s=0)
        for step, at in (("ui_output", 101.0), ("speak_start", 101.2)):
            with patch.object(turn_timing.time, "monotonic", return_value=at):
                turn_timing.mark(turn_id, step)
        with patch.object(turn_timing.time, "monotonic", return_value=102.0):
            lines = self.logged_lines(
                lambda: turn_timing.mark(turn_id, "audio", first_sentence_phonemes=42)
            )

        self.assertRegex(lines[0], r"ui_output_ms=1000 speak_start_ms=1200 audio_ms=2000 ")
        turn = report.parse(lines)[0]
        self.assertEqual(turn["first_sentence_phonemes"], "42")

    def test_a_turn_without_audio_is_logged_as_no_reply_when_the_next_starts(self):
        silent = turn_timing.start("ambient", 0.7)
        turn_timing.mark(silent, "dispatched")

        lines = self.logged_lines(lambda: turn_timing.start("ambient", 0.7))

        turn = report.parse(lines)[0]
        self.assertEqual(turn["turn_id"], silent)
        self.assertEqual(turn["reply"], "none")
        self.assertIn("dispatched_ms", turn)
        self.assertNotIn("audio_ms", turn)

    def test_later_audio_of_the_same_turn_is_ignored(self):
        turn_id = turn_timing.start("ptt", 0)
        self.logged_lines(lambda: turn_timing.mark(turn_id, "audio"))

        with self.assertNoLogs(turn_timing.logger):
            turn_timing.mark(turn_id, "audio")
            turn_timing.mark(None, "audio")


class TurnIdPropagationTests(TurnTimingTestCase):
    def test_dispatch_carries_the_turn_id_and_marks_dispatch(self):
        turn_id = turn_timing.start("ptt", 0)
        with patch.object(agent_events, "_post_json") as post:
            agent_events.dispatch_agent_event("hello", source="user", turn_id=turn_id)

        self.assertEqual(post.call_args.args[1]["sly_data"]["turn_id"], turn_id)
        self.assertIn("dispatched", turn_timing._OPEN[turn_id])

    def test_ui_output_echoes_the_turn_id(self):
        with patch("coded_tools.unigo2.ui_output.publish_ui_output", return_value=True) as publish:
            asyncio.run(UiOutputTool().async_invoke({"say": "hi"}, {"turn_id": "abc"}))

        self.assertEqual(publish.call_args.kwargs["turn_id"], "abc")

    def test_publish_ui_output_sends_the_turn_id(self):
        with patch.object(agent_events, "_post_json") as post:
            agent_events.publish_ui_output(say="hi", turn_id="abc")

        self.assertEqual(post.call_args.args[1]["turn_id"], "abc")

    def test_a_coalesced_ambient_turn_is_folded_into_the_newer_one(self):
        agent_events._PENDING_BY_SOURCE.clear()
        self.addCleanup(agent_events._PENDING_BY_SOURCE.clear)
        second = turn_timing.start("ambient", 0.7)
        # Starting `second` logged any earlier turn; put one back as still queued.
        first = "queued01"
        turn_timing._OPEN[first] = {"mode": "ambient", "t0": 0, "transcript": 0, "coalesced": 0}

        with patch.object(agent_events._EVENT_DISPATCHER, "submit"):
            agent_events.queue_agent_event("first", source="ambient", turn_id=first)
            agent_events.queue_agent_event("second", source="ambient", turn_id=second)

        self.assertNotIn(first, turn_timing._OPEN)
        self.assertEqual(turn_timing._OPEN[second]["coalesced"], 1)
        self.assertEqual(agent_events._PENDING_BY_SOURCE["ambient"], ("second", second))


class FlaskTurnFlowTests(TurnTimingTestCase):
    def setUp(self):
        super().setUp()
        patcher = patch.object(interface_flask, "speech_queue", queue.Queue())
        self.speech_queue = patcher.start()
        self.addCleanup(patcher.stop)
        patch.object(interface_flask, "tts_stop_speaking", return_value=False).start()
        self.addCleanup(patch.stopall)
        self.client = interface_flask.socketio.test_client(interface_flask.app, namespace="/chat")
        self.addCleanup(self.client.disconnect, namespace="/chat")

    def test_push_to_talk_starts_a_turn_and_dispatches_with_its_id(self):
        with (
            patch.object(interface_flask, "dispatch_agent_event", return_value=True) as dispatch,
            patch.object(interface_flask.socketio, "start_background_task", lambda fn: fn()),
        ):
            self.client.emit("user_input", {"data": "hi", "released_ms_ago": 800}, namespace="/chat")

        turn_id = dispatch.call_args.kwargs["turn_id"]
        self.assertEqual(turn_timing._OPEN[turn_id]["mode"], "ptt")

    def test_typed_input_has_no_timed_turn(self):
        with (
            patch.object(interface_flask, "dispatch_agent_event", return_value=True) as dispatch,
            patch.object(interface_flask.socketio, "start_background_task", lambda fn: fn()),
        ):
            self.client.emit("user_input", {"data": "hi"}, namespace="/chat")

        self.assertIsNone(dispatch.call_args.kwargs["turn_id"])

    def test_local_ambient_transcript_starts_a_turn(self):
        with patch.object(interface_flask, "queue_agent_event") as queue_event:
            self.client.emit(
                "ambient_transcript",
                {"data": "robot come here", "captured_ms_ago": 900},
                namespace="/chat",
            )

        turn_id = queue_event.call_args.kwargs["turn_id"]
        self.assertEqual(turn_timing._OPEN[turn_id]["mode"], "ambient")

    def test_agent_output_reaches_the_speaker_with_its_turn_id(self):
        turn_id = turn_timing.start("ptt", 0)

        response = interface_flask.app.test_client().post(
            "/api/agent-output", json={"say": "hello", "turn_id": turn_id}
        )

        self.assertEqual(response.status_code, 200)
        self.assertIn("ui_output", turn_timing._OPEN[turn_id])
        self.assertEqual(self.speech_queue.get_nowait()["turn_id"], turn_id)

    def test_a_non_string_turn_id_is_rejected(self):
        response = interface_flask.app.test_client().post(
            "/api/agent-output", json={"say": "hello", "turn_id": 5}
        )
        self.assertEqual(response.status_code, 400)

    def test_first_audio_is_announced_with_the_playing_turn(self):
        turn_id = turn_timing.start("ptt", 0)
        interface_flask._begin_utterance("hello", turn_id)
        self.addCleanup(interface_flask._end_utterance)
        self.client.get_received("/chat")

        lines = self.logged_lines(interface_flask._on_audio_started)

        self.assertIn(f"turn_id={turn_id}", lines[0])
        events = [e for e in self.client.get_received("/chat") if e["name"] == "speech_audio_started"]
        self.assertEqual(events[0]["args"][0], {"turn_id": turn_id})

    def run_worker(self, job):
        self.speech_queue.put(job)
        self.speech_queue.put(None)
        patch.object(interface_flask, "emit_speech_state").start()
        interface_flask.speech_worker()

    def test_the_speech_worker_marks_speak_start_just_before_speaking(self):
        turn_id = turn_timing.start("ptt", 0)
        seen = []
        with patch.object(interface_flask, "speak_text",
                          lambda text: seen.append("speak_start" in turn_timing._OPEN[turn_id])):
            interface_flask.enqueue_speech("hello", turn_id=turn_id)
            self.run_worker(self.speech_queue.get_nowait())

        self.assertEqual(seen, [True])

    def test_speech_without_a_turn_plays_and_logs_no_timing(self):
        spoken = []
        with (
            patch.object(interface_flask, "speak_text", spoken.append),
            self.assertNoLogs(turn_timing.logger),
        ):
            interface_flask.enqueue_speech("passing thought")
            self.run_worker(self.speech_queue.get_nowait())

        self.assertEqual(spoken, ["passing thought"])


class ReportTests(unittest.TestCase):
    LOG = [
        "INFO:coded_tools.unigo2.turn_timing:turn_timing turn_id=a mode=ptt transcript_ms=1000 "
        "dispatched_ms=1100 ui_output_ms=3000 audio_ms=4000 coalesced=0 reply=spoken",
        "INFO:coded_tools.unigo2.turn_timing:turn_timing turn_id=b mode=ptt transcript_ms=2000 "
        "dispatched_ms=2100 ui_output_ms=5000 audio_ms=6000 coalesced=0 reply=spoken",
        "INFO:coded_tools.unigo2.turn_timing:turn_timing turn_id=c mode=ptt transcript_ms=900 "
        "dispatched_ms=950 coalesced=0 reply=none",
        "INFO:coded_tools.unigo2.turn_timing:turn_timing turn_id=d mode=ambient transcript_ms=2700 "
        "dispatched_ms=2800 ui_output_ms=4800 audio_ms=5400 coalesced=1 reply=spoken",
        "INFO:root:unrelated line",
        "INFO:coded_tools.unigo2.turn_timing:turn_timing turn_id=e mode=ptt transcript_ms=1000 "
        "dispatched_ms=1100 ui_output_ms=3000 speak_start_ms=3500 audio_ms=5000 "
        "first_sentence_phonemes=40 coalesced=0 reply=spoken",
    ]

    def test_parse_reads_only_timing_lines(self):
        turns = report.parse(self.LOG)
        self.assertEqual([t["turn_id"] for t in turns], ["a", "b", "c", "d", "e"])
        self.assertEqual(turns[0]["audio_ms"], 4000.0)

    def test_p90_is_nearest_rank(self):
        self.assertEqual(report.p90([5]), 5)
        self.assertEqual(report.p90(list(range(1, 11))), 9)
        self.assertEqual(report.p90(list(range(1, 12))), 10)

    def test_ambient_splits_off_the_silence_hangover(self):
        turn = report.parse(self.LOG)[3]
        self.assertEqual(
            report.steps(turn),
            [
                ("end-of-speech hangover", 700),
                ("upload + STT", 2000),
                ("transcript -> dispatch", 100),
                ("dispatch -> ui_output", 2000),
                ("ui_output -> first audio", 600),
                ("total", 5400),
            ],
        )

    def test_speak_start_splits_the_wait_for_first_audio(self):
        rows = dict(report.steps(report.parse(self.LOG)[4]))
        self.assertEqual(rows["ui_output -> first audio"], 2000)
        self.assertEqual(rows["  queue wait"], 500)
        self.assertEqual(rows["  setup + 1st sentence"], 1500)

    def test_old_lines_without_speak_start_have_no_split(self):
        self.assertNotIn("  queue wait", dict(report.steps(report.parse(self.LOG)[0])))

    def test_ptt_table_excludes_no_reply_turns_from_the_maths(self):
        table = report.report(report.parse(self.LOG), "ptt")

        self.assertIn("3 spoken, 1 no reply", table)
        # median of 1000, 2000, 1000; shares 25 %, 33 %, 20 % -> median 25 %.
        self.assertIn("end-of-speech + STT: median 1000 ms, 25% of total", table)
        self.assertRegex(table, r"queue wait\s+500\s+500")
        self.assertNotIn("hangover", table)


if __name__ == "__main__":
    unittest.main()
