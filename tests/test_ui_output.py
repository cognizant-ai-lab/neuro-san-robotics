import unittest
from unittest.mock import patch

from coded_tools.unigo2.ui_output import UiOutputTool


class UiOutputToolTests(unittest.IsolatedAsyncioTestCase):
    async def test_empty_output_selects_silence_without_publishing(self):
        tool = UiOutputTool()

        with patch("coded_tools.unigo2.ui_output.publish_ui_output") as publish:
            result = await tool.async_invoke({"thought": "", "say": ""}, {})

        publish.assert_not_called()
        self.assertIn("Silence selected", result)
        self.assertIn("End this event turn now", result)

    async def test_delivered_output_tells_agent_to_end_turn(self):
        tool = UiOutputTool()

        with patch(
            "coded_tools.unigo2.ui_output.publish_ui_output",
            return_value=True,
        ) as publish:
            result = await tool.async_invoke(
                {"thought": "I reached the kitchen.", "say": ""},
                {},
            )

        publish.assert_called_once_with(
            thought="I reached the kitchen.",
            say="",
            heard="",
            epoch=None,
            turn_id=None,
        )
        self.assertIn("End this event turn now", result)
        self.assertIn("do not call ui_output again", result)

    async def test_the_turn_epoch_is_passed_back_with_the_output(self):
        tool = UiOutputTool()

        with patch(
            "coded_tools.unigo2.ui_output.publish_ui_output",
            return_value=True,
        ) as publish:
            await tool.async_invoke({"say": "Two plus two is four."}, {"speech_epoch": 7})

        self.assertEqual(publish.call_args.kwargs["epoch"], 7)

    async def test_observation_events_can_think_but_never_speak(self):
        tool = UiOutputTool()

        with patch(
            "coded_tools.unigo2.ui_output.publish_ui_output",
            return_value=True,
        ) as publish:
            await tool.async_invoke(
                {"thought": "A chair.", "say": "I see a chair!"},
                {"event_source": "observation"},
            )

        self.assertEqual(publish.call_args.kwargs["thought"], "A chair.")
        self.assertEqual(publish.call_args.kwargs["say"], "")

    async def test_observation_events_cannot_move_the_robot(self):
        from coded_tools.unigo2.nav_planner import NavPlannerTool
        from coded_tools.unigo2.robot_macros import RobotMacros

        observation = {"event_source": "observation"}
        with patch("coded_tools.unigo2.robot_macros.Go2Macros") as go2:
            result = await RobotMacros().async_invoke({"action": "dance"}, observation)
        go2.assert_not_called()
        self.assertIn("Refused", result)

        with (
            patch("coded_tools.unigo2.nav_core.NavCore.get_instance") as nav,
            patch("coded_tools.unigo2.nav_core.NavCore.set_status_callback"),
        ):
            result = await NavPlannerTool().async_invoke(
                {"command": "move_forward", "distance": 1}, observation
            )
        nav.assert_not_called()
        self.assertIn("Refused", result)


if __name__ == "__main__":
    unittest.main()
