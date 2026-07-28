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
        )
        self.assertIn("End this event turn now", result)
        self.assertIn("do not call ui_output again", result)


if __name__ == "__main__":
    unittest.main()
