import unittest
from unittest.mock import MagicMock, patch

from coded_tools.unigo2.nav_planner import NavPlannerTool


class NavPlannerToolTests(unittest.IsolatedAsyncioTestCase):
    async def test_read_only_queries_share_the_navigation_tool(self):
        nav = MagicMock()
        nav.get_status_summary.return_value = "status"
        nav.list_destinations.return_value = "destinations"
        nav.get_obstacle_summary.return_value = "obstacles"
        tool = NavPlannerTool()

        with (
            patch("coded_tools.unigo2.nav_core.NavCore.get_instance", return_value=nav),
            patch("coded_tools.unigo2.nav_core.NavCore.set_status_callback"),
            patch(
                "coded_tools.unigo2.agent_events.publish_ui_output",
                return_value=True,
            ) as publish,
        ):
            result = await tool.async_invoke({"command": "status"}, {})
            self.assertIn("delivered directly", result)
            publish.assert_called_once_with(thought="status", say="status")
            self.assertEqual(
                await tool.async_invoke({"command": "destinations"}, {}),
                "destinations",
            )
            self.assertEqual(
                await tool.async_invoke({"command": "obstacles"}, {}),
                "obstacles",
            )

    async def test_status_can_be_queried_without_announcing(self):
        nav = MagicMock()
        nav.get_status_summary.return_value = "status"
        tool = NavPlannerTool()

        with (
            patch("coded_tools.unigo2.nav_core.NavCore.get_instance", return_value=nav),
            patch("coded_tools.unigo2.nav_core.NavCore.set_status_callback"),
            patch("coded_tools.unigo2.agent_events.publish_ui_output") as publish,
        ):
            result = await tool.async_invoke(
                {"command": "status", "announce": False},
                {},
            )

        self.assertEqual(result, "status")
        publish.assert_not_called()


if __name__ == "__main__":
    unittest.main()
