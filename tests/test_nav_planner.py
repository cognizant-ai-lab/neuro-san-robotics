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
        ):
            self.assertEqual(await tool.async_invoke({"command": "status"}, {}), "status")
            self.assertEqual(
                await tool.async_invoke({"command": "destinations"}, {}),
                "destinations",
            )
            self.assertEqual(
                await tool.async_invoke({"command": "obstacles"}, {}),
                "obstacles",
            )


if __name__ == "__main__":
    unittest.main()
