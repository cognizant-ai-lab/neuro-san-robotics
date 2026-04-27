
# Copyright (C) 2023-2025 Cognizant Digital Business, Evolutionary AI.
# All Rights Reserved.
# Issued under the Academic Public License.
#
# You can be released from the terms, and requirements of the Academic Public
# License by purchasing a commercial license.
# Purchase of a commercial license is mandatory for any use of the
# neuro-san SDK Software in commercial settings.
#
# END COPYRIGHT

"""
NavStatusTool - Neuro SAN CodedTool for querying navigation state.

Read-only tool that reports the robot's navigation status, available
destinations, and obstacle information to the conscious agent.
"""

import logging
from typing import Any, Dict

from neuro_san.interfaces.coded_tool import CodedTool

logger = logging.getLogger(__name__)


class NavStatusTool(CodedTool):
    """
    CodedTool wrapper for navigation status queries.

    Queries:
    - status: Current navigation state and progress
    - destinations: List of known map destinations
    - obstacles: Nearby obstacle information from depth camera
    """

    async def async_invoke(self, args: Dict[str, Any], sly_data: Dict[str, Any]) -> Any:
        """Return navigation status information.

        Args:
            args: Optional 'query' key: 'status' (default), 'destinations', or 'obstacles'.
            sly_data: Neuro SAN inter-agent context (unused by this tool).

        Returns:
            Human-readable string with the requested navigation information.
        """
        from coded_tools.unigo2.nav_core import NavCore

        query = args.get("query", "status").lower().strip()
        nav = NavCore.get_instance()

        if query == "status":
            return nav.get_status_summary()

        elif query == "destinations":
            return nav.list_destinations()

        elif query == "obstacles":
            return nav.get_obstacle_summary()

        return nav.get_status_summary()
