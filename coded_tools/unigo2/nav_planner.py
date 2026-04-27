
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
NavPlannerTool - Neuro SAN CodedTool for navigation commands.

This is a thin command interface to NavCore. It sets goals and queries status
but does NOT participate in real-time navigation decisions.

Once a goal is set, the NavCore background loop (10 Hz) handles all
obstacle avoidance and path following autonomously.
"""

import math
import logging
from typing import Any, Dict

from neuro_san.interfaces.coded_tool import CodedTool

logger = logging.getLogger(__name__)


class NavPlannerTool(CodedTool):
    """
    CodedTool wrapper for agent-initiated navigation commands.

    Supported commands:
    - navigate_to: Go to a named location on the topological map
    - move_forward: Move forward a specified distance in meters
    - turn: Rotate by a specified angle in degrees
    - stop: Cancel current navigation
    - status: Get navigation state summary
    """

    async def async_invoke(self, args: Dict[str, Any], sly_data: Dict[str, Any]) -> Any:
        """Dispatch a navigation command to NavCore.

        Args:
            args: Must contain 'command'. Optional: 'target' (destination/direction),
                  'distance' (meters or degrees depending on command).
            sly_data: Neuro SAN inter-agent context (unused by this tool).

        Returns:
            Human-readable status string for the conscious agent.
        """
        from coded_tools.unigo2.nav_core import NavCore

        command = args.get("command", "").lower().strip()
        if not command:
            return "Please specify a navigation command: navigate_to, move_forward, turn, stop, or status."

        nav = NavCore.get_instance()

        if command == "navigate_to":
            target = args.get("target", "").strip()
            if not target:
                return "Please specify a destination. " + nav.list_destinations()
            success = nav.navigate_to(target)
            if success:
                return f"Navigating to '{target}'. I'll let you know when I arrive."
            return f"Cannot navigate to '{target}'. " + nav.list_destinations()

        elif command == "move_forward":
            distance = float(args.get("distance", 1.0))
            distance = max(0.1, min(distance, 10.0))
            nav.move_relative(distance, 0.0)
            return f"Moving forward {distance:.1f} meters."

        elif command == "turn":
            target = args.get("target", "").lower()
            angle_deg = float(args.get("distance", 90.0))

            if target == "left":
                angle_rad = math.radians(abs(angle_deg))
            elif target == "right":
                angle_rad = -math.radians(abs(angle_deg))
            elif target == "around":
                angle_rad = math.pi
            else:
                angle_rad = math.radians(angle_deg)

            nav.turn(angle_rad)
            return f"Turning {math.degrees(angle_rad):.0f} degrees."

        elif command == "stop":
            nav.stop()
            return "Navigation stopped."

        elif command == "status":
            return nav.get_status_summary()

        return f"Unknown navigation command: '{command}'. Use: navigate_to, move_forward, turn, stop, or status."
