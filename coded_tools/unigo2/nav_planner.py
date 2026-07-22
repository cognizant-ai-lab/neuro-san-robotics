
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
NavPlannerTool - Neuro SAN CodedTool for navigation commands and queries.

This is the single agent interface to NavCore. It sets goals and answers queries
but does NOT participate in real-time navigation decisions.

Once a goal is set, the NavCore background loop (10 Hz) handles all
obstacle avoidance and path following autonomously.
"""

import asyncio
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
    - set_location: Anchor the internal nav pose to a known map location
    - move_forward: Move forward with depth watchdog protection
    - move_until_obstacle: Move forward until an obstacle is near
    - turn: Rotate by a specified angle in degrees
    - stop: Cancel current navigation
    - status: Get navigation state summary
    - destinations: List known map destinations
    - obstacles: Describe current obstacle sensing
    """

    async def async_invoke(self, args: Dict[str, Any], sly_data: Dict[str, Any]) -> Any:
        """Dispatch a navigation command to NavCore.

        Args:
            args: Must contain 'command'. Optional: 'target' (destination, location,
                  or direction), 'distance' (meters or degrees depending on command),
                  and 'heading_degrees' for set_location.
            sly_data: Neuro SAN inter-agent context (unused by this tool).

        Returns:
            Human-readable status string for the conscious agent.
        """
        from coded_tools.unigo2.agent_events import queue_agent_event
        from coded_tools.unigo2.nav_core import NavCore, NavState

        NavCore.set_status_callback(
            lambda message: queue_agent_event(message, source="navigation"),
        )

        command = args.get("command", "").lower().strip()
        if not command:
            return (
                "Please specify a navigation command: navigate_to, set_location, "
                "move_forward, turn, stop, or status."
            )

        nav = NavCore.get_instance()

        if command == "navigate_to":
            target = args.get("target", "").strip()
            if not target:
                return "Please specify a destination. " + nav.list_destinations()
            success = nav.navigate_to(target)
            if success:
                status = nav.get_status_summary()
                if nav.state == NavState.IDLE and "Already at " in status:
                    return f"No movement needed for '{target}'."
                return f"Started toward '{target}'."

            status = nav.get_status_summary()
            if nav.state == NavState.E_STOP:
                return f"I could not start navigating to '{target}'. {status}"
            return f"Navigation to '{target}' was not started. " + nav.list_destinations()

        elif command in {"set_location", "reset_location", "localize"}:
            target = args.get("target", "").strip()
            if not target:
                return "Please specify the current location. " + nav.list_destinations()

            heading_degrees = float(args.get("heading_degrees") or 0.0)
            success = nav.set_location(target, heading_rad=math.radians(heading_degrees))
            if success:
                return f"Location set to '{target}'."
            return f"Cannot set location to '{target}'. " + nav.list_destinations()

        elif command == "move_forward":
            distance = float(args.get("distance", 1.0))
            distance = max(0.1, min(distance, 10.0))
            return await asyncio.to_thread(
                nav.move_forward_guarded,
                max_distance_m=distance,
            )

        elif command == "move_until_obstacle":
            stop_distance = float(args.get("distance", nav.FORWARD_STOP_DISTANCE_M))
            stop_distance = max(0.3, min(stop_distance, 2.0))
            return await asyncio.to_thread(
                nav.move_forward_guarded,
                max_distance_m=None,
                stop_distance_m=stop_distance,
            )

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

            if not nav.turn(angle_rad):
                return f"Could not turn. {nav.get_status_summary()}"
            return f"Turning {math.degrees(angle_rad):.0f} degrees."

        elif command == "stop":
            nav.stop()
            return "Navigation stopped."

        elif command == "status":
            return nav.get_status_summary()

        elif command == "destinations":
            return nav.list_destinations()

        elif command == "obstacles":
            return nav.get_obstacle_summary()

        return (
            f"Unknown navigation command: '{command}'. Use: navigate_to, "
            "set_location, move_forward, move_until_obstacle, turn, stop, "
            "status, destinations, or obstacles."
        )
