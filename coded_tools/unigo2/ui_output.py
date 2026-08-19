"""Agent-owned presentation tool for the robot's browser and speaker."""

from __future__ import annotations

import asyncio
from typing import Any
from typing import Dict

from neuro_san.interfaces.coded_tool import CodedTool

from coded_tools.unigo2.agent_events import publish_ui_output


class UiOutputTool(CodedTool):
    """Publish only explicit agent thoughts and speech to the presentation layer."""

    async def async_invoke(self, args: Dict[str, Any], sly_data: Dict[str, Any]) -> Any:
        thought = args.get("thought", "")
        say = args.get("say", "")
        heard = args.get("heard", "")
        if (
            not isinstance(thought, str)
            or not isinstance(say, str)
            or not isinstance(heard, str)
        ):
            return "thought, say, and heard must be strings. End this event turn now."
        if not thought.strip() and not say.strip():
            return (
                "Silence selected. End this event turn now without calling "
                "ui_output or any other tool again."
            )

        delivered = await asyncio.to_thread(
            publish_ui_output,
            thought=thought,
            say=say,
            heard=heard,
        )
        if delivered:
            return "Output delivered. End this event turn now; do not call ui_output again."
        return (
            "The presentation adapter is unavailable. End this event turn now; "
            "do not retry ui_output."
        )
