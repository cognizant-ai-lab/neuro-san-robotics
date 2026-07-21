"""Agent-owned presentation tool for CAIL-E's browser and speaker."""

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
        if not isinstance(thought, str) or not isinstance(say, str):
            return "thought and say must be strings"
        if not thought.strip() and not say.strip():
            return "No output requested."

        delivered = await asyncio.to_thread(
            publish_ui_output,
            thought=thought,
            say=say,
        )
        return "Output delivered." if delivered else "The presentation adapter is unavailable."
