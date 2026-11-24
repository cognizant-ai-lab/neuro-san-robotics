from pathlib import Path
from typing import Any
from typing import Dict

from neuro_san.interfaces.coded_tool import CodedTool

WORKSHOP_FILE = Path(__file__).with_name("uniper_ai_envision_workshop_2026-06.md")


class WorkshopKnowledgeTool(CodedTool):
    """Returns the full narrative of the Uniper AI Envision Workshop (June 2026)."""

    async def async_invoke(self, args: Dict[str, Any], sly_data: Dict[str, Any]) -> str:
        # shortcut: returns the whole ~10k-token file; switch to retrieval if more documents are added.
        return WORKSHOP_FILE.read_text(encoding="utf-8")
