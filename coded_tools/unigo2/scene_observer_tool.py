"""Expose the current camera scene to the native agent runtime."""

from __future__ import annotations

import asyncio
from typing import Any
from typing import Dict

from neuro_san.interfaces.coded_tool import CodedTool

from apps.conscious_assistant.scene_observer import SceneObserver
from coded_tools.unigo2.agent_events import publish_observation


_OBSERVER = SceneObserver()


class SceneObserverTool(CodedTool):
    """Capture one scene, retain its latest image, and return a concise observation."""

    async def async_invoke(self, args: Dict[str, Any], sly_data: Dict[str, Any]) -> Any:
        observation = await asyncio.to_thread(_OBSERVER.observe)
        if observation is None:
            return {"available": False, "summary": "No camera observation is available."}
        await asyncio.to_thread(publish_observation, observation)
        return {
            "available": True,
            "summary": observation.get("summary", "No detections"),
            "objects": observation.get("objects", []),
            "faces": observation.get("faces", []),
        }
