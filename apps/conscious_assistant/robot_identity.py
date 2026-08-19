"""Per-robot identity shared by the web UI and the transcription prompt.

Each robot overrides ROBOT_NAME and ROBOT_HOME from its own setmyenv.sh so one
checkout can serve CAIL-E in San Francisco, bit2 at another lab, and so on. The
defaults below keep the original CAIL-E unit working with no environment set.

Keep these values in step with the robot_name and robot_home keys in
registries/conscious_agent.hocon, which the agent persona resolves directly
through pyhocon rather than through this module.
"""

import os

DEFAULT_ROBOT_NAME = "CAIL-E"
DEFAULT_ROBOT_HOME = "the Cognizant AI Lab (CAIL) in San Francisco"


def _identity(variable: str, fallback: str) -> str:
    """Read an identity value from the environment, ignoring blank overrides."""
    return (os.environ.get(variable) or "").strip() or fallback


def robot_name() -> str:
    """Name this robot answers to, used in the persona, the UI, and the recogniser."""
    return _identity("ROBOT_NAME", DEFAULT_ROBOT_NAME)


def robot_home() -> str:
    """Human readable home lab for this robot."""
    return _identity("ROBOT_HOME", DEFAULT_ROBOT_HOME)
