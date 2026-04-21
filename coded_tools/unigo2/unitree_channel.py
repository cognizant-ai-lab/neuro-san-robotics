import os
from pathlib import Path
from typing import Any, Dict, Optional


_UNITREE_CHANNEL_STATE: Dict[str, Any] = {
    "modules": {},
}

_REPO_ROOT = Path(__file__).resolve().parents[2]
_CYCLONEDDS_CONFIG = _REPO_ROOT / "cyclonedds.xml"


def resolve_unitree_interface(default_ifname: Optional[str] = None) -> Optional[str]:
    """Resolve the preferred network interface for Unitree SDK access."""
    return (
        default_ifname
        or os.environ.get("VISION_CAMERA_INTERFACE")
        or os.environ.get("GO2_CAMERA_INTERFACE")
        or os.environ.get("CYCLONEDDS_NETWORK_INTERFACE")
        or os.environ.get("IFNAME")
    )


def _ensure_unitree_environment(ifname: Optional[str]) -> None:
    """Populate Unitree/CycloneDDS env vars with repo-local defaults."""
    if ifname:
        os.environ.setdefault("CYCLONEDDS_NETWORK_INTERFACE", ifname)

    if _CYCLONEDDS_CONFIG.exists():
        os.environ.setdefault("CYCLONEDDS_URI", _CYCLONEDDS_CONFIG.resolve().as_uri())


def initialize_unitree_channel(channel_factory_initialize, ifname: Optional[str] = None) -> Dict[str, Any]:
    """
    Initialize the Unitree DDS channel once per process and reuse it afterward.

    Vision and motion can both rely on the Unitree SDK. Reinitializing the
    CycloneDDS channel with different parameters in the same process can fail,
    so we keep a shared process-wide state here.
    """
    channel_ifname = resolve_unitree_interface(ifname)
    _ensure_unitree_environment(channel_ifname)
    module_key = (
        f"{getattr(channel_factory_initialize, '__module__', 'unknown')}::"
        f"{getattr(channel_factory_initialize, '__qualname__', getattr(channel_factory_initialize, '__name__', 'callable'))}"
    )
    initialized_modules = _UNITREE_CHANNEL_STATE.setdefault("modules", {})

    if module_key in initialized_modules:
        existing_ifname = initialized_modules[module_key]
        if existing_ifname and channel_ifname and existing_ifname != channel_ifname:
            raise RuntimeError(
                "Unitree DDS channel already initialized on "
                f"{existing_ifname}; cannot switch to {channel_ifname}"
            )
        return {
            "initialized": True,
            "ifname": existing_ifname,
            "module_key": module_key,
        }

    if channel_ifname:
        channel_factory_initialize(0, channel_ifname)
    else:
        channel_factory_initialize(0)

    initialized_modules[module_key] = channel_ifname
    return {
        "initialized": True,
        "ifname": channel_ifname,
        "module_key": module_key,
    }


def reset_unitree_channel_state() -> None:
    """Reset shared channel state for tests."""
    _UNITREE_CHANNEL_STATE["modules"] = {}
