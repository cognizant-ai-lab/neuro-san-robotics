import os
from typing import Any, Dict, Optional


_UNITREE_CHANNEL_STATE: Dict[str, Any] = {
    "initialized": False,
    "ifname": None,
}


def resolve_unitree_interface(default_ifname: Optional[str] = None) -> Optional[str]:
    """Resolve the preferred network interface for Unitree SDK access."""
    return (
        default_ifname
        or os.environ.get("VISION_CAMERA_INTERFACE")
        or os.environ.get("GO2_CAMERA_INTERFACE")
        or os.environ.get("CYCLONEDDS_NETWORK_INTERFACE")
        or os.environ.get("IFNAME")
    )


def initialize_unitree_channel(channel_factory_initialize, ifname: Optional[str] = None) -> Dict[str, Any]:
    """
    Initialize the Unitree DDS channel once per process and reuse it afterward.

    Vision and motion can both rely on the Unitree SDK. Reinitializing the
    CycloneDDS channel with different parameters in the same process can fail,
    so we keep a shared process-wide state here.
    """
    channel_ifname = resolve_unitree_interface(ifname)

    if _UNITREE_CHANNEL_STATE["initialized"]:
        existing_ifname = _UNITREE_CHANNEL_STATE["ifname"]
        if existing_ifname and channel_ifname and existing_ifname != channel_ifname:
            raise RuntimeError(
                "Unitree DDS channel already initialized on "
                f"{existing_ifname}; cannot switch to {channel_ifname}"
            )
        return dict(_UNITREE_CHANNEL_STATE)

    if channel_ifname:
        channel_factory_initialize(0, channel_ifname)
    else:
        channel_factory_initialize(0)

    _UNITREE_CHANNEL_STATE["initialized"] = True
    _UNITREE_CHANNEL_STATE["ifname"] = channel_ifname
    return dict(_UNITREE_CHANNEL_STATE)


def reset_unitree_channel_state() -> None:
    """Reset shared channel state for tests."""
    _UNITREE_CHANNEL_STATE["initialized"] = False
    _UNITREE_CHANNEL_STATE["ifname"] = None
