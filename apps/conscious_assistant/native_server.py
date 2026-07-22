"""Start the local Neuro SAN runtime after the robot control client is ready."""

from __future__ import annotations

import logging
import site
import sys
from pathlib import Path


def _preinitialize_robot_control() -> None:
    """Keep Unitree DDS initialization ahead of the optional vision stack."""
    try:
        from coded_tools.unigo2.go2_macros import Go2Macros

        robot = Go2Macros()
        logging.info("Native agent runtime robot control available=%s", getattr(robot, "available", False))
    except Exception:
        logging.exception("Native agent runtime could not preinitialize robot control")


def _prime_vision_runtime() -> None:
    """Load Torch dependencies before agent tools can initialize the camera stack."""
    if not sys.platform.startswith("linux"):
        return

    try:
        import ctypes

        for site_dir in site.getsitepackages():
            library = Path(site_dir) / "torch" / "lib" / "libgomp.so.1"
            if library.exists():
                ctypes.CDLL(str(library), mode=getattr(ctypes, "RTLD_GLOBAL", 0))
                break
        from ultralytics import YOLO  # noqa: F401
        logging.info("Native agent runtime vision dependencies are ready")
    except Exception:
        logging.exception("Native agent runtime could not preinitialize vision dependencies")


def main() -> None:
    """Run Neuro SAN's service main loop with the normal command-line arguments."""
    _preinitialize_robot_control()
    _prime_vision_runtime()
    from apps.conscious_assistant.scene_observer_service import SceneObserverService
    from neuro_san.service.main_loop.server_main_loop import ServerMainLoop

    observer_service = SceneObserverService()
    observer_service.start()
    try:
        ServerMainLoop().main_loop()
    finally:
        observer_service.stop()


if __name__ == "__main__":
    main()
