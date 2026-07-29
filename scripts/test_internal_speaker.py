#!/usr/bin/env python3
"""Probe whether Jetson audio can reach the Go2's internal speaker.

This is a hardware diagnostic, not an alternative TTS backend.  It asks the
Go2 VUI service to enable audio/set its volume, then plays a short WAV tone to
an ALSA device.  The VUI API does not itself accept arbitrary audio.
"""

from __future__ import annotations

import argparse
import math
import shutil
import struct
import subprocess
import tempfile
import wave
from pathlib import Path


def _import_unitree_vui():
    errors: list[Exception] = []
    for root in ("unitree_sdk2_python.unitree_sdk2py", "unitree_sdk2py"):
        try:
            channel_module = __import__(f"{root}.core.channel", fromlist=["ChannelFactoryInitialize"])
            vui_module = __import__(f"{root}.go2.vui.vui_client", fromlist=["VuiClient"])
            return channel_module.ChannelFactoryInitialize, vui_module.VuiClient
        except (ImportError, AttributeError) as exc:
            errors.append(exc)
    raise RuntimeError(
        "Unitree SDK2 Python with Go2 VUI support is not importable: "
        + "; ".join(str(error) for error in errors)
    )


def _configure_robot_speaker(interface: str | None, volume: int) -> tuple[object, int | None]:
    channel_factory_initialize, vui_client_type = _import_unitree_vui()
    if interface:
        channel_factory_initialize(0, interface)
    else:
        channel_factory_initialize(0)

    client = vui_client_type()
    client.SetTimeout(3.0)
    client.Init()

    code, old_volume = client.GetVolume()
    if code != 0:
        raise RuntimeError(f"Go2 VUI GetVolume failed with code {code}")

    switch_code = client.SetSwitch(1)
    if switch_code != 0:
        raise RuntimeError(f"Go2 VUI SetSwitch failed with code {switch_code}")

    volume_code = client.SetVolume(volume)
    if volume_code != 0:
        raise RuntimeError(f"Go2 VUI SetVolume failed with code {volume_code}")

    print(f"Go2 VUI connected; internal volume changed from {old_volume} to {volume}.")
    return client, old_volume


def _write_test_tone(path: Path, duration: float = 1.5) -> None:
    sample_rate = 48_000
    frequency = 660.0
    amplitude = 8_000
    sample_count = int(sample_rate * duration)
    with wave.open(str(path), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(sample_rate)
        frames = bytearray()
        for index in range(sample_count):
            sample = int(amplitude * math.sin(2 * math.pi * frequency * index / sample_rate))
            frames.extend(struct.pack("<h", sample))
        output.writeframes(frames)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--interface",
        help="Robot network interface for Unitree DDS, such as eth0 (default: SDK default)",
    )
    parser.add_argument(
        "--device",
        default="plughw:CARD=APE,DEV=0",
        help="ALSA playback device (default: plughw:CARD=APE,DEV=0)",
    )
    parser.add_argument(
        "--robot-volume",
        type=int,
        default=5,
        choices=range(1, 11),
        metavar="1-10",
        help="Temporary Go2 VUI volume (default: 5)",
    )
    parser.add_argument(
        "--skip-vui",
        action="store_true",
        help="Play the ALSA tone without contacting the Go2 VUI service",
    )
    parser.add_argument(
        "--wav",
        type=Path,
        help="Play this WAV file instead of generating a test tone",
    )
    args = parser.parse_args()

    if not shutil.which("aplay"):
        parser.error("aplay is not installed")

    client = None
    old_volume = None
    try:
        if not args.skip_vui:
            client, old_volume = _configure_robot_speaker(args.interface, args.robot_volume)

        with tempfile.TemporaryDirectory(prefix="go2-speaker-test-") as temp_dir:
            if args.wav:
                tone_path = args.wav.expanduser().resolve()
                if not tone_path.is_file():
                    parser.error(f"WAV file does not exist: {tone_path}")
                print(f"Playing {tone_path} through {args.device} ...")
            else:
                tone_path = Path(temp_dir) / "test-tone.wav"
                _write_test_tone(tone_path)
                print(f"Playing a 1.5-second test tone through {args.device} ...")
            result = subprocess.run(
                ["aplay", "-D", args.device, str(tone_path)],
                check=False,
            )
            if result.returncode != 0:
                print(f"aplay failed with exit code {result.returncode}.")
                return result.returncode

        print("aplay accepted the audio. Did the Go2's internal speaker emit the tone?")
        print("Note: success from aplay only proves the APE input accepted samples; it may not have a physical route.")
        return 0
    finally:
        if client is not None and old_volume is not None:
            code = client.SetVolume(old_volume)
            print(f"Restored Go2 VUI volume to {old_volume} (code {code}).")


if __name__ == "__main__":
    raise SystemExit(main())
