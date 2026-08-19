#!/usr/bin/env python3
"""Find the USB audio card, then set its playback volume.

Generalises the manual incantation ``amixer -c 0 sset PCM 100%``, which is
fragile for two reasons on this hardware:

* The card index is not stable.  On the Jetson, card 0 is the HDA/HDMI
  controller and the USB speaker lands on card 2, so a hardcoded ``-c 0``
  targets the wrong device.  USB enumeration order can also shift the index
  across reboots or re-plugs.
* The control name is not stable either.  Card 0 exposes only ``IEC958``
  (HDMI passthrough, no volume), while the USB speaker exposes exactly one
  control named ``PCM``.  Other cards use ``Master`` or ``Speaker``.

So: discover the card, discover a control on it that actually has a playback
volume, then set that.  Use ``--list`` to inspect without changing anything.
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys

# Preferred mixer controls, best first.  Anything with a playback volume is
# accepted as a fallback if none of these are present.
PREFERRED_CONTROLS = ("PCM", "Master", "Speaker", "Headphone", "Digital", "Volume")


def _run(cmd: list[str]) -> tuple[int, str]:
    """Run a command, returning (returncode, stdout+stderr)."""
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=5, check=False)
    except (OSError, subprocess.SubprocessError) as exc:
        return 1, str(exc)
    return result.returncode, result.stdout + result.stderr


def find_cards(match: str | None = None) -> list[tuple[int, str]]:
    """List playback-capable cards as (index, description).

    Parses ``aplay -l`` so we only ever consider cards that can actually play
    audio.  With ``match``, keeps cards whose line contains it (case
    insensitive); otherwise keeps USB audio cards.
    """
    code, out = _run(["aplay", "-l"])
    if code != 0:
        return []

    needle = (match or "USB Audio").lower()
    seen: dict[int, str] = {}
    for line in out.splitlines():
        if not line.startswith("card "):
            continue
        if needle not in line.lower():
            continue
        # e.g. "card 2: UACDemoV10 [UACDemoV1.0], device 0: USB Audio [USB Audio]"
        found = re.match(r"card (\d+):\s*(\S+)", line)
        if found:
            seen.setdefault(int(found.group(1)), found.group(2))
    return sorted(seen.items())


def find_volume_control(card: int) -> str | None:
    """Return a mixer control on ``card`` that has a playback volume, or None."""
    code, out = _run(["amixer", "-c", str(card), "scontrols"])
    if code != 0:
        return None

    # "Simple mixer control 'PCM',0" -> "PCM"
    names = re.findall(r"Simple mixer control '([^']+)'", out)
    if not names:
        return None

    def has_playback_volume(name: str) -> bool:
        code, out = _run(["amixer", "-c", str(card), "sget", name])
        return code == 0 and "pvolume" in out

    for preferred in PREFERRED_CONTROLS:
        if preferred in names and has_playback_volume(preferred):
            return preferred
    for name in names:
        if has_playback_volume(name):
            return name
    return None


def current_volume(card: int, control: str) -> str:
    """Return a one-line summary of the control's current level."""
    code, out = _run(["amixer", "-c", str(card), "sget", control])
    if code != 0:
        return "unknown"
    levels = re.findall(r"\[(\d+%)\]", out)
    state = "on"
    if "[off]" in out:
        state = "off"
    return f"{levels[0] if levels else '?'} ({state})"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Find the USB audio card and set its playback volume.",
    )
    parser.add_argument(
        "volume",
        nargs="?",
        type=int,
        default=100,
        help="Volume percent 0-100 (default: 100)",
    )
    parser.add_argument(
        "--card",
        help="Card index, or substring to match in 'aplay -l' "
        "(default: auto-detect the USB audio card)",
    )
    parser.add_argument(
        "--control",
        help="Mixer control name (default: auto-detect one with a playback volume)",
    )
    parser.add_argument(
        "--unmute",
        action="store_true",
        help="Also unmute the control after setting the volume",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="Show detected cards and controls, then exit without changing anything",
    )
    args = parser.parse_args()

    for tool in ("aplay", "amixer"):
        if shutil.which(tool) is None:
            print(f"error: {tool} not found on PATH (install alsa-utils)", file=sys.stderr)
            return 1

    # --- 1. Find the card ---
    if args.card is not None and args.card.isdigit():
        card = int(args.card)
        cards = [(card, "(explicit)")]
    else:
        cards = find_cards(args.card)
        if not cards:
            what = args.card or "USB Audio"
            print(f"error: no playback card matching {what!r} in 'aplay -l'", file=sys.stderr)
            print("       is the speaker plugged in? try: aplay -l", file=sys.stderr)
            return 1
        if len(cards) > 1:
            print(f"note: {len(cards)} matching cards; using the first", file=sys.stderr)
        card = cards[0][0]

    print(f"card    : {card} ({cards[0][1]})")

    # --- 2. Find the control ---
    control = args.control or find_volume_control(card)
    if control is None:
        print(f"error: no control with a playback volume on card {card}", file=sys.stderr)
        print(f"       inspect with: amixer -c {card} scontrols", file=sys.stderr)
        return 1
    print(f"control : {control}")
    print(f"before  : {current_volume(card, control)}")

    if args.list:
        return 0

    # --- 3. Set the volume ---
    volume = max(0, min(100, args.volume))
    if volume != args.volume:
        print(f"note: clamped {args.volume} to {volume}", file=sys.stderr)

    code, out = _run(["amixer", "-c", str(card), "sset", control, f"{volume}%"])
    if code != 0:
        print(f"error: amixer failed: {out.strip()}", file=sys.stderr)
        return 1

    if args.unmute:
        # Not every control is mutable; ignore failures here.
        _run(["amixer", "-c", str(card), "sset", control, "unmute"])

    print(f"after   : {current_volume(card, control)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
