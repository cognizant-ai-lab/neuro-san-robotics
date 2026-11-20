#!/usr/bin/env python3
"""
Fetch the Piper voice model the offline TTS fallback needs.

`pip install -r requirements.txt` gets the piper binary, but not the voice it
speaks with. Without the two files below, `_is_piper_available()` in
coded_tools/unigo2/tts_go2.py returns False and the robot falls all the way
through to espeak-ng, which sounds markedly worse. Nothing else in the repo
fetches these, so a fresh robot has no working neural fallback until this runs.

That fallback matters most where there is no hosted TTS to fall back *from* --
an Azure region with no gpt-4o-mini-tts deployment, or a robot off the network.

Usage:
    python scripts/install_piper_voice.py           # install if missing
    python scripts/install_piper_voice.py --check   # report status, change nothing
    python scripts/install_piper_voice.py --force   # re-download over existing files
    python scripts/install_piper_voice.py --voice en_GB-southern_english_female-low

Which voice, and where it goes, come from coded_tools/unigo2/piper_voice.py:
GO2_PIPER_VOICE names it, GO2_PIPER_DIR places it, and GO2_PIPER_MODEL /
GO2_PIPER_CONFIG pin exact paths. tts_go2.py reads the same module, so the
installer always writes where the robot will look.
"""

import argparse
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from coded_tools.unigo2 import piper_voice  # noqa: E402

# The .onnx is tens of MB at the smallest tier; anything tiny is an error page,
# not a model.
MIN_MODEL_BYTES = 1_000_000


def target_paths() -> tuple[Path, Path]:
    """Return where the model and its config belong, honouring the env vars."""
    return Path(piper_voice.model_path()), Path(piper_voice.config_path())


def downloads() -> list[tuple[str, Path]]:
    """The (url, destination) pairs that install the configured voice."""
    name = piper_voice.voice_name()
    model, config = target_paths()
    return [
        (piper_voice.download_url(name, "onnx"), model),
        (piper_voice.download_url(name, "onnx.json"), config),
    ]


def _download(url: str, destination: Path) -> None:
    """Download to a temporary sibling, then move it into place."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".partial")
    print(f"  downloading {url}")
    try:
        with urllib.request.urlopen(url, timeout=120) as response:  # nosec B310
            partial.write_bytes(response.read())
    except urllib.error.URLError as error:
        partial.unlink(missing_ok=True)
        raise SystemExit(f"  failed: {error}") from error

    # A truncated or HTML error body would otherwise sit there looking installed
    # and fail much later, inside a TTS call, as an opaque piper crash.
    if destination.suffix == ".onnx" and partial.stat().st_size < MIN_MODEL_BYTES:
        size = partial.stat().st_size
        partial.unlink(missing_ok=True)
        raise SystemExit(f"  failed: got {size} bytes, too small to be the model")

    partial.replace(destination)
    print(f"  wrote {destination} ({destination.stat().st_size:,} bytes)")


def report(model: Path, config: Path) -> bool:
    """Print what is present and return whether Piper can run."""
    ready = True
    for path in (model, config):
        if path.is_file():
            print(f"  present: {path} ({path.stat().st_size:,} bytes)")
        else:
            print(f"  MISSING: {path}")
            ready = False
    if ready:
        print(f"  sample rate: {piper_voice.sample_rate(str(config))} Hz")
    return ready


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--voice", default=None,
                        help="Piper voice name, e.g. en_GB-cori-high "
                             "(default: GO2_PIPER_VOICE, else en_GB-cori-high)")
    parser.add_argument("--check", action="store_true",
                        help="report status and exit without downloading")
    parser.add_argument("--force", action="store_true",
                        help="re-download even if the files are already present")
    args = parser.parse_args()

    if args.voice:
        os.environ["GO2_PIPER_VOICE"] = args.voice

    model, config = target_paths()
    print(f"Piper voice: {piper_voice.voice_name()}")

    if args.check:
        ready = report(model, config)
        print("\nOffline neural TTS is available."
              if ready else
              "\nOffline TTS would fall through to espeak-ng. Run without --check.")
        return 0 if ready else 1

    if not args.force and model.is_file() and config.is_file():
        report(model, config)
        print("\nAlready installed. Use --force to re-download.")
        return 0

    for url, destination in downloads():
        _download(url, destination)

    print("\nInstalled. The robot now falls back to Piper rather than espeak-ng.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
