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

Honours GO2_PIPER_MODEL and GO2_PIPER_CONFIG, so a robot that keeps its voices
somewhere other than the default path installs to the same place tts_go2.py
will look.
"""

import argparse
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path


VOICE = "en_GB-cori-high"

# Piper's voices are published on Hugging Face by the Rhasspy project.
BASE_URL = (
    "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_GB/cori/high"
)

# Defaults mirror tts_go2.py so both agree on where the voice lives.
DEFAULT_MODEL = f"/home/unitree/piper_models/{VOICE}.onnx"
DEFAULT_CONFIG = f"/home/unitree/piper_models/{VOICE}.onnx.json"

# The .onnx is around 110 MB; anything tiny is an error page, not a model.
MIN_MODEL_BYTES = 1_000_000


def target_paths() -> tuple[Path, Path]:
    """Return where the model and its config belong, honouring the env vars."""
    model = Path(os.environ.get("GO2_PIPER_MODEL", DEFAULT_MODEL))
    config = Path(os.environ.get("GO2_PIPER_CONFIG", DEFAULT_CONFIG))
    return model, config


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
    return ready


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--check", action="store_true",
                        help="report status and exit without downloading")
    parser.add_argument("--force", action="store_true",
                        help="re-download even if the files are already present")
    args = parser.parse_args()

    model, config = target_paths()
    print(f"Piper voice: {VOICE}")

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

    for url, destination in (
        (f"{BASE_URL}/{VOICE}.onnx", model),
        (f"{BASE_URL}/{VOICE}.onnx.json", config),
    ):
        _download(url, destination)

    print("\nInstalled. The robot now falls back to Piper rather than espeak-ng.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
