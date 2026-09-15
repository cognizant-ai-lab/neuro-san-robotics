#!/usr/bin/env python3
"""
Pre-fetch the offline speech-to-text weights.

faster-whisper downloads its own weights the first time it transcribes. That
is fine on a workstation and wrong on a robot: the download is a few hundred
megabytes and would land inside whichever request happened to need it first,
stalling that one for minutes. This pulls them at setup instead.

setmyenv.sh runs this by itself when GO2_STT_ENGINE="local", because declaring
that engine is the same as saying the robot depends on these weights.

Usage:
    python scripts/install_stt_model.py            # fetch if missing
    python scripts/install_stt_model.py --check    # report, change nothing
    python scripts/install_stt_model.py --model small
"""

import argparse
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def main() -> int:
    from coded_tools.unigo2 import local_stt

    parser = argparse.ArgumentParser(description="Pre-fetch the offline speech model")
    parser.add_argument("--model", default=None,
                        help="whisper size: tiny, base (default), small, medium")
    parser.add_argument("--check", action="store_true",
                        help="report whether the weights are cached, without fetching")
    args = parser.parse_args()

    if args.model:
        import os
        os.environ["GO2_STT_MODEL"] = args.model

    size = local_stt.model_size()

    try:
        import faster_whisper  # noqa: F401
    except ImportError:
        print("faster-whisper is not installed. Run: pip install -r requirements.txt",
              file=sys.stderr)
        return 1

    if local_stt.is_model_cached():
        print(f"Speech model whisper-{size}: already cached.")
        return 0

    if args.check:
        print(f"Speech model whisper-{size}: MISSING.")
        print("Fetch it with: python scripts/install_stt_model.py")
        return 1

    print(f"Fetching whisper-{size}. This happens once and takes a few minutes.")
    try:
        # Constructing the model is what triggers the download. Loading it here
        # rather than mid-request is the entire point of this script.
        local_stt.model()
    except Exception as error:  # noqa: BLE001 - surface whatever went wrong
        print(f"Failed: {error}", file=sys.stderr)
        return 1

    print(f"Speech model whisper-{size} ready on {local_stt.device()}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
