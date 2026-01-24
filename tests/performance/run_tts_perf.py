"""
Simple TTS Performance Test Runner

Run with:
    python tests/performance/run_tts_perf.py
"""

import json
import logging
import platform
import subprocess
import sys
from pathlib import Path
import os

# Add project root to Python path so we can import coded_tools
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from coded_tools.unigo2.tts_core import TtsCore
from tests.performance.perf_utils import run_perf_test, print_perf_table

try:
    from tqdm import tqdm
except ImportError:
    print("Installing tqdm...")
    subprocess.run([sys.executable, "-m", "pip", "install", "tqdm", "-q"], check=True)
    from tqdm import tqdm


# Setup logging - set to WARNING to suppress INFO logs during tqdm progress
logging.basicConfig(
    level=logging.WARNING,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


def get_system_info():
    """Collect system information for the test report."""
    info = {
        'platform': platform.system(),
        'platform_version': platform.version(),
        'machine': platform.machine(),
        'processor': platform.processor(),
        'python_version': platform.python_version(),
        'hostname': platform.node(),
    }

    # Try to get CPU count
    try:
        info['cpu_count'] = os.cpu_count()
    except Exception:
        pass

    # macOS specific info
    if platform.system() == "Darwin":
        try:
            # Get macOS version
            result = subprocess.run(
                ["sw_vers", "-productVersion"],
                capture_output=True,
                text=True,
                timeout=2
            )
            if result.returncode == 0:
                info['os_version'] = f"macOS {result.stdout.strip()}"

            # Get CPU model
            result = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True,
                text=True,
                timeout=2
            )
            if result.returncode == 0:
                info['cpu_model'] = result.stdout.strip()
        except Exception:
            pass

    # Linux specific info
    elif platform.system() == "Linux":
        try:
            # Get CPU model from /proc/cpuinfo
            with open('/proc/cpuinfo', 'r') as f:
                for line in f:
                    if 'model name' in line:
                        info['cpu_model'] = line.split(':')[1].strip()
                        break
        except Exception:
            pass

        try:
            # Get Linux distribution info
            result = subprocess.run(
                ["lsb_release", "-d"],
                capture_output=True,
                text=True,
                timeout=2
            )
            if result.returncode == 0:
                info['os_version'] = result.stdout.split(':')[1].strip()
        except Exception:
            pass

    return info


def save_volume():
    """Save current system volume."""
    system = platform.system()

    if system == "Darwin":  # macOS
        try:
            result = subprocess.run(
                ["osascript", "-e", "output volume of (get volume settings)"],
                capture_output=True,
                text=True,
                timeout=2
            )
            if result.returncode == 0:
                return int(result.stdout.strip())
        except Exception:
            pass
    elif system == "Linux":  # Unitree
        try:
            result = subprocess.run(
                ["amixer", "-c", "0", "sget", "PCM"],
                capture_output=True,
                text=True,
                timeout=2
            )
            if result.returncode == 0:
                import re
                match = re.search(r'\[(\d+)%\]', result.stdout)
                if match:
                    return int(match.group(1))
        except Exception:
            pass

    return None


def restore_volume(original_volume):
    """Restore system volume."""
    if original_volume is None:
        return

    system = platform.system()

    if system == "Darwin":
        try:
            subprocess.run(
                ["osascript", "-e", f"set volume output volume {original_volume}"],
                timeout=2,
                check=False
            )
            print(f"Restored volume to {original_volume}%")
        except Exception as e:
            print(f"Warning: Failed to restore volume: {e}")

    elif system == "Linux":
        try:
            subprocess.run(
                ["amixer", "-c", "0", "sset", "PCM", f"{original_volume}%"],
                timeout=2,
                check=False
            )
            print(f"Restored volume to {original_volume}%")
        except Exception as e:
            print(f"Warning: Failed to restore volume: {e}")


def main():
    """Run TTS performance tests."""

    # Load configuration
    fixtures_dir = Path(__file__).parent.parent.parent / "fixtures"

    with open(fixtures_dir / "perf_config.json") as f:
        config = json.load(f)["test_config"]

    with open(fixtures_dir / "text_samples.json") as f:
        samples = json.load(f)["tts_samples"]

    rounds = config["rounds"]

    # Get system info
    sys_info = get_system_info()

    # Print header with system information
    print("=" * 80)
    print("TTS Performance Tests")
    print("=" * 80)
    print("SYSTEM INFORMATION:")
    print(f"  OS: {sys_info.get('os_version', sys_info['platform'])}")
    if 'cpu_model' in sys_info:
        print(f"  CPU: {sys_info['cpu_model']}")
    if 'cpu_count' in sys_info:
        print(f"  CPU Cores: {sys_info['cpu_count']}")
    print(f"  Architecture: {sys_info['machine']}")
    print(f"  Hostname: {sys_info['hostname']}")
    print(f"  Python: {sys_info['python_version']}")
    print()
    print("TEST CONFIGURATION:")
    print(f"  Test rounds per sample: {rounds}")
    print(f"  Number of samples: {len(samples)}")
    print(f"  Volume: 0% (silent mode)")
    print("=" * 80)

    # Save current volume
    original_volume = save_volume()
    if original_volume is not None:
        print(f"Saved current volume: {original_volume}%")

    try:
        results = []

        # Create persistent TTS engine (model loaded once)
        print("Initializing TTS engine...")
        with TtsCore() as engine:
            print("✓ Engine ready\n")

            # Run tests with progress bar
            for sample in tqdm(samples, desc="Testing samples", unit="sample"):
                sample_name = sample['name']
                text = sample['text']
                word_count = len(text.split())

                # Run performance test with progress bar for rounds
                # Using persistent engine - no model reloading!
                stats = run_perf_test(
                    engine.say,
                    rounds=rounds,
                    text=text,
                    rate=150,
                    volume=0.0  # Silent
                )

                stats['name'] = f"{sample_name} ({word_count} words)"
                results.append(stats)

        # Display final results
        print_perf_table(results)

    finally:
        # Restore original volume
        restore_volume(original_volume)

        # Clean up lock file
        try:
            lock_file = Path("/tmp/go2_tts_engine.lock")
            if lock_file.exists():
                lock_file.unlink()
        except Exception:
            pass

    print("\n✓ All tests completed!")
    sys.stdout.flush()
    sys.stderr.flush()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nTests interrupted by user.")
        sys.stdout.flush()
        sys.exit(1)
    except Exception as e:
        print(f"\n\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        sys.exit(1)

    sys.exit(0)
