# Performance Testing

Simple performance tests for TTS components.

## Configuration

Edit `fixtures/perf_config.json` to set number of test rounds:

```json
{
  "test_config": {
    "rounds": 5
  }
}
```

## Running Tests

```bash
# Run TTS performance tests
python tests/performance/run_tts_perf.py
```

The script will:
- Show clean progress bars for samples and rounds
- Run at volume=0 (silent)
- Restore original volume after completion
- Display system information and results table
- Suppress logging to keep progress bars clean

## Output Example

```
================================================================================
TTS Performance Tests
================================================================================
SYSTEM INFORMATION:
  OS: macOS 15.2
  CPU: Apple M3 Max
  CPU Cores: 16
  Architecture: arm64
  Hostname: my-macbook.local
  Python: 3.12.10

TEST CONFIGURATION:
  Test rounds per sample: 5
  Number of samples: 5
  Volume: 0% (silent mode)
================================================================================
Saved current volume: 50%
Initializing TTS engine...
✓ Engine ready

Testing samples: 100%|████████████████████| 5/5 [00:02<00:00,  2.5 sample/s]

==========================================================================================
Test Name                      Min (s)      Max (s)      Mean (s)     Median (s)
==========================================================================================
short (2 words)                   0.0102       0.0154       0.0123       0.0121
medium (19 words)                 0.0457       0.0523       0.0489       0.0486
long (90 words)                   0.1235       0.1357       0.1286       0.1279
very_short (1 words)              0.0051       0.0068       0.0060       0.0059
command (7 words)                 0.0223       0.0289       0.0257       0.0251
==========================================================================================
Rounds per test: 5

Restored volume to 50%

✓ All tests completed!
```

## Adding New TTS Samples

Edit `fixtures/text_samples.json`:

```json
{
  "tts_samples": [
    {
      "name": "my_sample",
      "text": "Your text here",
      "description": "Description"
    }
  ]
}
```

## Dependencies

The script auto-installs `tqdm` if not present. No other dependencies needed.

## Performance Optimizations

The test suite uses a **persistent TTS engine** that loads the model once and reuses it for all tests. This provides:

- ✅ **Accurate measurements** - only TTS generation time, not model loading
- ✅ **Faster testing** - no model reload between samples
- ✅ **Real-world simulation** - matches how the robot actually uses TTS

On Jetson Orin with Piper TTS, this eliminates ~100-200ms of model loading overhead per call.

## Notes

- No pytest or test framework needed - just plain Python
- Logging is set to WARNING level to avoid breaking progress bars
- System volume is automatically preserved and restored
- Tests run with real TTS engine at volume=0 (silent mode)
- TTSEngine context manager ensures proper resource cleanup
