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
