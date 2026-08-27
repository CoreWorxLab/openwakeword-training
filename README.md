# OpenWakeWord Trainer

Train custom wake word models for [OpenWakeWord](https://github.com/dscripka/openWakeWord) using synthetic voices from Kokoro TTS combined with your real voice recordings.

**Why this exists:** The official OpenWakeWord training process relies on Google Colab notebooks that frequently break. This repo provides a working local training pipeline that produces quality models.

## What You Get

- A trained `.onnx` wake word model (~400KB)
- Works with OpenWakeWord, Home Assistant, or any system that supports ONNX models
- Typical results: 70%+ accuracy, <2 false positives per hour

## Requirements

- **NVIDIA GPU** with CUDA (RTX 3060 12GB or better recommended)
- **Docker** with [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
- **~20GB disk space** for training data

## Quick Start (Docker)

Docker is the recommended approach - it handles all the dependency hell for you.

### 1. Clone

```bash
git clone https://github.com/CoreWorxLab/openwakeword-training.git
cd openwakeword-training
```

### 2. Download Training Data (~17GB, one-time)

```bash
docker compose build trainer
docker compose run --rm trainer ./setup-data.sh
```

### 3. Record Your Voice (Optional but Recommended)

Recording 20-50 samples of your actual voice significantly improves detection. This runs on your host machine (needs microphone access):

```bash
pip install pyaudio numpy scipy
python record_samples.py --wake-word "hey cal"
```

- Press ENTER, then wait for "SPEAK NOW!" before speaking
- Say your wake word naturally
- Vary your tone, speed, and distance from the mic
- Press 'q' to quit

The cue fires only after the microphone has opened and settled (0.6s warm-up). Speaking before it means the word onset is never captured.

**Multiple speakers:** put each person in their own subdirectory — `train.py` searches `my_real_samples/` recursively and flattens the path into the training filename, so speakers with identical clip names don't collide:

```
my_real_samples/
├── jay/    hey_seeree_0001.wav …
└── alex/   hey_seeree_0001.wav …
```

Loose files directly in `my_real_samples/` still work. Keeping speakers separate lets you re-record or drop one without guessing which clips were whose.

### 3b. Check Your Recordings (Optional)

Before committing to a multi-hour training run, verify your samples are usable:

```bash
python check_alignment.py my_real_samples/ --verbose
```

This reports how far each clip's speech sits from the end of OpenWakeWord's detection window, and flags clips with no detectable speech. Training trims silence automatically, so large trailing-silence numbers here are expected and fine — what matters is that speech is actually present and not clipped.

### 4. Train Your Model

```bash
docker compose run --rm trainer python train.py --wake-word "hey cal" --data-dir /app/data
```

Training takes 4-8 hours depending on GPU.

### 5. Test Your Model

Test on your host machine (needs microphone access):

```bash
pip install openwakeword pyaudio numpy
python test_model.py --model my_custom_model/hey_cal.onnx
```

Speak your wake word into the microphone and watch for detections.

## Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--wake-word` | "hey cal" | The wake word/phrase to detect |
| `--samples-per-voice` | 200 | Samples generated per Kokoro voice |
| `--training-steps` | 50000 | More steps = better but slower |
| `--layer-size` | 64 | Network size (32, 64, or 128) |
| `--kokoro-url` | http://localhost:8880 | Kokoro TTS endpoint |
| `--data-dir` | `.` | Training data directory (`/app/data` for Docker) |
| `--no-trim` | off | Skip silence trimming before augmentation (not recommended) |

## How It Works

1. **Sample Generation** - Creates ~13K positive samples using 67 Kokoro voices with speed variation (0.7-1.3x), plus your real recordings (weighted 3x)

2. **Negative Samples** - Generates samples of clearly different phrases ("hello", "hey siri", "alexa") to teach the model what NOT to detect

3. **Silence Trimming** - Strips leading/trailing silence from all samples so speech lands where the model expects it (see below)

4. **Augmentation** - OpenWakeWord adds noise, reverb, and mixing to simulate real-world conditions

5. **Training** - Neural network learns to distinguish your wake word from everything else

### Key Insight

**Don't use similar-sounding negatives.** Training on phrases like "hey call" or "hey carl" actually hurts performance. Use only clearly different phrases like "hello", "hey siri", "alexa".

### Why Silence Trimming Matters

OpenWakeWord's `create_fixed_size_clip` aligns the **end of the array** with the end of the detection window, not the end of the speech. Untrimmed, a tight ~1.3s Kokoro clip and a fixed 2s recording place their speech at completely different offsets — so the two halves of the positive set teach contradictory alignments, and the real recordings are weighted 3x. Trimming first collapses that difference. Both positives and negatives are trimmed, so clip length never becomes a cue the model can learn instead of the phrase.

## Output

```
my_custom_model/
├── hey_cal.onnx          # Your trained model - use this!
└── hey_cal/
    ├── positive_train/   # Generated training samples
    ├── positive_test/    # Test samples
    ├── negative_train/   # Negative training samples
    └── negative_test/    # Negative test samples
```

## Using Your Model

```python
from openwakeword.model import Model

model = Model(wakeword_models=["my_custom_model/hey_cal.onnx"])

# Process 16kHz mono audio frames
prediction = model.predict(audio_frame)
if prediction["hey_cal"] > 0.5:
    print("Wake word detected!")
```

## Manual Setup (No Docker)

If you prefer not to use Docker, you can set up the environment directly:

```bash
./setup.sh
source venv/bin/activate

# Start Kokoro TTS separately
docker run -d --gpus all -p 8880:8880 ghcr.io/remsky/kokoro-fastapi-gpu:latest

python train.py --wake-word "hey cal"
```

Note: This requires Python 3.10+ and working CUDA. The pinned dependency versions in `requirements.txt` can conflict with other Python packages on your system, which is why Docker is recommended.

## Troubleshooting

### "Reached EOF prematurely" warnings
Normal - Kokoro's WAV headers have a quirk but the audio data is fine.

### Low recall in training metrics
Training metrics use synthetic test samples. Real-world performance is usually better.

### Model not detecting wake word
- Ensure audio is 16kHz mono
- Model needs ~2 seconds of audio buffer to warm up
- Try lowering detection threshold (default 0.5)

### TFLite conversion error at end
Ignore - the ONNX model is saved successfully before this error.

## Credits

- [OpenWakeWord](https://github.com/dscripka/openWakeWord) by David Scripka
- [Kokoro TTS](https://github.com/remsky/Kokoro-FastAPI) for synthetic voice generation
- Training data from [ACAV100M](https://huggingface.co/datasets/davidscripka/openwakeword_features)

## License

MIT
