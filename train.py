#!/usr/bin/env python3
"""
Train OpenWakeWord model using Kokoro TTS synthetic voices + real recordings.

Usage:
    python train.py --wake-word "hey cal"
    python train.py --wake-word "okay jarvis" --samples-per-voice 300 --training-steps 75000

Docker:
    docker compose run --rm trainer python train.py --wake-word "hey cal" --data-dir /app/data
"""

import argparse
import io
import os
import shutil
import subprocess
import sys
import time
import uuid
import warnings
from pathlib import Path

import numpy as np
import requests
import scipy.io.wavfile
import yaml
from tqdm import tqdm

warnings.filterwarnings("ignore", message="Reached EOF prematurely")

WORK_DIR = Path(__file__).parent.resolve()
os.chdir(WORK_DIR)


def get_kokoro_voices(kokoro_url: str) -> list:
    """Get all available English voices from Kokoro."""
    try:
        r = requests.get(f"{kokoro_url}/v1/audio/voices", timeout=5)
        voices = r.json().get("voices", [])
        # Filter to English voices (a = American, b = British)
        voices = [v["id"] if isinstance(v, dict) else v for v in voices]
        english = [v for v in voices if v.startswith(('af_', 'am_', 'bf_', 'bm_'))]
        print(f"Kokoro voices available: {len(english)}")
        return english
    except Exception as e:
        print(f"ERROR: Cannot connect to Kokoro at {kokoro_url}: {e}")
        print("Make sure Kokoro is running:")
        print("  docker run -d --gpus all -p 8880:8880 ghcr.io/remsky/kokoro-fastapi-gpu:latest")
        sys.exit(1)


def generate_kokoro_sample(kokoro_url: str, voice: str, text: str, output_dir: Path) -> bool:
    """Generate a single Kokoro TTS sample with speed variation."""
    try:
        speed = np.random.uniform(0.7, 1.3)
        r = requests.post(
            f"{kokoro_url}/v1/audio/speech",
            json={
                "model": "kokoro",
                "voice": voice,
                "input": text,
                "response_format": "wav",
                "speed": speed
            },
            timeout=30
        )
        if r.status_code != 200:
            return False

        audio_data = io.BytesIO(r.content)
        sr, data = scipy.io.wavfile.read(audio_data)

        # Resample to 16kHz if needed
        if sr != 16000:
            from scipy.signal import resample
            num_samples = int(len(data) * 16000 / sr)
            data = resample(data, num_samples)
            data = np.clip(data, -32768, 32767).astype(np.int16)

        filename = f"kokoro_{uuid.uuid4().hex}.wav"
        scipy.io.wavfile.write(str(output_dir / filename), 16000, data)
        return True
    except Exception:
        return False


def generate_kokoro_samples(kokoro_url: str, voices: list, output_dir: Path,
                           samples_per_voice: int, texts: list, desc: str):
    """Generate Kokoro samples for all voices."""
    output_dir.mkdir(parents=True, exist_ok=True)
    total = samples_per_voice * len(voices)
    pbar = tqdm(total=total, desc=desc)
    success = 0

    for voice in voices:
        for i in range(samples_per_voice):
            text = texts[i % len(texts)]
            if generate_kokoro_sample(kokoro_url, voice, text, output_dir):
                success += 1
            pbar.update(1)
            if i % 20 == 0:
                time.sleep(0.05)

    pbar.close()
    print(f"  Generated {success}/{total} samples")
    return success


def copy_real_samples(wake_word: str, output_dir: Path, copies: int = 3) -> int:
    """Copy real voice recordings to training directory."""
    # Look for samples in my_real_samples/ directory
    real_samples_dir = WORK_DIR / "my_real_samples"
    if not real_samples_dir.exists():
        print("  No real samples found (run record_samples.py first)")
        return 0

    output_dir.mkdir(parents=True, exist_ok=True)
    count = 0

    for wav_file in real_samples_dir.glob("*.wav"):
        try:
            sr, data = scipy.io.wavfile.read(wav_file)
            if sr != 16000:
                from scipy.signal import resample
                num_samples = int(len(data) * 16000 / sr)
                data = resample(data, num_samples)
                data = np.clip(data, -32768, 32767).astype(np.int16)

            # Create multiple copies to weight real samples higher
            for i in range(copies):
                dest = output_dir / f"real_{i}_{wav_file.name}"
                scipy.io.wavfile.write(str(dest), 16000, data)
                count += 1
        except Exception as e:
            print(f"  Error processing {wav_file}: {e}")

    print(f"  Copied {count} real voice samples ({copies}x weight)")
    return count


def setup_training_dirs(wake_word: str) -> Path:
    """Set up training directory structure."""
    # Convert wake word to safe directory name
    safe_name = wake_word.replace(" ", "_").lower()
    base_dir = WORK_DIR / "my_custom_model" / safe_name

    if base_dir.exists():
        print("Clearing previous training outputs...")
        shutil.rmtree(base_dir)

    for subdir in ["positive_train", "positive_test", "negative_train", "negative_test"]:
        (base_dir / subdir).mkdir(parents=True, exist_ok=True)

    return base_dir


def create_config(wake_word: str, n_samples: int, training_steps: int,
                  layer_size: int, data_dir: str):
    """Create training configuration."""
    safe_name = wake_word.replace(" ", "_").lower()

    # Load default config from OpenWakeWord
    default_path = WORK_DIR / "openwakeword/examples/custom_model.yml"
    with open(default_path, 'r') as f:
        config = yaml.load(f.read(), yaml.Loader)

    config["target_phrase"] = [safe_name]
    config["model_name"] = safe_name
    config["n_samples"] = n_samples
    config["n_samples_val"] = max(1000, n_samples // 10)
    config["steps"] = training_steps
    config["layer_size"] = layer_size
    config["target_accuracy"] = 0.7
    config["target_recall"] = 0.5
    config["target_false_positives_per_hour"] = 0.1
    config["output_dir"] = "./my_custom_model"
    config["max_negative_weight"] = 2000
    config["rir_paths"] = [f'{data_dir}/mit_rirs']
    config["background_paths"] = [f'{data_dir}/audioset_16k', f'{data_dir}/fma']
    config["false_positive_validation_data_path"] = f"{data_dir}/validation_set_features.npy"
    config["feature_data_files"] = {"ACAV100M_sample": f"{data_dir}/openwakeword_features_ACAV100M_2000_hrs_16bit.npy"}
    config.pop("piper_sample_generator_path", None)  # We use Kokoro, not Piper

    config_path = WORK_DIR / "training_config.yaml"
    with open(config_path, 'w') as f:
        yaml.dump(config, f)

    print(f"Config saved: {config_path}")
    return config


def run_augmentation():
    """Run OpenWakeWord augmentation pipeline."""
    print("\n" + "=" * 60)
    print("Running augmentation pipeline...")
    print("=" * 60)

    train_script = str(WORK_DIR / "openwakeword/openwakeword/train.py")
    subprocess.run([
        sys.executable, train_script,
        "--training_config", "training_config.yaml",
        "--augment_clips"
    ], check=True)


def run_training():
    """Run OpenWakeWord model training."""
    print("\n" + "=" * 60)
    print("Training model...")
    print("=" * 60)

    train_script = str(WORK_DIR / "openwakeword/openwakeword/train.py")
    result = subprocess.run([
        sys.executable, train_script,
        "--training_config", "training_config.yaml",
        "--train_model"
    ])
    return result.returncode


def main():
    parser = argparse.ArgumentParser(description="Train a custom OpenWakeWord model")
    parser.add_argument("--wake-word", default="hey cal", help="Wake word/phrase to train")
    parser.add_argument("--samples-per-voice", type=int, default=200, help="Samples per Kokoro voice")
    parser.add_argument("--training-steps", type=int, default=50000, help="Number of training steps")
    parser.add_argument("--layer-size", type=int, default=64, choices=[32, 64, 128], help="Network layer size")
    parser.add_argument("--kokoro-url", default=os.environ.get("KOKORO_URL", "http://localhost:8880"),
                        help="Kokoro TTS URL")
    parser.add_argument("--data-dir", default=".", help="Directory containing training data (features, audioset, fma, mit_rirs)")
    args = parser.parse_args()

    wake_word = args.wake_word
    safe_name = wake_word.replace(" ", "_").lower()

    print("=" * 60)
    print("OpenWakeWord Training")
    print("=" * 60)
    print(f"Wake word: {wake_word}")
    print(f"Samples per voice: {args.samples_per_voice}")
    print(f"Training steps: {args.training_steps}")
    print(f"Layer size: {args.layer_size}")
    print()

    # Get Kokoro voices
    kokoro_voices = get_kokoro_voices(args.kokoro_url)
    if not kokoro_voices:
        print("ERROR: No Kokoro voices available!")
        sys.exit(1)

    # Setup directories
    base_dir = setup_training_dirs(wake_word)
    pos_train = base_dir / "positive_train"
    pos_test = base_dir / "positive_test"
    neg_train = base_dir / "negative_train"
    neg_test = base_dir / "negative_test"

    # Text variations for positive samples
    positive_texts = [
        wake_word,
        wake_word.title(),
        wake_word.lower(),
        wake_word.upper(),
        f"{wake_word}!",
        f"{wake_word}.",
    ]

    # Negative phrases - ONLY clearly different words (not similar-sounding!)
    # Using similar-sounding phrases hurts model performance
    negative_phrases = [
        "hello", "hi there", "good morning", "excuse me", "okay",
        "hey google", "alexa", "hey jarvis", "computer",
    ]

    # === POSITIVE SAMPLES ===
    print("\n" + "=" * 60)
    print("Generating POSITIVE samples...")
    print("=" * 60)

    print("\n[Kokoro TTS]")
    generate_kokoro_samples(args.kokoro_url, kokoro_voices, pos_train,
                           args.samples_per_voice, positive_texts, "Kokoro positive train")
    generate_kokoro_samples(args.kokoro_url, kokoro_voices, pos_test,
                           args.samples_per_voice // 10, positive_texts, "Kokoro positive test")

    print("\n[Real Voice]")
    real_count = copy_real_samples(wake_word, pos_train)
    if real_count > 5:
        copy_real_samples(wake_word, pos_test)

    # === NEGATIVE SAMPLES ===
    print("\n" + "=" * 60)
    print("Generating NEGATIVE samples...")
    print("=" * 60)

    print("\n[Kokoro TTS]")
    generate_kokoro_samples(args.kokoro_url, kokoro_voices, neg_train,
                           args.samples_per_voice, negative_phrases, "Kokoro negative train")
    generate_kokoro_samples(args.kokoro_url, kokoro_voices, neg_test,
                           args.samples_per_voice // 10, negative_phrases, "Kokoro negative test")

    # === COUNT SAMPLES ===
    n_pos_train = len(list(pos_train.glob("*.wav")))
    n_pos_test = len(list(pos_test.glob("*.wav")))
    n_neg_train = len(list(neg_train.glob("*.wav")))
    n_neg_test = len(list(neg_test.glob("*.wav")))

    print("\n" + "=" * 60)
    print("Sample counts:")
    print(f"  Positive: {n_pos_train} train, {n_pos_test} test")
    print(f"  Negative: {n_neg_train} train, {n_neg_test} test")
    print("=" * 60)

    # Create config and run training
    create_config(wake_word, n_pos_train, args.training_steps, args.layer_size, args.data_dir)
    run_augmentation()
    run_training()

    # Done
    model_path = WORK_DIR / "my_custom_model" / f"{safe_name}.onnx"
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE!")
    print("=" * 60)
    if model_path.exists():
        size_kb = model_path.stat().st_size / 1024
        print(f"Model: {model_path} ({size_kb:.0f}KB)")
        print(f"\nTest with: python test_model.py --model {model_path}")
    else:
        print("WARNING: Model file not found!")


if __name__ == "__main__":
    main()
