#!/bin/bash
# Download training data (~17GB) into data/ directory.
# Run once before training. Skips files that already exist.
#
# Docker:  docker compose run --rm trainer ./setup-data.sh
# Manual:  ./setup-data.sh

set -e

# Use /app/data inside container, ./data on host
DATA_DIR="${DATA_DIR:-./data}"
mkdir -p "$DATA_DIR"

echo "=== Downloading training data to $DATA_DIR ==="
echo ""

# Pre-computed features (~17GB)
if [ ! -f "$DATA_DIR/openwakeword_features_ACAV100M_2000_hrs_16bit.npy" ]; then
    echo "Downloading ACAV100M features (17GB - this takes a while)..."
    curl -L -o "$DATA_DIR/openwakeword_features_ACAV100M_2000_hrs_16bit.npy" \
        'https://huggingface.co/datasets/davidscripka/openwakeword_features/resolve/main/openwakeword_features_ACAV100M_2000_hrs_16bit.npy'
else
    echo "ACAV100M features already exist, skipping."
fi

# Validation features (~177MB)
if [ ! -f "$DATA_DIR/validation_set_features.npy" ]; then
    echo "Downloading validation features..."
    curl -L -o "$DATA_DIR/validation_set_features.npy" \
        'https://huggingface.co/datasets/davidscripka/openwakeword_features/resolve/main/validation_set_features.npy'
else
    echo "Validation features already exist, skipping."
fi

# MIT Room Impulse Responses
if [ ! -d "$DATA_DIR/mit_rirs" ]; then
    echo "Downloading room impulse responses..."
    git lfs install
    git clone https://huggingface.co/datasets/davidscripka/MIT_environmental_impulse_responses "$DATA_DIR/MIT_environmental_impulse_responses_tmp"

    mkdir -p "$DATA_DIR/mit_rirs"
    python3 << EOF
import datasets
import scipy.io.wavfile
import numpy as np
from pathlib import Path
from tqdm import tqdm

rir_dataset = datasets.Dataset.from_dict({
    "audio": [str(i) for i in Path("$DATA_DIR/MIT_environmental_impulse_responses_tmp/16khz").glob("*.wav")]
}).cast_column("audio", datasets.Audio())

for row in tqdm(rir_dataset, desc="Processing RIRs"):
    name = row['audio']['path'].split('/')[-1]
    scipy.io.wavfile.write(
        "$DATA_DIR/mit_rirs/" + name, 16000,
        (row['audio']['array'] * 32767).astype(np.int16)
    )
EOF
    rm -rf "$DATA_DIR/MIT_environmental_impulse_responses_tmp"
else
    echo "MIT RIRs already exist, skipping."
fi

# AudioSet background audio
if [ ! -d "$DATA_DIR/audioset_16k" ]; then
    echo "Downloading AudioSet background audio..."
    mkdir -p "$DATA_DIR/audioset" "$DATA_DIR/audioset_16k"
    curl -L -o "$DATA_DIR/audioset/bal_train09.tar" \
        'https://huggingface.co/datasets/agkphysics/AudioSet/resolve/728fffe80088808b49db5aa700f2e264c035c880/data/bal_train09.tar'
    tar -xf "$DATA_DIR/audioset/bal_train09.tar" -C "$DATA_DIR/audioset"

    python3 << EOF
import datasets
import scipy.io.wavfile
import numpy as np
from pathlib import Path
from tqdm import tqdm

audioset = datasets.Dataset.from_dict({
    "audio": [str(i) for i in Path("$DATA_DIR/audioset/audio").glob("**/*.flac")]
}).cast_column("audio", datasets.Audio(sampling_rate=16000))

for row in tqdm(audioset, desc="Processing AudioSet"):
    name = row['audio']['path'].split('/')[-1].replace(".flac", ".wav")
    scipy.io.wavfile.write(
        "$DATA_DIR/audioset_16k/" + name, 16000,
        (row['audio']['array'] * 32767).astype(np.int16)
    )
EOF
    rm -rf "$DATA_DIR/audioset"
else
    echo "AudioSet already exists, skipping."
fi

# FMA music samples
if [ ! -d "$DATA_DIR/fma" ]; then
    echo "Downloading FMA music samples..."
    mkdir -p "$DATA_DIR/fma"
    python3 << EOF
import datasets
import scipy.io.wavfile
import numpy as np
from tqdm import tqdm

fma = datasets.load_dataset("rudraml/fma", name="small", split="train", streaming=True)
fma = iter(fma.cast_column("audio", datasets.Audio(sampling_rate=16000)))

for i in tqdm(range(120), desc="Processing FMA"):
    try:
        row = next(fma)
        name = row['audio']['path'].split('/')[-1].replace(".mp3", ".wav")
        scipy.io.wavfile.write(
            "$DATA_DIR/fma/" + name, 16000,
            (row['audio']['array'] * 32767).astype(np.int16)
        )
    except StopIteration:
        break
EOF
else
    echo "FMA already exists, skipping."
fi

echo ""
echo "=== Data download complete! ==="
echo ""
du -sh "$DATA_DIR"/*
