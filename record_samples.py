#!/usr/bin/env python3
"""
Record real voice samples for wake word training.
Creates 16kHz mono WAV files in my_real_samples/.

Usage:
    python record_samples.py --wake-word "hey cal"
"""

import argparse
import os
import time
import wave
from ctypes import CFUNCTYPE, POINTER, c_char_p, c_int, cdll
from pathlib import Path

# Suppress ALSA warnings on Linux
try:
    ERROR_HANDLER_FUNC = CFUNCTYPE(None, c_char_p, c_int, c_char_p, c_int, c_char_p)
    def _alsa_error_handler(filename, line, function, err, fmt):
        pass
    _c_error_handler = ERROR_HANDLER_FUNC(_alsa_error_handler)
    asound = cdll.LoadLibrary('libasound.so.2')
    asound.snd_lib_error_set_handler(_c_error_handler)
except Exception:
    pass

import pyaudio

SAMPLE_RATE = 16000
CHANNELS = 1
CHUNK = 1024
FORMAT = pyaudio.paInt16
DURATION = 2.0  # seconds of usable audio captured after the cue
WARMUP = 0.6    # seconds discarded after opening the device, before the cue


def record_sample(filename: str, p: pyaudio.PyAudio):
    """Record a single audio sample, cueing only once the stream is live."""
    stream = p.open(
        format=FORMAT,
        channels=CHANNELS,
        rate=SAMPLE_RATE,
        input=True,
        frames_per_buffer=CHUNK
    )

    # An input device does not deliver usable audio the instant it opens. Read and
    # discard WARMUP seconds first so the cue below lands on an already-running
    # stream - cueing before the device settles loses the first fraction of a
    # second of speech and clips the word onset (the "h" in "hey").
    for _ in range(int(SAMPLE_RATE / CHUNK * WARMUP)):
        stream.read(CHUNK, exception_on_overflow=False)

    print("SPEAK NOW!")

    frames = []
    for _ in range(int(SAMPLE_RATE / CHUNK * DURATION)):
        data = stream.read(CHUNK, exception_on_overflow=False)
        frames.append(data)

    stream.stop_stream()
    stream.close()

    with wave.open(filename, 'wb') as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(p.get_sample_size(FORMAT))
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(b''.join(frames))


def main():
    parser = argparse.ArgumentParser(description="Record voice samples for wake word training")
    parser.add_argument("--wake-word", default="hey cal", help="Wake word you're recording")
    parser.add_argument("--output-dir", default="my_real_samples", help="Output directory")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    safe_name = args.wake_word.replace(" ", "_").lower()

    print("=" * 50)
    print(f"Voice Sample Recorder - \"{args.wake_word}\"")
    print("=" * 50)
    print()
    print("Record at least 20-50 samples for best results.")
    print("Vary your tone, speed, distance from mic, etc.")
    print()
    print("  - Press ENTER to start recording")
    print(f"  - Wait for \"SPEAK NOW!\", then say \"{args.wake_word}\" naturally")
    print("  - Recording lasts 2 seconds; silence is trimmed at training time")
    print("  - Press 'q' + ENTER to quit")
    print()

    count = len(list(output_dir.glob("*.wav")))
    print(f"Existing samples: {count}")
    print()

    # Created once and reused - instantiating PyAudio enumerates devices, which is
    # slow enough to matter if it happens between the cue and the first read.
    p = pyaudio.PyAudio()
    try:
        while True:
            user_input = input(f"[Sample {count + 1}] Press ENTER to record (q to quit): ")

            if user_input.lower() == 'q':
                break

            print("Get ready...", end=" ", flush=True)
            time.sleep(0.5)

            filename = str(output_dir / f"{safe_name}_{count + 1:04d}.wav")
            record_sample(filename, p)

            print(f"Saved: {filename}")
            count += 1
            print()
    finally:
        p.terminate()

    print(f"\nDone! {count} total samples in {output_dir}/")


if __name__ == "__main__":
    main()
