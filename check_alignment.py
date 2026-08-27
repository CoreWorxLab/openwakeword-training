#!/usr/bin/env python3
"""
Measure where speech sits inside OpenWakeWord's fixed-size detection window.

OpenWakeWord aligns the END OF THE ARRAY with the end of the detection window
(openwakeword/data.py:719), so any trailing silence pushes the phrase earlier than
the alignment the model sees when streaming detection fires. This reports how far
each clip's speech actually ends from the window end, so you can check your samples
before committing to a multi-hour training run.

Usage:
    python check_alignment.py my_real_samples/
    python check_alignment.py my_custom_model/hey_cal/positive_train --verbose
    python check_alignment.py my_real_samples/ --total-length 32000

Only needs numpy + scipy, so it runs on the host where you record.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import scipy.io.wavfile

# Mirrors trim_silence() in train.py. Duplicated rather than imported so this script
# stays runnable on the host without train.py's dependencies (requests, yaml, tqdm).
TOP_DB = 40.0
FRAME_MS = 10.0

# create_fixed_size_clip picks end_jitter uniformly from [0, 200) ms, so even a
# perfectly trimmed clip sits up to this far from the window end.
END_JITTER_MS = 200.0


def speech_bounds(data: np.ndarray, sr: int):
    """Return (start_sample, end_sample) of speech energy, or None if silent."""
    frame = max(1, int(sr * FRAME_MS / 1000))
    n_frames = len(data) // frame
    if n_frames < 2:
        return None

    frames = data[:n_frames * frame].astype(np.float64).reshape(n_frames, frame)
    rms = np.sqrt(np.mean(frames ** 2, axis=1))
    peak = rms.max()
    if peak <= 0:
        return None

    voiced = np.flatnonzero(rms > peak * (10 ** (-TOP_DB / 20)))
    if voiced.size == 0:
        return None

    return voiced[0] * frame, (voiced[-1] + 1) * frame


def compute_total_length(durations: list) -> int:
    """Replicate OpenWakeWord's total_length calculation (openwakeword/train.py:754)."""
    total = int(round(np.median(durations) / 1000) * 1000) + 12000
    if total < 32000:
        return 32000
    if abs(total - 32000) <= 4000:
        return 32000
    return total


def main():
    parser = argparse.ArgumentParser(
        description="Check where speech lands in the OpenWakeWord detection window")
    parser.add_argument("directory", help="Directory of 16kHz WAV files to inspect")
    parser.add_argument("--total-length", type=int, default=None,
                        help="Window size in samples (default: derived as OpenWakeWord does)")
    parser.add_argument("--limit", type=int, default=None, help="Only inspect the first N files")
    parser.add_argument("--verbose", action="store_true", help="Print a row per file")
    args = parser.parse_args()

    directory = Path(args.directory)
    wavs = sorted(directory.glob("*.wav"))
    if args.limit:
        wavs = wavs[:args.limit]
    if not wavs:
        print(f"No WAV files found in {directory}/")
        sys.exit(1)

    clips = []
    silent = []
    for wav_file in wavs:
        sr, data = scipy.io.wavfile.read(wav_file)
        if data.ndim > 1:
            data = data[:, 0]
        bounds = speech_bounds(data, sr)
        if bounds is None:
            silent.append(wav_file.name)
            continue
        clips.append((wav_file.name, sr, len(data), bounds[0], bounds[1]))

    if not clips:
        print(f"No speech detected in any of the {len(wavs)} files.")
        sys.exit(1)

    total_length = args.total_length or compute_total_length([c[2] for c in clips])

    rows = []
    for name, sr, length, sp_start, sp_end in clips:
        lead_ms = sp_start / sr * 1000
        trail_ms = (length - sp_end) / sr * 1000

        # Replicate placement: a clip longer than the window keeps only its FIRST
        # total_length samples (data.py:676-677) - the tail is discarded, which can
        # cut the end of the phrase - then the array end aligns with the window end.
        placed_len = min(length, total_length)
        placed_end = min(sp_end, placed_len)
        start = max(0, total_length - placed_len)
        gap_ms = (total_length - (start + placed_end)) / sr * 1000

        rows.append((name, length / sr * 1000, lead_ms, trail_ms, gap_ms,
                     length > total_length))

    if args.verbose:
        print(f"{'file':<34}{'dur':>8}{'lead':>8}{'trail':>8}{'gap':>8}")
        print("-" * 66)
        for name, dur, lead, trail, gap, truncated in rows:
            flag = " *" if truncated else ""
            print(f"{name[:33]:<34}{dur:>7.0f}m{lead:>7.0f}m{trail:>7.0f}m{gap:>7.0f}m{flag}")
        print()

    gaps = np.array([r[4] for r in rows])
    trails = np.array([r[3] for r in rows])
    leads = np.array([r[2] for r in rows])
    truncated = sum(1 for r in rows if r[5])

    print("=" * 60)
    print(f"{len(rows)} clips from {directory}/")
    print(f"Window (total_length): {total_length} samples "
          f"({total_length / 16000:.2f}s)"
          f"{'' if args.total_length else ' [derived]'}")
    print("=" * 60)
    print(f"Leading silence:   mean {leads.mean():>6.0f}ms   max {leads.max():>6.0f}ms")
    print(f"Trailing silence:  mean {trails.mean():>6.0f}ms   max {trails.max():>6.0f}ms")
    print()
    print("Speech end -> window end (the number that matters):")
    print(f"  mean {gaps.mean():>6.0f}ms   median {np.median(gaps):>6.0f}ms   max {gaps.max():>6.0f}ms")
    print()

    if silent:
        print(f"WARNING: {len(silent)} file(s) had no detectable speech: "
              f"{', '.join(silent[:5])}{'...' if len(silent) > 5 else ''}")
    if truncated:
        print(f"WARNING: {truncated} clip(s) are longer than the {total_length / 16000:.2f}s "
              f"window. Only their first {total_length / 16000:.2f}s survives - the tail is "
              f"discarded, which can cut the end of the phrase (marked * above).")

    # A clip several times the median length is usually two utterances that the
    # recorder's silence splitter merged, not one slow one. Those make bad
    # positives: after truncation the array no longer ends where a phrase ends.
    durations = np.array([r[1] for r in rows])
    median_dur = float(np.median(durations))
    outliers = [(r[0], r[1]) for r in rows if r[1] > 2.0 * median_dur]
    if outliers:
        print(f"WARNING: {len(outliers)} clip(s) over 2x the median length "
              f"({median_dur:.0f}ms) - likely two merged utterances, worth listening to:")
        for name, dur in sorted(outliers, key=lambda t: -t[1])[:8]:
            print(f"    {name}  {dur:.0f}ms")

    # A well-trimmed set should sit within the jitter range, the residual being
    # decaying reverb tail rather than silence.
    if gaps.mean() <= END_JITTER_MS:
        print(f"OK: mean gap is within the {END_JITTER_MS:.0f}ms jitter "
              f"create_fixed_size_clip adds on purpose.")
    else:
        print(f"Speech sits {gaps.mean() - END_JITTER_MS:.0f}ms earlier than the "
              f"{END_JITTER_MS:.0f}ms jitter alone would explain - these clips need trimming.")


if __name__ == "__main__":
    main()
