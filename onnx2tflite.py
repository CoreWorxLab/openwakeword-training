#!/usr/bin/env python3
"""Convert an openWakeWord ONNX model to tflite.

onnx2tf relabels the input shape [1,16,96] -> [1,96,16] without moving the
underlying data, so we re-export wrapping it in a tf.reshape. Note: NOT a
transpose -- a transpose gives the right shape and the wrong numbers.

Run inside a container with tensorflow + onnxruntime + onnx2tf available:

    python onnx2tflite_oww.py hey_seeree.onnx -o hey_seeree_v0.1.tflite
"""

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import onnxruntime as ort
import tensorflow as tf


def convert(onnx_path: Path, out_path: Path, trials: int = 5, tol: float = 1e-4) -> float:
    workdir = Path(tempfile.mkdtemp())
    try:
        print("==> onnx2tf")
        subprocess.run(
            ["onnx2tf", "-i", str(onnx_path), "-o", str(workdir / "sm"), "-nuo", "-osd"],
            check=True,
            capture_output=True,
        )

        sess = ort.InferenceSession(str(onnx_path))
        name = sess.get_inputs()[0].name
        shape = [d if isinstance(d, int) else 1 for d in sess.get_inputs()[0].shape]

        loaded = tf.saved_model.load(str(workdir / "sm"))
        sig = loaded.signatures["serving_default"]
        key = next(iter(sig.structured_input_signature[1]))
        sm_shape = [int(d) for d in sig.structured_input_signature[1][key].shape]
        out_key = next(iter(sig.structured_outputs))

        print(f"==> re-exporting: {shape} -> reshape -> {sm_shape}")

        @tf.function(input_signature=[tf.TensorSpec(shape, tf.float32, name="input")])
        def wrapped(x):
            return sig(**{key: tf.reshape(x, sm_shape)})[out_key]

        converter = tf.lite.TFLiteConverter.from_concrete_functions(
            [wrapped.get_concrete_function()]
        )
        out_path.write_bytes(converter.convert())

        # verify against the ONNX
        interp = tf.lite.Interpreter(str(out_path))
        interp.allocate_tensors()
        inp, out = interp.get_input_details()[0], interp.get_output_details()[0]

        worst = 0.0
        for _ in range(trials):
            x = np.random.randn(*shape).astype(np.float32)
            ref = sess.run(None, {name: x})[0]
            interp.set_tensor(inp["index"], x)
            interp.invoke()
            got = interp.get_tensor(out["index"])
            worst = max(worst, float(np.abs(ref - got).max()))

        print(f"==> input shape {list(inp['shape'])}, max diff vs onnx {worst:.2e}")
        if worst >= tol:
            raise SystemExit(f"FAILED: diff {worst:.2e} exceeds tolerance {tol:.0e}")

        print(f"==> wrote {out_path}")
        return worst
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("onnx", type=Path, help="input .onnx model")
    p.add_argument("-o", "--output", type=Path, help="output .tflite (default: alongside input)")
    p.add_argument("--trials", type=int, default=5, help="verification trials (default: 5)")
    p.add_argument("--tol", type=float, default=1e-4, help="max allowed diff (default: 1e-4)")
    args = p.parse_args()

    if not args.onnx.is_file():
        sys.exit(f"not found: {args.onnx}")

    convert(args.onnx, args.output or args.onnx.with_suffix(".tflite"), args.trials, args.tol)


if __name__ == "__main__":
    main()
