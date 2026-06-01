"""ONNXRuntime wrapper — mirrors unitree_rl_lab/deploy/.../algorithms/algorithms.h."""

import numpy as np
import onnxruntime as ort


class OrtRunner:
    def __init__(self, onnx_path: str, expected_input_dim: int = None):
        self.sess = ort.InferenceSession(
            onnx_path, providers=["CPUExecutionProvider"]
        )
        self.inp_name = self.sess.get_inputs()[0].name
        self.out_name = self.sess.get_outputs()[0].name
        if expected_input_dim is not None:
            shape = self.sess.get_inputs()[0].shape
            dims = [s for s in shape if isinstance(s, int)]
            if expected_input_dim not in dims:
                raise RuntimeError(
                    f"ONNX expects {shape}, requested {expected_input_dim}"
                )

    def __call__(self, obs_flat: np.ndarray) -> np.ndarray:
        x = obs_flat.astype(np.float32).reshape(1, -1)
        out = self.sess.run([self.out_name], {self.inp_name: x})[0]
        return np.asarray(out[0], dtype=np.float64)
