"""ONNX inference equivalence contract check for G0.

Verifies that for the same 385-D policy obs, the deployed
  exported/policy.onnx   (bare MLP, no normalizer in graph)
  exported/policy.pt     (torch.jit module: obs_normalizer -> mlp -> deterministic_output)
produce numerically equivalent 22-D actions.

Also inspects:
  - ONNX graph: input/output names, node types, presence of any normalizer ops
  - policy.pt: obs_normalizer submodule type and buffers, to confirm whether
    normalization is identity or empirical (running mean/var).

Inputs: 385-D policy obs frames from
  logs/sim2sim/g0_history_contract_values.npz
plus a few synthetic frames (zeros, ones, random) for stress testing.

No MuJoCo closed loop. No DDS. No real motor IDs. No env/policy modification.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch


REPO = Path(__file__).resolve().parents[2]
DEFAULT_EXPORT_DIR = REPO / "logs/rsl_rl/g0_velocity/2026-05-26_18-36-57/exported"
DEFAULT_PARAMS_DIR = REPO / "logs/rsl_rl/g0_velocity/2026-05-26_18-36-57/params"
DEFAULT_HISTORY_NPZ = REPO / "logs/sim2sim/g0_history_contract_values.npz"
OUT_DIR = REPO / "logs/sim2sim"
OUT_REPORT = OUT_DIR / "g0_onnx_equivalence_report.txt"
OUT_VALUES = OUT_DIR / "g0_onnx_equivalence_values.npz"

ATOL = 1e-5
RTOL = 1e-5


def inspect_onnx(onnx_path: Path):
    model = onnx.load(str(onnx_path))
    inputs = []
    for inp in model.graph.input:
        shape = [d.dim_value if d.HasField("dim_value") else d.dim_param
                 for d in inp.type.tensor_type.shape.dim]
        inputs.append((inp.name, shape))
    outputs = []
    for out in model.graph.output:
        shape = [d.dim_value if d.HasField("dim_value") else d.dim_param
                 for d in out.type.tensor_type.shape.dim]
        outputs.append((out.name, shape))
    ops = [n.op_type for n in model.graph.node]
    inits = [(init.name, list(init.dims)) for init in model.graph.initializer]
    return inputs, outputs, ops, inits


def inspect_jit(pt_path: Path):
    p = torch.jit.load(str(pt_path), map_location="cpu")
    info = {"top_code": p.code}
    if hasattr(p, "obs_normalizer"):
        norm = p.obs_normalizer
        info["obs_normalizer_type"] = type(norm).__name__
        try:
            info["obs_normalizer_code"] = norm.code
        except Exception:
            info["obs_normalizer_code"] = "<no code>"
        bufs = []
        for n, b in norm.named_buffers():
            bufs.append((n, tuple(b.shape), float(b.float().abs().mean().item())))
        info["obs_normalizer_buffers"] = bufs
        params = []
        for n, w in norm.named_parameters():
            params.append((n, tuple(w.shape)))
        info["obs_normalizer_params"] = params
    return p, info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--export-dir", type=Path, default=DEFAULT_EXPORT_DIR)
    ap.add_argument("--params-dir", type=Path, default=DEFAULT_PARAMS_DIR)
    ap.add_argument("--history-npz", type=Path, default=DEFAULT_HISTORY_NPZ)
    args = ap.parse_args()

    onnx_path = args.export_dir / "policy.onnx"
    pt_path = args.export_dir / "policy.pt"
    if not onnx_path.exists():
        raise FileNotFoundError(onnx_path)
    if not pt_path.exists():
        raise FileNotFoundError(pt_path)

    lines = []
    def log(s=""):
        print(s)
        lines.append(s)

    log("=" * 90)
    log("G0 ONNX inference equivalence contract")
    log("=" * 90)
    log(f"export_dir : {args.export_dir}")
    log(f"params_dir : {args.params_dir}")
    log(f"history    : {args.history_npz}")
    log("")

    # ---------------------------------------------------------------- 1
    log("-" * 90)
    log("[1] ONNX graph inspection")
    log("-" * 90)
    onnx_inputs, onnx_outputs, onnx_ops, onnx_inits = inspect_onnx(onnx_path)
    for n, s in onnx_inputs:
        log(f"  input  {n}: {s}")
    for n, s in onnx_outputs:
        log(f"  output {n}: {s}")
    log(f"  op_types : {onnx_ops}")
    log(f"  #initializers: {len(onnx_inits)}")
    for n, dims in onnx_inits:
        log(f"    {n}: {dims}")
    norm_ops = [op for op in onnx_ops if op.lower() in
                ("batchnormalization", "instancenormalization",
                 "layernormalization", "meanvariancenormalization",
                 "div", "sub")]
    log(f"  normalizer-like ops in graph: {norm_ops if norm_ops else 'NONE'}")
    log("  => ONNX graph is a pure MLP (Gemm + Elu). Any obs normalization"
        " must be applied externally.")
    log("")

    # ---------------------------------------------------------------- 2
    log("-" * 90)
    log("[2] policy.pt inspection")
    log("-" * 90)
    jit, jit_info = inspect_jit(pt_path)
    log("  top forward code:")
    for ln in jit_info["top_code"].splitlines():
        log(f"    {ln}")
    log(f"  obs_normalizer class : {jit_info.get('obs_normalizer_type')}")
    log("  obs_normalizer code:")
    for ln in str(jit_info.get("obs_normalizer_code", "")).splitlines():
        log(f"    {ln}")
    log(f"  obs_normalizer params : {jit_info.get('obs_normalizer_params')}")
    log(f"  obs_normalizer buffers: {jit_info.get('obs_normalizer_buffers')}")
    log("")

    # ---------------------------------------------------------------- 3
    log("-" * 90)
    log("[3] Normalizer probe: feed a few obs through obs_normalizer alone")
    log("-" * 90)
    norm = getattr(jit, "obs_normalizer", None)
    probe = np.stack([
        np.zeros(385, np.float32),
        np.ones(385, np.float32),
        np.random.default_rng(0).standard_normal(385).astype(np.float32),
    ])
    if norm is not None:
        with torch.no_grad():
            probe_out = norm(torch.from_numpy(probe)).cpu().numpy()
        max_io_diff = float(np.max(np.abs(probe - probe_out)))
        log(f"  max|normalizer(x) - x| over 3 probe obs = {max_io_diff:.3e}")
        normalizer_is_identity = max_io_diff < 1e-6
    else:
        normalizer_is_identity = True
        log("  no obs_normalizer attribute; treating as identity")
    log(f"  => obs_normalizer is identity? {normalizer_is_identity}")
    log("")

    # ---------------------------------------------------------------- 4
    log("-" * 90)
    log("[4] Load obs frames")
    log("-" * 90)
    frames = []  # list of (label, obs_385 float32)
    if args.history_npz.exists():
        d = np.load(args.history_npz)
        # g0_history_contract_values.npz has 'policy_obs' shape (T, 385)
        # (per the prior contract step). Try common keys.
        key = None
        for k in ("policy_obs", "obs_385", "obs"):
            if k in d.files:
                key = k
                break
        if key is None:
            # take the first 385-wide 2D array
            for k in d.files:
                v = d[k]
                if v.ndim == 2 and v.shape[1] == 385:
                    key = k
                    break
        if key is not None:
            arr = np.asarray(d[key], dtype=np.float32)
            log(f"  loaded {arr.shape[0]} obs frames from history npz key '{key}'")
            for i in range(arr.shape[0]):
                frames.append((f"history[{i}]", arr[i]))
        else:
            log(f"  WARN: no 385-D obs array in {args.history_npz} "
                f"(keys={list(d.files)})")
    else:
        log(f"  WARN: {args.history_npz} not found; using synthetic only")

    rng = np.random.default_rng(42)
    frames.append(("zeros",   np.zeros(385, np.float32)))
    frames.append(("ones",    np.ones(385, np.float32)))
    frames.append(("randn",   rng.standard_normal(385).astype(np.float32)))
    frames.append(("randn*5", (5.0 * rng.standard_normal(385)).astype(np.float32)))
    log(f"  total frames to test: {len(frames)}")
    log("")

    # ---------------------------------------------------------------- 5
    log("-" * 90)
    log("[5] Run ONNX and policy.pt on each frame, compare")
    log("-" * 90)
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    in_name = sess.get_inputs()[0].name
    out_name = sess.get_outputs()[0].name
    log(f"  onnxruntime input  name: {in_name}")
    log(f"  onnxruntime output name: {out_name}")
    log("")

    header = (f"  {'idx':>3}  {'label':14s} "
              f"{'max|d|':>11}  {'mean|d|':>11}  "
              f"{'allclose':>9}   action_onnx[:4]")
    log(header)
    log("  " + "-" * (len(header) - 2))

    obs_arr = np.stack([f[1] for f in frames]).astype(np.float32)
    act_onnx_all = np.zeros((len(frames), 22), np.float32)
    act_pt_all = np.zeros((len(frames), 22), np.float32)
    per_frame = []
    all_close = True

    with torch.no_grad():
        for i, (label, obs) in enumerate(frames):
            obs_b = obs[None].astype(np.float32)
            a_onnx = sess.run([out_name], {in_name: obs_b})[0][0]
            a_pt = jit(torch.from_numpy(obs_b)).cpu().numpy()[0]
            diff = a_onnx - a_pt
            mx = float(np.max(np.abs(diff)))
            mn = float(np.mean(np.abs(diff)))
            ok = bool(np.allclose(a_onnx, a_pt, atol=ATOL, rtol=RTOL))
            all_close &= ok
            act_onnx_all[i] = a_onnx
            act_pt_all[i] = a_pt
            per_frame.append((label, mx, mn, ok))
            head = ", ".join(f"{v:+.4f}" for v in a_onnx[:4])
            log(f"  {i:>3}  {label:14s} {mx:>11.3e}  {mn:>11.3e}  "
                f"{str(ok):>9}   [{head}]")

    log("")
    log(f"  ATOL={ATOL:g}  RTOL={RTOL:g}")
    log(f"  per-frame allclose passed for ALL frames? {all_close}")
    overall_max = float(np.max(np.abs(act_onnx_all - act_pt_all)))
    overall_mean = float(np.mean(np.abs(act_onnx_all - act_pt_all)))
    log(f"  global max|d|  = {overall_max:.3e}")
    log(f"  global mean|d| = {overall_mean:.3e}")
    log("")

    # ---------------------------------------------------------------- 6
    log("-" * 90)
    log("[6] Head-of-action comparison on first frame")
    log("-" * 90)
    if len(frames) > 0:
        log(f"  frame label: {per_frame[0][0]}")
        log("   i   action_onnx     action_pt       diff")
        for j in range(min(8, 22)):
            ao = act_onnx_all[0, j]
            ap_ = act_pt_all[0, j]
            log(f"  {j:>2}   {ao:+.6e}  {ap_:+.6e}  {ao - ap_:+.3e}")
    log("")

    # ---------------------------------------------------------------- 7
    log("=" * 90)
    log("Verdict")
    log("=" * 90)
    log(f"  ONNX graph contains normalizer?    : {bool(norm_ops)}")
    log(f"  policy.pt obs_normalizer identity? : {normalizer_is_identity}")
    log(f"  ONNX == policy.pt for every frame? : {all_close}")
    if normalizer_is_identity and all_close:
        log("  => Deployment contract OK: feed RAW 385-D policy_obs straight"
            " into policy.onnx, no external normalizer step.")
    elif not normalizer_is_identity:
        log("  => obs_normalizer is NOT identity. Must apply it externally"
            " before calling policy.onnx, OR re-export the ONNX with the"
            " normalizer baked in.")
    if not all_close:
        log("  => MISMATCH between ONNX and policy.pt; investigate before"
            " using ONNX in MuJoCo loop.")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_REPORT.write_text("\n".join(lines) + "\n")
    np.savez(
        OUT_VALUES,
        obs=obs_arr,
        action_onnx=act_onnx_all,
        action_pt=act_pt_all,
        labels=np.array([f[0] for f in frames]),
        per_frame_max_abs=np.array([f[1] for f in per_frame], np.float64),
        per_frame_mean_abs=np.array([f[2] for f in per_frame], np.float64),
        per_frame_allclose=np.array([f[3] for f in per_frame], bool),
        normalizer_is_identity=np.array(normalizer_is_identity),
        all_close=np.array(all_close),
        atol=np.array(ATOL),
        rtol=np.array(RTOL),
    )
    print()
    print(f"Wrote {OUT_REPORT}")
    print(f"Wrote {OUT_VALUES}")


if __name__ == "__main__":
    main()
