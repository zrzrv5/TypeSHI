"""Export TypeSetClassifier checkpoints to ONNX and CoreML for on-device inference.

Fixed-shape (batch=1, T padded to MAX_TYPES=8) export. Descriptor computation
(src/typeid2elem/descriptors.py) must be reimplemented on-device -- this script
only exports the model that maps descriptors -> per-type element log-probs.

Usage:
    uv run python scripts/export_model.py                       # all runs/production/*.ckpt
    uv run python scripts/export_model.py --ckpt runs/production/sharp30.ckpt
    uv run python scripts/export_model.py --out-dir runs/export --n-parity 20
    uv run python scripts/export_model.py --ckpt ... --coreai    # + Apple Core AI .aimodel

--coreai adds an Apple Core AI (.aimodel) export via coreai-torch (2026 framework,
Xcode/OS 27+ to run; conversion works on macOS 26+). It goes straight from the fp32
checkpoint graph, so it avoids the int8 quantization of the committed deploy ONNX --
prefer ckpt -> .aimodel over onnx -> anything. Install: `pip install coreai-torch`.

CPU only (do not run on GPU -- it may be busy training).
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from typeid2elem.augment import MAX_TYPES  # noqa: E402
from typeid2elem.descriptors import K_ENV, M_ENV, R_ENV  # noqa: E402
from typeid2elem.model import TypeSetClassifier  # noqa: E402

torch.set_grad_enabled(False)

T_FIXED = MAX_TYPES  # 8: padded type-token count
B_FIXED = 1
N_CLASSES = 94
NEG = -1.0e4  # fp16-safe "effectively -inf" fill (fp16 max ~6.5e4)


# --------------------------------------------------------------------------- #
# Model wrapper
# --------------------------------------------------------------------------- #
class ManualEncoderLayer(nn.Module):
    """Re-implements nn.TransformerEncoderLayer(norm_first=True, activation=gelu)
    with explicit matmul attention instead of F.multi_head_attention_forward /
    scaled_dot_product_attention.

    Why: with src_key_padding_mask, PyTorch's fast attention path records
    tensor-valued aten::size/aten::Int ops for batch/seq dims during tracing.
    torch.onnx.export handles this fine, but coremltools' torch frontend
    chokes trying to resolve these to constants ("TypeError: only
    0-dimensional arrays can be converted to Python scalars" inside its
    `_int` op conversion, anchored at `.../self_attn/...`). Reimplementing
    attention manually with fixed python-int shapes (this script only ever
    exports B=1, T=8) avoids those ops entirely. Verified equivalent to the
    original nn.TransformerEncoder to <1e-5 abs diff (see `_selfcheck`).
    """

    def __init__(self, layer: nn.TransformerEncoderLayer, n_heads: int):
        super().__init__()
        self.self_attn = layer.self_attn  # reused for its trained weights
        self.linear1 = layer.linear1
        self.linear2 = layer.linear2
        self.norm1 = layer.norm1
        self.norm2 = layer.norm2
        self.n_heads = n_heads
        self.d_model = layer.self_attn.embed_dim
        self.head_dim = self.d_model // n_heads

    def _mha(self, x: torch.Tensor, key_padding_mask: torch.Tensor) -> torch.Tensor:
        B, T, D = B_FIXED, T_FIXED, self.d_model
        H, Hd = self.n_heads, self.head_dim
        qkv = F.linear(x, self.self_attn.in_proj_weight, self.self_attn.in_proj_bias)
        q, k, v = qkv.chunk(3, dim=-1)

        def split_heads(t):
            return t.view(B, T, H, Hd).transpose(1, 2)  # (B, H, T, Hd)

        q, k, v = split_heads(q), split_heads(k), split_heads(v)
        attn = torch.matmul(q, k.transpose(-2, -1)) * (Hd ** -0.5)  # (B, H, T, T)
        bias = torch.zeros(B, 1, 1, T, dtype=x.dtype).masked_fill(
            key_padding_mask[:, None, None, :], NEG)
        attn = (attn + bias).softmax(dim=-1)
        out = torch.matmul(attn, v).transpose(1, 2).reshape(B, T, D)
        return self.self_attn.out_proj(out)

    def forward(self, x: torch.Tensor, key_padding_mask: torch.Tensor) -> torch.Tensor:
        x = x + self._mha(self.norm1(x), key_padding_mask)
        h = self.linear2(F.gelu(self.linear1(self.norm2(x))))
        return x + h


class ExportModel(nn.Module):
    """Fixed-shape (B=1, T=8) inference wrapper around TypeSetClassifier.

    Takes the same fields collate()/features_to_batch() produce (rdf,
    pair_extra, frac, glob, mask), padded/truncated to T=8, and returns
    log-softmax class probabilities (1, 8, 94). `mask` is a float 0/1 tensor
    (not bool) -- bool graph inputs are flakier across ONNX/CoreML backends.

    use_env checkpoints take two extra fixed-shape inputs, env_d and env_t
    (1, 8, M_ENV=16, K_ENV=16): per-type sampled-atom neighbor environments as
    (distance A, partner type id). env_t is a FLOAT tensor for backend
    friendliness; -1.0 marks padding, real values are integral type indices
    cast to int inside the graph for the gather.
    """

    def __init__(self, lit_model: TypeSetClassifier):
        super().__init__()
        self.use_env = bool(lit_model.hparams.get("use_env", False))
        if self.use_env:
            self.partner_proj = lit_model.partner_proj
            self.env_nbr = lit_model.env_nbr
            self.env_atom = lit_model.env_atom
            self.env_out = lit_model.env_out
            self.d_env = int(lit_model.partner_proj.out_features)
            self.register_buffer("rbf_centers", lit_model.rbf_centers.clone(),
                                 persistent=False)
            self.rbf_gamma = lit_model.rbf_gamma
        self.phi = lit_model.phi
        self.proj = lit_model.proj
        n_heads = lit_model.hparams.n_heads
        self.layers = nn.ModuleList(
            ManualEncoderLayer(l, n_heads) for l in lit_model.encoder.layers)
        self.head = lit_model.head
        self.two_pass = bool(lit_model.hparams.two_pass)
        if self.two_pass:
            self.elem_embed = lit_model.elem_embed
        self.register_buffer(
            "eye", torch.eye(T_FIXED)[None].expand(B_FIXED, T_FIXED, T_FIXED).clone(),
            persistent=False)

    def _encode(self, tok: torch.Tensor, kpm: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            tok = layer(tok, kpm)
        return tok

    def _env_update(self, tok, env_d, env_t):
        B, T = B_FIXED, T_FIXED
        valid = (env_t > -0.5).float()                       # (B, T, M, K)
        idx = env_t.clamp(min=0.0).to(torch.int64)
        pemb = self.partner_proj(tok)                        # (B, T, d_env)
        g = torch.gather(
            pemb, 1,
            idx.reshape(B, T * M_ENV * K_ENV, 1).expand(B, T * M_ENV * K_ENV, self.d_env)
        ).reshape(B, T, M_ENV, K_ENV, self.d_env)
        rbf = torch.exp(-self.rbf_gamma *
                        (env_d[..., None] - self.rbf_centers) ** 2)
        hn = self.env_nbr(torch.cat([rbf, g], dim=-1)) * valid[..., None]
        atom = hn.sum(-2) / valid.sum(-1).clamp(min=1.0)[..., None]
        avalid = (valid.sum(-1) > 0.5).float()               # (B, T, M)
        atom = self.env_atom(atom) * avalid[..., None]
        env = atom.sum(-2) / avalid.sum(-1).clamp(min=1.0)[..., None]
        return tok + self.env_out(env)

    def forward(self, rdf, pair_extra, frac, glob, mask_f, env_d=None, env_t=None):
        mask = mask_f > 0.5
        kpm = ~mask  # True at padded key positions, matches src_key_padding_mask
        B, T = B_FIXED, T_FIXED
        pair = torch.cat(
            [torch.log1p(rdf), pair_extra, frac[:, None, :].expand(B, T, T)[..., None],
             self.eye[..., None]], dim=-1)
        h = self.phi(pair)
        pmask = mask[:, None, :, None]
        h = h * pmask
        n_valid = mask.sum(1).clamp(min=1)[:, None, None].float()
        h_mean = h.sum(2) / n_valid
        h_max = h.masked_fill(~pmask.expand_as(h), NEG).amax(2)
        tok = torch.cat(
            [h_mean, h_max, frac[..., None], glob[:, None, :].expand(B, T, -1)], dim=-1)
        tok = self.proj(tok)
        if self.use_env:
            tok = self._env_update(tok, env_d, env_t)
        henc = self._encode(tok, kpm)
        logits1 = self.head(henc)
        if not self.two_pass:
            return F.log_softmax(logits1, dim=-1)
        tok2 = tok + self.elem_embed(logits1.softmax(-1))
        henc2 = self._encode(tok2, kpm)
        logits2 = self.head(henc2)
        return F.log_softmax(logits2, dim=-1)


# --------------------------------------------------------------------------- #
# Reference forward through the original LightningModule (for parity checks)
# --------------------------------------------------------------------------- #
def reference_logprobs(lit_model: TypeSetClassifier, rdf, pair_extra, frac, glob,
                       mask_f, env_d=None, env_t=None):
    batch = dict(rdf=rdf, pair_extra=pair_extra, frac=frac, glob=glob, mask=mask_f > 0.5)
    if env_d is not None:
        batch["env_d"] = env_d
        batch["env_t"] = env_t.to(torch.int64)  # original model wants int env_t
    out = lit_model(batch)
    logits = out[1] if isinstance(out, tuple) else out
    return F.log_softmax(logits, dim=-1)


def random_inputs(rng: np.random.Generator, t_valid: int, use_env: bool = False):
    """Random padded (B=1, T=8) input tensors with the first t_valid type
    slots marked valid (mask=1), matching collate()'s zero-padding convention
    for the *padded* rows/cols (real descriptors are never negative, but for
    a pure numerical-parity check across backends any finite values suffice
    -- padded content never reaches the valid output rows, see docs/EXPORT.md).
    """
    T = T_FIXED
    rdf = np.zeros((1, T, T, 64), np.float32)
    pe = np.zeros((1, T, T, 8), np.float32)
    frac = np.zeros((1, T), np.float32)
    glob = rng.normal(size=(1, 3)).astype(np.float32)
    mask = np.zeros((1, T), np.float32)
    mask[0, :t_valid] = 1.0
    rdf[0, :t_valid, :t_valid] = rng.exponential(1.0, size=(t_valid, t_valid, 64))
    pe[0, :t_valid, :t_valid] = rng.normal(size=(t_valid, t_valid, 8))
    frac_v = rng.dirichlet(np.ones(t_valid)).astype(np.float32)
    frac[0, :t_valid] = frac_v
    arrays = [rdf, pe, frac, glob, mask]
    if use_env:
        env_d = np.zeros((1, T, M_ENV, K_ENV), np.float32)
        env_t = np.full((1, T, M_ENV, K_ENV), -1.0, np.float32)
        for a in range(t_valid):
            n_atoms = int(rng.integers(1, M_ENV + 1))
            for m in range(n_atoms):
                k = int(rng.integers(1, K_ENV + 1))
                env_d[0, a, m, :k] = np.sort(
                    rng.uniform(0.8, R_ENV, size=k)).astype(np.float32)
                env_t[0, a, m, :k] = rng.integers(0, t_valid, size=k)
        arrays += [env_d, env_t]
    return tuple(torch.from_numpy(a) for a in arrays)


# --------------------------------------------------------------------------- #
# Export routines
# --------------------------------------------------------------------------- #
INPUT_NAMES = ["rdf", "pair_extra", "frac", "glob", "mask"]
ENV_INPUT_NAMES = INPUT_NAMES + ["env_d", "env_t"]


def _names(dummy):
    return ENV_INPUT_NAMES if len(dummy) == 7 else INPUT_NAMES


def export_onnx(wrapper: ExportModel, dummy, path: Path) -> None:
    torch.onnx.export(
        wrapper, dummy, str(path),
        input_names=_names(dummy),
        output_names=["log_probs"],
        opset_version=17,
        dynamo=False,
    )


def onnx_parity(onnx_path: Path, wrapper: ExportModel, n: int, seed: int) -> float:
    import onnxruntime as ort

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    rng = np.random.default_rng(seed)
    max_diff = 0.0
    for _ in range(n):
        t_valid = int(rng.integers(1, T_FIXED + 1))
        inputs = random_inputs(rng, t_valid, wrapper.use_env)
        with torch.no_grad():
            ref = wrapper(*inputs).numpy()
        out = sess.run(None, {n: t.numpy() for n, t in zip(_names(inputs), inputs)})[0]
        diff = np.abs(out[0, :t_valid] - ref[0, :t_valid]).max()
        max_diff = max(max_diff, float(diff))
    return max_diff


def export_coreml(wrapper: ExportModel, dummy, base_path: Path) -> dict:
    import coremltools as ct

    traced = torch.jit.trace(wrapper, dummy, check_trace=False)
    with torch.no_grad():
        traced_out = traced(*dummy)
        ref_out = wrapper(*dummy)
    trace_diff = (traced_out - ref_out).abs().max().item()

    inputs = [ct.TensorType(name=n, shape=t.shape)
              for n, t in zip(_names(dummy), dummy)]
    outputs = [ct.TensorType(name="log_probs")]

    result = {"trace_vs_torch_diff": trace_diff, "fp32_saved": False, "fp16_warned": False}

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        mlmodel_fp16 = ct.convert(
            traced, inputs=inputs, outputs=outputs,
            minimum_deployment_target=ct.target.iOS16,
            convert_to="mlprogram",
            compute_precision=ct.precision.FLOAT16,
        )
    fp16_path = base_path.with_suffix(".mlpackage")
    mlmodel_fp16.save(str(fp16_path))
    result["fp16_path"] = fp16_path
    result["fp16_warned"] = any(
        "overflow" in str(w.message).lower() or issubclass(w.category, RuntimeWarning)
        for w in caught)

    if result["fp16_warned"]:
        mlmodel_fp32 = ct.convert(
            traced, inputs=inputs, outputs=outputs,
            minimum_deployment_target=ct.target.iOS16,
            convert_to="mlprogram",
            compute_precision=ct.precision.FLOAT32,
        )
        fp32_path = Path(str(base_path) + "_fp32").with_suffix(".mlpackage")
        mlmodel_fp32.save(str(fp32_path))
        result["fp32_path"] = fp32_path
        result["fp32_saved"] = True

    return result


def export_coreai(wrapper: ExportModel, dummy, base_path: Path) -> dict:
    """Export to Apple Core AI (`.aimodel`) via coreai-torch (2026 framework).

    Core AI is Apple's successor to Core ML for on-device inference (Xcode 27 /
    macOS|iOS 27+). This derives the on-device model straight from the fp32
    checkpoint graph -- NO int8 quantization, unlike the committed deploy ONNX --
    which is the whole reason to prefer ckpt -> .aimodel over onnx -> anything.

    Requires `coreai-torch` (pip; pulls a CPU/MPS torch). The conversion itself
    runs on macOS 26+, but the produced asset targets OS 27 (save_asset's
    minimum_os default) and needs an OS-27 device / the Core AI runtime to execute.
    Verified API surface: coreai-torch 0.4.1 / coreai-core 1.0.0b2.
    """
    import coreai_torch as ct  # optional heavy dep; imported only under --coreai

    names = _names(dummy)
    # torch.export (not jit.trace) is Core AI's front door; decompose to the ops
    # the converter lowers, then hand the ExportedProgram to TorchConverter.
    ep = torch.export.export(wrapper, tuple(dummy))
    ep = ep.run_decompositions(ct.get_decomp_table())

    # conversion-fidelity self-check: exported graph vs the eager wrapper
    with torch.no_grad():
        ep_out = ep.module()(*dummy)
        ref_out = wrapper(*dummy)
    export_diff = (ep_out - ref_out).abs().max().item()

    prog = (
        ct.TorchConverter()
        .add_exported_program(ep, input_names=list(names), output_names=["log_probs"])
        .to_coreai()
    )
    path = base_path.with_suffix(".aimodel")
    prog.save_asset(path)          # writes an .aimodel bundle (minimum_os=v27)
    return {"aimodel_path": path, "export_vs_torch_diff": export_diff}


def _dir_size(path: Path) -> int:
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


def export_one(ckpt: Path, out_dir: Path, n_parity: int, seed: int,
               coreai: bool = False) -> dict:
    name = ckpt.stem
    print(f"\n=== {name} ===")
    lit = TypeSetClassifier.load_from_checkpoint(str(ckpt), map_location="cpu")
    lit.eval()
    print(f"  two_pass={lit.hparams.two_pass} use_env={lit.hparams.get('use_env', False)} "
          f"n_heads={lit.hparams.n_heads} d_model={lit.hparams.d_model}")

    wrapper = ExportModel(lit)
    wrapper.eval()

    rng = np.random.default_rng(seed)
    dummy = random_inputs(rng, T_FIXED, wrapper.use_env)  # full T=8, for tracing/export

    with torch.no_grad():
        ref = reference_logprobs(lit, *dummy)
        wrap_out = wrapper(*dummy)
    selfcheck_diff = (ref - wrap_out).abs().max().item()
    print(f"  wrapper vs original torch forward, full-T dummy: max diff = {selfcheck_diff:.2e}")
    assert selfcheck_diff < 1e-4, "wrapper does not match original model within rtol 1e-4"

    n_params = sum(p.numel() for p in wrapper.parameters())

    onnx_path = out_dir / f"{name}.onnx"
    export_onnx(wrapper, dummy, onnx_path)
    onnx_diff = onnx_parity(onnx_path, wrapper, n_parity, seed + 1)
    print(f"  ONNX parity ({n_parity} random inputs, T in 1..8): max diff = {onnx_diff:.2e}")
    assert onnx_diff < 1e-3, "ONNX parity exceeds 1e-3"
    onnx_size = onnx_path.stat().st_size

    coreml_base = out_dir / name
    coreml_res = export_coreml(wrapper, dummy, coreml_base)
    print(f"  CoreML trace-vs-torch diff = {coreml_res['trace_vs_torch_diff']:.2e}")
    fp32_note = " (fp32 fallback also saved)" if coreml_res["fp32_saved"] else ""
    print(f"  CoreML fp16 conversion warned: {coreml_res['fp16_warned']}{fp32_note}")
    fp16_size = _dir_size(coreml_res["fp16_path"])
    fp32_size = _dir_size(coreml_res["fp32_path"]) if coreml_res["fp32_saved"] else None

    coreai_path = coreai_size = None
    if coreai:
        coreai_res = export_coreai(wrapper, dummy, out_dir / name)
        coreai_path = coreai_res["aimodel_path"]
        coreai_size = _dir_size(coreai_path)
        print(f"  Core AI (.aimodel) export-vs-torch diff = "
              f"{coreai_res['export_vs_torch_diff']:.2e}  ({coreai_size/1e6:.2f}MB)")

    return dict(
        name=name, n_params=n_params, selfcheck_diff=selfcheck_diff,
        onnx_path=onnx_path, onnx_diff=onnx_diff, onnx_size=onnx_size,
        coreml_fp16_path=coreml_res["fp16_path"], coreml_fp16_size=fp16_size,
        coreml_fp32_path=coreml_res.get("fp32_path"), coreml_fp32_size=fp32_size,
        coreml_fp16_warned=coreml_res["fp16_warned"],
        coreai_path=coreai_path, coreai_size=coreai_size,
        two_pass=lit.hparams.two_pass, use_env=lit.hparams.get("use_env", False),
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", nargs="*", default=None,
                     help="checkpoint path(s); default: runs/production/*.ckpt")
    ap.add_argument("--out-dir", default="runs/export")
    ap.add_argument("--n-parity", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--coreai", action="store_true",
                    help="also export Apple Core AI .aimodel (needs `pip install coreai-torch`)")
    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    ckpts = ([Path(c) for c in args.ckpt] if args.ckpt
             else sorted((repo_root / "runs/production").glob("*.ckpt")))
    if not ckpts:
        raise FileNotFoundError("no checkpoints found")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for ckpt in ckpts:
        results.append(export_one(ckpt, out_dir, args.n_parity, args.seed, args.coreai))

    print("\n=== summary ===")
    for r in results:
        fp32_mb = "n/a" if r["coreml_fp32_size"] is None else f"{r['coreml_fp32_size']/1e6:.2f}MB"
        coreai_mb = "" if r.get("coreai_size") is None else f" coreai={r['coreai_size']/1e6:.2f}MB"
        print(f"{r['name']}: params={r['n_params']:,} "
              f"onnx_diff={r['onnx_diff']:.2e} onnx={r['onnx_size']/1e6:.2f}MB "
              f"coreml_fp16={r['coreml_fp16_size']/1e6:.2f}MB "
              f"coreml_fp32={fp32_mb}{coreai_mb}")


if __name__ == "__main__":
    main()
