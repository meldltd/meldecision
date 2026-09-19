#!/usr/bin/env python3
"""Export a Laya checkpoint (https://github.com/NandhaKishorM/laya) to ONNX for the Go runtime.

    python export/export_onnx.py english            # convaiinnovations/laya (repo root)
    python export/export_onnx.py multilingual       # convaiinnovations/laya/multilingual
    python export/export_onnx.py typed-decisions    # convaiinnovations/laya/typed-decisions
    python export/export_onnx.py all [--int8]

Each checkpoint lands in  models/<name>/  as:

    model.onnx            graph:  (input_ids, attention_mask, marker_pos, marker_mask, qtype)
                                  -> (logits, act_logits)
    model.onnx.data       external weights (fp32)
    model.int8.onnx       optional dynamically quantised variant (--int8)
    tokenizer.json        HF tokenizers file, loaded by the Go side as-is
    laya_config.json      rl_agent_config.json + the special-token ids the Go side needs

The graph is self-contained: attention masks (padding + ModernBERT's sliding window) are built
inside the graph from `attention_mask`, so the caller only pads with the pad id and passes 0/1.

Requires:  pip install torch transformers safetensors huggingface_hub onnx onnxscript onnxruntime tokenizers
"""
import argparse
import json
import os
import shutil
import sys
import time

import numpy as np
import torch
import torch.nn as nn

BUNDLE_REPO = "convaiinnovations/laya"
CHECKPOINTS = {
    "english": None,
    "multilingual": "multilingual",
    "typed-decisions": "typed-decisions",
}
OPSET = 18
INPUT_NAMES = ["input_ids", "attention_mask", "marker_pos", "marker_mask", "qtype"]
OUTPUT_NAMES = ["logits", "act_logits"]


# --------------------------------------------------------------------------- model (mirrors laya/common.py)
class DecisionModel(nn.Module):
    """Bidirectional transformer encoder backbone + typed decision head (laya.common.DecisionModel)."""

    def __init__(self, encoder: nn.Module, head_layers: int = 2, n_act: int = 2, dropout: float = 0.1):
        super().__init__()
        self.encoder = encoder
        d = encoder.config.hidden_size
        nhead = max(1, d // 64)
        layer = nn.TransformerEncoderLayer(d, nhead, 4 * d, dropout, batch_first=True, norm_first=True)
        self.head = nn.TransformerEncoder(layer, head_layers, enable_nested_tensor=False) if head_layers > 0 else None
        self.type_emb = nn.Embedding(3, d)
        self.scorer = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d), nn.GELU(), nn.Linear(d, 1))
        self.act_head = nn.Sequential(nn.Linear(d + 4, 256), nn.GELU(), nn.Linear(256, n_act))
        self.register_buffer("temperature", torch.ones(3))


class ExportWrapper(nn.Module):
    """DecisionModel.forward rewritten with export-friendly ops and explicit attention masks.

    transformers' own mask helpers may skip the mask entirely for an all-ones attention_mask,
    which a traced graph would bake in; building the masks here keeps padding correct for any
    batch. Verified against the upstream model to ~1e-6.
    """

    def __init__(self, m: DecisionModel):
        super().__init__()
        self.m = m
        cfg = m.encoder.config
        self.window = int(cfg.sliding_window)  # local_attention // 2, inclusive distance

    def _masks(self, attention_mask: torch.Tensor, dtype: torch.dtype):
        # additive masks [B, 1, L, L]: 0 where attendable, finfo.min elsewhere
        B, L = attention_mask.shape
        neg = torch.finfo(dtype).min
        kv_ok = attention_mask.bool()[:, None, None, :]                      # [B,1,1,L]
        full = torch.where(kv_ok, torch.zeros((), dtype=dtype), torch.full((), neg, dtype=dtype))
        full = full.expand(B, 1, L, L)
        idx = torch.arange(L, device=attention_mask.device)
        dist = (idx[:, None] - idx[None, :]).abs() <= self.window               # [L,L]
        win_ok = kv_ok & dist[None, None, :, :]
        sliding = torch.where(win_ok, torch.zeros((), dtype=dtype), torch.full((), neg, dtype=dtype))
        return {"full_attention": full, "sliding_attention": sliding}

    def forward(self, input_ids, attention_mask, marker_pos, marker_mask, qtype):
        m = self.m
        emb_dtype = m.encoder.embeddings.tok_embeddings.weight.dtype
        masks = self._masks(attention_mask, emb_dtype)
        h = m.encoder(input_ids=input_ids, attention_mask=masks).last_hidden_state
        h = h + m.type_emb(qtype)[:, None, :]
        if m.head is not None:
            pad = attention_mask == 0
            for layer in m.head.layers:
                h = layer(h, src_key_padding_mask=pad)
        mm = marker_mask != 0
        idx = marker_pos.clamp(min=0)[:, :, None].expand(-1, -1, h.size(-1))
        picked = torch.gather(h, 1, idx)
        logits = m.scorer(picked).squeeze(-1).float()
        logits = logits.masked_fill(~mm, -1e4)

        p = torch.softmax(logits, -1)
        k = mm.sum(-1).clamp(min=2).float()
        ent = -(p * torch.log(p.clamp_min(1e-9))).sum(-1) / torch.log(k)
        top2 = p.topk(2, -1).values
        feats = torch.stack([top2[:, 0], top2[:, 0] - top2[:, 1], ent, k / 255.0], -1)
        pooled = h[:, 0].float()
        act_logits = m.act_head(torch.cat([pooled, feats], -1))
        return logits, act_logits


def build(model_dir: str):
    from safetensors.torch import load_file
    from transformers import AutoConfig, AutoModel

    cfg = json.load(open(os.path.join(model_dir, "rl_agent_config.json")))
    enc_dir = os.path.join(model_dir, "encoder")
    if os.path.isdir(enc_dir):
        ecfg = AutoConfig.from_pretrained(enc_dir)
        enc = AutoModel.from_config(ecfg, attn_implementation="eager")
    else:
        enc = AutoModel.from_pretrained(cfg["encoder"], attn_implementation="eager")
    try:
        enc.config.reference_compile = False
    except Exception:
        pass
    model = DecisionModel(enc, cfg.get("head_layers", 2), len(cfg.get("act_costs", {})) + 1)
    weights = load_file(os.path.join(model_dir, "model.safetensors"))
    weights = {k: v.float() for k, v in weights.items()}
    missing, unexpected = model.load_state_dict(weights, strict=False)
    missing = [k for k in missing if not k.endswith("temperature")]
    if missing or unexpected:
        raise RuntimeError("state dict mismatch: missing=%s unexpected=%s" % (missing[:5], unexpected[:5]))
    return model.float().eval(), cfg


def fetch(name: str, token=None) -> str:
    from huggingface_hub import snapshot_download

    sub = CHECKPOINTS[name]
    patterns = ["%s/*" % sub] if sub else ["model.safetensors", "rl_agent_config.json", "encoder/*", "tokenizer/*"]
    root = snapshot_download(BUNDLE_REPO, allow_patterns=patterns, token=token or os.environ.get("HF_TOKEN"))
    return os.path.join(root, sub) if sub else root


def special_ids(tok_json: str) -> dict:
    """Ids of cls/sep/mask/pad, taking the names from tokenizer_config.json next to tokenizer.json."""
    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(tok_json)
    tcfg_path = os.path.join(os.path.dirname(tok_json), "tokenizer_config.json")
    tcfg = json.load(open(tcfg_path)) if os.path.exists(tcfg_path) else {}
    names = {
        "cls": tcfg.get("cls_token", "[CLS]"),
        "sep": tcfg.get("sep_token", "[SEP]"),
        "mask": tcfg.get("mask_token", "[MASK]"),
        "pad": tcfg.get("pad_token", "[PAD]"),
    }
    names = {k: (v["content"] if isinstance(v, dict) else v) for k, v in names.items()}
    out = {}
    for k, v in names.items():
        i = tok.token_to_id(v)
        if i is None:
            raise RuntimeError("tokenizer has no %s token %r" % (k, v))
        out[k] = i
    out["names"] = names
    return out


def sample_batch(tok_json: str, sp: dict, n=3):
    """A small realistic batch to trace with (shapes are dynamic; content does not matter)."""
    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(tok_json)
    ids_of = lambda s: tok.encode(s, add_special_tokens=False).ids
    rows, markers = [], []
    texts = [
        ("choice question: Which department?", ["billing: refunds", "technical: bugs", "other"]),
        ("score question: How urgent?", ["level 0: not urgent", "level 1: soon"]),
        ("noul question: Is this spam?", ["false: no", "true: yes"]),
    ][:n]
    state = ids_of('{"subject": "Duplicate charge", "body": "We were billed twice, please refund."}')
    for head, opts in texts:
        ids = [sp["cls"]] + ids_of(head) + [sp["sep"]]
        mk = []
        for o in opts:
            mk.append(len(ids))
            ids += [sp["mask"]] + ids_of(" " + o)
        ids += [sp["sep"]] + state + [sp["sep"]]
        rows.append(ids)
        markers.append(mk)
    L = max(len(r) for r in rows)
    K = max(len(m) for m in markers)
    input_ids = torch.full((n, L), sp["pad"], dtype=torch.int64)
    att = torch.zeros((n, L), dtype=torch.int64)
    mpos = torch.zeros((n, K), dtype=torch.int64)
    mmask = torch.zeros((n, K), dtype=torch.int64)
    for i, (r, mk) in enumerate(zip(rows, markers)):
        input_ids[i, : len(r)] = torch.tensor(r)
        att[i, : len(r)] = 1
        mpos[i, : len(mk)] = torch.tensor(mk)
        mmask[i, : len(mk)] = 1
    qtype = torch.tensor([0, 1, 2][:n], dtype=torch.int64)
    return input_ids, att, mpos, mmask, qtype


def export_one(name: str, out_root: str, token=None, verify=True, quantize=False):
    import onnx

    t0 = time.time()
    src = fetch(name, token)
    out = os.path.join(out_root, name)
    os.makedirs(out, exist_ok=True)
    print("[%s] source: %s" % (name, src), flush=True)

    model, cfg = build(src)
    wrapper = ExportWrapper(model).eval()
    # nn.TransformerEncoderLayer's fused fast path (aten::_transformer_encoder_layer_fwd) is not
    # exportable; force the plain composite implementation.
    torch.backends.mha.set_fastpath_enabled(False)

    tok_dir = os.path.join(src, "tokenizer")
    shutil.copyfile(os.path.join(tok_dir, "tokenizer.json"), os.path.join(out, "tokenizer.json"))
    os.chmod(os.path.join(out, "tokenizer.json"), 0o644)
    sp = special_ids(os.path.join(tok_dir, "tokenizer.json"))

    laya_cfg = {
        "name": name,
        "repo": BUNDLE_REPO + ("/" + CHECKPOINTS[name] if CHECKPOINTS[name] else ""),
        "encoder": cfg.get("encoder"),
        "max_len": int(cfg.get("max_len", 512)),
        "head_max_len": int(cfg.get("head_max_len", 192)),
        "temperature": cfg.get("temperature", [1.0, 1.0, 1.0]),
        "temperature_by_options": cfg.get("temperature_by_options", {}),
        "act_costs": cfg.get("act_costs", {}),
        "special_tokens": {
            "cls_id": sp["cls"], "sep_id": sp["sep"], "mask_id": sp["mask"], "pad_id": sp["pad"],
            "cls": sp["names"]["cls"], "sep": sp["names"]["sep"],
            "mask": sp["names"]["mask"], "pad": sp["names"]["pad"],
        },
        "onnx": {"file": "model.onnx", "opset": OPSET, "inputs": INPUT_NAMES, "outputs": OUTPUT_NAMES},
    }
    json.dump(laya_cfg, open(os.path.join(out, "laya_config.json"), "w"), indent=2)

    ex = sample_batch(os.path.join(out, "tokenizer.json"), sp)
    with torch.no_grad():
        ref_logits, ref_act = wrapper(*ex)

    onnx_path = os.path.join(out, "model.onnx")
    data_name = "model.onnx.data"
    for f in (onnx_path, os.path.join(out, data_name)):
        if os.path.exists(f):
            os.remove(f)  # onnx appends to an existing external-data file
    print("[%s] exporting to %s ..." % (name, onnx_path), flush=True)
    B, S, K = torch.export.Dim("batch"), torch.export.Dim("seq"), torch.export.Dim("k", min=2)
    with torch.no_grad():
        torch.onnx.export(
            wrapper, ex, onnx_path,
            input_names=INPUT_NAMES,
            output_names=OUTPUT_NAMES,
            dynamic_shapes={
                "input_ids": {0: B, 1: S},
                "attention_mask": {0: B, 1: S},
                "marker_pos": {0: B, 1: K},
                "marker_mask": {0: B, 1: K},
                "qtype": {0: B},
            },
            opset_version=OPSET, dynamo=True, external_data=True, optimize=True,
        )
    for f in os.listdir(out):  # stray files from earlier runs
        p = os.path.join(out, f)
        if f not in ("model.onnx", data_name, "tokenizer.json", "laya_config.json", "model.int8.onnx") and os.path.isfile(p):
            os.remove(p)
    onnx.checker.check_model(onnx_path)
    print("[%s] wrote %s (%.0f MB data) in %.0fs" % (
        name, onnx_path, os.path.getsize(os.path.join(out, data_name)) / 1e6, time.time() - t0), flush=True)

    if verify:
        import onnxruntime as ort

        sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
        feeds = {n: t.numpy() for n, t in zip(INPUT_NAMES, ex)}
        t1 = time.time()
        lo, ac = sess.run(None, feeds)
        dt = time.time() - t1
        d1 = np.abs(lo - ref_logits.numpy()).max()
        d2 = np.abs(ac - ref_act.numpy()).max()
        print("[%s] verify: max|Δlogits|=%.2e max|Δact|=%.2e  (ort %.0f ms, batch=%d seq=%d)" % (
            name, d1, d2, dt * 1000, ex[0].shape[0], ex[0].shape[1]), flush=True)
        if d1 > 1e-2 or d2 > 1e-2:
            raise RuntimeError("ONNX output diverges from PyTorch")

    if quantize:
        from onnxruntime.quantization import QuantType, quantize_dynamic

        q_path = os.path.join(out, "model.int8.onnx")
        print("[%s] dynamic int8 quantization -> %s" % (name, q_path), flush=True)
        # The exporter leaves value_info shape annotations that trip ORT's pre-quantization shape
        # inference; the graph is correct without them, so strip them from a temporary copy.
        m = onnx.load(onnx_path, load_external_data=False)
        del m.graph.value_info[:]
        tmp = os.path.join(out, "model.novi.onnx")
        onnx.save(m, tmp)  # still references model.onnx.data
        try:
            quantize_dynamic(tmp, q_path, weight_type=QuantType.QInt8)
        finally:
            os.remove(tmp)
        print("[%s] wrote %s (%.0f MB). Note: int8 answers differ slightly from fp32 (dynamic quantization)." % (
            name, q_path, os.path.getsize(q_path) / 1e6), flush=True)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("name", choices=list(CHECKPOINTS) + ["all"])
    ap.add_argument("--out", default="models", help="output root (default: models/)")
    ap.add_argument("--token", default=None, help="HF token (or set HF_TOKEN)")
    ap.add_argument("--no-verify", action="store_true")
    ap.add_argument("--int8", action="store_true", help="also write a dynamically quantized model.int8.onnx")
    a = ap.parse_args()
    names = list(CHECKPOINTS) if a.name == "all" else [a.name]
    for n in names:
        export_one(n, a.out, token=a.token, verify=not a.no_verify, quantize=a.int8)


if __name__ == "__main__":
    sys.exit(main())
