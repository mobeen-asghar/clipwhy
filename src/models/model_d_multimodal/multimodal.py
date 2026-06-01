"""Late-fusion multimodal network: BERT [CLS] + audio + visual (+ optional metadata).

*** GPU REQUIRED. Do not run on Mac. ***

Variants and ablation:

  --variant original          (3 branches: text + audio + visual)
  --variant with_metadata     (4 branches: above + metadata)

  --ablate-branch {text,audio,visual,metadata}  (drop a single branch from the
                                                 selected variant; for the
                                                 ablation study)

  --all-branch-ablations      (run every meaningful branch ablation in one
                               invocation, sharing one CLS extraction. Used
                               by run_ablation_on_runpod.sh)

Output filenames:
  multimodal_{variant}_test.json                          full variant
  multimodal_{variant}_seed{N}_test.json                  per seed
  multimodal_{variant}_ablate_{branch}_test.json          ablation aggregated
  multimodal_{variant}_ablate_{branch}_seed{N}_test.json  per seed
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, Dataset
    from transformers import BertModel, BertTokenizerFast
    TORCH_AVAILABLE = True
except ImportError as e:
    TORCH_AVAILABLE = False
    _IMPORT_ERROR = e

from src.evaluation import metrics as eval_metrics
from src.models.shared.data import load_split, feature_categories
from src.models.shared.seeds import RANDOM_SEEDS
from src.post_extraction import config as post_config

RESULTS_DIR = Path(__file__).resolve().parent / "results"
CKPT_DIR = RESULTS_DIR / "multimodal_checkpoints"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
CKPT_DIR.mkdir(parents=True, exist_ok=True)

BERT_CHECKPOINT_DIR = Path(__file__).resolve().parents[1] / "model_c_bert" / "results" / "bert_checkpoints"
MAX_LENGTH = 256
BATCH_SIZE_TRAIN = 32
BATCH_SIZE_EVAL = 128
LR = 1e-4
EPOCHS = 5
PATIENCE = 1

AUDIO_CATS = ["audio_speech", "voice_quality", "audio_events", "audio_emotion"]
VISUAL_CATS = ["visual"]
METADATA_CATS = ["structural", "creator_context"]

log = logging.getLogger("clipwhy.model_d.multimodal")


class MultimodalNet(nn.Module):
    """Late fusion. Each branch can be turned on or off independently."""

    def __init__(self, *, bert_dim=768, audio_dim=20, visual_dim=42, metadata_dim=None,
                 use_text=True, use_audio=True, use_visual=True, use_metadata=False):
        super().__init__()
        self.use_text = use_text
        self.use_audio = use_audio
        self.use_visual = use_visual
        self.use_metadata = use_metadata
        n_branches = sum([use_text, use_audio, use_visual, use_metadata])
        if n_branches == 0:
            raise ValueError("At least one branch must be enabled")
        if use_text:
            self.text_head = nn.Sequential(
                nn.Linear(bert_dim, 128), nn.ReLU(), nn.Dropout(0.2),
            )
        if use_audio:
            self.audio_branch = nn.Sequential(
                nn.Linear(audio_dim, 64), nn.ReLU(),
                nn.Linear(64, 128), nn.ReLU(), nn.Dropout(0.2),
            )
        if use_visual:
            self.visual_branch = nn.Sequential(
                nn.Linear(visual_dim, 64), nn.ReLU(),
                nn.Linear(64, 128), nn.ReLU(), nn.Dropout(0.2),
            )
        if use_metadata:
            assert metadata_dim and metadata_dim > 0
            self.metadata_branch = nn.Sequential(
                nn.Linear(metadata_dim, 64), nn.ReLU(),
                nn.Linear(64, 128), nn.ReLU(), nn.Dropout(0.2),
            )
        self.fusion = nn.Sequential(
            nn.Linear(128 * n_branches, 128), nn.ReLU(),
            nn.Linear(128, 2),
        )

    def forward(self, cls_emb=None, audio_vec=None, visual_vec=None, metadata_vec=None):
        parts = []
        if self.use_text:
            parts.append(self.text_head(cls_emb))
        if self.use_audio:
            parts.append(self.audio_branch(audio_vec))
        if self.use_visual:
            parts.append(self.visual_branch(visual_vec))
        if self.use_metadata:
            parts.append(self.metadata_branch(metadata_vec))
        return self.fusion(torch.cat(parts, dim=1))


class MultimodalDataset(Dataset):
    def __init__(self, cls_embs, audio, visual, labels, video_ids, segment_ids,
                 metadata=None, *,
                 use_text=True, use_audio=True, use_visual=True, use_metadata=False):
        self.use_text = use_text
        self.use_audio = use_audio
        self.use_visual = use_visual
        self.use_metadata = use_metadata
        self.cls_embs = torch.as_tensor(cls_embs, dtype=torch.float32) if use_text else None
        self.audio = torch.as_tensor(audio, dtype=torch.float32) if use_audio else None
        self.visual = torch.as_tensor(visual, dtype=torch.float32) if use_visual else None
        self.metadata = (torch.as_tensor(metadata, dtype=torch.float32)
                         if use_metadata else None)
        self.labels = torch.as_tensor(labels, dtype=torch.long)
        self.video_ids = np.asarray(video_ids)
        self.segment_ids = np.asarray(segment_ids)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {"label": self.labels[idx], "video_id": self.video_ids[idx]}
        if self.use_text:
            item["cls"] = self.cls_embs[idx]
        if self.use_audio:
            item["audio"] = self.audio[idx]
        if self.use_visual:
            item["visual"] = self.visual[idx]
        if self.use_metadata:
            item["metadata"] = self.metadata[idx]
        return item


def _precompute_cls_embeddings(bert_model, tokenizer, transcripts, device):
    log.info("Precomputing CLS embeddings for %d segments...", len(transcripts))
    bert_model.eval()
    all_cls = []
    for i in range(0, len(transcripts), BATCH_SIZE_EVAL):
        batch = transcripts[i:i + BATCH_SIZE_EVAL]
        texts = []
        for t in batch:
            if t is None or (isinstance(t, float) and pd.isna(t)):
                texts.append("[empty]")
            else:
                s = str(t).strip()
                texts.append(s if s else "[empty]")
        enc = tokenizer(texts, truncation=True, padding=True,
                        max_length=MAX_LENGTH, return_tensors="pt").to(device)
        with torch.no_grad():
            out = bert_model(**enc)
        cls = out.last_hidden_state[:, 0, :].cpu().numpy()
        all_cls.append(cls)
        if i % 10000 == 0 and i > 0:
            log.info("  precomputed %d / %d", i, len(transcripts))
    return np.concatenate(all_cls, axis=0).astype(np.float32)


def _slice_features(X, feature_names, categories):
    groups = feature_categories()
    wanted = set()
    for c in categories:
        if c not in groups:
            raise ValueError(f"Unknown feature group {c!r}")
        wanted.update(groups[c])
    idx = [i for i, n in enumerate(feature_names) if n in wanted]
    return X[:, idx], [feature_names[i] for i in idx]


def _evaluate(model, loader, device, *, use_text, use_audio, use_visual, use_metadata):
    model.eval()
    scores, labels, vids = [], [], []
    with torch.no_grad():
        for b in loader:
            cls = b["cls"].to(device) if use_text else None
            au = b["audio"].to(device) if use_audio else None
            vi = b["visual"].to(device) if use_visual else None
            md = b["metadata"].to(device) if use_metadata else None
            logits = model(cls_emb=cls, audio_vec=au, visual_vec=vi, metadata_vec=md)
            probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
            scores.append(probs)
            labels.append(b["label"].numpy())
            vids.extend(b["video_id"])
    return np.concatenate(scores), np.concatenate(labels), np.asarray(vids)


def run_seed(*, seed, model_label, cls, audio, visual, metadata,
             y, vids, sids, device,
             use_text, use_audio, use_visual, use_metadata):
    """Train one seed of multimodal with the given branch configuration."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    md_dim = metadata["train"].shape[1] if use_metadata else None
    audio_dim = audio["train"].shape[1] if use_audio else 1
    visual_dim = visual["train"].shape[1] if use_visual else 1
    model = MultimodalNet(
        audio_dim=audio_dim, visual_dim=visual_dim, metadata_dim=md_dim,
        use_text=use_text, use_audio=use_audio, use_visual=use_visual,
        use_metadata=use_metadata,
    ).to(device)
    n_branches = sum([use_text, use_audio, use_visual, use_metadata])
    log.info("[%s seed=%d] params=%d branches=%d (text=%s audio=%s visual=%s meta=%s)",
             model_label, seed, sum(p.numel() for p in model.parameters()),
             n_branches, use_text, use_audio, use_visual, use_metadata)

    pos = int((y["train"] == 1).sum())
    neg = int((y["train"] == 0).sum())
    loss_fn = nn.CrossEntropyLoss(
        weight=torch.tensor([1.0, float(neg) / max(pos, 1)], device=device)
    )

    def _ds(split):
        return MultimodalDataset(
            cls[split] if use_text else None,
            audio[split] if use_audio else None,
            visual[split] if use_visual else None,
            y[split], vids[split], sids[split],
            metadata=metadata[split] if use_metadata else None,
            use_text=use_text, use_audio=use_audio,
            use_visual=use_visual, use_metadata=use_metadata,
        )

    train_loader = DataLoader(_ds("train"), batch_size=BATCH_SIZE_TRAIN, shuffle=True,
                              num_workers=2, pin_memory=True)
    val_loader = DataLoader(_ds("val"), batch_size=BATCH_SIZE_EVAL, shuffle=False,
                            num_workers=2, pin_memory=True)
    test_loader = DataLoader(_ds("test"), batch_size=BATCH_SIZE_EVAL, shuffle=False,
                             num_workers=2, pin_memory=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)

    best_val_auc = -1.0
    best_state = None
    stale = 0
    for epoch in range(EPOCHS):
        model.train()
        for b in train_loader:
            optimizer.zero_grad()
            cls_b = b["cls"].to(device) if use_text else None
            au_b = b["audio"].to(device) if use_audio else None
            vi_b = b["visual"].to(device) if use_visual else None
            md_b = b["metadata"].to(device) if use_metadata else None
            logits = model(cls_emb=cls_b, audio_vec=au_b, visual_vec=vi_b, metadata_vec=md_b)
            loss = loss_fn(logits, b["label"].to(device))
            loss.backward()
            optimizer.step()
        vs, vl, vv = _evaluate(model, val_loader, device,
                                use_text=use_text, use_audio=use_audio,
                                use_visual=use_visual, use_metadata=use_metadata)
        val_res = eval_metrics.evaluate(vl, vs, vv, split="val",
                                        model=model_label, seed=seed)
        log.info("  [%s seed=%d] epoch=%d val AUC=%.4f NDCG@10=%.4f",
                 model_label, seed, epoch, val_res.auc_roc, val_res.ndcg_at_k[10])
        if val_res.auc_roc > best_val_auc:
            best_val_auc = val_res.auc_roc
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale > PATIENCE:
                log.info("  [%s seed=%d] early stop at epoch %d", model_label, seed, epoch)
                break

    if best_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})

    ts, tl, tv = _evaluate(model, test_loader, device,
                            use_text=use_text, use_audio=use_audio,
                            use_visual=use_visual, use_metadata=use_metadata)
    return eval_metrics.evaluate(tl, ts, tv, split="test", model=model_label,
                                  seed=seed, notes=f"best_val_auc={best_val_auc:.4f}")


def _save_checkpoint(model_state, model_label, seed):
    """Save trained weights for explainability use later."""
    pass  # skip for ablation runs to save disk; full variant runs already saved


def _branch_flags_for(variant, ablate_branch):
    """Translate (variant, ablate_branch) -> (use_text, use_audio, use_visual, use_metadata)."""
    base = {
        "original": dict(use_text=True, use_audio=True, use_visual=True, use_metadata=False),
        "with_metadata": dict(use_text=True, use_audio=True, use_visual=True, use_metadata=True),
    }
    if variant not in base:
        raise ValueError(f"Unknown variant: {variant}")
    flags = dict(base[variant])
    if ablate_branch is None:
        return flags
    key = f"use_{ablate_branch}"
    if key not in flags or not flags[key]:
        raise ValueError(f"Cannot ablate {ablate_branch!r}: not active in variant {variant!r}")
    flags[key] = False
    return flags


def _label_for(variant, ablate_branch):
    if ablate_branch is None:
        return f"multimodal_{variant}"
    return f"multimodal_{variant}_ablate_{ablate_branch}"


def _filename_for(variant, ablate_branch, *, seed=None):
    base = _label_for(variant, ablate_branch)
    return f"{base}_seed{seed}_test.json" if seed is not None else f"{base}_test.json"


def run_one_configuration(variant, ablate_branch, *,
                           cls_dict, audio_dict, visual_dict, metadata_dict,
                           y_dict, vid_dict, sid_dict, device, seeds):
    """Train all 5 seeds for one (variant, ablation) configuration."""
    flags = _branch_flags_for(variant, ablate_branch)
    label = _label_for(variant, ablate_branch)
    log.info("=== %s ===", label.upper())

    per_seed_results = []
    for seed in seeds:
        res = run_seed(
            seed=seed, model_label=label,
            cls=cls_dict, audio=audio_dict, visual=visual_dict,
            metadata=metadata_dict if flags["use_metadata"] else {},
            y=y_dict, vids=vid_dict, sids=sid_dict, device=device,
            **flags,
        )
        out_path = RESULTS_DIR / _filename_for(variant, ablate_branch, seed=seed)
        out_path.write_text(json.dumps(res.to_dict(), indent=2))
        per_seed_results.append(res)
        log.info("  Wrote %s: AUC=%.4f", out_path.name, res.auc_roc)

    agg = eval_metrics.aggregate_seed_results(per_seed_results)
    agg["per_seed"] = [r.to_dict() for r in per_seed_results]
    agg["hyperparams"] = {
        "variant": variant,
        "ablate_branch": ablate_branch,
        "branches_used": flags,
        "max_length": MAX_LENGTH,
        "batch_size_train": BATCH_SIZE_TRAIN,
        "lr": LR, "epochs": EPOCHS,
    }
    out_path = RESULTS_DIR / _filename_for(variant, ablate_branch)
    out_path.write_text(json.dumps(agg, indent=2))
    log.info("Wrote %s (mean AUC=%.4f +/- %.4f)",
             out_path.name, agg["auc_roc"]["mean"], agg["auc_roc"]["std"])
    return agg


def main():
    if not TORCH_AVAILABLE:
        raise RuntimeError(
            f"torch/transformers not available ({_IMPORT_ERROR}). "
            "Install via: pip install 'torch>=2.3' 'transformers>=4.37'"
        )
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    p = argparse.ArgumentParser()
    p.add_argument("--variant", default="original",
                   choices=["original", "with_metadata"])
    p.add_argument("--ablate-branch", default=None,
                   choices=[None, "text", "audio", "visual", "metadata"])
    p.add_argument("--all-branch-ablations", action="store_true",
                   help="Run every meaningful single-branch ablation in one invocation, "
                        "sharing one CLS extraction. Implies --variant with_metadata.")
    p.add_argument("--bert-seed", type=int, default=42)
    p.add_argument("--bert-checkpoint-dir", type=Path, default=None)
    p.add_argument("--seeds", nargs="*", type=int, default=list(RANDOM_SEEDS))
    args = p.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available. Use a GPU pod.")
    device = torch.device("cuda")

    bert_ckpt = (args.bert_checkpoint_dir
                 or BERT_CHECKPOINT_DIR / f"bert_seed{args.bert_seed}")
    if not bert_ckpt.exists():
        raise FileNotFoundError(
            f"No BERT checkpoint at {bert_ckpt}. Run Model C first or pass --bert-checkpoint-dir."
        )
    log.info("BERT checkpoint: %s", bert_ckpt)

    tokenizer = BertTokenizerFast.from_pretrained(bert_ckpt)
    bert = BertModel.from_pretrained(bert_ckpt).to(device)

    splits = {s: load_split(s) for s in ("train", "val", "test")}

    tr = {s: pd.read_csv(post_config.OUT_DIR / f"bert_data_{s}.csv",
                         keep_default_na=False, na_values=[]).set_index("segment_id")
          for s in ("train", "val", "test")}

    def _texts_for(s):
        sd = splits[s]
        idx = tr[s].index
        return [tr[s].loc[sid, "transcript_text"] if sid in idx else ""
                for sid in sd.segment_ids]

    cls_dict = {s: _precompute_cls_embeddings(bert, tokenizer, _texts_for(s), device)
                for s in ("train", "val", "test")}
    audio_dict = {s: _slice_features(splits[s].X, splits[s].feature_names, AUDIO_CATS)[0]
                  for s in ("train", "val", "test")}
    visual_dict = {s: _slice_features(splits[s].X, splits[s].feature_names, VISUAL_CATS)[0]
                   for s in ("train", "val", "test")}
    metadata_dict = {s: _slice_features(splits[s].X, splits[s].feature_names, METADATA_CATS)[0]
                     for s in ("train", "val", "test")}
    y_dict = {s: splits[s].y for s in ("train", "val", "test")}
    vid_dict = {s: splits[s].video_ids for s in ("train", "val", "test")}
    sid_dict = {s: splits[s].segment_ids for s in ("train", "val", "test")}

    del bert
    torch.cuda.empty_cache()

    common = dict(
        cls_dict=cls_dict, audio_dict=audio_dict, visual_dict=visual_dict,
        metadata_dict=metadata_dict, y_dict=y_dict, vid_dict=vid_dict,
        sid_dict=sid_dict, device=device, seeds=args.seeds,
    )

    if args.all_branch_ablations:
        # Use with_metadata as base. Ablate each of the 4 branches.
        # ablate=metadata is equivalent to variant=original (already produced
        # in the main run), so we skip it here unless explicitly forced.
        for branch in ("text", "audio", "visual"):
            run_one_configuration("with_metadata", branch, **common)
    else:
        run_one_configuration(args.variant, args.ablate_branch, **common)


if __name__ == "__main__":
    main()
