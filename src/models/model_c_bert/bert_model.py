"""Fine-tune BERT (bert-base-uncased) on segment transcripts.

*** GPU REQUIRED. Do not run on Mac. ***
*** Intended environment: RunPod RTX 4090 pod or Google Colab. ***

Fine-tunes bert-base-uncased for binary classification (viral vs not)
using the segment transcript as input. 5 seeds (RANDOM_SEEDS), early
stopping on val AUC-ROC, AdamW optimizer, learning rate 2e-5.

Input CSVs (produced by prepare_data.py):
  data/post_extraction/bert_data_{train,val,test}.csv
Columns: segment_id, video_id, creator_id, transcript_text, label, split

Output (one per seed):
  results/bert_seed{N}_test.json        single-seed EvalResult
Plus aggregate:
  results/bert_test.json                 aggregated across seeds
Plus checkpoints:
  results/bert_checkpoints/bert_seed{N}/ HuggingFace save_pretrained output
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

# GPU-only imports; fail clearly if missing
try:
    import torch
    from torch.utils.data import DataLoader, Dataset
    from transformers import (
        BertForSequenceClassification, BertTokenizerFast,
        get_linear_schedule_with_warmup,
    )
    TORCH_AVAILABLE = True
except ImportError as e:
    TORCH_AVAILABLE = False
    _IMPORT_ERROR = e

from src.evaluation import metrics as eval_metrics
from src.models.shared.seeds import RANDOM_SEEDS
from src.post_extraction import config as post_config

RESULTS_DIR = Path(__file__).resolve().parent / "results"
CHECKPOINTS_DIR = RESULTS_DIR / "bert_checkpoints"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)

MODEL_NAME = "bert-base-uncased"
MAX_LENGTH = 256          # 30s of speech ~ 80-100 words ~ 150 tokens; 256 is safe
BATCH_SIZE_TRAIN = 32
BATCH_SIZE_EVAL = 128
LR = 2e-5
EPOCHS = 3                # with 2,277 positives, 3 epochs is typical before overfit
WARMUP_STEPS = 500
PATIENCE = 1              # early stop on val AUC-ROC worsening by 1 epoch

log = logging.getLogger("clipwhy.model_c.bert")


class TranscriptDataset(Dataset):
    def __init__(self, df, tokenizer, max_length):
        self.texts = df["transcript_text"].fillna("").astype(str).tolist()
        self.labels = df["label"].astype(int).tolist()
        self.segment_ids = df["segment_id"].tolist()
        self.video_ids = df["video_id"].tolist()
        self.tok = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        text = self.texts[idx] if self.texts[idx].strip() else "[empty]"
        enc = self.tok(text, truncation=True, padding="max_length",
                       max_length=self.max_length, return_tensors="pt")
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "label": torch.tensor(self.labels[idx], dtype=torch.long),
            "segment_id": self.segment_ids[idx],
            "video_id": self.video_ids[idx],
        }


def _seed_all(seed: int) -> None:
    import random as _random
    _random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _evaluate(model, loader, device):
    model.eval()
    all_scores, all_labels, all_vids = [], [], []
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attn = batch["attention_mask"].to(device)
            logits = model(input_ids=input_ids, attention_mask=attn).logits
            probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
            all_scores.append(probs)
            all_labels.append(batch["label"].numpy())
            all_vids.extend(batch["video_id"])
    return (np.concatenate(all_scores),
            np.concatenate(all_labels),
            np.asarray(all_vids))


def run_seed(seed: int, train_df, val_df, test_df, tokenizer, device) -> dict:
    _seed_all(seed)
    model = BertForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
    model.to(device)

    train_ds = TranscriptDataset(train_df, tokenizer, MAX_LENGTH)
    val_ds = TranscriptDataset(val_df, tokenizer, MAX_LENGTH)
    test_ds = TranscriptDataset(test_df, tokenizer, MAX_LENGTH)

    # Class-weighted loss for imbalance
    pos = int((train_df["label"] == 1).sum())
    neg = int((train_df["label"] == 0).sum())
    class_weights = torch.tensor([1.0, float(neg) / max(pos, 1)], device=device)
    loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE_TRAIN, shuffle=True,
                              num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE_EVAL, shuffle=False,
                            num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE_EVAL, shuffle=False,
                             num_workers=2, pin_memory=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    total_steps = len(train_loader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=WARMUP_STEPS, num_training_steps=total_steps,
    )

    best_val_auc = -1.0
    best_state = None
    stale = 0
    for epoch in range(EPOCHS):
        model.train()
        for i, batch in enumerate(train_loader):
            optimizer.zero_grad()
            logits = model(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
            ).logits
            loss = loss_fn(logits, batch["label"].to(device))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            if i % 200 == 0:
                log.info("seed=%d epoch=%d step=%d loss=%.4f", seed, epoch, i, loss.item())

        v_scores, v_labels, v_vids = _evaluate(model, val_loader, device)
        val_res = eval_metrics.evaluate(v_labels, v_scores, v_vids,
                                        split="val", model="bert", seed=seed)
        log.info("seed=%d epoch=%d val AUC=%.4f NDCG@10=%.4f",
                 seed, epoch, val_res.auc_roc, val_res.ndcg_at_k[10])
        if val_res.auc_roc > best_val_auc:
            best_val_auc = val_res.auc_roc
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale > PATIENCE:
                log.info("Early stop at epoch %d", epoch)
                break

    if best_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})

    ckpt_dir = CHECKPOINTS_DIR / f"bert_seed{seed}"
    model.save_pretrained(ckpt_dir)
    tokenizer.save_pretrained(ckpt_dir)

    t_scores, t_labels, t_vids = _evaluate(model, test_loader, device)
    test_res = eval_metrics.evaluate(t_labels, t_scores, t_vids,
                                     split="test", model="bert", seed=seed,
                                     notes=f"best_val_auc={best_val_auc:.4f}")
    return test_res.to_dict()


def main():
    if not TORCH_AVAILABLE:
        raise RuntimeError(
            f"torch/transformers not available ({_IMPORT_ERROR}). "
            "Install via: pip install 'torch>=2.3' 'transformers>=4.37'"
        )
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", nargs="*", type=int, default=list(RANDOM_SEEDS))
    args = p.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA not available. BERT fine-tuning on CPU is not practical. "
            "Use a GPU pod (RunPod RTX 4090 or Colab T4)."
        )
    device = torch.device("cuda")

    train_df = pd.read_csv(post_config.OUT_DIR / "bert_data_train.csv")
    val_df = pd.read_csv(post_config.OUT_DIR / "bert_data_val.csv")
    test_df = pd.read_csv(post_config.OUT_DIR / "bert_data_test.csv")
    log.info("Loaded train=%d val=%d test=%d (positives: %d / %d / %d)",
             len(train_df), len(val_df), len(test_df),
             int(train_df['label'].sum()), int(val_df['label'].sum()),
             int(test_df['label'].sum()))

    tokenizer = BertTokenizerFast.from_pretrained(MODEL_NAME)

    per_seed_results = []
    for seed in args.seeds:
        log.info("=== seed=%d ===", seed)
        res = run_seed(seed, train_df, val_df, test_df, tokenizer, device)
        (RESULTS_DIR / f"bert_seed{seed}_test.json").write_text(json.dumps(res, indent=2))
        per_seed_results.append(res)

    # Aggregate across seeds
    metric_arrays = {}
    for r in per_seed_results:
        for k, v in r["precision_at_k"].items():
            metric_arrays.setdefault(f"precision_at_{k}", []).append(v)
        for k, v in r["recall_at_k"].items():
            metric_arrays.setdefault(f"recall_at_{k}", []).append(v)
        for k, v in r["ndcg_at_k"].items():
            metric_arrays.setdefault(f"ndcg_at_{k}", []).append(v)
        metric_arrays.setdefault("auc_roc", []).append(r["auc_roc"])
        metric_arrays.setdefault("f1_at_0_5", []).append(r["f1_at_0_5"])

    agg = {"model": "bert", "split": "test", "seeds": [r["seed"] for r in per_seed_results],
           "per_seed": per_seed_results, "hyperparams": {
               "model_name": MODEL_NAME, "max_length": MAX_LENGTH,
               "batch_size_train": BATCH_SIZE_TRAIN, "lr": LR,
               "epochs": EPOCHS, "warmup_steps": WARMUP_STEPS,
           }}
    for name, vals in metric_arrays.items():
        arr = np.asarray(vals, dtype=float)
        agg[name] = {"mean": float(arr.mean()),
                     "std": float(arr.std(ddof=1)) if len(arr) > 1 else 0.0,
                     "n_seeds": len(arr), "values": arr.tolist()}
    (RESULTS_DIR / "bert_test.json").write_text(json.dumps(agg, indent=2))
    log.info("Wrote bert_test.json")


if __name__ == "__main__":
    main()
