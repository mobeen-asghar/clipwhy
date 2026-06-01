"""
Regenerate all nine paper figures from the original result files.

Design rules (per supervisor feedback):
  * 600 DPI output
  * No figure numbers in chart titles (charts carry no embedded title at all;
    the numbered caption lives in the Word document)
  * Error bars on the learning curve (Figure 2)

Data sources:
  Experiment 1 (15 creators, 35 features)  -> clipwhy-pipeline/output/...
  Experiment 2 (394 creators, 84 features) -> clipwhy-v2/data/post_extraction/...

Run:
  clipwhy-demo/venv/bin/python paper-publication/regenerate_figures.py
"""
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from scipy.stats import spearmanr

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
PIPE = ROOT / "clipwhy-pipeline"
V2 = ROOT / "clipwhy-v2"
OUT = Path(__file__).resolve().parent / "figures"
OUT.mkdir(exist_ok=True)

DPI = 600
plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "figure.autolayout": True,
    "savefig.dpi": DPI,
    "axes.grid": True,
    "grid.alpha": 0.3,
})

# Consistent palette
C_RF = "#1f77b4"
C_XGB = "#2ca02c"
C_E1 = "#ff7f0e"
C_E2 = "#1f77b4"
C_CHAMP = "#2ca02c"
C_NEG = "#c0392b"   # removing the category hurts -> important
C_POS = "#27ae60"   # removing the category helps -> noise

CAT_COLORS = {
    "structural": "#1f77b4",
    "creator_context": "#9467bd",
    "visual": "#2ca02c",
    "audio_emotion": "#ff7f0e",
    "audio_speech": "#d62728",
    "audio_events": "#8c564b",
    "voice_quality": "#e377c2",
    "text": "#7f7f7f",
}


def save(fig, name):
    path = OUT / name
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    # also emit a vector PDF (scales losslessly for print)
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {path.name} (+ .pdf)")


# ----------------------------------------------------------------------------
# Figure 1 - positive segments per content category, Exp 1 and Exp 2
# ----------------------------------------------------------------------------
def figure1():
    e1 = {"commentary": 33, "education": 30, "tech": 26, "entertainment": 20, "fitness": 18}
    e2 = {"entertainment": 1347, "fitness": 662, "commentary": 580, "education": 340, "tech": 334}
    cats = ["tech", "education", "entertainment", "fitness", "commentary"]
    labels = ["Technology", "Education", "Entertainment", "Fitness", "Commentary"]
    v1 = [e1[c] for c in cats]
    v2 = [e2[c] for c in cats]
    x = np.arange(len(cats))
    w = 0.38

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11, 4.6))
    b1 = axL.bar(x, v1, color=C_E1, edgecolor="black", linewidth=0.4)
    axL.set_ylabel("Positive segments")
    axL.set_xticks(x)
    axL.set_xticklabels(labels, rotation=30, ha="right")
    axL.set_title("Experiment 1 (127 positives)")
    axL.bar_label(b1, padding=2, fontsize=9)
    axL.margins(y=0.15)

    b2 = axR.bar(x, v2, color=C_E2, edgecolor="black", linewidth=0.4)
    axR.set_ylabel("Positive segments")
    axR.set_xticks(x)
    axR.set_xticklabels(labels, rotation=30, ha="right")
    axR.set_title("Experiment 2 (3,263 positives)")
    axR.bar_label(b2, padding=2, fontsize=9)
    axR.margins(y=0.15)
    save(fig, "fig1_positives_per_category.png")


# ----------------------------------------------------------------------------
# Figure 2 - Experiment 1 learning curve with error bars
# ----------------------------------------------------------------------------
def figure2():
    sc = json.load(open(PIPE / "output/results/scaling_experiment.json"))
    rf = sc["rf"]
    xgb = sc["xgboost"]
    pos = [r["approx_positives"] for r in rf]
    rf_m = [r["auc_roc_mean"] for r in rf]
    rf_s = [r["auc_roc_std"] for r in rf]
    xgb_m = [r["auc_roc_mean"] for r in xgb]
    xgb_s = [r["auc_roc_std"] for r in xgb]
    rule = sc["rule_based"]["auc_roc"]

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.errorbar(pos, rf_m, yerr=rf_s, marker="o", color=C_RF, capsize=4,
                lw=2, label="Random Forest (5 seeds)")
    ax.errorbar(pos, xgb_m, yerr=xgb_s, marker="s", color=C_XGB, capsize=4,
                lw=2, label="XGBoost (5 seeds)")
    ax.axhline(rule, ls="--", color=C_NEG, lw=1.5, label=f"Rule-based baseline ({rule:.3f})")
    ax.axhline(0.5, ls=":", color="grey", lw=1.5, label="Random (0.500)")
    ax.set_xlabel("Number of positive training examples")
    ax.set_ylabel("AUC-ROC (test split)")
    ax.legend(loc="lower right", framealpha=0.95)
    ax.set_ylim(0.45, 0.70)
    save(fig, "fig2_exp1_learning_curve.png")


# ----------------------------------------------------------------------------
# Figure 3 - Experiment 1 vs Experiment 2 AUC-ROC per matched model
# ----------------------------------------------------------------------------
def figure3():
    pm = json.load(open(V2 / "data/post_extraction/results/model_comparison.json"))["per_model"]
    mc = {e["model"]: e for e in pm}

    def v2auc(key):
        return mc[key]["auc_roc_mean"]

    models = ["Random", "Rule-based", "Random Forest", "XGBoost", "BERT", "Multi-modal\n(3-branch)"]
    # Experiment 1 (verified against Table 6 / aggregated JSONs)
    e1 = [0.5147, 0.6080, 0.6030, 0.5479, 0.5664, 0.4724]
    e2 = [v2auc("random"), v2auc("rule_based"), v2auc("rf"),
          v2auc("xgboost"), v2auc("bert"), v2auc("multimodal_original")]
    champ = v2auc("multimodal_with_metadata")

    x = np.arange(len(models))
    w = 0.38
    fig, ax = plt.subplots(figsize=(11, 5.5))
    b1 = ax.bar(x - w / 2, e1, w, color=C_E1, edgecolor="black", linewidth=0.4,
                label="Experiment 1 (127 positives)")
    b2 = ax.bar(x + w / 2, e2, w, color=C_E2, edgecolor="black", linewidth=0.4,
                label="Experiment 2 (3,263 positives)")
    # champion as a separate trailing bar
    xc = len(models)
    bc = ax.bar([xc], [champ], w, color=C_CHAMP, edgecolor="black", linewidth=0.4,
                hatch="//", label="Multi-modal 4-branch (Exp 2)")
    ax.axhline(0.5, ls=":", color="grey", lw=1.5, label="Random (0.500)")
    for bars in (b1, b2, bc):
        ax.bar_label(bars, fmt="%.3f", padding=2, fontsize=8)
    ax.set_ylabel("AUC-ROC (test split)")
    ax.set_xticks(list(x) + [xc])
    ax.set_xticklabels(models + ["Multi-modal\n4-branch"], fontsize=9)
    ax.set_ylim(0.44, 0.78)
    ax.legend(loc="upper left", framealpha=0.95, fontsize=9)
    save(fig, "fig3_exp1_vs_exp2_auc.png")


# ----------------------------------------------------------------------------
# Helpers for Spearman / category mapping (Exp 2)
# ----------------------------------------------------------------------------
def _v2_category_map(feat_cols):
    struct = {"position_ratio", "is_intro", "is_outro", "segment_duration",
              "video_duration", "is_first_segment", "is_last_segment",
              "segment_novelty_to_neighbors"}
    aspeech = {"energy_mean", "energy_var", "energy_first_3s_ratio", "pitch_range",
               "pitch_std", "speaking_rate_audio", "silence_ratio"}
    aevents = {"music_presence", "music_fraction", "speech_music_ratio", "laughter_peak"}
    aemotion = {"arousal_mean", "valence_mean", "dominance_mean", "arousal_std",
                "arousal_peak", "arousal_arc_direction", "valence_arc_direction"}
    vquality = {"jitter_local", "shimmer_local"}
    cmap = {}
    for c in feat_cols:
        if c in struct:
            cmap[c] = "structural"
        elif c.startswith("creator_category"):
            cmap[c] = "creator_context"
        elif c in aspeech:
            cmap[c] = "audio_speech"
        elif c in aevents:
            cmap[c] = "audio_events"
        elif c in aemotion:
            cmap[c] = "audio_emotion"
        elif c in vquality:
            cmap[c] = "voice_quality"
        elif c.startswith("clip_pca") or c in {
                "dover_aesthetic_score", "dover_technical_score", "colorfulness",
                "brightness_mean", "cut_count", "cuts_per_second", "face_present_ratio",
                "largest_face_area_ratio_max", "face_count_median"}:
            cmap[c] = "visual"
        else:
            cmap[c] = "text"
    return cmap


def _load_v2():
    return pd.read_csv(V2 / "data/post_extraction/segments_with_splits.csv")


# ----------------------------------------------------------------------------
# Figure 4 - top-30 features by absolute Spearman correlation (Exp 2)
# ----------------------------------------------------------------------------
def figure4(v2):
    meta = {"segment_id", "video_id", "creator_id", "category", "segment_index",
            "label", "features_version", "extracted_at", "split"}
    feat_cols = [c for c in v2.columns if c not in meta]
    cmap = _v2_category_map(feat_cols)
    y = v2.label.values
    rows = []
    for c in feat_cols:
        x = v2[c].values
        if np.nanstd(x) == 0:
            r = 0.0
        else:
            r, _ = spearmanr(x, y)
            r = 0.0 if r != r else r
        rows.append((c, abs(r)))
    rows.sort(key=lambda kv: -kv[1])
    top = rows[:30][::-1]  # reverse so largest at top of horizontal bar chart
    names = [c for c, _ in top]
    vals = [v for _, v in top]
    colors = [CAT_COLORS[cmap[c]] for c in names]

    fig, ax = plt.subplots(figsize=(9, 10))
    ax.barh(range(len(names)), vals, color=colors, edgecolor="black", linewidth=0.3)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=8)
    ax.set_xlabel("|Spearman correlation| with positive label")
    used = []
    for cat in CAT_COLORS:
        if any(cmap[c] == cat for c in names):
            used.append(Patch(facecolor=CAT_COLORS[cat], edgecolor="black", label=cat))
    ax.legend(handles=used, loc="lower right", fontsize=8, framealpha=0.95)
    ax.grid(axis="y", visible=False)
    save(fig, "fig4_exp2_spearman_top30.png")


# ----------------------------------------------------------------------------
# Figure 5 - position_ratio histogram, positive vs negative (Exp 2 train)
# ----------------------------------------------------------------------------
def figure5(v2):
    tr = v2[v2.split == "train"]
    pos = tr[tr.label == 1].position_ratio.values
    neg = tr[tr.label == 0].position_ratio.values
    bins = np.linspace(0, 1, 21)
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.hist(neg, bins=bins, density=True, alpha=0.55, color="#7f7f7f",
            label=f"Negative (n={len(neg):,})", edgecolor="white", linewidth=0.3)
    ax.hist(pos, bins=bins, density=True, alpha=0.7, color=C_NEG,
            label=f"Positive (n={len(pos):,})", edgecolor="white", linewidth=0.3)
    ax.set_xlabel("Segment position in video (position_ratio: 0 = start, 1 = end)")
    ax.set_ylabel("Density")
    ax.legend(framealpha=0.95)
    save(fig, "fig5_exp2_position_ratio_hist.png")


# ----------------------------------------------------------------------------
# Figure 6 - Experiment 1 remove-one-category ablation (Random Forest)
# ----------------------------------------------------------------------------
def figure6():
    ab = json.load(open(PIPE / "output/results/ablation_study.json"))
    full = ab["full"]["rf"]["auc_roc_mean"]
    cats = ["structural", "audio", "text", "sentiment"]
    deltas = {c: ab["ablation"]["rf"][c]["auc_roc_mean"] - full for c in cats}
    order = sorted(deltas, key=lambda c: deltas[c])  # most negative first (bottom)
    vals = [deltas[c] for c in order]
    colors = [C_NEG if v < 0 else C_POS for v in vals]
    fig, ax = plt.subplots(figsize=(9, 4.8))
    bars = ax.barh(range(len(order)), vals, color=colors, edgecolor="black", linewidth=0.4)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(order)
    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlabel(f"AUC-ROC change when category is removed (RF baseline {full:.4f})")
    for b, v in zip(bars, vals):
        ax.text(v + (0.0008 if v >= 0 else -0.0008), b.get_y() + b.get_height() / 2,
                f"{v:+.4f}", va="center", ha="left" if v >= 0 else "right", fontsize=9)
    ax.margins(x=0.18)
    ax.grid(axis="y", visible=False)
    save(fig, "fig6_exp1_ablation_remove.png")


# ----------------------------------------------------------------------------
# Figure 7 - Experiment 1 single-category isolation (Random Forest)
# ----------------------------------------------------------------------------
def figure7():
    ab = json.load(open(PIPE / "output/results/ablation_study.json"))
    full = ab["full"]["rf"]["auc_roc_mean"]
    cats = ["structural", "text", "audio", "sentiment"]
    vals = {c: ab["single"]["rf"][c]["auc_roc_mean"] for c in cats}
    order = sorted(cats, key=lambda c: -vals[c])
    v = [vals[c] for c in order]
    colors = [C_RF if vals[c] >= 0.55 else "#9e9e9e" for c in order]
    fig, ax = plt.subplots(figsize=(9, 5.2))
    bars = ax.bar(range(len(order)), v, color=colors, edgecolor="black", linewidth=0.4)
    ax.axhline(0.5, ls=":", color="grey", lw=1.5, label="Random (0.500)")
    ax.axhline(full, ls="--", color=C_NEG, lw=1.5, label=f"RF all-feature baseline ({full:.3f})")
    ax.bar_label(bars, fmt="%.3f", padding=2, fontsize=10)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(order)
    ax.set_ylabel("AUC-ROC (test split, 5 seeds)")
    ax.set_ylim(0.5, 0.76)
    ax.legend(framealpha=0.95)
    save(fig, "fig7_exp1_single_category.png")


# ----------------------------------------------------------------------------
# Figure 8 - Experiment 2 remove-one-category ablation (XGBoost)
# ----------------------------------------------------------------------------
def figure8():
    ab = json.load(open(V2 / "data/post_extraction/results/ablation_xgboost.json"))
    full = ab["baseline_xgboost_full"]["auc_roc"]["mean"]
    roc = ab["remove_one_category"]
    deltas = {c: roc[c]["auc_roc"]["mean"] - full for c in roc}
    order = sorted(deltas, key=lambda c: deltas[c])  # most negative first (bottom)
    vals = [deltas[c] for c in order]
    colors = [C_NEG if v < 0 else C_POS for v in vals]
    fig, ax = plt.subplots(figsize=(9, 5.5))
    bars = ax.barh(range(len(order)), vals, color=colors, edgecolor="black", linewidth=0.4)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(order)
    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlabel(f"AUC-ROC change when category is removed (XGBoost baseline {full:.4f})")
    for b, v in zip(bars, vals):
        ax.text(v + (0.0008 if v >= 0 else -0.0008), b.get_y() + b.get_height() / 2,
                f"{v:+.4f}", va="center", ha="left" if v >= 0 else "right", fontsize=9)
    ax.margins(x=0.18)
    ax.grid(axis="y", visible=False)
    save(fig, "fig8_exp2_ablation_remove.png")


# ----------------------------------------------------------------------------
# Figure 9 - Experiment 2 single-category isolation (XGBoost)
# ----------------------------------------------------------------------------
def figure9():
    ab = json.load(open(V2 / "data/post_extraction/results/ablation_xgboost.json"))
    full = ab["baseline_xgboost_full"]["auc_roc"]["mean"]
    sci = ab["single_category_isolation"]
    vals = {c: sci[c]["auc_roc"]["mean"] for c in sci}
    order = sorted(vals, key=lambda c: -vals[c])
    v = [vals[c] for c in order]
    colors = [C_RF if vals[c] >= 0.55 else "#9e9e9e" for c in order]
    fig, ax = plt.subplots(figsize=(10, 5.5))
    bars = ax.bar(range(len(order)), v, color=colors, edgecolor="black", linewidth=0.4)
    ax.axhline(0.5, ls=":", color="grey", lw=1.5, label="Random (0.500)")
    ax.axhline(full, ls="--", color=C_NEG, lw=1.5, label=f"All-feature baseline ({full:.3f})")
    ax.bar_label(bars, fmt="%.3f", padding=2, fontsize=9)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(order, rotation=25, ha="right")
    ax.set_ylabel("AUC-ROC (test split, 5 seeds)")
    ax.set_ylim(0.5, 0.74)
    ax.legend(framealpha=0.95)
    save(fig, "fig9_exp2_single_category.png")


if __name__ == "__main__":
    print("Regenerating figures at 600 DPI ->", OUT)
    figure1()
    figure2()
    figure3()
    v2 = _load_v2()
    figure4(v2)
    figure5(v2)
    figure6()
    figure7()
    figure8()
    figure9()
    print("Done.")
