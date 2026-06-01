"""Visual features (10 logical, 39 physical columns).

Components (5 frames per segment, sampled at t=0,6,12,18,24s):
  - dover_aesthetic_score, dover_technical_score         (DOVER, ICCV 2023)
  - clip_embedding  -> PCA to 32 dims                    (open_clip ViT-L/14)
  - colorfulness, brightness_mean                        (numpy + OpenCV)
  - cut_count, cuts_per_second                           (TransNetV2 on all frames)
  - face_present_ratio, largest_face_area_ratio_max,
    face_count_median                                    (InsightFace SCRFD)

Batched extractor:
  extract_batch(jobs, models) -> list of (dict, clip_raw_embedding or None)

Batching is primarily useful for CLIP and DOVER (ViT-based) where moving
K segments' worth of frames through in one forward is substantially faster
than K separate forwards. TransNetV2 and SCRFD are per-segment.
"""
import logging
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np

try:
    import cv2  # noqa: F401
    from PIL import Image  # noqa: F401
    HAS_CV = True
except ImportError:
    HAS_CV = False

from .. import config
from ..pipeline import SegmentJob

log = logging.getLogger("clipwhy.features.visual")

# Per-call timeouts for ffmpeg subprocesses. The 30s MP4s we process are small,
# so these are generous ceilings; anything above them means ffmpeg is stuck.
_FFMPEG_FRAME_TIMEOUT = 30
_FFMPEG_FULLSCAN_TIMEOUT = 120
_FFMPEG_TRANSCODE_TIMEOUT = 120

_ZERO_VISUAL = {
    "dover_aesthetic_score": 0.0, "dover_technical_score": 0.0,
    **{f"clip_pca_{i:02d}": 0.0 for i in range(config.CLIP_PCA_DIMS)},
    "colorfulness": 0.0, "brightness_mean": 0.0,
    "cut_count": 0.0, "cuts_per_second": 0.0,
    "face_present_ratio": 0.0, "largest_face_area_ratio_max": 0.0, "face_count_median": 0.0,
}


# ──────────────────────────────────────────────────────────────────────────
# Model loaders
# ──────────────────────────────────────────────────────────────────────────
def load_clip(device: str = "cuda"):
    try:
        import open_clip
        import torch
    except ImportError as e:
        log.error("open_clip / torch not installed: %s", e)
        return None
    try:
        model, _, preprocess = open_clip.create_model_and_transforms(
            "ViT-L-14", pretrained="laion2b_s32b_b82k",
            device=device, precision="fp16" if device == "cuda" else "fp32",
        )
        model.eval()
        # Also load a PCA model if present on disk (fit offline from train split).
        pca_path = config.LOCAL_MODELS / "clip_pca.pkl"
        pca = None
        if pca_path.exists():
            import joblib
            pca = joblib.load(pca_path)
        return {"model": model, "preprocess": preprocess, "device": device, "pca": pca}
    except Exception as e:
        log.error("CLIP load failed: %s", e)
        return None


def load_dover(device: str = "cuda"):
    """Load DOVER (ICCV 2023) and its frame samplers for one-video inference.

    Requires:
      - DOVER repo cloned at config.LOCAL_MODELS / "DOVER"
      - DOVER installed via `pip install -e . --no-deps`
      - decord, scikit-video, thop, timm installed
      - Weights at DOVER/pretrained_weights/DOVER.pth (~240 MB)
    """
    try:
        import sys as _sys
        dover_root = config.LOCAL_MODELS / "DOVER"
        if dover_root.exists() and str(dover_root) not in _sys.path:
            _sys.path.insert(0, str(dover_root))
        import torch
        import yaml
        from dover.datasets import UnifiedFrameSampler
        from dover.models import DOVER
    except Exception as e:
        log.warning("DOVER not available (%s); aesthetic/technical scores will be zero", e)
        return None
    try:
        cfg_path = config.LOCAL_MODELS / "DOVER" / "dover.yml"
        with open(cfg_path) as f:
            opt = yaml.safe_load(f)
        model = DOVER(**opt["model"]["args"])
        weights_path = config.LOCAL_MODELS / "DOVER" / "pretrained_weights" / "DOVER.pth"
        state = torch.load(str(weights_path), map_location="cpu", weights_only=False)
        model.load_state_dict(state, strict=False)
        model.eval()
        if device == "cuda":
            model = model.cuda()

        # Build samplers per the dover.yml val-l1080p args.
        dopt = opt["data"]["val-l1080p"]["args"]
        temporal_samplers = {}
        for stype, sopt in dopt["sample_types"].items():
            if "t_frag" not in sopt:
                temporal_samplers[stype] = UnifiedFrameSampler(
                    sopt["clip_len"], sopt["num_clips"], sopt["frame_interval"]
                )
            else:
                temporal_samplers[stype] = UnifiedFrameSampler(
                    sopt["clip_len"] // sopt["t_frag"],
                    sopt["t_frag"],
                    sopt["frame_interval"],
                    sopt["num_clips"],
                )

        mean = torch.FloatTensor([123.675, 116.28, 103.53])
        std = torch.FloatTensor([58.395, 57.12, 57.375])

        return {
            "model": model,
            "device": device,
            "opt": opt,
            "dopt": dopt,
            "samplers": temporal_samplers,
            "mean": mean,
            "std": std,
        }
    except Exception as e:
        log.warning("DOVER weights load failed (%s); aesthetic/technical will be zero", e)
        return None


def load_transnet(device: str = "cuda"):
    try:
        from transnetv2_pytorch import TransNetV2
        import torch
    except ImportError as e:
        log.warning("transnetv2_pytorch not installed (%s); cut_count will be zero", e)
        return None
    try:
        model = TransNetV2()
        ckpt = config.LOCAL_MODELS / "transnetv2-pytorch-weights.pth"
        if ckpt.exists():
            state = torch.load(str(ckpt), map_location="cpu")
            model.load_state_dict(state)
        model.eval()
        if device == "cuda":
            model = model.cuda()
        return {"model": model, "device": device}
    except Exception as e:
        log.warning("TransNetV2 load failed (%s); cut_count will be zero", e)
        return None


def load_scrfd(device: str = "cuda"):
    try:
        import insightface
        from insightface.app import FaceAnalysis
    except ImportError as e:
        log.warning("insightface not installed (%s); face features will be zero", e)
        return None
    try:
        app = FaceAnalysis(
            name="buffalo_sc",   # includes SCRFD 2.5g detector
            allowed_modules=["detection"],
            providers=["CUDAExecutionProvider" if device == "cuda" else "CPUExecutionProvider"],
        )
        app.prepare(ctx_id=0 if device == "cuda" else -1, det_size=(640, 640))
        return {"app": app}
    except Exception as e:
        log.warning("SCRFD init failed (%s); face features will be zero", e)
        return None


# ──────────────────────────────────────────────────────────────────────────
# Frame extraction
# ──────────────────────────────────────────────────────────────────────────
def _sample_frames(video_path: Path) -> list[np.ndarray]:
    """Extract 5 frames from the MP4 at the configured sample times in ONE
    ffmpeg invocation.

    Prefers a pre-transcoded H.264 sibling (created during pull for AV1
    sources) so ffmpeg doesn't re-decode AV1 multiple times per segment.

    Single-call extraction uses the `select` filter with `lt(abs(t-X),0.05)`
    to grab the frame within 50ms of each target timestamp. ~3x faster than
    5 separate -ss calls because we only pay process startup once and
    decode the video sequentially in one pass.

    Returns list of BGR numpy arrays in timestamp order. Empty list on failure.
    """
    if video_path is None:
        return []
    effective = _preferred_video_path(video_path)
    if effective is None or not effective.exists():
        return []
    if not HAS_CV:
        return []

    times = config.FRAME_SAMPLE_TIMES_SEC  # [0, 6, 12, 18, 24]
    select_expr = "+".join(f"lt(abs(t-{t}),0.05)" for t in times)
    n_target = len(times)

    with tempfile.TemporaryDirectory(prefix="clipwhy_frames_") as td:
        tdp = Path(td)
        try:
            # -frames:v <n_target> caps output to exactly the number of
            # target frames. Without it, the select filter matches multiple
            # frames near each timestamp (e.g., 14 outputs for 5 targets at
            # 30 fps), which wastes downstream model compute by 2-3x.
            subprocess.run(
                [
                    "ffmpeg", "-v", "error", "-y",
                    "-i", str(effective),
                    "-vf", f"select='{select_expr}',setpts=N/FRAME_RATE/TB",
                    "-vsync", "vfr",
                    "-frames:v", str(n_target),
                    "-q:v", "2",
                    f"{tdp}/f_%03d.jpg",
                ],
                check=False,
                timeout=_FFMPEG_FRAME_TIMEOUT * 2,  # one call decodes whole clip
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except subprocess.TimeoutExpired:
            log.warning("ffmpeg multi-frame extract timed out for %s", effective.name)
            return []
        frames: list[np.ndarray] = []
        for p in sorted(tdp.glob("f_*.jpg")):
            img = cv2.imread(str(p))
            if img is not None:
                frames.append(img)
        return frames


def _read_all_frames_downscaled(video_path: Path) -> Optional[np.ndarray]:
    """Read all frames from video, downscaled to (48, 27) for TransNetV2.

    Prefers the H.264 pre-transcoded sibling when available so libavcodec
    doesn't waste time re-decoding AV1 on every pipeline pass.
    """
    if video_path is None:
        return None
    effective = _preferred_video_path(video_path)
    if effective is None or not effective.exists():
        return None
    video_path = effective
    W = config.TRANSNET_INPUT_WIDTH
    H = config.TRANSNET_INPUT_HEIGHT
    cmd = [
        "ffmpeg", "-v", "error",
        "-i", str(video_path),
        "-vf", f"scale={W}:{H}",
        "-f", "rawvideo",
        "-pix_fmt", "rgb24",
        "-",
    ]
    try:
        proc = subprocess.run(
            cmd,
            check=False,
            timeout=_FFMPEG_FULLSCAN_TIMEOUT,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
    except subprocess.TimeoutExpired:
        log.warning("ffmpeg full-scan timed out for %s", video_path.name)
        return None
    raw = proc.stdout
    frame_bytes = W * H * 3
    n_frames = len(raw) // frame_bytes
    if n_frames < 2:
        return None
    buf = np.frombuffer(raw[: n_frames * frame_bytes], dtype=np.uint8)
    return buf.reshape(n_frames, H, W, 3).copy()  # (T, 27, 48, 3)


# ──────────────────────────────────────────────────────────────────────────
# Low-level per-frame features
# ──────────────────────────────────────────────────────────────────────────
def _colorfulness(frame_bgr: np.ndarray) -> float:
    """Hasler-Suesstrunk 2003 colorfulness metric."""
    b, g, r = [c.astype(np.float32) for c in np.dsplit(frame_bgr, 3)]
    rg = np.abs(r - g)
    yb = np.abs(0.5 * (r + g) - b)
    std_rg, std_yb = np.std(rg), np.std(yb)
    mean_rg, mean_yb = np.mean(rg), np.mean(yb)
    return float(np.sqrt(std_rg ** 2 + std_yb ** 2) + 0.3 * np.sqrt(mean_rg ** 2 + mean_yb ** 2))


def _brightness_hsv_v(frame_bgr: np.ndarray) -> float:
    try:
        import cv2
    except ImportError:
        return 0.0
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    return float(hsv[:, :, 2].mean() / 255.0)


# ──────────────────────────────────────────────────────────────────────────
# Main batched extractor
# ──────────────────────────────────────────────────────────────────────────
def extract_batch(jobs: list[SegmentJob], models: dict):
    """Extract all visual features for a batch of segments.

    Returns list of (feature_dict, clip_embedding_raw_or_None), same order as jobs.
    """
    clip_bundle = models.get("clip")
    dover_bundle = models.get("dover")
    transnet_bundle = models.get("transnet")
    scrfd_bundle = models.get("scrfd")

    # 1) Sample 5 frames per job + accumulate.
    sampled: list[list[np.ndarray]] = [_sample_frames(j.video_path) for j in jobs]

    # 2) CLIP on the sampled frames, batched across all frames in this batch.
    # NOTE: build each zero vector separately to avoid aliased references.
    clip_embeds_per_job: list[Optional[np.ndarray]] = [None] * len(jobs)
    clip_pca_per_job: list[np.ndarray] = [
        np.zeros(config.CLIP_PCA_DIMS) for _ in range(len(jobs))
    ]

    if clip_bundle is not None and HAS_CV:
        try:
            import torch
            model = clip_bundle["model"]
            preprocess = clip_bundle["preprocess"]
            device = clip_bundle["device"]
            pca = clip_bundle["pca"]
            # Flatten frames from all jobs into one big batch.
            flat_imgs = []
            owners: list[int] = []   # which job each frame belongs to
            for i, frames in enumerate(sampled):
                for f in frames:
                    pil = Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
                    flat_imgs.append(preprocess(pil))
                    owners.append(i)
            if flat_imgs:
                x = torch.stack(flat_imgs)
                if device == "cuda":
                    x = x.cuda().half()
                with torch.no_grad():
                    emb = model.encode_image(x).float().cpu().numpy()
                # Mean-pool per job
                for i in range(len(jobs)):
                    rows = emb[np.array(owners) == i]
                    if len(rows) > 0:
                        clip_embeds_per_job[i] = rows.mean(axis=0)
                # PCA if available (fitted offline on train split)
                if pca is not None:
                    for i, emb_vec in enumerate(clip_embeds_per_job):
                        if emb_vec is not None:
                            pca_vec = pca.transform(emb_vec.reshape(1, -1))[0]
                            pad = config.CLIP_PCA_DIMS - len(pca_vec)
                            if pad > 0:
                                pca_vec = np.concatenate([pca_vec, np.zeros(pad)])
                            clip_pca_per_job[i] = pca_vec[: config.CLIP_PCA_DIMS]
        except Exception as e:
            log.warning("CLIP batch failed: %s", e)

    # 3) DOVER (per-job; currently not as easily batchable across segments).
    dover_out: list[tuple[float, float]] = [(0.0, 0.0)] * len(jobs)
    if dover_bundle is not None:
        for i, j in enumerate(jobs):
            try:
                aes, tech = _dover_one(j.video_path, dover_bundle)
                dover_out[i] = (aes, tech)
            except Exception as e:
                log.debug("DOVER failed for %s: %s", j.segment_id, e)

    # 4) TransNetV2 (per-job).
    cut_counts: list[int] = [0] * len(jobs)
    if transnet_bundle is not None:
        for i, j in enumerate(jobs):
            try:
                cut_counts[i] = _transnet_cuts(j.video_path, transnet_bundle)
            except Exception as e:
                log.debug("TransNetV2 failed for %s: %s", j.segment_id, e)

    # 5) SCRFD faces (per-frame, per-job).
    face_stats: list[tuple[float, float, float]] = [(0.0, 0.0, 0.0)] * len(jobs)
    if scrfd_bundle is not None:
        for i, frames in enumerate(sampled):
            face_stats[i] = _face_stats(frames, scrfd_bundle)

    # 6) Low-level (colorfulness, brightness).
    low_level: list[tuple[float, float]] = []
    for frames in sampled:
        if frames:
            colorfulness_vals = [_colorfulness(f) for f in frames]
            brightness_vals = [_brightness_hsv_v(f) for f in frames]
            low_level.append((float(np.mean(colorfulness_vals)), float(np.mean(brightness_vals))))
        else:
            low_level.append((0.0, 0.0))

    # 7) Assemble per-job output dicts.
    results: list[tuple[dict, Optional[np.ndarray]]] = []
    for i, j in enumerate(jobs):
        aes, tech = dover_out[i]
        c, b = low_level[i]
        fpr, lfar, fcm = face_stats[i]
        cc = cut_counts[i]
        cps = cc / max(j.duration, 1e-6)

        row = dict(_ZERO_VISUAL)
        row["dover_aesthetic_score"] = round(aes, 6)
        row["dover_technical_score"] = round(tech, 6)
        for k in range(config.CLIP_PCA_DIMS):
            row[f"clip_pca_{k:02d}"] = float(round(clip_pca_per_job[i][k], 6))
        row["colorfulness"] = round(c, 6)
        row["brightness_mean"] = round(b, 6)
        row["cut_count"] = float(cc)
        row["cuts_per_second"] = round(cps, 6)
        row["face_present_ratio"] = round(fpr, 6)
        row["largest_face_area_ratio_max"] = round(lfar, 6)
        row["face_count_median"] = round(fcm, 6)

        results.append((row, clip_embeds_per_job[i]))
    return results


# ──────────────────────────────────────────────────────────────────────────
# Per-segment sub-extractors
# ──────────────────────────────────────────────────────────────────────────
def _preferred_video_path(video_path: Optional[Path]) -> Optional[Path]:
    """If a pre-transcoded H.264 sibling exists (from worker._pretranscode),
    prefer that for any downstream video decode. This gets us AV1 coverage
    without paying the transcode cost per-segment during extraction."""
    if video_path is None:
        return None
    h264_sibling = video_path.with_suffix(".h264.mp4")
    if h264_sibling.exists():
        return h264_sibling
    return video_path


def _probe_codec(video_path: Path) -> str:
    """Return the video codec name (e.g., 'h264', 'av1'). Empty string on failure."""
    try:
        result = subprocess.run(
            [
                "ffprobe", "-v", "error",
                "-select_streams", "v:0",
                "-show_entries", "stream=codec_name",
                "-of", "csv=p=0",
                str(video_path),
            ],
            check=False, timeout=10,
            capture_output=True, text=True,
        )
        return result.stdout.strip()
    except Exception:
        return ""


def _dover_one(video_path: Optional[Path], bundle: dict) -> tuple[float, float]:
    """Run DOVER on one 30-s MP4. Returns (aesthetic, technical) scalars.

    Mirrors DOVER's evaluate_one_video.py reference inference. DOVER uses
    decord which doesn't decode AV1, so the pull phase (worker._pretranscode
    _av1_segments) prepares an H.264 sibling for any non-H.264 source.
    _preferred_video_path picks the H.264 version when it exists.

    The model outputs two tensors; we return them as (aesthetic, technical).
    """
    if video_path is None:
        return 0.0, 0.0
    effective_path = _preferred_video_path(video_path)
    if effective_path is None or not effective_path.exists():
        return 0.0, 0.0

    try:
        import torch
        from dover.datasets import spatial_temporal_view_decomposition
        model = bundle["model"]
        device = bundle["device"]
        dopt = bundle["dopt"]
        samplers = bundle["samplers"]
        mean = bundle["mean"]
        std = bundle["std"]

        views, _ = spatial_temporal_view_decomposition(
            str(effective_path), dopt["sample_types"], samplers
        )
        for k, v in views.items():
            num_clips = dopt["sample_types"][k].get("num_clips", 1)
            views[k] = (
                ((v.permute(1, 2, 3, 0) - mean) / std)
                .permute(3, 0, 1, 2)
                .reshape(v.shape[0], num_clips, -1, *v.shape[2:])
                .transpose(0, 1)
                .to(device)
            )

        with torch.no_grad():
            results = [r.mean().item() for r in model(views)]
        # DOVER.forward returns [technical, aesthetic] per the reference script.
        technical, aesthetic = float(results[0]), float(results[1])
        return aesthetic, technical
    except Exception as e:
        log.debug("DOVER inference failed for %s: %s", video_path.name, e)
        return 0.0, 0.0


def _transnet_cuts(video_path: Optional[Path], bundle: dict) -> int:
    """TransNetV2 hard-cut count for one segment."""
    if video_path is None or not video_path.exists():
        return 0
    frames = _read_all_frames_downscaled(video_path)
    if frames is None or frames.shape[0] < 2:
        return 0
    try:
        import torch
        model = bundle["model"]
        device = bundle["device"]
        x = torch.from_numpy(frames).unsqueeze(0)  # (1, T, H, W, 3) uint8
        if device == "cuda":
            x = x.cuda()
        with torch.no_grad():
            single_frame_pred, _ = model(x)
            probs = torch.sigmoid(single_frame_pred).squeeze().cpu().numpy()
        # Count hard cuts using the standard threshold 0.5.
        cuts = _count_transitions(probs, threshold=0.5)
        return int(cuts)
    except Exception as e:
        log.debug("TransNetV2 inference issue: %s", e)
        return 0


def _count_transitions(probs: np.ndarray, threshold: float) -> int:
    """Classic TransNetV2 threshold-based transition counter (non-max suppression)."""
    if probs.ndim == 0:
        return 0
    bin_ = probs > threshold
    # Count rising edges.
    edges = np.diff(bin_.astype(int)) == 1
    return int(edges.sum())


def _face_stats(frames: list[np.ndarray], bundle: dict) -> tuple[float, float, float]:
    """face_present_ratio, largest_face_area_ratio_max, face_count_median."""
    if not frames:
        return 0.0, 0.0, 0.0
    app = bundle["app"]
    presences, max_areas, counts = [], [], []
    for frame in frames:
        try:
            faces = app.get(frame) or []
        except Exception:
            faces = []
        # Filter by confidence
        faces = [f for f in faces if float(getattr(f, "det_score", 1.0)) >= config.FACE_CONF_THRESHOLD]
        present = 1.0 if faces else 0.0
        presences.append(present)
        counts.append(len(faces))
        if faces:
            h, w = frame.shape[:2]
            frame_area = h * w
            areas = [
                (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1])
                for f in faces
            ]
            max_areas.append(float(max(areas)) / max(frame_area, 1))
        else:
            max_areas.append(0.0)

    face_present_ratio = float(np.mean(presences))
    largest_face_area_ratio_max = float(max(max_areas)) if max_areas else 0.0
    face_count_median = float(np.median(counts)) if counts else 0.0
    return face_present_ratio, largest_face_area_ratio_max, face_count_median
