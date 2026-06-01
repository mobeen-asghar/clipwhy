"""Entry point for BERT fine-tuning (GPU required).

1. python -m src.models.model_c_bert.prepare_data   # once, ~5 min, pulls labeled/ and joins transcripts
2. python -m src.models.model_c_bert.bert_model     # 5 seeds, GPU, ~2-4 h per seed on RTX 4090
"""
from __future__ import annotations

import sys


def main():
    print("Run in order on a GPU pod:")
    print("  python -m src.models.model_c_bert.prepare_data")
    print("  python -m src.models.model_c_bert.bert_model")
    print("Intended environment: RunPod RTX 4090 or Google Colab T4.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
