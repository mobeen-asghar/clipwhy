"""Multimodal fusion entry point. Requires Model C (BERT) to be trained first."""
from __future__ import annotations

import sys


def main():
    print("Run in order on a GPU pod:")
    print("  python -m src.models.model_c_bert.prepare_data")
    print("  python -m src.models.model_c_bert.bert_model")
    print("  python -m src.models.model_d_multimodal.multimodal --bert-seed 42")
    return 0


if __name__ == "__main__":
    sys.exit(main())
