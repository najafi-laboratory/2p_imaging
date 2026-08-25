"""SRDTrans network construction and CUDA runtime setup shared by both workflows."""

from __future__ import annotations

import os
import warnings


def runtime(gpu: str):
    """Load PyTorch and expose the requested CUDA devices."""
    if gpu.lower() != "all":
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    try:
        import torch
    except ImportError as error:
        raise RuntimeError("Install the SRDTrans dependencies from requirements.txt") from error
    if not torch.cuda.is_available():
        raise RuntimeError("SRDTrans requires a CUDA-capable PyTorch installation")
    return torch, torch.cuda.device_count()


def network(patch: int):
    """Create the local SRDTrans network architecture."""
    warnings.filterwarnings(
        "ignore", message="Importing from timm.models.layers", category=FutureWarning)
    from srdtrans_model import SRDTrans

    return SRDTrans(
        img_dim=patch, img_time=patch, in_channel=1, embedding_dim=128,
        num_heads=8, hidden_dim=512, window_size=7, num_transBlock=1,
        attn_dropout_rate=.1, f_maps=[8, 16, 32, 64], input_dropout_rate=0,
    )
