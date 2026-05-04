"""VideoGPT package exports.

Heavy training dependencies such as pytorch_lightning are imported lazily so
lightweight utility modules can run in CPU-only smoke tests without the full
legacy VideoGPT environment.
"""

__all__ = [
    "VQVAE",
    "VideoGPT",
    "VideoData",
    "load_vqvae",
    "load_videogpt",
    "load_i3d_pretrained",
    "download",
]


def __getattr__(name):
    if name == "VQVAE":
        from .vqvae import VQVAE

        return VQVAE
    if name == "VideoGPT":
        from .gpt import VideoGPT

        return VideoGPT
    if name == "VideoData":
        from .data import VideoData

        return VideoData
    if name in {"load_vqvae", "load_videogpt", "load_i3d_pretrained", "download"}:
        from . import download as download_module

        return getattr(download_module, name)
    raise AttributeError(f"module 'videogpt' has no attribute {name!r}")
