# src/npi_mt/npi/normalization.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import torch


@dataclass(frozen=True)
class NormalizationStats:
    """
    Stores normalization for:
      - inputs: (2, F) or (2, 1) broadcastable
      - labels: (Z,)
    """
    input_mean: torch.Tensor  # (2,F) or (2,1)
    input_std: torch.Tensor   # (2,F) or (2,1)
    label_mean: torch.Tensor  # (Z,)
    label_std: torch.Tensor   # (Z,)

    @property
    def device(self):
        return self.input_mean.device

    def to(self, device: torch.device) -> "NormalizationStats":
        return NormalizationStats(
            input_mean=self.input_mean.to(device),
            input_std=self.input_std.to(device),
            label_mean=self.label_mean.to(device),
            label_std=self.label_std.to(device),
        )


def _canon_input_stat(x: torch.Tensor, F: int) -> torch.Tensor:
    """
    Accepts (2,), (1,2), (2,1), (1,2,F), (2,F) -> returns (2,F) or (2,1).
    If channel-only stats (2,), returns (2,1) for broadcast.
    """
    x = x.detach()
    while x.ndim > 2:
        # e.g. (1,2,F) -> (2,F)
        x = x.squeeze(0)

    if x.ndim == 1:
        if x.shape[0] != 2:
            raise ValueError(f"Expected input stat shape (2,), got {tuple(x.shape)}")
        return x.view(2, 1)  # (2,1), broadcast over F

    if x.ndim == 2:
        if x.shape[0] == 2 and x.shape[1] == F:
            return x
        if x.shape == (2, 1):
            return x
        if x.shape == (1, 2):
            return x.view(2, 1)
        raise ValueError(f"Unexpected input stat shape {tuple(x.shape)} for F={F}")

    raise ValueError(f"Unexpected input stat ndim={x.ndim}")


def load_normalization_stats(stats_dir: str | Path, device: torch.device, F: int) -> NormalizationStats:
    """
    Loads:
      label_mean.pth: {'label_mean': ...}
      label_std.pth:  {'label_std': ...}
      input_mean.pth: {'input_mean': ...}
      input_std.pth:  {'input_std': ...}
    """
    stats_dir = Path(stats_dir)

    lm = torch.load(stats_dir / "label_mean.pth", map_location="cpu", weights_only=False)["label_mean"]
    ls = torch.load(stats_dir / "label_std.pth",  map_location="cpu", weights_only=False)["label_std"]
    im = torch.load(stats_dir / "input_mean.pth", map_location="cpu", weights_only=False)["input_mean"]
    is_ = torch.load(stats_dir / "input_std.pth", map_location="cpu", weights_only=False)["input_std"]

    lm = torch.as_tensor(lm, dtype=torch.float32)
    ls = torch.as_tensor(ls, dtype=torch.float32)
    im = torch.as_tensor(im, dtype=torch.float32)
    is_ = torch.as_tensor(is_, dtype=torch.float32)

    im = _canon_input_stat(im, F)
    is_ = _canon_input_stat(is_, F)

    # protect against divide-by-zero
    is_ = torch.clamp(is_, min=1e-8)
    ls = torch.clamp(ls, min=1e-8)

    stats = NormalizationStats(
        input_mean=im,
        input_std=is_,
        label_mean=lm,
        label_std=ls,
    )
    return stats.to(device)
