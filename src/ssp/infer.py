"""The network, and how a trained model segments a whole volume."""
from __future__ import annotations

import numpy as np

from . import paths, preprocess

N_CLASSES = 4   # background, LV, MYO, RV


def build_net(cfg: dict):
    from monai.networks.nets import UNet

    channels = tuple(cfg["channels"])
    return UNet(spatial_dims=2, in_channels=1, out_channels=N_CLASSES, channels=channels,
                strides=(2,) * (len(channels) - 1), num_res_units=cfg["num_res_units"],
                norm=cfg.get("norm", "instance"))


def pick_device(name: str | None = None) -> str:
    import torch

    return name or ("cuda" if torch.cuda.is_available() else "cpu")


def checkpoint_path(model_id: str):
    return paths.runs() / model_id / "best.pt"


def load_model(model_id: str, device: str):
    import torch

    path = checkpoint_path(model_id)
    if not path.exists():
        raise SystemExit(f"No trained weights at {path}. Train first:  python -m ssp.train --model {model_id}")
    ckpt = torch.load(path, map_location=device, weights_only=False)
    net = build_net(ckpt["config"]).to(device)
    net.load_state_dict(ckpt["state_dict"])
    net.eval()
    return net, ckpt


def predict_volume(net, image: np.ndarray, device: str, batch: int = 16) -> np.ndarray:
    """Segment a prepared volume [X,Y,Z] slice by slice. Returns labels [X,Y,Z]."""
    import torch

    padded, (x, y) = preprocess.pad_to_multiple(image, 32)             # safe for up to 5 downsamplings
    slices = torch.from_numpy(np.moveaxis(padded, 2, 0)[:, None])       # [Z,1,X,Y]
    out = []
    with torch.no_grad():
        for i in range(0, len(slices), batch):
            out.append(net(slices[i:i + batch].to(device)).argmax(1).cpu())
    pred = torch.cat(out).numpy().astype(np.uint8)                      # [Z,X,Y]
    return np.moveaxis(pred, 0, 2)[:x, :y]
