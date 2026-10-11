"""Train one model of the population on vendor B.

    python -m ssp.train --model m01 --data-root /path/to/mnms

Uses only the training and validation patients in configs/split.json.
Writes weights to runs/<model>/best.pt and a run record to results/runs/.
"""
from __future__ import annotations

import argparse
import random
import time
from datetime import datetime

import numpy as np

from . import config, data, infer, metrics, paths, preprocess, results


def augmentation(level: int):
    """Training augmentation. Level 0 is none; 1 to 3 scale the intensity changes up."""
    if level == 0:
        return None
    from monai import transforms as T

    k, keys = float(level), ["image", "label"]
    return T.Compose([
        T.RandAffined(keys, prob=0.5, rotate_range=0.26, scale_range=0.1, mode=("bilinear", "nearest"), padding_mode="border"),
        T.RandFlipd(keys, prob=0.5, spatial_axis=0),
        T.RandAdjustContrastd("image", prob=min(0.2 * k, 0.6), gamma=(1 - 0.15 * k, 1 + 0.15 * k)),
        T.RandGaussianSmoothd("image", prob=0.1 * k, sigma_x=(0.25, 0.5 * k), sigma_y=(0.25, 0.5 * k)),
        T.RandGaussianNoised("image", prob=0.1 * k, std=0.05 * k),
        T.RandBiasFieldd("image", prob=0.1 * k, coeff_range=(0.0, 0.1 * k)),
        T.RandScaleIntensityd("image", prob=0.15 * k, factors=0.1 * k),
        T.RandShiftIntensityd("image", prob=0.15 * k, offsets=0.1 * k),
    ])


class SliceSet:
    """All 2D slices of the given patients (both phases), prepared once and kept in memory."""

    def __init__(self, cases, patients, target, patch, aug, rng):
        import torch

        self.torch, self.patch, self.aug, self.rng = torch, patch, aug, rng
        self.items = []
        for _, case in cases[cases["patient"].isin(patients)].iterrows():
            for phase in data.PHASES:
                image, label, spacing = data.load_volume(case, phase)
                img, lab, _ = preprocess.prepare(image, label, spacing, target)
                self.items += [(img[:, :, z], lab[:, :, z]) for z in range(img.shape[2])]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        img, lab = self.items[i]
        if lab.any():                                   # crop around the heart, with jitter
            xs, ys = np.nonzero(lab)
            centre = (int(xs.mean() + self.rng.integers(-24, 25)), int(ys.mean() + self.rng.integers(-24, 25)))
        else:
            centre = (img.shape[0] // 2, img.shape[1] // 2)
        img, lab = preprocess.crop_or_pad_2d(img, lab, self.patch, centre)
        sample = {"image": self.torch.from_numpy(img)[None], "label": self.torch.from_numpy(lab.astype(np.int64))[None]}
        if self.aug is not None:
            sample = self.aug(sample)
        return self.torch.as_tensor(sample["image"]).float(), self.torch.as_tensor(sample["label"]).long()


def validate(net, cases, patients, target, device) -> float:
    """Mean case-level Dice on the validation patients, scored per volume like the final metric."""
    scores = []
    for _, case in cases[cases["patient"].isin(patients)].iterrows():
        per_case = []
        for phase in data.PHASES:
            image, label, spacing = data.load_volume(case, phase)
            img, lab, _ = preprocess.prepare(image, label, spacing, target)
            pred = infer.predict_volume(net, img, device)
            per_case += [metrics.dice(pred == v, lab == v) for v in data.STRUCTURES.values()]
        scores.append(np.nanmean(per_case))
    return float(np.mean(scores))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="model id from configs/models.yaml, e.g. m01")
    ap.add_argument("--data-root")
    ap.add_argument("--epochs", type=int, help="override the configured number of epochs (for quick tests)")
    ap.add_argument("--device")
    args = ap.parse_args()

    import torch
    from monai.losses import DiceCELoss

    cfg = config.model(args.model)
    if args.epochs:
        cfg["epochs"] = args.epochs
    root = paths.data_root(args.data_root)
    split, dcfg = config.split(), config.data()
    if bool(split.get("mock")) != paths.is_mock(root):
        raise SystemExit("configs/split.json was made for a different dataset (mock vs real). Re-make the split.")
    device = infer.pick_device(args.device)

    random.seed(cfg["seed"]); np.random.seed(cfg["seed"]); torch.manual_seed(cfg["seed"])
    rng = np.random.default_rng(cfg["seed"])
    cases = data.discover_cases(root)
    target = dcfg["target_spacing_mm"]

    print(f"[{args.model}] seed {cfg['seed']}, augmentation level {cfg['aug']}, {cfg['epochs']} epochs, device {device}")
    train_set = SliceSet(cases, split["train"], target, cfg["patch_size"], augmentation(cfg["aug"]), rng)
    loader = torch.utils.data.DataLoader(train_set, batch_size=cfg["batch_size"], shuffle=True, drop_last=True, num_workers=0)
    print(f"[{args.model}] {len(split['train'])} training patients, {len(train_set)} slices; {len(split['val'])} validation patients")

    net = infer.build_net(cfg).to(device)
    loss_fn = DiceCELoss(to_onehot_y=True, softmax=True)
    opt = torch.optim.Adam(net.parameters(), lr=cfg["lr"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg["epochs"])
    scaler = torch.amp.GradScaler(enabled=(device == "cuda"))

    out_dir = paths.runs() / args.model
    out_dir.mkdir(parents=True, exist_ok=True)
    log, best, best_epoch, start = [], -1.0, 0, time.time()
    for epoch in range(1, cfg["epochs"] + 1):
        net.train()
        losses = []
        for image, label in loader:
            image, label = image.to(device), label.to(device)
            opt.zero_grad()
            with torch.autocast(device_type=device, enabled=(device == "cuda")):
                loss = loss_fn(net(image), label)
            scaler.scale(loss).backward()
            scaler.step(opt); scaler.update()
            losses.append(loss.item())
        sched.step()
        net.eval()
        val = validate(net, cases, split["val"], target, device)
        log.append({"epoch": epoch, "loss": float(np.mean(losses)), "val_dice": val})
        if val > best:
            best, best_epoch = val, epoch
            torch.save({"state_dict": net.state_dict(), "config": cfg, "epoch": epoch, "val_dice": val}, out_dir / "best.pt")
        print(f"[{args.model}] epoch {epoch:3d}  loss {np.mean(losses):.4f}  val Dice {val:.4f}{'  *' if epoch == best_epoch else ''}")

    import pandas as pd
    pd.DataFrame(log).to_csv(out_dir / "train_log.csv", index=False)
    record = results.update_run(
        paths.results(root), args.model,
        seed=cfg["seed"], aug=cfg["aug"], epochs=cfg["epochs"], best_epoch=best_epoch, best_val_dice=round(best, 4),
        trained_at=datetime.now().isoformat(timespec="seconds"), train_commit=config.git_commit(),
        train_minutes=round((time.time() - start) / 60, 1), device=device, n_train_patients=len(split["train"]),
    )
    print(f"[{args.model}] done. Best validation Dice {record['best_val_dice']} at epoch {best_epoch}. Weights: {out_dir / 'best.pt'}")


if __name__ == "__main__":
    main()
