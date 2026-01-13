# DiffusionFreeGuidence/TrainCondition.py
from typing import Dict, List
import os
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import save_image
from tqdm import tqdm

from VeinDataset import PalmVeinDataset
from DiffusionFreeGuidence.ModelCondition import UNet
from DiffusionFreeGuidence.DiffusionCondition import (
    GaussianDiffusionTrainer,
    GaussianDiffusionSampler,
    DynamicGuidanceDiffusionSampler,
    extract,
)

from Scheduler import GradualWarmupScheduler
import copy
import glob


def train(model_cfg: Dict):
    device = torch.device(model_cfg["device"])
    dataset = PalmVeinDataset(
        root=model_cfg["data_root"],
        transform=transforms.Compose(
            [
                transforms.Resize((model_cfg["img_size"], model_cfg["img_size"])),
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,)),
            ]
        ),
    )
    dataloader = DataLoader(
        dataset,
        batch_size=model_cfg["batch_size"],
        shuffle=True,
        num_workers=4,
        drop_last=True,
        pin_memory=True,
    )

    num_labels = len(dataset.class_to_idx)
    print(f"Detected {num_labels} user labels.")

    net = UNet(
        T=model_cfg["T"],
        num_labels=num_labels,
        ch=model_cfg["channel"],
        ch_mult=model_cfg["channel_mult"],
        num_res_blocks=model_cfg["num_res_blocks"],
        dropout=model_cfg["dropout"],
    ).to(device)

    if model_cfg["training_load_weight"]:
        net.load_state_dict(
            torch.load(
                os.path.join(
                    model_cfg["save_dir"], model_cfg["training_load_weight"]
                ),
                map_location=device,
            ),
            strict=False,
        )
        print("Pretrained weight loaded.")

    optimizer = torch.optim.AdamW(net.parameters(), lr=model_cfg["lr"], weight_decay=1e-4)
    cosine = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=model_cfg["epoch"], eta_min=0
    )
    warmup = GradualWarmupScheduler(
        optimizer,
        multiplier=model_cfg["multiplier"],
        warm_epoch=model_cfg["epoch"] // 10,
        after_scheduler=cosine,
    )

    trainer = GaussianDiffusionTrainer(
        net, model_cfg["beta_1"], model_cfg["beta_T"], model_cfg["T"]
    ).to(device)

    os.makedirs(model_cfg["save_dir"], exist_ok=True)
    all_losses: List[float] = []

    for e in range(model_cfg["epoch"]):
        epoch_losses = []
        with tqdm(dataloader, dynamic_ncols=True) as tq:
            for x_0, labels in tq:
                x_0 = x_0.to(device)
                labels = labels.to(device) + 1
                if np.random.rand() < 0.1:
                    labels = torch.zeros_like(labels)

                optimizer.zero_grad()
                loss = trainer(x_0, labels).mean()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    net.parameters(), model_cfg["grad_clip"]
                )
                optimizer.step()

                tq.set_postfix(
                    epoch=e,
                    loss=f"{loss.item():.4f}",
                    LR=optimizer.state_dict()["param_groups"][0]["lr"],
                )
                epoch_losses.append(loss.item())

        mean_loss = float(np.mean(epoch_losses))
        all_losses.append(mean_loss)
        warmup.step()

        save_path = os.path.join(model_cfg["save_dir"], f"ckpt_{e:02d}.pt")
        torch.save(net.state_dict(), save_path)

    plt.figure()
    plt.plot(all_losses, label="train loss")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title("DDPM training curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(model_cfg["save_dir"], "loss.png"))
    plt.close()
    print("Training finished.")


@torch.no_grad()
def infer_single(cfg: Dict):
    from PIL import Image

    device = torch.device(cfg["device"])

    tfm = transforms.Compose(
        [
            transforms.Resize((cfg["img_size"], cfg["img_size"])),
            transforms.ToTensor(),
        ]
    )
    raw = tfm(Image.open(cfg["noisy_img_path"]).convert("L"))
    raw = 1.0 - raw
    raw = transforms.Normalize((0.5,), (0.5,))(raw).repeat(3, 1, 1).to(device)
    y = raw.unsqueeze(0)

    net = UNet(
        T=cfg["T"],
        num_labels=cfg["num_labels"],
        ch=cfg["channel"],
        ch_mult=cfg["channel_mult"],
        num_res_blocks=cfg["num_res_blocks"],
        dropout=cfg["dropout"],
    ).to(device)
    ckpt = torch.load(
        os.path.join(cfg["save_dir"], cfg["test_load_weight"]), map_location=device
    )
    net.load_state_dict(ckpt)
    net.eval()

    os.makedirs(cfg["sampled_dir"], exist_ok=True)

    sampler_type = cfg.get("sampler", "dgd")
    labels = torch.zeros(1, dtype=torch.long, device=device)

    if sampler_type == "basic":
        sampler = GaussianDiffusionSampler(
            net, cfg["beta_1"], cfg["beta_T"], cfg["T"], w=cfg["w"]
        ).to(device)

        t_guide = cfg["t_guide"]
        eps = torch.randn_like(raw)
        guide_img = (
            extract(
                sampler.sqrt_alphas_bar,
                torch.tensor([t_guide], device=device),
                raw.shape,
            )
            * raw
            + extract(
                sampler.sqrt_one_minus_alphas_bar,
                torch.tensor([t_guide], device=device),
                raw.shape,
            )
            * eps
        ).unsqueeze(0)

        x_T = torch.randn_like(guide_img)
        out = sampler(
            x_T,
            labels,
            guide_img=guide_img,
            t_guide=t_guide,
            fuse_alpha=cfg["fuse_alpha"],
        )
    else:
        sampler = DynamicGuidanceDiffusionSampler(
            net,
            cfg["beta_1"],
            cfg["beta_T"],
            cfg["T"],
            gamma=cfg.get("dgd_gamma", 1.0),
            zeta=cfg.get("dgd_zeta", 0.5),
            mu=cfg.get("dgd_mu", 0.1),
            c=cfg.get("dgd_c", 1.0),
            eta=cfg.get("dgd_eta", 0.0),
            sigma_e=cfg.get("dgd_sigma_e", 1.0),
        ).to(device)

        x_T = torch.randn_like(y)
        out = sampler(
            x_T,
            labels,
            y=y,
            A_op=None,
            AT_op=None,
        )

    out = out * 0.5 + 0.5
    save_image(out, os.path.join(cfg["sampled_dir"], "recovered.png"))


def infer_folder(cfg: Dict):
    assert "input_dir" in cfg, "cfg must contain input_dir"
    img_list = sorted(glob.glob(os.path.join(cfg["input_dir"], "*.*")))
    if len(img_list) == 0:
        raise RuntimeError("No images found in input_dir.")
    os.makedirs(cfg["sampled_dir"], exist_ok=True)
    for p in tqdm(img_list, desc="Batch infer"):
        sub_cfg = copy.deepcopy(cfg)
        sub_cfg["noisy_img_path"] = p
        infer_single(sub_cfg)
        src = os.path.join(cfg["sampled_dir"], "recovered.png")
        dst_name = os.path.splitext(os.path.basename(p))[0] + "_rec.png"
        os.rename(src, os.path.join(cfg["sampled_dir"], dst_name))
    print(f"All images recovered and saved in: {cfg['sampled_dir']}")

    gt_dir = cfg.get("gt_dir", None)
    if gt_dir and os.path.isdir(gt_dir):
        print("Evaluating metrics between recovered images and clean GT ...")
        evaluate_metrics(gt_dir, cfg["sampled_dir"], suffix="_rec.png")


def evaluate_metrics(gt_dir: str, rec_dir: str, suffix: str = "_rec.png"):
    import numpy as np
    import os
    from skimage.metrics import peak_signal_noise_ratio as psnr
    from skimage.metrics import structural_similarity as ssim
    from PIL import Image
    import pandas as pd

    try:
        import lpips

        loss_fn = lpips.LPIPS(net="alex").cuda()
        has_lpips = True
    except Exception:
        has_lpips = False
        loss_fn = None

    rows = []
    for fname in sorted(os.listdir(gt_dir)):
        if not fname.lower().endswith((".png", ".jpg", ".bmp", ".tif")):
            continue
        name = os.path.splitext(fname)[0]
        gt_path = os.path.join(gt_dir, fname)
        rec_path = os.path.join(rec_dir, name + suffix)
        if not os.path.exists(rec_path):
            continue

        gt = (
            np.array(Image.open(gt_path).convert("L")).astype(np.float32) / 255.0
        )
        rec = (
            np.array(Image.open(rec_path).convert("L")).astype(np.float32) / 255.0
        )

        if gt.shape != rec.shape:
            from skimage.transform import resize

            rec = resize(rec, gt.shape, mode="reflect", anti_aliasing=True)

        psnr_rec = psnr(gt, rec, data_range=1.0)
        ssim_rec = ssim(gt, rec, data_range=1.0)

        if has_lpips:
            gt3 = np.stack([gt] * 3, axis=2)
            rec3 = np.stack([rec] * 3, axis=2)
            import torch as th

            lp = (
                loss_fn(
                    th.from_numpy(gt3).permute(2, 0, 1)[None].cuda(),
                    th.from_numpy(rec3).permute(2, 0, 1)[None].cuda(),
                )
                .item()
            )
        else:
            lp = np.nan

        print(
            f"{fname}:  PSNR={psnr_rec:.2f},  SSIM={ssim_rec:.3f},  LPIPS={lp:.4f}"
        )
        rows.append([fname, psnr_rec, ssim_rec, lp])

    df = pd.DataFrame(rows, columns=["filename", "PSNR", "SSIM", "LPIPS"])
    csv_path = os.path.join(rec_dir, "recover_metrics.csv")
    df.to_csv(csv_path, index=False)

    print("\n==== Mean metrics ====")
    print("Mean PSNR :", np.mean([r[1] for r in rows]))
    print("Mean SSIM :", np.mean([r[2] for r in rows]))
    if has_lpips:
        print("Mean LPIPS:", np.mean([r[3] for r in rows]))
    print(f"\nDetailed scores are saved to {csv_path}")
