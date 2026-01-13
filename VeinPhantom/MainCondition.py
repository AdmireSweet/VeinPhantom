# MainCondition.py
from DiffusionFreeGuidence.TrainCondition import train, infer_single, infer_folder

def main(custom_cfg=None):
    cfg = {
        "state": "train",               # "train" / "infer" / "infer_folder"
        "epoch": 100,
        "batch_size": 16,
        "img_size": 128,
        "T": 500,
        "channel": 128,
        "channel_mult": [1, 2, 2, 2],
        "num_res_blocks": 2,
        "dropout": 0.15,
        "lr": 1e-4,
        "multiplier": 2.5,
        "beta_1": 1e-4,
        "beta_T": 0.02,
        "grad_clip": 1.0,
        "device": "cuda:0",
        "save_dir": "./CheckpointsCondition/",
        "training_load_weight": None,
        "test_load_weight": "ckpt_99.pt",
        "num_labels": 500,
        "input_dir": "./test_input",
        "gt_dir": "./test_gt/",
        "sampled_dir": "./test_output/",
        "t_guide": 250,
        "fuse_alpha": 1.0,
        "w": 0.0,
        "sampler": "dgd",
        "dgd_gamma": 1.0,
        "dgd_zeta": 0.5,
        "dgd_mu": 0.1,
        "dgd_c": 1.0,
        "dgd_eta": 0.0,
        "dgd_sigma_e": 1.0,
    }

   
    if custom_cfg:
        cfg.update(custom_cfg)

    if cfg["state"] == "train":
        train(cfg)
    elif cfg["state"] == "infer":
        infer_single(cfg)
    elif cfg["state"] == "infer_folder":
        infer_folder(cfg)
    else:
        raise ValueError(f"Unknown state: {cfg['state']}")


if __name__ == "__main__":
    main()
