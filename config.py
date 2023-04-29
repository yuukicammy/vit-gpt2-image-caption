config_dict = {
    # For modal
    "project_name": "vit-gpt2-image-caption",
    "shared_vol": "caption-vol",
    # For NN training
    "evaluation_strategy": "step",  # ["no", "step", "epoch"]
    "batch_size_per_device": {"train": 4, "val": 4, "log": 2},
    "steps": {"train": 10, "val": 5},
    "save_limits": 3,
    "seed": 42,
    "learaning_rate": 5e-5,
    "optimizer": "adamw_hf",
    "log_dir": "image-caption-outputs",
    # For Hugging Face Dataset
    "dataset_path": "red_caps",
    "dataset_name": "all",
}


class to_obj(object):
    def __init__(self, d):
        for k, v in d.items():
            if isinstance(k, (list, tuple)):
                setattr(self, k, [to_obj(x) if isinstance(x, dict) else x for x in v])
            else:
                setattr(self, k, to_obj(v) if isinstance(v, dict) else v)


config_obj = to_obj(config_dict)
