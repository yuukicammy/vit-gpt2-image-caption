import modal
from config import Config

stub = modal.Stub(Config.project_name + "-make-splited-dataset")
SHARED_ROOT = "/root/model_cache"


@stub.function(
    image=modal.Image.debian_slim().pip_install("datasets"),
    shared_volumes={SHARED_ROOT: modal.SharedVolume.from_name(Config.shared_vol)},
    retries=3,
    cpu=14,
    secret=modal.Secret.from_name("huggingface-secret"),
    timeout=86400,
)
def make_splited_dataset(
    dataset_path: str = "red_caps",
    dataset_name: str = "all",
    split: str = "train",
    push_hub_rep: str = None,
    save_disk_name: str = "red_caps/yuukicammy/",
) -> None:
    import os
    from pathlib import Path
    from datasets import load_dataset, DatasetDict

    cache_root = Path(SHARED_ROOT)

    dataset = load_dataset(
        dataset_path,
        dataset_name,
        split=split,
        cache_dir=cache_root,
        num_proc=14,
    )
    print(dataset)
    trainval_test = dataset.train_test_split(
        test_size=0.2, seed=Config.seed, shuffle=True
    )
    train_valid = trainval_test["train"].train_test_split(
        test_size=0.1, seed=Config.seed, shuffle=True
    )
    print(trainval_test)
    print(train_valid)
    dataset = DatasetDict(
        {
            "train": train_valid["train"],
            "test": trainval_test["test"],
            "val": train_valid["test"],
        }
    )
    try:
        if save_disk_name:
            dataset.save_to_disk(cache_root / save_disk_name)
        if push_hub_rep:
            dataset.push_to_hub(
                push_hub_rep,
                private=True,
                token=os.environ["HUGGINGFACE_TOKEN"],
            )
    except Exception as e:
        print(e)
        print("type: " + str(type(e)))
        print("args: " + str(e.args))
        print("message: " + e.message)


@stub.local_entrypoint()
def main():
    make_splited_dataset.call(
        push_hub_rep="yuukicammy/red_caps", save_disk_name="red_caps/yuukicammy"
    )
