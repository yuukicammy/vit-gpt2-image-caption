""" Split the Hugging Face dataset into train, val, and test.
MIT License
Copyright (c) 2023 yuukicammy
"""

import modal
from model_training.config import Config

stub = modal.Stub(Config.project_name + "-split-dataset")
SHARED_ROOT = "/root/model_cache"


@stub.function(
    image=modal.Image.debian_slim().pip_install("datasets"),
    shared_volumes={SHARED_ROOT: modal.SharedVolume().persist(Config.shared_vol)},
    retries=3,
    cpu=14,
    secret=modal.Secret.from_name("huggingface-secret"),
    timeout=3600,
)
def make_splited_dataset(
    dataset_path: str = "red_caps",
    dataset_name: str = "all",
    split: str = None,
    push_hub_rep: str = None,
    save_dir: str = None,
) -> None:
    """Load a dataset and split it into train, validation, and test sets.

    Args:
        dataset_path (str, optional): Path to the dataset. Defaults to "red_caps".
        dataset_name (str, optional): Name of the dataset. Defaults to "all".
        split (str, optional): The split of the dataset to use. Defaults to None.
        push_hub_rep (str, optional): The name of the repository on Hugging Face to push the dataset to. Defaults to None.
        save_dir (str, optional): The name of the directory to save the dataset to the shared volume. Defaults to None.

    Returns:
        None

    Raises:
        Exception: An exception is raised if saving or pushing the dataset fails.

    """
    import os
    from pathlib import Path
    from datasets import load_dataset, DatasetDict

    cache_root = Path(SHARED_ROOT) / ".hf_cache"

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
        if save_dir:
            dataset.save_to_disk(Path(SHARED_ROOT) / save_dir)
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
def main(
    dataset_path: str = "red_caps",
    dataset_name: str = "all",
    push_hub_rep: str = "yuukicammy/red_caps",
    split: str = "train",
    save_dir: str = None,
):
    make_splited_dataset.call(
        dataset_path=dataset_path,
        dataset_name=dataset_name,
        push_hub_rep=push_hub_rep,
        split=split,
        save_dir=save_dir,
    )
