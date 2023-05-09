import modal
from datasets import Dataset
from model_training.config import Config
import model_training.utils as utils

stub = modal.Stub(Config.project_name + "-build-dataset-subset")
SHARED_ROOT = "/root/model_cache"


@stub.function(
    image=modal.Image.debian_slim().pip_install("Pillow", "datasets"),
    shared_volumes={SHARED_ROOT: modal.SharedVolume.from_name(Config.shared_vol)},
    retries=0,
    cpu=5,
    secret=modal.Secret.from_name("huggingface-secret"),
    timeout=3600,
)
def build_dataset_subset(
    from_dataset_path: str = "red_caps",
    to_dataset_path: str = "red-caps-5k-01",
    num_train: int = 3500,
    num_val: int = 500,
    num_test: int = 1000,
    push_hub_rep: str = None,
):
    """
    Downloads images from the specified Hugging Face dataset and saves them to a local folder. If the dataset has already been downloaded to the local folder, it can be loaded from disk
    instead of being downloaded again.

    Args:
        from_dataset_path (str): The path of the Hugging Face dataset to download from.
        to_dataset_path (str): The path to upload the dataset to on Hugging Face after it has been processed.
        dataset_name (str): The name of the dataset being processed.
        num_train (int): The number of training examples to download.
        num_val (int): The number of validation examples to download.
        num_test (int): The number of test examples to download.
        save_to_hub (bool): Whether or not to save the processed dataset to Hugging Face.

    Returns:
        None
    """
    import os
    from pathlib import Path
    import multiprocessing
    from datasets import load_dataset, Dataset, DatasetDict, load_from_disk

    data_root = Path(SHARED_ROOT) / to_dataset_path
    if os.path.exists(Path(SHARED_ROOT) / from_dataset_path):
        from_dataset_path = Path(SHARED_ROOT) / from_dataset_path
        dataset = load_from_disk(from_dataset_path)
    else:
        dataset = load_dataset(
            path=from_dataset_path,
            cache_dir=Path(SHARED_ROOT) / ".hf_cache",
            num_proc=5,
            use_auth_token=os.environ["HUGGINGFACE_TOKEN_READ"],
        )
    print(dataset)
    result_dict = multiprocessing.Manager().dict()
    jobs = []
    for split, num_examples in zip(
        ["train", "val", "test"], [num_train, num_val, num_test]
    ):
        p = multiprocessing.Process(
            target=build_splited_dataset,
            args=(dataset[split], data_root, split, num_examples, result_dict),
        )
        jobs.append(p)
        p.start()
    for proc in jobs:
        proc.join()

    created_dataset = DatasetDict(
        {
            "train": Dataset.from_dict(result_dict["train"]),
            "val": Dataset.from_dict(result_dict["val"]),
            "test": Dataset.from_dict(result_dict["test"]),
        }
    )
    print(created_dataset)
    created_dataset.save_to_disk(data_root)
    if push_hub_rep:
        created_dataset = created_dataset.remove_columns("crosspost_parents")
        created_dataset.push_to_hub(
            push_hub_rep,
            private=True,
            token=os.environ["HUGGINGFACE_TOKEN"],
        )


def build_splited_dataset(
    ds: Dataset,
    data_root: str,
    split: str,
    num_examples: int,
    result_dict,
) -> None:
    import os

    split_dict = {}
    print(f"Shuffle {split}...")
    ds = ds.shuffle(Config.seed)
    print(f"Done shuffling {split}.")
    cls_names = ds.info.features["subreddit"].names
    for name in cls_names:
        os.makedirs(
            os.path.join(data_root, "images", split, name),
            exist_ok=True,
        )
    for k in ds.info.features.keys():
        split_dict[k] = []
    split_dict["subreddit_str"] = []
    valid_count = 0
    total_count = 0
    while valid_count < num_examples and total_count < len(ds):
        item = ds[total_count]
        total_count += 1
        cls_name = cls_names[item["subreddit"]]
        filepath = os.path.join(
            data_root,
            "images",
            split,
            cls_name,
            f"{item['image_id']}.jpg",
        )
        if not os.path.isfile(filepath):
            image = utils.download_image(item["image_url"], timeout=1)
            if image is None:
                total_count += 1
                continue
            image = image.convert("RGB")
            image.save(filepath)
        valid_count += 1
        for k in ds.info.features.keys():
            split_dict[k] += [item[k]]
        split_dict["subreddit_str"] += [cls_names[item["subreddit"]]]
        # print(f"Done {filepath}")
        total_count += 1

    result_dict[split] = split_dict


@stub.local_entrypoint()
def main(
    from_dataset_path: str = "red_caps",
    to_dataset_path: str = "red-caps-5k-01",
    num_train: int = 3500,
    num_val: int = 500,
    num_test: int = 1000,
    push_hub_rep: str = None,
):
    build_dataset_subset.call(
        from_dataset_path=from_dataset_path,
        to_dataset_path=to_dataset_path,
        num_train=num_train,
        num_val=num_val,
        num_test=num_test,
        push_hub_rep=push_hub_rep,
    )
