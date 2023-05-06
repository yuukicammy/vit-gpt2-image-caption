import modal
from config import Config
import utils

stub = modal.Stub(Config.project_name + "-make-splited-dataset")
SHARED_ROOT = "/root/model_cache"


@stub.function(
    image=modal.Image.debian_slim().pip_install("datasets"),
    shared_volumes={SHARED_ROOT: modal.SharedVolume().persist(Config.shared_vol)},
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


@stub.function(
    image=modal.Image.debian_slim().pip_install("Pillow", "datasets"),
    shared_volumes={SHARED_ROOT: modal.SharedVolume().persist("red-caps-vol")},
    retries=0,
    cpu=1,
    # cloud="gcp",
    secret=modal.Secret.from_name("huggingface-secret"),
    timeout=86400,
)
def set_dataset_in_local(
    from_dataset_path: str = "yuukicammy/red_caps",
    to_dataset_path: str = "yuukicammy/red-caps-5k-01",
    dataset_name="5k-01",
    save_path: str = "red_caps",
    num_train: int = 3500,
    num_val: int = 500,
    num_test: int = 1000,
    save_to_hub: bool = True,
    load_from_local: bool = True,
):
    import os
    from pathlib import Path
    from datasets import load_dataset, Dataset, DatasetDict, load_from_disk

    data_root = Path(SHARED_ROOT) / save_path / dataset_name
    if load_from_local:
        created_dataset = load_from_disk(str(data_root))
    else:
        dataset = load_dataset(
            path=from_dataset_path,
            cache_dir=Path(SHARED_ROOT) / ".hf_cache",
            num_proc=14,
            use_auth_token=os.environ["HUGGINGFACE_TOKEN_READ"],
        )

        created_dataset = DatasetDict()
        train_dict = {}
        val_dict = {}
        test_dict = {}

        for split, num_examples in zip(
            ["train", "val", "test"], [num_train, num_val, num_test]
        ):
            ds = dataset[split].shuffle(Config.seed)
            cls_names = ds.info.features["subreddit"].names
            for name in cls_names:
                os.makedirs(
                    os.path.join(data_root, "images", split, name),
                    exist_ok=True,
                )
            target_dict = (
                train_dict
                if split == "train"
                else test_dict
                if split == "test"
                else val_dict
            )
            for k in ds.info.features.keys():
                target_dict[k] = []
            target_dict["subreddit_str"] = []
            valid_count = 0
            total_count = 0
            while valid_count < num_examples:
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
                    target_dict[k] += [item[k]]
                target_dict["subreddit_str"] += [cls_names[item["subreddit"]]]
                # print(f"Done {filepath}")
                total_count += 1
            created_dataset[split] = Dataset.from_dict(target_dict)
        print(created_dataset)
        created_dataset.save_to_disk(data_root)
    if save_to_hub:
        created_dataset = created_dataset.remove_columns("crosspost_parents")
        created_dataset.push_to_hub(
            to_dataset_path,
            private=True,
            token=os.environ["HUGGINGFACE_TOKEN"],
        )


@stub.local_entrypoint()
def main():
    # make_splited_dataset.call(
    #     push_hub_rep="yuukicammy/red_caps", save_disk_name="red_caps/yuukicammy"
    # )
    set_dataset_in_local.call()
