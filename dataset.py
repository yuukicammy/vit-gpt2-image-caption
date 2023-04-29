import modal
from config import config_dict as config
from typing import List, Any, Dict
import PIL


stub = modal.Stub(config["project_name"] + "-red-cap-dataset")
SHARED_ROOT = "/root/model_cache"


@stub.cls(
    image=modal.Image.debian_slim().pip_install("datasets", "Pillow"),
    shared_volumes={SHARED_ROOT: modal.SharedVolume.from_name(config["shared_vol"])},
    retries=3,
    cpu=14,
    secret=modal.Secret.from_name("huggingface-secret"),
    timeout=86400,
)
class DatasetPreprocess:
    def __enter__(self):
        from pathlib import Path
        from datasets.utils.file_utils import get_datasets_user_agent

        self.user_agent = get_datasets_user_agent()
        self.cache_root = Path(SHARED_ROOT)

    @modal.method()
    def fetch_single_image(self, image_url: str, timeout: int = None, retries: int = 0):
        import io
        import urllib
        import PIL

        image = None
        for _ in range(retries + 1):
            try:
                request = urllib.request.Request(
                    image_url,
                    data=None,
                    headers={"user-agent": self.user_agent},
                )
                with urllib.request.urlopen(request, timeout=timeout) as req:
                    image = PIL.Image.open(io.BytesIO(req.read()))
                break
            except Exception:
                print(f"Failed to download an image from url: {image_url}.")
        return image

    @modal.method()
    def fetch_images(
        self, batch, num_workers: int = 4, timeout: int = None, retries: int = 0
    ):
        from concurrent.futures import ThreadPoolExecutor
        from functools import partial

        fetch_single_image_with_args = partial(
            self.fetch_single_image, timeout=timeout, retries=retries
        )
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            batch["image"] = list(
                executor.map(fetch_single_image_with_args, batch["image_url"])
            )
        return batch

    @modal.method()
    def make_splits(
        self,
        dataset_path: str = "red_caps",
        dataset_name: str = "all",
        split: str = "train",
        push_hub_rep: str = None,
        save_disk_name: str = "red_caps/yuukicammy/",
    ):
        import os
        from datasets import load_dataset, DatasetDict

        dataset = load_dataset(
            dataset_path,
            dataset_name,
            split=split,
            cache_dir=self.cache_root,
            num_proc=14,
        )
        print(dataset)
        trainval_test = dataset.train_test_split(
            test_size=0.2, seed=config["seed"], shuffle=True
        )
        train_valid = trainval_test["train"].train_test_split(
            test_size=0.1, seed=config["seed"], shuffle=True
        )
        print(trainval_test)
        print(train_valid)
        dataset = DatasetDict(
            {
                "train": train_valid["train"],
                "test": trainval_test["test"],
                "valid": train_valid["test"],
            }
        )
        try:
            # dataset.save_to_disk(self.cache_root / save_disk_name)
            if push_hub_rep:
                dataset.push_to_hub(
                    push_hub_rep,
                    private=True,
                    token=os.environ["HUGGINGFACE_TOKEN"],
                )
        except Exception as e:
            print("type: " + str(type(e)))
            print("args: " + str(e.args))
            print("message: " + e.message)
            print(e)


@stub.local_entrypoint()
def main():
    DatasetPreprocess().make_splits.call(push_hub_rep="yuukicammy/red_caps")
