import modal
from config import config_dict as config
from typing import List, Any, Dict
import PIL


stub = modal.Stub(config["project_name"] + "-download-dataset")
SHARED_ROOT = "/root/model_cache"


@stub.cls(
    image=modal.Image.debian_slim().pip_install("datasets", "Pillow"),
    shared_volumes={SHARED_ROOT: modal.SharedVolume.from_name(config["shared_vol"])},
    retries=3,
    cpu=5,
    cpu=14,
)
class DownloadDataset:
    def __enter__(self):
        from pathlib import Path
        from datasets.utils.file_utils import get_datasets_user_agent

        self.dataset_path = config["dataset_path"]
        self.dataset_name = config["dataset_name"]
        self.user_agent = get_datasets_user_agent()
        self.cache_root = Path(SHARED_ROOT)
        self.seed = config["seed"]

    @modal.method()
    def _process_batch(self, batch):
        batch["image_url"] = self.process_image_urls(batch["image_url"])
        images = self.download_images(batch["image_url"])
        image_paths = []
        for img, _id in zip(images, batch["image_id"]):
            if img:
                img_path = self.image_dir / f"{_id}.jpg"
                img.save(img_path)
                image_paths.append(img_path)
            else:
                image_paths.append("")
        batch["image_path"] = img_path
        return batch

    @modal.method()
    def process(
        self,
        split: str,
        batch_size: int = 16,
        num_workers: int = 8,
        download_retries: int = 5,
        download_percentage: int = 1,  # [0,100]
    ) -> None:
        import os
        from datasets import load_dataset

        self.batch_size = batch_size
        self.split = split
        self.num_workers = num_workers
        self.download_retries = download_retries

        dataset = load_dataset(
            self.dataset_path,
            self.dataset_name,
            split=f"{self.split}[:{download_percentage}%]",
            cache_dir=self.cache_root,
            num_proc=self.num_workers,
        )
        self.image_dir = (
            self.cache_root
            / self.dataset_path
            / self.dataset_name
            / self.split
            / "images"
        )
        os.makedirs(self.image_dir, exist_ok=True)
        dataset = dataset.map(
            self._process_batch, batched=True, batch_size=self.batch_size
        )
        dataset.save_to_disk(self.cache_root / self.split)

    @modal.method()
    def process_image_urls(self, urls):
        import os
        import re

        processed_batch_image_urls = []
        for image_url in urls:
            processed_example_image_urls = []
            image_url_splits = re.findall(r"http\S+", image_url)
            for image_url_split in image_url_splits:
                if "imgur" in image_url_split and "," in image_url_split:
                    for image_url_part in image_url_split.split(","):
                        if not image_url_part:
                            continue
                        image_url_part = image_url_part.strip()
                        root, ext = os.path.splitext(image_url_part)
                        if not root.startswith("http"):
                            root = "http://i.imgur.com/" + root
                        root = root.split("#")[0]
                        if not ext:
                            ext = ".jpg"
                        ext = re.split(r"[?%]", ext)[0]
                        image_url_part = root + ext
                        processed_example_image_urls.append(image_url_part)
                else:
                    processed_example_image_urls.append(image_url_split)
            processed_batch_image_urls.append(processed_example_image_urls)
        return processed_batch_image_urls

    @modal.method()
    def download_image(self, image_url: str, timeout: int = None, retries: int = None):
        import urllib
        import io
        from PIL import Image

        if not retries or retries < 0:
            retries = self.download_retries
        for _ in range(retries + 1):
            try:
                request = urllib.request.Request(
                    image_url,
                    data=None,
                    headers={"user-agent": self.user_agent},
                )
                with urllib.request.urlopen(request, timeout=timeout) as req:
                    image = Image.open(io.BytesIO(req.read()))
                break
            except Exception:
                print(f"Failed to download an image from url: {image_url}.")
                image = None
        return image

    @modal.method()
    def download_images(
        self,
        image_urls: List[str],
        timeout: int = None,
        retries: int = None,
    ):
        from functools import partial
        from concurrent.futures import ThreadPoolExecutor

        fetch_single_image_with_args = partial(
            self.download_image,
            timeout=timeout,
            retries=retries if retries else self.download_retries,
        )
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            images = list(executor.map(fetch_single_image_with_args, image_urls))
        return images


@stub.local_entrypoint()
def main():
    for split in ["train", "validation"]:
        DownloadDataset().process.call(split)
