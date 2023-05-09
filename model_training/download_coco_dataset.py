""" Split the Hugging Face dataset into train, val, and test.
MIT License
Copyright (c) 2023 yuukicammy
"""
import modal
from model_training.config import Config

stub = modal.Stub(Config.project_name + "-download-coco")
SHARED_ROOT = "/root/model_cache"

docker_command = [
    "RUN apt-get update && apt-get install -y git",
    # For COCO dataset
    "RUN git clone https://github.com/cocodataset/cocoapi && cd cocoapi/PythonAPI && make",
]


@stub.function(
    image=modal.Image.debian_slim()
    .pip_install("torchvision", "cython")
    .dockerfile_commands(docker_command),
    shared_volumes={SHARED_ROOT: modal.SharedVolume().persist(Config.shared_vol)},
    retries=3,
    cpu=2,
    secret=modal.Secret.from_name("huggingface-secret"),
    timeout=3600,
)
def download_coco_dataset() -> None:
    import os
    import multiprocessing
    from pathlib import Path
    from torchvision.datasets.utils import download_and_extract_archive

    dataset_root = Path(SHARED_ROOT) / "coco"
    os.makedirs(dataset_root, exist_ok=True)
    download_targetd = [
        {
            "url": "http://images.cocodataset.org/zips/train2017.zip",
            "save_path": f"{dataset_root}/train2017.zip",
        },
        {
            "url": "http://images.cocodataset.org/zips/val2017.zip",
            "save_path": f"{dataset_root}/val2017.zip",
        },
        {
            "url": "http://images.cocodataset.org/annotations/annotations_trainval2017.zip",
            "save_path": f"{dataset_root}/annotations_trainval2017.zip",
        },
    ]
    try:
        jobs = []
        for target in download_targetd:
            p = multiprocessing.Process(
                target=download_and_extract_archive,
                args=(
                    target["url"],
                    dataset_root,
                    dataset_root,
                    target["save_path"],
                    None,
                    True,  # delete archive
                ),
            )
            jobs.append(p)
            p.start()
        for proc in jobs:
            proc.join()

    except Exception as e:
        print(e)
        print("type: " + str(type(e)))
        print("args: " + str(e.args))
        print("message: " + e.message)


@stub.local_entrypoint()
def main():
    download_coco_dataset.call()
