import io
import urllib
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from functools import partial
import PIL
from datasets.utils.file_utils import get_datasets_user_agent

USER_AGENT = get_datasets_user_agent()


def fetch_single_image(image_url: str, timeout: int = None, retries: int = 0):
    for _ in range(retries + 1):
        try:
            request = urllib.request.Request(
                image_url,
                data=None,
                headers={"user-agent": USER_AGENT},
            )
            with urllib.request.urlopen(request, timeout=timeout) as req:
                image = PIL.Image.open(io.BytesIO(req.read()))
            break
        except Exception as e:
            print(e)
            print(f"Failed to download an image from url: {image_url}.")
            print("Set Image to None. Must remove it with a filter function.")
            image = None
    return image


def fetch_images(batch, num_workers: int = 4, timeout: int = None, retries: int = 0):
    if isinstance(batch["image_url"], str):
        batch["image"] = fetch_single_image(batch["image_url"])
    else:
        fetch_single_image_with_args = partial(
            fetch_single_image, timeout=timeout, retries=retries
        )
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            batch["image"] = list(
                executor.map(fetch_single_image_with_args, batch["image_url"])
            )
    return batch
