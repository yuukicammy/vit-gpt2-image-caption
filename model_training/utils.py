import io
from typing import List
from datasets.utils.file_utils import get_datasets_user_agent

USER_AGENT = get_datasets_user_agent()


def download_image(image_url: str, timeout: int = None, retries: int = 0):
    import urllib
    from PIL import Image

    for _ in range(retries + 1):
        try:
            request = urllib.request.Request(
                image_url,
                data=None,
                headers={"user-agent": USER_AGENT},
            )
            with urllib.request.urlopen(request, timeout=timeout) as req:
                image = Image.open(io.BytesIO(req.read()))
            break
        except Exception as e:
            print(e)
            print(f"Failed to download an image from url: {image_url}.")
            print("Set Image to None.")
            image = None
    return image


def download_images(
    image_urls: List[str], num_workers: int = 4, timeout: int = None, retries: int = 0
):
    from concurrent.futures import ThreadPoolExecutor
    from functools import partial

    images = None
    if isinstance(image_urls, str):
        images = download_image(image_urls)
    else:
        fetch_single_image_with_args = partial(
            download_image, timeout=timeout, retries=retries
        )
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            images = list(executor.map(fetch_single_image_with_args, image_urls))
    return images


class to_obj(object):
    def __init__(self, d):
        for k, v in d.items():
            if isinstance(k, (list, tuple)):
                setattr(self, k, [to_obj(x) if isinstance(x, dict) else x for x in v])
            else:
                setattr(self, k, to_obj(v) if isinstance(v, dict) else v)
