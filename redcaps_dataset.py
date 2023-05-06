from torch.utils.data import Dataset
import utils


class RedCapsDataset(Dataset):
    def __init__(
        self,
        split: str,
        dataset_path: str,
        image_processor_pretrained: str,
        tokenizer,
        max_length: int,
        cache_root: str,
        # download_retries: int = 0,
        # download_timeout: int = 300,
        seed: int = 42,
        # num_examples: int = None,
        use_image: bool = False,
    ):
        import os
        from pathlib import Path
        from datasets import load_dataset
        from transformers import ViTImageProcessor

        self.image_processor = ViTImageProcessor.from_pretrained(
            image_processor_pretrained
        )
        self.tokenizer = tokenizer
        self.max_length = max_length

        self.cache_root = cache_root
        self.split = split

        self.hf_dataset = load_dataset(
            dataset_path,
            streaming=False,
            use_auth_token=os.environ["HUGGINGFACE_TOKEN_READ"],
            cache_dir=Path(cache_root) / ".hf_cache",
            split=split,
            keep_in_memory=True,
            save_infos=True,
            num_proc=8,
        )
        # self.download_retries = download_retries
        # self.download_timeout = download_timeout
        self.hf_dataset = self.hf_dataset.shuffle(seed=seed)

        self.num_examples = self.hf_dataset.info.splits[split].num_examples
        # (
        #     num_examples
        #     if num_examples
        #     else self.hf_dataset.info.splits[split].num_examples
        # )
        self.use_image = use_image
        self.im_size = 256

    def __len__(self):
        return self.num_examples

    def __getitem__(self, idx):
        metadata = self.hf_dataset[idx]
        import os
        from pathlib import Path
        from PIL import Image
        import numpy as np

        filepath = (
            Path(self.cache_root)
            / "red_caps/5k-01/images"
            / self.split
            / metadata["subreddit_str"]
            / f"{metadata['image_id']}.jpg"
        )
        if not os.path.isfile(filepath):
            return self.__getitem__(idx=(idx + 1) % self.num_examples)
        image = Image.open(filepath)
        pixel_values = self.image_processor(
            image, return_tensors="pt", data_format="channels_first"
        ).pixel_values.squeeze()  # ViTImageProcessor.preprocess() returns in batch format.
        # print(pixel_values.shape)
        labels = self.tokenizer(
            metadata["caption"], padding="max_length", max_length=self.max_length
        ).input_ids
        if self.use_image:
            return {
                "pixel_values": pixel_values,
                "labels": labels,
                "image": np.array(image.resize((self.im_size, self.im_size))),
            }
        else:
            return {"pixel_values": pixel_values, "labels": labels}
