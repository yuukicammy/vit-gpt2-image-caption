""" RedCaps dataset <https://redcaps.xyz/>

MIT License
Copyright (c) 2023 yuukicammy
"""

from torch.utils.data import Dataset


class RedCapsDataset(Dataset):
    f"""Dataset class for RedCaps <https://redcaps.xyz/> .

    Args:
        split (str): The split to load from the dataset. train, val, or test.
        dataset_path (str): The path to the dataset.
        image_processor_pretrained (str): The name or path of the pre-trained image processor.
        tokenizer: The tokenizer to use.
        max_length (int): The maximum length of tokens for the captions.
        cache_root (str): The root path for caching data.
        seed (int, optional): The random seed. Defaults to 42.
        use_input (bool, optional): Whether to use the original images and captions in the returned dict. Defaults to False.
    """

    def __init__(
        self,
        split: str,
        dataset_path: str,
        image_processor_pretrained: str,
        tokenizer,
        max_length: int,
        cache_root: str,
        seed: int = 42,
        use_input: bool = False,
    ):
        import os
        from pathlib import Path
        from datasets import load_dataset, load_from_disk
        from transformers import ViTImageProcessor

        self.image_processor = ViTImageProcessor.from_pretrained(
            image_processor_pretrained
        )
        self.tokenizer = tokenizer
        self.max_length = max_length

        self.cache_root = cache_root
        self.split = split
        self.dataset_path = dataset_path

        if os.path.exists(self.dataset_path):
            self.hf_dataset = load_from_disk(self.dataset_path)[split]
        else:
            self.hf_dataset = load_dataset(
                dataset_path,
                streaming=False,
                use_auth_token=os.environ["HUGGINGFACE_TOKEN_READ"],
                cache_dir=cache_root,
                split=split,
                keep_in_memory=True,
                save_infos=True,
                num_proc=8,
            )
        self.hf_dataset = self.hf_dataset.shuffle(seed=seed)
        self.num_examples = self.hf_dataset.num_rows
        self.use_input = use_input
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
            Path(self.dataset_path)
            / "images"
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
            metadata["caption"],
            padding="max_length",
            max_length=self.max_length,
        ).input_ids
        if self.use_input:
            return {
                "ann_caption": metadata["caption"],
                "pixel_values": pixel_values,
                "labels": labels,
                "image": np.array(image.resize((self.im_size, self.im_size))),
            }
        else:
            return {"pixel_values": pixel_values, "labels": labels}
