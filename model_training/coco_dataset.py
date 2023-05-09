""" Download COCO dataset from the official repositories.
MIT License
Copyright (c) 2023 yuukicammy
"""
from torch.utils.data import Dataset


class CocoDataset(Dataset):
    """Download COCO dataset from the official repositories.

    This class implements a PyTorch Dataset for the COCO dataset. It downloads the dataset from the official repositories and preprocesses the images and captions for training. The dataset contains images with corresponding captions.

    Args:
        split (str): The split to use for the dataset, either 'train' or 'val'.
        dataset_path (str): The path to the COCO dataset on disk.
        image_processor_pretrained (str): The name of the ViTImageProcessor model to use for preprocessing images.
        tokenizer (obj): A tokenizer object to use for tokenizing captions.
        max_length (int): The maximum length to truncate captions to.
        seed (int, optional): The random seed to use for shuffling. Default is 42.
        use_input (bool, optional): Whether to include the image and caption in the output. Default is False.

    Returns:
        dict: A dictionary containing the pixel values of the preprocessed image and the tokenized labels for the captions.
    """

    def __init__(
        self,
        split: str,
        dataset_path: str,
        image_processor_pretrained: str,
        tokenizer,
        max_length: int,
        seed: int = 42,
        use_input: bool = False,
    ):
        import random
        from pathlib import Path
        from transformers import ViTImageProcessor
        from torchvision.datasets import CocoCaptions

        random.seed(seed)

        self.image_processor = ViTImageProcessor.from_pretrained(
            image_processor_pretrained
        )
        self.tokenizer = tokenizer
        self.max_length = max_length

        self.split = split
        self.dataset_path = dataset_path

        image_dir = (
            Path(dataset_path) / "train2017"
            if split == "train"
            else Path(dataset_path) / "val2017"
        )
        annotation_filepath = (
            Path(dataset_path) / "annotations" / "captions_train2017.json"
            if split == "train"
            else Path(dataset_path) / "annotations" / "captions_val2017.json"
        )
        self.dataset = CocoCaptions(
            root=image_dir,
            annFile=annotation_filepath,
        )
        self.use_input = use_input
        self.im_size = 256

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        import random
        import numpy as np

        i = random.randint(0, 4)
        image, captions = self.dataset[idx]
        pixel_values = self.image_processor(
            image, return_tensors="pt", data_format="channels_first"
        ).pixel_values.squeeze()  # ViTImageProcessor.preprocess() returns in batch format.
        # print(pixel_values.shape)
        labels = self.tokenizer(
            captions[i],
            padding="max_length",
            max_length=self.max_length,
        ).input_ids
        if self.use_input:
            return {
                "ann_caption": captions[i],
                "pixel_values": pixel_values,
                "labels": labels,
                "image": np.array(image.resize((self.im_size, self.im_size))),
            }
        else:
            return {"pixel_values": pixel_values, "labels": labels}
