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
        download_retries: int = 0,
        download_timeout: int = 300,
        seed: int = 42,
        num_examples: int = None,
    ):
        import os
        from datasets import load_dataset
        from transformers import ViTImageProcessor

        self.image_processor = ViTImageProcessor.from_pretrained(
            image_processor_pretrained
        )
        self.tokenizer = tokenizer
        self.max_length = max_length

        self.hf_dataset = load_dataset(
            dataset_path,
            streaming=True,
            use_auth_token=os.environ["HUGGINGFACE_TOKEN_READ"],
            cache_dir=cache_root,
            split=split,
        )
        self.download_retries = download_retries
        self.download_timeout = download_timeout
        self.hf_dataset = self.hf_dataset.shuffle(seed=seed)

        self.num_examples = (
            num_examples
            if num_examples
            else self.hf_dataset.info.splits[split].num_examples
        )

    def __len__(self):
        return self.num_examples

    def __getitem__(self, idx):
        metadata = next(iter(self.hf_dataset))
        image = utils.download_image(
            metadata["image_url"],
            retries=self.download_retries,
            timeout=self.download_timeout,
        )
        while image is None:
            metadata = next(iter(self.hf_dataset))
            image = utils.download_image(
                metadata["image_url"],
                retries=self.download_retries,
                timeout=self.download_timeout,
            )
        pixel_values = self.image_processor(
            image, return_tensors="pt", data_format="channels_first"
        ).pixel_values.squeeze()  # ViTImageProcessor.preprocess() returns in batch format.
        # print(pixel_values.shape)
        labels = self.tokenizer(
            metadata["caption"], padding="max_length", max_length=self.max_length
        ).input_ids
        return {"pixel_values": pixel_values, "labels": labels}
