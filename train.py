from transformers import default_data_collator
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments
from config import config_dict as config

import modal
from config import config_dict as config
from dataset import DatasetPreprocess

stub = modal.Stub(config["project_name"] + "-train")


@stub.function(
    gpu="any",
    image=modal.Image.debian_slim().pip_install("Pillow", "transformers", "torch"),
    shared_volumes={"/root/model_cache": modal.SharedVolume.from_name(config["shared_vol"])},
    retries=3,
)
def train(dataset_disk_path:str="red_caps/yuukicammy", batch_size:int=32,):
    from datasets import load_dataset
    import torch
    from torch.utils.data import DataLoader
    train_dataset = load_dataset.load_from_disk(dataset_disk_path, split="train")
    val_dataset = load_dataset.load_from_disk(dataset_disk_path, split="val")
    dataset.set_format(type='torch', columns=['input_ids', 'token_type_ids', 'attention_mask', 'labels'])
    train_dataloader = DataLoader(dataset, batch_size=32)
    
    DatasetPreprocess().fetch_images


        text_tokenizer: str = "nlpconnect/vit-gpt2-image-captioning",
        image_processer: str = "nlpconnect/vit-gpt2-image-captioning",
        vision_encoder_decoder_model: str = "nlpconnect/vit-gpt2-image-captioning",
 # image feature extractor
        self.feature_extractor = ViTImageProcessor.from_pretrained(image_processer)

        # text tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(text_tokenizer)

        # GPT2 only has bos/eos tokens but not decoder_start/pad tokens
        self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = VisionEncoderDecoderModel.from_pretrained(
            vision_encoder_decoder_model
        )
        # update the model config
        self.max_target_length = max_target_length
        self.model.config.eos_token_id = self.tokenizer.eos_token_id
        self.model.config.decoder_start_token_id = self.tokenizer.bos_token_id
        self.model.config.pad_token_id = self.tokenizer.pad_token_id

            def tokenization(self, captions, max_target_length):
        """Run tokenization on captions."""

        labels = self.tokenizer(
            captions, padding="max_length", max_length=max_target_length
        ).input_ids

        return labels

def train(
    model,
    feature_extractor,
    compute_metrics,
    processed_dataset,
):
    import time
    from pathlib import Path

    output_dir = Path(config["log_dir"]) / f"train-{time.strftime('%Y%m%d-%H%M%S')}"
    training_args = Seq2SeqTrainingArguments(
        predict_with_generate=True,  # Whether to use generate to calculate generative metrics (ROUGE, BLEU).
        evaluation_strategy=config["evaluation_strategy"],  # ["no", "epoch", "step"]
        per_device_train_batch_size=config["batch_size_per_device"]["train"],
        per_device_eval_batch_size=config["batch_size_per_device"]["eval"],
        output_dir=output_dir,
        eval_steps=config["steps"]["eval"],
        logging_steps=config["steps"]["log"],
        learning_rate=config["learaning_rate"],
        seed=config["seed"],
        save_total_limit=config["save_limits"],
    )
    # instantiate trainer
    trainer = Seq2SeqTrainer(
        model=model,
        tokenizer=feature_extractor,
        args=training_args,
        compute_metrics=compute_metrics,
        train_dataset=processed_dataset["train"],
        eval_dataset=processed_dataset["validation"],
        data_collator=default_data_collator,
    )
    trainer.train()
