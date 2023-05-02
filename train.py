from config import config_dict as config
import modal
from module import CaptionTransformer
from redcaps_dataset import RedCapsDataset

docker_command = [
    "RUN apt-get update && apt-get install -y git",
    #    "RUN pip uninstall datasets",
    "RUN git clone --branch add-fn-kwargs-to-iterable-map-and-filter https://github.com/yuukicammy/datasets.git",
    'RUN cd datasets && pip install -e ".[dev]"',
]

stub = modal.Stub(
    config["project_name"] + "-train",
    image=modal.Image.debian_slim()
    .pip_install(
        "Pillow",
        "tensorboard",
        "transformers",
        "torchvision",
        "evaluate",
        "numpy",
        "nltk",
        "torch",
        "kornia",
        "rouge_score",
    )
    .dockerfile_commands(docker_command, force_build=False),
)
SHARED_ROOT = "/root/model_cache"


@stub.cls(
    gpu="any",
    cpu=14,
    shared_volumes={SHARED_ROOT: modal.SharedVolume.from_name(config["shared_vol"])},
    retries=0,
    secret=modal.Secret.from_name("huggingface-secret"),
    timeout=86400,
    interactive=False,
)
class FineTune:
    def __enter__(self):
        import evaluate

        self.metric = evaluate.load(config["metric"])
        self.ignore_pad_token_for_loss = config["ignore_pad_token_for_loss"]
        self.module = CaptionTransformer(
            tokenizer_pretrained_path=config["tokenizer_pretrained"],
            model_pretrained_path=config["encoder_decoder_pretrained"],
            max_target_length=config["max_target_length"],
            ignore_pad_token_for_loss=config["ignore_pad_token_for_loss"],
        )

    @modal.method()
    def run(self):
        import os
        import time
        import json
        from pathlib import Path

        from transformers import default_data_collator
        from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments

        print("start!")
        output_dir = Path(SHARED_ROOT) / config["log_dir"]
        # # os.makedirs(
        # #     output_dir / "run" / f"{time.strftime('%b%d_%H_%M_%S')}_modal",
        # #     exist_ok=True,
        # # )
        # with open(
        #     output_dir / f"{time.strftime('%b%d_%H_%M_%S')}_config.json",
        #     mode="w",
        #     encoding="utf-8",
        # ) as f:
        #     json.dump(config, f)

        training_args = Seq2SeqTrainingArguments(
            predict_with_generate=True,  # Whether to use generate to calculate generative metrics (ROUGE, BLEU).
            evaluation_strategy=config[
                "evaluation_strategy"
            ],  # ["no", "epoch", "step"]
            per_device_train_batch_size=config["batch_size_per_device"]["train"],
            per_device_eval_batch_size=config["batch_size_per_device"]["val"],
            output_dir=output_dir,
            eval_steps=config["steps"]["val"],
            logging_steps=config["steps"]["log"],
            learning_rate=config["learaning_rate"],
            logging_first_step=True,
            seed=config["seed"],
            save_total_limit=config["save_limits"],
            max_steps=config["max_steps"],
            # log_level="debug",
            dataloader_num_workers=config["dataloader_num_workers"],
            disable_tqdm=True,
            push_to_hub=False,
        )
        # instantiate trainer
        trainer = Seq2SeqTrainer(
            model=self.module.model,
            tokenizer=self.module.tokenizer,
            args=training_args,
            compute_metrics=self.compute_metrics,
            train_dataset=RedCapsDataset(
                split="train",
                dataset_path=config["dataset_path"],
                image_processor_pretrained=config["image_processor_pretrained"],
                tokenizer=self.module.tokenizer,
                max_length=config["max_target_length"],
                cache_root=SHARED_ROOT,
                download_retries=config["download_retries"],
                seed=config["seed"],
            ),
            eval_dataset=RedCapsDataset(
                split="val",
                dataset_path=config["dataset_path"],
                image_processor_pretrained=config["image_processor_pretrained"],
                tokenizer=self.module.tokenizer,
                max_length=config["max_target_length"],
                cache_root=SHARED_ROOT,
                download_retries=config["download_retries"],
                seed=config["seed"],
                num_examples=10,
            ),
            data_collator=default_data_collator,
        )
        # breakpoint()
        trainer.train()

    def compute_metrics(self, eval_preds):
        # receive data as numpy
        import numpy as np

        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]
        if self.ignore_pad_token_for_loss:
            # Replace -100 in the labels as we can't decode them.
            labels = np.where(
                labels != -100, labels, self.module.tokenizer.pad_token_id
            )
        decoded_preds = self.module.decode(preds)
        decoded_labels = self.module.decode(labels)
        result = self.metric.compute(
            predictions=decoded_preds, references=decoded_labels, use_stemmer=True
        )
        result = {k: round(v * 100, 4) for k, v in result.items()}
        prediction_lens = [
            np.count_nonzero(pred != self.module.tokenizer.pad_token_id)
            for pred in preds
        ]
        result["gen_len"] = np.mean(prediction_lens)
        return result

    @modal.method()
    def test(self):
        import os

        os.system("cd datasets && pytest")


@stub.local_entrypoint()
def main():
    FineTune().run.call()


if __name__ == "__main__":
    main()
