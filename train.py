from config import config_dict as cnfg
import utils
from encoder_decoder import EnocoderDecoder
import modal

docker_command = [
    "RUN apt-get update && apt-get install -y git",
    #    "RUN pip uninstall datasets",
    "RUN git clone --branch add-fn_kwargs-to-IterableDatasetDict https://github.com/yuukicammy/datasets.git",
    'RUN cd datasets && pip install -e ".[dev]"',
]

stub = modal.Stub(
    cnfg["project_name"] + "-train",
    image=modal.Image.debian_slim()
    .pip_install(
        "Pillow",
        "tensorboard",
        "transformers",
        "evaluate",
        "numpy",
        "nltk",
        "rouge_score",
    )
    .dockerfile_commands(docker_command),
)
SHARED_ROOT = "/root/model_cache"


@stub.cls(
    gpu="any",
    cpu=14,
    shared_volumes={SHARED_ROOT: modal.SharedVolume.from_name(cnfg["shared_vol"])},
    retries=0,
    secret=modal.Secret.from_name("huggingface-secret"),
    timeout=86400,
)
class FineTune:
    def __enter__(self):
        import evaluate

        self.encoder_decoder = EnocoderDecoder()
        self.metric = evaluate.load(cnfg["metric"])
        self.ignore_pad_token_for_loss = cnfg["ignore_pad_token_for_loss"]

    @modal.method()
    def run(self):
        import os
        import time
        from pathlib import Path
        from datasets import load_dataset
        from transformers import default_data_collator
        from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments

        os.system("pwd")

        dataset = load_dataset(
            cnfg["dataset_path"],
            streaming=True,
            use_auth_token=os.environ["HUGGINGFACE_TOKEN_READ"],
            cache_dir=SHARED_ROOT,
        )

        dataset = dataset.map(
            function=utils.fetch_images,
            fn_kwargs={"retries": 5},
        )
        dataset = dataset.filter(lambda x: x["image"] is not None)  # remove None
        dataset = dataset.map(function=self.encoder_decoder.preprocess)

        print(next(iter(dataset["train"])))

        output_dir = Path(cnfg["log_dir"]) / f"train-{time.strftime('%Y%m%d-%H%M%S')}"
        training_args = Seq2SeqTrainingArguments(
            predict_with_generate=True,  # Whether to use generate to calculate generative metrics (ROUGE, BLEU).
            evaluation_strategy=cnfg["evaluation_strategy"],  # ["no", "epoch", "step"]
            per_device_train_batch_size=cnfg["batch_size_per_device"]["train"],
            per_device_eval_batch_size=cnfg["batch_size_per_device"]["val"],
            output_dir=output_dir,
            eval_steps=cnfg["steps"]["val"],
            logging_steps=cnfg["steps"]["log"],
            learning_rate=cnfg["learaning_rate"],
            seed=cnfg["seed"],
            save_total_limit=cnfg["save_limits"],
            max_steps=cnfg["max_steps"],
        )
        # instantiate trainer
        trainer = Seq2SeqTrainer(
            model=self.encoder_decoder.model,
            tokenizer=self.encoder_decoder.feature_extractor,
            args=training_args,
            compute_metrics=self.compute_metrics,
            train_dataset=dataset["train"],
            eval_dataset=dataset["val"],
            data_collator=default_data_collator,
        )
        # trainer.train()

    def compute_metrics(self, eval_preds):
        import numpy as np

        preds, labels = eval_preds
        if self.ignore_pad_token_for_loss:
            # Replace -100 in the labels as we can't decode them.
            labels = np.where(
                labels != -100, labels, self.encoder_decoder.tokenizer.pad_token_id
            )
        decoded_preds = self.encoder_decoder.decode(preds)
        decoded_labels = self.encoder_decoder.decode(labels)
        result = self.metric.compute(
            predictions=decoded_preds, references=decoded_labels, use_stemmer=True
        )
        result = {k: round(v * 100, 4) for k, v in result.items()}
        prediction_lens = [
            np.count_nonzero(pred != self.encoder_decoder.tokenizer.pad_token_id)
            for pred in preds
        ]
        result["gen_len"] = np.mean(prediction_lens)
        return result

    @modal.method()
    def test(self):
        import os

        os.system("pwd")


@stub.local_entrypoint()
def main():
    FineTune().run.call()
