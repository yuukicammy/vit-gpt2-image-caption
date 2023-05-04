import modal
import transformers
from transformers import (
    TrainingArguments,
    TrainerState,
    TrainerControl,
)

from redcaps_dataset import RedCapsDataset
from config import Config

docker_command = [
    "RUN apt-get update && apt-get install -y git",
    #    "RUN pip uninstall datasets",
    # For using forked datasets
    "RUN git clone --branch add-fn-kwargs-to-iterable-map-and-filter https://github.com/yuukicammy/datasets.git",
    'RUN cd datasets && pip install -e ".[dev]"',
    # For Git Large File Storage
    "RUN curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | bash",
    "RUN apt-get install git-lfs && git lfs install",
]

stub = modal.Stub(
    Config.project_name + "-train",
    image=modal.Image.debian_slim()
    .pip_install(
        "Pillow",
        "tensorboard",
        "transformers",
        "torchvision",
        "evaluate",
        "numpy",
        "nltk",
        "jaxlib",
        "torch",
        "kornia",
        "rouge_score",
    )
    .dockerfile_commands(docker_command, force_build=False),
)
SHARED_ROOT = "/root/model_cache"


@stub.cls(
    gpu=modal.gpu.A10G(count=4),
    #    cloud="gcp",
    cpu=10,
    shared_volumes={SHARED_ROOT: modal.SharedVolume.from_name(Config.shared_vol)},
    retries=0,
    secret=modal.Secret.from_name("huggingface-secret"),
    interactive=False,
    timeout=8000,
)
class FineTune:
    def __enter__(self):
        from pathlib import Path
        import evaluate
        from transformers import VisionEncoderDecoderModel, AutoTokenizer
        from torch.utils.tensorboard import SummaryWriter

        self.metric = evaluate.load(Config.metric)
        self.ignore_pad_token_for_loss = Config.ignore_pad_token_for_loss

        self.tokenizer = AutoTokenizer.from_pretrained(Config.tokenizer_pretrained)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = VisionEncoderDecoderModel.from_pretrained(
            Config.encoder_decoder_pretrained
        )
        # update the model config
        self.model.config.eos_token_id = self.tokenizer.eos_token_id
        self.model.config.decoder_start_token_id = self.tokenizer.bos_token_id
        self.model.config.pad_token_id = self.tokenizer.pad_token_id

        self.output_dir = Path(SHARED_ROOT) / Config.log_dir
        # self.tb_writer = SummaryWriter(self.output_dir)
        # self.tb_writer.add_hparams(Config.train_args)

    @modal.method()
    def run(self):
        from transformers import default_data_collator
        from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments
        from config import Config

        print("start!")

        training_args = Seq2SeqTrainingArguments(
            output_dir=self.output_dir,
            # hub_token=os.environ["HUGGINGFACE_TOKEN"],
            **Config.train_args,
        )
        # instantiate trainer
        self.trainer = Seq2SeqTrainer(
            model=self.model,
            tokenizer=self.tokenizer,
            args=training_args,
            compute_metrics=self.compute_metrics,
            train_dataset=RedCapsDataset(
                split="train",
                dataset_path=Config.dataset_path,
                image_processor_pretrained=Config.image_processor_pretrained,
                tokenizer=self.tokenizer,
                max_length=Config.max_target_length,
                cache_root=SHARED_ROOT,
                download_retries=Config.download_retries,
                seed=Config.seed,
            ),
            eval_dataset=RedCapsDataset(
                split="val",
                dataset_path=Config.dataset_path,
                image_processor_pretrained=Config.image_processor_pretrained,
                tokenizer=self.tokenizer,
                max_length=Config.max_target_length,
                cache_root=SHARED_ROOT,
                download_retries=Config.download_retries,
                seed=Config.seed,
                num_examples=320,
            ),
            data_collator=default_data_collator,
            callbacks=[ImageCaptionTensorBoardCallback],
        )
        # breakpoint()
        self.trainer.train()

    def decode(self, preds):
        decoded_preds = self.tokenizer.batch_decode(preds, skip_special_tokens=True)
        # simple post-processing
        decoded_preds = [decoded_pred.strip() for decoded_pred in decoded_preds]
        return decoded_preds

    def compute_metrics(self, eval_preds):
        # receive data as numpy
        import numpy as np

        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]
        if self.ignore_pad_token_for_loss:
            # Replace -100 in the labels as we can't decode them.
            labels = np.where(labels != -100, labels, self.tokenizer.pad_token_id)
        decoded_preds = self.decode(preds)
        decoded_labels = self.decode(labels)
        result = self.metric.compute(
            predictions=decoded_preds, references=decoded_labels, use_stemmer=True
        )
        result = {k: round(v * 100, 4) for k, v in result.items()}
        prediction_lens = [
            np.count_nonzero(pred != self.tokenizer.pad_token_id) for pred in preds
        ]
        result["gen_len"] = np.mean(prediction_lens)
        return result


class ImageCaptionTensorBoardCallback(transformers.integrations.TensorBoardCallback):
    def on_log(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        model,
        tokenizer,
        eval_dataloader,
    ):
        import torch

        with torch.no_grad():
            input = next(iter(eval_dataloader))
            loss, logits, labels = model(**input)

        decoded_preds = tokenizer.batch_decode(logits, skip_special_tokens=True)
        decoded_preds = [decoded_pred.strip() for decoded_pred in decoded_preds]
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        decoded_labels = [decoded_label.strip() for decoded_label in decoded_labels]

        self.tb_writer.add_embedding(
            mat=logits,
            metadata={"gt": decoded_preds, "pred": decoded_labels},
            label_img=input["pixel_values"],
            global_step=state.global_step,
            tag="embedding",
        )


@stub.local_entrypoint()
def main():
    FineTune().run.call()
