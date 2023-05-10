""" Fine-tuning the model for image captioning. 

MIT License
Copyright (c) 2023 yuukicammy
"""
import modal
import transformers
from transformers import (
    TrainingArguments,
    TrainerState,
    TrainerControl,
)

from model_training.coco_dataset import CocoDataset
from model_training.redcaps_dataset import RedCapsDataset
from model_training.config import Config

docker_command = [
    "RUN apt-get update && apt-get install -y git",
    # For using forked datasets
    "RUN git clone --branch add-fn-kwargs-to-iterable-map-and-filter https://github.com/yuukicammy/datasets.git",
    'RUN cd datasets && pip install -e ".[dev]"',
    # For Git Large File Storage
    "RUN curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | bash",
    "RUN apt-get install git-lfs && git lfs install",
    # For COCO dataset
    "RUN git clone https://github.com/cocodataset/cocoapi && cd cocoapi/PythonAPI && make",
]

stub = modal.Stub(
    Config.project_name + "-train",
    image=modal.Image.debian_slim()
    .env({"TOKENIZERS_PARALLELISM": "false"})
    .pip_install(
        "Pillow",
        "tensorboardX",
        "transformers",
        "torchvision",
        "evaluate",
        "numpy",
        "nltk",
        "jaxlib",
        "torch",
        "kornia",
        "matplotlib",
        "rouge_score",
        "cython",
        "pycocotools",
    )
    .dockerfile_commands(docker_command, force_build=False),
)
SHARED_ROOT = "/root/model_cache"


@stub.cls(
    gpu=modal.gpu.A10G(count=1),
    # cloud="gcp",
    cpu=8,
    shared_volumes={SHARED_ROOT: modal.SharedVolume.from_name(Config.shared_vol)},
    retries=0,
    secret=modal.Secret.from_name("huggingface-secret"),
    interactive=False,
    timeout=3600,
)
class FineTune:
    def __enter__(self):
        import os
        import datetime
        from pathlib import Path
        import evaluate
        from transformers import (
            VisionEncoderDecoderModel,
            AutoTokenizer,
        )

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
        self.output_dir = Path(SHARED_ROOT) / Config.output_dir
        now = datetime.datetime.now()
        formatted_date = now.strftime("%b%d_%H-%M-%S")
        self.logging_dir = self.output_dir / "runs" / f"{formatted_date}_modal"

        if os.path.exists(Path(SHARED_ROOT) / Config.dataset_path):
            self.dataset_path = Path(SHARED_ROOT) / Config.dataset_path
        else:
            self.dataset_path = Config.dataset_path

    @modal.method()
    def run(self):
        import os
        from pathlib import Path
        from transformers import default_data_collator
        import tensorboardX
        import torch
        from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments

        print("start!")

        if Config.dataset_path == "coco":
            train_dataset = CocoDataset(
                split="train",
                dataset_path=self.dataset_path,
                image_processor_pretrained=Config.image_processor_pretrained,
                tokenizer=self.tokenizer,
                max_length=Config.max_target_length,
                seed=Config.seed,
            )
            val_dataset = CocoDataset(
                split="val",
                dataset_path=self.dataset_path,
                image_processor_pretrained=Config.image_processor_pretrained,
                tokenizer=self.tokenizer,
                max_length=Config.max_target_length,
                seed=Config.seed,
                num_examples=Config.val_examples,
            )
            log_dataset = CocoDataset(
                split="val",
                dataset_path=self.dataset_path,
                image_processor_pretrained=Config.image_processor_pretrained,
                tokenizer=self.tokenizer,
                max_length=Config.max_target_length,
                seed=Config.seed,
                use_input=True,
                num_examples=Config.val_examples,
            )
        else:
            train_dataset = RedCapsDataset(
                split="train",
                dataset_path=self.dataset_path,
                image_processor_pretrained=Config.image_processor_pretrained,
                tokenizer=self.tokenizer,
                max_length=Config.max_target_length,
                cache_root=Path(SHARED_ROOT) / ".hf_cache",
                seed=Config.seed,
            )
            val_dataset = RedCapsDataset(
                split="val",
                dataset_path=self.dataset_path,
                image_processor_pretrained=Config.image_processor_pretrained,
                tokenizer=self.tokenizer,
                max_length=Config.max_target_length,
                cache_root=Path(SHARED_ROOT) / ".hf_cache",
                seed=Config.seed,
            )
            log_dataset = RedCapsDataset(
                split="val",
                dataset_path=self.dataset_path,
                image_processor_pretrained=Config.image_processor_pretrained,
                tokenizer=self.tokenizer,
                max_length=Config.max_target_length,
                cache_root=Path(SHARED_ROOT) / ".hf_cache",
                seed=Config.seed,
                use_input=True,
            )
        training_args = Seq2SeqTrainingArguments(
            output_dir=str(self.output_dir),
            logging_dir=str(self.logging_dir),
            hub_token=os.environ["HUGGINGFACE_TOKEN"],
            **Config.train_args,
        )
        # instantiate trainer
        self.trainer = Seq2SeqTrainer(
            model=self.model,
            tokenizer=self.tokenizer,
            args=training_args,
            compute_metrics=self.compute_metrics,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            data_collator=default_data_collator,
        )
        self.trainer.add_callback(
            ImageCaptionTensorBoardCallback(
                tensorboardX.SummaryWriter(self.logging_dir),
                torch.utils.data.DataLoader(
                    log_dataset,
                    batch_size=Config.log_examples,
                    shuffle=True,
                ),
            )
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

        if self.ignore_pad_token_for_loss:
            # Replace -100 in the labels as we can't decode them.
            labels = np.where(labels != -100, labels, self.tokenizer.pad_token_id)
            preds = np.where(preds != -100, preds, self.tokenizer.pad_token_id)

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
    def __init__(self, tf_writer, eval_dataloader=None):
        super().__init__(tf_writer)
        self.eval_dataloader = eval_dataloader

    def on_train_begin(self, args, state, control, **kwargs):
        super().on_train_begin(args, state, control, **kwargs)
        if "model" in kwargs and "tokenizer" in kwargs:
            self.logging_impl(
                args=args,
                state=state,
                control=control,
                model=kwargs["model"],
                tokenizer=kwargs["tokenizer"],
            )

    def on_log(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        model,
        tokenizer,
        logs=None,
        **kwargs,
    ):
        super().on_log(args=args, state=state, control=control, logs=logs, **kwargs)
        self.logging_impl(
            args=args,
            state=state,
            control=control,
            model=model,
            tokenizer=tokenizer,
            **kwargs,
        )

    def logging_impl(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        model,
        tokenizer,
        **kwargs,
    ):
        import torch
        import matplotlib.pyplot as plt
        import numpy as np

        with torch.no_grad():
            input = next(iter(self.eval_dataloader))
            gen_kwargs = {
                "max_length": Config.max_target_length,
                "num_beams": Config.num_beams,
            }
            output_ids = model.generate(input["pixel_values"].to("cuda"), **gen_kwargs)
            # print("Done forwarding.")
            # print(output_ids.shape)
        decoded_preds = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
        decoded_preds = [decoded_pred.strip() for decoded_pred in decoded_preds]
        # print(f"output_ids shape: {output_ids.shape}")
        # print(f"pixel_values shape: {input['pixel_values'].shape}")
        # print(f"decoded_preds len: {len(decoded_preds)}")
        # self.tb_writer.add_embedding(
        #     mat=output_ids.to("cpu"),
        #     metadata=decoded_preds,
        #     label_img=input["pixel_values"],
        #     # global_step=state.global_step,
        #     tag="embedding",
        # )
        n_images = output_ids.shape[0]
        n_cols = 10

        n_rows = (n_images // n_cols) * 2
        n_rows += 2 if n_images % n_cols != 0 else 0
        fig, axs = plt.subplots(n_rows, n_cols, figsize=(20, 2 * n_rows))
        fig.subplots_adjust(wspace=0.5, hspace=0)

        for i in range(output_ids.shape[0]):
            row, col = (i // n_cols) * 2, i % n_cols
            caption = insert_newlines(decoded_preds[i])
            ann_caption = "[annotation]\n" + insert_newlines(input["ann_caption"][i])
            axs[row, col].imshow(np.asarray(input["image"][i]))
            axs[row, col].axis("off")
            axs[row, col].set_title(caption, fontsize=10)
            axs[row + 1, col].text(0, 0, ann_caption, fontsize=10, color="red")
            axs[row + 1, col].axis("off")
        self.tb_writer.add_figure(
            tag="generated caption", figure=fig, global_step=state.global_step
        )


def insert_newlines(in_caption: str) -> str:
    vocabs = in_caption.split()
    n_line = 0
    out_caption = ""
    for vocab in vocabs:
        if 20 < n_line + len(vocab):
            out_caption += "\n" + vocab + " "
            n_line = len(vocab) + 1
        else:
            out_caption += vocab + " "
            n_line += len(vocab) + 1
    return out_caption


@stub.local_entrypoint()
def main():
    FineTune().run.call()
