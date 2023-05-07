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
    )
    .dockerfile_commands(docker_command, force_build=False),
)
SHARED_ROOT = "/root/model_cache"


@stub.cls(
    gpu=modal.gpu.A10G(count=1),
    # cloud="gcp",
    cpu=14,
    shared_volumes={SHARED_ROOT: modal.SharedVolume.from_name(Config.shared_vol)},
    retries=0,
    secret=modal.Secret.from_name("huggingface-secret"),
    interactive=False,
    timeout=8000,
)
class FineTune:
    def __enter__(self):
        import datetime
        from pathlib import Path
        import evaluate
        from transformers import VisionEncoderDecoderModel, AutoTokenizer

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
        # self.tb_writer = SummaryWriter(self.output_dir)
        # self.tb_writer.add_hparams(Config.train_args)

    @modal.method()
    def run(self):
        from pathlib import Path
        from transformers import default_data_collator
        import tensorboardX
        import torch
        from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments
        from config import Config

        print("start!")

        training_args = Seq2SeqTrainingArguments(
            output_dir=str(self.output_dir),
            logging_dir=str(self.logging_dir),
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
                cache_root=Path(SHARED_ROOT),
                seed=Config.seed,
            ),
            eval_dataset=RedCapsDataset(
                split="val",
                dataset_path=Config.dataset_path,
                image_processor_pretrained=Config.image_processor_pretrained,
                tokenizer=self.tokenizer,
                max_length=Config.max_target_length,
                cache_root=Path(SHARED_ROOT),
                seed=Config.seed,
            ),
            data_collator=default_data_collator,
        )
        self.trainer.add_callback(
            ImageCaptionTensorBoardCallback(
                tensorboardX.SummaryWriter(self.logging_dir),
                torch.utils.data.DataLoader(
                    RedCapsDataset(
                        split="val",
                        dataset_path=Config.dataset_path,
                        image_processor_pretrained=Config.image_processor_pretrained,
                        tokenizer=self.tokenizer,
                        max_length=Config.max_target_length,
                        cache_root=Path(SHARED_ROOT),
                        seed=Config.seed,
                        use_image=True,
                    ),
                    batch_size=Config.log_batch_size,
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
            gen_kwargs = {"max_length": 128, "num_beams": 4}
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
        n_cols = 5

        n_rows = n_images // n_cols
        n_rows += 1 if n_images % n_cols != 0 else 0
        fig, axs = plt.subplots(n_rows, n_cols, figsize=(20, 4 * n_rows))
        fig.subplots_adjust(wspace=0.2, hspace=0.4)
        fig.subplots_adjust(wspace=0.5)

        if n_images < 5:
            for i in range(output_ids.shape[0]):
                caption = ""
                for j in range(0, len(decoded_preds[i]), 20):
                    caption += decoded_preds[i][j : j + 20] + "\n"

                axs[i].imshow(np.asarray(input["image"][i]))
                axs[i].axis("off")
                axs[i].set_title(caption, fontsize=12)
        else:
            for i in range(output_ids.shape[0]):
                row, col = i // n_cols, i % n_cols
                caption = ""
                for j in range(0, len(decoded_preds[i]), 20):
                    caption += decoded_preds[i][j : j + 20] + "\n"

                axs[row, col].imshow(np.asarray(input["image"][i]))
                axs[row, col].axis("off")
                axs[row, col].set_title(caption, fontsize=12)
        self.tb_writer.add_figure(
            tag="generated caption", figure=fig, global_step=state.global_step
        )


@stub.local_entrypoint()
def main():
    FineTune().run.call()
