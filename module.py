# from pytorch_lightning import LightningModule
import torch


class CaptionTransformer:
    def __init__(
        self,
        tokenizer_pretrained_path: str,
        model_pretrained_path: str,
        max_target_length: int,
        ignore_pad_token_for_loss: bool,
    ):
        from transformers import AutoTokenizer, VisionEncoderDecoderModel
        import torch

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_pretrained_path)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.max_target_length = max_target_length
        self.model = VisionEncoderDecoderModel.from_pretrained(model_pretrained_path)
        # update the model config
        self.model.config.eos_token_id = self.tokenizer.eos_token_id
        self.model.config.decoder_start_token_id = self.tokenizer.bos_token_id
        self.model.config.pad_token_id = self.tokenizer.pad_token_id

        self.loss_fn = torch.nn.CrossEntropyLoss(
            ignore_index=self.tokenizer.pad_token_id
        )

        self.gen_kwargs = {"max_length": self.max_target_length}
        self.ignore_pad_token_for_loss = ignore_pad_token_for_loss

    def decode(self, preds):
        decoded_preds = self.tokenizer.batch_decode(preds, skip_special_tokens=True)
        # simple post-processing
        decoded_preds = postprocess_text(decoded_preds)
        return decoded_preds

    def compute_loss(self, inputs, labels):
        if self.ignore_pad_token_for_loss:
            logits = self.model(**inputs, use_cache=False)[0]
            loss = self.loss_fn(logits.view(-1, logits.shape[-1]), labels.view(-1))
        else:
            # compute usual loss via models
            loss, logits = self.model(**inputs, labels=labels, use_cache=False)[:2]
        return loss, logits

    def training_step(self, batch, batch_idx) -> torch.Tensor:
        inputs = {"pixel_values": batch["pixel_values"]}
        labels = self.tokenizer(
            batch["caption"], padding="max_length", max_length=self.max_target_length
        ).input_ids
        loss, _ = self.compute_loss(inputs, labels)
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        inputs = {"pixel_values": batch["pixel_values"]}
        labels = self.tokenizer(
            batch["caption"], padding="max_length", max_length=self.max_target_length
        ).input_ids
        loss, logits = self.compute_loss(inputs, labels)
        preds = logits.squeeze()
        return {"loss": loss, "preds": preds, "labels": labels}

    def configure_optimizers(self):
        return


def postprocess_text(preds):
    # import nltk

    preds = [pred.strip() for pred in preds]
    # rougeLSum expects newline after each sentence
    # preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
    return preds
