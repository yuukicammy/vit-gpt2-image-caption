# https://huggingface.co/nlpconnect/vit-gpt2-image-captioning

import urllib.request
import modal

stub = modal.Stub("vit-gpt2-image-caption")
volume = modal.SharedVolume.from_name("red-caps-vol")


@stub.function(
    gpu="any",
    image=modal.Image.debian_slim().pip_install("Pillow", "transformers", "torch"),
    shared_volumes={"/root/model_cache": volume},
    secret=modal.Secret.from_name("huggingface-secret"),
    retries=3,
)
def predict(image, max_length=64, num_beams=4):
    import io
    import os
    from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
    import torch
    from PIL import Image

    model = VisionEncoderDecoderModel.from_pretrained(
        pretrained_model_name_or_path="/root/model_cache/image-caption"
        # pretrained_model_name_or_path="yuukicammy/image-caption",
        # use_auth_token=os.environ["HUGGINGFACE_TOKEN"],
    )
    feature_extractor = ViTImageProcessor.from_pretrained(
        pretrained_model_name_or_path="nlpconnect/vit-gpt2-image-captioning"
        # pretrained_model_name_or_path="yuukicammy/image-caption",
        # use_auth_token=os.environ["HUGGINGFACE_TOKEN_READ"],
    )
    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path="/root/model_cache/image-caption"
        # pretrained_model_name_or_path="yuukicammy/image-caption",
        # use_auth_token=os.environ["HUGGINGFACE_TOKEN"],
    )

    if not isinstance(max_length, int):
        max_length = int(max_length)

    if not isinstance(num_beams, int):
        num_beams = int(num_beams)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    gen_kwargs = {"max_length": max_length, "num_beams": num_beams}
    input_img = Image.open(io.BytesIO(image))
    pixel_values = feature_extractor(
        images=[input_img], return_tensors="pt"
    ).pixel_values
    pixel_values = pixel_values.to(device)

    output_ids = model.generate(pixel_values, **gen_kwargs)

    preds = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
    preds = [pred.strip() for pred in preds]
    return preds


@stub.local_entrypoint()
def main():
    from pathlib import Path

    image_filepath = Path(__file__).parent / "sample.png"
    if image_filepath.exists():
        with open(image_filepath, "rb") as f:
            image = f.read()
    else:
        try:
            image = urllib.request.urlopen(
                "https://drive.google.com/uc?id=0B0TjveMhQDhgLTlpOENiOTZ6Y00&export=download"
            ).read()
        except urllib.error.URLError as e:
            print(e.reason)
    print(predict.call(image)[0])
