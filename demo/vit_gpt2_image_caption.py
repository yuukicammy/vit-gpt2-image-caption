""" A microservice components for demonstration of image captioning runnning on Modal

MIT License
Copyright (c) 2023 yuukicammy
"""
from typing import Union
import modal

stub = modal.Stub("vit-gpt2-image-caption")
volume = modal.SharedVolume.from_name("image-caption-vol")

docker_command = [
    "RUN apt-get update && apt-get install -y git",
    # For Git Large File Storage
    "RUN curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | bash",
    "RUN apt-get install git-lfs && git lfs install",
]


@stub.function(
    # gpu="any",
    shared_volumes={"/root/model_cache": volume},
    image=modal.Image.debian_slim()
    .pip_install("Pillow", "transformers", "torch")
    .dockerfile_commands(docker_command),
    secret=modal.Secret.from_name("huggingface-secret"),
    retries=0,
)
def predict(
    image: bytes, max_length: Union[str, int] = 64, num_beams: Union[str, int] = 4
) -> str:
    """Generate a caption for an input image using pretrained VisionEncoderDecoderModel and ViTImageProcessor.

    Args:
        image (bytes): Input image in bytes format.
        max_length (int, optional): Maximum length of the generated caption. Defaults to 64.
        num_beams (int, optional): Number of beams to use for beam search. Defaults to 4.

    Returns:
        str: A generated caption for the input image.
    """
    import io
    from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
    import torch
    from PIL import Image

    latest_checkpoint = search_latest_checkpoint("/root/model_cache/image-caption")
    if latest_checkpoint is None:
        print(
            f"Pretrained model does not exist in /image-caption for the shared volume: `image-caption-vol` ."
        )
        return ["Error! Cannot find model to generate caption."]

    print(f"latest_checkpoint: {latest_checkpoint}")

    model = VisionEncoderDecoderModel.from_pretrained(
        pretrained_model_name_or_path="/root/model_cache/image-caption/"
        + latest_checkpoint,
    )
    feature_extractor = ViTImageProcessor.from_pretrained(
        pretrained_model_name_or_path="nlpconnect/vit-gpt2-image-captioning",
        cache_dir="/root/model_cache/.hf_cache",
    )
    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path="/root/model_cache/image-caption/"
        + latest_checkpoint,
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


def search_latest_checkpoint(target_dir):
    import os

    latest_dir = None
    latest_time = 0
    for dirname in os.listdir(target_dir):
        if dirname.startswith("checkpoint-"):
            fullpath = os.path.join(target_dir, dirname)
            if os.path.isdir(fullpath):
                mtime = os.stat(fullpath).st_mtime
                if mtime > latest_time:
                    latest_dir = dirname
                    latest_time = mtime
    return latest_dir


@stub.local_entrypoint()
def main():
    import requests
    from pathlib import Path

    image_filepath = Path(__file__).parent / "sample.png"
    if image_filepath.exists():
        with open(image_filepath, "rb") as f:
            image = f.read()
    else:
        try:
            image = requests.get(
                "https://drive.google.com/uc?id=0B0TjveMhQDhgLTlpOENiOTZ6Y00&export=download",
                # "https://drive.google.com/uc?id=1M5Z0LqieL5Jem_WyjlQ0-HnG048WYMk5&export=download",
                timeout=360,
            ).content
        except Exception as e:
            print(e)
    print(predict.call(image)[0])
