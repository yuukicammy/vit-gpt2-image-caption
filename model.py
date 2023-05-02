import modal

from config import config_dict as cnfg

from encoder_decoder import ImageCaption
import numpy as np

stub = modal.Stub("tmp")


@stub.cls(
    gpu="any",
    cpu=14,
    image=modal.Image.debian_slim().pip_install(
        "transformers",
    ),
    # .dockerfile_commands(docker_command),
    retries=0,
    secret=modal.Secret.from_name("huggingface-secret"),
    timeout=86400,
)
class A:
    @modal.method()
    def method(self):
        import os

        os.system("pwd")


@stub.local_entrypoint()
def main():
    A().method.call()
