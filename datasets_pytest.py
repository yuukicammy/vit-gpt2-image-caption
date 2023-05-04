import modal

stub = modal.Stub("datasets-pytest")


@stub.function(
    gpu="any",
    cpu=14,
    image=modal.Image.debian_slim()
    .pip_install("jiwer")
    .dockerfile_commands(
        [
            "RUN apt-get update && apt-get install -y git",
            # "RUN pip uninstall datasets",
            "RUN git clone --branch add-fn-kwargs-to-iterable-map-and-filter https://github.com/yuukicammy/datasets.git",
            'RUN cd datasets && pip install -e ".[dev]" && pytest tests/test_dataset_dict.py tests/test_iterable_dataset.py',
        ],
        force_build=False,
    ),
    timeout=3000,
)
def test():
    import os

    os.system("pwd")
    os.system("ls")


@stub.local_entrypoint()
def main():
    test.call()
