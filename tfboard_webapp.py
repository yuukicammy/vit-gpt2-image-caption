import modal
from config import Config

SHARED_ROOT: str = "/root/model_cache"
ONLY_LATEST_DIR: bool = True

stub = modal.Stub(Config.project_name + "-tfboard-webapp")


@stub.function(
    image=modal.Image.debian_slim().pip_install("tensorboardX", "tensorboard"),
    shared_volumes={SHARED_ROOT: modal.SharedVolume.from_name(Config.shared_vol)},
)
@modal.wsgi_app()
def tensorboard_app():
    import os
    import tensorboard
    from pathlib import Path

    tfboard_log_root = Path(SHARED_ROOT) / Config.output_dir / "runs"
    tfboard_log_dir = tfboard_log_root
    if ONLY_LATEST_DIR:
        tfboard_log_dir = tfboard_log_root / search_latest_dir(tfboard_log_root)

    board = tensorboard.program.TensorBoard()
    board.configure(logdir=str(tfboard_log_dir))
    (data_provider, deprecated_multiplexer) = board._make_data_provider()
    wsgi_app = tensorboard.backend.application.TensorBoardWSGIApp(
        board.flags,
        board.plugin_loaders,
        data_provider,
        board.assets_zip_provider,
        deprecated_multiplexer,
    )
    return wsgi_app


def search_latest_dir(target_dir):
    import os

    os.system("pwd")
    latest_dir = None
    latest_time = 0
    for dirname in os.listdir(target_dir):
        fullpath = os.path.join(target_dir, dirname)
        if os.path.isdir(fullpath):
            ctime = os.path.getctime(fullpath)
            if ctime > latest_time:
                latest_dir = dirname
                latest_time = ctime
    return latest_dir


# @stub.local_entrypoint()
# def main():
#     tensorboard_app().call()
