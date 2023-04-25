from pathlib import Path

import fastapi
import fastapi.staticfiles

from modal import Function, Mount, Stub, asgi_app

stub = Stub("vit-gpt2-image-caption-webapp")
web_app = fastapi.FastAPI()


@web_app.post("/parse")
async def parse(request: fastapi.Request):
    predict_step = Function.lookup("vit-gpt2-image-captioning", "predict_step")

    form = await request.form()
    image = await form["image"].read()  # type: ignore
    call = predict_step.spawn(image)
    return {"call_id": call.object_id}


@web_app.get("/result/{call_id}")
async def poll_results(call_id: str):
    from modal.functions import FunctionCall

    function_call = FunctionCall.from_id(call_id)
    try:
        result = function_call.get(timeout=0)
    except TimeoutError:
        return fastapi.responses.JSONResponse(content="", status_code=202)

    return result[0]


assets_path = Path(__file__).parent / "frontend"


@stub.function(mounts=[Mount.from_local_dir(assets_path, remote_path="/assets")])
@asgi_app()
def wrapper():
    web_app.mount("/", fastapi.staticfiles.StaticFiles(directory="/assets", html=True))

    return web_app
