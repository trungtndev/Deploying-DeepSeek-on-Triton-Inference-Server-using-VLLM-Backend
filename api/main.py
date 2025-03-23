from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from .client import LLMClient

client = LLMClient(
    model="deepseek",
    url="localhost:8001",
    verbose=False,
    stream_timeout=10.0,
    offset=0,
    streaming_mode=False,
    exclude_inputs_in_outputs=False,
)


class TextInput(BaseModel):
    text_input: str


app = FastAPI()


@app.get("/")
def root():
    return {"message": "Hello, World!"}


@app.post("/deepseek/generate/")
async def inference_endpoint(request: TextInput):
    sampling_parameters = {
        "temperature": "0.1",
        "top_p": "0.95",
        "max_tokens": "100",
    }
    success = await client.process_stream(
        [request.text_input],
        sampling_parameters,
    )

    if not success:
        raise HTTPException(status_code=500, detail="Inference processing failed")

    result = {
        k: "".join([
            r.decode("utf-8") if isinstance(r, bytes) else r for r in v
        ]).strip()
        for k, v in client._results_dict.items()
    }
    return result
