# Deploying DeepSeek on Triton Inference Server using VLLM Backend

## Ubuntu, GPU and CUDA Information
- Ubuntu 20.04
- CUDA 12.4
- GPU: NVIDIA GeForce RTX 3050Ti
## Pulling the Docker Image
```bash
docker pull nvcr.io/nvidia/tritonserver:24.07-vllm-python-py3
docker pull python:3.10.16-bookworm

```

## Configuring the model

You can use this config file to testing
```json
{
"model":"facebook/opt-125m",
"disable_log_requests": true,
"gpu_memory_utilization": 0.5,
"enforce_eager": true
}
```

This is the config file for the Qwen model, mainly used for this project.
You can change the config file depending on your hardware and model.
```json
{
"model":"Qwen/Qwen1.5-0.5B",
"max_model_len": 300,
"disable_log_requests": true,
"gpu_memory_utilization": 0.5,
"enforce_eager": true
}
```


## Running the Application
This step you have 2 options, you can run the application using the docker-compose or running the docker commands manually.

### Option 1: Running the application using docker-compose

Docker-compose file is used to run the Triton Inference Server and the FastAPI application.
```yaml
services:
  api:
    network_mode: "host"
    build:
      context: .
      dockerfile: Dockerfile
    command: fastapi run  api/main.py --host 0.0.0.0 --port 8888
    depends_on:
      - triton
    ports:
      - "8888:8888"
    volumes:
      - .:/app
    working_dir: /app

  triton:
    image: nvcr.io/nvidia/tritonserver:24.07-vllm-python-py3
    network_mode: "host"
    environment:
      - TORCH_USE_CUDA_DSA=1
      - CUDA_LAUNCH_BLOCKING=1
    ports:
      - "8000:8000"
      - "8001:8001"
      - "8002:8002"
    shm_size: "1g"
    ulimits:
      memlock: -1
      stack: 67108864
    volumes:
      - ${PWD}/models:/models
    working_dir: /workspace
    deploy:
      resources:
        reservations:
          devices:
            - capabilities: [ gpu ]
    command: [ "tritonserver", "--model-repository", "/models" ]

```

Start the application using the following command:
```bash
docker-compose up
```

### Option 2: Running the application using docker commands
1. Running the Triton Inference Server (with VLLM Backend)
```bash
bash run_triton.sh
```
Or full command:
```bash
docker run  -it \
            --gpus all \
            --net=host \
            --rm \
            -e TORCH_USE_CUDA_DSA=1 \
            -e CUDA_LAUNCH_BLOCKING=1 \
            -p 8000:8000 \
            -p 8001:8001 \
            -p 8002:8002 \
            --shm-size=1G \
            --ulimit memlock=-1 \
            --ulimit stack=67108864 \
            -v ${PWD}/models:/models \
            -w /work nvcr.io/nvidia/tritonserver:24.07-vllm-python-py3 \
            tritonserver \
            --model-repository /models
```
2. Running the FastAPI application

Install the required packages using the following command:
```bash
pip install -r requirements.txt
```

Start the FastAPI application using the following command:
```bash
fastapi run  api/main.py --host 0.0.0.0 --port 8888
```
### The output should be similar to this:
For the Triton Inference Server:
```bash
=============================
== Triton Inference Server ==
=============================

...

I0323 18:29:04.044553 1 server.cc:674] 
+----------+---------+--------+
| Model    | Version | Status |
+----------+---------+--------+
| deepseek | 1       | READY  |
+----------+---------+--------+

...

I0323 18:29:04.110731 1 grpc_server.cc:2463] "Started GRPCInferenceService at 0.0.0.0:8001"
I0323 18:29:04.111301 1 http_server.cc:4692] "Started HTTPService at 0.0.0.0:8000"
I0323 18:29:04.156651 1 http_server.cc:362] "Started Metrics Service at 0.0.0.0:8002"
W0323 18:29:05.096491 1 metrics.cc:631] "Unable to get power limit for GPU 0. Status:Success, value:0.000000"
W0323 18:29:06.099012 1 metrics.cc:631] "Unable to get power limit for GPU 0. Status:Success, value:0.000000"
W0323 18:29:07.099395 1 metrics.cc:631] "Unable to get power limit for GPU 0. Status:Success, value:0.000000"
```

## Create LLMClient

File name `client.py` is used to create a client for the FastAPI application. 
This client is used to send requests to triton inference server and get the results.
Define the client as follows:

```python
from client import LLMClient
client = LLMClient(
    ...
    )
sampling_parameters = {
        "temperature": "0.1",
        "top_p": "0.95",
        "max_tokens": "100",
    }
success = await client.process_stream(
        [str],
        sampling_parameters,
    )

## Check and get the result
print(success)
result = client._results_dict()
```

## Testing the Application
There are two ways to test the application:
- Sending a request to the Triton Inference Server
- Sending a request to the FastAPI application

### Sending a request to the Triton Inference Server
```bash
curl -X POST localhost:8000/v2/models/deepseek/generate \
        -d '{
        "text_input": "Hello, my name is", 
        "parameters": {
                "stream": false, 
                "temperature": 0
                }
        }'
```

### Sending a request to the FastAPI application
```bash
curl -X POST "http://localhost:8888/deepseek/generate/" \
     -H "Content-Type: application/json" \
     -d '{
     "text_input": "Hello, my name is"
     }'
```