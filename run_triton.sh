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
