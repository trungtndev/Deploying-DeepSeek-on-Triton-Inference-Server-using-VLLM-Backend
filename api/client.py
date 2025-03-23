#!/usr/bin/env python3

# Copyright 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import argparse
import asyncio
import json
import sys

import numpy as np
import tritonclient.grpc.aio as grpcclient
from tritonclient.utils import *


class LLMClient:
    def __init__(self,
                 model="vllm_model",
                 url="localhost:8001",
                 verbose=False,
                 stream_timeout=None,
                 offset=0,
                 input_prompts="prompts.txt",
                 results_file="results.txt",
                 iterations=1,
                 streaming_mode=False,
                 exclude_inputs_in_outputs=False,
                 lora_name=None
                 ):
        self.model = model
        self.url = url
        self.verbose = verbose
        self.stream_timeout = stream_timeout
        self.offset = offset
        self.input_prompts = input_prompts
        self.results_file = results_file
        self.iterations = iterations
        self.streaming_mode = streaming_mode
        self.exclude_inputs_in_outputs = exclude_inputs_in_outputs
        self.lora_name = lora_name
        self._results_dict = {}

    def get_triton_client(self):
        try:
            triton_client = grpcclient.InferenceServerClient(
                url=self.url,
                verbose=self.verbose,
            )
        except Exception as e:
            print("channel creation failed: " + str(e))
            sys.exit(1)
        return triton_client

    async def async_request_iterator(self, prompts, sampling_parameters):
        try:
            for iter in range(self.iterations):
                for i, prompt in enumerate(prompts):
                    prompt_id = self.offset + (len(prompts) * iter) + i
                    self._results_dict[str(prompt_id)] = []
                    yield self.create_request(
                        prompt,
                        self.streaming_mode,
                        prompt_id,
                        sampling_parameters,
                        self.exclude_inputs_in_outputs,
                    )
        except Exception as error:
            print(f"Caught an error in the request iterator: {error}")

    async def stream_infer(self, prompts, sampling_parameters):
        try:
            triton_client = self.get_triton_client()
            # Start streaming
            response_iterator = triton_client.stream_infer(
                inputs_iterator=self.async_request_iterator(prompts, sampling_parameters),
                stream_timeout=self.stream_timeout,
            )
            async for response in response_iterator:
                yield response
        except InferenceServerException as error:
            print(error)
            sys.exit(1)

    async def process_stream(self, prompts, sampling_parameters):
        # Clear results in between process_stream calls
        self._results_dict = {}
        success = True
        # Read response from the stream
        async for response in self.stream_infer(prompts, sampling_parameters):
            result, error = response
            if error:
                print(f"Encountered error while processing: {error}")
                success = False
            else:
                output = result.as_numpy("text_output")
                for i in output:
                    self._results_dict[result.get_response().id].append(i)
        return success

    async def run(self):
        # Sampling parameters cho text generation, ví dụ:
        sampling_parameters = {
            "temperature": "0.1",
            "top_p": "0.95",
            "max_tokens": "100",
        }
        if self.lora_name is not None:
            sampling_parameters["lora_name"] = self.lora_name

        # Đọc các prompt từ file input
        with open(self.input_prompts, "r") as file:
            print(f"Loading inputs from `{self.input_prompts}`...")
            prompts = file.readlines()

        success = await self.process_stream(prompts, sampling_parameters)

        # Ghi kết quả ra file
        with open(self.results_file, "w") as file:
            for id in self._results_dict.keys():
                for result in self._results_dict[id]:
                    file.write(result.decode("utf-8"))
                file.write("\n")
                file.write("\n=========\n\n")
            print(f"Storing results into `{self.results_file}`...")

        if self.verbose:
            with open(self.results_file, "r") as file:
                print(f"\nContents of `{self.results_file}` ===>")
                print(file.read())
        if success:
            print("PASS: vLLM example")
        else:
            print("FAIL: vLLM example")

    def run_async(self):
        asyncio.run(self.run())

    def create_request(self, prompt, stream, request_id, sampling_parameters, exclude_input_in_output,
                       send_parameters_as_tensor=True):
        inputs = []
        prompt_data = np.array([prompt.encode("utf-8")], dtype=np.object_)
        try:
            inputs.append(grpcclient.InferInput("text_input", [1], "BYTES"))
            inputs[-1].set_data_from_numpy(prompt_data)
        except Exception as error:
            print(f"Encountered an error during request creation: {error}")

        stream_data = np.array([stream], dtype=bool)
        inputs.append(grpcclient.InferInput("stream", [1], "BOOL"))
        inputs[-1].set_data_from_numpy(stream_data)

        if send_parameters_as_tensor:
            sampling_parameters_data = np.array(
                [json.dumps(sampling_parameters).encode("utf-8")], dtype=np.object_
            )
            inputs.append(grpcclient.InferInput("sampling_parameters", [1], "BYTES"))
            inputs[-1].set_data_from_numpy(sampling_parameters_data)

        inputs.append(grpcclient.InferInput("exclude_input_in_output", [1], "BOOL"))
        inputs[-1].set_data_from_numpy(np.array([exclude_input_in_output], dtype=bool))

        outputs = []
        outputs.append(grpcclient.InferRequestedOutput("text_output"))

        return {
            "model_name": self.model,
            "inputs": inputs,
            "outputs": outputs,
            "request_id": str(request_id),
            "parameters": sampling_parameters,
        }

