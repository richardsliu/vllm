from typing import Dict, Optional, List
import logging
import json

from fastapi import FastAPI
from starlette.requests import Request
from starlette.responses import Response

from ray import serve

from vllm import LLM, SamplingParams

logger = logging.getLogger("ray.serve")

app = FastAPI()


@serve.deployment(
    num_replicas=1,
    max_ongoing_requests=10,
)
#@serve.ingress(app)
class VLLMDeployment:
    def __init__(
        self,
    ):
        self.llm = LLM(model="meta-llama/Meta-Llama-3.1-8B", #model="google/gemma-2b", 
                       tensor_parallel_size=8,
                       enforce_eager=True)


    async def __call__(self, request: Request) -> Response:
    #@app.post("/v1/generate")
    #async def generate(
    #    self, request: Request
    #):
        request_dict = await request.json()
        prompts = request_dict.pop("prompt")
        print("Processing prompt ", prompts)
        sampling_params = SamplingParams(temperature=0.7, 
                                         top_p=1.0,
                                         n=1,
                                         max_tokens=1000)

        outputs = self.llm.generate(prompts, sampling_params)
        for output in outputs:
            prompt = output.prompt
            generated_text = ''
            token_ids = []
            for completion_output in output.outputs:
                generated_text += completion_output.text
                token_ids.extend(list(completion_output.token_ids))

            print("Generated text: ", generated_text)
            ret = {"prompt": prompt, "text": generated_text, "token_ids": token_ids}

        return Response(content=json.dumps(ret))


def build_app() -> serve.Application:
    """Builds the Serve app based on CLI arguments.
    """  # noqa: E501
    tp = 8 #engine_args.tensor_parallel_size
    logger.info(f"Tensor parallelism = {tp}")
    pg_resources = []
    pg_resources.append({"CPU": 1})  # for the deployment replica
    for i in range(tp):
        pg_resources.append({"CPU": 1, "TPU": 1})  # for the vLLM actors

    # We use the "STRICT_PACK" strategy below to ensure all vLLM actors are placed on
    # the same Ray node.
    return VLLMDeployment.options(
        placement_group_bundles=pg_resources, placement_group_strategy="PACK"
    ).bind()


deployment = build_app()
serve.run(deployment)
