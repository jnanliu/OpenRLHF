import os
import json

from bespokelabs import curator
from datasets import load_dataset


class Qwen3Generator(curator.LLM):
    return_completions_object = True
    
    def prompt(self, input: dict) -> str:
        return f"{input['question']}"
    
    def parse(self, input: dict, response: str) -> dict:
        return {
            "question": input["question"],
            "solution": input["solution"],
            "idx": input["idx"],
            "reasoning_content": response["choices"][0]["message"]["reasoning_content"],
            "content": response["choices"][0]["message"]["content"],
        }
    

if __name__ == "__main__":
    os.environ["HOSTED_VLLM_API_KEY"] = "YOUR_API_KEY"

    d = load_dataset("simplescaling/s1K-1.1", split="train")
    inputs = []
    for idx, example in enumerate(d):
        for _ in range(1):
            inputs.append(
                {
                    "question": example["question"],
                    "solution": example["solution"],
                    "idx": idx,
                }
            )
    
    qwen3_generator = Qwen3Generator(
        model_name="hosted_vllm/Qwen/Qwen3-235B-A22B",
        backend="litellm",
        backend_params={ 
            "base_url": "https://sd082he1te3scmnaj20a0.apigateway-cn-beijing.volceapi.com/mlp/s-20250429091742-nzgg7/v1", 
            "request_timeout": 600,
            "max_concurrent_requests": 16
        },
        generation_params={
            "max_tokens": 32768,
            "temperature": 0.6,
            "extra_body": {
                "top_p": 0.9,
                "min_p": 0.0,
                "top_k": 40
            },
        }
    )
    response = qwen3_generator(
        inputs, 
        working_dir="data/qwen3_multi_trajectories"
    )
    