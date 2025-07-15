import os
import json

from bespokelabs import curator
from datasets import load_dataset


class DeepSeekR1Generator(curator.LLM):
    return_completions_object = True
    
    def prompt(self, input: dict) -> str:
        return f"{input['question']}"
    
    def parse(self, input: dict, response: dict) -> dict:
        return {
            "question": input["question"],
            "solution": input["solution"],
            "idx": input["idx"],
            "reasoning_content": response["choices"][0]["message"]["reasoning_content"],
            "content": response["choices"][0]["message"]["content"],
        }
    

if __name__ == "__main__":
    os.environ["HOSTED_VLLM_API_KEY"] = "sk-admin"

    d = load_dataset("simplescaling/s1K-1.1", split="train")
    inputs = []
    for idx, example in enumerate(d):
        for _ in range(5):
            inputs.append(
                {
                    "question": example["question"],
                    "solution": example["solution"],
                    "idx": idx,
                }
            )
    
    deepseek_generator = DeepSeekR1Generator(
        model_name="hosted_vllm/DeepSeek-R1-0528-FP8", # USD-guiji/deepseek-r1 "hosted_vllm/DeepSeek-R1-0528-FP8 DeepSeek-R1-0528-FP8'
        backend="litellm", # openai litellm
        backend_params={ 
            "base_url": "http://106.15.231.215:40007/v1/", # "http://35.220.164.252:3888/v1/", "https://sd0shm2ujbbc44uo9a3n0.apigateway-cn-beijing.volceapi.com/v1", 
            "request_timeout": 600,
            "max_concurrent_requests": 8,
            "max_retries": 10,
            "api_key": "sk-admin", # "sk-BDZB1IO7c5bUDFpnOImFXci2vSuR5HXNLyPTKaAJLban0Lwy"
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
    response = deepseek_generator(
        inputs, 
        working_dir="data/deepseek_multi_trajectories"
    )
    