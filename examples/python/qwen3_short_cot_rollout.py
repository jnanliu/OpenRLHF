import os
import json

from bespokelabs import curator
from datasets import load_dataset


class Qwen3Generator(curator.LLM):
    return_completions_object = True
    
    def prompt(self, input: dict) -> str:
        return f"{input['question']}"
    
    def parse(self, input: dict, response: dict) -> dict:
        return {
            "question": input["question"],
            "answer": input["answer"],
            "idx": input["idx"],
            "reasoning_content": response["choices"][0]["message"]["reasoning_content"],
            "content": response["choices"][0]["message"]["content"],
        }
    

if __name__ == "__main__":
    os.environ["HOSTED_VLLM_API_KEY"] = "YOUR_API_KEY"
    
    inputs = []
    idxs = {}
    with open("data/questions.jsonl") as f:
        for line in f:
            example = json.loads(line)
            if example["source"] not in idxs:
                idxs[example["source"]] = 0
            else:
                idxs[example["source"]] += 1
            for _ in range(4):
                inputs.append(
                    {
                        "question": example["question"],
                        "answer": example["answer"],
                        "source": example["source"],
                        "idx": idxs[example["source"]]
                    }
                )

    qwen3_generator = Qwen3Generator(
        model_name="Qwen/Qwen3-30B-A3B-Instruct-2507",
        # backend="litellm",
        backend_params={ 
            "base_url": "https://sd2b27ra80c6ft26ua11g.apigateway-cn-beijing.volceapi.com/v1",
            "request_timeout": 600,
            "max_concurrent_requests": 256,
            "max_retries": 10,
            "api_key": "YOUR_API_KEY"
        },
        generation_params={
            "max_tokens": 32768,
            "temperature": 0.6,
            "extra_body": {
                "top_p": 0.95,
                "top_k": 40
            },
        }
    )
    response = qwen3_generator(
        inputs, 
        working_dir="data/qwen3_short_cot",
        cache_dir="data/qwen3_short_cot"
    )
    