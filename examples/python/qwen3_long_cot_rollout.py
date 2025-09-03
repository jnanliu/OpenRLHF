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
            "solution": input["answer"],
            "idx": input["idx"],
            "reasoning_content": "reasoning_content" in response["choices"][0]["message"]["reasoning_content"] if "reasoning_content" in response["choices"][0]["message"] else "",
            "content": response["choices"][0]["message"]["content"],
        }
    

if __name__ == "__main__":
    os.environ["HOSTED_VLLM_API_KEY"] = "sk-admin"
    os.environ["OPENAI_API_KEY"] = "sk-admin"
    
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
        model_name="hosted_vllm/DeepSeek-V3_1",
        backend="litellm",
        backend_params={ 
            "base_url": "http://106.15.231.215:40007/v1/",
            "request_timeout": 600,
            "max_concurrent_requests": 512,
            "max_retries": 10,
            "api_key": "sk-admin"
        },
        generation_params={
            "max_tokens": 32768,
            "temperature": 0.6,
            "extra_body": {
                "top_p": 0.95,
                "top_k": 40,
                "chat_template_kwargs": {
                    "enable_thinking": True,
                    "thinking": True
                }
            },
        }
    )
    response = qwen3_generator(
        inputs, 
        working_dir="data/qwen3_long_cot",
        cache_dir="data/qwen3_long_cot"
    )
    

# curl http://106.15.231.215:40007/v1/models -H "Content-Type: application/json" -H "Authorization: Bearer sk-admin" 