import os
import json

from bespokelabs import curator
from datasets import load_dataset

SYSTEM = """
You are ChatGPT, a large language model trained by OpenAI.
Knowledge cutoff: 2024-06
Current date: 2025-08-08
Reasoning: high
# Valid channels: analysis, commentary, final. Channel must be included for every message.
""".strip()


class GPTOSSGenerator(curator.LLM):
    return_completions_object = True
    
    def prompt(self, input: dict) -> str:
        return [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": f"{input['question']}"}
        ]
    
    def parse(self, input: dict, response: dict) -> dict:
        return {
            "question": input["question"],
            "solution": input["answer"],
            "idx": input["idx"],
            "reasoning_content": response["choices"][0]["message"]["reasoning_content"],
            "content": response["choices"][0]["message"]["content"],
        }
    

if __name__ == "__main__":
    os.environ["HOSTED_VLLM_API_KEY"] = "EMPTY"
    
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

    gpt_generator = GPTOSSGenerator(
        model_name="hosted_vllm/openai/gpt-oss-120b",
        # "hosted_vllm//fs-computility/llm/shared/llmeval/models/opencompass_hf_hub/models--openai--gpt-oss-120b/snapshots/eabf0c518da7584a2e7dab4ab272709785a72126",
        backend="litellm",
        backend_params={ 
            "base_url": "https://sd2e3s0scck1fc4k3685g.apigateway-cn-beijing.volceapi.com/v1",
            "request_timeout": 600,
            "max_concurrent_requests": 32,
            "max_retries": 10,
            "api_key": "EMPTY",
            "invalid_finish_reasons": []
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
    response = gpt_generator(
        inputs, 
        working_dir="data/gpt_long_cot"
    )
    