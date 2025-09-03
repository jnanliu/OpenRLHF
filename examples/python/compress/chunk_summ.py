import os
import json

from tqdm import tqdm
from bespokelabs import curator
from datasets import load_dataset
from llama_index.core.node_parser import SentenceSplitter


PROMPT = """
Your task is to compress the linguistic presentation of the following reasoning trajectory, aiming for enhanced conciseness and clarity.

**Reasoning Trajectory:**
{}

**Guidelines for Simplification:**
1.  **Comprehensive Retention:** Ensure that no discrete reasoning steps are omitted. This emphatically includes all instances of reflection, analysis, verification, and articulation of errors or corrective actions. Explicit verbalizations of internal thought, such as "wait" or "hmm," must be preserved.
2.  **Exclusive Output:** Produce solely the rephrased content. Avoid any introductory or concluding remarks, or other extraneous conversational text.
3.  **Scope Adherence:** Do not generate additional reasoning steps or extend the ongoing problem-solving sequence.
4.  **Structural Integrity:** Adhere strictly to the original formatting and structure of the input reasoning trajectory.

**Output Format**

<start> (Compressed Content) <end>
""".strip()

splitter = SentenceSplitter(
    chunk_size=256,
    chunk_overlap=0
)


class Generator(curator.LLM):
    return_completions_object = True
    
    def prompt(self, input: dict) -> str:
        return PROMPT.format(input["trajectory"])
    
    def parse(self, input: dict, response: dict) -> dict:
        return {
            "trajectory": input["trajectory"],
            "idx": input["idx"],
            "source": input["source"]
        }


if __name__ == "__main__":
    os.environ["HOSTED_VLLM_API_KEY"] = "YOUR_API_KEY"
    
    inputs = []
    with open("data/gpt-oss-processed-rollout.jsonl") as f:
        for line in tqdm(f):
            example = json.loads(line)
            for text in splitter.split_text(example["reasoning_content"]):
                for _ in range(4):
                    inputs.append(
                        {
                            "trajectory": text,
                            "idx": example["idx"],
                            "source": example["source"]
                        }
                    )

    generator = Generator(
        model_name="hosted_vllm/Qwen3-30B-A3B-Instruct-2507",
        backend="litellm",
        backend_params={ 
            "base_url": "https://sd2b27ra80c6ft26ua11g.apigateway-cn-beijing.volceapi.com/v1",
            "request_timeout": 600,
            "max_concurrent_requests": 256,
            "max_retries": 10,
            "api_key": "YOUR_API_KEY",
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
    response = generator(
        inputs, 
        working_dir="data/long_cot_chunk_summ",
        cache_dir="data/long_cot_chunk_summ"
    )
