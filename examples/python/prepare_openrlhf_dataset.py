import json
import random
random.seed(42)
from collections import defaultdict

from datasets import load_dataset, Dataset


def load_s1_data(only_deepseek=False):
    d1 = load_dataset("simplescaling/s1K-1.1", split="train")
    d2 = load_dataset("simplescaling/s1K-claude-3-7-sonnet", split="train")

    question_and_source2example = {}
    for example in d1:
        question_and_source2example[f"{example['question']}_{example['source_type']}"] = example
    
    for example in d2:
        key = f"{example['question']}_{example['source_type']}"
        question_and_source2example[key]["claude_thinking_trajectory"] = example["claude_thinking_trajectory"]
        question_and_source2example[key]["claude_attempt"] = example["claude_attempt"]

    examples = []
    for item in list(question_and_source2example.values()):
        if not only_deepseek:
            examples.append(
                {
                    "question": item["question"], 
                    "answer": item["solution"], 
                    "source": "claude_s1",
                    "input": [
                        {
                            "role": "user",
                            "content": item["question"], 
                        },
                    ],
                    "output": [
                        {
                            "role": "assistant",
                            "content": "\n<think>\n" + item["claude_thinking_trajectory"] + "\n</think>\n" + item["claude_attempt"], 
                        }
                    ]
                }
            )
            examples.append(
                {
                    "question": item["question"], 
                    "answer": item["solution"],
                    "source": "gemini_s1",
                    "input": [
                        {
                            "role": "user",
                            "content": item["question"], 
                        },
                    ],
                    "output": [
                        {
                            "role": "assistant",
                            "content": "\n<think>\n" + item["gemini_thinking_trajectory"] + "\n</think>\n" + item["gemini_attempt"], 
                        }
                    ]
                }
            )
        examples.append(
            {
                "question": item["question"], 
                "answer": item["solution"], 
                "source": "deepseek_s1",
                "input": [
                    {
                        "role": "user",
                        "content": item["question"], 
                    },
                ],
                "output": [
                    {
                        "role": "assistant",
                        "content": "\n<think>\n" + item["deepseek_thinking_trajectory"] + "\n</think>\n" + item["deepseek_attempt"], 
                    }
                ]
            }
        )
    
    if not only_deepseek:
        with open("data/deepseek_1xtrajectories+gemini_1xtrajectories+claude_1xtrajectories_s1.jsonl", "w") as f:
            for example in examples:
                f.write(json.dumps(example, ensure_ascii=False) + "\n")
    else:
        with open("data/deepseek_1xtrajectories_s1.jsonl", "w") as f:
            for example in examples:
                f.write(json.dumps(example, ensure_ascii=False) + "\n")

    return examples


def load_qwen3_data(n=3):
    examples = []
    idx2item = defaultdict(list)
    with open("data/qwen3_multi_trajectories/qwen3_multi_trajectories.jsonl", "r") as f:
        for line in f:
            item = json.loads(line)
            idx2item[item["idx"]].append(item)
    
    idx2item = {k: random.sample(v, n) for k, v in idx2item.items()}
    all_items = []
    for v in idx2item.values():
        all_items.extend(v)
    for item in all_items:
        examples.append(
            {
                "question": item["question"], 
                "answer": item["solution"],
                "source": "deepseek_curator_rollout",
                "input": [
                    {
                        "role": "user",
                        "content": item["question"], 
                    }
                ],
                "output": [
                    {
                        "role": "assistant",
                        "content": "\n<think>\n" + item["reasoning_content"] + "\n</think>\n" + item["content"], 
                    }
                ]
            }
        )

    with open(f"data/qwen3_{n}xtrajectories.jsonl", "w") as f:
        for example in examples:
            f.write(json.dumps(example, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    load_qwen3_data(n=5)
    # load_s1_data(only_deepseek=True)