import json

from datasets import load_dataset
from transformers import AutoTokenizer


if __name__ == "__main__":
    # dataset = load_dataset("math-dataset/DeepScaleR-Preview-Dataset", split="train")
    # tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")
    # examples = []
    # for example in dataset:
    #     messages = [
    #         {"role": "user", "content": f'{example["problem"]}'},
    #     ]
    #     examples.append(
    #         {
    #             "input": tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True),
    #             "answer": example["answer"],
    #             "datasource": "deepscaler-preview",
    #         }
    #     )
    
    # with open("data/deepscaler-preview-qwen3.jsonl", "w") as f:
    #     for example in examples:
    #         f.write(json.dumps(example) + "\n")
    dataset = load_dataset("zwhe99/DeepMath-103K", split="train", cache_dir="/fs-computility/llmeval/liujunnan/project/misc/DeepMath-103K")
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")
    examples = []
    for example in dataset:
        messages = [
            {"role": "system", "content": "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process is enclosed within <think> </think> tags and answer is enclosed in \\boxed{}, respectively, i.e., <think> reasoning process here </think> ... \\boxed{answer here}."},
            {"role": "user", "content": f'{example["question"]}'},
        ]
        examples.append(
            {
                "input": tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True),
                "answer": example["final_answer"],
                "datasource": "deepmath-103k",
            }
        )
    
    with open("data/deepmath-103k-qwen3.jsonl", "w") as f:
        for example in examples:
            f.write(json.dumps(example) + "\n")


    dataset = load_dataset("HuggingFaceH4/aime_2024", split="train")
    examples = []
    for example in dataset:
        messages = [
            {"role": "system", "content": "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>."},
            {"role": "user", "content": f'{example["problem"]}'},
        ]
        examples.append(
            {
                "input": tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True),
                "answer": example["answer"],
                "datasource": "aime-2024",
            }
        )
    
    with open("data/aime-2024-qwen3.jsonl", "w") as f:
        for example in examples:
            f.write(json.dumps(example) + "\n")