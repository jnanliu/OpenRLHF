import json
import random

from datasets import load_dataset
import datasets
from transformers import AutoTokenizer

SYSTEM = """
A conversation between a User and an Assistant. The User poses a question, and the Assistant provides a solution. The Assistant's response follows these structured steps:

1. **Reasoning Process**: The Assistant comprehensively thinks about the problem through a reasoning process.
2. **Conclusion**: The Assistant reaches a conclusion, which is enclosed within `<conclusion>` and `</conclusion>` tags. The final answer is highlighted within `\\boxed{...final answer...}`.
3. **Response Format**: The complete response should be formatted as:

...reasoning process...
<conclusion>
...conclusion...
The answer is \\boxed{...final answer...}
</conclusion>
""".strip()


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


    def last_boxed_only_string(string):
        idx = string.rfind('\\boxed')
        if idx < 0:
            idx = string.rfind('\\fbox')
            if idx < 0:
                return None

        i = idx
        right_brace_idx = None
        num_left_braces_open = 0
        while i < len(string):
            if string[i] == '{':
                num_left_braces_open += 1
            if string[i] == '}':
                num_left_braces_open -= 1
                if num_left_braces_open == 0:
                    right_brace_idx = i
                    break
            i += 1

        if right_brace_idx is None:
            retval = None
        else:
            retval = string[idx:right_brace_idx + 1]

        return retval

    def remove_boxed(s):
        left = '\\boxed{'
        try:
            assert s[:len(left)] == left
            assert s[-1] == '}'
            return s[len(left):-1]
        except Exception:
            return None

    def extract_boxed(string, strip_double_curly_brace=False):
        if string.count("\\boxed") > 1:
            return None
        boxed_string = last_boxed_only_string(string)
        if boxed_string is None:
            return None
        answer = remove_boxed(boxed_string)
        if answer is None:
            return None
        if strip_double_curly_brace:
            match = re.match('^\{(.*)\}$', answer)  # noqa: W605
            if match:
                answer = match.group(1)
        return answer

    random.seed(42)

    dataset = []

    math_dataset = datasets.load_dataset("SynthLabsAI/Big-Math-RL-Verified", split="train")
    math_dataset = math_dataset.filter(lambda example: example["llama8b_solve_rate"] is not None)
    for example in random.choices(math_dataset.filter(lambda example: example["llama8b_solve_rate"] >= 0.7), k=1000):
        dataset.append(
            {
                "question": example["problem"],
                "final_answer": example["answer"],
            }
        )
    for example in random.choices(math_dataset.filter(lambda example: (example["llama8b_solve_rate"] > 0.3 and example["llama8b_solve_rate"] < 0.7)), k=1000):
        dataset.append(
            {
                "question": example["problem"],
                "final_answer": example["answer"],
            }
        )
    for example in random.choices(math_dataset.filter(lambda example: example["llama8b_solve_rate"] <= 0.3), k=8000):
        dataset.append(
            {
                "question": example["problem"],
                "final_answer": example["answer"],
            }
        )

    for config in ['algebra', 'counting_and_probability', 'geometry', 'intermediate_algebra', 'number_theory', 'prealgebra', 'precalculus']:
        math_dataset = datasets.load_dataset("EleutherAI/hendrycks_math", config, split="train")
        for example in math_dataset:
            if example["level"] not in ["Level 5", "Level 4", "Level 3"]:
                continue
            answer = extract_boxed(example["solution"])
            if answer is None:
                continue
            dataset.append(
                {
                    "question": example["problem"],
                    "final_answer": answer,
                }
            )   
            

    # dataset = load_dataset("zwhe99/DeepMath-103K", split="train", cache_dir="/fs-computility/llmeval/liujunnan/project/misc/DeepMath-103K")
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")
    examples = []
    for example in dataset:
        messages = [
            # {"role": "system", "content": "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process is enclosed within <think> </think> tags and answer is enclosed in \\boxed{}, respectively, i.e., <think> reasoning process here </think> ... \\boxed{answer here}."},
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": f'{example["question"]}'},
        ]
        examples.append(
            {
                "input": tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True),
                "answer": example["final_answer"],
                # "datasource": "deepmath-103k",
                "datasource": "big-math",
            }
        )
    
    with open("data/big-math-qwen3.jsonl", "w") as f:
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