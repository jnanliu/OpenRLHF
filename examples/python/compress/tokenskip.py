import json
from copy import deepcopy

from tqdm import tqdm
from llmlingua import PromptCompressor


if __name__ == "__main__":
    dataset = []
    with open("data/gpt-oss-processed-rollout.jsonl") as f:
        for line in f:
            example = json.loads(line)
            dataset.append(example)

    compressor = PromptCompressor(
        model_name="microsoft/llmlingua-2-xlm-roberta-large-meetingbank",
        use_llmlingua2=True
    )

    compressed_examples = []
    for example in tqdm(dataset): 
        result = compressor.compress_prompt_llmlingua2(
            example["reasoning_content"],
            rate=0.6,
            chunk_end_tokens=[".", "\n", "\n\n"]
        )
        compressed_reasoning_content = result.pop("compressed_prompt")
        copy_example = deepcopy(example)
        copy_example["reasoning_content"] = compressed_reasoning_content
        compressed_examples.append(
            copy_example
        )

    with open("data/gpt-oss-processed-rollout-token-level-compressed.jsonl", "w") as f:
        for example in compressed_examples:
            f.write(json.dumps(example, ensure_ascii=False) + "\n")