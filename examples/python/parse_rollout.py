import json


if __name__ == "__main__":
    responses = []
    with open("data/qwen3_multi_trajectories/266acb02229d07a7/responses_0.jsonl", "r") as f:
        for line in f:
            responses.append(json.loads(line))

    examples = []
    for idx, response in enumerate(responses):
        examples.append(
            {
                "question": response["generic_request"]["original_row"]["question"],
                "solution": response["generic_request"]["original_row"]["solution"],
                "idx": response["generic_request"]["original_row"]["idx"],
                "reasoning_content": response["raw_response"]["choices"][0]["message"]["reasoning_content"],
                "content": response["raw_response"]["choices"][0]["message"]["content"],
            }
        )

    with open("data/qwen3_multi_trajectories/qwen3_multi_trajectories.jsonl", "w") as f:
        for example in examples:
            f.write(json.dumps(example, ensure_ascii=False) + "\n")