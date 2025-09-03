import json
from argparse import ArgumentParser

from datasets import Dataset


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--input_path", type=str)
    parser.add_argument("--output_path", type=str)
    args = parser.parse_args()

    examples = []
    with open(args.input_path, "r") as f:
        for line in f:
            examples.append(json.loads(line))

    datasets = []
    for example in examples:
        datasets.append(
            {
                "prompt": example["question"],
                "response": example["trajectory"]
            }
        )

    dataset = Dataset.from_list(datasets)
    dataset.to_parquet(args.output_path)
