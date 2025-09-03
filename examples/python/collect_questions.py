import json
from pathlib import Path
import random

import reasoning_gym
from tqdm import tqdm
from datasets import load_dataset


reasoning_gym_configs = lambda is_train, size: [
    ("ab", 1.0, {"seed": 42, "length": 10, "size": size}),
    ("ab", 1.0, {"seed": 42, "length": 15, "size": size}),
    ("acre", 1.0, {"seed": 42, "train": 1 if is_train else 0, "size": size}),
    ("advanced_geometry", 1.0, {"seed": 42, "min_coord": -100, "max_coord": 100, "size": size}),
    ("aiw", 1.0, {"seed": 42, "max_entities": 10, "size": size}),
    ("cryptarithm", 1.0, {"seed": 42, "min_words": 5, "max_words": 20, "size": size}),
    ("dice", 1.0, {"seed": 42, "num_dice": 5, "max_dice_size": 30, "size": size}),
    ("futoshiki", 1.0, {"seed": 42, "size": size}),
    ("game_of_life", 1.0, {"seed": 42, "grid_size_x": 30, "grid_size_y": 30, "simulation_steps": 3, "size": size}),
    ("game_of_life", 1.0, {"seed": 42, "grid_size_x": 30, "grid_size_y": 30, "simulation_steps": 4, "size": size}),
    ("game_of_life", 1.0, {"seed": 42, "grid_size_x": 30, "grid_size_y": 30, "simulation_steps": 5, "size": size}),
    ("game_of_life_halting", 1.0, {"seed": 42, "grid_size_x": 30, "grid_size_y": 30, "difficulty": 3, "num_oscillators": 8, "max_simulation_steps": 40, "size": size}),
    ("jugs", 1.0, {"seed": 42, "difficulty": 20, "size": size}),
    ("knight_swap", 1.0, {"seed": 42, "size": size}),
    ("knights_knaves", 1.0, {"seed": 42, "n_people": 3, "depth_constraint": 3, "width_constraint": 3, "size": size}),
    ("knights_knaves", 1.0, {"seed": 42, "n_people": 5, "depth_constraint": 5, "width_constraint": 5, "size": size}),
    ("mahjong_puzzle", 1.0, {"seed": 42, "min_num_rounds": 30, "size": size}),
    ("needle_haystack", 1.0, {"seed": 42, "min_num_statements": 50, "size": size}),
    ("quantum_lock", 1.0, {"seed": 42, "difficulty": 10, "size": size}),
    ("quantum_lock", 1.0, {"seed": 42, "difficulty": 20, "size": size}),
    ("rush_hour", 1.0, {"seed": 42, "min_moves": 10, "size": size}),
    ("self_reference", 1.0, {"seed": 42, "difficulty": 10, "size": size}),
    ("sudoku", 1.0, {"seed": 42, "size": size}),
    ("zebra_puzzles", 1.0, {"seed": 42, "num_people": 4, "num_characteristics": 4, "size": size}),
    ("zebra_puzzles", 1.0, {"seed": 42, "num_people": 5, "num_characteristics": 5, "size": size}),
    ("zebra_puzzles", 1.0, {"seed": 42, "num_people": 6, "num_characteristics": 6, "size": size}),("zebra_puzzles", 1.0, {"seed": 42, "num_people": 7, "num_characteristics": 7, "size": size})
]

def prepare_data():
    random.seed(42)
    samples = []

    science_dataset = load_dataset(
        "nvidia/OpenScienceReasoning-2", 
        split="train", 
        cache_dir=Path(__file__).parents[3].joinpath(".cache")
    )
    science_dataset = science_dataset.select(random.sample(list(range(len(science_dataset))), 20_000))
    for item in science_dataset:
        samples.append(
            {
                "question": item["input"],
                "answer": item["expected_answer"],
                "source": "science"
            }
        )

    math_dataset = load_dataset(
        "zwhe99/DeepMath-103K", 
        split="train", 
        cache_dir=Path(__file__).parents[3].joinpath(".cache")
    )
    math_dataset = math_dataset.filter(lambda x: x["difficulty"] >= 7)
    for item in math_dataset:
        samples.append(
            {
                "question": item["question"] + "\n\nPlease reason step by step, and put your final answer within \\boxed{{}}",
                "answer": item["final_answer"],
                "source": "math"
            }
        )

    puzzle_datasets = [
        reasoning_gym.create_dataset(name, **config)
        for name, _, config in reasoning_gym_configs(True, 1000)
    ]
    for d in tqdm(puzzle_datasets):
        for item in tqdm(d, total=1000):
            item.pop("metadata")
            samples.append(
                {
                    "question": item["question"]+ "\n\nPlease reason step by step, and put your final answer within \\boxed{{}}",
                    "answer": item["answer"],
                    "source": "puzzle"
                }
            )

    with open("data/questions.jsonl", "w") as f:
        for sample in samples:
            f.write(json.dumps(sample) + "\n")


if __name__ == "__main__":
    prepare_data()
    