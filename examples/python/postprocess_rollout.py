import re
from typing import Any
from argparse import ArgumentParser
import json

from tqdm import tqdm
from transformers import AutoTokenizer

from antlr4.PredictionContext import PredictionContextFromRuleContext
from latex2sympy2_extended import NormalizationConfig
from math_verify import ExprExtractionConfig, LatexExtractionConfig, parse, verify


# Constants for normalization
SUBSTITUTIONS = [
    ("an ", ""),
    ("a ", ""),
    (".$", "$"),
    ("\\$", ""),
    (r"\ ", ""),
    (" ", ""),
    ("mbox", "text"),
    (",\\text{and}", ","),
    ("\\text{and}", ","),
    ("\\text{m}", "\\text{}"),
]

REMOVED_EXPRESSIONS = [
    "square",
    "ways",
    "integers",
    "dollars",
    "mph",
    "inches",
    "hours",
    "km",
    "units",
    "\\ldots",
    "sue",
    "points",
    "feet",
    "minutes",
    "digits",
    "cents",
    "degrees",
    "cm",
    "gm",
    "pounds",
    "meters",
    "meals",
    "edges",
    "students",
    "childrentickets",
    "multiples",
    "\\text{s}",
    "\\text{.}",
    "\\text{\ns}",
    "\\text{}^2",
    "\\text{}^3",
    "\\text{\n}",
    "\\text{}",
    r"\mathrm{th}",
    r"^\circ",
    r"^{\circ}",
    r"\;",
    r",\!",
    "{,}",
    '"',
    "\\dots",
]

def last_boxed_only_string(string: str) -> str:
    """Extract the last LaTeX boxed expression from a string.

    Args:
        string: Input string containing LaTeX code

    Returns:
        The last boxed expression or None if not found
    """
    idx = string.rfind("\\boxed{")
    if idx < 0:
        return ""

    i = idx
    right_brace_idx = None
    num_left_braces_open = 0

    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1

    return string[idx + 7: right_brace_idx] if right_brace_idx is not None else ""

def last_text_only_string(string: str) -> str:
    """Extract the last LaTeX text expression from a string.

    Args:
        string: Input string containing LaTeX code

    Returns:
        The last text expression or None if not found
    """
    idx = string.rfind("\\text{")
    if idx < 0:
        return ""

    i = idx
    right_brace_idx = None
    num_left_braces_open = 0

    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1

    return string[idx + 6: right_brace_idx] if right_brace_idx is not None else ""

def normalize(text: str) -> str:
    text = text.split("=")[-1]

    # Apply substitutions and removals
    for before, after in SUBSTITUTIONS:
        text = text.replace(before, after)
    for expr in REMOVED_EXPRESSIONS:
        text = text.replace(expr, "")

    # Extract and normalize LaTeX math
    text = re.sub(r"(.*?)(\$)(.*?)(\$)(.*)", "$\\3$", text)
    text = re.sub(r"(\\text\{)(.*?)(\})", "\\2", text)
    text = re.sub(r"(\\textbf\{)(.*?)(\})", "\\2", text)
    text = re.sub(r"(\\overline\{)(.*?)(\})", "\\2", text)
    text = re.sub(r"(\\boxed\{)(.*)(\})", "\\2", text)

    # Normalize shorthand TeX:
    #  \fracab -> \frac{a}{b}
    #  \frac{abc}{bef} -> \frac{abc}{bef}
    #  \fracabc -> \frac{a}{b}c
    #  \sqrta -> \sqrt{a}
    #  \sqrtab -> sqrt{a}b
    text = re.sub(r"(frac)([^{])(.)", "frac{\\2}{\\3}", text)
    text = re.sub(r"(sqrt)([^{])", "sqrt{\\2}", text)
    text = text.replace("$", "")

    # Normalize numbers
    if text.replace(",", "").isdigit():
        text = text.replace(",", "")

    return text.strip()

def math_scorer(prediction: str, ground_truth: str, timeout: int = 10) -> float:
    if prediction is None:
        prediction = ""

    prediction = normalize(prediction)
    ground_truth = normalize(ground_truth)

    prediction = f"\\boxed{{{prediction}}}"
    ground_truth = f"\\boxed{{{ground_truth}}}"

    parsed_ground_truth = parse(
        ground_truth, 
        extraction_config=[LatexExtractionConfig(boxed_match_priority=0), ExprExtractionConfig()],
        parsing_timeout=timeout
    )
    if len(parsed_ground_truth) == 0:
        parsed_ground_truth_with_env = f'${ground_truth}$'
        parsed_ground_truth = parse(
            parsed_ground_truth_with_env,
            extraction_config=[LatexExtractionConfig(), ExprExtractionConfig()],
            parsing_timeout=timeout
        )

    if len(parsed_ground_truth) != 0:
        parsed_prediction = parse(
            prediction,
            extraction_config=[
                LatexExtractionConfig(
                    boxed_match_priority=0,
                    try_extract_without_anchor=False,
                    normalization_config=NormalizationConfig(),
                )
            ],
            parsing_timeout=timeout
        )
        if verify(parsed_prediction, parsed_ground_truth, timeout_seconds=timeout):
            return 1.0
        else:
            return 0.0
    return 1.0

def science_scorer(prediction: str, ground_truth: str) -> float:
    if prediction.startswith("("):
        prediction = prediction[1:]
    if prediction.endswith(")"):
        prediction = prediction[:-1]

    if "\\text" in prediction:
        prediction
        prediction = last_text_only_string(prediction)
    if "." in prediction:
        prediction = prediction.split(".")[0]

    if "\\text" in ground_truth:
        ground_truth = last_boxed_only_string(ground_truth)
    
    if prediction.lower() == ground_truth.lower():
        return 1.0
    else:
        return 0.0

def reasongym_scorer(prediction: str, ground_truth: str) -> float:
    if prediction == ground_truth:
        return 1.0
    else:
        return 0.0


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--input_path", type=str)
    parser.add_argument("--output_path", type=str)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained("openai/gpt-oss-120b")

    processed_dataset = []
    summ = []
    with open(args.input_path, "r") as f:
        for line in tqdm(f):
            example = json.loads(line)

            processed_data_item = {}
            parsed_response_message = example["parsed_response_message"]
            processed_data_item.update(parsed_response_message[0])
            if "solution" in processed_data_item:
                processed_data_item["answer"] = processed_data_item["solution"]
                processed_data_item.pop("solution")

            processed_data_item["source"] = example["generic_request"]["original_row"]["source"]
            if not processed_data_item['reasoning_content']:
                processed_data_item["trajectory"] = f"{processed_data_item['content']}"
            else:
                processed_data_item["trajectory"] = f"<think>\n{processed_data_item['reasoning_content']}\n</think>\n{processed_data_item['content']}"

            processed_data_item["meta_info"] = {
                k: example[k] 
                for k in [
                    "generic_request",
                    "created_at",
                    "finished_at",
                    "token_usage",
                    "response_cost",
                    "finish_reason"
                ]
            }

            extracted_predicted_answer = last_boxed_only_string(processed_data_item["content"])
            if processed_data_item["source"] == "math":
                label = math_scorer(extracted_predicted_answer, processed_data_item["answer"], 10)
            elif processed_data_item["source"] == "science":
                label = science_scorer(extracted_predicted_answer, processed_data_item["answer"])
            else:
                label = reasongym_scorer(extracted_predicted_answer, processed_data_item["answer"])
            processed_data_item["prediction"] = extracted_predicted_answer
            processed_data_item["label"] = label

            if label == 1.0:
                processed_dataset.append(processed_data_item)

    with open(args.output_path, "w") as f:
        for data_item in processed_dataset:
            f.write(json.dumps(data_item) + "\n")
