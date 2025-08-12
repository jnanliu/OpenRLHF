import re
from concurrent.futures import ThreadPoolExecutor
import traceback

import torch
from openai import OpenAI
from transformers import AutoTokenizer
import requests
import numpy as np

from latex2sympy2_extended import NormalizationConfig
from math_verify import ExprExtractionConfig, LatexExtractionConfig, parse, verify


model_name="Qwen/Qwen3-30B-A3B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
url_list=["http://172.30.6.89:8000", "http://172.30.6.88:8000"]


# Compile a regular expression pattern to match a specific format of strings.
# This pattern is used to extract reasoning, conclusion, and answer from a given string.
format_pattern = re.compile(
    r"""
    ^                                 # Match the start of the string.
    (?!.*<conclusion>.*<conclusion>)  # Negative lookahead to ensure <conclusion> tag appears only once.
    (?!.*</conclusion>.*</conclusion>)# Negative lookahead to ensure </conclusion> tag appears only once.
    (?!.*\\boxed.*\\boxed)            # Negative lookahead to ensure \boxed appears only once.
    (?P<reasoning>.*?)                # Non-greedily capture any characters and name this group 'reasoning'.
    \n<conclusion>\n                  # Match a newline, the <conclusion> tag, and another newline.
    # Capture the conclusion part
    (?P<conclusion>                   # Start capturing the conclusion part and name this group 'conclusion'.
        (.*?(?P<answer>\\boxed\{      # Non-greedily capture any characters until "\boxed{", 
                                     # then start capturing the answer and name this group 'answer'.
    # Capture the boxed part
    (.*)                          # Capture any characters that are not curly braces.
    \}).*?)                           # Match the closing brace of \boxed{} and any remaining characters non-greedily.
    )
    \n</conclusion>                   # Match a newline, the </conclusion> tag.
    $                                 # Match the end of the string.
    """,
    re.DOTALL | re.VERBOSE
)
boxed_pattern = re.compile(
    r"""
    \\boxed\{                       # Match the literal string "\boxed{"
    # Capture the boxed part
    (.*)
    \}                              # Match the closing brace
    """,
    re.DOTALL | re.VERBOSE,
)

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

    return string[idx : right_brace_idx + 1] if right_brace_idx is not None else ""

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


def math_equiv(prediction: str, ground_truth: str, timeout: int = 10) -> bool:
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
            return True
        else:
            return False
    return True

def parse_response(completion: str) -> dict[str, str] | None:
    """
    Parse the given completion string to extract the reasoning process, conclusion block, and answer.

    This function checks if the completion string contains exactly one <conclusion> tag, 
    one </conclusion> tag, and the closing tag comes after the opening tag. It also checks 
    if the string contains exactly one \\boxed{} environment. If all checks pass, it uses 
    predefined regular expression patterns to extract the reasoning process, conclusion block, 
    and answer.

    Args:
        completion (str): The completion string to be parsed.

    Returns:
        Dict[str, str] | None: A dictionary containing the keys "reasoning", "conclusion", 
        and "answer", or None if parsing fails.
    """
    # Attempt to match the entire completion string using the predefined format_pattern
    # The format_pattern is designed to enforce specific structural rules on the completion string
    # such as single occurrence of <conclusion> tags and \boxed{} environment.
    match = format_pattern.match(completion)
    if not match:
        # Return None if the matching fails, indicating the string does not meet the required format.
        return None

    # Extract the reasoning process from the match result and strip leading and trailing whitespace
    # The "reasoning" group is defined in the format_pattern regular expression.
    reasoning = match.group("reasoning").strip()
    # Extract the conclusion block from the match result and strip leading and trailing whitespace
    # The "conclusion" group is defined in the format_pattern regular expression.
    conclusion_block = match.group("conclusion").strip()
    # Extract the answer from the match result and strip leading and trailing whitespace
    # The "answer" group is defined in the format_pattern regular expression.
    answer = match.group("answer").strip()
    
    # Return a dictionary containing the reasoning process, conclusion block, and answer
    return {
        "reasoning": reasoning,
        "conclusion": conclusion_block,
        "answer": answer
    }

def extract_boxed(completion: str) -> str | None:
    """
    Extract the content enclosed within the \\boxed{} environment from the given completion string.

    This function uses a predefined regular expression pattern to search for all occurrences of
    content within the \\boxed{} environment in the input string. It then returns the last match
    if any matches are found.

    Args:
        completion (str): The input string from which the \\boxed{} content is to be extracted.

    Returns:
        str | None: The extracted content if found, otherwise None.
    """
    # Use the predefined regular expression pattern `boxed_pattern` to find all occurrences
    # of the \\boxed{} environment in the completion string.
    # `findall` returns a list of all non-overlapping matches in the string.
    match = boxed_pattern.findall(completion)
    # Check if any matches were found. If so, return the content of the last match.
    # If no matches were found, return None.
    return match[-1] if match else None

def get_log_probs(url: str, input: str):
    retry = 0
    suc = False
    while not suc and retry < 5:
        try:
            response = requests.post( 
                f"{url}/generate",
                json={ 
                    "text": input, 
                    "sampling_params": { 
                        "temperature": 1.0, 
                        "max_new_tokens": 1, 
                        "n": 1
                    }, 
                    "return_logprob": True, 
                    "top_logprobs_num": 20,
                    "logprob_start_len": 0, 
                }, 
            )
            suc = True
        except:
            print(traceback.format_exc())
            retry += 1
    data = response.json()
    input_top_log_probs = data["meta_info"]["input_top_logprobs"][1:]
    tokens = [[top[1] for top in pos] for pos in input_top_log_probs]
    probs = [[np.exp(top[0]) for top in pos] for pos in input_top_log_probs]
    input_log_probs = data["meta_info"]["input_token_logprobs"][1:]
    log_probs = torch.tensor([pos[0] for pos in input_log_probs])
    return tokens, probs, log_probs

def reward_func(
    queries: list[str], 
    prompts: list[str], 
    labels: list[str],
    **kwargs
):
    """
    Reward function for distillation.
    """
    inputs = queries
    
    with ThreadPoolExecutor(max_workers=16) as executor:
        results = list(
            executor.map(
                get_log_probs, 
                [url_list[i % len(url_list)] for i in range(len(inputs))], 
                inputs
            )
        )
        tokens, probs, log_probs = zip(*results)
    
    # tokens = torch.tensor(tokens)
    # probs = torch.tensor(probs)
    # log_probs = torch.tensor(log_probs)

    def compute_reward(completion, ground_truth):
        pared_completion = parse_response(completion)
        if pared_completion is None:
            prediction = extract_boxed(completion) or ""
            if math_equiv(prediction, ground_truth):
                return 1.0
            return -1.0
        else:
            prediction = pared_completion["answer"]
            if math_equiv(prediction, ground_truth):
                return 1.0
            return 0.0

    rewards = torch.tensor(
        [
            compute_reward(query, label) 
            for query, label in zip(queries, labels)
        ]
    )

    return {
        "rewards": rewards,  # Rewards for advantage calculation
        "scores": rewards,   # Scores for dynamic filtering (0-1 reward)
        "teacher_log_probs": log_probs,
        # "teacher_tokens": tokens,
        # "teacher_probs": probs,
        "extra_logs": {"dummy_scores": rewards},  # Additional logging info for wandb
    }