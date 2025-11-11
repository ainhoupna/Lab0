"""
Core logic for data preprocessing operations, including cleaning, numerical transformation,
text processing, and structural manipulation.
"""

import math
import random
import re
from typing import List, Any, Union

MISSING_VALUES = {None, "", math.nan}

# --- Data Cleaning Functions (Clean Group) ---


def remove_missing_values(data: List[Any]) -> List[Any]:
    """
    Removes missing values (None, '', or math.nan) from a list of values.

    Args:
        data: List of values, potentially including missing ones.

    Returns:
        List of values with missing ones removed.
    """
    return [
        item
        for item in data
        if item is not None
        and item != ""
        and not (isinstance(item, float) and math.isnan(item))
    ]


def fill_missing_values(data: List[Any], fill_value: Any = 0) -> List[Any]:
    """
    Replaces missing values (None, '', or math.nan) with a specified fill value.

    Args:
        data: List of values, potentially including missing ones.
        fill_value: The value to use for replacement (default is 0).

    Returns:
        List of values with missing ones replaced.
    """
    filled_data = []
    for item in data:
        is_nan = isinstance(item, float) and math.isnan(item)
        if item is None or item == "" or is_nan:
            filled_data.append(fill_value)
        else:
            filled_data.append(item)
    return filled_data


def remove_duplicated_values(data: List[Any]) -> List[Any]:
    """
    Removes duplicated values from a list, preserving order of first appearance.

    Args:
        data: List of values.

    Returns:
        List of unique values.
    """
    seen = set()
    unique_list = []
    for item in data:
        if item not in seen:
            seen.add(item)
            unique_list.append(item)
    return unique_list


# --- Numerical Functions (Numeric Group) ---


def min_max_normalize(
    data: List[Union[int, float]], new_min: float = 0.0, new_max: float = 1.0
) -> List[float]:
    """
    Normalizes numerical values using the Min-Max method.

    Args:
        data: List of numerical values.
        new_min: The minimum value of the new range (default is 0.0).
        new_max: The maximum value of the new range (default is 1.0).

    Returns:
        List of normalized values (float).
    """
    if not data:
        return []

    data_min = min(data)
    data_max = max(data)

    if data_max == data_min:
        return [new_min for _ in data]

    denominator = data_max - data_min

    normalized_data = [
        ((val - data_min) / denominator) * (new_max - new_min) + new_min
        for val in data
    ]
    return normalized_data


def z_score_standardize(data: List[Union[int, float]]) -> List[float]:
    """
    Standardizes numerical values using the Z-score method.

    Args:
        data: List of numerical values.

    Returns:
        List of standardized values (float).
    """
    if not data:
        return []

    mean = sum(data) / len(data)

    # Use generator expression for variance calculation (Pylint R1728 fix)
    variance = sum((x - mean) ** 2 for x in data) / len(data)
    std_dev = math.sqrt(variance)

    if std_dev == 0:
        return [0.0 for _ in data]

    standardized_data = [(x - mean) / std_dev for x in data]
    return standardized_data


def clip_values(
    data: List[Union[int, float]],
    min_clip: Union[int, float],
    max_clip: Union[int, float],
) -> List[Union[int, float]]:
    """
    Clips numerical values to a specified minimum and maximum range.

    Args:
        data: List of numerical values.
        min_clip: Minimum value to clip to.
        max_clip: Maximum value to clip to.

    Returns:
        List of clipped values.
    """
    # The logic returns a list of numbers, which may be floats if input was float.
    return [max(min_clip, min(max_clip, val)) for val in data]


def convert_to_integers(data: List[str]) -> List[int]:
    """
    Converts elements in a list of strings to integers, excluding non-numerical
    and incomplete numerical strings.

    Args:
        data: List of strings (can include numerical and non-numerical values).

    Returns:
        List of values converted to integers (non-numerical values are excluded).
    """
    integer_list = []
    for val in data:
        try:
            if isinstance(val, str) and val.strip().isdigit():
                integer_list.append(int(val))
            elif isinstance(val, str):
                float_val = float(val)
                if float_val == int(float_val):
                    integer_list.append(int(float_val))
        except ValueError:
            pass
    return integer_list


def log_transform(data: List[Union[int, float]]) -> List[float]:
    """
    Applies a logarithmic scale transformation (natural log, ln) to a list of values.
    Only original positive numbers are transformed (log(x) for x > 0).

    Args:
        data: List of numerical values.

    Returns:
        List of values converted to logarithmic scale.
    """
    if not data:
        return []

    return [math.log(x) for x in data if x > 0]


# --- Text Functions (Text Group) ---


def tokenize_text(text: str) -> str:
    """
    Tokenizes text into words, keeping only alphanumeric characters and lower-casing words.

    Args:
        text: Text to be processed.

    Returns:
        Processed text (string).
    """
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def select_alphanumeric_and_spaces(text: str) -> str:
    """
    Processes text by selecting only alphanumeric characters and spaces (removes punctuation).

    Args:
        text: Text to be processed.

    Returns:
        Processed text (string).
    """
    return re.sub(r"[^a-zA-Z0-9\s]", "", text)


def remove_stop_words(text: str, stop_words: List[str]) -> str:
    """
    Removes specified stop-words from the text. The text is lower-cased first.

    Args:
        text: Text to be processed.
        stop_words: List of strings to be removed.

    Returns:
        Processed text (string) with stop words removed.
    """
    text_lower = text.lower()
    words = re.findall(r"\b\w+\b", text_lower)
    stop_words_set = set(stop_words)
    filtered_words = [word for word in words if word not in stop_words_set]
    return " ".join(filtered_words)


# --- Structure Functions (Struct Group) ---


def flatten_list(list_of_lists: List[List[Any]]) -> List[Any]:
    """
    Flattens a list of lists into a single list.

    Args:
        list_of_lists: A list containing sub-lists.

    Returns:
        A flattened list.
    """
    return [item for sublist in list_of_lists for item in sublist]


def random_shuffle_list(data: List[Any], seed: int = None) -> List[Any]:
    """
    Randomly shuffles a list of values using an optional seed for reproducibility.

    Args:
        data: List of values to be shuffled.
        seed: Seed value for the random number generator (default is None).

    Returns:
        A new list of shuffled values.
    """
    shuffled_data = data[:]

    if seed is not None:
        random.seed(seed)

    random.shuffle(shuffled_data)

    return shuffled_data
