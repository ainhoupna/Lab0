import math
import random
import re
from typing import List, Any, Union

"""
Core logic for data preprocessing operations, including cleaning, numerical transformation, 
text processing, and structural manipulation.
"""

# Define missing values constant for clarity
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
    # Handle math.nan specifically, as it doesn't equal itself
    return [
        item for item in data 
        if item is not None and item != "" and not (isinstance(item, float) and math.isnan(item))
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
    # Use a set to track seen elements while building a new list to preserve order
    seen = set()
    unique_list = []
    for item in data:
        # Check if the item has been seen already
        if item not in seen:
            seen.add(item)
            unique_list.append(item)
    return unique_list


# --- Numerical Functions (Numeric Group) ---

def min_max_normalize(
    data: List[Union[int, float]], 
    new_min: float = 0.0, 
    new_max: float = 1.0
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
    
    # Avoid division by zero if all values are the same
    if data_max == data_min:
        return [new_min for _ in data]

    # Formula: new_value = (old_value - old_min) / (old_max - old_min) * (new_max - new_min) + new_min
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
    
    # Calculate Mean
    mean = sum(data) / len(data)
    
    # R1728 Fix: Use generator expression instead of list comprehension for sum (better style)
    # Calculate Standard Deviation (population standard deviation used here)
    variance = sum((x - mean) ** 2 for x in data) / len(data)
    std_dev = math.sqrt(variance)
    
    # Avoid division by zero if all values are the same (std_dev = 0)
    if std_dev == 0:
        return [0.0 for _ in data]
        
    # Formula: z = (x - mean) / std_dev
    standardized_data = [(x - mean) / std_dev for x in data]
    return standardized_data


def clip_values(
    data: List[Union[int, float]], 
    min_clip: Union[int, float], 
    max_clip: Union[int, float]
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
    # Use max() to enforce the lower bound and min() to enforce the upper bound
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
            # Check if it's a whole number representation (e.g., '10')
            if isinstance(val, str) and val.strip().isdigit():
                integer_list.append(int(val))
            # Also handle if input is a float string that is an integer (e.g., '10.0')
            elif isinstance(val, str):
                float_val = float(val)
                if float_val == int(float_val):
                    integer_list.append(int(float_val))
        except ValueError:
            # Non-numerical strings (like 'text') are ignored
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
        
    # Use math.log (natural logarithm) and filter for x > 0
    return [math.log(x) for x in data if x > 0]


# --- Text Functions (Text Group) ---

def tokenize_text(text: str) -> str:
    """
    Tokenizes text into words, keeping only alphanumeric characters and lower-casing words.
    The output is a single string of processed words separated by spaces.
    
    Args:
        text: Text to be processed.
        
    Returns:
        Processed text (string).
    """
    # 1. Lower-casing
    text = text.lower()
    # 2. Select only alphanumeric characters and spaces, replacing others with space
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    # 3. Collapse multiple spaces into a single space and strip leading/trailing spaces
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def select_alphanumeric_and_spaces(text: str) -> str:
    """
    Processes text by selecting only alphanumeric characters and spaces (removes punctuation).
    
    Args:
        text: Text to be processed.
        
    Returns:
        Processed text (string).
    """
    # Keeps letters (a-zA-Z), numbers (0-9), and spaces (\s)
    return re.sub(r'[^a-zA-Z0-9\s]', '', text)


def remove_stop_words(text: str, stop_words: List[str]) -> str:
    """
    Removes specified stop-words from the text. The text is lower-cased first.
    
    Args:
        text: Text to be processed.
        stop_words: List of strings to be removed.
        
    Returns:
        Processed text (string) with stop words removed.
    """
    # Lower-case the text for consistent matching
    text_lower = text.lower()
    
    # Simple word tokenization
    words = re.findall(r'\b\w+\b', text_lower) 
    
    # Convert stop_words list to a set for faster lookup
    stop_words_set = set(stop_words)
    
    # Filter words: keep the word if it is NOT in the stop_words set
    filtered_words = [word for word in words if word not in stop_words_set]
    
    # Reconstruct the text
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
    # Use a list comprehension to iterate over sublists and elements
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
    # Create a copy to avoid modifying the original list in place
    shuffled_data = data[:] 
    
    if seed is not None:
        random.seed(seed)
        
    # Shuffle the list in place (we operate on the copy)
    random.shuffle(shuffled_data)
    
    return shuffled_data