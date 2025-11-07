import pytest
import math
from src.preprocessing import (
    remove_missing_values,
    fill_missing_values,
    remove_duplicated_values,
    min_max_normalize,
    z_score_standardize,
    clip_values,
    convert_to_integers,
    log_transform,
    tokenize_text,
    select_alphanumeric_and_spaces,
    remove_stop_words,
    flatten_list,
    random_shuffle_list,
)

# --- FIXTURES: For reusable test data ---

@pytest.fixture
def numeric_list_fixture():
    """Fixture for a base list of numbers used in several tests."""
    return [10, 20, 30, 40, 50]

@pytest.fixture
def list_with_missing_values():
    """Fixture for a list containing various types of missing values."""
    return ["a", None, 10, "", "b", math.nan, 20]

# --- UNIT TESTS: Data Cleaning Functions (Clean) ---

# Parametrize used on a function WITHOUT required options (remove_missing_values)
@pytest.mark.parametrize(
    "input_data, expected_output",
    [
        (["a", None, 10, ""], ["a", 10]),
        ([1, 2, math.nan, 4], [1, 2, 4]),
        ([], []),
        ([None, "", math.nan], []),
    ]
)
def test_remove_missing_values(input_data, expected_output):
    """Tests the correct removal of missing values."""
    assert remove_missing_values(input_data) == expected_output

# Parametrize used on a function WITH options (fill_missing_values)
@pytest.mark.parametrize(
    "input_data, fill_value, expected_output",
    [
        # Case 1: Default fill value (0)
        (["a", None, 10, ""], 0, ["a", 0, 10, 0]),
        # Case 2: Custom fill value (N/A)
        (["a", None, 10, ""], "N/A", ["a", "N/A", 10, "N/A"]),
        # Case 3: nan with custom fill value (99)
        ([1, math.nan, 2], 99, [1, 99, 2]),
    ]
)
def test_fill_missing_values_parametrized(input_data, fill_value, expected_output):
    """Tests the correct filling of missing values using different options."""
    assert fill_missing_values(input_data, fill_value) == expected_output

def test_remove_duplicated_values():
    """Tests the removal of duplicates while preserving the order."""
    data = [1, 2, 2, 3, 1, 'a', 'a', 4]
    expected = [1, 2, 3, 'a', 4]
    assert remove_duplicated_values(data) == expected

# --- UNIT TESTS: Numerical Functions (Numeric) ---

def test_min_max_normalize(numeric_list_fixture):
    """Tests Min-Max normalization with default ranges (0-1)."""
    expected = [0.0, 0.25, 0.5, 0.75, 1.0]
    result = min_max_normalize(numeric_list_fixture)
    # Compare floats using pytest.approx()
    assert result == pytest.approx(expected)

def test_min_max_normalize_constant_data():
    """Tests Min-Max normalization when all data points are the same (covers edge case)."""
    data = [10, 10, 10, 10]
    new_min = 5.0
    expected = [5.0, 5.0, 5.0, 5.0]
    assert min_max_normalize(data, new_min=new_min) == expected
    
def test_min_max_normalize_empty_data():
    """Tests Min-Max normalization with an empty list (covers edge case)."""
    assert min_max_normalize([]) == []

def test_z_score_standardize(numeric_list_fixture):
    """Tests Z-score standardization."""
    expected = [-1.414, -0.707, 0.0, 0.707, 1.414]
    result = z_score_standardize(numeric_list_fixture)
    assert result == pytest.approx(expected, abs=1e-3)

def test_z_score_standardize_constant_data():
    """Tests Z-score standardization when std_dev is 0 (covers edge case)."""
    data = [5, 5, 5, 5]
    expected = [0.0, 0.0, 0.0, 0.0]
    assert z_score_standardize(data) == expected
    
def test_z_score_standardize_empty_data():
    """Tests Z-score standardization with an empty list (covers edge case)."""
    assert z_score_standardize([]) == []

def test_clip_values():
    """Tests clipping numerical values to a range."""
    data = [1, 10, 20, 30]
    min_clip = 5
    max_clip = 25
    expected = [5, 10, 20, 25]
    assert clip_values(data, min_clip, max_clip) == expected

def test_convert_to_integers():
    """Tests conversion to integers, excluding non-numerical and non-integer floats."""
    data = ["10", "20.0", "30.5", "text", "40"]
    expected = [10, 20, 40]
    assert convert_to_integers(data) == expected

def test_log_transform():
    """Tests logarithmic transformation (only positives)."""
    data = [1, math.e, 10, -5, 0] # e ≈ 2.718
    expected = [0.0, 1.0, 2.302] 
    result = log_transform(data)
    assert result == pytest.approx(expected, abs=1e-3)

def test_log_transform_empty_data():
    """Tests log transform with an empty list (covers edge case)."""
    assert log_transform([]) == []

# --- UNIT TESTS: Text Functions (Text) ---

def test_tokenize_text():
    """Tests tokenization with cleaning and lower-casing."""
    text = "Hello World! This is Text 123. \n\t"
    expected = "hello world this is text 123"
    assert tokenize_text(text) == expected
    
def test_tokenize_text_extra_spaces():
    """Tests tokenization handles multiple spaces correctly (covers edge case)."""
    text = "   too   many   spaces   "
    expected = "too many spaces"
    assert tokenize_text(text) == expected

def test_select_alphanumeric_and_spaces():
    """Tests selection of alphanumeric characters and spaces."""
    text = "Data Processing $100.00 (OK)"
    expected = "Data Processing 10000 OK"
    assert select_alphanumeric_and_spaces(text) == expected

def test_remove_stop_words():
    """Tests removal of stop-words."""
    text = "The quick brown fox jumps over the lazy dog."
    stop_words = ["the", "over", "a"]
    expected = "quick brown fox jumps lazy dog"
    assert remove_stop_words(text, stop_words) == expected

def test_remove_stop_words_no_match():
    """Tests stop word removal when no words match (covers edge case)."""
    text = "alpha beta gamma"
    stop_words = ["the", "a"]
    expected = "alpha beta gamma"
    assert remove_stop_words(text, stop_words) == expected

# --- UNIT TESTS: Structural Functions (Struct) ---

def test_flatten_list():
    """Tests flattening a list of lists."""
    list_of_lists = [[1, 2], [3], ['a', 'b', 'c']]
    expected = [1, 2, 3, 'a', 'b', 'c']
    assert flatten_list(list_of_lists) == expected

def test_random_shuffle_list_reproducible():
    """Tests random shuffle with a seed for reproducibility (FIXED expected output)."""
    data = [1, 2, 3, 4, 5]
    seed = 42
    
    # CORRECT deterministic output for seed=42 (as determined by your previous failed run)
    expected_shuffle = [4, 2, 3, 5, 1] 
    
    assert random_shuffle_list(data, seed) == expected_shuffle
    
    # Test without seed to ensure it is randomized (must be different from the original)
    data_copy = [1, 2, 3, 4, 5]
    result_no_seed = random_shuffle_list(data_copy)
    assert result_no_seed != [1, 2, 3, 4, 5]