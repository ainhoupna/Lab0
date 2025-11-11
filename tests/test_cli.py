import pytest
from click.testing import CliRunner
from src.cli import cli

# --- FICTURE: CliRunner ---

@pytest.fixture
def runner():
    """Fixture to instantiate the CliRunner, shared by all integration tests."""
    return CliRunner()

# --- INTEGRATION TESTS: Helper Function Coverage (Negative Tests) ---

def test_parse_list_argument_failure(runner: CliRunner):
    """Tests the failure path of parse_list_argument (invalid JSON input) 
    to cover the 'except json.JSONDecodeError' block."""
    invalid_json = '["a", 10' # Invalid JSON: Missing closing bracket
    
    # We invoke a simple command that uses parse_list_argument
    result = runner.invoke(cli, ['clean', 'remove-missing', invalid_json])
    
    # Assert that the command failed (non-zero exit code)
    assert result.exit_code != 0
    # Assert that the error message is descriptive
    assert "Must be a valid JSON list" in result.output

def test_parse_numeric_argument_failure(runner: CliRunner):
    """Tests the failure path of parse_numeric_list_argument (non-numerical element) 
    to cover the internal argument validation."""
    non_numeric_list = '["a", 10, "text"]' # Valid JSON, but contains non-numeric string
    
    # We invoke a command that uses parse_numeric_list_argument
    result = runner.invoke(cli, ['numeric', 'standardize', non_numeric_list])
    
    assert result.exit_code != 0
    assert "must contain only numerical values" in result.output

# --- INTEGRATION TESTS: Group Clean (Includes fix for previously failed test) ---

def test_clean_remove_missing_integration(runner: CliRunner):
    """Tests 'cli clean remove-missing' command."""
    input_list_str = '["a", null, 10, ""]' 
    result = runner.invoke(cli, ['clean', 'remove-missing', input_list_str])
    
    assert result.exit_code == 0
    assert "['a', 10]" in result.output
    
def test_clean_fill_missing_integration(runner: CliRunner):
    """Tests 'cli clean fill-missing' command with a string fill value (was failing on exit code 2)."""
    input_list_str = '["a", null, 10, ""]'
    fill_value = "MISSING"
    
    # The fix in cli.py (type=str) allows this command to run successfully.
    result = runner.invoke(cli, ['clean', 'fill-missing', input_list_str, '--fill_value', fill_value])
    
    assert result.exit_code == 0
    assert f"Result: ['a', '{fill_value}', 10, '{fill_value}']" in result.output

# --- INTEGRATION TESTS: Group Numeric ---

def test_numeric_normalize_integration(runner: CliRunner):
    """Tests 'cli numeric normalize' command with default min/max."""
    input_list_str = '[10, 20, 30]'
    
    result = runner.invoke(cli, ['numeric', 'normalize', input_list_str])
    
    assert result.exit_code == 0
    assert "[0.0, 0.5, 1.0]" in result.output

def test_numeric_clip_integration(runner: CliRunner):
    """Tests 'cli numeric clip' command with required min/max options."""
    input_list_str = '[1, 10, 20]'
    
    result = runner.invoke(cli, ['numeric', 'clip', input_list_str, '--min_clip', '5', '--max_clip', '15'])
    
    assert result.exit_code == 0
    # Expected output matches the float representation from the clip_values logic
    assert "Result: [5.0, 10, 15.0]" in result.output

# --- INTEGRATION TESTS: Group Text ---

def test_text_tokenize_integration(runner: CliRunner):
    """Tests 'cli text tokenize' command."""
    input_text = "Hello World! This is Text 123."
    
    result = runner.invoke(cli, ['text', 'tokenize', input_text])
    
    assert result.exit_code == 0
    assert "hello world this is text 123" in result.output

# --- INTEGRATION TESTS: Group Struct ---

def test_struct_shuffle_integration(runner: CliRunner):
    """Tests 'cli struct shuffle' command with a fixed seed for predictability."""
    input_list_str = '[1, 2, 3, 4, 5]'
    seed = 42
    
    result = runner.invoke(cli, ['struct', 'shuffle', input_list_str, '--seed', str(seed)])
    
    assert result.exit_code == 0
    # Correct deterministic output for seed=42
    assert "[4, 2, 3, 5, 1]" in result.output

def test_struct_flatten_integration(runner: CliRunner):
    """Tests 'cli struct flatten' command."""
    input_list_of_lists_str = '[[1, 2], [3], [4, 5]]'
    
    result = runner.invoke(cli, ['struct', 'flatten', input_list_of_lists_str])
    
    assert result.exit_code == 0
    assert "[1, 2, 3, 4, 5]" in result.output

# The test for 'unique-values' is covered by clean/remove-duplicates logic, 
# but a direct test can be included for completeness if desired.
def test_struct_unique_values_integration(runner: CliRunner):
    """Tests 'cli struct unique-values' command."""
    input_list_str = '[1, 2, 2, "a", "a", 3]'
    result = runner.invoke(cli, ['struct', 'unique-values', input_list_str])
    
    assert result.exit_code == 0
    assert "[1, 2, 'a', 3]" in result.output