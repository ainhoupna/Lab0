import json 
from typing import List, Any, Union 

import click  

"""
Command Line Interface (CLI) for data preprocessing functionalities, built with Click.
It organizes functions into groups: clean, numeric, text, and struct.
"""

# Local Imports
from .preprocessing import (
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

# --- CLI Groups Definition ---

@click.group(help="CLI for the MLOps assignment: Fundamentals of Continuous Integration (CI).")
def cli():
    """
    Main command group for data preprocessing tasks.
    """

@cli.group(help="Functions related to data cleaning and validation.")
def clean():
    """Command group for data cleaning."""

@cli.group(help="Functions for preprocessing numerical attributes.")
def numeric():
    """Command group for numerical preprocessing."""

@cli.group(help="Functions for preprocessing textual information.")
def text():
    """Command group for text processing."""

@cli.group(help="Functions related to data structure manipulation (lists, etc.).")
def struct():
    """Command group for structure manipulation."""

# --- Helper functions for argument parsing (Requires coverage for 'except' blocks) ---

def parse_list_argument(_ctx, _param, value) -> List[Any]:
    """Attempts to parse a string argument as a JSON list.
    
    This function is a Click callback for arguments expecting a list.
    """
    # Arguments ctx, param renamed to _ctx, _param to satisfy W0613
    try:
        return json.loads(value)
    except json.JSONDecodeError as exc:
        error_msg = (
            f"Must be a valid JSON list (e.g., '[\"a\", \"b\"]'). "
            f"Value received: {value}"
        )
        # Explicitly chain exception for W0707
        raise click.BadParameter(error_msg) from exc


def parse_numeric_list_argument(_ctx, _param, value) -> List[Union[int, float]]:
    """Attempts to parse a string argument as a list of numbers.
    
    This function validates that all list elements are numerical.
    """
    parsed_list = parse_list_argument(_ctx, _param, value)
    
    # This loop checks for the non-numerical failure case (needs negative test coverage)
    for item in parsed_list:
        if not isinstance(item, (int, float)):
             raise click.BadParameter(
                 f"The list must contain only numerical values for this command. "
                 f"Invalid value: {item}"
             )
    
    return parsed_list


# --- CLEAN Group Commands ---

@clean.command(
    help="Removes missing values (None, '', nan) from a list."
)
@click.argument('values', type=str, callback=parse_list_argument)
@click.pass_context
def remove_missing(_ctx, values):
    """Example: cli clean remove-missing '["a", null, 10, ""]'"""
    result = remove_missing_values(values)
    click.echo(f"Result: {result}")

@clean.command(
    help="Replaces missing values (None, '', nan) with a given fill value."
)
@click.argument('values', type=str, callback=parse_list_argument)
@click.option(
    '--fill_value',
    '-f',
    default=0,
    type=str, # FIX: Explicitly set type=str to allow string inputs like "MISSING"
    help='The value that will replace the missing values (default: 0).',
    show_default=True
)
@click.pass_context
def fill_missing(_ctx, values, fill_value):
    """Example: cli clean fill-missing '["a", null, 10, ""]' --fill-value 'N/A'"""
    result = fill_missing_values(values, fill_value)
    click.echo(f"Result: {result}")


# --- NUMERIC Group Commands (Standardize/Clip rely on parse_numeric_list_argument) ---

@numeric.command(
    help="Normalizes numerical values using the Min-Max method."
)
@click.argument('values', type=str, callback=parse_numeric_list_argument)
@click.option(
    '--new_min',
    '-min',
    type=float,
    default=0.0,
    help='The new minimum value of the range (default: 0.0).',
    show_default=True
)
@click.option(
    '--new_max',
    '-max',
    type=float,
    default=1.0,
    help='The new maximum value of the range (default: 1.0).',
    show_default=True
)
@click.pass_context
def normalize(_ctx, values, new_min, new_max):
    """Example: cli numeric normalize '[10, 20, 30]' --new-min 5 --new-max 15"""
    result = min_max_normalize(values, new_min, new_max)
    click.echo(f"Result: {result}")

@numeric.command(
    help="Standardizes numerical values using the Z-score method."
)
@click.argument('values', type=str, callback=parse_numeric_list_argument)
@click.pass_context
def standardize(_ctx, values):
    """Example: cli numeric standardize '[10, 20, 30, 40, 50]'"""
    result = z_score_standardize(values)
    click.echo(f"Result: {result}")

@numeric.command(
    help="Clips numerical values to a minimum and maximum range."
)
@click.argument('values', type=str, callback=parse_numeric_list_argument)
@click.option(
    '--min_clip',
    type=float,
    required=True, 
    help='The minimum value to clip to.'
)
@click.option(
    '--max_clip',
    type=float,
    required=True, 
    help='The maximum value to clip to.'
)
@click.pass_context
def clip(_ctx, values, min_clip, max_clip):
    """Example: cli numeric clip '[1, 10, 20]' --min-clip 5 --max-clip 15"""
    result = clip_values(values, min_clip, max_clip)
    click.echo(f"Result: {result}")

@numeric.command(
    name="to-int",
    help="Converts numerical strings to integers (excluding non-numerics)."
)
@click.argument('values', type=str, callback=parse_list_argument)
@click.pass_context
def conversion_to_integers(_ctx, values):
    """Example: cli numeric to-int '[\"10\", \"20.0\", \"text\", \"30\"]'"""
    result = convert_to_integers(values)
    click.echo(f"Result: {result}")

@numeric.command(
    name="log-scale",
    help="Logarithmic scale transformation (only positive numbers)."
)
@click.argument('values', type=str, callback=parse_numeric_list_argument)
@click.pass_context
def transformation_to_logarithmic_scale(_ctx, values):
    """Example: cli numeric log-scale '[1, 10, -5, 100]'"""
    result = log_transform(values)
    click.echo(f"Result: {result}")


# --- TEXT Group Commands ---

@text.command(
    help="Tokenizes text, selecting only alphanumeric characters and lower-casing."
)
@click.argument('input_text', type=str)
@click.pass_context
def tokenize(_ctx, input_text):
    """Example: cli text tokenize 'Hello World! This is Text 123.'"""
    result = tokenize_text(input_text)
    click.echo(f"Result: {result}")

@text.command(
    help="Selects only alphanumeric characters and spaces from the text."
)
@click.argument('input_text', type=str)
@click.pass_context
def remove_punctuation(_ctx, input_text):
    """Example: cli text remove-punctuation 'Hello World! This is Text 123.'"""
    result = select_alphanumeric_and_spaces(input_text)
    click.echo(f"Result: {result}")

@text.command(
    help="Removes specified stop-words from the text (lower-cased)."
)
@click.argument('input_text', type=str)
@click.option(
    '--stop_words',
    '-s',
    type=str,
    required=True,
    callback=parse_list_argument,
    help='List of strings defining the stop-words to remove (e.g., \'[\"the\", \"a\"]\').'
)
@click.pass_context
def remove_stopwords(_ctx, input_text, stop_words):
    """Example: cli text remove-stopwords 'The dog jumps over the fence' -s '[\"the\", \"over\"]'"""
    result = remove_stop_words(input_text, stop_words)
    click.echo(f"Result: {result}")


# --- STRUCT Group Commands ---

@struct.command(
    help="Randomly shuffles a list, with an optional seed for reproducibility."
)
@click.argument('values', type=str, callback=parse_list_argument)
@click.option(
    '--seed',
    '-s',
    type=int,
    default=None,
    help='Integer seed value for the random number generator (default: None).',
    show_default=True
)
@click.pass_context
def shuffle(_ctx, values, seed):
    """Example: cli struct shuffle '[1, 2, 3, 4, 5]' --seed 42"""
    result = random_shuffle_list(values, seed)
    click.echo(f"Result: {result}")

@struct.command(
    help="Flattens a list of lists into a single list."
)
@click.argument('list_of_lists', type=str, callback=parse_list_argument)
@click.pass_context
def flatten(_ctx, list_of_lists):
    """Example: cli struct flatten '[[1, 2], [3], [4, 5]]'"""
    result = flatten_list(list_of_lists)
    click.echo(f"Result: {result}")

@struct.command(
    name="unique-values",
    help="Returns a list of unique values (equivalent to removing duplicates)."
)
@click.argument('values', type=str, callback=parse_list_argument)
@click.pass_context
def unique_values_struct(_ctx, values):
    """Example: cli struct unique-values '[1, 2, 2, 3, 1]'"""
    result = remove_duplicated_values(values)
    click.echo(f"Result: {result}")

if __name__ == '__main__':
    cli()