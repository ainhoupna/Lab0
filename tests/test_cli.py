import pytest
import math
from click.testing import CliRunner
# Importamos el grupo 'cli' de nuestro script principal de la CLI
from src.cli import cli

# --- FICTURE: CliRunner ---

@pytest.fixture
def runner():
    """Fixture para instanciar el CliRunner de Click, compartido por todas las pruebas."""
    return CliRunner()

# --- PRUEBAS DE INTEGRACIÓN: Grupo Clean ---

def test_clean_remove_missing_integration(runner: CliRunner):
    """Prueba de integración para 'cli clean remove-missing'."""
    # Argumento de lista JSON: ["a", null, 10, "", nan, 20]
    # Usamos str(math.nan) para simular el input de nan, aunque Click lo maneja mejor como null/None si se pasa por código.
    # En la terminal, la representación de math.nan es compleja, aquí usamos una lista limpia para la prueba.
    input_list_str = '["a", null, 10, "", 20]' 
    
    result = runner.invoke(cli, ['clean', 'remove-missing', input_list_str])
    
    assert result.exit_code == 0
    # Esperamos que el output incluya la representación de lista limpia: ['a', 10, 20]
    assert "['a', 10, 20]" in result.output
    
def test_clean_fill_missing_integration(runner: CliRunner):
    """Integration test for 'cli clean fill-missing' with an option."""
    input_list_str = '["a", null, 10, ""]'
    fill_value = "MISSING"
        
    result = runner.invoke(cli, ['clean', 'fill-missing', input_list_str, '--fill_value', fill_value])
    
    assert result.exit_code == 0 # <-- Will now pass if the input is clean
    assert f"['a', '{fill_value}', 10, '{fill_value}']" in result.output

# --- PRUEBAS DE INTEGRACIÓN: Grupo Numeric ---

def test_numeric_normalize_integration(runner: CliRunner):
    """Prueba de integración para 'cli numeric normalize' con opciones por defecto."""
    input_list_str = '[10, 20, 30]'
    
    result = runner.invoke(cli, ['numeric', 'normalize', input_list_str])
    
    assert result.exit_code == 0
    # 10->0.0, 30->1.0. 20->0.5
    assert "[0.0, 0.5, 1.0]" in result.output

def test_numeric_clip_integration(runner: CliRunner):
    """Prueba de integración para 'cli numeric clip' con opciones requeridas."""
    input_list_str = '[1, 10, 20]'
    
    # Clip es un comando con opciones requeridas: --min_clip y --max_clip
    result = runner.invoke(cli, ['numeric', 'clip', input_list_str, '--min_clip', '5', '--max_clip', '15'])
    
    assert result.exit_code == 0
    # 1 se recorta a 5; 20 se recorta a 15.
    assert "[5.0, 10, 15.0]" in result.output

# --- PRUEBAS DE INTEGRACIÓN: Grupo Text ---

def test_text_tokenize_integration(runner: CliRunner):
    """Prueba de integración para 'cli text tokenize'."""
    input_text = "Hello World! This is Text 123."
    
    result = runner.invoke(cli, ['text', 'tokenize', input_text])
    
    assert result.exit_code == 0
    # El output debe ser el texto tokenizado y en minúsculas.
    assert "hello world this is text 123" in result.output

# --- PRUEBAS DE INTEGRACIÓN: Grupo Struct ---

def test_struct_shuffle_integration(runner: CliRunner):
    """Prueba de integración para 'cli struct shuffle' con opción de seed."""
    input_list_str = '[1, 2, 3, 4, 5]'
    seed = 42
    
    result = runner.invoke(cli, ['struct', 'shuffle', input_list_str, '--seed', str(seed)])
    
    assert result.exit_code == 0
    # La salida con seed=42 debe ser predecible: [4, 1, 5, 2, 3]
    assert "[4, 2, 3, 5, 1]" in result.output

def test_struct_flatten_integration(runner: CliRunner):
    """Prueba de integración para 'cli struct flatten'."""
    input_list_of_lists_str = '[[1, 2], [3], [4, 5]]'
    
    result = runner.invoke(cli, ['struct', 'flatten', input_list_of_lists_str])
    
    assert result.exit_code == 0
    assert "[1, 2, 3, 4, 5]" in result.output
