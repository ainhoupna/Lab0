## Lab0: Fundamentals of Continuous Integration (CI)

This project focuses on establishing the fundamentals of Continuous Integration (CI). It involves implementing basic CI practices, including linting, automatic code formatting, and testing, to guarantee the functionality and correctness of the code.


The core application is a Command Line Interface (CLI) used to execute various data preprocessing techniques

## Project Structure and Setup

Lab0: Fundamentals of Continuous Integration (CI)
This project focuses on establishing the fundamentals of Continuous Integration (CI). It involves implementing basic CI practices, including linting, automatic code formatting, and testing, to guarantee the functionality and correctness of the code.


The core application is a Command Line Interface (CLI) used to execute various data preprocessing techniques.

Project Structure and Setup
The project follows the best practices of MLOps by utilizing a specific virtual environment. The final structure includes the source code (src), testing files (tests), and necessary configuration files.

### 1\. File Structure

| Directory/File | Description |
| :--- | :--- |
| `pyproject.toml` | The primary project configuration file. |
| `pytest.ini` | Pytest configuration settings. |
| `README.md` | Provides documentation for the project. |
| `src/` | Stores the project's logic and implementation of the CLI. |
| `src/cli.py` | Implements the CLI using the `click` package. |
| `src/__init__.py` | Marks the `src` folder as a Python package. |
| `src/preprocessing.py` | Stores the main logic of the project, programming data preprocessing techniques. |
| `tests/` | Stores the testing files. |
| `tests/test_cli.py` | Implementation of integration testing between the logic and the CLI. |
| `tests/test_logic.py` | Implementation of unit testing for the core project functionalities. |
| `uv.lock` | Locked file managing environment dependencies. |

### 2\. Dependencies

The project uses a specific virtual environment. The specific dependencies were installed using `uv init` and `uv sync`:

  * **`click`**: To develop the Command Line Interface (CLI) for the logic of the project.
  * **`black`**: To perform automatic code formatting.
  * **`pylint`**: To perform linting and check code correctness.
  * **`pytest`**: The framework used for all project testing (both unit tests and integration tests).
  * **`pytest-cov`**: A plugin to measure the coverage of the testing process.


##  CI Process Execution

These commands allow you to run the CI tools from the project root directoryç.

### 1\. Code Linting (Quality Check)

Pylint checks for errors, enforces a coding standard, looks for code smells, and can make suggestions about how the code could be refactored. To run the linting of the project, execute the following command:

```bash
uv run python -m pylint src/*.py
```

### 2\. Code Formatting (Style Enforcement)

A Python library devoted to this end is **black**. You can run it executing the following command:

```bash
uv run black src/*.py
```

### 3\. Testing and Coverage

The testing files must start with `test_`, and the name of each test inside the testing files also must start with `test_`.

```bash
# Run all tests (unit and integration)
uv run python -m pytest -v
```

```bash
# Run tests and measure coverage
uv run python -m pytest -v --cov=src
```