# Contributing to imagen-mcp

Thank you for your interest in contributing!

## Development Setup

1.  **Clone the repository**:
    ```bash
    git clone https://github.com/michaeljabbour/imagen-mcp.git
    cd imagen-mcp
    ```

2.  **Create and activate a virtual environment**:
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

3.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    pip install -r dev-requirements.txt
    ```

## Code Quality

We use `ruff` for linting and formatting, and `mypy` for static type checking.

### Linting & Formatting

Run the following command to format code and fix linting issues:

```bash
ruff format .
ruff check . --fix
```

### Type Checking

Run `mypy` to check for type errors:

```bash
mypy src/
```

Ensure all type checks pass before submitting a pull request.

## Testing

Run the test suite using `pytest`:

```bash
pytest tests/ -v
```

## Pull Request Process

1.  Create a new branch for your feature or fix.
2.  Make your changes.
3.  Run linters, type checks, and tests.
4.  Submit a PR with a clear description of your changes.
