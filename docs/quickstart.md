# Quick Start

## 1. Create an Environment

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .[dev,docs,gui]
```

## 2. Run the Test Suite

```bash
pytest
```

## 3. Build Documentation

```bash
mkdocs build --strict
```

## 4. Launch the GUI

```bash
python scripts/launch_gui.py
```

## 5. Run an Example Script

```bash
python examples/basic_synthetic_case.py
```
