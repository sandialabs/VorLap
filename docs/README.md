# VorLap Documentation

This directory contains the Sphinx documentation for VorLap.

## Building Documentation Locally

### Prerequisites

Install the documentation dependencies:

```bash
pip install sphinx sphinx-rtd-theme sphinx-autobuild
```

### Build HTML Documentation

```bash
cd docs/
make html
```

The built documentation will be available in `_build/html/index.html`.

### Live Reload Development

For development with automatic rebuilding:

```bash
cd docs/
make livehtml
```

This will start a local server with live reload at `http://localhost:8000`.

## Deployment

The documentation is automatically built and deployed by GitHub Actions on every push to the main branch.
