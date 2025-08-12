# Documentation

This directory contains the Sphinx documentation for BellmanFilterDFSV.

## 🌐 Online Documentation

The documentation is automatically built and deployed to GitHub Pages:

**📖 [https://givani30.github.io/BellmanFilterDFSV/](https://givani30.github.io/BellmanFilterDFSV/)**

## 🏗️ Building Locally

### Prerequisites

```bash
# Install documentation dependencies
uv sync --group dev
```

### Build HTML Documentation

```bash
cd docs/
make html
```

The built documentation will be available at `docs/build/html/index.html`.

### Alternative Build Command

```bash
# From project root
uv run sphinx-build -b html docs/source docs/build/html
```

## 📁 Documentation Structure

```
docs/
├── source/
│   ├── index.rst              # Main landing page
│   ├── installation.rst       # Installation guide
│   ├── usage.rst             # Usage examples
│   ├── examples.rst          # Complete examples
│   ├── contributing.rst      # Development guide
│   ├── changelog.rst         # Release notes
│   ├── conf.py              # Sphinx configuration
│   └── api/                 # API documentation
│       ├── index.rst
│       └── core.rst
├── build/                   # Generated documentation
├── Makefile                # Build commands
└── make.bat               # Windows build commands
```

## 🔄 Automatic Deployment

Documentation is automatically built and deployed via GitHub Actions:

- **Trigger**: Push to `main` branch
- **Workflow**: `.github/workflows/docs.yml`
- **Deployment**: GitHub Pages

## ✏️ Contributing to Documentation

1. **Edit source files** in `docs/source/`
2. **Build locally** to test changes
3. **Commit and push** to trigger automatic deployment

### Adding New Pages

1. Create new `.rst` file in `docs/source/`
2. Add to `toctree` in `index.rst`
3. Build and test locally

### API Documentation

API documentation is automatically generated from docstrings using Sphinx autodoc. Ensure all public functions have proper Google-style docstrings.

## 🛠️ Troubleshooting

**Build Errors:**
- Check that all dependencies are installed
- Verify Python path in `conf.py`
- Ensure all referenced files exist

**Missing Modules:**
- Install package in development mode: `uv sync`
- Check `sys.path` configuration in `conf.py`

**Formatting Issues:**
- Validate RST syntax
- Check indentation (RST is whitespace-sensitive)
- Use `sphinx-build -W` for warnings as errors
