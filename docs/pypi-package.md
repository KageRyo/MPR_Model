# PyPI Package Boundary

The `wqsurrogatemodels` PyPI distribution contains the reusable backend runtime,
not the complete research repository.

## Included

- `wqsurrogatemodels` Python modules and their package metadata
- `README-PYPI.md` and `LICENSE`
- Runtime dependency metadata from `pyproject.toml`

## Excluded

- Research datasets and row-level exports
- Serialized model binaries such as `.pkl` and `.joblib`
- Generated results, figures, and statistics
- Experiment configurations and reproducibility scripts
- Repository-only tests and documentation files

The direct WQI5 functions and API code are installable from PyPI. Surrogate
artifacts and datasets remain separately supplied local resources, so installing
the package does not grant access to research data or trained model parameters.
Set `PROJECT_ROOT` when the API needs to load local resources outside a source
checkout.
