# PyPI Release Process

This workflow builds and publishes the package through PyPI Trusted Publishing.
It is available only while the repository is public.

## Required order

1. Confirm the public-release audit and repository policies are complete.
2. Confirm `main` CI is green and the canonical package version in
   `src/wqsurrogatemodels/version.py` has been updated.
3. Configure the PyPI and TestPyPI Trusted Publishers for
   `KageRyo/WQSurrogateModels` and `.github/workflows/release.yml`:

   | Index | Package | Environment |
   | --- | --- | --- |
   | PyPI | `wqsurrogatemodels` | `pypi` |
   | TestPyPI | `wqsurrogatemodels` | `testpypi` |

4. From `main`, run the workflow manually with the `testpypi` target and
   verify installation and the direct WQI5 smoke test.
5. Create and publish a GitHub Release from `main` with:

   - tag: `WQSurrogateModels-v<package-version>`
   - release title: `WQSurrogateModels v<package-version>`

   The published release triggers the production PyPI job. The workflow checks
   the repository visibility, tag, release title, canonical package version,
   tests, and distribution contents before the upload.

## Authentication and approval

The workflow uses PyPI Trusted Publishing through GitHub Actions OIDC. No
long-lived PyPI or TestPyPI API token is stored in repository secrets. Configure
the `pypi` environment with required reviewers before the first production
release; the `testpypi` environment can also require approval for manual dry
runs.

The workflow does not publish on branch pushes or pull requests. A release
event must be published for production, and manual dispatch is restricted to
TestPyPI from `main`.
