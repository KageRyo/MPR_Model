# PyPI Release Process

This workflow prepares the package release but does not publish while the
repository is private. The repository owner must complete the public-release
review and manually change the repository visibility before any PyPI release.

## Required order

1. Merge WQSurrogateModels #24, #25, and #26 in order.
2. Confirm the public-release audit and repository policies are complete.
3. Manually change the repository visibility to public.
4. Configure the PyPI and TestPyPI Trusted Publishers for
   `KageRyo/WQSurrogateModels` and `.github/workflows/release.yml`:

   | Index | Package | Environment |
   | --- | --- | --- |
   | PyPI | `wqsurrogatemodels` | `pypi` |
   | TestPyPI | `wqsurrogatemodels` | `testpypi` |

5. From `main`, run the workflow manually with the `testpypi` target and
   verify installation and the direct WQI5 smoke test.
6. Create and publish a GitHub Release from `main` with:

   - tag: `WQSurrogateModels-v2.1.0`
   - release title: `WQSurrogateModels v2.1.0`

   The published release triggers the production PyPI job. The workflow checks
   the repository visibility, tag, release title, package version, tests, and
   distribution contents before the upload.

## Authentication and approval

The workflow uses PyPI Trusted Publishing through GitHub Actions OIDC. No
long-lived PyPI or TestPyPI API token is stored in repository secrets. Configure
the `pypi` environment with required reviewers before the first production
release; the `testpypi` environment can also require approval for manual dry
runs.

The workflow does not publish on branch pushes or pull requests. A release
event must be published for production, and manual dispatch is restricted to
TestPyPI from `main`.
