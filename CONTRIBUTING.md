# Contributing Guide

Thank you for considering contributing to WQSurrogateModels, the FastAPI backend and reproducibility repository for WQI5-based water quality assessment.

## Contents
- [Workflow](#workflow)
- [How to Contribute](#how-to-contribute)
- [Branch Naming](#branch-naming)
- [Commit Messages](#commit-messages)
- [Pull Request Process](#pull-request-process)
- [Code of Conduct](#code-of-conduct)
- [License](#license)

---

## Workflow

This project follows GitHub Flow with `main` as the primary branch.

1. `main` stays releasable.
2. Create a topic branch from `main` for each change.
3. Push your branch regularly.
4. Open a Pull Request for review.
5. Merge only after approval and passing checks.

---

## How to Contribute

1. Fork the repository.
2. Clone your fork and add the upstream remote.
3. Create a topic branch from `main`.
4. Make your changes and add or update tests when appropriate.
5. Commit your work using the conventions below.
6. Push your branch and open a Pull Request against `main`.

---

## Branch Naming

Use descriptive names prefixed by category:

- `feature/…` for new features
- `fix/…` for bug fixes
- `docs/…` for documentation-only changes
- `refactor/…` for refactoring
- `test/…` for tests

Examples:
- `feature/add-lstm-model`
- `fix/csv-parsing-error`
- `docs/update-readme`

---

## Commit Messages

Follow the Conventional Commits specification:

```
type(scope): short description
```

Common types: `feat`, `fix`, `docs`, `style`, `refactor`, `perf`, `test`, `chore`.

Examples:

```
feat(api): add batch prediction endpoint
fix(model): resolve memory leak in prediction
docs(readme): update installation instructions
```

---

## Pull Request Process

Before opening a PR, make sure:

- Tests pass locally.
- You added or updated tests where appropriate.
- Documentation is updated if needed.
- Your branch is up to date with `main`.

PRs should include a clear description of the change and how to test it.

---

## Code of Conduct

Be respectful and constructive. Harassment, discrimination, or personal attacks are not acceptable.

## License

By contributing, you agree that your contributions will be licensed under the project's
[Apache License 2.0](LICENSE).
